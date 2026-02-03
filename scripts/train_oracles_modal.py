"""
Train Ctrl-DNA oracle models on Modal GPU with improved training.

Usage:
    # Train all oracles with 50 epochs + early stopping
    modal run scripts/train_oracles_modal.py --epochs 50

    # Train single cell type
    modal run scripts/train_oracles_modal.py --cell JURKAT --epochs 50

    # Quick test (5 epochs)
    modal run scripts/train_oracles_modal.py --cell JURKAT --epochs 5

Cost estimate: ~$5-15 total for all 3 oracles (T4 GPU, 50 epochs with early stopping)

Training improvements (v2):
- Early stopping (patience=7) to prevent overfitting
- LR scheduling (ReduceLROnPlateau) for better convergence
- Reverse complement augmentation (doubles effective data)
- Gradient clipping for stability

IMPORTANT: This script uses the SAME EnformerModel class as inference
(from Ctrl-DNA/ctrl_dna/src/reglm/regression.py) with loss="mse" to ensure
checkpoint compatibility. Previous versions defined a local class that
caused loss type mismatch (MSE training but Poisson inference).
"""

import modal
from pathlib import Path

app = modal.App(name="ctrl-dna-oracle-training")

# Get the repo root for mounting
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Image with all dependencies and local repo snapshot
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2+cu121",
        "torchvision==0.16.2+cu121",
        "torchaudio==2.1.2+cu121",
        "pytorch-lightning==1.9.5",
        "enformer-pytorch",
        "pandas",
        "numpy<2",
        "scipy",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
)

# Volume to persist checkpoints
checkpoint_volume = modal.Volume.from_name("ctrl-dna-checkpoints", create_if_missing=True)

@app.function(
    image=training_image,
    gpu="T4",
    timeout=14400,  # 4 hours max per cell type
    volumes={"/checkpoints": checkpoint_volume},
)
def train_oracle(
    cell: str,
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    patience: int = 7,
    use_augmentation: bool = True,
    dim: int = 384,
    depth: int = 4,
    ensemble_id: int = 0,
) -> dict:
    """
    Train EnformerModel oracle for a single cell type on GPU.

    Improvements over v1:
    - Early stopping (patience=7) to prevent overfitting
    - LR scheduling (ReduceLROnPlateau) for better convergence
    - Reverse complement augmentation (doubles effective data)
    - Gradient clipping for stability

    CRITICAL: Uses the SAME EnformerModel class signature as regression.py
    with loss="mse" explicitly set. This ensures checkpoint compatibility
    at inference time (base_optimizer.py loads with regression.EnformerModel).
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    import numpy as np
    from pathlib import Path
    import pandas as pd
    import sys

    # Use canonical Ctrl-DNA model/dataset to avoid drift
    sys.path.insert(0, "/repo/Ctrl-DNA/ctrl_dna")
    from src.reglm.regression import EnformerModel, SeqDataset

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Ensure the Modal image installs a CUDA-enabled "
            "PyTorch wheel and that a GPU is attached."
        )

    # Reverse complement augmentation dataset wrapper
    class AugmentedSeqDataset(Dataset):
        """Wraps SeqDataset with reverse complement augmentation."""

        COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

        def __init__(self, base_dataset: SeqDataset, augment: bool = True):
            self.base = base_dataset
            self.augment = augment

        def __len__(self):
            return len(self.base) * 2 if self.augment else len(self.base)

        def __getitem__(self, idx):
            if not self.augment or idx < len(self.base):
                return self.base[idx]

            # Get original data
            orig_idx = idx - len(self.base)
            seq_tensor, label = self.base[orig_idx]

            # Reverse complement: flip both dimensions
            # seq_tensor is (L, 4) one-hot: A=0, C=1, G=2, T=3
            # RC swaps A<->T, C<->G and reverses
            rc_tensor = torch.flip(seq_tensor, dims=[0, 1])

            return rc_tensor, label

    # Custom model with LR scheduling
    class EnformerModelWithScheduler(EnformerModel):
        """EnformerModel with ReduceLROnPlateau scheduler."""

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

    aug_status = "ON (2x data)" if use_augmentation else "OFF"
    ensemble_str = f" (ensemble {ensemble_id})" if ensemble_id > 0 else ""
    print(f"\n{'='*60}")
    print(f"Training oracle for {cell}{ensemble_str} (v2 with improvements)")
    print(f"  Train samples: {len(train_data)} (x2 with augmentation = {len(train_data) * 2})")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Max epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial LR: {lr}")
    print(f"  Architecture: dim={dim}, depth={depth}")
    print(f"  LR scheduling: ReduceLROnPlateau (factor=0.5, patience=3)")
    print(f"  Reverse complement augmentation: {aug_status}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Loss type: MSE (explicitly set for checkpoint compatibility)")
    print(f"{'='*60}\n")

    # Create datasets
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    base_train_dataset = SeqDataset(train_df[["sequence", "label"]], seq_len=250)
    train_dataset = AugmentedSeqDataset(base_train_dataset, augment=use_augmentation)
    val_dataset = SeqDataset(val_df[["sequence", "label"]], seq_len=250)
    test_dataset = SeqDataset(test_df[["sequence", "label"]], seq_len=250)

    # Initialize model with scheduler
    model = EnformerModelWithScheduler(
        lr=lr,
        loss="mse",
        pretrained=False,
        dim=dim,
        depth=depth,
        n_downsamples=4,
    )

    # Setup training directory
    save_dir = Path(f"/checkpoints/{cell}_training")
    save_dir.mkdir(parents=True, exist_ok=True)
    for old_ckpt in save_dir.rglob("*.ckpt"):
        old_ckpt.unlink()

    # Create dataloaders
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Create trainer with all improvements
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[0],
        logger=CSVLogger(str(save_dir)),
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        gradient_clip_val=1.0,  # Gradient clipping for stability
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print(f"\n  Training stopped at epoch {trainer.current_epoch}")
    print(f"  Best val_loss: {checkpoint_callback.best_model_score:.4f}")

    # Copy best checkpoint to final location
    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt or not Path(best_ckpt).exists():
        ckpts = list(save_dir.rglob("*.ckpt"))
        if ckpts:
            best_ckpt = str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1])

    # Match expected naming in base_optimizer.py: THP1 stays uppercase, others lowercase
    cell_name = cell if cell == "THP1" else cell.lower()
    if ensemble_id > 0:
        final_path = f"/checkpoints/human_paired_{cell_name}_ensemble{ensemble_id}.ckpt"
    else:
        final_path = f"/checkpoints/human_paired_{cell_name}.ckpt"

    import shutil
    if best_ckpt:
        shutil.copy(str(best_ckpt), final_path)
        print(f"\nSaved checkpoint: {final_path}")
        print(f"  loss_type in checkpoint: mse (will NOT exponentiate on inference)")

    # Reload best checkpoint for evaluation (model object may be from last epoch, not best)
    if final_path and Path(final_path).exists():
        print("  Reloading best checkpoint for evaluation...")
        model = EnformerModel.load_from_checkpoint(final_path)
        model.cuda()

    # PyTorch Lightning's current_epoch is 0-indexed and may overshoot at end of training
    epochs_trained = min(trainer.current_epoch + 1, epochs)

    # Compute validation loss on the reloaded model
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_dl:
            x, y = batch
            x = x.cuda()
            pred = model(x, return_logits=True)
            val_preds.extend(pred.cpu().numpy().flatten().tolist())
            val_targets.extend(y.numpy().flatten().tolist())
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    best_val_loss = float(np.mean((val_targets - val_preds) ** 2)) if len(val_targets) else float("nan")

    # Evaluate on test set
    print(f"\n  Evaluating on test set ({len(test_data)} samples)...")
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_dl:
            x, y = batch
            x = x.cuda()
            pred = model(x, return_logits=True)
            predictions.extend(pred.cpu().numpy().flatten().tolist())
            targets.extend(y.numpy().flatten().tolist())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    from scipy import stats

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    spearman_r, spearman_p = stats.spearmanr(predictions, targets)
    pearson_r, pearson_p = stats.pearsonr(predictions, targets)
    invalid_corr = False
    if not np.isfinite(spearman_r):
        spearman_r = 0.0
        spearman_p = float("nan")
        invalid_corr = True
    if not np.isfinite(pearson_r):
        pearson_r = 0.0
        pearson_p = float("nan")
        invalid_corr = True
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    print(f"\n  === Test Set Metrics for {cell} ===")
    print(f"  R²:              {r2:.4f}")
    print(f"  Spearman ρ:      {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Pearson r:       {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  MAE:             {mae:.4f}")
    print(f"  RMSE:            {rmse:.4f}")

    if invalid_corr:
        print(f"\n  ⚠️  WARNING: Invalid correlation metrics (NaN) detected!")
    elif spearman_r < 0.5:
        print(f"\n  ⚠️  WARNING: Spearman ρ < 0.5 indicates weak predictive power!")

    # Commit volume changes
    checkpoint_volume.commit()

    return {
        "cell": cell,
        "best_val_loss": best_val_loss,
        "checkpoint_path": final_path,
        "epochs_trained": epochs_trained,
        "max_epochs": epochs,
        "early_stopped": epochs_trained < epochs,
        "loss_type": "mse",
        "test_r2": r2,
        "test_spearman_r": spearman_r,
        "test_pearson_r": pearson_r,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_n_samples": len(test_data),
        "metrics_invalid": invalid_corr,
        "dim": dim,
        "depth": depth,
        "ensemble_id": ensemble_id,
    }


@app.local_entrypoint()
def main(
    cell: str = "all",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    patience: int = 7,
    no_augmentation: bool = False,
    data_dir: str = "./data",
    dim: int = 384,
    depth: int = 4,
    seed: int = 97,
    ensemble_id: int = 0,
):
    """Train oracle models on Modal GPU with improved training."""
    import pandas as pd
    from pathlib import Path
    import traceback

    try:
        print("Modal local_entrypoint starting...")
        data_path = Path(data_dir).resolve()
        print(f"Using data_dir: {data_path}")

        # Load processed data
        processed_path = data_path / "mpra_cache" / "processed_expression.csv"
        print(f"Looking for processed data at: {processed_path}")
        if not processed_path.exists():
            print(f"Error: {processed_path} not found. Run prepare_data.py first.")
            return

        print("Loading data...")
        df = pd.read_csv(processed_path)

        # Split into train/val - must match local script exactly (stratify by GC content only)
        # Note: prepare_data.py doesn't save class labels, so we stratify by GC content only
        # This is consistent with the data available from processed_expression.csv
        import numpy as np
        np.random.seed(seed)
        print(f"Using random seed: {seed}")

        def gc_content(seq):
            return sum(1 for c in seq if c in "GC") / len(seq)

        df["GC_bin"] = (np.floor(df["sequence"].apply(gc_content) / 0.05)).astype(int)

        train_idx, val_idx, test_idx = [], [], []
        for gc_bin in df["GC_bin"].unique():
            bin_idx = df[df["GC_bin"] == gc_bin].index.tolist()
            np.random.shuffle(bin_idx)
            # Use 70/10/20 split (train/val/test) to match local script
            n_train = int(np.ceil(len(bin_idx) * 0.7))
            n_val = int(np.floor(len(bin_idx) * 0.1))
            train_idx.extend(bin_idx[:n_train])
            val_idx.extend(bin_idx[n_train:n_train + n_val])
            test_idx.extend(bin_idx[n_train + n_val:])

        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Determine which cells to train
        cells = ["JURKAT", "K562", "THP1"] if cell == "all" else [cell]

        results = []
        for c in cells:
            print(f"\n{'='*50}")
            print(f"Launching training for {c}...")
            print(f"{'='*50}")

            # Prepare data for this cell type
            train_data = [
                {"sequence": row["sequence"], "label": row[c]}
                for _, row in train_df.iterrows()
            ]
            val_data = [
                {"sequence": row["sequence"], "label": row[c]}
                for _, row in val_df.iterrows()
            ]
            test_data = [
                {"sequence": row["sequence"], "label": row[c]}
                for _, row in test_df.iterrows()
            ]

            # Train on Modal
            result = train_oracle.remote(
                cell=c,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                use_augmentation=not no_augmentation,
                dim=dim,
                depth=depth,
                ensemble_id=ensemble_id,
            )
            results.append(result)

        # Print results
        print("\n" + "="*80)
        print("TRAINING COMPLETE - TEST SET EVALUATION SUMMARY (v2 with improvements)")
        print(f"Architecture: dim={dim}, depth={depth}")
        print("="*80)
        print(f"\n{'Cell Type':<12} {'Ens':>4} {'Epochs':>8} {'Val Loss':>10} {'R²':>8} {'Spearman ρ':>12} {'RMSE':>8}")
        print("-"*72)

        any_warnings = False
        for r in results:
            epoch_str = f"{r['epochs_trained']}/{r['max_epochs']}"
            if r.get('early_stopped'):
                epoch_str += "*"  # Mark early stopped
            ens_str = str(r.get('ensemble_id', 0)) if r.get('ensemble_id', 0) > 0 else "-"
            print(f"{r['cell']:<12} {ens_str:>4} {epoch_str:>8} {r['best_val_loss']:>10.4f} {r['test_r2']:>8.4f} {r['test_spearman_r']:>12.4f} {r['test_rmse']:>8.4f}")
            if r.get("metrics_invalid") or r['test_spearman_r'] < 0.5:
                any_warnings = True

        print("-"*72)
        print("  * = early stopped")

        if any_warnings:
            print("\n⚠️  WARNING: One or more oracles have Spearman ρ < 0.5")
            print("   This indicates weak predictive power. Consider training longer.")
        else:
            print("\n✓ All oracles have acceptable predictive power (Spearman ρ ≥ 0.5)")

        print("\nCheckpoints saved to Modal volume 'ctrl-dna-checkpoints'")
        print("Download with: modal volume get ctrl-dna-checkpoints human_paired_*.ckpt ./checkpoints/")
    except Exception:
        print("ERROR in local_entrypoint:")
        traceback.print_exc()
        raise
