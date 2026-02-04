"""
Train JURKAT ensemble (5 models with different seeds) on Modal GPU.

Usage:
    modal run scripts/train_jurkat_ensemble.py

This trains 5 JURKAT models with seeds 1-5, similar to the THP1 ensemble
that achieved ρ=0.89 (vs ρ=0.39 baseline). Expected improvement for JURKAT:
from ρ=0.50 baseline to potentially ρ=0.7+.

Cost estimate: ~$10-20 (5 models × ~$2-4 each on T4 GPU)
"""

import modal
from pathlib import Path

app = modal.App(name="ctrl-dna-jurkat-ensemble")

REPO_ROOT = Path(__file__).parent.parent.resolve()

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

checkpoint_volume = modal.Volume.from_name("ctrl-dna-checkpoints", create_if_missing=True)


@app.function(
    image=training_image,
    gpu="T4",
    timeout=14400,
    volumes={"/checkpoints": checkpoint_volume},
)
def train_single_model(
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
    seed: int,
    ensemble_id: int,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    patience: int = 7,
    dim: int = 384,
    depth: int = 4,
) -> dict:
    """Train a single JURKAT ensemble member."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    import numpy as np
    from pathlib import Path
    import pandas as pd
    import sys

    sys.path.insert(0, "/repo/Ctrl-DNA/ctrl_dna")
    from src.reglm.regression import EnformerModel, SeqDataset

    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class AugmentedSeqDataset(Dataset):
        COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

        def __init__(self, base_dataset: SeqDataset, augment: bool = True):
            self.base = base_dataset
            self.augment = augment

        def __len__(self):
            return len(self.base) * 2 if self.augment else len(self.base)

        def __getitem__(self, idx):
            if not self.augment or idx < len(self.base):
                return self.base[idx]
            orig_idx = idx - len(self.base)
            seq_tensor, label = self.base[orig_idx]
            rc_tensor = torch.flip(seq_tensor, dims=[0, 1])
            return rc_tensor, label

    class EnformerModelWithScheduler(EnformerModel):
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

    print(f"\n{'='*60}")
    print(f"Training JURKAT ensemble member {ensemble_id} (seed={seed})")
    print(f"  Train: {len(train_data)} samples (x2 with augmentation)")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Max epochs: {epochs}, Early stopping patience: {patience}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    base_train_dataset = SeqDataset(train_df[["sequence", "label"]], seq_len=250)
    train_dataset = AugmentedSeqDataset(base_train_dataset, augment=True)
    val_dataset = SeqDataset(val_df[["sequence", "label"]], seq_len=250)
    test_dataset = SeqDataset(test_df[["sequence", "label"]], seq_len=250)

    model = EnformerModelWithScheduler(
        lr=lr,
        loss="mse",
        pretrained=False,
        dim=dim,
        depth=depth,
        n_downsamples=4,
    )

    save_dir = Path(f"/checkpoints/JURKAT_ensemble{ensemble_id}_training")
    save_dir.mkdir(parents=True, exist_ok=True)
    for old_ckpt in save_dir.rglob("*.ckpt"):
        old_ckpt.unlink()

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[0],
        logger=CSVLogger(str(save_dir)),
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print(f"\n  Training stopped at epoch {trainer.current_epoch}")
    print(f"  Best val_loss: {checkpoint_callback.best_model_score:.4f}")

    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt or not Path(best_ckpt).exists():
        ckpts = list(save_dir.rglob("*.ckpt"))
        if ckpts:
            best_ckpt = str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1])

    final_path = f"/checkpoints/human_paired_jurkat_ensemble{ensemble_id}.ckpt"

    import shutil
    if best_ckpt:
        shutil.copy(str(best_ckpt), final_path)
        print(f"\nSaved checkpoint: {final_path}")

    # Reload best checkpoint for evaluation
    if final_path and Path(final_path).exists():
        model = EnformerModel.load_from_checkpoint(final_path)
        model.cuda()

    # Evaluate on test set
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

    from scipy import stats
    spearman_r, spearman_p = stats.spearmanr(predictions, targets)
    pearson_r, _ = stats.pearsonr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    print(f"\n  === Test Metrics (ensemble {ensemble_id}) ===")
    print(f"  Spearman ρ: {spearman_r:.4f}")
    print(f"  Pearson r:  {pearson_r:.4f}")
    print(f"  RMSE:       {rmse:.4f}")

    checkpoint_volume.commit()

    return {
        "ensemble_id": ensemble_id,
        "seed": seed,
        "spearman_r": spearman_r,
        "pearson_r": pearson_r,
        "rmse": rmse,
        "checkpoint_path": final_path,
        "epochs_trained": min(trainer.current_epoch + 1, epochs),
    }


@app.local_entrypoint()
def main(epochs: int = 50, seeds: str = "1,2,3,4,5"):
    """Train all JURKAT ensemble members."""
    import pandas as pd
    from pathlib import Path
    import numpy as np

    seed_list = [int(s) for s in seeds.split(",")]
    print(f"Training JURKAT ensemble with {len(seed_list)} models (seeds: {seed_list})")

    # Load data
    data_path = Path("./data/mpra_cache/processed_expression.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run prepare_data.py first.")
        return

    df = pd.read_csv(data_path)

    # Split data (same as training script, seed=97)
    np.random.seed(97)

    def gc_content(seq):
        return sum(1 for c in seq if c in "GC") / len(seq)

    df["GC_bin"] = (np.floor(df["sequence"].apply(gc_content) / 0.05)).astype(int)

    train_idx, val_idx, test_idx = [], [], []
    for gc_bin in df["GC_bin"].unique():
        bin_idx = df[df["GC_bin"] == gc_bin].index.tolist()
        np.random.shuffle(bin_idx)
        n_train = int(np.ceil(len(bin_idx) * 0.7))
        n_val = int(np.floor(len(bin_idx) * 0.1))
        train_idx.extend(bin_idx[:n_train])
        val_idx.extend(bin_idx[n_train:n_train + n_val])
        test_idx.extend(bin_idx[n_train + n_val:])

    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    test_df = df.loc[test_idx]

    print(f"Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Prepare JURKAT data
    train_data = [{"sequence": row["sequence"], "label": row["JURKAT"]} for _, row in train_df.iterrows()]
    val_data = [{"sequence": row["sequence"], "label": row["JURKAT"]} for _, row in val_df.iterrows()]
    test_data = [{"sequence": row["sequence"], "label": row["JURKAT"]} for _, row in test_df.iterrows()]

    # Launch all ensemble members in parallel
    print(f"\nLaunching {len(seed_list)} training jobs in parallel...")

    futures = []
    for i, seed in enumerate(seed_list, 1):
        future = train_single_model.spawn(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            seed=seed,
            ensemble_id=i,
            epochs=epochs,
        )
        futures.append(future)

    # Collect results
    results = [f.get() for f in futures]

    # Summary
    print("\n" + "="*70)
    print("JURKAT ENSEMBLE TRAINING COMPLETE")
    print("="*70)
    print(f"{'Model':<12} {'Seed':>6} {'Epochs':>8} {'Spearman ρ':>12} {'RMSE':>10}")
    print("-"*50)

    spearman_values = []
    for r in results:
        print(f"Ensemble {r['ensemble_id']:<4} {r['seed']:>6} {r['epochs_trained']:>8} {r['spearman_r']:>12.4f} {r['rmse']:>10.4f}")
        spearman_values.append(r['spearman_r'])

    print("-"*50)
    print(f"{'Mean':>18} {np.mean(spearman_values):>12.4f}")
    print(f"{'Best':>18} {max(spearman_values):>12.4f}")
    print(f"{'Worst':>18} {min(spearman_values):>12.4f}")

    baseline = 0.50
    best = max(spearman_values)
    print(f"\nBaseline (single model): ρ={baseline}")
    print(f"Best individual: ρ={best:.4f} ({(best-baseline)/baseline*100:+.1f}%)")

    print("\nCheckpoints saved. Download with:")
    print("  modal volume get ctrl-dna-checkpoints human_paired_jurkat_ensemble*.ckpt ./checkpoints/")
    print("\nThen run ensemble evaluation:")
    print("  python scripts/evaluate_ensemble.py --cell JURKAT")
