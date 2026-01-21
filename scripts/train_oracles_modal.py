"""
Train Ctrl-DNA oracle models on Modal GPU.

Usage:
    # Train all oracles
    modal run scripts/train_oracles_modal.py

    # Train single cell type
    modal run scripts/train_oracles_modal.py --cell JURKAT

    # Quick test (1 epoch)
    modal run scripts/train_oracles_modal.py --cell JURKAT --epochs 1

Cost estimate: ~$2-5 total for all 3 oracles (T4 GPU)

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

# Image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pytorch-lightning",
        "enformer-pytorch",
        "pandas",
        "numpy",
        "scipy",
    )
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
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
) -> dict:
    """
    Train EnformerModel oracle for a single cell type on GPU.

    CRITICAL: Uses the SAME EnformerModel class signature as regression.py
    with loss="mse" explicitly set. This ensures checkpoint compatibility
    at inference time (base_optimizer.py loads with regression.EnformerModel).
    """
    import torch
    from torch import nn, optim
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader, Dataset
    from enformer_pytorch import Enformer
    from enformer_pytorch.data import str_to_one_hot
    import numpy as np
    from pathlib import Path

    print(f"\n{'='*50}")
    print(f"Training oracle for {cell}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Loss type: MSE (explicitly set for checkpoint compatibility)")
    print(f"{'='*50}\n")

    # Dataset class
    class SeqDataset(Dataset):
        def __init__(self, data: list[dict], seq_len: int = 250):
            self.seqs = [d["sequence"] for d in data]
            self.labels = torch.tensor([d["label"] for d in data], dtype=torch.float32)
            self.seq_len = seq_len

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            seq = self.seqs[idx]
            if len(seq) < self.seq_len:
                seq = seq + "N" * (self.seq_len - len(seq))
            seq_onehot = str_to_one_hot(seq)
            return seq_onehot, self.labels[idx].unsqueeze(0)

    # EnformerModel class - MUST match regression.py signature exactly!
    # This ensures load_from_checkpoint() restores loss_type correctly.
    class EnformerModel(pl.LightningModule):
        """
        Enformer-based regression model matching regression.py signature.

        CRITICAL: The __init__ signature must match regression.py exactly,
        including the 'loss' parameter, so that save_hyperparameters()
        stores it and load_from_checkpoint() restores it correctly.
        """
        def __init__(
            self,
            lr=1e-4,
            loss="mse",  # CRITICAL: Must match regression.py signature
            pretrained=False,
            dim=1536,
            depth=11,
            n_downsamples=7,
        ):
            super().__init__()
            self.n_tasks = 1
            self.save_hyperparameters()  # Saves all args including 'loss'

            # Build model
            if pretrained:
                self.trunk = Enformer.from_pretrained(
                    "EleutherAI/enformer-official-rough", target_length=-1
                )._trunk
            else:
                self.trunk = Enformer.from_hparams(
                    dim=dim,
                    depth=depth,
                    heads=8,
                    num_downsamples=n_downsamples,
                    target_length=-1,
                )._trunk
            self.head = nn.Linear(dim * 2, self.n_tasks, bias=True)

            # Training params
            self.lr = lr
            self.loss_type = loss  # CRITICAL: Store for forward() check
            if loss == "poisson":
                self.loss = nn.PoissonNLLLoss(log_input=True, full=True)
            else:
                self.loss = nn.MSELoss()

        def forward(self, x, return_logits=False):
            if (isinstance(x, list)) or (isinstance(x, tuple)):
                if isinstance(x[0], str):
                    x = str_to_one_hot(x)
                else:
                    x = x[0]
            x = x.to(self.device)
            x = self.trunk(x)
            x = self.head(x)
            x = x.mean(1)

            # CRITICAL: Only exponentiate for Poisson loss
            if (self.loss_type == "poisson") and (not return_logits):
                x = torch.exp(x)
            return x

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x, return_logits=True)
            loss = self.loss(logits, y)
            self.log("train_loss", loss, logger=True, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x, return_logits=True)
            loss = self.loss(logits, y)
            self.log("val_loss", loss, logger=True, on_step=False, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.parameters(), lr=self.lr)

    # Create datasets
    train_dataset = SeqDataset(train_data)
    val_dataset = SeqDataset(val_data)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model with EXPLICIT loss="mse"
    # Using smaller architecture for faster training (same as train_oracles.py)
    model = EnformerModel(
        lr=lr,
        loss="mse",  # CRITICAL: Explicitly set MSE loss
        pretrained=False,
        dim=384,
        depth=4,
        n_downsamples=4,
    )

    # Verify loss type is correct
    print(f"  Model loss_type: {model.loss_type}")
    assert model.loss_type == "mse", f"Expected loss_type='mse', got '{model.loss_type}'"

    # Checkpoint callback
    save_dir = Path(f"/checkpoints/{cell}_training")
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        filename=f"human_paired_{cell.lower()}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=CSVLogger(str(save_dir)),
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, train_dl, val_dl)

    # Copy best checkpoint to final location
    best_ckpt = checkpoint_callback.best_model_path
    # Match expected naming in base_optimizer.py: THP1 stays uppercase, others lowercase
    cell_name = cell if cell == "THP1" else cell.lower()
    final_path = f"/checkpoints/human_paired_{cell_name}.ckpt"

    import shutil
    if best_ckpt:
        shutil.copy(best_ckpt, final_path)
        print(f"\nSaved checkpoint: {final_path}")
        print(f"  loss_type in checkpoint: mse (will NOT exponentiate on inference)")

    # Reload best checkpoint for evaluation (model object may be from last epoch, not best)
    if best_ckpt:
        print(f"  Reloading best checkpoint for evaluation...")
        model = EnformerModel.load_from_checkpoint(best_ckpt)
        model.cuda()

    # Evaluate on test set
    print(f"\n  Evaluating on test set ({len(test_data)} samples)...")
    test_dataset = SeqDataset(test_data)
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
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    print(f"\n  === Test Set Metrics for {cell} ===")
    print(f"  R²:              {r2:.4f}")
    print(f"  Spearman ρ:      {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Pearson r:       {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  MAE:             {mae:.4f}")
    print(f"  RMSE:            {rmse:.4f}")

    if spearman_r < 0.5:
        print(f"\n  ⚠️  WARNING: Spearman ρ < 0.5 indicates weak predictive power!")

    # Commit volume changes
    checkpoint_volume.commit()

    return {
        "cell": cell,
        "best_val_loss": float(checkpoint_callback.best_model_score),
        "checkpoint_path": final_path,
        "epochs_trained": epochs,
        "loss_type": "mse",
        "test_r2": r2,
        "test_spearman_r": spearman_r,
        "test_pearson_r": pearson_r,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_n_samples": len(test_data),
    }


@app.local_entrypoint()
def main(
    cell: str = "all",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
    data_dir: str = "./data",
):
    """Train oracle models on Modal GPU."""
    import pandas as pd
    from pathlib import Path

    data_path = Path(data_dir)

    # Load processed data
    processed_path = data_path / "mpra_cache" / "processed_expression.csv"
    if not processed_path.exists():
        print(f"Error: {processed_path} not found. Run prepare_data.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(processed_path)

    # Split into train/val - must match local script exactly (stratify by GC content only)
    # Note: prepare_data.py doesn't save class labels, so we stratify by GC content only
    # This is consistent with the data available from processed_expression.csv
    import numpy as np
    np.random.seed(97)

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
        )
        results.append(result)

    # Print results
    print("\n" + "="*70)
    print("TRAINING COMPLETE - TEST SET EVALUATION SUMMARY")
    print("="*70)
    print(f"\n{'Cell Type':<12} {'Val Loss':>10} {'R²':>8} {'Spearman ρ':>12} {'RMSE':>8}")
    print("-"*56)

    any_warnings = False
    for r in results:
        print(f"{r['cell']:<12} {r['best_val_loss']:>10.4f} {r['test_r2']:>8.4f} {r['test_spearman_r']:>12.4f} {r['test_rmse']:>8.4f}")
        if r['test_spearman_r'] < 0.5:
            any_warnings = True

    print("-"*56)

    if any_warnings:
        print("\n⚠️  WARNING: One or more oracles have Spearman ρ < 0.5")
        print("   This indicates weak predictive power. Consider training longer.")
    else:
        print("\n✓ All oracles have acceptable predictive power (Spearman ρ ≥ 0.5)")

    print("\nCheckpoints saved to Modal volume 'ctrl-dna-checkpoints'")
    print("Download with: modal volume get ctrl-dna-checkpoints human_paired_*.ckpt ./checkpoints/")
