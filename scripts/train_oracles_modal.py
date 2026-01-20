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
"""

import modal

app = modal.App(name="ctrl-dna-oracle-training")

# Image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pytorch-lightning",
        "enformer-pytorch",
        "pandas",
        "numpy",
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
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
) -> dict:
    """Train EnformerModel oracle for a single cell type on GPU."""
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader, Dataset
    from enformer_pytorch import Enformer
    from enformer_pytorch.data import str_to_one_hot
    import numpy as np
    import pandas as pd
    from pathlib import Path

    print(f"\n{'='*50}")
    print(f"Training oracle for {cell}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
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

    # Model class
    class EnformerModel(pl.LightningModule):
        def __init__(self, lr=1e-4, dim=384, depth=4, n_downsamples=4):
            super().__init__()
            self.save_hyperparameters()
            self.trunk = Enformer.from_hparams(
                dim=dim,
                depth=depth,
                heads=8,
                num_downsamples=n_downsamples,
                target_length=-1,
            )._trunk
            self.head = torch.nn.Linear(dim * 2, 1, bias=True)
            self.lr = lr
            self.loss = torch.nn.MSELoss()

        def forward(self, x):
            x = self.trunk(x)
            x = self.head(x)
            x = x.mean(1)
            return x

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = self.loss(logits, y)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = self.loss(logits, y)
            self.log("val_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    # Create datasets
    train_dataset = SeqDataset(train_data)
    val_dataset = SeqDataset(val_data)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    model = EnformerModel(lr=lr, dim=384, depth=4, n_downsamples=4)

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
    final_path = f"/checkpoints/human_paired_{cell.lower()}.ckpt"

    import shutil
    if best_ckpt:
        shutil.copy(best_ckpt, final_path)
        print(f"\nSaved checkpoint: {final_path}")

    # Commit volume changes
    checkpoint_volume.commit()

    return {
        "cell": cell,
        "best_val_loss": float(checkpoint_callback.best_model_score),
        "checkpoint_path": final_path,
        "epochs_trained": epochs,
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

    # Split into train/val (same logic as local script)
    import numpy as np
    np.random.seed(97)

    def gc_content(seq):
        return sum(1 for c in seq if c in "GC") / len(seq)

    df["GC_bin"] = (df["sequence"].apply(gc_content) / 0.05).astype(int)

    train_idx, val_idx = [], []
    for gc_bin in df["GC_bin"].unique():
        bin_idx = df[df["GC_bin"] == gc_bin].index.tolist()
        np.random.shuffle(bin_idx)
        n_train = int(len(bin_idx) * 0.8)
        train_idx.extend(bin_idx[:n_train])
        val_idx.extend(bin_idx[n_train:])

    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

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

        # Train on Modal
        result = train_oracle.remote(
            cell=c,
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        results.append(result)

    # Print results
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    for r in results:
        print(f"\n{r['cell']}:")
        print(f"  Best val loss: {r['best_val_loss']:.6f}")
        print(f"  Checkpoint: {r['checkpoint_path']}")

    print("\nCheckpoints saved to Modal volume 'ctrl-dna-checkpoints'")
    print("Download with: modal volume get ctrl-dna-checkpoints human_paired_*.ckpt ./checkpoints/")
