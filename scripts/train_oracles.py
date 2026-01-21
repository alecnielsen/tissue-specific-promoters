#!/usr/bin/env python
"""
Train EnformerModel oracle models for Ctrl-DNA tissue-specific promoter design.

This script:
1. Downloads MPRA data from Reddy et al. (JURKAT, K562, THP1)
2. Preprocesses into train/val splits
3. Trains separate EnformerModel regressors for each cell type
4. Saves checkpoints in Ctrl-DNA expected format

Usage:
    python scripts/train_oracles.py --cell JURKAT --epochs 10 --device 0
    python scripts/train_oracles.py --cell K562 --epochs 10 --device 0
    python scripts/train_oracles.py --cell THP1 --epochs 10 --device 0

Data source: https://github.com/anikethjr/promoter_models
"""

import argparse
import os
import sys
import shlex
import numpy as np
import pandas as pd
from pathlib import Path

# Add Ctrl-DNA to path
CTRL_DNA_PATH = Path(__file__).parent.parent / "Ctrl-DNA" / "ctrl_dna"
sys.path.insert(0, str(CTRL_DNA_PATH))

from src.reglm.regression import EnformerModel, SeqDataset


def validate_download(path: Path, min_size_kb: int = 10) -> bool:
    """Validate downloaded file is not an HTML error page from Google Drive."""
    if not path.exists():
        return False
    size_kb = path.stat().st_size / 1024
    if size_kb < min_size_kb:
        print(f"  WARNING: File too small ({size_kb:.1f} KB), may be an error page")
        return False
    try:
        with open(path, 'rb') as f:
            header = f.read(100).decode('utf-8', errors='ignore').lower()
            if '<!doctype' in header or '<html' in header:
                print(f"  WARNING: File appears to be HTML (error page), not data")
                return False
    except Exception:
        pass
    return True


def download_data(cache_dir: Path) -> tuple[Path, Path]:
    """Download raw MPRA data from Google Drive."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    counts_path = cache_dir / "Raw_Promoter_Counts.csv"
    if not counts_path.exists() or not validate_download(counts_path, min_size_kb=100):
        if counts_path.exists():
            counts_path.unlink()
        print(f"Downloading Raw_Promoter_Counts.csv to {counts_path}...")
        cmd = f"curl -L 'https://drive.google.com/uc?export=download&id=15p6GhDop5BsUPryZ6pfKgwJ2XEVHRAYq' -o {shlex.quote(str(counts_path))}"
        ret = os.system(cmd)
        if ret != 0 or not validate_download(counts_path, min_size_kb=100):
            raise RuntimeError(f"Failed to download {counts_path}. Check Google Drive quota or network.")

    seqs_path = cache_dir / "final_list_of_all_promoter_sequences_fixed.tsv"
    if not seqs_path.exists() or not validate_download(seqs_path, min_size_kb=100):
        if seqs_path.exists():
            seqs_path.unlink()
        print(f"Downloading sequence list to {seqs_path}...")
        cmd = f"curl -L 'https://drive.google.com/uc?export=download&id=1kTfsZvsCz7EWUhl-UZgK0B31LtxJH4qG' -o {shlex.quote(str(seqs_path))}"
        ret = os.system(cmd)
        if ret != 0 or not validate_download(seqs_path, min_size_kb=100):
            raise RuntimeError(f"Failed to download {seqs_path}. Check Google Drive quota or network.")

    return counts_path, seqs_path


def process_mpra_data(
    counts_path: Path,
    seqs_path: Path,
    min_reads: int = 5,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process raw MPRA counts into expression values per cell type.

    Returns train, val, test DataFrames with columns:
        sequence, JURKAT, K562, THP1
    """
    cell_names = ["JURKAT", "K562", "THP1"]
    num_replicates = 2

    print("Loading raw measurements...")
    measurements = pd.read_csv(counts_path)

    # Filter by minimum reads
    measurements["keep"] = True
    for col in measurements.columns:
        if col.endswith("_sum") and col != "cum_sum":
            measurements["keep"] = measurements["keep"] & (measurements[col] >= min_reads)
    measurements = measurements[measurements["keep"]].drop("keep", axis=1).reset_index(drop=True)
    print(f"Kept {len(measurements)} sequences with >= {min_reads} reads")

    # Normalize counts (add pseudocount, divide by total)
    for col in measurements.columns:
        if not (col.endswith("_sum") or col == "sequence"):
            measurements[col] = measurements[col] + 1.0  # pseudocount
            measurements[col] = measurements[col] / measurements[col].sum()

    # Calculate log2(P4/P7) expression for each cell type
    for cell in cell_names:
        first_letter = cell[0]
        measurements[cell] = 0
        for rep in range(num_replicates):
            p4_col = f"{first_letter}{rep+1}_P4"
            p7_col = f"{first_letter}{rep+1}_P7"
            measurements[cell] += np.log2(measurements[p4_col] / measurements[p7_col])
        measurements[cell] /= num_replicates

    # Load sequence properties and merge
    seq_props = pd.read_csv(seqs_path, sep="\t")
    merged = measurements.merge(seq_props, on="sequence", how="inner")
    print(f"Merged dataset: {len(merged)} sequences")

    # Calculate GC content for stratified splitting
    def gc_content(seq):
        return sum(1 for c in seq if c in "GC") / len(seq)

    merged["GC_content"] = merged["sequence"].apply(gc_content)
    merged["GC_content_bin"] = (np.floor(merged["GC_content"] / 0.05)).astype(int)

    # Stratified split by GC content (consistent with Modal script and prepare_data.py)
    np.random.seed(97)
    all_train_inds, all_val_inds, all_test_inds = [], [], []

    for gc_bin in merged["GC_content_bin"].unique():
        bin_subset = merged[merged["GC_content_bin"] == gc_bin]
        inds = bin_subset.index.to_numpy()
        np.random.shuffle(inds)

        n_train = int(np.ceil(len(inds) * train_fraction))
        n_val = int(np.floor(len(inds) * val_fraction))

        all_train_inds.extend(inds[:n_train])
        all_val_inds.extend(inds[n_train:n_train + n_val])
        all_test_inds.extend(inds[n_train + n_val:])

    # Create splits
    train_df = merged.loc[all_train_inds, ["sequence"] + cell_names].reset_index(drop=True)
    val_df = merged.loc[all_val_inds, ["sequence"] + cell_names].reset_index(drop=True)
    test_df = merged.loc[all_test_inds, ["sequence"] + cell_names].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def compute_fitness_stats(df: pd.DataFrame, cell: str) -> tuple[float, float]:
    """Compute min/max fitness values for normalization."""
    values = df[cell].values
    return float(values.min()), float(values.max())


def evaluate_oracle(
    model,
    test_df: pd.DataFrame,
    cell: str,
    device: int = 0,
    use_cpu: bool = False,
) -> dict:
    """
    Evaluate oracle model on held-out test set.

    Returns dict with R², Spearman correlation, Pearson correlation, MAE, RMSE.
    """
    from scipy import stats

    # Create test dataset
    test_single = test_df[["sequence", cell]].copy()
    test_dataset = SeqDataset(test_single, seq_len=250)

    # Get predictions
    model.eval()
    predictions = []
    targets = []

    from torch.utils.data import DataLoader
    test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    import torch
    device_obj = torch.device("cpu" if use_cpu else f"cuda:{device}")
    model = model.to(device_obj)

    with torch.no_grad():
        for batch in test_dl:
            x, y = batch
            x = x.to(device_obj)
            pred = model(x, return_logits=True)  # Get raw logits for MSE comparison
            predictions.extend(pred.cpu().numpy().flatten().tolist())
            targets.extend(y.numpy().flatten().tolist())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    # R² (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Correlations
    spearman_r, spearman_p = stats.spearmanr(predictions, targets)
    pearson_r, pearson_p = stats.pearsonr(predictions, targets)

    # Error metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return {
        "r2": r2,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mae": mae,
        "rmse": rmse,
        "n_samples": len(targets),
    }


def train_oracle(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cell: str,
    checkpoint_dir: Path,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
    device: int = 0,
    use_cpu: bool = False,
) -> tuple:
    """
    Train an EnformerModel oracle for a single cell type and evaluate on test set.

    Returns:
        tuple: (model, test_metrics dict)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create single-task datasets (sequence + one label column)
    train_single = train_df[["sequence", cell]].copy()
    val_single = val_df[["sequence", cell]].copy()

    train_dataset = SeqDataset(train_single, seq_len=250)
    val_dataset = SeqDataset(val_single, seq_len=250)

    print(f"\nTraining oracle for {cell}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Device: {'CPU' if use_cpu else f'GPU {device}'}")

    # Initialize model
    model = EnformerModel(
        lr=lr,
        loss="mse",
        pretrained=False,
        dim=384,  # Smaller than default for faster training
        depth=4,
        n_downsamples=4,
    )

    # Train
    save_dir = checkpoint_dir / f"{cell}_training"
    trainer = model.train_on_dataset(
        train_dataset,
        val_dataset,
        device=device,
        batch_size=batch_size if not use_cpu else min(batch_size, 32),  # Smaller batch on CPU
        num_workers=4 if not use_cpu else 0,
        save_dir=str(save_dir),
        max_epochs=epochs,
        use_cpu=use_cpu,
    )

    # Copy best checkpoint to expected location
    best_ckpt = list(save_dir.rglob("*.ckpt"))
    checkpoint_path = None
    if best_ckpt:
        # Sort by modification time, get most recent (best)
        best_ckpt = sorted(best_ckpt, key=lambda p: p.stat().st_mtime)[-1]
        # Match expected naming in base_optimizer.py: THP1 stays uppercase, others lowercase
        cell_name = cell if cell == "THP1" else cell.lower()
        checkpoint_path = checkpoint_dir / f"human_paired_{cell_name}.ckpt"
        import shutil
        shutil.copy(best_ckpt, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    # Reload best checkpoint for evaluation (model object may be from last epoch, not best)
    if checkpoint_path and checkpoint_path.exists():
        print(f"  Reloading best checkpoint for evaluation...")
        import torch
        device_obj = torch.device("cpu" if use_cpu else f"cuda:{device}")
        model = EnformerModel.load_from_checkpoint(str(checkpoint_path), map_location=device_obj)
        model.to(device_obj)

    # Evaluate on test set
    print(f"\n  Evaluating on test set...")
    test_metrics = evaluate_oracle(model, test_df, cell, device=device, use_cpu=use_cpu)

    print(f"\n  === Test Set Metrics for {cell} ===")
    print(f"  R²:              {test_metrics['r2']:.4f}")
    print(f"  Spearman ρ:      {test_metrics['spearman_r']:.4f} (p={test_metrics['spearman_p']:.2e})")
    print(f"  Pearson r:       {test_metrics['pearson_r']:.4f} (p={test_metrics['pearson_p']:.2e})")
    print(f"  MAE:             {test_metrics['mae']:.4f}")
    print(f"  RMSE:            {test_metrics['rmse']:.4f}")
    print(f"  N samples:       {test_metrics['n_samples']}")

    # Warn if correlation is low
    if test_metrics['spearman_r'] < 0.5:
        print(f"\n  ⚠️  WARNING: Spearman ρ < 0.5 indicates weak predictive power!")
        print(f"      This oracle may not reliably guide optimization.")
    elif test_metrics['spearman_r'] < 0.7:
        print(f"\n  ⚠️  CAUTION: Spearman ρ < 0.7 indicates moderate predictive power.")

    return model, test_metrics


def generate_rl_init_data(
    df: pd.DataFrame,
    cell: str,
    output_dir: Path,
    top_n: int = 1000,
):
    """Generate RL initialization data for Ctrl-DNA (top promoters for each cell type)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by expression in target cell, get top sequences
    sorted_df = df.sort_values(cell, ascending=False)

    # Create output with all cell type scores
    output = sorted_df[["sequence", "JURKAT", "K562", "THP1"]].head(top_n).copy()
    output.columns = ["sequence", "JURKAT_mean", "K562_mean", "THP1_mean"]

    output_path = output_dir / f"{cell}_hard.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved RL init data: {output_path} ({len(output)} sequences)")


def main():
    parser = argparse.ArgumentParser(description="Train Ctrl-DNA oracle models")
    parser.add_argument("--cell", type=str, default="all",
                        choices=["JURKAT", "K562", "THP1", "all"],
                        help="Cell type to train (or 'all')")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--cpu", action="store_true", help="Train on CPU (slow but works without GPU)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store/cache data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--download_only", action="store_true",
                        help="Only download and process data, don't train")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    checkpoint_dir = project_root / args.checkpoint_dir

    # Download and process data
    cache_dir = data_dir / "mpra_cache"
    counts_path, seqs_path = download_data(cache_dir)

    processed_path = cache_dir / "processed_expression.csv"
    if processed_path.exists():
        print(f"Loading cached processed data from {processed_path}")
        merged = pd.read_csv(processed_path)
        # Check if split column exists (from train_oracles.py) or not (from prepare_data.py)
        if "split" in merged.columns:
            train_df = merged[merged["split"] == "train"][["sequence", "JURKAT", "K562", "THP1"]]
            val_df = merged[merged["split"] == "val"][["sequence", "JURKAT", "K562", "THP1"]]
            test_df = merged[merged["split"] == "test"][["sequence", "JURKAT", "K562", "THP1"]]
        else:
            # File from prepare_data.py - need to create splits
            # Use GC-based stratification (consistent with Modal script)
            print("Creating train/val/test splits...")
            np.random.seed(97)

            def gc_content(seq):
                return sum(1 for c in seq if c in "GC") / len(seq)

            merged["GC_bin"] = (np.floor(merged["sequence"].apply(gc_content) / 0.05)).astype(int)

            train_idx, val_idx, test_idx = [], [], []
            for gc_bin in merged["GC_bin"].unique():
                bin_idx = merged[merged["GC_bin"] == gc_bin].index.tolist()
                np.random.shuffle(bin_idx)
                n_train = int(np.ceil(len(bin_idx) * 0.7))
                n_val = int(np.floor(len(bin_idx) * 0.1))
                train_idx.extend(bin_idx[:n_train])
                val_idx.extend(bin_idx[n_train:n_train + n_val])
                test_idx.extend(bin_idx[n_train + n_val:])

            train_df = merged.loc[train_idx, ["sequence", "JURKAT", "K562", "THP1"]].reset_index(drop=True)
            val_df = merged.loc[val_idx, ["sequence", "JURKAT", "K562", "THP1"]].reset_index(drop=True)
            test_df = merged.loc[test_idx, ["sequence", "JURKAT", "K562", "THP1"]].reset_index(drop=True)
    else:
        train_df, val_df, test_df = process_mpra_data(counts_path, seqs_path)
        # Save processed data (without split labels for compatibility with prepare_data.py)
        merged = pd.concat([train_df, val_df, test_df])
        merged.to_csv(processed_path, index=False)
        print(f"Saved processed data to {processed_path}")

    # Print fitness statistics
    print("\nFitness statistics (for normalization):")
    for cell in ["JURKAT", "K562", "THP1"]:
        all_data = pd.concat([train_df, val_df, test_df])
        min_fit, max_fit = compute_fitness_stats(all_data, cell)
        print(f"  {cell}: min={min_fit:.6f}, max={max_fit:.6f}")

    # Generate RL initialization data
    rl_data_dir = data_dir / "human_promoters" / "rl_data_large"
    all_data = pd.concat([train_df, val_df, test_df])
    for cell in ["JURKAT", "K562", "THP1"]:
        generate_rl_init_data(all_data, cell, rl_data_dir)

    if args.download_only:
        print("\nData download complete. Skipping training.")
        return

    # Train oracles and collect metrics
    cells = ["JURKAT", "K562", "THP1"] if args.cell == "all" else [args.cell]
    all_metrics = {}

    for cell in cells:
        model, metrics = train_oracle(
            train_df,
            val_df,
            test_df,
            cell,
            checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            use_cpu=args.cpu,
        )
        all_metrics[cell] = metrics

    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - TEST SET EVALUATION SUMMARY")
    print("="*70)
    print(f"\n{'Cell Type':<12} {'R²':>8} {'Spearman ρ':>12} {'Pearson r':>12} {'RMSE':>8}")
    print("-"*56)

    any_warnings = False
    for cell, metrics in all_metrics.items():
        print(f"{cell:<12} {metrics['r2']:>8.4f} {metrics['spearman_r']:>12.4f} {metrics['pearson_r']:>12.4f} {metrics['rmse']:>8.4f}")
        if metrics['spearman_r'] < 0.5:
            any_warnings = True

    print("-"*56)

    if any_warnings:
        print("\n⚠️  WARNING: One or more oracles have Spearman ρ < 0.5")
        print("   This indicates weak predictive power. Consider:")
        print("   - Training for more epochs")
        print("   - Using more training data")
        print("   - Checking data quality")
        print("   - Using a larger model architecture")
    else:
        print("\n✓ All oracles have acceptable predictive power (Spearman ρ ≥ 0.5)")

    print(f"\nCheckpoints saved to: {checkpoint_dir}")

    # Save metrics to file
    metrics_path = checkpoint_dir / "oracle_test_metrics.csv"
    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = "cell_type"
    metrics_df.to_csv(metrics_path)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
