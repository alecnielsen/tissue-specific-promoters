"""
Evaluate THP1 ensemble by averaging predictions from multiple models.

Usage:
    python scripts/evaluate_ensemble.py
"""

import sys
from pathlib import Path

# Add Ctrl-DNA to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Ctrl-DNA" / "ctrl_dna"))

import numpy as np
import pandas as pd
import torch
import pathlib
from scipy import stats
from torch.utils.data import DataLoader

# Fix for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([pathlib.PosixPath])

from src.reglm.regression import EnformerModel, SeqDataset


def evaluate_ensemble():
    """Evaluate ensemble of THP1 models."""

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "mpra_cache" / "processed_expression.csv"
    df = pd.read_csv(data_path)

    # Same split as training (seed=97, but we'll use seed=5 for test consistency)
    # Actually we need to use the same split as training
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

    test_df = df.loc[test_idx]
    print(f"Test set: {len(test_df)} samples")

    # Prepare test data
    test_data = test_df[["sequence", "THP1"]].copy().reset_index(drop=True)
    test_data.columns = ["sequence", "label"]
    test_dataset = SeqDataset(test_data, seq_len=250)
    test_dl = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Load ensemble models
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    ensemble_paths = sorted(checkpoint_dir.glob("human_paired_THP1_ensemble*.ckpt"))

    print(f"\nLoading {len(ensemble_paths)} ensemble models...")
    models = []
    for ckpt_path in ensemble_paths:
        model = EnformerModel.load_from_checkpoint(str(ckpt_path))
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        models.append(model)
        print(f"  Loaded: {ckpt_path.name}")

    # Get predictions from each model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_predictions = []

    for i, model in enumerate(models):
        preds = []
        with torch.no_grad():
            for batch in test_dl:
                x, _ = batch
                x = x.to(device)
                pred = model(x, return_logits=True)
                preds.extend(pred.cpu().numpy().flatten().tolist())
        all_predictions.append(np.array(preds))
        print(f"  Model {i+1} predictions: mean={np.mean(preds):.4f}, std={np.std(preds):.4f}")

    # Ground truth
    targets = test_data["label"].values

    # Evaluate individual models
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*70)
    print(f"{'Model':<10} {'Spearman ρ':>12} {'Pearson r':>12} {'RMSE':>10}")
    print("-"*50)

    for i, preds in enumerate(all_predictions):
        spearman_r, _ = stats.spearmanr(preds, targets)
        pearson_r, _ = stats.pearsonr(preds, targets)
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        print(f"Model {i+1:<4} {spearman_r:>12.4f} {pearson_r:>12.4f} {rmse:>10.4f}")

    # Ensemble by averaging
    ensemble_preds = np.mean(all_predictions, axis=0)

    spearman_r, spearman_p = stats.spearmanr(ensemble_preds, targets)
    pearson_r, pearson_p = stats.pearsonr(ensemble_preds, targets)
    rmse = np.sqrt(np.mean((ensemble_preds - targets) ** 2))
    ss_res = np.sum((targets - ensemble_preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("-"*50)
    print(f"{'Ensemble':<10} {spearman_r:>12.4f} {pearson_r:>12.4f} {rmse:>10.4f}")
    print("="*70)

    print(f"\n=== ENSEMBLE RESULTS ===")
    print(f"  R²:         {r2:.4f}")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  RMSE:       {rmse:.4f}")

    # Compare to baseline
    baseline_rho = 0.39
    improvement = (spearman_r - baseline_rho) / baseline_rho * 100
    print(f"\n  Baseline (single model): ρ={baseline_rho}")
    print(f"  Improvement: {improvement:+.1f}%")

    # Best individual vs ensemble
    best_individual = max(stats.spearmanr(p, targets)[0] for p in all_predictions)
    ensemble_vs_best = (spearman_r - best_individual) / best_individual * 100
    print(f"\n  Best individual: ρ={best_individual:.4f}")
    print(f"  Ensemble vs best individual: {ensemble_vs_best:+.1f}%")


if __name__ == "__main__":
    evaluate_ensemble()
