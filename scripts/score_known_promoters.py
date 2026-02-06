#!/usr/bin/env python3
"""Score known immune promoters with the trained oracles."""

import sys
import os
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "Ctrl-DNA" / "ctrl_dna"))
sys.path.insert(0, str(REPO_ROOT / "Ctrl-DNA"))

import torch
import numpy as np
import pandas as pd
import pathlib

# Fix for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

# Known immune promoter sequences (250bp upstream of TSS)
IMMUNE_PROMOTERS = {
    # T-cell specific
    'CD4': 'GGCTGCAGGAATGAAATGAAGTAGATGAAATTGGATTAGGTTGAGATTCTCATAAGGTGAGTTTTTAAAAAGAAAAATAATTAAATGTTTTAAATAAAAGATTTCAGAGGTAAATTTCCAGAGCTGCAGGAGAGGGAACAAAAGTTACAGCTGGGGAATGGGAGCCCTCAGCTCTCTGATGTTTCTGCTCCCTCCAGAAGTCAGGGGCTGGGCAGGCAGCTGGCGGCCACTAAAGGGAGCAGG',

    'IL2': 'ACAGGATGCAACTCCTGTCTTGCATTGCACTAAGTCTTGCACTTGTCACAAACAGTGCACCTACTTCAAGTTCTACAAAGAAAACACAGCTACAACTGGAGCATTTACTGCTGGATTTACAGATGATTTTGAATGGAATTAATAATTACAAGAATCCCAAACTCACCAGGATGCTCACATTTAAGTTTTACATGCCCAAGAAGGCCACAGAACTGAAACATCTTCAGTGTCTAGAAGAAGAA',

    'IFNG': 'TGCAGAGCCAAATTGTCTCCTTTTACTTCAAACTTTTTAAAAACTTTAAAATCAAATCAAGTTATCAATAGCAACAAAAGAAACAATCTGACTTCTTTAAAAGCATTTAAAGATAGCAGATTATTCATTAAAATAATAAACATCCTTTAATACTAGTAAACAGACATGTATCCAAAAAATCTAATAGCACTATTGATACAAAATCTAAAATTATAAAAATATTAAAAGGAGGGGAAAGATA',

    'CD8A': 'GGGCAGATGTACCCTGGGACTCAGAGAGAGCCTGGGTGACAGAGCAAGACTCCATCTCAAAAAAAAAAAAAAAAAGAAAGAAAGAAAGAGACAGAGAAAGACACTGAGGCTGTGAGCGCGGGGCAAGCATGGGCTGTGTGTGTGGGGAGGGGGCTGAGGGTGTGTGCACGTGTGTGAGGCGGGGACGAGCCGGGAGCGCCTGGGAGATAAAGACGCCCTGGAGCCGCCGCACCTGGGAGAG',

    # Myeloid/macrophage
    'TNF': 'GCTGAGCTCAGAGGGAGAGAAGCAACTACAGACCCCCCCTGAAAACAACCCTCAGACGCCACATCCCCTGACAAGCTGCCAGGCAGGTTCTCTTCCTCTCACATACTGACCCACGGCTCCACCCTCTCTCCCCTGGAAAGGACACCATGAGCACTGAAAGCATGATCCGGGACGTGGAGCTGGCCGAGGAGGCGCTCCCCAAGAAGACAGGGGGGCCCCAGGGCTCCAGGCGGTGCTTGTTC',

    'IL6': 'CCTCACAGGGAGAGCCAGAACACAGAAGAACTCAGATGACTGGTAGTATTACCTTCTTCATAATCCCAGGCTTGGGGGGCTGCGATGGAGTCAGAGGAAACTCAGTTCAGAACATCTTTGGTTTTTACAAATACAAATATTTATTTTATTTGTGAATTCAAAGTATTTATCATGCATAATGTGAGGAGACATCTTTAAAGAAATTATTTATGTACATGAGTCAAGGATGCCTTCCAAGTGGAG',

    # Cytotoxic
    'GZMB': 'CAGGAACAGAGAGAAGCCTTCTCCTTGTGGCACTGGGGCTGAACAGACTGTGGCTCTTTGAGCCCAGCAGGGACAGGAGCAGAGGCCTGAGGCCCGTGGGACAGAGTCAGGGAAGGAGCCCTGGGCTGGAGTTGACAATGGAGATGCTGCTGCTGAGCCTGGCCCCCTGGGGCCTGCAGGAGGGGGCCAGAATGTCACCCCTACCTCTGAGATCAATGGGATCAGGATCAGAATTCTCTCCC',

    'PRF1': 'TGACAGAACTCAGAAGAGGCCAGCCCCAGCCCCAGCTCTGCCAATGCCACAGATGAACCTACAGCTGCTGCAGTGGGCCAGCAGGGGGCTGCCCCAGCACCTGTGGGACATCCCCACAGATGCCACAGCAGGCCAGCCCTGGATCTATCTGGATCCCCTGCAGAAGCTTGGAGGGCAGAAGAGAGCCTCCGCTCAGCGCTGAGGGTTCTCCTGAGCCATGAGTGCCGAGGGCCTGTCAGGGC',
}

# Add some control sequences
CONTROLS = {
    # Ubiquitous promoters
    'GAPDH': 'GGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACTACATGGTTTACATGTTCCAATATGATTCCACCCATGGCAAATTCCATGGCACCGTCAAGGCTGAGAACGGGAAGCTTGTCATCAATGGAAATCCCATCACCATCTTCCAGGAGCGAGATCCCTCCAAAATCAAGTGGGGCGATGCTGGCGCTGAGTACGTCGTGGAG',

    'ACTB': 'GCGCGGCGCGGCGCGGTGGGGGGCGGCAGCGCGGCGGCGGCGGGCGCGGGGGCGCGGCGCGCTGGCGGCAGCCCCGCGCTGCGCCCGCGCCTCGGCCCCGCCCTCCCCGCGCCCGCTCGGTGAGCTGCGCGAGCGGGCCCGGCGAGCGGCGCGGGGCACCTCGCGCGCCCGCGCGCTCACTCCGCCCCATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAGGCCGGCTTCGCG',
}

def load_enformer_model(checkpoint_path: str, device: str = "cpu"):
    """Load EnformerModel from checkpoint."""
    from src.reglm.regression import EnformerModel

    # Patch torch.load to use weights_only=False for older checkpoints
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

    try:
        model = EnformerModel.load_from_checkpoint(checkpoint_path, map_location=device)
    finally:
        torch.load = original_load

    model.to(device)
    model.eval()
    return model

def get_normalization_stats(cell_type: str):
    """Get min/max fitness values for normalization."""
    from dna_optimizers_multi.base_optimizer import get_fitness_info
    _, min_val, max_val = get_fitness_info(cell_type, "paired")
    return min_val, max_val

def score_sequences(model, sequences: list, min_val: float, max_val: float, device: str = "cpu"):
    """Score sequences and normalize to [0, 1]."""
    scores = []
    with torch.no_grad():
        for seq in sequences:
            # Ensure sequence is 250bp (pad or truncate)
            if len(seq) < 250:
                seq = seq + 'N' * (250 - len(seq))
            elif len(seq) > 250:
                seq = seq[:250]

            raw_score = model([seq]).squeeze().item()
            normalized = (raw_score - min_val) / (max_val - min_val)
            scores.append(normalized)
    return scores

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_dir = REPO_ROOT / "checkpoints"

    # Load a single JURKAT model (faster than ensemble for this test)
    print("\nLoading JURKAT model...")
    jurkat_model = load_enformer_model(
        str(checkpoint_dir / "human_paired_jurkat.ckpt"), device
    )
    jurkat_min, jurkat_max = get_normalization_stats("JURKAT")

    print("Loading THP1 model...")
    thp1_model = load_enformer_model(
        str(checkpoint_dir / "human_paired_THP1.ckpt"), device
    )
    thp1_min, thp1_max = get_normalization_stats("THP1")

    # Also load K562 for comparison
    print("Loading K562 model...")
    k562_model = load_enformer_model(
        str(checkpoint_dir / "human_paired_k562.ckpt"), device
    )
    k562_min, k562_max = get_normalization_stats("K562")

    # Load optimized sequences for comparison
    opt_df = pd.read_csv(REPO_ROOT / "results" / "dual_on_hek293_20260203_215622" / "top_100_sequences.csv")
    opt_seqs = opt_df.head(10)['sequence'].tolist()

    # Combine all sequences
    all_seqs = {}
    all_seqs.update(IMMUNE_PROMOTERS)
    all_seqs.update(CONTROLS)
    all_seqs['OPT_TOP1'] = opt_seqs[0]
    all_seqs['OPT_TOP5'] = opt_seqs[4]
    all_seqs['OPT_TOP10'] = opt_seqs[9]

    print("\nScoring sequences...")
    results = []
    for name, seq in all_seqs.items():
        jurkat_score = score_sequences(jurkat_model, [seq], jurkat_min, jurkat_max, device)[0]
        thp1_score = score_sequences(thp1_model, [seq], thp1_min, thp1_max, device)[0]
        k562_score = score_sequences(k562_model, [seq], k562_min, k562_max, device)[0]

        # Determine category
        if name.startswith('OPT_'):
            category = 'Optimized'
        elif name in CONTROLS:
            category = 'Ubiquitous'
        else:
            category = 'Immune'

        results.append({
            'Name': name,
            'Category': category,
            'JURKAT': jurkat_score,
            'THP1': thp1_score,
            'K562': k562_score,
            'Avg_Immune': (jurkat_score + thp1_score) / 2,
            'GC%': (seq.count('G') + seq.count('C')) / len(seq) * 100,
        })

    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("ORACLE PREDICTIONS: KNOWN PROMOTERS vs OPTIMIZED SEQUENCES")
    print("=" * 90)
    print(f"\n{'Name':<12} {'Category':<12} {'JURKAT':>8} {'THP1':>8} {'K562':>8} {'Avg':>8} {'GC%':>6}")
    print("-" * 70)

    for _, row in df.sort_values(['Category', 'JURKAT'], ascending=[True, False]).iterrows():
        print(f"{row['Name']:<12} {row['Category']:<12} {row['JURKAT']:>8.3f} {row['THP1']:>8.3f} {row['K562']:>8.3f} {row['Avg_Immune']:>8.3f} {row['GC%']:>5.1f}%")

    print("\n" + "=" * 90)
    print("SUMMARY BY CATEGORY")
    print("=" * 90)

    summary = df.groupby('Category').agg({
        'JURKAT': 'mean',
        'THP1': 'mean',
        'K562': 'mean',
        'GC%': 'mean',
    }).round(3)

    print(f"\n{'Category':<12} {'JURKAT':>10} {'THP1':>10} {'K562':>10} {'GC%':>8}")
    print("-" * 50)
    for cat, row in summary.iterrows():
        print(f"{cat:<12} {row['JURKAT']:>10.3f} {row['THP1']:>10.3f} {row['K562']:>10.3f} {row['GC%']:>7.1f}%")

    print("\n" + "=" * 90)
    print("KEY FINDING")
    print("=" * 90)

    immune_avg = df[df['Category'] == 'Immune']['JURKAT'].mean()
    opt_avg = df[df['Category'] == 'Optimized']['JURKAT'].mean()
    ubiq_avg = df[df['Category'] == 'Ubiquitous']['JURKAT'].mean()

    print(f"""
Oracle JURKAT predictions:
  - Known immune promoters (CD4, IL2, etc.): {immune_avg:.3f}
  - Optimized sequences:                     {opt_avg:.3f}
  - Ubiquitous promoters (GAPDH, ACTB):      {ubiq_avg:.3f}

The oracles predict {'HIGHER' if opt_avg > immune_avg else 'LOWER'} expression for
optimized sequences compared to real immune promoters.

This {'confirms' if opt_avg > immune_avg else 'suggests'} the oracles learned a different
pattern than what drives natural immune-specific expression.
""")

if __name__ == "__main__":
    main()
