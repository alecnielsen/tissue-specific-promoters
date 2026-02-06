#!/usr/bin/env python3
"""Analyze top promoter sequences from optimization run."""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import re

# Load data
RESULTS_DIR = Path(__file__).parent.parent / "results" / "dual_on_hek293_20260203_215622"
df = pd.read_csv(RESULTS_DIR / "top_100_sequences.csv")

print("=" * 70)
print("TOP SEQUENCE ANALYSIS")
print("=" * 70)

# Basic stats
print(f"\n## Score Summary (n={len(df)})")
print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 60)
for col in ['JURKAT', 'THP1', 'HEK293', 'reward']:
    print(f"{col:<20} {df[col].mean():>10.3f} {df[col].std():>10.3f} {df[col].min():>10.3f} {df[col].max():>10.3f}")

# Specificity ratios
df['JURKAT_HEK_ratio'] = df['JURKAT'] / df['HEK293']
df['THP1_HEK_ratio'] = df['THP1'] / df['HEK293']
print(f"\n## Specificity Ratios")
print(f"JURKAT/HEK293: {df['JURKAT_HEK_ratio'].mean():.2f}x (range: {df['JURKAT_HEK_ratio'].min():.2f}-{df['JURKAT_HEK_ratio'].max():.2f})")
print(f"THP1/HEK293:   {df['THP1_HEK_ratio'].mean():.2f}x (range: {df['THP1_HEK_ratio'].min():.2f}-{df['THP1_HEK_ratio'].max():.2f})")

# Nucleotide composition
print(f"\n## Nucleotide Composition")
def calc_composition(seq):
    seq = seq.upper()
    n = len(seq)
    return {
        'A': seq.count('A') / n,
        'T': seq.count('T') / n,
        'G': seq.count('G') / n,
        'C': seq.count('C') / n,
        'GC': (seq.count('G') + seq.count('C')) / n,
        'length': n
    }

compositions = df['sequence'].apply(calc_composition).apply(pd.Series)
df = pd.concat([df, compositions], axis=1)

print(f"{'Nucleotide':<12} {'Mean %':>10} {'Std %':>10}")
print("-" * 32)
for nt in ['A', 'T', 'G', 'C', 'GC']:
    print(f"{nt:<12} {compositions[nt].mean()*100:>10.1f} {compositions[nt].std()*100:>10.1f}")
print(f"Sequence length: {compositions['length'].iloc[0]} bp")

# Dinucleotide frequencies
print(f"\n## Top 10 Dinucleotide Frequencies")
all_seqs = ''.join(df['sequence'].tolist())
dinucs = [all_seqs[i:i+2] for i in range(len(all_seqs)-1)]
dinuc_counts = Counter(dinucs)
total_dinucs = sum(dinuc_counts.values())
for dinuc, count in dinuc_counts.most_common(10):
    pct = count / total_dinucs * 100
    print(f"  {dinuc}: {pct:.1f}%")

# CpG analysis
print(f"\n## CpG Analysis")
cpg_counts = [seq.count('CG') for seq in df['sequence']]
print(f"CpG per sequence: {np.mean(cpg_counts):.1f} ± {np.std(cpg_counts):.1f}")
print(f"CpG observed/expected: {dinuc_counts['CG']/total_dinucs / (compositions['C'].mean() * compositions['G'].mean()):.2f}")

# Motif analysis - look for common patterns
print(f"\n## Common Motifs (k-mers)")
def find_kmers(seqs, k):
    kmers = []
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmers.append(seq[i:i+k])
    return Counter(kmers)

for k in [6, 8]:
    print(f"\nTop 10 {k}-mers:")
    kmer_counts = find_kmers(df['sequence'].tolist(), k)
    total = sum(kmer_counts.values())
    for kmer, count in kmer_counts.most_common(10):
        pct = count / total * 100
        # Check if it's a known TFBS pattern
        print(f"  {kmer}: {count} ({pct:.2f}%)")

# Known TFBS motif search
print(f"\n## Known TFBS Motif Hits")
tfbs_patterns = {
    'SP1 (GGGCGG)': 'GGGCGG',
    'SP1 alt (CCGCCC)': 'CCGCCC',
    'NF-kB (GGGRNNTYYCC)': 'GGG[AG][ACGT]{2}[CT][CT][CT]CC',
    'AP-1 (TGAGTCA)': 'TGA[GC]TCA',
    'CREB (TGACGTCA)': 'TGACGTCA',
    'ETS (GGAA)': 'GGAA',
    'GATA (GATA)': '[AT]GATA[AG]',
    'E-box (CANNTG)': 'CA[ACGT]{2}TG',
    'GC-box': 'GGGGCGGGG',
    'TATA-box': 'TATA[AT]A',
    'Initiator (INR)': '[CT][CT]A[ACGT][CT][CT]',
}

for name, pattern in tfbs_patterns.items():
    hits = sum(len(re.findall(pattern, seq, re.IGNORECASE)) for seq in df['sequence'])
    avg_hits = hits / len(df)
    if hits > 0:
        print(f"  {name}: {hits} total hits ({avg_hits:.1f}/seq)")

# Sequence diversity - pairwise similarity
print(f"\n## Sequence Diversity")
def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# Sample pairwise distances (full matrix would be 100x100)
np.random.seed(42)
n_samples = 500
seqs = df['sequence'].tolist()
distances = []
for _ in range(n_samples):
    i, j = np.random.choice(len(seqs), 2, replace=False)
    distances.append(hamming_distance(seqs[i], seqs[j]))

print(f"Pairwise Hamming distance (sampled):")
print(f"  Mean: {np.mean(distances):.1f} bp ({np.mean(distances)/len(seqs[0])*100:.1f}%)")
print(f"  Range: {np.min(distances)}-{np.max(distances)} bp")

# Check for near-duplicates
print(f"\nNear-duplicate check (≤10 bp difference):")
near_dupes = 0
for i in range(len(seqs)):
    for j in range(i+1, len(seqs)):
        if hamming_distance(seqs[i], seqs[j]) <= 10:
            near_dupes += 1
print(f"  Found {near_dupes} near-duplicate pairs")

# Positional nucleotide bias
print(f"\n## Positional Bias (5' vs 3' half)")
first_half = [seq[:125] for seq in df['sequence']]
second_half = [seq[125:] for seq in df['sequence']]

first_gc = np.mean([(s.count('G') + s.count('C'))/len(s) for s in first_half])
second_gc = np.mean([(s.count('G') + s.count('C'))/len(s) for s in second_half])
print(f"  5' half GC: {first_gc*100:.1f}%")
print(f"  3' half GC: {second_gc*100:.1f}%")

# Top 10 best sequences summary
print(f"\n## Top 10 Sequences")
print(f"{'Rank':<6} {'JURKAT':>8} {'THP1':>8} {'HEK293':>8} {'Reward':>8} {'GC%':>6}")
print("-" * 50)
for i, row in df.head(10).iterrows():
    print(f"{i+1:<6} {row['JURKAT']:>8.3f} {row['THP1']:>8.3f} {row['HEK293']:>8.3f} {row['reward']:>8.3f} {row['GC']*100:>5.1f}%")

# Correlation analysis
print(f"\n## Score Correlations")
print(f"  JURKAT-THP1:   r={df['JURKAT'].corr(df['THP1']):.3f}")
print(f"  JURKAT-HEK293: r={df['JURKAT'].corr(df['HEK293']):.3f}")
print(f"  THP1-HEK293:   r={df['THP1'].corr(df['HEK293']):.3f}")
print(f"  GC%-JURKAT:    r={df['GC'].corr(df['JURKAT']):.3f}")
print(f"  GC%-HEK293:    r={df['GC'].corr(df['HEK293']):.3f}")

# Save enhanced dataframe
output_path = RESULTS_DIR / "top_100_analyzed.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved enhanced data to {output_path}")
