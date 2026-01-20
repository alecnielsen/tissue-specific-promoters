#!/usr/bin/env python
"""
Prepare all data required for Ctrl-DNA tissue-specific promoter design.

This script:
1. Downloads MPRA data and generates RL initialization files
2. Downloads JASPAR motifs and creates TFBS files
3. Generates TFBS frequency files per cell type
4. Creates the complete data directory structure

Usage:
    python scripts/prepare_data.py --download_all
    python scripts/prepare_data.py --tfbs_only
"""

import argparse
import os
import sys
import shlex
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Constants
CELL_TYPES = ["JURKAT", "K562", "THP1"]
SEQ_LEN = 250

# JASPAR 2024 vertebrate core motifs (MEME format)
JASPAR_URL = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"


def download_file(url: str, dest: Path) -> bool:
    """Download file if it doesn't exist."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True

    print(f"  Downloading {url}")
    cmd = f"curl -L '{url}' -o {shlex.quote(str(dest))} 2>/dev/null"
    ret = os.system(cmd)
    if ret != 0 or not dest.exists():
        print(f"  Failed to download {url}")
        return False
    return True


def download_mpra_data(cache_dir: Path) -> tuple[Path, Path]:
    """Download raw MPRA data from Google Drive."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    counts_path = cache_dir / "Raw_Promoter_Counts.csv"
    if not counts_path.exists():
        print("Downloading Raw_Promoter_Counts.csv...")
        cmd = f"curl -L 'https://drive.google.com/uc?export=download&id=15p6GhDop5BsUPryZ6pfKgwJ2XEVHRAYq' -o {shlex.quote(str(counts_path))}"
        os.system(cmd)

    seqs_path = cache_dir / "final_list_of_all_promoter_sequences_fixed.tsv"
    if not seqs_path.exists():
        print("Downloading sequence list...")
        cmd = f"curl -L 'https://drive.google.com/uc?export=download&id=1kTfsZvsCz7EWUhl-UZgK0B31LtxJH4qG' -o {shlex.quote(str(seqs_path))}"
        os.system(cmd)

    return counts_path, seqs_path


def process_mpra_data(counts_path: Path, seqs_path: Path, min_reads: int = 5) -> pd.DataFrame:
    """Process raw MPRA counts into expression values."""
    print("Processing MPRA data...")
    num_replicates = 2

    measurements = pd.read_csv(counts_path)

    # Filter by minimum reads
    measurements["keep"] = True
    for col in measurements.columns:
        if col.endswith("_sum") and col != "cum_sum":
            measurements["keep"] = measurements["keep"] & (measurements[col] >= min_reads)
    measurements = measurements[measurements["keep"]].drop("keep", axis=1).reset_index(drop=True)
    print(f"  Kept {len(measurements)} sequences with >= {min_reads} reads")

    # Normalize and compute expression
    for col in measurements.columns:
        if not (col.endswith("_sum") or col == "sequence"):
            measurements[col] = measurements[col] + 1.0
            measurements[col] = measurements[col] / measurements[col].sum()

    for cell in CELL_TYPES:
        first_letter = cell[0]
        measurements[cell] = 0
        for rep in range(num_replicates):
            p4_col = f"{first_letter}{rep+1}_P4"
            p7_col = f"{first_letter}{rep+1}_P7"
            measurements[cell] += np.log2(measurements[p4_col] / measurements[p7_col])
        measurements[cell] /= num_replicates

    # Merge with sequence properties
    seq_props = pd.read_csv(seqs_path, sep="\t")
    merged = measurements.merge(seq_props, on="sequence", how="inner")
    print(f"  Merged: {len(merged)} sequences")

    return merged[["sequence"] + CELL_TYPES]


def generate_rl_init_data(df: pd.DataFrame, output_dir: Path, top_n: int = 1000):
    """Generate RL initialization data for each cell type."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for cell in CELL_TYPES:
        sorted_df = df.sort_values(cell, ascending=False).head(top_n).copy()

        # Rename columns to expected format
        output = sorted_df[["sequence"] + CELL_TYPES].copy()
        output.columns = ["sequence"] + [f"{c}_mean" for c in CELL_TYPES]

        output_path = output_dir / f"{cell}_hard.csv"
        output.to_csv(output_path, index=False)
        print(f"  Saved: {output_path} ({len(output)} sequences)")


def download_jaspar_motifs(tfbs_dir: Path) -> Optional[Path]:
    """Download JASPAR 2024 motifs in MEME format."""
    tfbs_dir.mkdir(parents=True, exist_ok=True)

    # Use a consistent filename
    meme_path = tfbs_dir / "20250424153556_JASPAR2024_combined_matrices_735317_meme.txt"

    if meme_path.exists():
        print(f"  JASPAR motifs already exist: {meme_path}")
        return meme_path

    print("Downloading JASPAR 2024 motifs...")
    if download_file(JASPAR_URL, meme_path):
        return meme_path
    return None


def create_selected_ppms(tfbs_dir: Path) -> Path:
    """Create selected_ppms.csv with immune-relevant TFs."""
    ppms_path = tfbs_dir / "selected_ppms.csv"

    if ppms_path.exists():
        print(f"  Selected PPMs already exist: {ppms_path}")
        return ppms_path

    # Key immune cell transcription factors
    # These are JASPAR matrix IDs for TFs important in T-cells, macrophages, and hematopoiesis
    immune_tfs = [
        # T-cell TFs
        "MA0523.1",  # TCF7 (T-cell factor)
        "MA0002.2",  # RUNX1
        "MA0139.1",  # CTCF
        "MA0107.1",  # RELA (NF-kB)
        "MA0105.4",  # NFKB1
        "MA0099.3",  # AP1 (JUN::FOS)
        "MA0476.1",  # FOS
        "MA0488.1",  # JUN
        "MA0150.2",  # NFE2L2 (NRF2)
        "MA0517.1",  # STAT1::STAT2
        "MA0144.2",  # STAT3

        # Macrophage TFs
        "MA0080.5",  # SPI1 (PU.1)
        "MA0466.2",  # CEBPB
        "MA0102.4",  # CEBPA
        "MA0062.3",  # GABPA (ETS factor)
        "MA0098.3",  # ETS1
        "MA0081.2",  # SPIB
        "MA0765.2",  # ETV6

        # Hematopoietic TFs (K562)
        "MA0035.4",  # GATA1
        "MA0036.3",  # GATA2
        "MA0140.2",  # GATA3
        "MA0079.4",  # SP1
        "MA0003.4",  # TFAP2A

        # General regulatory TFs
        "MA0004.1",  # Arnt
        "MA0006.1",  # Ahr::Arnt
        "MA0058.3",  # MAX
        "MA0059.1",  # MAX::MYC
        "MA0147.3",  # MYC
        "MA0119.1",  # MAFK::NFE2L1
        "MA0506.1",  # NRF1
    ]

    # Create DataFrame
    ppms_df = pd.DataFrame({"Matrix_id": immune_tfs})
    ppms_df.to_csv(ppms_path, index=False)
    print(f"  Created: {ppms_path} ({len(immune_tfs)} TFs)")

    return ppms_path


def generate_tfbs_frequencies(
    df: pd.DataFrame,
    tfbs_dir: Path,
    output_dir: Path,
    meme_path: Path,
    ppms_path: Path
):
    """Generate TFBS frequency files per cell type using motif scanning."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if pymemesuite is available
    try:
        from pymemesuite.common import MotifFile, Sequence
        from pymemesuite.fimo import FIMO
    except ImportError:
        print("  Warning: pymemesuite not installed. Creating placeholder TFBS files.")
        print("  Install with: pip install pymemesuite")
        _create_placeholder_tfbs(df, output_dir, ppms_path)
        return

    # Load motifs
    print("Loading motifs...")
    motiffile = MotifFile(str(meme_path))
    motifs = []
    while True:
        motif = motiffile.read()
        if motif is None:
            break
        motifs.append(motif)
    bg = motiffile.background
    print(f"  Loaded {len(motifs)} motifs")

    # Load selected TFs
    selected = pd.read_csv(ppms_path).Matrix_id.tolist()
    motifs = [m for m in motifs if m.name.decode() in selected]
    print(f"  Selected {len(motifs)} motifs")

    # Create sequences
    sequences = [
        Sequence(row.sequence, name=str(row.Index).encode())
        for row in df.itertuples()
    ]

    # Scan for motifs
    print("Scanning sequences for motifs (this may take a while)...")
    fimo = FIMO(both_strands=True, threshold=0.001)

    from collections import defaultdict
    d = defaultdict(list)

    for motif in motifs:
        matches = fimo.score_motif(motif, sequences, bg).matched_elements
        for m in matches:
            d['Matrix_id'].append(motif.name.decode())
            d['SeqID'].append(m.source.accession.decode())
            d['strand'].append(m.strand)
            d['start'].append(m.start if m.strand == '+' else m.stop)

    tfbs_sites = pd.DataFrame(d)
    print(f"  Found {len(tfbs_sites)} motif occurrences")

    # Generate frequency files per cell type
    for cell in CELL_TYPES:
        # Get top sequences for this cell type
        top_seqs = df.nlargest(1000, cell).index.astype(str).tolist()
        cell_sites = tfbs_sites[tfbs_sites.SeqID.isin(top_seqs)]

        # Compute frequency pivot table
        freq_df = pd.pivot_table(
            cell_sites,
            values="start",
            index="SeqID",
            columns="Matrix_id",
            aggfunc="count"
        ).fillna(0)

        # Add SeqID as column (expected format)
        freq_df = freq_df.reset_index()

        output_path = output_dir / f"{cell}_tfbs_freq_all.csv"
        freq_df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")


def _create_placeholder_tfbs(df: pd.DataFrame, output_dir: Path, ppms_path: Path):
    """Create placeholder TFBS files when pymemesuite is not available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(ppms_path).Matrix_id.tolist()

    for cell in CELL_TYPES:
        # Create empty frequency table with correct columns
        freq_df = pd.DataFrame(columns=["SeqID"] + selected)

        # Add a few rows with zeros
        top_seqs = df.nlargest(100, cell).index.tolist()
        for seq_id in top_seqs[:10]:
            row = {"SeqID": str(seq_id)}
            row.update({tf: 0 for tf in selected})
            freq_df = pd.concat([freq_df, pd.DataFrame([row])], ignore_index=True)

        output_path = output_dir / f"{cell}_tfbs_freq_all.csv"
        freq_df.to_csv(output_path, index=False)
        print(f"  Saved placeholder: {output_path}")


def print_fitness_stats(df: pd.DataFrame):
    """Print fitness statistics needed for Ctrl-DNA normalization."""
    print("\n=== Fitness Statistics (for base_optimizer.py) ===")
    for cell in CELL_TYPES:
        min_val = df[cell].min()
        max_val = df[cell].max()
        print(f"'{cell}': min_fitness={min_val:.6f}, max_fitness={max_val:.6f}")
    print("===================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Ctrl-DNA")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--download_all", action="store_true",
                        help="Download and process all data")
    parser.add_argument("--tfbs_only", action="store_true",
                        help="Only regenerate TFBS files")
    parser.add_argument("--skip_tfbs_scan", action="store_true",
                        help="Skip TFBS scanning (create placeholders)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir

    print("=== Ctrl-DNA Data Preparation ===\n")

    # Step 1: Download MPRA data
    print("Step 1: MPRA Data")
    cache_dir = data_dir / "mpra_cache"
    counts_path, seqs_path = download_mpra_data(cache_dir)

    # Step 2: Process MPRA data
    processed_path = cache_dir / "processed_expression.csv"
    if processed_path.exists():
        print(f"  Loading cached: {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        df = process_mpra_data(counts_path, seqs_path)
        df.to_csv(processed_path, index=False)
        print(f"  Saved: {processed_path}")

    print_fitness_stats(df)

    # Step 3: Generate RL initialization data
    print("Step 2: RL Initialization Data")
    rl_data_dir = data_dir / "human_promoters" / "rl_data_large"
    generate_rl_init_data(df, rl_data_dir)

    # Step 4: Download JASPAR motifs
    print("\nStep 3: JASPAR Motifs")
    tfbs_dir = data_dir / "TFBS"
    meme_path = download_jaspar_motifs(tfbs_dir)
    if meme_path is None:
        print("  Warning: Could not download JASPAR motifs")
        return

    # Step 5: Create selected PPMs
    ppms_path = create_selected_ppms(tfbs_dir)

    # Step 6: Generate TFBS frequency files
    print("\nStep 4: TFBS Frequencies")
    tfbs_freq_dir = data_dir / "human_promoters" / "tfbs"
    if args.skip_tfbs_scan:
        _create_placeholder_tfbs(df, tfbs_freq_dir, ppms_path)
    else:
        generate_tfbs_frequencies(df, tfbs_dir, tfbs_freq_dir, meme_path, ppms_path)

    print("\n=== Data Preparation Complete ===")
    print(f"\nDirectory structure:")
    print(f"  {data_dir}/")
    print(f"  ├── mpra_cache/")
    print(f"  │   ├── Raw_Promoter_Counts.csv")
    print(f"  │   ├── final_list_of_all_promoter_sequences_fixed.tsv")
    print(f"  │   └── processed_expression.csv")
    print(f"  ├── TFBS/")
    print(f"  │   ├── *_meme.txt (JASPAR motifs)")
    print(f"  │   └── selected_ppms.csv")
    print(f"  └── human_promoters/")
    print(f"      ├── rl_data_large/")
    print(f"      │   ├── JURKAT_hard.csv")
    print(f"      │   ├── K562_hard.csv")
    print(f"      │   └── THP1_hard.csv")
    print(f"      └── tfbs/")
    print(f"          ├── JURKAT_tfbs_freq_all.csv")
    print(f"          ├── K562_tfbs_freq_all.csv")
    print(f"          └── THP1_tfbs_freq_all.csv")


if __name__ == "__main__":
    main()
