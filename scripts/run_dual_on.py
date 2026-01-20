#!/usr/bin/env python
"""
Run Ctrl-DNA for dual ON targets (JURKAT + THP1 ON, K562 OFF).

This wrapper modifies the reward structure for the immune-cell-specific use case:
- ON targets: JURKAT (T-cells) + THP1 (macrophages)
- OFF target: K562 (hematopoietic progenitor, off-target proxy)

Usage:
    python scripts/run_dual_on.py \
        --epoch 5 \
        --lambda_lr 3e-4 \
        --tfbs_dir ./data/TFBS \
        --data_dir ./data \
        --checkpoint_dir ./checkpoints
"""

import argparse
import os
import sys
from pathlib import Path

# Add Ctrl-DNA to path
CTRL_DNA_PATH = Path(__file__).parent.parent / "Ctrl-DNA" / "ctrl_dna"
sys.path.insert(0, str(CTRL_DNA_PATH))
os.chdir(CTRL_DNA_PATH)

import torch
import numpy as np
import random
import pandas as pd

from dna_optimizers_multi.base_optimizer import get_fitness_info
from dna_optimizers_multi.lagrange_optimizer import Lagrange_optimizer


def get_meme_and_ppms_path(tfbs_dir):
    """Get paths to TFBS motif files."""
    meme_path = f"{tfbs_dir}/20250424153556_JASPAR2024_combined_matrices_735317_meme.txt"
    ppms_path = f"{tfbs_dir}/selected_ppms.csv"
    return meme_path, ppms_path


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Ctrl-DNA for dual ON targets (JURKAT+THP1)")

    # Basic params
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Training params
    parser.add_argument("--max_oracle_calls", type=int, default=4000000)
    parser.add_argument("--max_strings", type=int, default=384000000)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=5)

    # Logging
    parser.add_argument("--wandb_log", action='store_true')
    parser.add_argument("--train_log_interval", type=int, default=1)
    parser.add_argument("--env_log_interval", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default='results')
    parser.add_argument("--project_name", type=str, default='Ctrl_DNA_Immune')

    # Experience replay
    parser.add_argument("--level", type=str, default="hard")
    parser.add_argument("--e_size", type=int, default=100)
    parser.add_argument("--e_batch_size", type=int, default=24)
    parser.add_argument("--priority", type=bool, default=True)

    # GRPO params
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--grpo", type=bool, default=True)
    parser.add_argument("--num_rewards", type=int, default=3)
    parser.add_argument("--top", type=int, default=10)

    # Oracle params
    parser.add_argument('--oracle_type', default='paired')

    # Lagrangian params
    parser.add_argument('--lambda_lr', default=3e-4, type=float)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lambda_value', default=[0.1, 0.9], nargs='+', type=float,
                        help="Initial lambda values for OFF constraint")

    # Constraints for dual ON: [JURKAT_min, THP1_min, K562_max]
    # Positive = minimize (OFF), Negative = maximize (ON)
    parser.add_argument('--on_weight', default=0.4, type=float,
                        help="Weight for each ON target in reward")
    parser.add_argument('--off_constraint', default=0.3, type=float,
                        help="Constraint threshold for K562 (OFF target)")

    parser.add_argument('--tfbs', default=False, type=bool)
    parser.add_argument('--tfbs_lambda', type=int, default=1)
    parser.add_argument('--tfbs_ratio', default=0.01, type=float)
    parser.add_argument('--tfbs_upper', default=0.01, type=float)
    parser.add_argument('--lambda_upper', default=1.0, type=float)

    # Path configuration
    parser.add_argument('--tfbs_dir', type=str, default='./data/TFBS')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    return parser.parse_args()


class DualOnOptimizer(Lagrange_optimizer):
    """
    Modified Lagrange optimizer for dual ON targets.

    Target: JURKAT (0) + THP1 (2) ON, K562 (1) OFF

    Reward structure:
        reward = on_weight * JURKAT + on_weight * THP1 - (K562 - off_constraint)
    """

    def __init__(self, cfg):
        # Set task to JURKAT for initialization (will be overridden)
        cfg.task = "JURKAT"
        cfg.prefix_label = "100"  # Dummy, will compute dual reward
        cfg.constraint = [cfg.off_constraint, 0.0, 0.0]  # K562 constraint
        super().__init__(cfg)

        self.on_weight = cfg.on_weight
        self.off_constraint = cfg.off_constraint

        # Task indices: JURKAT=0, K562=1, THP1=2
        self.on_indices = [0, 2]  # JURKAT, THP1
        self.off_index = 1  # K562

    @torch.no_grad()
    def score_enformer(self, dna):
        """Score sequences with dual ON reward structure."""
        if len(self.dna_buffer) > self.max_oracle_calls:
            return 0

        scores = []
        for cell, model in self.targets.items():
            raw_score = model([dna]).squeeze(0).item()
            norm_score = self.normalize_target(raw_score, cell)
            scores.append(norm_score)

        # Dual ON reward:
        # reward = on_weight * JURKAT + on_weight * THP1 - lambda * (K562 - constraint)
        reward = (
            self.on_weight * scores[0]  # JURKAT ON
            + self.on_weight * scores[2]  # THP1 ON
            - 0.2 * scores[1]  # K562 OFF
        )

        if dna in self.dna_buffer:
            self.dna_buffer[dna][2] += 1
            self.redundant_count += 1
        else:
            self.dna_buffer[dna] = [torch.tensor(scores), reward, len(self.dna_buffer) + 1, 1]

        return self.dna_buffer[dna][0]


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get path info
    args.max_len, _, _ = get_fitness_info("JURKAT", args.oracle_type)
    args.meme_path, args.ppms_path = get_meme_and_ppms_path(args.tfbs_dir)

    # Build run name
    args.wandb_run_name = (
        f"dual_ON_JURKAT_THP1_vs_K562_"
        f"lr_{args.lr}_"
        f"epoch_{args.epoch}_"
        f"seed_{args.seed}_"
        f"on_weight_{args.on_weight}_"
        f"off_constraint_{args.off_constraint}"
    )

    print(f"Optimizer Arguments: {args}")
    print(f"\n=== Dual ON Target Setup ===")
    print(f"ON targets: JURKAT + THP1 (weight={args.on_weight} each)")
    print(f"OFF target: K562 (constraint={args.off_constraint})")
    print("============================\n")

    # Initialize optimizer
    optimizer = DualOnOptimizer(args)

    # Load starting sequences - use JURKAT as primary (can also use THP1)
    data_dir = args.data_dir
    init_data_path = f'{data_dir}/human_promoters/rl_data_large/JURKAT_hard.csv'
    cell_types = ['JURKAT', 'K562', 'THP1']

    starting_sequences = pd.read_csv(init_data_path)

    # Normalize scores
    for cell in cell_types:
        _, min_fitness, max_fitness = get_fitness_info(cell, args.oracle_type)
        col_name = f"{cell}_mean"
        starting_sequences[col_name] = (starting_sequences[col_name] - min_fitness) / (max_fitness - min_fitness)

    # Compute dual ON target score
    # Score = on_weight * JURKAT + on_weight * THP1 - (K562 - off_constraint)
    starting_sequences['target'] = (
        args.on_weight * starting_sequences['JURKAT_mean']
        + args.on_weight * starting_sequences['THP1_mean']
        - (starting_sequences['K562_mean'] - args.off_constraint)
    )

    starting_sequences['rewards'] = starting_sequences[
        ['JURKAT_mean', 'K562_mean', 'THP1_mean']
    ].values.tolist()

    # Sort by combined target score
    starting_sequences = starting_sequences.sort_values('target', ascending=False).head(128)

    print(f"Starting sequences (top 5):")
    print(starting_sequences[['sequence', 'JURKAT_mean', 'THP1_mean', 'K562_mean', 'target']].head())

    # Run optimization
    optimizer.optimize(args, starting_sequences)


if __name__ == "__main__":
    main()
