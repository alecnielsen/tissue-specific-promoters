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


def str2bool(v):
    """Parse boolean from string for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')

# Resolve paths before changing directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add Ctrl-DNA to path
CTRL_DNA_PATH = PROJECT_ROOT / "Ctrl-DNA" / "ctrl_dna"
sys.path.insert(0, str(CTRL_DNA_PATH))
os.chdir(CTRL_DNA_PATH)

import torch
import numpy as np
import random
import pandas as pd

from dna_optimizers_multi.base_optimizer import get_fitness_info
from dna_optimizers_multi.lagrange_optimizer import Lagrange_optimizer


def validate_tfbs_files(data_dir: str, tfbs_enabled: bool) -> None:
    """
    Validate TFBS files before starting optimization.

    Checks for:
    1. Placeholder marker file (created when pymemesuite was missing)
    2. All-zero TFBS frequency files

    Raises:
        SystemExit: If tfbs_enabled=True and files are invalid
    """
    tfbs_freq_dir = Path(data_dir) / "human_promoters" / "tfbs"

    # Check for placeholder marker
    marker_path = tfbs_freq_dir / ".PLACEHOLDER_WARNING"
    if marker_path.exists():
        if tfbs_enabled:
            print("\n" + "="*70)
            print("ERROR: TFBS files are PLACEHOLDERS (all zeros)")
            print("="*70)
            print(f"Marker file found: {marker_path}")
            print("")
            print("You have --tfbs enabled, but the TFBS files are invalid.")
            print("The TFBS correlation penalty would be meaningless.")
            print("")
            print("Options:")
            print("  1. Disable TFBS: Remove --tfbs flag (default is False)")
            print("  2. Fix TFBS files:")
            print("     pip install pymemesuite")
            print("     rm -rf", tfbs_freq_dir)
            print("     python scripts/prepare_data.py --download_all")
            print("="*70 + "\n")
            sys.exit(1)
        else:
            print(f"Note: TFBS files are placeholders, but --tfbs is disabled (OK)")
            return

    # Even without marker, check if files exist and have non-zero values
    if tfbs_enabled:
        cell_types = ["JURKAT", "K562", "THP1"]
        for cell in cell_types:
            tfbs_path = tfbs_freq_dir / f"{cell}_tfbs_freq_all.csv"
            if not tfbs_path.exists():
                print(f"\nERROR: TFBS file not found: {tfbs_path}")
                print("Run: python scripts/prepare_data.py --download_all")
                sys.exit(1)

            # Check if file has any non-zero values
            try:
                df = pd.read_csv(tfbs_path)
                # Drop SeqID column and check numeric columns
                numeric_cols = [c for c in df.columns if c != "SeqID"]
                if len(numeric_cols) == 0:
                    print(f"\nERROR: TFBS file has no motif columns: {tfbs_path}")
                    sys.exit(1)
                total_sum = df[numeric_cols].sum().sum()
                if total_sum == 0:
                    print(f"\nERROR: TFBS file is all zeros: {tfbs_path}")
                    print("This indicates placeholder data. TFBS optimization would be invalid.")
                    print("Run: python scripts/prepare_data.py --download_all")
                    sys.exit(1)
            except Exception as e:
                print(f"\nERROR: Could not validate TFBS file {tfbs_path}: {e}")
                sys.exit(1)

        print("TFBS files validated successfully")


def resolve_path(path_str):
    """Resolve a path string relative to the project root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def get_meme_and_ppms_path(tfbs_dir):
    """Get paths to TFBS motif files (tfbs_dir should already be resolved)."""
    tfbs_path = Path(tfbs_dir)
    meme_path = str(tfbs_path / "20250424153556_JASPAR2024_combined_matrices_735317_meme.txt")
    ppms_path = str(tfbs_path / "selected_ppms.csv")
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
    parser.add_argument("--priority", type=str2bool, default=True)

    # GRPO params
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--grpo", type=str2bool, default=True)
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

    parser.add_argument('--tfbs', default=False, type=str2bool)
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
            # Return zero tensor matching expected shape for consistency
            return torch.zeros(3)

        # Score each cell type explicitly by key (not relying on dict order)
        scores_dict = {}
        for cell, model in self.targets.items():
            raw_score = model([dna]).squeeze(0).item()
            scores_dict[cell] = self.normalize_target(raw_score, cell)

        # Build scores list in consistent order: JURKAT, K562, THP1
        # This ensures buffer format matches base_optimizer expectations
        scores = [scores_dict['JURKAT'], scores_dict['K562'], scores_dict['THP1']]

        # Dual ON reward:
        # reward = on_weight * JURKAT + on_weight * THP1 - (K562 - off_constraint)
        # The Lagrangian optimizer will handle dynamic lambda adjustment
        reward = (
            self.on_weight * scores_dict['JURKAT']  # JURKAT ON
            + self.on_weight * scores_dict['THP1']  # THP1 ON
            - (scores_dict['K562'] - self.off_constraint)  # K562 OFF with constraint
        )

        if dna in self.dna_buffer:
            self.dna_buffer[dna][3] += 1  # Increment count (index 3), not order (index 2)
            self.redundant_count += 1
        else:
            self.dna_buffer[dna] = [torch.tensor(scores), reward, len(self.dna_buffer) + 1, 1]

        return self.dna_buffer[dna][0]

    def compute_combined_score(self, scores_multi, cfg):
        """
        Override to compute dual ON target reward.
        scores_multi columns: [JURKAT (0), K562 (1), THP1 (2)]

        Reward = on_weight * JURKAT + on_weight * THP1 - (K562 - off_constraint)
        """
        jurkat_scores = scores_multi[:, 0]  # ON target 1
        k562_scores = scores_multi[:, 1]    # OFF target
        thp1_scores = scores_multi[:, 2]    # ON target 2

        scores = (
            self.on_weight * jurkat_scores
            + self.on_weight * thp1_scores
            - (k562_scores - self.off_constraint)
        )
        return scores

    def update(self, obs, old_logprobs, rewards, nonterms, episode_lens, correlations, cfg, metrics, log, iteration, epoch):
        """
        Override for dual-ON optimization (Option 1: combined reward approach).

        Key difference from parent:
        - Main advantages from combined ON signal (JURKAT + THP1)
        - Only K562 treated as constraint (single lambda)

        Indices: JURKAT=0, K562=1, THP1=2
        """
        from dna_optimizers_multi.lagrange_optimizer import get_advantages
        import numpy as np

        self.agent.train()

        # --- Lambda update for K562 constraint only ---
        # Shape must be [N, 1] for update_lambda to iterate over
        k562_cost = rewards[-1, 1, :].mean().view(1, 1)  # K562 score, shape [1, 1]
        zeros = torch.zeros(1, 1, device=k562_cost.device)

        if correlations is not None:
            corr_cost = correlations.mean().view(1, 1)
            self.update_lambda(torch.cat([k562_cost, zeros, corr_cost], dim=0))  # [3, 1]
        else:
            # Only update first lambda (K562), zero out others
            self.update_lambda(torch.cat([k562_cost, zeros], dim=0))  # [2, 1]

        # Clamp lambdas
        for i in range(3):
            self.lagrangian_multipliers[i].data.clamp_(min=0)

        lambdas = np.array([lag.detach().cpu().numpy() for lag in self.lagrangian_multipliers])
        for i in range(2):
            self.lagrangian_multipliers[i].data.clamp_(min=lambdas[i], max=self.lambda_upper)
        self.lagrangian_multipliers[2].data.clamp_(min=lambdas[2], max=self.tfbs_upper)

        # --- Compute combined ON advantages ---
        jurkat_scores = rewards[-1, 0]  # JURKAT
        thp1_scores = rewards[-1, 2]    # THP1
        k562_scores = rewards[-1, 1]    # K562

        # Combined ON signal
        combined_on = self.on_weight * jurkat_scores + self.on_weight * thp1_scores
        combined_advantages = get_advantages(combined_on)

        # K562 constraint advantages
        k562_advantages = get_advantages(k562_scores)

        # TFBS correlation advantages if present
        if correlations is not None:
            correlations_adv = get_advantages(correlations)
            total_lambda = self.lagrangian_multipliers[0] + self.lagrangian_multipliers[2]
        else:
            total_lambda = self.lagrangian_multipliers[0]

        # Soft inverse weighting (high lambda → lower boost)
        boost = torch.clamp(2 + self.tfbs_upper - total_lambda, min=1.0)

        # Final advantages: boost ON, penalize K562
        if correlations is not None:
            advantages = boost * combined_advantages - self.lagrangian_multipliers[0] * k562_advantages - self.lagrangian_multipliers[2] * correlations_adv
        else:
            advantages = boost * combined_advantages - self.lagrangian_multipliers[0] * k562_advantages

        # --- PPO update ---
        logprobs = self.agent.sequences_log_probs(obs, nonterms)
        old_per_token_logps = old_logprobs.detach().to(logprobs.device)

        coef_1 = torch.exp(logprobs - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)

        per_token_loss1 = coef_1 * advantages.unsqueeze(0)
        per_token_loss2 = coef_2 * advantages.unsqueeze(0)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # KL penalty if beta != 0
        if self.beta != 0.0:
            ref_per_token_logps = self.ref_model.sequences_log_probs(obs, nonterms)
            per_token_kl = (torch.exp(ref_per_token_logps - logprobs) - (ref_per_token_logps - logprobs) - 1)
            per_token_loss += self.beta * per_token_kl

        loss = per_token_loss.sum() / nonterms[:-1].sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()

        # --- Logging ---
        if log:
            import wandb
            step = iteration * cfg.epoch + epoch

            log_data = {
                "update/pgloss": loss.item(),
                "update/combined_on_mean": combined_on.mean().item(),
                "update/jurkat_mean": jurkat_scores.mean().item(),
                "update/thp1_mean": thp1_scores.mean().item(),
                "update/k562_mean": k562_scores.mean().item(),
                "update/kl_loss": per_token_kl.mean().item() if self.beta != 0.0 else 0.0,
                "update/advantages": advantages.mean().item(),
                "update/iteration": iteration,
                "update/epoch": epoch,
                "update/step": step,
                "update/lambda_k562": self.lagrangian_multipliers[0].item(),
            }

            if correlations is not None and cfg.tfbs:
                log_data["update/corr"] = (-correlations).mean().item()
                log_data["update/lambda_tfbs"] = self.lagrangian_multipliers[2].item()

            if cfg.epoch > 1:
                wandb.log(log_data, step=step)
            else:
                wandb.log(log_data)

            print(log_data)


def main():
    args = parse_args()

    # Resolve all paths relative to project root before they're used
    args.tfbs_dir = resolve_path(args.tfbs_dir)
    args.data_dir = resolve_path(args.data_dir)
    args.checkpoint_dir = resolve_path(args.checkpoint_dir)

    # Validate TFBS files if TFBS optimization is enabled
    validate_tfbs_files(args.data_dir, args.tfbs)

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
