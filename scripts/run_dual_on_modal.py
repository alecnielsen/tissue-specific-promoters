"""
Run Ctrl-DNA dual ON optimization on Modal GPU.

Usage:
    # Run optimization with defaults
    modal run scripts/run_dual_on_modal.py

    # Quick test (1 iteration, 1 epoch)
    modal run scripts/run_dual_on_modal.py --max-iter 1 --epochs 1

    # Full run with custom params
    modal run scripts/run_dual_on_modal.py --max-iter 100 --epochs 5

Cost estimate: ~$5-15 depending on iterations (T4 GPU)
"""

import modal
from pathlib import Path

app = modal.App(name="ctrl-dna-dual-on-optimization")

REPO_ROOT = Path(__file__).parent.parent.resolve()
HF_CACHE = Path.home() / ".cache" / "huggingface"

# Image with all dependencies - pinned versions for fast resolution
optimization_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2+cu121",
        "torchvision==0.16.2+cu121",
        "torchaudio==2.1.2+cu121",
        "pytorch-lightning==1.9.5",
        "pandas==2.0.3",
        "numpy==1.26.4",
        "scipy==1.11.4",
        "transformers==4.44.0",
        "einops==0.7.0",
        "safetensors==0.4.3",
        "huggingface_hub==0.23.4",
        "anndata==0.10.8",
        "enformer-pytorch==0.8.10",
        "pymemesuite",
        "wandb",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
    # Copy flattened HyenaDNA model (no symlinks)
    .add_local_dir(str(REPO_ROOT / "data" / "hyenadna_model"),
                   remote_path="/hyenadna_model",
                   copy=True)
)

# Volume with trained checkpoints
checkpoint_volume = modal.Volume.from_name("ctrl-dna-checkpoints", create_if_missing=True)

# Volume to persist optimization results
results_volume = modal.Volume.from_name("ctrl-dna-results", create_if_missing=True)


@app.function(
    image=optimization_image,
    gpu="A10G",  # 24GB - T4 (14GB) runs out of memory
    timeout=14400,  # 4 hours max
    volumes={
        "/checkpoints": checkpoint_volume,
        "/results": results_volume,
    },
)
def run_optimization(
    max_iter: int = 100,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-4,
    on_weight: float = 0.4,
    off_constraint: float = 0.3,
    seed: int = 0,
    wandb_log: bool = False,
) -> dict:
    """Run dual ON optimization on GPU."""
    import os
    import sys
    import torch
    import numpy as np
    import random
    import pandas as pd
    import json
    from datetime import datetime

    # Set up paths
    sys.path.insert(0, "/repo/Ctrl-DNA/ctrl_dna")
    sys.path.insert(0, "/repo/Ctrl-DNA")
    os.chdir("/repo/Ctrl-DNA/ctrl_dna")

    # Copy checkpoints from volume to expected location
    os.makedirs("/repo/checkpoints", exist_ok=True)
    import shutil
    for ckpt in ["human_paired_jurkat.ckpt", "human_paired_k562.ckpt", "human_paired_THP1.ckpt"]:
        src = f"/checkpoints/{ckpt}"
        dst = f"/repo/checkpoints/{ckpt}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {ckpt} to /repo/checkpoints/")

    print(f"\n{'='*60}")
    print("DUAL ON OPTIMIZATION - Modal GPU")
    print(f"{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Max iterations: {max_iter}")
    print(f"Epochs per iteration: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"ON weight: {on_weight}")
    print(f"OFF constraint: {off_constraint}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import after path setup
    from dna_optimizers_multi.base_optimizer import get_fitness_info
    from dna_optimizers_multi.lagrange_optimizer import Lagrange_optimizer

    # Create config namespace
    class Config:
        pass

    cfg = Config()
    cfg.lr = lr
    cfg.batch_size = batch_size
    cfg.device = "cuda"
    cfg.seed = seed
    cfg.max_oracle_calls = 4000000
    cfg.max_strings = 384000000
    cfg.max_iter = max_iter
    cfg.epoch = epochs
    cfg.wandb_log = wandb_log
    cfg.train_log_interval = 1
    cfg.env_log_interval = 256
    cfg.out_dir = "/results"
    cfg.project_name = "Ctrl_DNA_Immune"
    cfg.level = "hard"
    cfg.e_size = 100
    cfg.e_batch_size = 24
    cfg.priority = True
    cfg.beta = 0.01
    cfg.epsilon = 0.2
    cfg.grpo = True
    cfg.num_rewards = 3
    cfg.top = 10
    cfg.oracle_type = "paired"
    cfg.lambda_lr = 3e-4
    cfg.optimizer = "Adam"
    cfg.lambda_value = [0.1, 0.9]
    cfg.on_weight = on_weight
    cfg.off_constraint = off_constraint
    cfg.tfbs = False
    cfg.tfbs_lambda = 1
    cfg.tfbs_ratio = 0.01
    cfg.tfbs_upper = 0.01
    cfg.lambda_upper = 1.0
    cfg.tfbs_dir = "/repo/data/TFBS"
    cfg.data_dir = "/repo/data"
    cfg.checkpoint_dir = "/repo/checkpoints"

    # Get sequence length
    cfg.max_len, _, _ = get_fitness_info("JURKAT", cfg.oracle_type)

    # TFBS paths (not used but needed for init)
    cfg.meme_path = f"{cfg.tfbs_dir}/20250424153556_JASPAR2024_combined_matrices_735317_meme.txt"
    cfg.ppms_path = f"{cfg.tfbs_dir}/selected_ppms.csv"

    cfg.wandb_run_name = (
        f"dual_ON_JURKAT_THP1_vs_K562_"
        f"lr_{cfg.lr}_"
        f"epoch_{cfg.epoch}_"
        f"seed_{cfg.seed}_"
        f"on_weight_{cfg.on_weight}_"
        f"off_constraint_{cfg.off_constraint}"
    )

    # Define DualOnOptimizer inline
    class DualOnOptimizer(Lagrange_optimizer):
        """Modified Lagrange optimizer for dual ON targets."""

        def __init__(self, cfg):
            cfg.task = "JURKAT"
            cfg.prefix_label = "100"
            cfg.constraint = [cfg.off_constraint, 0.0, 0.0]
            super().__init__(cfg)
            self.on_weight = cfg.on_weight
            self.off_constraint = cfg.off_constraint
            self.on_indices = [0, 2]
            self.off_index = 1

        @torch.no_grad()
        def score_enformer(self, dna):
            if len(self.dna_buffer) > self.max_oracle_calls:
                return torch.zeros(3)

            scores_dict = {}
            for cell, model in self.targets.items():
                raw_score = model([dna]).squeeze(0).item()
                scores_dict[cell] = self.normalize_target(raw_score, cell)

            scores = [scores_dict['JURKAT'], scores_dict['K562'], scores_dict['THP1']]

            reward = (
                self.on_weight * scores_dict['JURKAT']
                + self.on_weight * scores_dict['THP1']
                - (scores_dict['K562'] - self.off_constraint)
            )

            if dna in self.dna_buffer:
                self.dna_buffer[dna][3] += 1
                self.redundant_count += 1
            else:
                self.dna_buffer[dna] = [torch.tensor(scores), reward, len(self.dna_buffer) + 1, 1]

            return self.dna_buffer[dna][0]

        def compute_combined_score(self, scores_multi, cfg):
            jurkat_scores = scores_multi[:, 0]
            k562_scores = scores_multi[:, 1]
            thp1_scores = scores_multi[:, 2]
            scores = (
                self.on_weight * jurkat_scores
                + self.on_weight * thp1_scores
                - (k562_scores - self.off_constraint)
            )
            return scores

    print("Initializing optimizer...")
    optimizer = DualOnOptimizer(cfg)

    # Load starting sequences
    data_dir = cfg.data_dir
    init_data_path = f'{data_dir}/human_promoters/rl_data_large/JURKAT_hard.csv'
    cell_types = ['JURKAT', 'K562', 'THP1']

    starting_sequences = pd.read_csv(init_data_path)

    for cell in cell_types:
        _, min_fitness, max_fitness = get_fitness_info(cell, cfg.oracle_type)
        col_name = f"{cell}_mean"
        starting_sequences[col_name] = (starting_sequences[col_name] - min_fitness) / (max_fitness - min_fitness)

    starting_sequences['target'] = (
        cfg.on_weight * starting_sequences['JURKAT_mean']
        + cfg.on_weight * starting_sequences['THP1_mean']
        - (starting_sequences['K562_mean'] - cfg.off_constraint)
    )

    starting_sequences['rewards'] = starting_sequences[
        ['JURKAT_mean', 'K562_mean', 'THP1_mean']
    ].values.tolist()

    starting_sequences = starting_sequences.sort_values('target', ascending=False).head(128)

    print(f"\nStarting sequences (top 5):")
    print(starting_sequences[['sequence', 'JURKAT_mean', 'THP1_mean', 'K562_mean', 'target']].head())

    print("\n" + "="*60)
    print("STARTING OPTIMIZATION")
    print("="*60 + "\n")

    # Run optimization (Ctrl-DNA calls sys.exit(0) when done, which we catch)
    try:
        optimizer.optimize(cfg, starting_sequences)
    except SystemExit:
        pass  # Normal termination from Ctrl-DNA

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/results/dual_on_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Extract top sequences from buffer
    buffer_items = list(optimizer.dna_buffer.items())
    results_data = []
    for seq, (scores, reward, order, count) in buffer_items:
        results_data.append({
            "sequence": seq,
            "JURKAT": scores[0].item() if torch.is_tensor(scores[0]) else scores[0],
            "K562": scores[1].item() if torch.is_tensor(scores[1]) else scores[1],
            "THP1": scores[2].item() if torch.is_tensor(scores[2]) else scores[2],
            "reward": reward,
            "order": order,
            "count": count,
        })

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values("reward", ascending=False)

    # Save full results
    results_df.to_csv(f"{results_dir}/all_sequences.csv", index=False)

    # Save top 100
    top_100 = results_df.head(100)
    top_100.to_csv(f"{results_dir}/top_100_sequences.csv", index=False)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "config": {
            "max_iter": max_iter,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "on_weight": on_weight,
            "off_constraint": off_constraint,
            "seed": seed,
        },
        "results": {
            "total_sequences": len(results_df),
            "top_reward": float(results_df["reward"].max()),
            "top_10_avg_reward": float(results_df.head(10)["reward"].mean()),
            "top_10_avg_jurkat": float(results_df.head(10)["JURKAT"].mean()),
            "top_10_avg_thp1": float(results_df.head(10)["THP1"].mean()),
            "top_10_avg_k562": float(results_df.head(10)["K562"].mean()),
        }
    }

    with open(f"{results_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Commit volume
    results_volume.commit()

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_dir}")
    print(f"Total sequences evaluated: {len(results_df)}")
    print(f"Top reward: {summary['results']['top_reward']:.4f}")
    print(f"Top 10 avg JURKAT: {summary['results']['top_10_avg_jurkat']:.4f}")
    print(f"Top 10 avg THP1: {summary['results']['top_10_avg_thp1']:.4f}")
    print(f"Top 10 avg K562: {summary['results']['top_10_avg_k562']:.4f}")
    print("="*60 + "\n")

    return summary


@app.local_entrypoint()
def main(
    max_iter: int = 100,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-4,
    on_weight: float = 0.4,
    off_constraint: float = 0.3,
    seed: int = 0,
    wandb_log: bool = False,
):
    """Run dual ON optimization on Modal GPU."""
    print("Launching optimization on Modal...")

    result = run_optimization.remote(
        max_iter=max_iter,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        on_weight=on_weight,
        off_constraint=off_constraint,
        seed=seed,
        wandb_log=wandb_log,
    )

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total sequences: {result['results']['total_sequences']}")
    print(f"Top reward: {result['results']['top_reward']:.4f}")
    print(f"Top 10 avg JURKAT: {result['results']['top_10_avg_jurkat']:.4f}")
    print(f"Top 10 avg THP1: {result['results']['top_10_avg_thp1']:.4f}")
    print(f"Top 10 avg K562: {result['results']['top_10_avg_k562']:.4f}")
    print("="*60)
    print("\nDownload results with:")
    print("  modal volume get ctrl-dna-results . ./results/")
