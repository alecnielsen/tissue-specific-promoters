"""
Run Ctrl-DNA dual ON optimization on Modal GPU with HEK293 (PARM) as OFF target.

This version uses:
- JURKAT and THP1 as ON targets (EnformerModel ENSEMBLES - 5 models each)
- HEK293 as OFF target (PARM pretrained model)

Ensemble oracles:
- JURKAT: 5 models, ensemble ρ=0.54 (+8% vs single model)
- THP1: 5 models, ensemble ρ=0.89 (+127% vs single model)

Usage:
    # Run optimization with defaults
    modal run scripts/run_dual_on_hek293_modal.py

    # Quick test (1 iteration, 1 epoch)
    modal run scripts/run_dual_on_hek293_modal.py --max-iter 1 --epochs 1

    # Full run with custom params
    modal run scripts/run_dual_on_hek293_modal.py --max-iter 100 --epochs 5

Cost estimate: ~$5-15 depending on iterations (A10G GPU)
"""

import modal
from pathlib import Path

app = modal.App(name="ctrl-dna-dual-on-hek293")

REPO_ROOT = Path(__file__).parent.parent.resolve()
HF_CACHE = Path.home() / ".cache" / "huggingface"

# Image with all dependencies including PARM
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
        "biopython",  # For PARM
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
    gpu="A10G",  # 24GB
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
    """Run dual ON optimization with HEK293 (PARM) as OFF target."""
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
    sys.path.insert(0, "/repo/PARM")  # Add PARM to path
    os.chdir("/repo/Ctrl-DNA/ctrl_dna")

    # Copy ensemble checkpoints from volume to expected location
    os.makedirs("/repo/checkpoints", exist_ok=True)
    import shutil
    import glob as glob_module

    # Copy JURKAT ensemble (5 models)
    for i in range(1, 6):
        ckpt = f"human_paired_jurkat_ensemble{i}.ckpt"
        src = f"/checkpoints/{ckpt}"
        dst = f"/repo/checkpoints/{ckpt}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {ckpt}")

    # Copy THP1 ensemble (5 models)
    for i in range(1, 6):
        ckpt = f"human_paired_THP1_ensemble{i}.ckpt"
        src = f"/checkpoints/{ckpt}"
        dst = f"/repo/checkpoints/{ckpt}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {ckpt}")

    # Also copy single model checkpoints (needed for base class init, then replaced)
    for ckpt in ["human_paired_jurkat.ckpt", "human_paired_THP1.ckpt", "human_paired_k562.ckpt"]:
        src = f"/checkpoints/{ckpt}"
        dst = f"/repo/checkpoints/{ckpt}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    print(f"\n{'='*60}")
    print("DUAL ON OPTIMIZATION with HEK293 (PARM) OFF target")
    print(f"{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ON targets: JURKAT, THP1 (EnformerModel ENSEMBLES, 5 models each)")
    print(f"OFF target: HEK293 (PARM pretrained, 5 folds)")
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

    # Import Ctrl-DNA components
    from dna_optimizers_multi.base_optimizer import get_fitness_info
    from dna_optimizers_multi.lagrange_optimizer import Lagrange_optimizer
    from src.reglm.regression import EnformerModel

    # Import PARM components
    from PARM.PARM_utils_load_model import load_PARM
    from PARM.PARM_predict import sequence_to_onehot

    class EnsembleModel:
        """Wrapper that loads multiple EnformerModel checkpoints and averages predictions."""

        def __init__(self, checkpoint_paths: list, device: str = "cuda"):
            self.models = []
            self.device = device
            for path in checkpoint_paths:
                model = EnformerModel.load_from_checkpoint(path, map_location=device)
                model.to(device)
                model.eval()
                self.models.append(model)
            print(f"  Loaded {len(self.models)} ensemble models")

        def __call__(self, sequences):
            """Score sequences by averaging predictions from all models."""
            all_preds = []
            with torch.no_grad():
                for model in self.models:
                    pred = model(sequences)
                    all_preds.append(pred)
            # Average predictions across ensemble
            return torch.stack(all_preds).mean(dim=0)

    def load_ensemble(cell: str, checkpoint_dir: str, device: str = "cuda"):
        """Load ensemble of models for a cell type."""
        # Handle case: JURKAT -> jurkat, THP1 -> THP1
        cell_name = cell if cell == "THP1" else cell.lower()
        paths = sorted(glob_module.glob(f"{checkpoint_dir}/human_paired_{cell_name}_ensemble*.ckpt"))
        if not paths:
            raise FileNotFoundError(f"No ensemble checkpoints found for {cell} in {checkpoint_dir}")
        print(f"Loading {cell} ensemble ({len(paths)} models)...")
        return EnsembleModel(paths, device)

    # Load PARM HEK293 models (ensemble of 5 folds)
    print("Loading PARM HEK293 models...")
    parm_model_dir = "/repo/PARM/pre_trained_models/HEK293"
    parm_models = []
    for fold_file in sorted(os.listdir(parm_model_dir)):
        if fold_file.endswith(".parm"):
            model_path = os.path.join(parm_model_dir, fold_file)
            model = load_PARM(model_path, filter_size=125, train=False, type_loss="poisson")
            model.cuda()
            model.eval()
            parm_models.append(model)
            print(f"  Loaded {fold_file}")
    print(f"  Total PARM folds: {len(parm_models)}")

    def score_hek293_parm(sequences: list[str]) -> np.ndarray:
        """Score sequences using PARM HEK293 ensemble."""
        if not sequences:
            return np.array([])

        # Convert to one-hot
        L_max = max(len(s) for s in sequences)
        onehot = torch.tensor(
            np.float32(sequence_to_onehot(sequences, L_max=L_max))
        ).permute(0, 2, 1).cuda()

        # Get predictions from all folds
        all_preds = []
        with torch.no_grad():
            for model in parm_models:
                pred = model(onehot).cpu().numpy()
                all_preds.append(pred)

        # Average across folds
        return np.mean(all_preds, axis=0).flatten()

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
    cfg.project_name = "Ctrl_DNA_Immune_HEK293"
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
        f"dual_ON_JURKAT_THP1_vs_HEK293_"
        f"lr_{cfg.lr}_"
        f"epoch_{cfg.epoch}_"
        f"seed_{cfg.seed}_"
        f"on_weight_{cfg.on_weight}_"
        f"off_constraint_{cfg.off_constraint}"
    )

    # Get normalization stats for HEK293 (use PARM's typical range)
    # PARM outputs are Log2RPM values, typically in range [-2, 8]
    hek293_min = -2.0
    hek293_max = 8.0

    # Load JURKAT and THP1 ensembles
    print("\nLoading ON target ensembles...")
    jurkat_ensemble = load_ensemble("JURKAT", "/repo/checkpoints", "cuda")
    thp1_ensemble = load_ensemble("THP1", "/repo/checkpoints", "cuda")

    # Define DualOnOptimizer with HEK293
    class DualOnOptimizerHEK293(Lagrange_optimizer):
        """Modified Lagrange optimizer for dual ON targets with HEK293 OFF.

        Uses ensemble models for JURKAT and THP1 (5 models each, predictions averaged).
        """

        def __init__(self, cfg):
            cfg.task = "JURKAT"
            cfg.prefix_label = "100"
            cfg.constraint = [cfg.off_constraint, 0.0, 0.0]
            super().__init__(cfg)
            self.on_weight = cfg.on_weight
            self.off_constraint = cfg.off_constraint
            self.on_indices = [0, 2]  # JURKAT, THP1
            self.off_index = 1  # HEK293

            # Replace single models with ensembles
            print("Replacing single models with ensembles...")
            self.targets['JURKAT'] = jurkat_ensemble
            self.targets['THP1'] = thp1_ensemble
            print("  JURKAT: 5-model ensemble (ρ=0.54)")
            print("  THP1: 5-model ensemble (ρ=0.89)")
            print("Setting up HEK293 (PARM) as OFF target...")

        def normalize_hek293(self, score):
            """Normalize HEK293 score to [0, 1]."""
            return (score - hek293_min) / (hek293_max - hek293_min)

        @torch.no_grad()
        def score_enformer(self, dna):
            """Score a single sequence with all oracles."""
            if len(self.dna_buffer) > self.max_oracle_calls:
                return torch.zeros(3)

            scores_dict = {}

            # Score JURKAT and THP1 with EnformerModel
            for cell in ['JURKAT', 'THP1']:
                model = self.targets[cell]
                raw_score = model([dna]).squeeze(0).item()
                scores_dict[cell] = self.normalize_target(raw_score, cell)

            # Score HEK293 with PARM
            hek293_raw = score_hek293_parm([dna])[0]
            scores_dict['HEK293'] = self.normalize_hek293(hek293_raw)

            # Order: JURKAT, HEK293, THP1
            scores = [scores_dict['JURKAT'], scores_dict['HEK293'], scores_dict['THP1']]

            reward = (
                self.on_weight * scores_dict['JURKAT']
                + self.on_weight * scores_dict['THP1']
                - (scores_dict['HEK293'] - self.off_constraint)
            )

            if dna in self.dna_buffer:
                self.dna_buffer[dna][3] += 1
                self.redundant_count += 1
            else:
                self.dna_buffer[dna] = [torch.tensor(scores), reward, len(self.dna_buffer) + 1, 1]

            return self.dna_buffer[dna][0]

        def compute_combined_score(self, scores_multi, cfg):
            """Compute combined score for batch."""
            jurkat_scores = scores_multi[:, 0]
            hek293_scores = scores_multi[:, 1]
            thp1_scores = scores_multi[:, 2]
            scores = (
                self.on_weight * jurkat_scores
                + self.on_weight * thp1_scores
                - (hek293_scores - self.off_constraint)
            )
            return scores

    print("Initializing optimizer...")
    optimizer = DualOnOptimizerHEK293(cfg)

    # Load starting sequences
    data_dir = cfg.data_dir
    init_data_path = f'{data_dir}/human_promoters/rl_data_large/JURKAT_hard.csv'

    starting_sequences = pd.read_csv(init_data_path)

    # Normalize JURKAT and THP1 scores
    for cell in ['JURKAT', 'THP1']:
        _, min_fitness, max_fitness = get_fitness_info(cell, cfg.oracle_type)
        col_name = f"{cell}_mean"
        starting_sequences[col_name] = (starting_sequences[col_name] - min_fitness) / (max_fitness - min_fitness)

    # Score starting sequences with PARM for HEK293
    print("Scoring starting sequences with PARM HEK293...")
    hek293_scores = score_hek293_parm(starting_sequences['sequence'].tolist())
    starting_sequences['HEK293_mean'] = (hek293_scores - hek293_min) / (hek293_max - hek293_min)

    starting_sequences['target'] = (
        cfg.on_weight * starting_sequences['JURKAT_mean']
        + cfg.on_weight * starting_sequences['THP1_mean']
        - (starting_sequences['HEK293_mean'] - cfg.off_constraint)
    )

    starting_sequences['rewards'] = starting_sequences[
        ['JURKAT_mean', 'HEK293_mean', 'THP1_mean']
    ].values.tolist()

    starting_sequences = starting_sequences.sort_values('target', ascending=False).head(128)

    print(f"\nStarting sequences (top 5):")
    print(starting_sequences[['sequence', 'JURKAT_mean', 'THP1_mean', 'HEK293_mean', 'target']].head())

    print("\n" + "="*60)
    print("STARTING OPTIMIZATION")
    print("="*60 + "\n")

    # Run optimization
    try:
        optimizer.optimize(cfg, starting_sequences)
    except SystemExit:
        pass  # Normal termination from Ctrl-DNA

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/results/dual_on_hek293_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Extract top sequences from buffer
    buffer_items = list(optimizer.dna_buffer.items())
    results_data = []
    for seq, (scores, reward, order, count) in buffer_items:
        results_data.append({
            "sequence": seq,
            "JURKAT": scores[0].item() if torch.is_tensor(scores[0]) else scores[0],
            "HEK293": scores[1].item() if torch.is_tensor(scores[1]) else scores[1],
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
            "off_target": "HEK293_PARM",
            "on_targets": {
                "JURKAT": {"type": "ensemble", "n_models": 5, "rho": 0.54},
                "THP1": {"type": "ensemble", "n_models": 5, "rho": 0.89},
            },
        },
        "results": {
            "total_sequences": len(results_df),
            "top_reward": float(results_df["reward"].max()),
            "top_10_avg_reward": float(results_df.head(10)["reward"].mean()),
            "top_10_avg_jurkat": float(results_df.head(10)["JURKAT"].mean()),
            "top_10_avg_thp1": float(results_df.head(10)["THP1"].mean()),
            "top_10_avg_hek293": float(results_df.head(10)["HEK293"].mean()),
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
    print(f"Top 10 avg HEK293: {summary['results']['top_10_avg_hek293']:.4f}")
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
    """Run dual ON optimization with HEK293 (PARM) as OFF target."""
    print("Launching optimization on Modal...")
    print("Using ENSEMBLE oracles:")
    print("  - JURKAT: 5-model ensemble (ρ=0.54)")
    print("  - THP1: 5-model ensemble (ρ=0.89)")
    print("  - HEK293: PARM pretrained (5 folds)")

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
    print(f"Top 10 avg HEK293: {result['results']['top_10_avg_hek293']:.4f}")
    print("="*60)
    print("\nDownload results with:")
    print("  modal volume get ctrl-dna-results . ./results/")
