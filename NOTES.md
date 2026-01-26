# Session Notes

## Project Goal

Design tissue-specific promoters that are:
- **ON** in JURKAT (T-cells) and THP1 (macrophages)
- **OFF** in K562 (hematopoietic, proxy for off-target)

## Key Decisions

1. **Approach**: Ctrl-DNA with constrained RL (Apache-2.0, commercial OK)
2. **Phase 1 cell types**: JURKAT (T-cell ON), THP1 (macrophage ON), K562 (OFF proxy)
3. **K562 limitation**: It's hematopoietic, not epithelial like HEK293. Designed promoters may still be active in HEK293 - validate experimentally.

## Critical Assumptions (Explicit)

- **Off-target proxy**: K562 is used as the OFF target proxy for HEK293 (hematopoietic vs epithelial).
- **Assay context**: MPRA data are episomal; genomic integration can change activity.
- **Sequence window**: 250 bp promoters may miss distal regulatory signals.
- **Chromatin context**: Oracles ignore chromatin state and 3D context.
- **Oracle quality**: Optimization assumes oracles rank sequences reliably (Spearman ρ ≥ 0.5; prefer ≥ 0.7).
- **TFBS constraints**: TFBS-based constraints are only valid if motif scans are real (not placeholders).

## Workflow (Gated)

Use this as the **go/no-go** checklist before moving to wet lab.

1. **Prepare data**: `python scripts/prepare_data.py --download_all`
2. **Train oracles** (Modal or local).
3. **Quality gate**: Spearman ρ ≥ 0.5 for all oracles in `checkpoints/oracle_test_metrics.csv`.
4. **Write fitness ranges** after retrain: `python scripts/train_oracles.py --write_fitness_ranges`
5. **Run RL optimization** with `scripts/run_dual_on.py`.
6. **Analyze top sequences** (specificity, motifs, controls).
7. **Validate experimentally** in JURKAT, THP1, and **HEK293** (required).

---

## Scientific Validity & Known Limitations

### Critical Bug Fixed (2025-01-21)

**Loss Type Mismatch**: The Modal-trained checkpoints had a critical bug where:
- Training used MSE loss (correct for regression)
- Inference incorrectly applied `exp()` to outputs (Poisson loss default)
- This made all oracle scores scientifically invalid

**Status**: Fixed. Must retrain oracles with updated `train_oracles_modal.py`.

### Minor Bug Fixed (2025-01-21)

**Checkpoint/Eval Mismatch**: Test metrics were computed on last-epoch model, not best checkpoint:
- PyTorch Lightning saves best checkpoint by val_loss
- But evaluation used the in-memory model (last epoch)
- Reported metrics could differ slightly from actual checkpoint performance

**Impact**: Low - metrics were still MSE-based and from same training run, just potentially from different epoch. Retrain to get accurate metrics.

### Oracle Model Quality

Training scripts now evaluate on held-out test set and report:
- **R²** (coefficient of determination)
- **Spearman ρ** (rank correlation - most important for optimization)
- **Pearson r** (linear correlation)
- **MAE/RMSE** (error magnitude)

**Interpretation**:
| Spearman ρ | Interpretation |
|------------|----------------|
| ≥ 0.7 | Good - oracle reliably ranks sequences |
| 0.5–0.7 | Moderate - use with caution |
| < 0.5 | Weak - oracle may misguide optimization |

**Check your metrics**: After training, see `checkpoints/oracle_test_metrics.csv`

### Biological Limitations

| Issue | Severity | Mitigation |
|-------|----------|------------|
| **K562 ≠ HEK293** | High | K562 is hematopoietic, HEK293 is epithelial. Different TF repertoires. Must validate experimentally in HEK293. |
| **MPRA vs genomic context** | Medium | MPRA uses episomal reporters. Promoter activity may differ when integrated. |
| **250 bp sequences** | Medium | May miss distal regulatory elements. Consider longer context in future. |
| **No chromatin context** | Medium | Oracle predicts from sequence alone, ignoring cell-type chromatin state. |

### TFBS Constraint Validation

If `pymemesuite` is not installed, TFBS files are placeholders (all zeros).

**Check**: Look for `data/human_promoters/tfbs/.PLACEHOLDER_WARNING`

If this file exists and you run with `--tfbs True`, optimization will fail with a clear error. Either:
1. Install pymemesuite and regenerate: `pip install pymemesuite && python scripts/prepare_data.py`
2. Run without TFBS: omit `--tfbs` flag (default is False)

**Strict validation** (fast-fail): TFBS CSVs are now validated for non-empty, numeric-only motif columns, finite values (no NaN/inf), and non-negative counts. Any violation halts optimization to avoid silent scientific invalidity.

### Fitness Range Overrides (Normalization)

Ctrl-DNA normalizes oracle scores using min/max fitness ranges. Hardcoded defaults can drift after occasional retraining.

**Override file**: `checkpoints/fitness_ranges.json`
- Format: `{"JURKAT": {"length": 250, "min": -5.1, "max": 8.2}, ...}`
- Written by `scripts/train_oracles.py` with `--write_fitness_ranges`
- Automatically picked up by Ctrl-DNA at runtime if present

### Recommendations Before Wet Lab

1. **Retrain oracles** with fixed Modal script
2. **Check test metrics** - Spearman ρ ≥ 0.5 for all cell types
3. **Run optimization** and analyze top sequences
4. **Include controls** in validation:
   - Known T-cell promoters (e.g., CD4, IL2)
   - Known ubiquitous promoters (e.g., EF1α, CMV)
   - Random sequences (negative control)
5. **Test in HEK293** - K562 OFF does not guarantee HEK293 OFF

---

## Setup Status

### Completed
- [x] Fork Ctrl-DNA to [alecnielsen/Ctrl-DNA](https://github.com/alecnielsen/Ctrl-DNA)
- [x] Fix hardcoded paths - now configurable via CLI args
- [x] Create oracle training script (`scripts/train_oracles.py`)
- [x] Create dual ON target runner (`scripts/run_dual_on.py`)
- [x] Create Modal optimization script (`scripts/run_dual_on_modal.py`)
- [x] Create data preparation script (`scripts/prepare_data.py`)
- [x] Run `scripts/prepare_data.py` - MPRA data and JASPAR motifs downloaded
- [x] Fix critical loss mismatch bug in Modal training (2025-01-21)
- [x] Add test set evaluation with R², Spearman ρ metrics
- [x] Fix checkpoint/eval mismatch - now reloads best checkpoint before test eval (2025-01-21)
- [x] Train oracles on Modal GPU (10 epochs) - Spearman ρ ~0.4-0.5
- [x] Fix HuggingFace cache proxy issue on Modal (local HyenaDNA model)
- [x] Successfully run test optimization on Modal A10G GPU (2026-01-26)

### Test Optimization Results (2026-01-26)

Quick validation run (1 iteration, 1 epoch) on A10G GPU:
- Total sequences: 256
- Top reward: 0.387
- Top 10 avg JURKAT: 0.38
- Top 10 avg THP1: 0.37
- Top 10 avg K562: 0.22

The optimization shows expected behavior (ON > OFF). Ready for full runs.

### ⚠️ Note on Oracle Quality

Current oracles have Spearman ρ ~0.4-0.5 (weak). Validation loss plateaued during training while training loss kept decreasing (overfitting). Options to improve:

1. **More epochs with early stopping**: Current models overfit. Add early stopping or try different learning rates.
2. **Different architecture**: Try larger Enformer models or alternative architectures (e.g., regLM, HyenaDNA fine-tuning).
3. **More data**: The MPRA dataset has ~17K sequences. Additional data sources could help.
4. **Ensemble methods**: Combine multiple oracles to improve robustness.

For initial proof-of-concept experiments, current oracles are usable but results should be interpreted cautiously.

### Next Steps
- [x] ~~Retrain oracles with fixed Modal script~~
- [x] ~~Run test optimization on Modal GPU~~
- [ ] **Run full optimization** (100 iterations, 5 epochs) ← **START HERE**
- [ ] Analyze top sequences for cell-type specificity
- [ ] Consider oracle improvements (if optimization results are poor)
- [ ] Experimental validation in cell lines (including HEK293)

---

## Quick Start

### 1. Prepare Data

```bash
cd /Users/alec/kernel/tissue-specific-promoters

# Download and process all data
python scripts/prepare_data.py --download_all

# Or skip TFBS scanning if pymemesuite not installed
python scripts/prepare_data.py --download_all --skip_tfbs_scan
```

### 2. Train Oracle Models

#### Option A: Modal (recommended, ~$2-5 total)

```bash
# Train all oracles on Modal GPU
modal run scripts/train_oracles_modal.py

# Train single cell type
modal run scripts/train_oracles_modal.py --cell JURKAT

# Quick test (1 epoch, ~$0.50)
modal run scripts/train_oracles_modal.py --cell JURKAT --epochs 1

# Download checkpoints from Modal
modal volume get ctrl-dna-checkpoints human_paired_jurkat.ckpt ./checkpoints/
modal volume get ctrl-dna-checkpoints human_paired_k562.ckpt ./checkpoints/
modal volume get ctrl-dna-checkpoints human_paired_THP1.ckpt ./checkpoints/
```

#### Option B: Local GPU

```bash
# Train all three cell type oracles
python scripts/train_oracles.py --cell all --epochs 10 --device 0

# Or train individually
python scripts/train_oracles.py --cell JURKAT --epochs 10 --device 0

# CPU fallback (very slow, ~20-40 hrs per model)
python scripts/train_oracles.py --cell JURKAT --epochs 10 --cpu
```

### 3. Run Ctrl-DNA Optimization

For dual ON targets (JURKAT + THP1 ON, K562 OFF):

#### Option A: Modal GPU (Recommended)

```bash
# Full optimization (~$5-15, A10G GPU)
modal run scripts/run_dual_on_modal.py --max-iter 100 --epochs 5

# Quick test (~$1)
modal run scripts/run_dual_on_modal.py --max-iter 1 --epochs 1

# Download results
modal volume get ctrl-dna-results . ./results/
```

#### Option B: Local GPU

```bash
cd Ctrl-DNA/ctrl_dna

python ../../scripts/run_dual_on.py \
    --epoch 5 \
    --max_iter 100 \
    --on_weight 0.4 \
    --off_constraint 0.3 \
    --tfbs_dir ../../data/TFBS \
    --data_dir ../../data \
    --checkpoint_dir ../../checkpoints \
    --wandb_log
```

Or use original single-ON target:

```bash
python reinforce_multi_lagrange.py \
    --task JURKAT \
    --oracle_type paired \
    --grpo True \
    --epoch 5 \
    --lambda_lr 3e-4 \
    --lambda_value 0.1 0.9 \
    --tfbs_dir ../../data/TFBS \
    --data_dir ../../data \
    --checkpoint_dir ../../checkpoints
```

---

## Data & Model Requirements

### Data Directory Structure

After running `prepare_data.py`:

```
data/
├── mpra_cache/
│   ├── Raw_Promoter_Counts.csv      # ~17K promoter MPRA measurements
│   ├── final_list_of_all_promoter_sequences_fixed.tsv
│   └── processed_expression.csv      # Processed log2 ratios
├── TFBS/
│   ├── *_JASPAR2024_*_meme.txt      # JASPAR motifs (MEME format)
│   └── selected_ppms.csv             # Selected immune-relevant TFs
└── human_promoters/
    ├── rl_data_large/
    │   ├── JURKAT_hard.csv           # RL init: top JURKAT promoters
    │   ├── K562_hard.csv             # RL init: top K562 promoters
    │   └── THP1_hard.csv             # RL init: top THP1 promoters
    └── tfbs/
        ├── JURKAT_tfbs_freq_all.csv  # TFBS frequencies
        ├── K562_tfbs_freq_all.csv
        └── THP1_tfbs_freq_all.csv

checkpoints/
├── human_paired_jurkat.ckpt          # JURKAT oracle
├── human_paired_k562.ckpt            # K562 oracle
└── human_paired_THP1.ckpt            # THP1 oracle
```

### Oracle Models

EnformerModel regressors trained on MPRA data:
- Architecture: Enformer trunk (dim=384, depth=4) + linear head
- Loss: MSE on log2(P4/P7) expression values
- Training: ~12K sequences, 10 epochs

### Fitness Statistics

From processed MPRA data (for normalization in base_optimizer.py):

| Cell Type | Min Fitness | Max Fitness |
|-----------|-------------|-------------|
| JURKAT    | -5.574782   | 8.413577    |
| K562      | -4.088671   | 8.555965    |
| THP1      | -7.271035   | 12.485513   |

---

## Code Changes Summary

### Scripts Created

1. **scripts/train_oracles.py** - Train EnformerModel oracles (local)
   - Downloads MPRA data from Google Drive
   - Processes into train/val splits
   - Trains per-cell-type regression models

2. **scripts/train_oracles_modal.py** - Train oracles on Modal GPU
   - Same training logic, runs on cloud GPU
   - ~$2-5 total for all 3 models

3. **scripts/run_dual_on.py** - Run Ctrl-DNA with dual ON targets (local)
   - Modified reward: `on_weight * JURKAT + on_weight * THP1 - (K562 - constraint)`
   - Custom `DualOnOptimizer` class

4. **scripts/run_dual_on_modal.py** - Run optimization on Modal GPU (recommended)
   - Same optimization logic as local script
   - Runs on A10G GPU (24GB) - T4 runs out of memory
   - Includes local HyenaDNA model (bypasses broken HF cache proxy)
   - Results saved to Modal volume `ctrl-dna-results`
   - ~$5-15 for full run (100 iter, 5 epochs)

5. **scripts/prepare_data.py** - Prepare all data files
   - Downloads MPRA data
   - Downloads JASPAR 2024 motifs
   - Generates RL init and TFBS frequency files

### Ctrl-DNA Fork Changes

1. **reinforce_multi_lagrange.py**
   - Added `--tfbs_dir`, `--data_dir`, `--checkpoint_dir` CLI args

2. **dna_optimizers_multi/base_optimizer.py**
   - Updated `load_target_model()` for configurable paths

3. **dna_optimizers_multi/lagrange_optimizer.py**
   - Updated to use configurable data_dir

---

## Dual ON Target Reward Structure

Current Ctrl-DNA: 1 ON + 2 OFF constraints
- Reward = ON - (OFF1 - c1) + ON - (OFF2 - c2)

Modified for JURKAT+THP1 ON, K562 OFF:
- Reward = w * JURKAT + w * THP1 - (K562 - c)
- Default: w=0.4 (each ON target), c=0.3 (OFF constraint)

The Lagrangian optimizer adjusts the K562 constraint dynamically.

---

## Dependencies

```bash
# Core
pip install torch pytorch-lightning enformer-pytorch pandas numpy

# For TFBS scanning
pip install pymemesuite

# For logging
pip install wandb
```

---

## Reference Resources

| Resource | URL | Use Case |
|----------|-----|----------|
| Ctrl-DNA (fork) | https://github.com/alecnielsen/Ctrl-DNA | Main codebase |
| promoter_models | https://github.com/anikethjr/promoter_models | MPRA data source |
| JASPAR 2024 | https://jaspar.elixir.no | TF motif database |
| regLM | https://github.com/Genentech/regLM | Oracle architecture |

---

## Code Review (Ralph Wiggum Loop)

Automated iterative code review using the [Ralph Wiggum technique](https://awesomeclaude.ai/ralph-wiggum).

```bash
# Run the review loop (iterates until clean or max iterations)
./review/ralph_review.sh

# Run in macOS sandbox (recommended - blocks ~/.ssh, ~/.aws, etc.)
./review/sandbox_review.sh

# Check current status
./review/ralph_review.sh --status

# Reset and start fresh
./review/ralph_review.sh --reset

# Increase max iterations (default: 5)
MAX_ITERATIONS=10 ./review/ralph_review.sh
```

The loop will:
1. Read all scripts and the review prompt
2. Ask Claude to review and fix issues
3. If issues found/fixed, loop again with fresh context
4. Stop when `NO_ISSUES` or max iterations reached
5. Detect stuck loops (same output twice = human intervention needed)

**IMPORTANT for Claude sessions**: The loop is FULLY AUTOMATIC. Run it ONCE and let it complete on its own. Do NOT manually re-invoke between iterations - the script handles iteration automatically. If it exits with issues_found, just run it again once (it continues from tracking state). Never babysit or manually continue the loop.

Review artifacts:
- `review/tracking.yaml` - Status tracking
- `review/logs/scripts_history.md` - Iteration history
- `review/prompts/scripts.md` - Review criteria

---

## Next Steps (Priority Order)

1. ~~**Code Review**: Run `./review/ralph_review.sh` to fix known issues~~ ✅ Done
2. ~~**Run Data Prep**: `python scripts/prepare_data.py --download_all`~~ ✅ Done
3. ~~**Train Oracles**: `modal run scripts/train_oracles_modal.py`~~ ✅ Done (Spearman ρ ~0.4-0.5)
4. ~~**Test Optimization**: Verified pipeline on Modal A10G~~ ✅ Done (2026-01-26)
5. **Run Full Optimization**: 100 iterations, 5 epochs ← **START HERE**
6. **Analyze Results**: Check top sequences for cell-type specificity
7. **Improve Oracles**: If results are poor, retrain with better architecture
8. **Experimental Validation**: Test in JURKAT, THP1, K562, **and HEK293**

### Step 5: Run Full Optimization

The optimization pipeline is working. Run a full optimization:

```bash
cd /Users/alec/kernel/tissue-specific-promoters

# Full run on Modal (~$5-15, A10G GPU)
modal run scripts/run_dual_on_modal.py --max-iter 100 --epochs 5

# Download results when done
modal volume get ctrl-dna-results . ./results/
```

Results are saved to Modal volume `ctrl-dna-results` with:
- `all_sequences.csv` - All evaluated sequences with scores
- `top_100_sequences.csv` - Top 100 by reward
- `summary.json` - Run configuration and metrics

### Step 6: Analyze Results

After optimization, analyze the top sequences:
- **Specificity ratio**: JURKAT/K562 and THP1/K562 ratios
- **Motif analysis**: TFBS enrichment in top sequences
- **Sequence diversity**: Check for convergence/collapse

### Step 7: Oracle Improvements (If Needed)

Current oracles have weak predictive power (Spearman ρ ~0.4-0.5). If optimization results are poor:

```bash
# Try more epochs with early stopping
modal run scripts/train_oracles_modal.py --epochs 50

# Download improved checkpoints
modal volume get ctrl-dna-checkpoints human_paired_jurkat.ckpt ./checkpoints/
modal volume get ctrl-dna-checkpoints human_paired_k562.ckpt ./checkpoints/
modal volume get ctrl-dna-checkpoints human_paired_THP1.ckpt ./checkpoints/
```

Alternative approaches:
- Larger Enformer models (`dim=768, depth=6`)
- Different architectures (HyenaDNA fine-tuning, regLM)
- Additional training data

**Remember**: K562 OFF ≠ HEK293 OFF. Always validate in HEK293 experimentally.
