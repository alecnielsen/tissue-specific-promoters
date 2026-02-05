# Session Notes

## Project Goal

Design tissue-specific promoters that are:
- **ON** in JURKAT (T-cells) and THP1 (macrophages)
- **OFF** in HEK293 (epithelial cells)

## Key Decisions

1. **Approach**: Ctrl-DNA with constrained RL (Apache-2.0, commercial OK)
2. **Phase 1 cell types**: JURKAT (T-cell ON), THP1 (macrophage ON), HEK293 (OFF)
3. **HEK293 as OFF target**: Using PARM pretrained model (5-fold ensemble) for HEK293 predictions.

## Critical Assumptions (Explicit)

- **OFF target**: HEK293 (epithelial) via PARM pretrained model - proper tissue contrast with immune cells.
- **Assay context**: MPRA data are episomal; genomic integration can change activity.
- **Sequence window**: 250 bp promoters may miss distal regulatory signals.
- **Chromatin context**: Oracles ignore chromatin state and 3D context.
- **Oracle quality**: Optimization assumes oracles rank sequences reliably (Spearman ρ ≥ 0.5; prefer ≥ 0.7).
- **TFBS constraints**: TFBS-based constraints are only valid if motif scans are real (not placeholders).

## Workflow (Gated)

Use this as the **go/no-go** checklist before moving to wet lab.

1. **Prepare data**: `python scripts/prepare_data.py --download_all`
2. **Train oracles** (Modal): `modal run scripts/train_oracles_modal.py` (trains JURKAT + THP1 with v2 improvements)
3. **Quality gate**: Spearman ρ ≥ 0.5 for ON target oracles (JURKAT passed, THP1 is weak but usable).
4. **Run RL optimization**: `modal run scripts/run_dual_on_hek293_modal.py --max-iter 100 --epochs 5`
5. **Analyze top sequences** (specificity, motifs, controls).
6. **Validate experimentally** in JURKAT, THP1, and HEK293.

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
| ~~**THP1 oracle weak**~~ | ~~Medium~~ | ✅ Resolved. Ensemble achieves ρ=0.89. |
| **MPRA vs genomic context** | Medium | MPRA uses episomal reporters. Promoter activity may differ when integrated. |
| **250 bp sequences** | Medium | May miss distal regulatory elements. Consider longer context in future. |
| **No chromatin context** | Medium | Oracle predicts from sequence alone, ignoring cell-type chromatin state. |

### Performance Ceiling (~0.45-0.47 top reward)

Optimization plateaus around the same top reward regardless of iterations or target configuration. Likely causes:

1. **Oracle quality ceiling**: JURKAT ρ=0.54 means oracle explains only ~29% of variance. Above a threshold, it can't distinguish "good" from "great" sequences.

2. **Starting sequence quality**: Optimization starts from top 128 pre-selected high performers in `JURKAT_hard.csv`. Early rounds often just rediscover these.

3. **Generator distribution limits**: HyenaDNA trained on natural sequences may struggle to generate out-of-distribution sequences.

**More iterations help with diversity, not peak performance**: 100 iterations yields hundreds of diverse good candidates vs ~10 from quick test, but top score only improves marginally (0.45 → 0.47).

**To break ceiling**: Better oracles, different generator (Evo 2), or experimental validation + active learning.

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
- [x] Add training improvements v2 (augmentation, early stopping, LR scheduling) (2026-02-02)
- [x] Retrain JURKAT oracle - now passes quality gate (ρ=0.50)
- [x] Clone and integrate PARM for HEK293 predictions
- [x] Create new optimization script with HEK293 OFF target (`scripts/run_dual_on_hek293_modal.py`)
- [x] Successfully test end-to-end pipeline with PARM HEK293 (2026-02-02)
- [x] Train THP1 ensemble (5 models) achieving ρ=0.89 (2026-02-03)
- [x] Create ensemble evaluation script (`scripts/evaluate_ensemble.py`)
- [x] Update optimization script to use JURKAT and THP1 ensembles (2026-02-04)

### Test Optimization Results (2026-02-02)

Quick validation run with HEK293 (PARM) as OFF target (1 iteration, 1 epoch) on A10G GPU:
- Total sequences: 256
- Top reward: 0.31
- Top 10 avg JURKAT: 0.43
- Top 10 avg THP1: 0.38
- Top 10 avg HEK293: 0.34

The optimization shows expected behavior (ON > OFF). Ready for full runs.

### Oracle Quality Summary (v2 Training + Ensemble)

Training improvements in v2:
- Reverse complement augmentation (2x effective data)
- Early stopping (patience=7)
- LR scheduling (ReduceLROnPlateau)
- Gradient clipping

| Cell Type | Source | Spearman ρ | Status |
|-----------|--------|------------|--------|
| JURKAT | EnformerModel ensemble (5 models) | **0.54** | ✅ Passed quality gate |
| THP1 | EnformerModel ensemble (5 models) | **0.89** | ✅ Excellent |
| HEK293 | PARM (pretrained) | N/A | ✅ Pretrained model |

### THP1 Oracle Improvement (2026-02-03)

Explored two approaches to improve THP1 oracle from baseline ρ=0.39:

**Experiment 1: Deeper Architecture**
- Configuration: dim=384, depth=6 (vs baseline depth=4)
- Result: ρ=0.38 (no improvement)
- Conclusion: Deeper model doesn't help with this dataset size

**Experiment 2: Ensemble of 5 Models** ✅
- Configuration: dim=384, depth=4, trained with seeds 1-5
- Individual model results:
  - Model 1 (seed=1): ρ=0.73
  - Model 2 (seed=2): ρ=0.44
  - Model 3 (seed=3): ρ=0.82
  - Model 4 (seed=4): ρ=0.82
  - Model 5 (seed=5): ρ=0.84
- **Ensemble (average predictions): ρ=0.89, R²=0.83**
- Improvement: +127% over baseline single model

Key insight: Random seed significantly affects model quality. Ensemble averaging provides robust predictions that exceed any individual model.

### JURKAT Oracle Improvement (2026-02-04)

Applied same ensemble approach to JURKAT:

**Ensemble of 5 Models**
- Configuration: dim=384, depth=4, trained with seeds 1-5
- Individual model results:
  - Model 1 (seed=1): ρ=0.49
  - Model 2 (seed=2): ρ=0.45
  - Model 3 (seed=3): ρ=0.51
  - Model 4 (seed=4): ρ=0.50
  - Model 5 (seed=5): ρ=0.48
- **Ensemble (average predictions): ρ=0.54, R²=0.34**
- Improvement: +8.2% over baseline single model (ρ=0.50)

Note: JURKAT improvement is more modest than THP1 (+8% vs +127%) because the baseline single model was already better calibrated. The JURKAT oracle appears to have a harder performance ceiling around ρ~0.54 with current data/architecture.

### Next Steps
- [x] ~~Retrain oracles with fixed Modal script~~
- [x] ~~Run test optimization on Modal GPU~~
- [x] ~~Integrate PARM HEK293 as OFF target~~
- [x] ~~Improve THP1 oracle~~ (ensemble ρ=0.89)
- [x] ~~Improve JURKAT oracle~~ (ensemble ρ=0.54)
- [x] ~~Run full optimization~~ (100 iterations, 5 epochs) ✅ **COMPLETE**
- [x] ~~Update optimization script to use ensembles~~ ✅ **COMPLETE** (2026-02-04)
- [x] ~~Re-run optimization with improved ensemble oracles~~ ✅ **COMPLETE** (2026-02-04)
- [ ] Analyze top sequences for cell-type specificity
- [ ] Experimental validation in cell lines

### Full Optimization Results (Ensemble Oracles) — 2026-02-04

**Run configuration**: 100 iterations, 5 epochs, batch_size=256, lr=0.0001, **ensemble oracles**

| Metric | Single Models | Ensembles | Change |
|--------|---------------|-----------|--------|
| Total sequences | 25,600 | 25,600 | — |
| Top reward | 0.4499 | **0.4723** | +5.0% |
| Top 10 avg JURKAT | 0.50 | **0.5529** | +10.6% |
| Top 10 avg THP1 | 0.39 | **0.4052** | +3.9% |
| Top 10 avg HEK293 | 0.21 | 0.2192 | ~same |

**Output files (ensemble run)**: `results/dual_on_hek293_20260204_192544/`
- `top_100_sequences.csv` — Best candidates for experimental validation
- `all_sequences.csv` — Full dataset (25,600 sequences)
- `summary.json` — Run configuration and metrics

**Observations**:
- Ensemble oracles improved all ON-target metrics
- JURKAT saw largest improvement (+10.6%), likely due to better-calibrated ensemble
- Tissue specificity maintained: JURKAT/HEK293 = 2.52x
- Top sequences are GC-rich with typical promoter motifs (CpG islands, GGG repeats)

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

For dual ON targets (JURKAT + THP1 ON, HEK293 OFF):

#### Option A: Modal GPU (Recommended)

```bash
# Dual ON: JURKAT + THP1 ON, HEK293 OFF (~$5-15, A10G GPU)
modal run scripts/run_dual_on_hek293_modal.py --max-iter 100 --epochs 5

# Single ON: JURKAT ON, HEK293 OFF (ignore THP1)
modal run scripts/run_dual_on_hek293_modal.py --jurkat-only --max-iter 100 --epochs 5

# Quick test (~$1)
modal run scripts/run_dual_on_hek293_modal.py --max-iter 1 --epochs 1

# Download results
modal volume get ctrl-dna-results . ./results/
```

#### Option B: Local GPU (Legacy - uses K562)

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
├── human_paired_jurkat.ckpt              # JURKAT single model (ρ=0.50, superseded by ensemble)
├── human_paired_jurkat_ensemble[1-5].ckpt # JURKAT ensemble (5 models, ρ=0.54 combined)
├── human_paired_THP1.ckpt                # THP1 single model (ρ=0.39, superseded by ensemble)
├── human_paired_THP1_ensemble[1-5].ckpt  # THP1 ensemble (5 models, ρ=0.89 combined)
└── human_paired_k562.ckpt                # K562 oracle (not used - replaced by PARM)

PARM/pre_trained_models/HEK293/
├── HEK293_fold0.parm                 # PARM HEK293 ensemble (5 folds)
├── HEK293_fold1.parm
├── HEK293_fold2.parm
├── HEK293_fold3.parm
└── HEK293_fold4.parm
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
   - v2 training with augmentation, early stopping, LR scheduling
   - Trains JURKAT and THP1 oracles
   - ~$2-5 total for all models

3. **scripts/evaluate_ensemble.py** - Evaluate ensemble models (THP1 or JURKAT)
   - Loads 5 ensemble checkpoints and averages predictions
   - Reports individual and ensemble Spearman ρ metrics
   - Usage: `python scripts/evaluate_ensemble.py --cell JURKAT` or `--cell THP1`

4. **scripts/run_dual_on.py** - Run Ctrl-DNA with dual ON targets (local, legacy)
   - Modified reward: `on_weight * JURKAT + on_weight * THP1 - (K562 - constraint)`
   - Custom `DualOnOptimizer` class

4. **scripts/run_dual_on_modal.py** - Run optimization on Modal GPU (legacy - K562 OFF)
   - Uses K562 as OFF target (hematopoietic, same lineage as ON targets)
   - Runs on A10G GPU (24GB)

5. **scripts/run_dual_on_hek293_modal.py** - **Main optimization script (recommended)**
   - Uses PARM HEK293 (epithelial) as OFF target
   - PARM 5-fold ensemble for robust predictions
   - **JURKAT ensemble** (5 models, ρ=0.54) for ON target
   - **THP1 ensemble** (5 models, ρ=0.89) for ON target
   - `--jurkat-only` flag: Single ON mode (ignore THP1 in reward)
   - Results saved to Modal volume `ctrl-dna-results`
   - ~$5-15 for full run (100 iter, 5 epochs)

6. **scripts/prepare_data.py** - Prepare all data files
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

Modified for JURKAT+THP1 ON, HEK293 OFF:
- Reward = w * JURKAT + w * THP1 - max(0, HEK293 - c)
- Default: w=0.4 (each ON target), c=0.3 (OFF constraint)

The HEK293 penalty only applies when predicted activity exceeds the constraint threshold (0.3). This allows some baseline expression while penalizing high off-target activity.

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
5. ~~**Run Full Optimization**: 100 iterations, 5 epochs~~ ✅ Done (2026-02-03)
6. **Analyze Results**: Check top sequences for cell-type specificity ← **START HERE**
7. **Improve Oracles**: Train JURKAT ensemble (same approach as THP1)
8. **Experimental Validation**: Test in JURKAT, THP1, and HEK293

### Step 5: Run Full Optimization ✅ COMPLETE

First full optimization completed 2026-02-03:
- 100 iterations, 5 epochs on Modal A10G
- 25,600 sequences evaluated
- Top reward: 0.4499
- Results: `results/dual_on_hek293_20260203_215622/`

To run another optimization:
```bash
modal run --detach scripts/run_dual_on_hek293_modal.py --max-iter 100 --epochs 5
modal volume get ctrl-dna-results dual_on_hek293_YYYYMMDD_HHMMSS ./results/
```

### Step 6: Analyze Results

Analyze the top sequences from `results/dual_on_hek293_20260203_215622/top_100_sequences.csv`:
- **Specificity ratio**: JURKAT/HEK293 and THP1/HEK293 ratios
- **Motif analysis**: TFBS enrichment in top sequences
- **Sequence diversity**: Check for convergence/collapse
- **GC content**: Distribution of nucleotide composition

### Step 7: Oracle Improvements ✅ COMPLETE

Both oracles now have trained ensembles:
- JURKAT: ρ=0.54 (5-model ensemble) ✅ Improved from 0.50
- THP1: ρ=0.89 (5-model ensemble) ✅ Excellent

Training scripts:
- `scripts/train_jurkat_ensemble.py` - Trains 5 JURKAT models in parallel
- `scripts/train_oracles_modal.py --ensemble-id N` - Train individual ensemble members

Evaluate ensembles:
```bash
python scripts/evaluate_ensemble.py --cell JURKAT  # ρ=0.54
python scripts/evaluate_ensemble.py --cell THP1    # ρ=0.89
```

### Step 8: Optimization with Ensemble Oracles ✅ COMPLETE (2026-02-04)

Updated `scripts/run_dual_on_hek293_modal.py` to use ensemble oracles:
- Added `EnsembleModel` wrapper class that loads 5 checkpoints and averages predictions
- JURKAT and THP1 ensembles replace single models after base class initialization

**Full run results** (100 iter, 5 epochs):
| Metric | Single Models | Ensembles | Change |
|--------|---------------|-----------|--------|
| Top reward | 0.4499 | **0.4723** | +5.0% |
| Top 10 avg JURKAT | 0.50 | **0.5529** | +10.6% |
| Top 10 avg THP1 | 0.39 | **0.4052** | +3.9% |
| Top 10 avg HEK293 | 0.21 | 0.2192 | ~same |

Results: `results/dual_on_hek293_20260204_192544/`

**Next**: Analyze top sequences for motifs and cell-type specificity.
