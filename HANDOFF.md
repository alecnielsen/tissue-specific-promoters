# Handoff (2026-02-04)

## Project Goal
Tissue-specific promoter design using Ctrl-DNA with:
- **Dual ON targets**: JURKAT + THP1 (high expression in T-cells and macrophages)
- **OFF target**: HEK293 (low expression in epithelial cells)

## Current Status: ENSEMBLE OPTIMIZATION COMPLETE ✅

### Results Summary (Ensemble Oracles)
Full optimization with ensemble oracles (100 iterations, 5 epochs):
- **Total sequences evaluated**: 25,600
- **Top reward**: 0.4723 (+5.0% vs single models)
- **Top 10 avg JURKAT (ON)**: 0.5529 (+10.6% vs single models)
- **Top 10 avg THP1 (ON)**: 0.4052 (+3.9% vs single models)
- **Top 10 avg HEK293 (OFF)**: 0.2192

Results saved to: `results/dual_on_hek293_20260204_192544/`

### Previous Results (Single Model Oracles)
- **Top reward**: 0.4499
- **Top 10 avg JURKAT**: 0.50
- **Top 10 avg THP1**: 0.39
- **Top 10 avg HEK293**: 0.21

Results: `results/dual_on_hek293_20260203_215622/`

### What's Done
1. **Environment**: Python 3.11 venv at `.venv/`
2. **Data prepared**: MPRA data (17,104 sequences) in `data/mpra_cache/`
3. **Improved oracles trained** (v2 with augmentation, early stopping, LR scheduling):
   - JURKAT: **ρ=0.50** ✅ Passed quality gate
   - THP1: **ρ=0.89** ✅ Passed quality gate (5-model ensemble)
4. **PARM HEK293 integrated**: Pretrained model as OFF target (replaces K562)
5. **New optimization script**: `scripts/run_dual_on_hek293_modal.py`
6. **Test run completed**: PARM integration working end-to-end
7. **Full optimization completed**: 100 iterations, 25,600 sequences evaluated

### Oracle Quality Summary

| Cell Type | Source | Spearman ρ | Status |
|-----------|--------|------------|--------|
| JURKAT | EnformerModel ensemble (5 models) | **0.5408** | ✅ Improved |
| THP1 | EnformerModel ensemble (5 models) | **0.8878** | ✅ Excellent |
| HEK293 | PARM (pretrained) | N/A | ✅ Pretrained |

### THP1 Ensemble Details
- 5 models trained with different random seeds (1-5)
- Individual model performance: ρ=0.44 to ρ=0.84
- Ensemble (prediction averaging): **ρ=0.8878, R²=0.83**
- Improvement: +127% over single model baseline (ρ=0.39)

### JURKAT Ensemble Details (2026-02-04)
- 5 models trained with different random seeds (1-5)
- Individual model performance: ρ=0.45 to ρ=0.51
- Ensemble (prediction averaging): **ρ=0.5408, R²=0.34**
- Improvement: +8.2% over single model baseline (ρ=0.50)
- Note: More modest improvement than THP1 due to better-calibrated baseline

### Training Improvements (v2)
- Reverse complement augmentation (2x data)
- Early stopping (patience=7)
- LR scheduling (ReduceLROnPlateau)
- Gradient clipping

---

## Optimization Results (2026-02-03)

### Top Candidates

| Rank | JURKAT (ON) | THP1 (ON) | HEK293 (OFF) | Reward |
|------|-------------|-----------|--------------|--------|
| 1 | 0.512 | 0.386 | 0.209 | 0.450 |
| 2 | 0.541 | 0.422 | 0.236 | 0.450 |
| 3 | 0.504 | 0.385 | 0.211 | 0.445 |
| 4 | 0.495 | 0.379 | 0.206 | 0.444 |
| 5 | 0.507 | 0.390 | 0.216 | 0.443 |

**Observations**:
- Good tissue specificity: ~2.5x higher expression in JURKAT vs HEK293
- THP1 expression is moderate (ensemble oracle may be conservative)
- Top sequences are GC-rich with typical promoter motifs

### Output Files (Ensemble Run - Latest)
- `results/dual_on_hek293_20260204_192544/top_100_sequences.csv` — Best candidates for validation
- `results/dual_on_hek293_20260204_192544/all_sequences.csv` — Full 25,600 sequence dataset
- `results/dual_on_hek293_20260204_192544/summary.json` — Run config and metrics

### Output Files (Single Model Run - Previous)
- `results/dual_on_hek293_20260203_215622/top_100_sequences.csv`
- `results/dual_on_hek293_20260203_215622/all_sequences.csv`
- `results/dual_on_hek293_20260203_215622/summary.json`

---

## Next Steps

1. ~~**Update optimization script**: Use JURKAT and THP1 ensembles instead of single models~~ ✅ Done
2. ~~**Re-run optimization**: With improved ensemble oracles~~ ✅ Done (2026-02-04)
3. **Analyze top sequences**: Motif enrichment, GC content, sequence diversity
4. **Experimental validation**: Test top 10-20 candidates in JURKAT, THP1, HEK293
5. **Add B-cell oracle**: From SynBP data
6. **Swap generator**: HyenaDNA → Evo 2

---

## Key Files

### Scripts
- `scripts/run_dual_on_hek293_modal.py` - **Main optimization script (HEK293 OFF target)**
- `scripts/train_oracles_modal.py` - Oracle training with v2 improvements
- `scripts/run_dual_on_modal.py` - Old script (K562 OFF target)
- `scripts/prepare_data.py` - Data preparation

### Checkpoints
```
checkpoints/
├── human_paired_jurkat.ckpt (87MB)             - ρ=0.50 (single model, superseded)
├── human_paired_jurkat_ensemble1.ckpt (87MB)   - ρ=0.49 (ensemble member)
├── human_paired_jurkat_ensemble2.ckpt (87MB)   - ρ=0.45 (ensemble member)
├── human_paired_jurkat_ensemble3.ckpt (87MB)   - ρ=0.51 (ensemble member)
├── human_paired_jurkat_ensemble4.ckpt (87MB)   - ρ=0.50 (ensemble member)
├── human_paired_jurkat_ensemble5.ckpt (87MB)   - ρ=0.48 (ensemble member)
├── human_paired_THP1.ckpt (87MB)               - ρ=0.39 (single model, superseded)
├── human_paired_THP1_ensemble1.ckpt (87MB)     - ρ=0.73 (ensemble member)
├── human_paired_THP1_ensemble2.ckpt (87MB)     - ρ=0.44 (ensemble member)
├── human_paired_THP1_ensemble3.ckpt (87MB)     - ρ=0.82 (ensemble member)
├── human_paired_THP1_ensemble4.ckpt (87MB)     - ρ=0.82 (ensemble member)
├── human_paired_THP1_ensemble5.ckpt (87MB)     - ρ=0.84 (ensemble member)
└── human_paired_k562.ckpt (87MB)               - Not used (replaced by PARM)

PARM/pre_trained_models/HEK293/
├── HEK293_fold0.parm - HEK293_fold4.parm (5 ensemble models)
```

**JURKAT Ensemble**: Use all 5 `human_paired_jurkat_ensemble*.ckpt` files together.
Average predictions achieve ρ=0.54. See `scripts/evaluate_ensemble.py --cell JURKAT`.

**THP1 Ensemble**: Use all 5 `human_paired_THP1_ensemble*.ckpt` files together.
Average predictions achieve ρ=0.89. See `scripts/evaluate_ensemble.py --cell THP1`.

### Data
- `data/mpra_cache/processed_expression.csv` - Training data
- `Ctrl-DNA/` - Git submodule with core optimization code
- `PARM/` - PARM model for HEK293 predictions

---

## Environment Setup
```bash
# Activate existing venv
source .venv/bin/activate

# Or recreate if needed:
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[modal]"
git submodule update --init --recursive
```

## Session Notes (2026-02-04)

### Completed This Session
1. **Updated optimization script to use ensembles**
   - Modified `scripts/run_dual_on_hek293_modal.py` to load JURKAT and THP1 ensembles
   - Added `EnsembleModel` wrapper class that averages predictions from 5 models
   - Ensembles replace single models after base class initialization

2. **Ran full optimization with ensemble oracles**
   - 100 iterations, 5 epochs on Modal A10G
   - Top reward: 0.4723 (+5.0% vs single models)
   - Top 10 avg JURKAT: 0.5529 (+10.6% improvement)
   - Results: `results/dual_on_hek293_20260204_192544/`

3. **Trained JURKAT ensemble** (5 models with seeds 1-5)
   - Created `scripts/train_jurkat_ensemble.py` for parallel training
   - Individual models: ρ=0.45 to ρ=0.51
   - Ensemble: ρ=0.5408 (+8.2% over baseline)
2. **Updated `scripts/evaluate_ensemble.py`** to support both JURKAT and THP1
3. **Downloaded JURKAT ensemble checkpoints** to local `checkpoints/`

### JURKAT Ensemble Results

| Model | Seed | Epochs | Spearman ρ |
|-------|------|--------|------------|
| Ensemble 1 | 1 | 26 | 0.4943 |
| Ensemble 2 | 2 | 9 | 0.4509 |
| Ensemble 3 | 3 | 33 | 0.5070 |
| Ensemble 4 | 4 | 49 | 0.4992 |
| Ensemble 5 | 5 | 35 | 0.4761 |
| **Combined** | - | - | **0.5408** |

**Observation**: JURKAT ensemble improvement (+8%) is more modest than THP1 (+127%) because the baseline JURKAT model was already better calibrated. The oracle appears to hit a ceiling around ρ~0.54 with current data/architecture.

---

## Session Notes (2026-02-03)

### Completed This Session
1. **THP1 oracle improvement experiments**:
   - Deeper architecture (depth=6): ρ=0.38 (no improvement over baseline)
   - **Ensemble (5 models)**: ρ=0.89 ✅ Major improvement (+127% over baseline)
2. Created `scripts/evaluate_ensemble.py` for ensemble inference
3. Extended `scripts/train_oracles_modal.py` with `--seed` and `--ensemble-id` parameters

### THP1 Improvement Experiments (2026-02-03)

| Experiment | Configuration | Spearman ρ | Result |
|------------|---------------|------------|--------|
| Baseline | dim=384, depth=4, seed=97 | 0.39 | Weak |
| Deeper model | dim=384, depth=6, seed=97 | 0.38 | ❌ No improvement |
| Ensemble (5 models) | dim=384, depth=4, seeds 1-5 | **0.89** | ✅ Excellent |

**Key finding**: Ensemble averaging outperforms any individual model. Different random seeds produce models with varying quality (ρ=0.44 to ρ=0.84), but averaging predictions yields robust ρ=0.89.

## Session Notes (2026-02-02)

### Completed This Session
1. Added training improvements (augmentation, early stopping, LR scheduling)
2. Retrained oracles - JURKAT now passes quality gate (ρ=0.50)
3. Cloned and integrated PARM for HEK293 predictions
4. Created new optimization script using HEK293 as OFF target
5. Tested end-to-end pipeline successfully

### Why HEK293 Instead of K562
- K562 is hematopoietic (same lineage as JURKAT/THP1)
- HEK293 is epithelial (different tissue, proper "off-target")
- PARM has pretrained HEK293 model (no training needed)

### Architecture Decisions
- Using PARM as drop-in oracle (not retraining)
- PARM ensemble of 5 folds for robust predictions
- Normalized PARM outputs to [0,1] range (Log2RPM typically -2 to 8)

## Future Phases
- ~~Improve THP1 oracle~~ ✅ Done (ensemble ρ=0.89)
- Add B-cell oracle from SynBP data
- Swap HyenaDNA generator for Evo 2
