# Handoff (2026-02-03)

## Project Goal
Tissue-specific promoter design using Ctrl-DNA with:
- **Dual ON targets**: JURKAT + THP1 (high expression in T-cells and macrophages)
- **OFF target**: HEK293 (low expression in epithelial cells)

## Current Status: READY FOR FULL OPTIMIZATION

### What's Done
1. **Environment**: Python 3.11 venv at `.venv/`
2. **Data prepared**: MPRA data (17,104 sequences) in `data/mpra_cache/`
3. **Improved oracles trained** (v2 with augmentation, early stopping, LR scheduling):
   - JURKAT: **ρ=0.50** ✅ Passed quality gate
   - THP1: **ρ=0.89** ✅ Passed quality gate (5-model ensemble)
4. **PARM HEK293 integrated**: Pretrained model as OFF target (replaces K562)
5. **New optimization script**: `scripts/run_dual_on_hek293_modal.py`
6. **Test run completed**: PARM integration working end-to-end

### Oracle Quality Summary

| Cell Type | Source | Spearman ρ | Status |
|-----------|--------|------------|--------|
| JURKAT | EnformerModel (v2) | **0.5019** | ✅ Passed |
| THP1 | EnformerModel ensemble (5 models) | **0.8878** | ✅ Excellent |
| HEK293 | PARM (pretrained) | N/A | ✅ Pretrained |

### THP1 Ensemble Details
- 5 models trained with different random seeds (1-5)
- Individual model performance: ρ=0.44 to ρ=0.84
- Ensemble (prediction averaging): **ρ=0.8878, R²=0.83**
- Improvement: +127% over single model baseline (ρ=0.39)

### Training Improvements (v2)
- Reverse complement augmentation (2x data)
- Early stopping (patience=7)
- LR scheduling (ReduceLROnPlateau)
- Gradient clipping

---

## Next Step: Run Full Optimization

Both oracles now pass quality gates. The pipeline is ready.

```bash
# Full optimization run (100 iterations)
modal run scripts/run_dual_on_hek293_modal.py --max-iter 100 --epochs 5

# Download results
modal volume get ctrl-dna-results . ./results/
```

The THP1 ensemble (5 models) is used via `scripts/evaluate_ensemble.py` for inference.

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
├── human_paired_jurkat.ckpt (87MB)         - ρ=0.50 ✅
├── human_paired_THP1.ckpt (87MB)           - ρ=0.39 (single model, superseded)
├── human_paired_THP1_ensemble1.ckpt (87MB) - ρ=0.73 (ensemble member)
├── human_paired_THP1_ensemble2.ckpt (87MB) - ρ=0.44 (ensemble member)
├── human_paired_THP1_ensemble3.ckpt (87MB) - ρ=0.82 (ensemble member)
├── human_paired_THP1_ensemble4.ckpt (87MB) - ρ=0.82 (ensemble member)
├── human_paired_THP1_ensemble5.ckpt (87MB) - ρ=0.84 (ensemble member)
└── human_paired_k562.ckpt (87MB)           - Not used (replaced by PARM)

PARM/pre_trained_models/HEK293/
├── HEK293_fold0.parm - HEK293_fold4.parm (5 ensemble models)
```

**THP1 Ensemble**: Use all 5 `human_paired_THP1_ensemble*.ckpt` files together.
Average predictions achieve ρ=0.89. See `scripts/evaluate_ensemble.py`.

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
