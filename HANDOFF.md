# Handoff (2026-02-02)

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
   - THP1: ρ=0.39 (weak but usable)
4. **PARM HEK293 integrated**: Pretrained model as OFF target (replaces K562)
5. **New optimization script**: `scripts/run_dual_on_hek293_modal.py`
6. **Test run completed**: PARM integration working end-to-end

### Oracle Quality Summary

| Cell Type | Source | Spearman ρ | Status |
|-----------|--------|------------|--------|
| JURKAT | EnformerModel (v2) | **0.5019** | ✅ Passed |
| THP1 | EnformerModel (v2) | 0.3932 | ⚠️ Weak |
| HEK293 | PARM (pretrained) | N/A | ✅ Pretrained |

### Training Improvements (v2)
- Reverse complement augmentation (2x data)
- Early stopping (patience=7)
- LR scheduling (ReduceLROnPlateau)
- Gradient clipping

---

## Next Steps

### Option A: Run Full Optimization Now
The pipeline is ready. JURKAT oracle passed quality gate, PARM HEK293 is strong.

```bash
# Full optimization run (100 iterations)
modal run scripts/run_dual_on_hek293_modal.py --max-iter 100 --epochs 5

# Download results
modal volume get ctrl-dna-results . ./results/
```

### Option B: Improve THP1 Oracle First
THP1 (macrophage) oracle is weak (ρ=0.39). Options:
1. **Scale up architecture** - Increase dim from 384 to 512-768
2. **Multi-task learning** - Train one model with 3 output heads
3. **Ensemble** - Train 3-5 models, average predictions
4. **Additional data** - Macrophage MPRA from ETS2 study

```bash
# Example: Retrain with larger architecture
# Edit scripts/train_oracles_modal.py to change dim=512, depth=5
modal run scripts/train_oracles_modal.py --cell THP1 --epochs 50
```

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
├── human_paired_jurkat.ckpt (87MB) - ρ=0.50 ✅
├── human_paired_THP1.ckpt (87MB)   - ρ=0.39
└── human_paired_k562.ckpt (87MB)   - Not used (replaced by PARM)

PARM/pre_trained_models/HEK293/
├── HEK293_fold0.parm - HEK293_fold4.parm (5 ensemble models)
```

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
- Improve THP1 oracle (architecture scaling or multi-task)
- Add B-cell oracle from SynBP data
- Swap HyenaDNA generator for Evo 2
