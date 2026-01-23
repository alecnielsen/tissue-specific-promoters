# Handoff (2026-01-22)

## Project Goal
Tissue-specific promoter design using Ctrl-DNA with:
- **Dual ON targets**: JURKAT + THP1 (high expression)
- **OFF target**: K562 (low expression)

## Current Status: READY FOR OPTIMIZATION

### What's Done
1. **Environment**: Python 3.11 venv created at `.venv/`
2. **Data prepared**: MPRA data (17,104 sequences) in `data/mpra_cache/`
3. **Oracles trained**: All 3 cell types trained on Modal GPU (10 epochs)
4. **Checkpoints downloaded**: Ready in `checkpoints/`

### Checkpoints Available
```
checkpoints/
├── human_paired_jurkat.ckpt (87MB)
├── human_paired_k562.ckpt (87MB)
└── human_paired_THP1.ckpt (87MB)
```

### Training Results (10 epochs, T4 GPU)
| Cell Type | Val Loss | R² | Spearman ρ | RMSE |
|-----------|----------|-----|------------|------|
| JURKAT | 1.1663 | 0.2134 | 0.4369 | 1.1124 |
| K562 | 1.1230 | 0.2037 | 0.4632 | 1.1067 |
| THP1 | 0.5112 | 0.1570 | 0.3469 | 0.7313 |

Note: Spearman ρ < 0.5 = weak predictive power. Usable for initial experiments, but consider retraining with more epochs (50+) for production.

## Environment Setup (if starting fresh)
```bash
# Activate existing venv
source .venv/bin/activate

# Or recreate if needed:
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[modal]"
git submodule update --init --recursive
```

## Next Steps

### Option A: Run Dual-ON Optimization (primary goal)
```bash
source .venv/bin/activate
python scripts/run_dual_on.py --help  # Check available options
# Then run optimization with desired parameters
```

### Option B: Retrain Oracles with More Epochs (if better performance needed)
```bash
source .venv/bin/activate
python -m modal run scripts/train_oracles_modal.py --epochs 50
# Then re-download checkpoints:
cd checkpoints
python -m modal volume get ctrl-dna-checkpoints human_paired_jurkat.ckpt .
python -m modal volume get ctrl-dna-checkpoints human_paired_k562.ckpt .
python -m modal volume get ctrl-dna-checkpoints human_paired_THP1.ckpt .
```

### Option C: Enable Real TFBS Constraints (optional)
Current TFBS files are placeholders. For real motif constraints:
```bash
pip install pymemesuite
python scripts/prepare_data.py --tfbs_only
```

## Key Files
- `scripts/run_dual_on.py` - Main optimization script
- `scripts/train_oracles_modal.py` - Modal GPU training
- `scripts/prepare_data.py` - Data preparation
- `Ctrl-DNA/` - Git submodule with core Ctrl-DNA code

## Recent Fixes Applied
- Pinned `numpy<2` in Modal image (PyTorch 2.1.2 compatibility)
- Pinned `pytorch-lightning==1.9.5`
- Fixed Google Drive download confirm tokens
- Added NaN guards for correlation metrics
- Checkpoint selection prefers Lightning's best checkpoint

## Fitness Range Overrides
If needed, create `checkpoints/fitness_ranges.json`:
```json
{
  "JURKAT": {"length": 250, "min": -5.57, "max": 8.41},
  "K562": {"length": 250, "min": -4.09, "max": 8.43},
  "THP1": {"length": 250, "min": -7.27, "max": 12.49}
}
```

## TFBS Validation
- Optimization aborts if TFBS CSVs are invalid
- Placeholder marker: `data/human_promoters/tfbs/.PLACEHOLDER_WARNING`
- Remove this file only after generating real TFBS data
