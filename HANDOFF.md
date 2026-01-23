# Handoff (2026-01-23)

## Context
- Goal: tissue-specific promoter design using Ctrl-DNA with dual ON targets (JURKAT + THP1) and K562 OFF.
- This handoff captures recent fixes for Modal training reliability, checkpoint selection, metric handling, and downloads.

## Key Changes
- Modal training image now installs CUDA-enabled PyTorch and fails fast if CUDA is unavailable.
- Oracle checkpoint selection prefers Lightning's best checkpoint (falls back to latest mtime).
- Metrics handling now guards against NaN Spearman/Pearson and flags as invalid.
- Google Drive downloads now handle confirm tokens (fewer HTML error failures).
- Pinned `pytorch-lightning==1.9.5` in `pyproject.toml` for consistency with Modal.

## How to Use (Quick)
- Prepare data: `python scripts/prepare_data.py --download_all`
- Train oracles on Modal: `modal run scripts/train_oracles_modal.py`
- Write fitness ranges after retrain (optional but recommended):
  `python scripts/train_oracles.py --cell all --epochs 10 --write_fitness_ranges`
- Run dual-ON optimization: `python scripts/run_dual_on.py ...`

## Fitness Range Overrides
- File: `checkpoints/fitness_ranges.json`
- Auto-loaded by Ctrl-DNA if present.
- Format example:
  {"JURKAT": {"length": 250, "min": -5.1, "max": 8.2}, "K562": {...}, "THP1": {...}}
- You can also set `CTRL_DNA_FITNESS_RANGES` to point to a custom JSON path.

## TFBS Validation (Strict)
- Optimization will abort if TFBS CSVs are empty, have non-numeric motif columns,
  contain NaN/inf, or include negative counts.
- Placeholder detection via `data/human_promoters/tfbs/.PLACEHOLDER_WARNING` remains.

## Files Touched
- `scripts/prepare_data.py`
- `scripts/train_oracles.py`
- `scripts/train_oracles_modal.py`
- `pyproject.toml`
- `HANDOFF.md`

## Open Follow-ups (Optional)
- Modal training script could optionally write `checkpoints/fitness_ranges.json`
  after training to keep normalization in sync.

## Latest Status (2026-01-23)
- Modal training command failed locally because `modal` CLI is not installed in this environment.
- Modal image now installs CUDA-enabled PyTorch (cu121) and will error if CUDA is unavailable.
- Checkpoint selection now prefers Lightning best checkpoint path.
- Correlation metrics now treat NaN as invalid (flags warnings).
- Google Drive downloads now handle confirm tokens.

## Action Required
- Install Modal CLI if you plan to train on Modal: `pip install modal`
- **Retrain all 3 oracles** with the fixed pipeline when ready (Modal).
