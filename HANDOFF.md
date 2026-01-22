# Handoff (2026-01-22)

## Context
- Goal: tissue-specific promoter design using Ctrl-DNA with dual ON targets (JURKAT + THP1) and K562 OFF.
- This handoff captures recent changes for strict TFBS validation and optional fitness range overrides.

## Key Changes
- Added strict TFBS CSV validation (fast-fail) to prevent silent scientific invalidity.
- Added optional fitness range overrides via JSON for occasional oracle retraining.
- Documented the above in `README.md` and `NOTES.md`.

## How to Use (Quick)
- Prepare data: `python scripts/prepare_data.py --download_all`
- Train oracles (optional): `python scripts/train_oracles.py --cell all --epochs 10`
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
- `Ctrl-DNA/ctrl_dna/dna_optimizers_multi/base_optimizer.py`
- `scripts/run_dual_on.py`
- `scripts/train_oracles.py`
- `README.md`
- `NOTES.md`
- `HANDOFF.md`

## Open Follow-ups (Optional)
- Modal training script could optionally write `checkpoints/fitness_ranges.json`
  after training to keep normalization in sync.

## Latest Status (2026-01-22)
- Modal dry-run succeeded for **JURKAT** (1 epoch) after pinning `pytorch-lightning==1.9.5`.
  - Test metrics (1 epoch): Spearman ρ ≈ 0.4227, R² ≈ 0.1010 (expectedly weak).
  - This was a pipeline sanity check, not a full retrain.
- Root cause of earlier Modal failures: Lightning v2 removed `validation_epoch_end`. Pin fixed this.
- `scripts/prepare_data.py` now writes a GC-stratified `split` column and uses **train+val only**
  for RL init + normalization stats to avoid test leakage.

## Action Required
- **Retrain all 3 oracles** with the fixed pipeline when ready (Modal or local).
