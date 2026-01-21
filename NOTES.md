# Session Notes

## Project Goal

Design tissue-specific promoters that are:
- **ON** in JURKAT (T-cells) and THP1 (macrophages)
- **OFF** in K562 (hematopoietic, proxy for off-target)

## Key Decisions

1. **Approach**: Ctrl-DNA with constrained RL (Apache-2.0, commercial OK)
2. **Phase 1 cell types**: JURKAT (T-cell ON), THP1 (macrophage ON), K562 (OFF proxy)
3. **K562 limitation**: It's hematopoietic, not epithelial like HEK293. Designed promoters may still be active in HEK293 - validate experimentally.

---

## Setup Status

### Completed
- [x] Fork Ctrl-DNA to [alecnielsen/Ctrl-DNA](https://github.com/alecnielsen/Ctrl-DNA)
- [x] Fix hardcoded paths - now configurable via CLI args
- [x] Create oracle training script (`scripts/train_oracles.py`)
- [x] Create dual ON target runner (`scripts/run_dual_on.py`)
- [x] Create data preparation script (`scripts/prepare_data.py`)

### Remaining Setup
- [ ] Run `scripts/prepare_data.py` to download MPRA data and JASPAR motifs
- [ ] Train oracle models on GPU (~2-4 hrs each)
- [ ] Run Ctrl-DNA optimization

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

3. **scripts/run_dual_on.py** - Run Ctrl-DNA with dual ON targets
   - Modified reward: `on_weight * JURKAT + on_weight * THP1 - (K562 - constraint)`
   - Custom `DualOnOptimizer` class

4. **scripts/prepare_data.py** - Prepare all data files
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

1. **Code Review**: Run `./review/ralph_review.sh` to fix known issues
2. **Run Data Prep**: `python scripts/prepare_data.py --download_all`
3. **Train Oracles**: `modal run scripts/train_oracles_modal.py` (or local GPU)
4. **Run Optimization**: `python scripts/run_dual_on.py`
5. **Analyze Results**: Check top sequences for cell-type specificity
6. **Experimental Validation**: Test designed promoters in cell lines
