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

### Remaining Setup
- [ ] Download/prepare MPRA training data
- [ ] Train oracle models (~6-12 hrs total GPU time)
- [ ] Adapt code for dual ON targets (currently 1 ON + 2 OFF)

---

## Data & Model Requirements

### Oracle Models (Reward Functions)

The RL training requires pre-trained regression models that predict expression from sequence. These are NOT publicly available for promoters - must train from scratch.

**Required checkpoints** (place in `./checkpoints/`):
```
human_paired_jurkat.ckpt   # JURKAT oracle
human_paired_k562.ckpt     # K562 oracle
human_paired_THP1.ckpt     # THP1 oracle
```

**Training estimate**: ~2-4 hours per model on single GPU (10 epochs, ~17K sequences)

### MPRA Training Data

**Source**: Reddy et al. 2024 - "Strategies for effectively modelling promoter-driven gene expression using transfer learning"

| Resource | URL | Contents |
|----------|-----|----------|
| Code + Data | https://github.com/anikethjr/promoter_models | Training scripts, data loaders |
| Paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/ | Methods, data description |
| Pretraining data | https://huggingface.co/datasets/anikethjr/promoter_design | SuRE data |
| Sharpr-MPRA | https://mitra.stanford.edu/kundaje/projects/mpra/data/ | train.hdf5, valid.hdf5, test.hdf5 |

**Data details**: ~17,104 promoter expression measurements across JURKAT, K562, THP1 cells

### TFBS Motif Files

**Required files** (place in `./data/TFBS/`):
```
20250424153556_JASPAR2024_combined_matrices_735317_meme.txt
selected_ppms.csv
```

**Source**: JASPAR 2024 database - need to download or extract from TACO repo

### RL Initialization Data

**Required files** (place in `./data/human_promoters/`):
```
rl_data_large/JURKAT_hard.csv
rl_data_large/K562_hard.csv
rl_data_large/THP1_hard.csv
tfbs/JURKAT_tfbs_freq_all.csv
tfbs/K562_tfbs_freq_all.csv
tfbs/THP1_tfbs_freq_all.csv
```

These are derived from the MPRA data - likely need to generate during oracle training.

---

## To Run Ctrl-DNA

```bash
cd Ctrl-DNA/ctrl_dna

python reinforce_multi_lagrange.py \
    --task JURKAT \
    --oracle_type paired \
    --grpo True \
    --epoch 5 \
    --lambda_lr 3e-4 \
    --lambda_value 0.1 0.9 \
    --tfbs_dir ./data/TFBS \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints
```

### CLI Path Configuration

| Arg | Default | Description |
|-----|---------|-------------|
| `--tfbs_dir` | `./data/TFBS` | JASPAR motif files (meme, ppms) |
| `--data_dir` | `./data` | MPRA training data |
| `--checkpoint_dir` | `./checkpoints` | Oracle model checkpoints |

---

## Next Steps (Priority Order)

### 1. Train Oracle Models
The oracle models are EnformerModel regression models from regLM. Training code is in `Ctrl-DNA/ctrl_dna/src/reglm/regression.py`.

```python
from src.reglm.regression import EnformerModel, SeqDataset

model = EnformerModel(lr=1e-4, loss="mse", pretrained=False)
model.train_on_dataset(train_dataset, val_dataset, max_epochs=10)
```

**TODO**: Write a training script that:
1. Loads Reddy et al. MPRA data
2. Splits into train/val per cell type
3. Trains 3 separate regression models
4. Saves checkpoints in expected format

### 2. Adapt for Dual ON Targets
Current code optimizes: 1 ON target + 2 OFF constraints
Need to modify for: 2 ON targets (JURKAT + THP1) + 1 OFF (K562)

Key file: `Ctrl-DNA/ctrl_dna/reinforce_multi_lagrange.py`
- Modify reward calculation in `main()`
- Update constraint handling in `lagrange_optimizer.py`

### 3. Prepare Data Directory Structure
```
data/
├── TFBS/
│   ├── 20250424153556_JASPAR2024_combined_matrices_735317_meme.txt
│   └── selected_ppms.csv
├── human_promoters/
│   ├── rl_data_large/
│   │   ├── JURKAT_hard.csv
│   │   ├── K562_hard.csv
│   │   └── THP1_hard.csv
│   └── tfbs/
│       ├── JURKAT_tfbs_freq_all.csv
│       ├── K562_tfbs_freq_all.csv
│       └── THP1_tfbs_freq_all.csv
checkpoints/
├── human_paired_jurkat.ckpt
├── human_paired_k562.ckpt
└── human_paired_THP1.ckpt
```

---

## Reference: Related Resources

| Resource | URL | Use Case |
|----------|-----|----------|
| regLM (Genentech) | https://github.com/Genentech/regLM | Oracle architecture, training code |
| regLM Zenodo | https://zenodo.org/records/10669334 | Enhancer models (hepg2/k562/sknsh) - NOT promoters |
| promoter_design | https://github.com/young-geng/promoter_design | Alternative approach, has finetuning data |
| PARM | https://github.com/vansteensellab/PARM | HEK293 oracle (future) |
| Evo 2 | https://github.com/ArcInstitute/evo2 | Better sequence generator (future) |

---

## Code Changes Made

### Files Modified in Ctrl-DNA Fork

1. **reinforce_multi_lagrange.py**
   - Added `--tfbs_dir`, `--data_dir`, `--checkpoint_dir` CLI args
   - Updated `get_meme_and_ppms_path()` to accept tfbs_dir parameter
   - Updated `main()` to use configurable data_dir

2. **dna_optimizers_multi/base_optimizer.py**
   - Updated `load_target_model()` to use configurable checkpoint_dir

3. **dna_optimizers_multi/lagrange_optimizer.py**
   - Updated to use configurable data_dir for TFBS frequency files
