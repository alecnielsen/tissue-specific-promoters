# Handoff (2026-02-02)

## Project Goal
Tissue-specific promoter design using Ctrl-DNA with:
- **Dual ON targets**: JURKAT + THP1 (high expression)
- **OFF target**: K562 (low expression) → **Replace with HEK293**

## Current Status: ORACLES NEED IMPROVEMENT

### What's Done
1. **Environment**: Python 3.11 venv at `.venv/`
2. **Data prepared**: MPRA data (17,104 sequences) in `data/mpra_cache/`
3. **Oracles trained**: All 3 cell types (10 epochs) - **weak predictive power**
4. **Modal optimization pipeline**: Working end-to-end on A10G GPU
5. **Test run completed**: 1 iteration, 1 epoch (proof of concept works)

### Problem: Oracle Quality is Borderline
Current oracles have Spearman ρ ~0.35-0.46, **below the 0.5 quality gate**. Running full optimization with weak oracles risks chasing noise rather than real biological signal.

| Cell Type | Val Loss | R² | Spearman ρ | RMSE |
|-----------|----------|-----|------------|------|
| JURKAT | 1.1663 | 0.2134 | 0.4369 | 1.1124 |
| K562 | 1.1230 | 0.2037 | 0.4632 | 1.1067 |
| THP1 | 0.5112 | 0.1570 | 0.3469 | 0.7313 |

### Additional Problem: K562 is Wrong OFF Target
K562 is hematopoietic (same lineage as immune cells). HEK293 is epithelial - the actual off-target we care about. Need a proper HEK293 oracle.

---

## Priority: Improve Oracles Before Running Full Optimization

### Phase 1: Add HEK293 Oracle (High Priority)

**Why**: K562 is a poor proxy for HEK293. They have different TF repertoires.

**Option A: Use PARM pretrained model (Recommended)**
- PARM (van Steensel lab, 2024) has pretrained models for HEK116 (similar to HEK293)
- Trained on millions of MPRA fragments, likely better than our small dataset
- GitHub: https://github.com/vansteensellab/PARM
- Paper: https://www.biorxiv.org/content/10.1101/2024.07.09.602649v1

```bash
# Install PARM
pip install parm  # or clone repo

# Use PARM to score sequences directly
# PARM supports: AGS, HAP1, HCT116, HEK116, HepG2, K562, LNCaP, MCF7, U2OS
```

**Option B: scMPRA dataset**
- Single-cell MPRA with matched HEK293 + K562 on 676 core promoters
- Small but directly comparable
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9931678/

**Implementation approach**:
1. Integrate PARM as an alternative oracle in the optimization loop
2. Or: retrain EnformerModel on PARM's HEK293 training data
3. Update `scripts/run_dual_on_modal.py` to use HEK293 instead of K562 as OFF target

### Phase 2: Improve T-cell/Macrophage Oracles

**Why**: Current JURKAT (ρ=0.44) and THP1 (ρ=0.35) oracles have weak predictive power.

**Training improvements (low effort, do first)**:
1. **Early stopping** - Current oracles overfit (val loss plateaus while train loss drops)
2. **LR scheduling** - Add ReduceLROnPlateau or cosine annealing
3. **More epochs** - Train 50+ epochs with early stopping
4. **Data augmentation** - Reverse complement (doubles data for free)

**Architecture improvements (medium effort)**:
1. **Scale up Enformer** - Current: dim=384, depth=4. Try: dim=512-768, depth=5-6
2. **Multi-task learning** - One model with 3 output heads (shared representation)
3. **Ensemble** - Train 3-5 models, average predictions (+0.02-0.05 ρ for free)

**Additional data (if above insufficient)**:
- Primary CD4 T-cell MPRA: https://www.embopress.org/doi/abs/10.15252/emmm.202312112
  - Note: Effects differ from Jurkat cell line
- Macrophage MPRA from ETS2 study (inflammatory disease loci)

### Phase 3: Architecture Upgrades (if Phases 1-2 insufficient)

**Model options ranked by effort**:

| Model | Promoter Performance | Effort | Notes |
|-------|---------------------|--------|-------|
| PARM CNN | Already trained for promoter activity | Low | Lightweight, multi-cell-type |
| Nucleotide Transformer 2.5B | +10% over HyenaDNA on promoter tasks | Medium | Fine-tune pretrained |
| Larger Enformer (dim=768) | ~4x compute | Medium | Known architecture |
| Evo 2 (7B/40B) | State-of-art, learns TFBS | High | Feb 2025 release, very large |

**Resources**:
- Nucleotide Transformer: https://www.nature.com/articles/s41592-024-02523-z
- Evo 2: https://github.com/ArcInstitute/evo2
- DNA Foundation Model Benchmark: https://www.nature.com/articles/s41467-025-65823-8

---

## Concrete Next Steps

### Step 1: Improve training (do immediately)
```bash
# Edit scripts/train_oracles_modal.py to add:
# - Early stopping (patience=5)
# - LR scheduling (ReduceLROnPlateau)
# - Reverse complement augmentation

# Then retrain with more epochs
modal run scripts/train_oracles_modal.py --epochs 50

# Check metrics - target Spearman ρ ≥ 0.5
```

### Step 2: Integrate PARM for HEK293
```bash
# Clone PARM
git clone https://github.com/vansteensellab/PARM.git

# Explore their API for scoring sequences
# Modify run_dual_on_modal.py to use PARM HEK116 model as OFF target
```

### Step 3: Scale up architecture (if Step 1 insufficient)
```python
# In train_oracles.py, change:
model = EnformerModel(
    dim=512,   # was 384
    depth=5,   # was 4
    ...
)
```

### Step 4: Run full optimization (only after oracles pass quality gate)
```bash
# Only run this after Spearman ρ ≥ 0.5 for all oracles
modal run scripts/run_dual_on_modal.py --max-iter 100 --epochs 5
modal volume get ctrl-dna-results . ./results/
```

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

## Key Files
- `scripts/train_oracles_modal.py` - Modal GPU training (needs improvements)
- `scripts/run_dual_on_modal.py` - Modal GPU optimization
- `scripts/prepare_data.py` - Data preparation
- `Ctrl-DNA/` - Git submodule with core code
- `NOTES.md` - Detailed technical notes and assumptions

## Checkpoints
```
checkpoints/
├── human_paired_jurkat.ckpt (87MB) - ρ=0.44, needs improvement
├── human_paired_k562.ckpt (87MB)   - ρ=0.46, replace with HEK293
└── human_paired_THP1.ckpt (87MB)   - ρ=0.35, needs improvement
```

## Recent Session Notes (2026-02-02)
- Identified oracle quality as blocking issue before full optimization
- K562 is wrong OFF target (hematopoietic, not epithelial like HEK293)
- PARM model is best path to HEK293 oracle
- Training improvements (early stopping, LR scheduling, augmentation) should help
- Full optimization should wait until oracles pass quality gate (ρ ≥ 0.5)

## Future Phases (unchanged)
- Add B-cell oracle from SynBP data (https://zenodo.org/records/8008545)
- Swap HyenaDNA generator for Evo 2
