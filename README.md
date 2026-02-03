# Tissue-Specific Promoters

Design promoters that are **active in immune cells** (T-cells, macrophages) and **inactive in non-immune cells** (HEK293).

## Results (2026-02-03)

First full optimization completed:
- **25,600 sequences** evaluated over 100 RL iterations
- **Top reward**: 0.4499 (combined objective)
- **Tissue specificity**: ~2.5x higher expression in JURKAT vs HEK293

| Metric | Top 10 Average |
|--------|----------------|
| JURKAT (ON) | 0.50 |
| THP1 (ON) | 0.39 |
| HEK293 (OFF) | 0.21 |

Top candidates saved in `results/dual_on_hek293_20260203_215622/top_100_sequences.csv`.

## Approach

Using [Ctrl-DNA](https://github.com/bowang-lab/Ctrl-DNA) - constrained reinforcement learning for cell-type-specific promoter design.

### Phase 1 (Complete)

| Cell Line | Type | Objective | Source | Status |
|-----------|------|-----------|--------|--------|
| JURKAT | T-cell | ON (maximize) | EnformerModel (ρ=0.50) | ✅ Passed quality gate |
| THP1 | Macrophage | ON (maximize) | EnformerModel ensemble (ρ=0.89) | ✅ Excellent |
| HEK293 | Epithelial | OFF (constrain) | PARM pretrained | ✅ Ready |

**Note**: HEK293 (epithelial) is the proper OFF target for tissue-specific immune promoters. We use the [PARM](https://github.com/vansteensellab/PARM) pretrained model (5-fold ensemble) rather than training our own oracle.

> ⚠️ **Scientific Validity**: See [NOTES.md](NOTES.md#scientific-validity--known-limitations) for important caveats about oracle model validation and biological assumptions.

## Scientific Assumptions & Decisions (Critical)

This project makes explicit biological and modeling assumptions that affect validity. Short version:

- **HEK293 OFF target**: Using PARM pretrained model for HEK293 (epithelial) as the proper OFF target.
- **MPRA context**: Oracles are trained on episomal MPRA data; activity may differ in genomic integration.
- **Sequence length**: 250 bp promoters may miss distal regulatory context.
- **No chromatin context**: Oracles score sequence only; chromatin state is ignored.
- **Oracle quality gate**: Use Spearman ρ ≥ 0.5 (prefer ≥ 0.7) on held-out test data; weaker models can misguide RL.
- **TFBS constraints**: If TFBS files are placeholders (all zeros), do not use `--tfbs True`.
- **Normalization**: Ctrl-DNA normalizes scores using min/max fitness ranges; update `checkpoints/fitness_ranges.json` after retraining oracles.

For full context and mitigations, see [NOTES.md](NOTES.md).

## Workflow (End-to-End)

1. **Prepare data** (downloads MPRA + motifs, builds RL init + TFBS files): `python scripts/prepare_data.py --download_all`
2. **Train oracles** (Modal GPU): `modal run scripts/train_oracles_modal.py` (trains JURKAT + THP1 with v2 improvements)
3. **Quality gate**: Check Spearman ρ in training output (≥ 0.5).
4. **Run optimization** (Modal GPU, uses PARM HEK293 as OFF target):
   ```bash
   modal run scripts/run_dual_on_hek293_modal.py --max-iter 100 --epochs 5
   modal volume get ctrl-dna-results . ./results/
   ```
5. **Analyze top sequences** and **validate experimentally**.

### Future Phases

- Add B-cell oracle (from [SynBP](https://zenodo.org/records/8008545) data)
- Swap HyenaDNA generator for [Evo 2](https://github.com/ArcInstitute/evo2)

## How It Works

```
HyenaDNA (generator) → Candidate sequences → Enformer oracles (per cell type)
        ↑                                              ↓
        └──────── Lagrangian RL optimization ──────────┘
                  (maximize ON, constrain OFF)
```

**Normalization note**: Ctrl-DNA normalizes oracle scores using min/max fitness ranges. After retraining oracles, you can write overrides to `checkpoints/fitness_ranges.json` via:
```bash
python scripts/train_oracles.py --cell all --epochs 10 --write_fitness_ranges
```

## Dependencies

- [Ctrl-DNA](https://github.com/bowang-lab/Ctrl-DNA) (Apache-2.0) - included as submodule

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/alecnielsen/tissue-specific-promoters.git
cd tissue-specific-promoters

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install Ctrl-DNA dependencies
cd Ctrl-DNA && pip install -r requirements.txt
```

## References

- [Ctrl-DNA](https://arxiv.org/abs/2505.20578) - Chen et al., 2025
- [regLM](https://genome.cshlp.org/content/early/2024/09/24/gr.279173.124) - Lal et al., 2024
