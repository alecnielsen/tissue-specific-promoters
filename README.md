# Tissue-Specific Promoters

Design promoters that are **active in immune cells** (T-cells, macrophages) and **inactive in non-immune cells** (HEK293).

## Approach

Using [Ctrl-DNA](https://github.com/bowang-lab/Ctrl-DNA) - constrained reinforcement learning for cell-type-specific promoter design.

### Phase 1 (Current)

| Cell Line | Type | Objective | Status |
|-----------|------|-----------|--------|
| JURKAT | T-cell | ON (maximize) | ✅ Oracle available |
| THP1 | Macrophage | ON (maximize) | ✅ Oracle available |
| K562 | Myeloid | OFF (constrain) | ✅ Oracle available |

**Note**: K562 is a proxy for HEK293. Both are off-targets, but K562 is hematopoietic while HEK293 is epithelial. Results should be validated in HEK293.

> ⚠️ **Scientific Validity**: See [NOTES.md](NOTES.md#scientific-validity--known-limitations) for important caveats about oracle model validation and biological assumptions.

## Scientific Assumptions & Decisions (Critical)

This project makes explicit biological and modeling assumptions that affect validity. Short version:

- **Off-target proxy**: K562 is used as an OFF target proxy for HEK293 (hematopoietic vs epithelial). Promoters must still be validated in HEK293.
- **MPRA context**: Oracles are trained on episomal MPRA data; activity may differ in genomic integration.
- **Sequence length**: 250 bp promoters may miss distal regulatory context.
- **No chromatin context**: Oracles score sequence only; chromatin state is ignored.
- **Oracle quality gate**: Use Spearman ρ ≥ 0.5 (prefer ≥ 0.7) on held-out test data; weaker models can misguide RL.
- **TFBS constraints**: If TFBS files are placeholders (all zeros), do not use `--tfbs True`.
- **Normalization**: Ctrl-DNA normalizes scores using min/max fitness ranges; update `checkpoints/fitness_ranges.json` after retraining oracles.

For full context and mitigations, see [NOTES.md](NOTES.md).

## Workflow (End-to-End)

1. **Prepare data** (downloads MPRA + motifs, builds RL init + TFBS files): `python scripts/prepare_data.py --download_all`
2. **Train oracles** (Modal GPU or local): `modal run scripts/train_oracles_modal.py` or `python scripts/train_oracles.py --cell all --epochs 10`
3. **Quality gate**: Check Spearman ρ in `checkpoints/oracle_test_metrics.csv` (≥ 0.5).
4. **Write fitness ranges** (recommended after retrain): `python scripts/train_oracles.py --cell all --epochs 10 --write_fitness_ranges`
5. **Run optimization** (Modal GPU recommended):
   ```bash
   modal run scripts/run_dual_on_modal.py --max-iter 100 --epochs 5
   modal volume get ctrl-dna-results . ./results/
   ```
6. **Analyze top sequences** and **validate experimentally** (including HEK293).

### Future Phases

- Add HEK293 oracle (from [PARM](https://github.com/vansteensellab/PARM) data)
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
