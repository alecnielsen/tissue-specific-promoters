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
