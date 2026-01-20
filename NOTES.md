# Session Notes

## Key Decisions

1. **Approach**: Ctrl-DNA with constrained RL (Apache-2.0, commercial OK)
2. **Phase 1 cell types**: JURKAT (T-cell ON), THP1 (macrophage ON), K562 (OFF proxy)
3. **K562 limitation**: It's hematopoietic, not epithelial like HEK293. Designed promoters may still be active in HEK293 - validate experimentally.

## To Run Ctrl-DNA

```bash
cd Ctrl-DNA/ctrl_dna

# Modify for dual ON targets (JURKAT + THP1)
# Default script optimizes single target with 2 OFF constraints
# You'll need to adapt reinforce_multi_lagrange.py for 2 ON + 1 OFF

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

### Path Configuration

Three new CLI args configure data/model locations:

| Arg | Default | Description |
|-----|---------|-------------|
| `--tfbs_dir` | `./data/TFBS` | JASPAR motif files (meme, ppms) |
| `--data_dir` | `./data` | MPRA training data |
| `--checkpoint_dir` | `./checkpoints` | Oracle model checkpoints |

## Gotchas

- ~~Ctrl-DNA has **hardcoded paths** in the code~~ **FIXED** - now configurable via CLI args
- Need to download pre-trained oracle checkpoints (not included in repo)
- Need MPRA data files for initialization

## Future Enhancements

| Enhancement | Effort | Benefit |
|-------------|--------|---------|
| Add HEK293 oracle (PARM) | Medium | True OFF target |
| Add B-cell oracle (SynBP) | Medium | Cover B-cells |
| Swap to Evo 2 | Higher | Better sequence generation |

## Resources

- Ctrl-DNA oracles: Need to contact authors or train from regLM
- B-cell data: https://zenodo.org/records/8008545
- HEK293 model: https://github.com/vansteensellab/PARM
- Evo 2: https://github.com/ArcInstitute/evo2
