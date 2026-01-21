# Scientific Review: Ctrl-DNA Scripts

You are reviewing the `scripts/` directory of a tissue-specific promoter design project using Ctrl-DNA (constrained RL). The goal is to design promoters that are:
- **ON** in JURKAT (T-cells) and THP1 (macrophages)
- **OFF** in K562 (hematopoietic progenitor, off-target proxy)

## Scripts Under Review

1. **train_oracles.py** - Train EnformerModel regressors on MPRA data (local GPU)
2. **train_oracles_modal.py** - Same training logic but runs on Modal GPU
3. **prepare_data.py** - Download MPRA data, JASPAR motifs, generate init files
4. **run_dual_on.py** - Modified Ctrl-DNA optimizer for 2 ON + 1 OFF targets

## Review Focus Areas

### 1. Data Pipeline Consistency

The three data scripts must produce compatible outputs:

**Verify:**
- Processed data filenames are consistent across scripts
- Train/val split logic produces the same splits (same seed, same stratification)
- Column names match what downstream scripts expect
- Fitness statistics (min/max) are computed consistently

**Known issue pattern:**
- Check if `processed.csv` vs `processed_expression.csv` naming is consistent
- Check if stratification uses both GC content AND class, or just GC content

### 2. Checkpoint Path and Naming

Oracle checkpoints must be named correctly for Ctrl-DNA to load them.

**Verify:**
- Checkpoint filenames match what `base_optimizer.py` expects
- Case sensitivity is handled correctly (e.g., `THP1` vs `thp1`)
- The checkpoint format is: `human_{oracle_type}_{cell}.ckpt`

**Expected format:**
```
checkpoints/
├── human_paired_jurkat.ckpt
├── human_paired_k562.ckpt
└── human_paired_THP1.ckpt   # Note: case may matter
```

### 3. Reward Formula Correctness (run_dual_on.py)

This is the core scientific logic. The reward must correctly implement:
- Maximize expression in JURKAT and THP1
- Minimize expression in K562

**Verify:**
- The reward formula in `score_enformer()` matches documentation
- The `off_constraint` parameter is actually used (not hardcoded)
- The starting sequence ranking uses the same formula as the optimizer
- Lambda/constraint handling follows Lagrangian optimization principles

**Expected reward structure:**
```
reward = on_weight * JURKAT + on_weight * THP1 - penalty * (K562 - constraint)
```

### 4. Train/Val Split Consistency

The local and Modal training scripts should produce the same data splits for reproducibility.

**Verify:**
- Both scripts use the same random seed (97)
- Both scripts use the same stratification strategy (GC content bins, class labels)
- Both scripts use the same train/val/test proportions

### 5. Path Handling and Working Directory

**Verify:**
- `os.chdir()` doesn't break relative path arguments
- Paths are absolute or resolved correctly before any directory changes
- Default paths work when running from different directories

### 6. JASPAR Motif File Handling

**Verify:**
- MEME format is correctly parsed
- Selected PPMs list contains valid JASPAR matrix IDs
- Motif file path is used consistently across scripts

### 7. Return Type Consistency

**Verify:**
- Functions return consistent types (not sometimes int, sometimes tensor)
- Error cases return appropriate values
- Buffer overflow handling is correct

### 8. Dict Ordering Assumptions

**Verify:**
- Code doesn't rely on implicit dict ordering for cell type indices
- If indices are used (0, 1, 2), they match the actual dict insertion order
- Prefer explicit key-based access over positional assumptions

## Output Format

**CRITICAL RULE**: If you use the Edit tool or Write tool AT ALL during this review, you MUST NOT output "NO_ISSUES". You must output "FIXES_MADE" instead.

If you find **code issues**:
1. **FIX THEM** by editing the code files directly
2. Then report what you fixed:
   - File path and line number
   - Description of what was wrong
   - What you changed and why
3. End your response with exactly: FIXES_MADE

If you reviewed the code and made **ZERO file edits**, respond with exactly:
NO_ISSUES

**Important notes:**
- "NO_ISSUES" means you used ZERO Edit/Write tools this iteration
- If you edited ANY file for ANY reason, you MUST say "FIXES_MADE"
- The next iteration will verify your fixes with fresh context
