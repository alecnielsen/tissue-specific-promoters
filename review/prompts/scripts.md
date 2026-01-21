# Scientific Review: Ctrl-DNA Scripts & Optimizer

You are reviewing the tissue-specific promoter design project using Ctrl-DNA (constrained RL). The goal is to design promoters that are:
- **ON** in JURKAT (T-cells) and THP1 (macrophages)
- **OFF** in K562 (hematopoietic progenitor, off-target proxy)

## Files Under Review

### Scripts (`scripts/`)
1. **train_oracles.py** - Train EnformerModel regressors on MPRA data (local GPU)
2. **train_oracles_modal.py** - Same training logic but runs on Modal GPU
3. **prepare_data.py** - Download MPRA data, JASPAR motifs, generate init files
4. **run_dual_on.py** - Modified Ctrl-DNA optimizer for 2 ON + 1 OFF targets

### Ctrl-DNA Optimizer (`Ctrl-DNA/ctrl_dna/dna_optimizers_multi/`)
5. **base_optimizer.py** - Base optimizer class, model loading, scoring
6. **lagrange_optimizer.py** - Lagrangian constrained optimization, training loop

## Review Focus Areas

### 1. PyTorch/Python Type Mixing (CRITICAL)

Python builtins (`max`, `min`, `abs`) don't work with tensors in boolean contexts.

**Verify:**
- `max(a, b)` where either arg is a tensor → use `torch.maximum()` or `torch.clamp()`
- `min(a, b)` where either arg is a tensor → use `torch.minimum()` or `torch.clamp()`
- `if tensor:` comparisons → use `.item()` or explicit comparison

**Known issue pattern:**
```python
# WRONG - will raise "Boolean value of Tensor is ambiguous"
boost = max(1, 2 + self.tfbs_upper - total_lambda)

# CORRECT
boost = torch.clamp(2 + self.tfbs_upper - total_lambda, min=1.0)
```

### 2. Device Handling

Code must work on CPU, single GPU, and multi-GPU setups.

**Verify:**
- No hardcoded `map_location='cuda:0'` in checkpoint loading
- Use `self.device` or `cfg.device` consistently
- Tensors created mid-function use the correct device

**Known issue pattern:**
```python
# WRONG - breaks on CPU or cuda:1
model = Model.load_from_checkpoint(path, map_location='cuda:0')

# CORRECT
model = Model.load_from_checkpoint(path, map_location=self.device)
```

### 3. Argparse Boolean Flags

`type=bool` in argparse doesn't work as expected - `--flag False` evaluates to `True`.

**Verify:**
- Boolean args use `action='store_true'`/`'store_false'` OR a custom `str2bool` function
- No `type=bool` in any argparse argument

**Known issue pattern:**
```python
# WRONG - --priority False will be True because bool("False") == True
parser.add_argument("--priority", type=bool, default=True)

# CORRECT - use str2bool helper
parser.add_argument("--priority", type=str2bool, default=True)
```

### 4. Parent/Child Class Scoring Consistency

When a child class overrides scoring (e.g., `DualOnOptimizer`), the parent's training loop must use the override.

**Verify:**
- Training loop scoring uses an overridable method, not inline computation
- Child class `score_enformer()` and training loop scoring produce consistent rewards
- The `compute_combined_score()` method exists and is used in the training loop

**Known issue pattern:**
```python
# WRONG - inline formula in parent ignores child's reward structure
scores = scores_multi[:,task_idx] - (scores_multi[:,c1] - self.constraint[0]) + ...

# CORRECT - use overridable method
scores = self.compute_combined_score(scores_multi, cfg)
```

### 5. Wandb Initialization

Multiple `wandb.init()` calls (e.g., in parent and child class) can error without `reinit=True`.

**Verify:**
- All `wandb.init()` calls include `reinit=True`
- Or wandb is only initialized once in the inheritance chain

### 6. Data Pipeline Consistency

**Verify:**
- Processed data filenames are consistent across scripts
- Train/val split logic produces the same splits (same seed, same stratification)
- Column names match what downstream scripts expect
- Fitness statistics (min/max) are computed consistently

### 7. Checkpoint Path and Naming

**Verify:**
- Checkpoint filenames match what `base_optimizer.py` expects
- Case sensitivity is handled correctly (e.g., `THP1` vs `thp1`)
- The checkpoint format is: `human_{oracle_type}_{cell}.ckpt`

### 8. Reward Formula Correctness

**Verify:**
- The reward formula in `score_enformer()` matches documentation
- The `compute_combined_score()` override matches the child class intent
- The `off_constraint` parameter is actually used (not hardcoded)

**Expected dual-ON reward structure:**
```
reward = on_weight * JURKAT + on_weight * THP1 - (K562 - off_constraint)
```

### 9. Path Handling and Working Directory

**Verify:**
- `os.chdir()` doesn't break relative path arguments
- Paths are absolute or resolved correctly before any directory changes

### 10. Return Type Consistency

**Verify:**
- Functions return consistent types (not sometimes int, sometimes tensor)
- Error cases return appropriate values
- Buffer overflow handling is correct

### 11. Dict Ordering Assumptions

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
