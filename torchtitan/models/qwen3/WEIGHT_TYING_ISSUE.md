# Qwen3 Weight Tying Issue

## Summary

The Qwen3 model configuration specifies `enable_weight_tying=True` for smaller models (debugmodel, 0.6B, 1.7B, 4B), but weight tying is **not actually working** in practice. Models train with separate `tok_embeddings.weight` and `output.weight` parameters that have different values.

## Root Cause

Weight tying is attempted in `parallelize_qwen3()` AFTER FSDP wrapping:

```python
# Lines 176-183 in infra/parallelize.py
# NOTE: Weight tying attempted here, but does NOT actually work with FSDP.
# By this point, FSDP has already wrapped both tok_embeddings and output as separate
# parameters in its internal state. The assignment below does not properly tie weights
# in FSDP's view - they remain independent parameters during training.
if model.model_args.enable_weight_tying:
    model.output.weight = model.tok_embeddings.weight
```

By the time this code executes, FSDP has already wrapped both parameters as separate FSDP-managed parameters. A simple pointer assignment does not properly tie weights in FSDP's internal state, so FSDP continues treating them as independent parameters with separate sharding and synchronization.

### Parallelization Order Issue

The parallelization flow in `parallelize_qwen3()`:
1. Apply Tensor Parallelism (TP) to transformer layers
2. Apply MoE Expert Parallelism (EP) if enabled
3. Apply Activation Checkpointing
4. Apply Compilation
5. **Apply FSDP/HSDP wrapping** ← FSDP wraps parameters here
6. **Attempt weight tying** ← Too late, FSDP has already wrapped both parameters

FSDP wraps parameters in its own management layer and maintains internal state for sharding and synchronization. After FSDP wrapping, assigning `model.output.weight = model.tok_embeddings.weight` only updates the module's attribute, but doesn't inform FSDP that these parameters should be treated as the same tensor.

## Evidence

Validation testing clearly demonstrates weight tying is not active:

- **Converting checkpoint assuming weight tying** (using only `tok_embeddings.weight` for both positions) → **gibberish output**
- **Converting checkpoint with both weights separately** (using distinct values) → **correct output**

This proves the two weights have different learned values during training, confirming they are not tied.

## Current Status: Documented Workaround in Place

**This workaround is correct and should be kept as-is.**

### Workaround Components

1. **Checkpoint saving (automatic):**
   - Checkpoints naturally save both `tok_embeddings.weight` and `output.weight` separately
   - They have different values due to untied training

2. **HuggingFace config conversion:**
   - File: `convert_ckpts.sh` (and related scripts)
   - Uses `sed` to set `"tie_word_embeddings": false` in the config
   - Tells HuggingFace not to expect tied weights

3. **State dict adapter:**
   - File: `torchtitan/models/qwen3/model/state_dict_adapter.py` (lines 149-153)
   - Code that was commented out to skip `lm_head.weight` when weight tying is enabled
   - Now always saves both weights to match checkpoint reality

## Impact Analysis

### Memory Overhead

**Qwen3-0.6B weight size:**
- Embedding dimensions: `vocab_size × dim = 151,936 × 1,024`
- Memory per weight: `151,936 × 1,024 × 2 bytes (bf16) = ~311 MB`
- **Total memory overhead: ~311 MB per model replica**

**With FSDP sharding (8 GPUs):**
- Per-GPU overhead: `~311 MB / 8 ≈ 39 MB`
- **Impact: negligible** (<0.5% of typical 40GB GPU memory)

**Larger models (1.7B, 4B):**
- 1.7B: ~770 MB per replica (~96 MB per GPU with 8-way sharding)
- 4B: ~1.2 GB per replica (~150 MB per GPU with 8-way sharding)
- **Still <1% of available GPU memory in typical setups**

### Functional Impact

**None.** The model trains correctly and generates proper output. The only impact is wasted memory, which is negligible in GPU-rich environments.

## Decision: Do Not Fix (For Now)

### Reasoning

1. **Small memory savings:** 311-1200 MB per replica is <1% of GPU memory in typical setups
2. **High implementation risk:** Requires refactoring parallelization order, high chance of breaking existing training
3. **Better time allocation:** Engineering effort better spent on research tasks than memory optimization for well-resourced training
4. **Current workaround works:** The sed hack and explicit dual-weight conversion is proven and stable in production
5. **No functional degradation:** Model quality is unaffected; only memory efficiency is suboptimal

### When to Reconsider

Fix if any of these conditions change:
- Training becomes memory-constrained (e.g., larger models, smaller GPUs)
- Weight tying becomes essential for model convergence
- Timeline allows for high-risk refactoring with thorough testing
- Other models in the suite require proper weight tying

## Future Fix Plan (If Needed)

If weight tying becomes necessary, here are the approach options:

### Option 1: Tie Weights Before FSDP (Recommended)

Refactor `parallelize_qwen3()` to apply weight tying before FSDP wrapping:

```python
def parallelize_qwen3(...):
    # ... existing code for TP and EP ...

    # Enable weight tying BEFORE FSDP wrapping
    if model.model_args.enable_weight_tying:
        model.output.weight = model.tok_embeddings.weight

    # ... activation checkpointing, compilation ...

    # Now apply FSDP with weights already tied
    model = apply_fsdp(model, ...)
```

**Advantages:**
- Weights are tied before FSDP's parameter wrapping
- FSDP will recognize them as the same tensor and manage them accordingly
- Minimal code changes

**Challenges:**
- Tensor Parallelism may shard weights differently before/after tying
- Need to verify FSDP correctly handles the shared parameter
- May require FSDP configuration adjustments
- Requires thorough testing before production use

**Implementation difficulty:** Medium
**Risk level:** Medium-High (requires parallelization refactoring)

### Option 2: Architectural Fix

Modify `Qwen3Model` to not create separate output layer when weight tying is enabled:

```python
# In torchtitan/models/qwen3/model/model.py
def __init__(self, model_args):
    # ...
    self.tok_embeddings = nn.Embedding(...)

    if model_args.enable_weight_tying:
        # Don't create separate output layer
        self.output = None
    else:
        self.output = nn.Linear(dim, vocab_size, ...)
```

Then in forward pass, use embedding weights directly for output projection.

**Advantages:**
- True weight tying at the architecture level
- No FSDP interaction issues
- Works correctly with any parallelism scheme

**Challenges:**
- Requires forward pass modifications
- May affect checkpoint compatibility
- Need to handle the "no self.output" case in multiple places
- Harder to reason about vs. simple assignment

**Implementation difficulty:** Medium-High
**Risk level:** High (affects core model architecture)

### Option 3: Use FSDP's Parameter Tying Mechanisms

Investigate and use FSDP's official parameter tying support (if available):

```python
# After FSDP wrapping, use FSDP's API to tie parameters
# (Requires deep investigation of FSDP internals)
```

**Advantages:**
- Would be the "official" way if FSDP supports it
- Minimal code changes

**Challenges:**
- Requires investigating FSDP source code and internals
- May not be officially supported or documented
- Version-dependent behavior
- Unclear if it works post-wrapping

**Implementation difficulty:** High
**Risk level:** Very High (depends on FSDP internals)

## Recommended Approach (If Fix Needed)

**Option 1 (Tie Before FSDP)** is recommended because:
1. Simplest to understand and verify
2. Lowest architectural impact
3. Clear parallelization order: "tie weights → parallelize"
4. Can be tested incrementally

**Implementation steps (if pursued):**
1. Move weight tying code before FSDP wrapping
2. Run training with weight tying enabled on a small model (debugmodel)
3. Verify checkpoint has identical values for both weights
4. Convert checkpoint and test inference quality
5. Run full training pipeline to ensure stability
6. Compare training convergence with and without weight tying

## Files Involved

- `torchtitan/models/qwen3/__init__.py` - Model configs with `enable_weight_tying=True`
- `torchtitan/models/qwen3/infra/parallelize.py` - Weight tying attempt (lines 176-183)
- `torchtitan/models/qwen3/model/state_dict_adapter.py` - HF conversion with workaround
- `torchtitan/models/qwen3/model/model.py` - Qwen3Model class defining output layer
- `convert_ckpts.sh` - sed workaround for config files
- `torchtitan/models/qwen3/WEIGHT_TYING_ISSUE.md` - This documentation

## Related Documentation

- [PyTorch FSDP Parameter Management](https://pytorch.org/docs/stable/fsdp.html)
- [HuggingFace Weight Tying](https://huggingface.co/docs/transformers/model_sharing)
- [TorchTitan Parallelism Guide](https://github.com/pytorch/torchtitan)

---

**Last updated:** 2026-02-03
**Status:** Documented, no fix planned
**Memory overhead:** ~311 MB (0.6B), ~770 MB (1.7B), ~1.2 GB (4B) per replica
**Functional impact:** None - model trains and generates correctly
**Risk if fixed:** High - affects core parallelization
**Priority:** Low - memory impact negligible, time better spent on research
