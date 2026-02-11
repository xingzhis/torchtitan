# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gradient diagnostics for dispersion loss.

Verifies:
  1. Gradient flow from dispersion loss to model parameters
  2. Gradient norm comparison: ||grad_disp|| vs ||grad_CE||
  3. Accumulation scaling correctness (empirical)

Usage:
  NGPU=1 TRAIN_FILE=tests.test_gradient_diagnostics \
    CONFIG_FILE=configs/final/qwen_test_dispersion.toml ./run_train.sh
"""

import os

import torch

from torchtitan.components.dispersion import DispersionLossWrapper
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_batch(trainer):
    """Get a single batch from the trainer's dataloader."""
    data_iterator = trainer.batch_generator(trainer.dataloader)
    return next(data_iterator)


def grad_norm(grads):
    """Total L2 norm of a list of (possibly None) gradient tensors."""
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.float().pow(2).sum().item()
    return total ** 0.5


# ---------------------------------------------------------------------------
# Test 1: Gradient Flow
# ---------------------------------------------------------------------------

def test_gradient_flow(trainer):
    """Verify that dispersion loss produces non-zero gradients on model params."""
    print("\n" + "=" * 60)
    print("=== Test 1: Gradient Flow ===")
    print("=" * 60)

    model = trainer.model_parts[0]
    loss_fn = trainer.loss_fn
    assert isinstance(loss_fn, DispersionLossWrapper), (
        "Expected DispersionLossWrapper, got " + type(loss_fn).__name__
    )

    input_dict, labels = get_batch(trainer)
    inputs = input_dict["input"]

    params = [p for p in model.parameters() if p.requires_grad]

    # Forward pass — hooks capture hidden_states
    loss_fn.hidden_states = None
    with torch.no_grad():
        pass  # dummy to ensure no stale graph
    with trainer.maybe_enable_amp:
        pred = model(inputs)
        hidden_states = list(loss_fn.hidden_states)

    # Compute dispersion loss only (no CE)
    disp_loss = loss_fn.disperse_hidden_states(hidden_states)

    print(f"disp_loss requires_grad: {disp_loss.requires_grad}")
    print(f"disp_loss value: {disp_loss.item():.6f}")

    # Gradients of dispersion loss w.r.t. model parameters
    disp_grads = torch.autograd.grad(
        disp_loss, params, retain_graph=False, allow_unused=True,
    )

    non_zero = sum(
        1 for g in disp_grads if g is not None and g.abs().max() > 0
    )
    total = len(params)

    print(f"Params with non-zero grad from dispersion: {non_zero} / {total}")
    if non_zero == 0:
        print("FAIL: No gradients flow from dispersion loss!")
    else:
        print("PASS")

    del pred, hidden_states, disp_grads
    return non_zero > 0


# ---------------------------------------------------------------------------
# Test 2: Gradient Norm Comparison
# ---------------------------------------------------------------------------

def test_gradient_norm_comparison(trainer):
    """Compare ||grad(CE/N)|| vs ||grad(coeff*disp/N)|| on the same batch.

    Uses two separate forward passes to avoid retain_graph issues with
    activation checkpointing.
    """
    print("\n" + "=" * 60)
    print("=== Test 2: Gradient Norm Comparison ===")
    print("=" * 60)

    model = trainer.model_parts[0]
    loss_fn = trainer.loss_fn
    assert isinstance(loss_fn, DispersionLossWrapper)

    N = trainer.gradient_accumulation_steps
    coeff = loss_fn.dispersion_coeff

    # Disable subsampling so both forward passes see identical dispersion
    orig_max_tokens = loss_fn.disp_loss_fn.max_tokens
    loss_fn.disp_loss_fn.max_tokens = 0

    # Use eval mode for deterministic forward passes (no dropout)
    model.eval()

    input_dict, labels = get_batch(trainer)
    inputs = input_dict["input"]

    params = [p for p in model.parameters() if p.requires_grad]

    # --- Forward pass 1: CE only ---
    loss_fn.hidden_states = None
    with trainer.maybe_enable_amp:
        pred1 = model(inputs)
        # base_loss_fn is RescaleAccumulatedLoss → already /N
        ce_loss = loss_fn.base_loss_fn(pred1, labels)
    ce_grads = torch.autograd.grad(
        ce_loss, params, allow_unused=True,
    )
    ce_gn = grad_norm(ce_grads)
    raw_ce = ce_loss.item() * N

    del pred1, ce_grads

    # --- Forward pass 2: dispersion only ---
    loss_fn.hidden_states = None
    with trainer.maybe_enable_amp:
        pred2 = model(inputs)
        hidden_states = list(loss_fn.hidden_states)
        disp_loss = loss_fn.disperse_hidden_states(hidden_states)
    scaled_disp = coeff * disp_loss / N
    disp_grads = torch.autograd.grad(
        scaled_disp, params, allow_unused=True,
    )
    disp_gn = grad_norm(disp_grads)

    # Restore
    loss_fn.disp_loss_fn.max_tokens = orig_max_tokens
    model.train()

    print(f"accumulation_steps (N)    = {N}")
    print(f"dispersion_coeff          = {coeff}")
    print(f"CE loss (raw)             = {raw_ce:.6f}")
    print(f"Dispersion loss (raw)     = {disp_loss.item():.6f}")
    print()
    print(f"||grad(CE/N)||            = {ce_gn:.6e}")
    print(f"||grad(coeff*disp/N)||    = {disp_gn:.6e}  (coeff={coeff})")

    if ce_gn > 0:
        ratio = disp_gn / ce_gn
        print(f"Ratio disp/CE             = {ratio:.6e}")
        if ratio > 0:
            suggested = 0.1 / ratio * coeff
            print(f"Suggested coeff for 10%% contribution: {suggested:.6e}")
    else:
        print("WARNING: CE gradient norm is 0!")

    del pred2, hidden_states, disp_grads
    return ce_gn, disp_gn


# ---------------------------------------------------------------------------
# Test 3: Accumulation Scaling Correctness
# ---------------------------------------------------------------------------

def test_accumulation_scaling(trainer):
    """
    Empirically verify that the /accumulation_steps fix is correct.

    Run A: accum=1 — single forward-backward with full loss
    Run B: accum=N — N forward-backwards with loss/N each, same batch

    Both should produce the same accumulated gradient.
    """
    print("\n" + "=" * 60)
    print("=== Test 3: Scaling Correctness ===")
    print("=" * 60)

    model = trainer.model_parts[0]
    loss_fn = trainer.loss_fn
    assert isinstance(loss_fn, DispersionLossWrapper)

    N = trainer.gradient_accumulation_steps
    coeff = loss_fn.dispersion_coeff

    # Raw CE function (unwrap RescaleAccumulatedLoss)
    raw_ce_fn = loss_fn.base_loss_fn.unwrapped_loss_fn

    # Disable subsampling for determinism
    orig_max_tokens = loss_fn.disp_loss_fn.max_tokens
    loss_fn.disp_loss_fn.max_tokens = 0

    # Use eval mode so dropout is disabled → deterministic forward passes
    model.eval()

    input_dict, labels = get_batch(trainer)
    inputs = input_dict["input"]

    # --- Run A: accum=1, single forward-backward ---
    model.zero_grad()
    loss_fn.hidden_states = None
    with trainer.maybe_enable_amp:
        pred_a = model(inputs)
        hidden_a = list(loss_fn.hidden_states)
        ce_a = raw_ce_fn(pred_a, labels)
        disp_a = loss_fn.disperse_hidden_states(hidden_a)
        loss_a = ce_a + coeff * disp_a
    loss_a.backward()

    grad_a = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_a[name] = p.grad.clone()
    norm_a = grad_norm(list(grad_a.values()))

    del pred_a, hidden_a

    # --- Run B: accum=N, N forward-backwards on same batch ---
    model.zero_grad()
    for _ in range(N):
        loss_fn.hidden_states = None
        with trainer.maybe_enable_amp:
            pred_b = model(inputs)
            hidden_b = list(loss_fn.hidden_states)
            ce_b = raw_ce_fn(pred_b, labels) / N
            disp_b = loss_fn.disperse_hidden_states(hidden_b) / N
            loss_b = ce_b + coeff * disp_b
        loss_b.backward()
        del pred_b, hidden_b

    grad_b = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_b[name] = p.grad.clone()
    norm_b = grad_norm(list(grad_b.values()))

    # Restore
    loss_fn.disp_loss_fn.max_tokens = orig_max_tokens
    model.train()

    print(f"accumulation_steps (N)    = {N}")
    print(f"accum=1  total grad norm: {norm_a:.6e}")
    print(f"accum={N}  total grad norm: {norm_b:.6e}")

    if norm_a > 0:
        rel_error = abs(norm_a - norm_b) / norm_a * 100
        print(f"Relative error: {rel_error:.4f}%  (should be <1%)")
        if rel_error < 1.0:
            print("PASS")
        else:
            print("FAIL: Large relative error — scaling may be wrong.")
    else:
        rel_error = float("inf")
        print("WARNING: norm_a is 0, cannot compute relative error.")

    # Per-parameter worst-case error
    max_param_err = 0.0
    worst_param = ""
    for name in grad_a:
        if name in grad_b:
            diff = (grad_a[name].float() - grad_b[name].float()).abs().max().item()
            denom = grad_a[name].float().abs().max().item()
            if denom > 0:
                err = diff / denom
                if err > max_param_err:
                    max_param_err = err
                    worst_param = name
    print(f"Max per-param relative error: {max_param_err:.6e}  ({worst_param})")

    return rel_error < 1.0 if norm_a > 0 else False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_logger()

    config_manager = ConfigManager()
    config = config_manager.parse_args()

    # Force single-GPU, diagnostic-friendly settings
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = -1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1

    # Need N > 1 for Test 3 to be meaningful
    ACCUM_N = 4
    config.training.global_batch_size = config.training.local_batch_size * ACCUM_N
    config.training.steps = 1
    config.checkpoint.enable = False
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = False

    assert int(os.environ.get("WORLD_SIZE", 1)) == 1, (
        "This diagnostic must run with NGPU=1"
    )

    logger.info(
        f"Gradient diagnostics: local_batch_size={config.training.local_batch_size}, "
        f"ACCUM_N={ACCUM_N}, dispersion_coeff={config.dispersion.dispersion_coeff}"
    )

    trainer = None
    try:
        trainer = Trainer(config)

        assert isinstance(trainer.loss_fn, DispersionLossWrapper), (
            "Dispersion must be enabled in config (set [dispersion] variant)"
        )

        ok1 = test_gradient_flow(trainer)
        ce_gn, disp_gn = test_gradient_norm_comparison(trainer)
        ok3 = test_accumulation_scaling(trainer)

        print("\n" + "=" * 60)
        print("=== Summary ===")
        print("=" * 60)
        print(f"Test 1 (gradient flow):       {'PASS' if ok1 else 'FAIL'}")
        print(f"Test 2 (norm comparison):      CE={ce_gn:.4e}  disp={disp_gn:.4e}")
        print(f"Test 3 (accum scaling):        {'PASS' if ok3 else 'FAIL'}")

    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Diagnostics complete.")
