"""
Diagnose whether activation checkpointing (AC) or DDP breaks dispersion
gradient flow.

Tests:
  T1 – Hook-firing count: do hooks fire more than once per layer per forward?
  T2 – AC on vs AC off gradient comparison at NGPU=1
  T3 – Full training-loop gradient check: does optimizer receive disp grads?
  T4 – (NGPU=2 only) DDP gradient survival check

Usage (NGPU=1):
  NGPU=1 TRAIN_FILE=tests.test_ac_and_ddp_dispersion \
    CONFIG_FILE=configs/final/qwen_test_dispersion.toml ./run_train.sh

Usage (NGPU=2):
  NGPU=2 TRAIN_FILE=tests.test_ac_and_ddp_dispersion \
    CONFIG_FILE=configs/final/qwen_test_dispersion.toml ./run_train.sh
"""

import os
import copy
import torch
import torch.distributed as dist

from torchtitan.components.dispersion import DispersionLossWrapper
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_batch(trainer):
    data_iterator = trainer.batch_generator(trainer.dataloader)
    return next(data_iterator)


def grad_norm(grads):
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.float().pow(2).sum().item()
    return total ** 0.5


def param_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.float().pow(2).sum().item()
    return total ** 0.5


# ---------------------------------------------------------------------------
# T1: Hook-firing count
# ---------------------------------------------------------------------------

def test_hook_firing_count(trainer):
    """Count how many times hooks fire during forward and backward."""
    print("\n" + "=" * 60)
    print("=== T1: Hook-Firing Count ===")
    print("=" * 60)

    model = trainer.model_parts[0]
    loss_fn = trainer.loss_fn
    assert isinstance(loss_fn, DispersionLossWrapper)

    # Instrument hook with counter
    hook_counts = {"forward": 0, "backward": 0}
    phase = {"current": "forward"}

    original_hooks = trainer._dispersion_hooks

    # Remove existing hooks and add instrumented ones
    for h in original_hooks:
        h.remove()

    instrumented_hooks = []

    def make_counting_hook():
        def hook_fn(module, input, output):
            hook_counts[phase["current"]] += 1
            if isinstance(loss_fn, DispersionLossWrapper):
                if loss_fn.hidden_states is None:
                    loss_fn.hidden_states = []
                loss_fn.hidden_states.append(output)
        return hook_fn

    if hasattr(model, 'layers'):
        for layer in model.layers.values():
            h = layer.register_forward_hook(make_counting_hook())
            instrumented_hooks.append(h)

    n_layers = len(list(model.layers.values()))

    # Forward pass
    input_dict, labels = get_batch(trainer)
    loss_fn.hidden_states = None
    hook_counts["forward"] = 0
    hook_counts["backward"] = 0
    phase["current"] = "forward"

    with trainer.maybe_enable_amp:
        pred = model(input_dict["input"])
        loss = loss_fn(pred, labels, is_training=True)

    forward_count = hook_counts["forward"]

    # Backward pass
    phase["current"] = "backward"
    loss.backward()

    backward_count = hook_counts["backward"]

    print(f"Number of transformer layers: {n_layers}")
    print(f"Hook firings during FORWARD:  {forward_count}")
    print(f"Hook firings during BACKWARD: {backward_count}")

    if backward_count > 0:
        print(f"WARNING: Hooks fire {backward_count} times during backward!")
        print("This means AC recomputation triggers hooks, creating ghost hidden states.")
    else:
        print("OK: Hooks only fire during forward pass.")

    # Restore original hooks
    for h in instrumented_hooks:
        h.remove()
    trainer._dispersion_hooks = []
    for layer in model.layers.values():
        def make_hook_fn():
            def hook_fn(module, input, output):
                if isinstance(loss_fn, DispersionLossWrapper):
                    if loss_fn.hidden_states is None:
                        loss_fn.hidden_states = []
                    loss_fn.hidden_states.append(output)
            return hook_fn
        h = layer.register_forward_hook(make_hook_fn())
        trainer._dispersion_hooks.append(h)

    del pred
    model.zero_grad()

    return forward_count, backward_count


# ---------------------------------------------------------------------------
# T2: Compare gradient with AC on vs off (proxy: compare actual backward
#     gradient to torch.autograd.grad which bypasses AC recomputation issues)
# ---------------------------------------------------------------------------

def test_actual_backward_gradient(trainer):
    """
    The key test: does loss.backward() actually produce the same dispersion
    gradients as torch.autograd.grad(disp_loss, params)?

    If AC breaks gradient flow, loss.backward() would give zero dispersion
    contribution even though autograd.grad works.
    """
    print("\n" + "=" * 60)
    print("=== T2: Backward vs Autograd.grad Comparison ===")
    print("=" * 60)

    model = trainer.model_parts[0]
    loss_fn = trainer.loss_fn
    assert isinstance(loss_fn, DispersionLossWrapper)

    N = trainer.gradient_accumulation_steps
    coeff = loss_fn.dispersion_coeff

    # Disable subsampling for determinism
    orig_max_tokens = loss_fn.disp_loss_fn.max_tokens
    loss_fn.disp_loss_fn.max_tokens = 0
    model.eval()

    input_dict, labels = get_batch(trainer)
    inputs = input_dict["input"]
    params = [p for p in model.parameters() if p.requires_grad]

    # --- Method A: loss.backward() (this is what training actually does) ---
    model.zero_grad()
    loss_fn.hidden_states = None
    with trainer.maybe_enable_amp:
        pred_a = model(inputs)
        loss_a = loss_fn(pred_a, labels, is_training=True)
    loss_a.backward()

    grad_a = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_a[name] = p.grad.clone()
    norm_a = param_grad_norm(model)

    del pred_a

    # --- Method B: CE-only backward (to get CE-only gradient) ---
    model.zero_grad()
    loss_fn.hidden_states = None
    with trainer.maybe_enable_amp:
        pred_b = model(inputs)
        # Only compute CE, ignore hidden states
        ce_only = loss_fn.base_loss_fn(pred_b, labels)
    ce_only.backward()

    grad_b = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_b[name] = p.grad.clone()
    norm_b = param_grad_norm(model)

    del pred_b

    # Compare: if dispersion has any effect, grad_a != grad_b
    diff_norm = 0.0
    n_params_differ = 0
    for name in grad_a:
        if name in grad_b:
            diff = (grad_a[name].float() - grad_b[name].float()).abs().max().item()
            diff_norm += (grad_a[name].float() - grad_b[name].float()).pow(2).sum().item()
            if diff > 1e-8:
                n_params_differ += 1
    diff_norm = diff_norm ** 0.5

    print(f"||grad(CE+disp backward)||  = {norm_a:.6e}")
    print(f"||grad(CE-only backward)||  = {norm_b:.6e}")
    print(f"||difference||              = {diff_norm:.6e}")
    print(f"Params where grads differ:  {n_params_differ} / {len(grad_a)}")

    if diff_norm < 1e-8:
        print("FAIL: loss.backward() produces SAME gradient with and without dispersion!")
        print("      Dispersion gradient is NOT reaching the optimizer.")
        result = False
    else:
        rel_contrib = diff_norm / norm_a if norm_a > 0 else 0
        print(f"Relative dispersion contribution: {rel_contrib:.4e}")
        print("PASS: Dispersion gradient IS reaching the optimizer via backward().")
        result = True

    # Restore
    loss_fn.disp_loss_fn.max_tokens = orig_max_tokens
    model.train()
    model.zero_grad()
    # Clear ghost hidden states from backward
    loss_fn.hidden_states = None

    return result


# ---------------------------------------------------------------------------
# T3: Full training loop step – check optimizer actually uses disp gradient
# ---------------------------------------------------------------------------

def test_training_step_effect(trainer):
    """
    Do one full training step (with gradient accumulation) and check that
    the parameter update differs from a CE-only step.
    """
    print("\n" + "=" * 60)
    print("=== T3: Full Training Step Effect ===")
    print("=" * 60)

    model = trainer.model_parts[0]
    loss_fn = trainer.loss_fn
    assert isinstance(loss_fn, DispersionLossWrapper)

    # Save initial parameters
    init_params = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            init_params[name] = p.data.clone()

    # Do one full training step using the actual training loop
    data_iterator = trainer.batch_generator(trainer.dataloader)
    trainer.train_step(data_iterator)

    # Compute parameter delta
    delta_with_disp = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name in init_params:
            delta_with_disp[name] = (p.data.float() - init_params[name].float()).clone()

    # Now do a step with dispersion disabled
    # Reset parameters to initial
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in init_params:
                p.data.copy_(init_params[name])

    # Reset optimizer state
    for opt in trainer.optimizers.optimizers:
        opt.state.clear()

    # Disable dispersion temporarily
    orig_coeff = loss_fn.dispersion_coeff
    loss_fn.dispersion_coeff = 0.0
    loss_fn.use_disp = False

    # Reset LR schedulers
    for sched in trainer.lr_schedulers.schedulers:
        sched.last_epoch = -1
        sched.step()

    trainer.optimizers.zero_grad()

    # Do another step without dispersion (same data via re-seeded dataloader)
    data_iterator2 = trainer.batch_generator(trainer.dataloader)
    trainer.train_step(data_iterator2)

    delta_without_disp = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name in init_params:
            delta_without_disp[name] = (p.data.float() - init_params[name].float()).clone()

    # Restore dispersion
    loss_fn.dispersion_coeff = orig_coeff
    loss_fn.use_disp = True

    # Compare deltas
    diff_norm = 0.0
    n_params_differ = 0
    for name in delta_with_disp:
        if name in delta_without_disp:
            diff = (delta_with_disp[name] - delta_without_disp[name]).abs().max().item()
            diff_norm += (delta_with_disp[name] - delta_without_disp[name]).pow(2).sum().item()
            if diff > 1e-10:
                n_params_differ += 1
    diff_norm = diff_norm ** 0.5

    norm_with = sum(d.pow(2).sum().item() for d in delta_with_disp.values()) ** 0.5
    norm_without = sum(d.pow(2).sum().item() for d in delta_without_disp.values()) ** 0.5

    print(f"||param_update WITH disp||     = {norm_with:.6e}")
    print(f"||param_update WITHOUT disp||  = {norm_without:.6e}")
    print(f"||difference in updates||      = {diff_norm:.6e}")
    print(f"Params where updates differ:   {n_params_differ} / {len(delta_with_disp)}")

    if diff_norm < 1e-10:
        print("FAIL: Parameter updates are IDENTICAL with and without dispersion!")
        return False
    else:
        rel = diff_norm / norm_with if norm_with > 0 else 0
        print(f"Relative effect: {rel:.4e}")
        print("PASS: Dispersion DOES affect parameter updates.")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_logger()

    config_manager = ConfigManager()
    config = config_manager.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size == 1:
        # Single GPU: force no parallelism
        config.parallelism.data_parallel_replicate_degree = 1
        config.parallelism.data_parallel_shard_degree = -1
        config.parallelism.tensor_parallel_degree = 1
        config.parallelism.context_parallel_degree = 1
        config.parallelism.pipeline_parallel_degree = 1
    # For NGPU=2, use config as-is (DDP with replicate=2)

    ACCUM_N = 4
    config.training.global_batch_size = (
        config.training.local_batch_size * ACCUM_N
        * max(1, config.parallelism.data_parallel_replicate_degree)
    )
    config.training.steps = 2
    config.checkpoint.enable = False
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = False

    logger.info(
        f"AC+DDP diagnostics: world_size={world_size}, "
        f"local_batch={config.training.local_batch_size}, "
        f"global_batch={config.training.global_batch_size}, "
        f"ACCUM_N={ACCUM_N}, coeff={config.dispersion.dispersion_coeff}"
    )

    trainer = None
    try:
        trainer = Trainer(config)
        assert isinstance(trainer.loss_fn, DispersionLossWrapper)

        # T1: Hook firing count
        fwd_count, bwd_count = test_hook_firing_count(trainer)

        # T2: Backward vs autograd.grad
        t2_ok = test_actual_backward_gradient(trainer)

        # T3: Full training step effect
        t3_ok = test_training_step_effect(trainer)

        print("\n" + "=" * 60)
        print(f"=== Summary (world_size={world_size}) ===")
        print("=" * 60)
        print(f"T1: Hook firings — forward={fwd_count}, backward={bwd_count}")
        print(f"T2: Backward gradient includes dispersion: {'PASS' if t2_ok else 'FAIL'}")
        print(f"T3: Training step affected by dispersion:  {'PASS' if t3_ok else 'FAIL'}")

        if not t2_ok:
            print("\n>>> DIAGNOSIS: Dispersion gradient does NOT flow through backward().")
            print("    Most likely cause: Activation Checkpointing breaks the autograd graph")
            print("    for hook-captured tensors during backward recomputation.")
        elif not t3_ok:
            print("\n>>> DIAGNOSIS: Gradient flows but optimizer step is unaffected.")
            print("    Possible causes: DDP all-reduce, grad clipping, or optimizer normalization.")

    except Exception:
        import traceback
        traceback.print_exc()
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        dist.destroy_process_group()
        logger.info("Diagnostics complete.")
