"""
End-to-end comparison: Run N training steps with dispersion on vs off
and compare CE loss trajectories.

Usage:
  NGPU=2 TRAIN_FILE=tests.test_ce_comparison \
    CONFIG_FILE=configs/final/qwen_test_dispersion.toml ./run_train.sh
"""

import os
import json
import torch
import torch.distributed as dist

from torchtitan.components.dispersion import DispersionLossWrapper
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


N_STEPS = 30


def run_steps(trainer, n_steps, label):
    """Run n_steps of training and return per-step CE losses."""
    ce_losses = []
    disp_losses = []
    grad_norms = []

    data_iterator = trainer.batch_generator(trainer.dataloader)

    for step in range(n_steps):
        trainer.optimizers.zero_grad()

        accumulated_ce = 0.0
        accumulated_disp = 0.0

        for _micro in range(trainer.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            loss = trainer.forward_backward_step(input_dict, labels)

        # Get metrics from loss_fn before they're cleared
        loss_fn = trainer.loss_fn
        if isinstance(loss_fn, DispersionLossWrapper):
            metrics = loss_fn.get_metrics()
            ce_val = metrics.get("loss_metrics/ce_loss", 0.0)
            disp_val = metrics.get("loss_metrics/dispersion_loss", 0.0)
        else:
            ce_val = loss.item() * trainer.gradient_accumulation_steps
            disp_val = 0.0

        # Gradient norm before clipping
        raw_gn = 0.0
        for m in trainer.model_parts:
            for p in m.parameters():
                if p.grad is not None:
                    raw_gn += p.grad.float().pow(2).sum().item()
        raw_gn = raw_gn ** 0.5

        # Clip and step
        from torchtitan.distributed import utils as dist_utils
        gn = dist_utils.clip_grad_norm_(
            [p for m in trainer.model_parts for p in m.parameters()],
            trainer.job_config.training.max_norm,
            foreach=True,
            pp_mesh=None,
            ep_enabled=False,
        )
        trainer.optimizers.step()
        trainer.lr_schedulers.step()

        ce_losses.append(ce_val)
        disp_losses.append(disp_val)
        grad_norms.append(raw_gn)

        if step % 10 == 0 or step == n_steps - 1:
            print(f"  [{label}] step {step:3d}: CE={ce_val:.4f}  disp={disp_val:.6f}  raw_gn={raw_gn:.4f}  clipped_gn={gn:.4f}")

    return ce_losses, disp_losses, grad_norms


if __name__ == "__main__":
    init_logger()

    config_manager = ConfigManager()
    config = config_manager.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size == 1:
        config.parallelism.data_parallel_replicate_degree = 1
        config.parallelism.data_parallel_shard_degree = -1
        config.parallelism.tensor_parallel_degree = 1
        config.parallelism.context_parallel_degree = 1
        config.parallelism.pipeline_parallel_degree = 1

    ACCUM_N = 4
    dp = max(1, config.parallelism.data_parallel_replicate_degree)
    config.training.global_batch_size = config.training.local_batch_size * ACCUM_N * dp
    config.training.steps = N_STEPS * 2 + 10
    config.checkpoint.enable = False
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = False

    logger.info(
        f"CE comparison: world_size={world_size}, "
        f"global_batch={config.training.global_batch_size}, "
        f"ACCUM_N={ACCUM_N}, N_STEPS={N_STEPS}, "
        f"coeff={config.dispersion.dispersion_coeff}"
    )

    trainer = None
    try:
        trainer = Trainer(config)

        loss_fn = trainer.loss_fn
        assert isinstance(loss_fn, DispersionLossWrapper)

        # ---- Run A: WITH dispersion ----
        print(f"\n{'='*60}")
        print(f"=== Run A: WITH dispersion (coeff={loss_fn.dispersion_coeff}) ===")
        print(f"{'='*60}")

        # Save initial state
        init_params = {}
        for name, p in trainer.model_parts[0].named_parameters():
            init_params[name] = p.data.clone()
        init_opt_state = {}
        for opt in trainer.optimizers.optimizers:
            for group in opt.param_groups:
                for p in group['params']:
                    if p in opt.state:
                        init_opt_state[id(p)] = {
                            k: v.clone() if isinstance(v, torch.Tensor) else v
                            for k, v in opt.state[p].items()
                        }

        ce_with, disp_with, gn_with = run_steps(trainer, N_STEPS, "disp_ON")

        # ---- Reset model, optimizer, LR scheduler ----
        with torch.no_grad():
            for name, p in trainer.model_parts[0].named_parameters():
                if name in init_params:
                    p.data.copy_(init_params[name])

        for opt in trainer.optimizers.optimizers:
            opt.state.clear()

        # Reset LR schedulers
        for sched in trainer.lr_schedulers.schedulers:
            sched.last_epoch = -1
            sched.step()

        # ---- Run B: WITHOUT dispersion ----
        loss_fn.dispersion_coeff = 0.0
        loss_fn.use_disp = False

        print(f"\n{'='*60}")
        print(f"=== Run B: WITHOUT dispersion (coeff=0) ===")
        print(f"{'='*60}")

        ce_without, _, gn_without = run_steps(trainer, N_STEPS, "disp_OFF")

        # ---- Compare ----
        print(f"\n{'='*60}")
        print(f"=== CE Loss Comparison ({N_STEPS} steps) ===")
        print(f"{'='*60}")
        print(f"{'Step':>5}  {'CE(disp=ON)':>12}  {'CE(disp=OFF)':>12}  {'Diff':>12}  {'Rel%':>8}")
        print("-" * 60)

        max_diff = 0.0
        total_diff = 0.0
        for i in range(N_STEPS):
            diff = ce_with[i] - ce_without[i]
            rel = abs(diff) / ce_without[i] * 100 if ce_without[i] > 0 else 0
            total_diff += abs(diff)
            max_diff = max(max_diff, abs(diff))
            if i % 5 == 0 or i == N_STEPS - 1:
                print(f"{i:5d}  {ce_with[i]:12.4f}  {ce_without[i]:12.4f}  {diff:+12.4f}  {rel:7.3f}%")

        avg_diff = total_diff / N_STEPS
        print(f"\nAverage |CE diff|:  {avg_diff:.6f}")
        print(f"Max |CE diff|:      {max_diff:.6f}")
        print(f"Final CE with:      {ce_with[-1]:.4f}")
        print(f"Final CE without:   {ce_without[-1]:.4f}")

        # NOTE: Different data might be used if dataloader isn't seeded identically
        # So we should look at overall trajectory shape, not exact step-by-step match
        print(f"\nAvg grad norm WITH disp:    {sum(gn_with)/len(gn_with):.4f}")
        print(f"Avg grad norm WITHOUT disp: {sum(gn_without)/len(gn_without):.4f}")

        if avg_diff < 0.001:
            print("\nCE curves appear IDENTICAL — dispersion has negligible effect on CE.")
        else:
            print(f"\nCE curves DIFFER by avg {avg_diff:.4f} — dispersion IS affecting CE.")

    except Exception:
        import traceback
        traceback.print_exc()
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        dist.destroy_process_group()
        logger.info("Comparison complete.")
