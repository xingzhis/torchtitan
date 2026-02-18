#!/usr/bin/env python
"""Generate Qwen3 training sweep configs with Chinchilla-optimal schedules.

Quick start (local 2x A100-40GB test):
    python tools/generate_sweep_configs.py --models 0.6B --tag test \\
        --gpu-memory-gb 40 --available-gpus 2

Migrating to a large HPC cluster (e.g. 128x H100-80GB):
    python tools/generate_sweep_configs.py --tag full_sweep \\
        --gpu-memory-gb 80 --available-gpus 128

Key flags to adjust per-cluster:
    --gpu-memory-gb     Per-GPU memory in GB (default: 80 for H100).
                        Controls auto-estimated local_batch_size.
                        Set to 40 for A100-40GB, 80 for A100-80GB / H100.
    --available-gpus    Total GPUs you can allocate. Caps the DP degree
                        (accounting for TP) so configs don't request more
                        GPUs than you have. If omitted, uses conservative
                        per-model defaults from MODEL_SIZE_MAX_GPU_DEFAULTS.
    --local-batch-size  Override auto-estimated LBS (e.g. if you know your
                        GPU can handle a specific size from prior runs).
    --global-batch-size Override GBS across all models.
    --models            Comma-separated subset, e.g. "0.6B,1.7B" to skip
                        larger models you can't run yet.
    --tokens-per-param  Chinchilla multiplier (default: 20).
    --dry-run           Print configs without writing files.

What the script auto-computes:
    - local_batch_size: largest power-of-2 that fits in GPU memory
      (analytical estimate from model arch + FSDP sharding + optimizer states
      + activation memory including output logits and loss workspace).
    - dp_degree: largest DP that keeps grad_accum <= target (default 4).
    - tensor_parallel_degree: from MODEL_TP_DEGREE (TP=2 for 14B, TP=4 for 32B).
    - training steps + warmup: Chinchilla-optimal from instantiated param count.
    - checkpoint interval: ~20 saves per run, clamped to [250, 2000].

Example: same model set, different clusters:
    # 4x A100-40GB workstation
    python tools/generate_sweep_configs.py --models 0.6B,1.7B \\
        --available-gpus 4 --gpu-memory-gb 40 --tag workstation

    # 256x H100-80GB HPC partition
    python tools/generate_sweep_configs.py \\
        --available-gpus 256 --gpu-memory-gb 80 --tag hpc_run
"""
import argparse
import copy
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch

from torchtitan.models.qwen3 import Qwen3Model, qwen3_args


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_CONFIG = os.path.join(BASE_DIR, "configs", "final", "qwen_test_baseline.toml")
# DISPERSION_COEFFS = [1.0, 0.5, 0.1, 0.01]
DISPERSION_COEFFS = [0.1]
SUPPORTED_MODELS = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]

# Conservative defaults for dense pretraining; always override if you have measured
# a better setting on your exact cluster.
MODEL_DEFAULTS = {
    "0.6B": {"lr": 3.0e-4, "weight_decay": 0.1, "local_batch_size": 2, "global_batch_size": 128},
    "1.7B": {"lr": 2.5e-4, "weight_decay": 0.1, "local_batch_size": 2, "global_batch_size": 256},
    "4B": {"lr": 2.0e-4, "weight_decay": 0.1, "local_batch_size": 2, "global_batch_size": 1024},
    "8B": {"lr": 1.5e-4, "weight_decay": 0.1, "local_batch_size": 1, "global_batch_size": 1024},
    "14B": {"lr": 1.2e-4, "weight_decay": 0.1, "local_batch_size": 1, "global_batch_size": 1536},
    "32B": {"lr": 1.0e-4, "weight_decay": 0.1, "local_batch_size": 1, "global_batch_size": 2048},
}

# HuggingFace asset paths per model size (tokenizer, config, etc.)
MODEL_HF_ASSETS = {
    "0.6B": "./assets/hf/Qwen3-0.6B",
    "1.7B": "./assets/hf/Qwen3-1.7B",
    "4B": "./assets/hf/Qwen3-4B",
    "8B": "./assets/hf/Qwen3-8B",
    "14B": "./assets/hf/Qwen3-14B",
    "32B": "./assets/hf/Qwen3-32B",
}

# Tensor parallel degree per model size. Larger models need TP to fit in GPU memory.
MODEL_TP_DEGREE = {
    "0.6B": 1,
    "1.7B": 1,
    "4B": 1,
    "8B": 1,
    "14B": 2,
    "32B": 4,
}

# Default communication-aware GPU caps used when --available-gpus is not provided.
# Smaller models cap earlier to avoid over-scaling into communication bottlenecks.
MODEL_SIZE_MAX_GPU_DEFAULTS = {
    "0.6B": 32,
    "1.7B": 64,
    "4B": 128,
    "8B": 192,
    "14B": 512,
    "32B": 512,
}

# Fallback only if instantiated counting fails.
APPROX_PARAM_FALLBACK = {
    "0.6B": 600_000_000,
    "1.7B": 1_700_000_000,
    "4B": 4_000_000_000,
    "8B": 8_000_000_000,
    "14B": 14_000_000_000,
    "32B": 32_000_000_000,
}


@dataclass
class BatchPlan:
    local_batch_size: int
    global_batch_size: int
    dp_degree: int
    grad_accum_steps: int


def estimate_max_local_batch_size(
    model_name: str,
    seq_len: int,
    dp_degree: int,
    tp_degree: int,
    gpu_memory_gb: float,
    safety_margin: float = 0.70,
) -> tuple[int, dict[str, float]]:
    """Estimate the largest local batch size that fits in GPU memory.

    Uses analytical memory model:
      - Parameters (FSDP-sharded, bf16): params / dp / tp * 2 bytes
      - Gradients (same shard): params / dp / tp * 2 bytes
      - Optimizer states (AdamW fp32 copy + momentum + variance): params / dp / tp * 12 bytes
      - Activations per sample (selective AC "op" mode):
          * Per-layer: 2 * seq_len * dim (attention + FFN residuals, bf16)
                       + seq_len * hidden_dim/tp (FFN intermediate, bf16)
          * Output logits: seq_len * vocab_size * 2 bytes (the dominant cost!)
          * Loss / cross-entropy workspace: ~seq_len * vocab_size * 4 bytes (fp32 softmax)

    Returns (max_lbs, breakdown_gb) where max_lbs is the largest power-of-2 batch size
    that fits, and breakdown_gb is a dict of memory components in GB for diagnostics.
    """
    model_args = qwen3_args[model_name]
    nparams = APPROX_PARAM_FALLBACK.get(model_name, 1_000_000_000)
    try:
        nparams = _count_params_from_instantiated_model(model_name, "off")
    except Exception:
        pass

    shard_factor = dp_degree * tp_degree
    bytes_per_bf16 = 2

    # Sharded parameter memory
    param_mem = nparams / shard_factor * bytes_per_bf16
    # Sharded gradient memory
    grad_mem = nparams / shard_factor * bytes_per_bf16
    # Sharded optimizer states: fp32 param copy (4B) + momentum (4B) + variance (4B) = 12B
    opt_mem = nparams / shard_factor * 12

    fixed_mem = param_mem + grad_mem + opt_mem

    dim = model_args.dim
    hidden_dim = model_args.hidden_dim
    n_layers = model_args.n_layers
    vocab_size = model_args.vocab_size

    # Per-layer activations kept under selective AC:
    #   2 * seq_len * dim (attention + FFN input residuals, bf16)
    #   + seq_len * hidden_dim/tp (FFN gate/up intermediate, bf16, sharded by TP)
    act_per_layer = seq_len * (2 * dim + hidden_dim // tp_degree) * bytes_per_bf16
    layer_act_per_sample = n_layers * act_per_layer

    # Output logits: (seq_len, vocab_size) in bf16 — this is often the largest single
    # allocation.  For Qwen3 vocab_size=151936, seq_len=4096: ~1.16 GB per sample.
    logits_per_sample = seq_len * vocab_size * bytes_per_bf16

    # Cross-entropy loss workspace: softmax computed in fp32 over vocab dim.
    # PyTorch keeps the fp32 softmax output for backward: seq_len * vocab_size * 4 bytes.
    loss_workspace_per_sample = seq_len * vocab_size * 4

    act_per_sample = layer_act_per_sample + logits_per_sample + loss_workspace_per_sample

    usable_mem = gpu_memory_gb * (1024 ** 3) * safety_margin
    available_for_act = usable_mem - fixed_mem

    if available_for_act <= 0:
        return 1, {
            "params_gb": param_mem / 1e9,
            "grads_gb": grad_mem / 1e9,
            "optim_gb": opt_mem / 1e9,
            "act_per_sample_gb": act_per_sample / 1e9,
            "logits_per_sample_gb": logits_per_sample / 1e9,
            "usable_gb": usable_mem / 1e9,
        }

    max_samples = int(available_for_act / act_per_sample)
    # Round down to largest power of 2
    if max_samples <= 0:
        lbs = 1
    else:
        lbs = 1 << (max_samples.bit_length() - 1)
    lbs = max(1, lbs)

    breakdown = {
        "params_gb": param_mem / 1e9,
        "grads_gb": grad_mem / 1e9,
        "optim_gb": opt_mem / 1e9,
        "act_per_sample_gb": act_per_sample / 1e9,
        "layer_act_gb": layer_act_per_sample / 1e9,
        "logits_gb": logits_per_sample / 1e9,
        "loss_ws_gb": loss_workspace_per_sample / 1e9,
        "usable_gb": usable_mem / 1e9,
        "max_samples_exact": max_samples,
    }
    return lbs, breakdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3 sweep configs without submitting jobs."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(SUPPORTED_MODELS),
        help="Comma-separated model flavors to generate.",
    )
    parser.add_argument("--dataset", type=str, default="fineweb_edu_dedup_full", help="Dataset name.")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag.")
    parser.add_argument(
        "--description",
        type=str,
        default="Qwen training sweep (config generation only)",
        help="Run description written into generated TOMLs.",
    )
    parser.add_argument(
        "--baseline-template",
        type=str,
        default="",
        help="Path to baseline TOML. If omitted, tries model-specific baseline then default.",
    )
    parser.add_argument(
        "--tokens-per-param",
        type=float,
        default=20.0,
        help="Chinchilla token multiplier.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio before clamping.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=0,
        help="Override global batch size across all selected models.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=0,
        help="Override local batch size across all selected models.",
    )
    parser.add_argument(
        "--dp-degree",
        type=int,
        default=0,
        help="Override data parallel degree. If set, script computes grad accumulation from GBS.",
    )
    parser.add_argument(
        "--available-gpus",
        type=int,
        default=0,
        help="Maximum GPUs available. Used to auto-pick dp_degree if --dp-degree is not set.",
    )
    parser.add_argument(
        "--target-grad-accum",
        type=int,
        default=4,
        help="Prefer grad accumulation at or below this value when auto-selecting dp_degree.",
    )
    parser.add_argument(
        "--max-grad-accum",
        type=int,
        default=8,
        help="Hard upper bound for auto-selected grad accumulation.",
    )
    parser.add_argument(
        "--weight-tying-for-count",
        type=str,
        choices=["auto", "on", "off"],
        default="off",
        help="Whether parameter counting assumes tied embeddings. 'off' matches untied bug behavior.",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=0,
        help="Override tensor parallel degree for all models. "
             "Set to 1 for clusters without NVLink (PCIe only). "
             "If 0 (default), uses per-model defaults from MODEL_TP_DEGREE.",
    )
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=80.0,
        help="Per-GPU memory in GB. Used to auto-estimate max local batch size.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=os.path.join("configs", "sweep_generated"),
        help="Directory under repo root to save generated configs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned configs and hyperparameters without writing files.",
    )
    return parser.parse_args()


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def parse_model_list(raw: str) -> list[str]:
    models = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in models if m not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. Supported: {', '.join(SUPPORTED_MODELS)}"
        )
    return models


def choose_baseline_template(baseline_template: str, model_name: str) -> str:
    if baseline_template:
        return os.path.abspath(baseline_template)
    model_specific = os.path.join(
        BASE_DIR, "configs", "final", f"qwen_test_baseline_{model_name}.toml"
    )
    if os.path.exists(model_specific):
        return model_specific
    return BASELINE_CONFIG


def read_template_defaults(config_path: str) -> dict[str, Any]:
    defaults = {
        "seq_len": 4096,
        "local_batch_size": 4,
        "global_batch_size": 128,
        "steps": 1000,
        "warmup_steps": 100,
        "lr": 3e-4,
        "weight_decay": 0.1,
        "data_parallel_replicate_degree": 1,
    }
    section = ""
    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if "=" not in line:
                continue

            key, value = [part.strip() for part in line.split("=", 1)]
            if value.startswith('"') and value.endswith('"'):
                parsed: Any = value.strip('"')
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        continue

            if section == "training":
                if key in {"seq_len", "local_batch_size", "global_batch_size", "steps"}:
                    defaults[key] = int(parsed)
            elif section == "lr_scheduler" and key == "warmup_steps":
                defaults["warmup_steps"] = int(parsed)
            elif section == "optimizer":
                if key == "lr":
                    defaults["lr"] = float(parsed)
                elif key == "weight_decay":
                    defaults["weight_decay"] = float(parsed)
            elif section == "parallelism" and key == "data_parallel_replicate_degree":
                defaults["data_parallel_replicate_degree"] = int(parsed)
    return defaults


def compute_chinchilla_schedule(
    param_count: int,
    tokens_per_param: float,
    seq_len: int,
    global_batch_size: int,
    warmup_ratio: float,
) -> dict[str, int]:
    target_tokens = int(round(param_count * tokens_per_param))
    tokens_per_step = global_batch_size * seq_len
    steps = math.ceil(target_tokens / tokens_per_step)
    warmup_steps = max(100, min(4000, int(round(steps * warmup_ratio))))
    if warmup_steps >= steps:
        warmup_steps = max(1, steps // 10)
    return {
        "target_tokens": target_tokens,
        "tokens_per_step": tokens_per_step,
        "steps": steps,
        "warmup_steps": warmup_steps,
    }


def _format_toml_value(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _ensure_section(lines: list[str], section_name: str) -> list[str]:
    header = f"[{section_name}]"
    if any(line.strip() == header for line in lines):
        return lines
    if lines and lines[-1].strip():
        lines.append("\n")
    lines.append(f"{header}\n")
    return lines


def create_modified_config(
    template_path: str,
    output_path: str,
    updates: dict[str, Any],
    section_updates: dict[str, dict[str, Any]] | None = None,
) -> None:
    with open(template_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        replaced = False
        for key, value in updates.items():
            if stripped.startswith(f"{key} ="):
                new_lines.append(f"{key} = {_format_toml_value(value)}\n")
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    existing_sections: dict[str, int] = {}
    for idx, line in enumerate(new_lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            existing_sections[stripped[1:-1]] = idx

    if section_updates:
        for section_name, kv_pairs in section_updates.items():
            if section_name not in existing_sections:
                _ensure_section(new_lines, section_name)
                existing_sections[section_name] = len(new_lines) - 1

            section_start = existing_sections[section_name]
            section_end = len(new_lines)
            for i in range(section_start + 1, len(new_lines)):
                stripped = new_lines[i].strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    section_end = i
                    break

            for key, value in kv_pairs.items():
                replaced = False
                for i in range(section_start + 1, section_end):
                    stripped = new_lines[i].strip()
                    if stripped.startswith(f"{key} ="):
                        new_lines[i] = f"{key} = {_format_toml_value(value)}\n"
                        replaced = True
                        break
                if not replaced:
                    insert_at = section_end
                    new_lines.insert(insert_at, f"{key} = {_format_toml_value(value)}\n")
                    section_end += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _count_params_from_instantiated_model(model_name: str, tying_mode: str) -> int:
    model_args = copy.deepcopy(qwen3_args[model_name])
    if tying_mode == "on":
        model_args.enable_weight_tying = True
    elif tying_mode == "off":
        model_args.enable_weight_tying = False

    with torch.device("meta"):
        model = Qwen3Model(model_args)

    if model_args.enable_weight_tying:
        model.output.weight = model.tok_embeddings.weight

    return sum(p.numel() for p in model.parameters())


def get_param_count(model_name: str, tying_mode: str) -> tuple[int, str]:
    try:
        nparams = _count_params_from_instantiated_model(model_name, tying_mode)
        return nparams, "instantiated"
    except Exception as exc:
        fallback = APPROX_PARAM_FALLBACK[model_name]
        print(
            f"  -> Warning: failed instantiated param count for {model_name}: {exc}. "
            f"Using fallback ~{fallback:,}."
        )
        return fallback, "fallback"


def choose_batch_plan(
    local_batch_size: int,
    global_batch_size: int,
    requested_dp_degree: int,
    available_gpus: int,
    target_grad_accum: int,
    max_grad_accum: int,
) -> BatchPlan:
    if local_batch_size <= 0 or global_batch_size <= 0:
        raise ValueError("local_batch_size and global_batch_size must be positive.")

    if requested_dp_degree > 0:
        denom = local_batch_size * requested_dp_degree
        if global_batch_size % denom != 0:
            raise ValueError(
                "Infeasible batch plan: global_batch_size must be divisible by "
                f"local_batch_size * dp_degree ({global_batch_size} % {denom} != 0)."
            )
        return BatchPlan(
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            dp_degree=requested_dp_degree,
            grad_accum_steps=global_batch_size // denom,
        )

    max_dp = available_gpus if available_gpus > 0 else global_batch_size // local_batch_size
    max_dp = max(1, max_dp)

    candidates: list[BatchPlan] = []
    for dp in range(1, max_dp + 1):
        denom = local_batch_size * dp
        if global_batch_size % denom != 0:
            continue
        grad_accum = global_batch_size // denom
        candidates.append(
            BatchPlan(
                local_batch_size=local_batch_size,
                global_batch_size=global_batch_size,
                dp_degree=dp,
                grad_accum_steps=grad_accum,
            )
        )

    if not candidates:
        raise ValueError(
            "No feasible (dp_degree, grad_accum) found. "
            "Try adjusting local/global batch sizes."
        )

    preferred = [c for c in candidates if c.grad_accum_steps <= target_grad_accum]
    bounded = [c for c in candidates if c.grad_accum_steps <= max_grad_accum]
    pool = preferred or bounded or candidates

    # Choose largest dp (maximizes throughput) among acceptable candidates.
    best = max(pool, key=lambda c: c.dp_degree)
    return best


def resolve_effective_gpu_cap(model_name: str, available_gpus: int) -> int:
    if available_gpus > 0:
        return available_gpus
    return MODEL_SIZE_MAX_GPU_DEFAULTS[model_name]


def main() -> None:
    args = parse_args()
    models = parse_model_list(args.models)
    run_id = get_timestamp()
    if args.tag:
        run_id = f"{run_id}_{args.tag}"

    output_root = os.path.join(BASE_DIR, args.output_root, run_id)
    if not args.dry_run:
        os.makedirs(output_root, exist_ok=True)

    print(f"=== Generating configs only: {run_id} ===")
    print(f"Models: {', '.join(models)}")
    print(f"Dataset: {args.dataset}")
    print(f"Output root: {output_root}")
    print(
        "Batch principle: global_batch = local_batch * dp_degree * grad_accum. "
        "Use largest fitting local_batch, then scale dp to keep grad_accum low."
    )

    for model_name in models:
        print(f"\n--- {model_name} ---")
        if model_name not in qwen3_args:
            raise ValueError(f"Model flavor '{model_name}' not found in torchtitan qwen3_args.")

        baseline_template = choose_baseline_template(args.baseline_template, model_name)
        template_defaults = read_template_defaults(baseline_template)
        model_defaults = MODEL_DEFAULTS[model_name]

        global_batch_size = int(
            args.global_batch_size
            if args.global_batch_size > 0
            else model_defaults.get("global_batch_size", template_defaults["global_batch_size"])
        )
        lr = float(model_defaults.get("lr", template_defaults["lr"]))
        weight_decay = float(
            model_defaults.get("weight_decay", template_defaults["weight_decay"])
        )
        seq_len = int(template_defaults["seq_len"])
        effective_gpu_cap = resolve_effective_gpu_cap(model_name, args.available_gpus)
        tp_degree = args.tp_degree if args.tp_degree > 0 else MODEL_TP_DEGREE.get(model_name, 1)

        # DP can only use GPUs not consumed by tensor parallelism.
        dp_gpu_cap = effective_gpu_cap // tp_degree

        # Determine local batch size: CLI override > auto-estimate.
        if args.local_batch_size > 0:
            local_batch_size = args.local_batch_size
        else:
            estimated_lbs, mem_breakdown = estimate_max_local_batch_size(
                model_name=model_name,
                seq_len=seq_len,
                dp_degree=dp_gpu_cap,
                tp_degree=tp_degree,
                gpu_memory_gb=args.gpu_memory_gb,
            )
            fallback_lbs = model_defaults.get(
                "local_batch_size", template_defaults["local_batch_size"]
            )
            local_batch_size = estimated_lbs
            print(
                f"  mem estimate ({args.gpu_memory_gb}GB, 70% usable): "
                f"params={mem_breakdown['params_gb']:.1f}GB, "
                f"grads={mem_breakdown['grads_gb']:.1f}GB, "
                f"optim={mem_breakdown['optim_gb']:.1f}GB, "
                f"act/sample={mem_breakdown['act_per_sample_gb']:.2f}GB "
                f"(layers={mem_breakdown.get('layer_act_gb', 0):.2f} + "
                f"logits={mem_breakdown.get('logits_gb', 0):.2f} + "
                f"loss_ws={mem_breakdown.get('loss_ws_gb', 0):.2f}), "
                f"max_samples={mem_breakdown.get('max_samples_exact', '?')}"
            )
            if estimated_lbs != fallback_lbs:
                print(
                    f"  lbs: {fallback_lbs} (table default) -> {estimated_lbs} (estimated)"
                )

        local_batch_size = int(local_batch_size)

        plan = choose_batch_plan(
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            requested_dp_degree=args.dp_degree,
            available_gpus=dp_gpu_cap,
            target_grad_accum=args.target_grad_accum,
            max_grad_accum=args.max_grad_accum,
        )

        nparams, count_source = get_param_count(model_name, args.weight_tying_for_count)
        schedule = compute_chinchilla_schedule(
            param_count=nparams,
            tokens_per_param=args.tokens_per_param,
            seq_len=seq_len,
            global_batch_size=plan.global_batch_size,
            warmup_ratio=args.warmup_ratio,
        )

        model_dir = os.path.join(output_root, model_name)
        config_dir = os.path.join(model_dir, "configs")
        base_output_dir = os.path.join(BASE_DIR, "outputs", run_id, model_name)

        hf_assets = MODEL_HF_ASSETS.get(model_name, f"./assets/hf/Qwen3-{model_name}")

        # Scale checkpoint interval: save ~20 checkpoints per run, clamped to [250, 2000].
        ckpt_interval = max(250, min(2000, schedule["steps"] // 20))

        common_updates = {
            "flavor": model_name,
            "hf_assets_path": hf_assets,
            "dataset": args.dataset,
            "local_batch_size": plan.local_batch_size,
            "global_batch_size": plan.global_batch_size,
            "data_parallel_replicate_degree": plan.dp_degree,
            "tensor_parallel_degree": tp_degree,
            "lr": lr,
            "weight_decay": weight_decay,
            "description": args.description,
            "steps": schedule["steps"],
            "warmup_steps": schedule["warmup_steps"],
            "interval": ckpt_interval,
            "enable": True,
            "async_mode": "async",
        }

        print(
            f"  params={nparams:,} ({count_source}, tying={args.weight_tying_for_count}), "
            f"target_tokens={schedule['target_tokens']:,}, "
            f"steps={schedule['steps']}, warmup={schedule['warmup_steps']}"
        )
        print(
            f"  lr={lr:g}, wd={weight_decay:g}, lbs={plan.local_batch_size}, "
            f"gbs={plan.global_batch_size}, dp={plan.dp_degree}, tp={tp_degree}, "
            f"grad_accum={plan.grad_accum_steps}, ckpt_interval={ckpt_interval}"
        )
        total_gpus = plan.dp_degree * tp_degree
        if args.available_gpus > 0:
            print(f"  gpu_cap={effective_gpu_cap} (from --available-gpus), total_gpus={total_gpus} (dp={plan.dp_degree} x tp={tp_degree})")
        else:
            print(f"  gpu_cap={effective_gpu_cap} (model-size default), total_gpus={total_gpus} (dp={plan.dp_degree} x tp={tp_degree})")
        print(f"  template={baseline_template}")

        if args.dry_run:
            continue

        baseline_cfg_path = os.path.join(config_dir, "baseline.toml")
        baseline_updates = common_updates.copy()
        baseline_updates["dump_folder"] = os.path.join(base_output_dir, "baseline")
        create_modified_config(baseline_template, baseline_cfg_path, baseline_updates)

        for coeff in DISPERSION_COEFFS:
            disp_cfg_path = os.path.join(config_dir, f"dispersion_{coeff}.toml")
            disp_updates = common_updates.copy()
            disp_updates["dump_folder"] = os.path.join(base_output_dir, f"dispersion_{coeff}")
            dispersion_section = {
                "variant": "angular_spread",
                "dispersion_coeff": coeff,
                "dispersion_loc": "all",
                "tau_l2": 1.0,
                "tau_cos": 1.0,
                "max_tokens": 256,
            }
            create_modified_config(
                baseline_template,
                disp_cfg_path,
                disp_updates,
                section_updates={"dispersion": dispersion_section},
            )

    if args.dry_run:
        print("\nDry run complete: no files written.")
    else:
        print("\nGeneration complete.")
        print(f"Configs saved in: {output_root}")


if __name__ == "__main__":
    main()
