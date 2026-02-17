#!/usr/bin/env python
import argparse
import csv
import os
import sys
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools.generate_sweep_configs import (
    MODEL_DEFAULTS,
    SUPPORTED_MODELS,
    choose_baseline_template,
    choose_batch_plan,
    compute_chinchilla_schedule,
    get_param_count,
    parse_model_list,
    read_template_defaults,
    resolve_effective_gpu_cap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a planning table for Qwen3 sweep defaults."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(SUPPORTED_MODELS),
        help="Comma-separated model flavors to include.",
    )
    parser.add_argument(
        "--baseline-template",
        type=str,
        default="",
        help="Optional baseline TOML path.",
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
        help="Override global batch size for all models.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=0,
        help="Override local batch size for all models.",
    )
    parser.add_argument(
        "--dp-degree",
        type=int,
        default=0,
        help="Force data parallel degree.",
    )
    parser.add_argument(
        "--available-gpus",
        type=int,
        default=0,
        help="Cluster GPU cap. If omitted, uses model-size default cap.",
    )
    parser.add_argument(
        "--target-grad-accum",
        type=int,
        default=4,
        help="Prefer grad accumulation at or below this value.",
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
        help="Mode for parameter counting.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="",
        help="Optional output markdown file path.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional output CSV file path.",
    )
    return parser.parse_args()


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = parse_model_list(args.models)
    for model_name in models:
        baseline_template = choose_baseline_template(args.baseline_template, model_name)
        template_defaults = read_template_defaults(baseline_template)
        model_defaults = MODEL_DEFAULTS[model_name]

        local_batch_size = int(
            args.local_batch_size
            if args.local_batch_size > 0
            else model_defaults.get("local_batch_size", template_defaults["local_batch_size"])
        )
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
        gpu_cap = resolve_effective_gpu_cap(model_name, args.available_gpus)

        plan = choose_batch_plan(
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            requested_dp_degree=args.dp_degree,
            available_gpus=gpu_cap,
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

        rows.append(
            {
                "model": model_name,
                "params": nparams,
                "param_source": count_source,
                "lr": lr,
                "weight_decay": weight_decay,
                "seq_len": seq_len,
                "local_batch_size": plan.local_batch_size,
                "global_batch_size": plan.global_batch_size,
                "dp_degree": plan.dp_degree,
                "grad_accum_steps": plan.grad_accum_steps,
                "gpu_cap": gpu_cap,
                "target_tokens": schedule["target_tokens"],
                "tokens_per_step": schedule["tokens_per_step"],
                "steps": schedule["steps"],
                "warmup_steps": schedule["warmup_steps"],
                "baseline_template": baseline_template,
            }
        )
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| model | params | lr | wd | lbs | gbs | dp | grad_accum | gpu_cap | "
        "target_tokens | steps | warmup |\n"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = ""
    for r in rows:
        body += (
            f"| {r['model']} | {_fmt_int(r['params'])} | {r['lr']:.2e} | {r['weight_decay']:.3f} | "
            f"{r['local_batch_size']} | {r['global_batch_size']} | {r['dp_degree']} | "
            f"{r['grad_accum_steps']} | {r['gpu_cap']} | {_fmt_int(r['target_tokens'])} | "
            f"{_fmt_int(r['steps'])} | {_fmt_int(r['warmup_steps'])} |\n"
        )
    return header + sep + body


def maybe_write(path: str, content: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def maybe_write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "model",
        "params",
        "param_source",
        "lr",
        "weight_decay",
        "seq_len",
        "local_batch_size",
        "global_batch_size",
        "dp_degree",
        "grad_accum_steps",
        "gpu_cap",
        "target_tokens",
        "tokens_per_step",
        "steps",
        "warmup_steps",
        "baseline_template",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    md = to_markdown(rows)
    print(md)
    maybe_write(args.output_md, md)
    maybe_write_csv(args.output_csv, rows)
    if args.output_md:
        print(f"Wrote markdown table: {args.output_md}")
    if args.output_csv:
        print(f"Wrote CSV table: {args.output_csv}")


if __name__ == "__main__":
    main()
