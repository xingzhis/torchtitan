import os
import json
import argparse
import re
import torch
from lm_eval import simple_evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(s, filepath=None, to_console=True):
    if to_console:
        print(s)
    if filepath is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a+") as o:
            o.write(s + "\n")


def pick_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return "cuda"
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def pick_dtype(dtype_arg: str, device: str) -> str:
    if dtype_arg and dtype_arg != "auto":
        if device == "cpu" and dtype_arg in {"float16", "bfloat16"}:
            return "float32"
        return dtype_arg
    if device == "cpu":
        return "float32"
    return "float16"


def build_model_args(args, dtype: str, parallelize: bool, model_id: str) -> str:
    parts = [
        f"pretrained={model_id}",
        f"dtype={dtype}",
        "trust_remote_code=True",
    ]
    if args.cache_dir:
        parts.append(f"cache_dir={args.cache_dir}")
    if args.hf_token and args.hf_token.strip():
        parts.append(f"token={args.hf_token.strip()}")
    if parallelize:
        parts.append("parallelize=True")
        # Avoid multi-GPU device mismatch: inputs on cuda:0, embeddings on cuda:1.
        parts.append("device_map=sequential")
    return ",".join(parts)


def list_checkpoints(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    entries = []
    for name in os.listdir(checkpoint_dir):
        if not name.startswith("step-"):
            continue
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isdir(path):
            continue
        match = re.match(r"step-(\d+)$", name)
        if not match:
            continue
        step = int(match.group(1))
        entries.append((step, name, path))
    entries.sort(key=lambda x: x[0])
    return entries


def resolve_log_path(base_log_path: str, output_dir: str, checkpoint_name: str) -> str:
    if base_log_path:
        if os.path.isdir(base_log_path):
            return os.path.join(base_log_path, f"{checkpoint_name}.txt")
        root, ext = os.path.splitext(base_log_path)
        if not ext:
            ext = ".txt"
        return f"{root}_{checkpoint_name}{ext}"
    return os.path.join(output_dir, "log.txt")


def run_eval(args, model_id: str, output_dir: str, log_path: str):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    parallelize = args.parallelize
    if parallelize is None:
        parallelize = torch.cuda.is_available() and torch.cuda.device_count() > 1

    # When parallelize=True, use "cuda" so inputs match model device placement.
    eval_device = "cuda" if (parallelize and device.startswith("cuda")) else device
    model_args = build_model_args(args, dtype, parallelize, model_id)

    log("=== LMEval (pretrained) ===", filepath=log_path)
    log(f"Model: {model_id}", filepath=log_path)
    log(f"Device: {device} | Dtype: {dtype} | Parallelize: {parallelize}", filepath=log_path)
    log(f"Zero-shot tasks: {args.zeroshot_tasks}", filepath=log_path)
    log(f"Few-shot tasks: {args.fewshot_tasks}", filepath=log_path)
    log(f"Num fewshot: {args.num_fewshot}", filepath=log_path)
    log(f"Max eval samples: {args.max_eval_samples}", filepath=log_path)
    log(f"Context length: {args.context_len}", filepath=log_path)
    log(f"Max gen tokens: {args.max_gen_tokens}", filepath=log_path)
    log(f"Batch size: {args.batch_size}", filepath=log_path)
    log(f"Seed: {args.seed}", filepath=log_path)

    if args.max_gen_tokens > args.context_len:
        raise ValueError("max_gen_tokens must be <= context_len.")

    with torch.inference_mode():
        res_zeroshot = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=args.zeroshot_tasks,
            num_fewshot=0,
            batch_size=args.batch_size,
            device=eval_device,
            limit=args.max_eval_samples,
            gen_kwargs={"max_gen_toks": args.max_gen_tokens, "do_sample": False},
            log_samples=False,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

        res_fewshot = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=args.fewshot_tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=eval_device,
            limit=args.max_eval_samples,
            gen_kwargs={"max_gen_toks": args.max_gen_tokens, "do_sample": False},
            log_samples=False,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

    if "results" not in res_zeroshot or "results" not in res_fewshot:
        raise RuntimeError("LMEval did not return expected results.")

    merged_dict = {**res_zeroshot["results"], **res_fewshot["results"]}
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "lm_eval_begin_0.json")
    with open(out, "w") as f:
        json.dump({"results": merged_dict}, f, indent=2)
    log(f"Results saved to {out}", filepath=log_path)

    for task, metrics in merged_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    log(f"{task}.{metric_name}: {value:.4f}", filepath=log_path)


def main():
    ap = argparse.ArgumentParser(description="Run lm-eval on a pretrained Qwen3 model.")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B",
                    help="Hugging Face model id to evaluate.")
    ap.add_argument("--hf_token", type=str, default=None,
                    help="HF token if needed for gated/private models.")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="HF cache dir.")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for lm_eval JSON and logs.")
    ap.add_argument("--log_path", type=str, default=None,
                    help="Path to log file (default: <output_dir>/log.txt).")
    ap.add_argument("--checkpoint_dir", type=str,
                    default="./qwen3_0.6B_dispersion_huggingface",
                    help="Directory containing step-* checkpoints to evaluate.")
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--max_eval_samples", type=int, default=200)
    ap.add_argument("--context_len", type=int, default=4096,
                    help="Context length used during training (for consistency checks).")
    ap.add_argument("--max_gen_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=str, default="auto",
                    help="Batch size for lm_eval ('auto' or integer).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto",
                    help="Device for lm_eval (e.g. 'cuda', 'cuda:0', 'cpu').")
    ap.add_argument("--dtype", type=str, default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--parallelize", action="store_true", default=None,
                    help="Enable model parallelism across GPUs.")
    ap.add_argument("--zeroshot_tasks", type=str, nargs="+",
                    default=[
                        "anli",
                        "hellaswag",
                        "lambada",
                        "openbookqa",
                        "paloma_wikitext_103",
                        "piqa",
                        "truthfulqa_mc2",
                        "winogrande",
                    ])
    ap.add_argument("--fewshot_tasks", type=str, nargs="+",
                    default=[
                        "arc_challenge",
                        "arc_easy",
                        # "mathqa", # Disabled: HF datasets no longer supports dataset scripts (math_qa.py).
                        "mmlu",
                        "medmcqa",
                    ])
    args = ap.parse_args()

    model_str = args.model_name.replace("/", "-")
    args.dataset_name = "Salesforce/wikitext"
    args.lr = 0
    args.train_tokens = "156G"
    args.dispersion = "none"
    args.dispersion_coeff = 0.1
    args.dispersion_loc = "all"
    args.tau_cos = 1.0
    args.tau_l2 = 1.0
    if args.output_dir is None:
        args.output_dir = f'./results/pretrain_{model_str}_{"-".join(args.dataset_name.split("/"))}_lr-{args.lr}_token-{args.train_tokens}_disp-{args.dispersion}-{args.dispersion_coeff}-{args.dispersion_loc}-tau_cos-{args.tau_cos}-tau_l2-{args.tau_l2}_fewshot-{args.num_fewshot}_maxsample-{args.max_eval_samples}_seed-{args.seed}'

    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)

    checkpoints = list_checkpoints(args.checkpoint_dir)

    if not checkpoints:
        # Single model directory (no step-* subdirs).
        output_dir = args.output_dir
        log_path = resolve_log_path(args.log_path, output_dir, "model")
        run_eval(args, args.checkpoint_dir, output_dir, log_path)
    else:
        for _, checkpoint_name, checkpoint_path in checkpoints:
            output_dir = os.path.join(args.output_dir, checkpoint_name)
            log_path = resolve_log_path(args.log_path, output_dir, checkpoint_name)
            run_eval(args, checkpoint_path, output_dir, log_path)


if __name__ == "__main__":
    main()