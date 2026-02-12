import os
import subprocess
import argparse
import math
from datetime import datetime

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_CONFIG = os.path.join(BASE_DIR, "configs", "final", "qwen_test_baseline.toml")
SBATCH_FILE = os.path.join(BASE_DIR, "train.sbatch")

DISPERSION_COEFFS = [1.0, 0.5, 0.1, 0.01]

MODEL_DEFAULTS = {
    "width_256": {
        "params": 60_000_000,
        "lr": 3e-4,
        "weight_decay": 0.1,
        "local_batch_size": 4,
        "global_batch_size": 128,
    },
    "width_512": {
        "params": 173_248_000,
        "lr": 3e-4,
        "weight_decay": 0.1,
        "local_batch_size": 4,
        "global_batch_size": 128,
    },
    "0.6B": {
        "params": 600_000_000,
        "lr": 3e-4,
        "weight_decay": 0.1,
        "local_batch_size": 2,
        "global_batch_size": 256,
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Submit training sweep jobs.")
    parser.add_argument("--model", type=str, default="width_256", help="Model flavor (e.g., width_256, width_512, 0.6B)")
    parser.add_argument("--dataset", type=str, default="fineweb_edu", help="Dataset name in torchtitan")
    parser.add_argument("--tag", type=str, default="", help="Tag for this run (folders and wandb)")
    parser.add_argument("--description", type=str, default="Qwen training sweep", help="Job description")
    parser.add_argument(
        "--baseline-template",
        type=str,
        default="",
        help="Path to baseline template TOML. If omitted, uses configs/final/qwen_test_baseline_<model>.toml when present.",
    )
    parser.add_argument(
        "--tokens-per-param",
        type=float,
        default=20.0,
        help="Chinchilla token budget multiplier (default: 20).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup as ratio of total steps before clamping.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=0,
        help="Override global batch size for all generated configs (0 keeps defaults).",
    )
    return parser.parse_args()

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M")

def submit_job(config_path, job_name, env_vars=None):
    """Submits a Slurm job with exported environment variables."""
    config_path = os.path.abspath(config_path)
    
    # Base exports
    exports = [f"ALL", f"CONFIG_FILE={config_path}"]
    
    # Add custom env vars (WandB etc)
    if env_vars:
        for k, v in env_vars.items():
            exports.append(f"{k}={v}")
            
    export_str = ",".join(exports)
    
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--export={export_str}",
        SBATCH_FILE
    ]
    
    print(f"Submitting {job_name}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=BASE_DIR)
        print(f"  -> {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  -> FAILED: {e.stderr}")


def choose_baseline_template(args):
    if args.baseline_template:
        return os.path.abspath(args.baseline_template)
    model_specific = os.path.join(BASE_DIR, "configs", "final", f"qwen_test_baseline_{args.model}.toml")
    if os.path.exists(model_specific):
        return model_specific
    return BASELINE_CONFIG


def read_template_defaults(config_path):
    defaults = {
        "seq_len": 4096,
        "local_batch_size": 4,
        "global_batch_size": 128,
        "steps": 1000,
        "warmup_steps": 100,
        "lr": 3e-4,
        "weight_decay": 0.1,
    }
    section = ""
    with open(config_path, "r") as f:
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
                parsed = value.strip('"')
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
    return defaults


def compute_chinchilla_schedule(param_count, tokens_per_param, seq_len, global_batch_size, warmup_ratio):
    target_tokens = int(round(param_count * tokens_per_param))
    tokens_per_step = global_batch_size * seq_len
    steps = math.ceil(target_tokens / tokens_per_step)
    warmup_steps = max(100, min(2000, int(round(steps * warmup_ratio))))
    if warmup_steps >= steps:
        warmup_steps = max(1, steps // 10)
    return {
        "target_tokens": target_tokens,
        "tokens_per_step": tokens_per_step,
        "steps": steps,
        "warmup_steps": warmup_steps,
    }

def _format_toml_value(value):
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _ensure_section(lines, section_name):
    header = f"[{section_name}]"
    if any(line.strip() == header for line in lines):
        return lines
    if lines and lines[-1].strip():
        lines.append("\n")
    lines.append(f"{header}\n")
    return lines


def create_modified_config(template_path, output_path, updates, section_updates=None):
    """Reads template and applies top-level key updates and optional section updates."""
    with open(template_path, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Simple parser replacement
        replaced = False
        for key, value in updates.items():
            # Check for exact key match at start of line
            # This is naive TOML parsing but generally safe for this structure
            if stripped.startswith(f"{key} ="):
                new_lines.append(f"{key} = {_format_toml_value(value)}\n")
                replaced = True
                break
        
        if not replaced:
            new_lines.append(line)
            
    existing_sections = {}
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

    with open(output_path, "w") as f:
        f.writelines(new_lines)

def main():
    args = parse_args()
    timestamp = get_timestamp()
    baseline_template = choose_baseline_template(args)
    
    # Construct Run ID and Paths
    run_id = f"{timestamp}_{args.model}"
    if args.tag:
        run_id += f"_{args.tag}"
        
    # Output and Config Dirs
    # outputs/20260204_1030_width_512_large_run
    base_output_dir = os.path.join(BASE_DIR, "outputs", run_id)
    # configs/sweep/20260204_1030_width_512_large_run
    epoch_config_dir = os.path.join(BASE_DIR, "configs", "sweep", run_id)
    
    if not os.path.exists(epoch_config_dir):
        os.makedirs(epoch_config_dir)
    
    print(f"=== Starting Sweep: {run_id} ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output Dir: {base_output_dir}")
    print(f"Baseline Template: {baseline_template}")
    
    # WandB Setup
    wandb_tags = [args.model]
    if args.tag:
        wandb_tags.append(args.tag)
    else:
        wandb_tags.append("sweep")

    wandb_env = {
        "WANDB_RUN_GROUP": run_id,
        "WANDB_TAGS": ",".join(wandb_tags),
    }

    # --- Auto-Tuning Logic ---
    model_defaults = MODEL_DEFAULTS.get(args.model, {})
    template_defaults = read_template_defaults(baseline_template)

    local_batch_size = int(model_defaults.get("local_batch_size", template_defaults["local_batch_size"]))
    global_batch_size = int(
        args.global_batch_size if args.global_batch_size > 0 else model_defaults.get("global_batch_size", template_defaults["global_batch_size"])
    )
    lr = float(model_defaults.get("lr", template_defaults["lr"]))
    weight_decay = float(model_defaults.get("weight_decay", template_defaults["weight_decay"]))
    seq_len = int(template_defaults["seq_len"])

    schedule = None
    if "params" in model_defaults:
        schedule = compute_chinchilla_schedule(
            param_count=int(model_defaults["params"]),
            tokens_per_param=args.tokens_per_param,
            seq_len=seq_len,
            global_batch_size=global_batch_size,
            warmup_ratio=args.warmup_ratio,
        )
        print(
            "  -> Chinchilla schedule: "
            f"params={model_defaults['params']:,}, "
            f"tokens_per_param={args.tokens_per_param}, "
            f"target_tokens={schedule['target_tokens']:,}, "
            f"tokens_per_step={schedule['tokens_per_step']:,}, "
            f"steps={schedule['steps']}, "
            f"warmup_steps={schedule['warmup_steps']}"
        )
    else:
        print(f"  -> Warning: No parameter count for {args.model}; using template steps/warmup.")

    common_updates = {
        "flavor": args.model,
        "dataset": args.dataset,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "description": args.description
    }

    if schedule is not None:
        common_updates["steps"] = schedule["steps"]
        common_updates["warmup_steps"] = schedule["warmup_steps"]
    else:
        common_updates["steps"] = template_defaults["steps"]
        common_updates["warmup_steps"] = template_defaults["warmup_steps"]

    # 1. Baseline
    baseline_out = os.path.join(base_output_dir, "baseline")
    baseline_cfg_path = os.path.join(epoch_config_dir, "baseline.toml")
    
    baseline_updates = common_updates.copy()
    baseline_updates["dump_folder"] = baseline_out
    
    create_modified_config(baseline_template, baseline_cfg_path, baseline_updates)
    submit_job(baseline_cfg_path, f"{args.model}_base", wandb_env)
    
    # 2. Dispersion Sweep
    for coeff in DISPERSION_COEFFS:
        disp_out = os.path.join(base_output_dir, f"dispersion_{coeff}")
        disp_cfg_path = os.path.join(epoch_config_dir, f"dispersion_{coeff}.toml")
        
        disp_updates = common_updates.copy()
        disp_updates["dump_folder"] = disp_out
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
        submit_job(disp_cfg_path, f"{args.model}_d{coeff}", wandb_env)

    print(f"\n=== Sweep Submitted ===")
    print(f"Configs saved in: {epoch_config_dir}")
    print(f"Outputs will be in: {base_output_dir}")

if __name__ == "__main__":
    main()
