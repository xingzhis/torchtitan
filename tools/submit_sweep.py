import os
import subprocess
import argparse
from datetime import datetime

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISPERSION_TEMPLATE = os.path.join(BASE_DIR, "configs", "final", "qwen_test_dispersion.toml")
BASELINE_CONFIG = os.path.join(BASE_DIR, "configs", "final", "qwen_test_baseline.toml")
SBATCH_FILE = os.path.join(BASE_DIR, "train.sbatch")

DISPERSION_COEFFS = [1.0, 0.5, 0.1, 0.01]

def parse_args():
    parser = argparse.ArgumentParser(description="Submit training sweep jobs.")
    parser.add_argument("--model", type=str, default="width_256", help="Model flavor (e.g., width_256, width_512, 0.6B)")
    parser.add_argument("--dataset", type=str, default="fineweb_edu", help="Dataset name in torchtitan")
    parser.add_argument("--tag", type=str, default="", help="Tag for this run (folders and wandb)")
    parser.add_argument("--description", type=str, default="Qwen training sweep", help="Job description")
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

def create_modified_config(template_path, output_path, updates):
    """Reads template and applies updates to specific keys."""
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
                if isinstance(value, str):
                    new_lines.append(f'{key} = "{value}"\n')
                else:
                    new_lines.append(f'{key} = {value}\n')
                replaced = True
                break
        
        if not replaced:
            new_lines.append(line)
            
    with open(output_path, "w") as f:
        f.writelines(new_lines)

def main():
    args = parse_args()
    timestamp = get_timestamp()
    
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
    
    # WandB Setup
    wandb_env = {
        "WANDB_RUN_GROUP": run_id,
        "WANDB_TAGS": args.tag if args.tag else "sweep"
    }

    # --- Auto-Tuning Logic ---
    # Defaults based on Chinchilla Law (20 tokens per param)
    # Assuming Global Batch Size = 128 (0.5M tokens)
    MODEL_CONFIGS = {
        "width_256": {"steps": 2480,  "lr": 3e-4, "warmup_steps": 100},  # 60M  -> 1.2B tokens
        "width_512": {"steps": 6610,  "lr": 3e-4, "warmup_steps": 500},  # 173M -> 3.46B tokens (Calculated: 173M * 20 / 0.5M)
        "0.6B":      {"steps": 24000, "lr": 3e-4, "warmup_steps": 1000}, # 600M -> 12B tokens
    }

    # Heuristic for Local Batch Size (Output of 0.6B is large!)
    local_batch_size = 4
    if args.model in ["width_512", "width_768"]:
        local_batch_size = 4  # Should fit on 40GB
    elif args.model in ["0.6B", "1.7B"]:
        local_batch_size = 2  # Safer for 0.6B+
        print(f"  -> Large model {args.model}: Dropping local_batch_size to {local_batch_size}")

    common_updates = {
        "flavor": args.model,
        "dataset": args.dataset,
        "local_batch_size": local_batch_size,
        "description": args.description
    }

    # Apply Model-Specific Defaults if known
    if args.model in MODEL_CONFIGS:
        defaults = MODEL_CONFIGS[args.model]
        print(f"  -> Applying Chinchilla defaults for {args.model}: {defaults}")
        common_updates.update(defaults)
    else:
        print(f"  -> Warning: No defaults for {args.model}, using baseline settings.")

    # 1. Baseline
    baseline_out = os.path.join(base_output_dir, "baseline")
    baseline_cfg_path = os.path.join(epoch_config_dir, "baseline.toml")
    
    baseline_updates = common_updates.copy()
    baseline_updates["dump_folder"] = baseline_out
    
    create_modified_config(BASELINE_CONFIG, baseline_cfg_path, baseline_updates)
    submit_job(baseline_cfg_path, f"{args.model}_base", wandb_env)
    
    # 2. Dispersion Sweep
    for coeff in DISPERSION_COEFFS:
        disp_out = os.path.join(base_output_dir, f"dispersion_{coeff}")
        disp_cfg_path = os.path.join(epoch_config_dir, f"dispersion_{coeff}.toml")
        
        disp_updates = common_updates.copy()
        disp_updates["dump_folder"] = disp_out
        disp_updates["dispersion_coeff"] = coeff
        
        create_modified_config(DISPERSION_TEMPLATE, disp_cfg_path, disp_updates)
        submit_job(disp_cfg_path, f"{args.model}_d{coeff}", wandb_env)

    print(f"\n=== Sweep Submitted ===")
    print(f"Configs saved in: {epoch_config_dir}")
    print(f"Outputs will be in: {base_output_dir}")

if __name__ == "__main__":
    main()
