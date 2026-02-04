import os
import subprocess
import sys

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISPERSION_TEMPLATE = os.path.join(BASE_DIR, "configs", "final", "qwen_test_dispersion.toml")
BASELINE_CONFIG = os.path.join(BASE_DIR, "configs", "final", "qwen_test_baseline.toml")
SWEEP_CONFIG_DIR = os.path.join(BASE_DIR, "configs", "sweep")
SBATCH_FILE = os.path.join(BASE_DIR, "train.sbatch")

DISPERSION_COEFFS = [1.0, 0.5, 0.1, 0.01]

def submit_job(config_path, job_name_suffix=""):
    """Submits a Slurm job for the given config."""
    config_path = os.path.abspath(config_path)
    job_name = f"qwen_{job_name_suffix}"
    
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--export=ALL,CONFIG_FILE={config_path}",
        SBATCH_FILE
    ]
    
    print(f"Submitting {job_name} ({os.path.basename(config_path)})...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=BASE_DIR)
        print(f"  -> {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  -> FAILED: {e.stderr}")

def create_config(coeff):
    """Reads template, modifies coeff and dump_folder, writes new file."""
    with open(DISPERSION_TEMPLATE, "r") as f:
        content = f.read()
    
    # Simple string replacement for robustness without TOML dependencies
    # Replace dispersion_coeff
    # We look for 'dispersion_coeff = 1.0' or similar. 
    # Note: The template currently has 'dispersion_coeff = 1.0'. 
    # To be safe, we'll replace the whole line if we find the key.
    
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("dispersion_coeff ="):
            new_lines.append(f"dispersion_coeff = {coeff}")
        elif line.strip().startswith("dump_folder ="):
            # Update dump folder to be unique
            # Original: dump_folder = "./outputs/qwen_test_dispersion"
            new_lines.append(f'dump_folder = "./outputs/qwen_test_dispersion_coeff_{coeff}"')
        else:
            new_lines.append(line)
            
    output_filename = f"qwen_dispersion_{coeff}.toml"
    output_path = os.path.join(SWEEP_CONFIG_DIR, output_filename)
    
    with open(output_path, "w") as f:
        f.write("\n".join(new_lines))
    
    return output_path

def main():
    if not os.path.exists(SWEEP_CONFIG_DIR):
        os.makedirs(SWEEP_CONFIG_DIR)
        
    print("=== Starting Sweep Submission ===")
    
    # 1. Submit Baseline
    print("\n--- Submitting Baseline ---")
    submit_job(BASELINE_CONFIG, "baseline")
    
    # 2. Submit Dispersion Sweep
    print("\n--- Submitting Dispersion Sweep ---")
    for coeff in DISPERSION_COEFFS:
        new_config = create_config(coeff)
        submit_job(new_config, f"disp_{coeff}")
        
    print("\n=== All Jobs Submitted ===")
    print(f"Check logs in {os.path.join(BASE_DIR, 'logs')}")
    print("Use 'squeue' to check status.")

if __name__ == "__main__":
    main()
