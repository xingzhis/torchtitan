#!/bin/bash
#SBATCH --job-name=qwen_sweep
#SBATCH --partition=test
#SBATCH --nodelist=misha02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- Environment Setup ---
# Attempt to load micromamba/conda
if [ -f "/diskarray/home/$USER/.bashrc" ]; then
    source "/diskarray/home/$USER/.bashrc"
fi

# Explicit fallback to include micromamba in path if not loaded
export PATH="/diskarray/home/$USER/micromamba/bin:$PATH"

# Hook micromamba for this shell
eval "$(micromamba shell hook --shell bash)"

# Activate environment
micromamba activate torchtitan

# --- Settings ---
export NGPU=2
# CONFIG_FILE is expected to be passed via --export=CONFIG_FILE=...

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE environment variable is not set."
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Config File: $CONFIG_FILE"
echo "GPUs: $NGPU"

# --- Execution ---
# We use the existing run_train.sh which handles the torchrun command
# We pass the config file as an argument
./run_train.sh --job.config_file "$CONFIG_FILE"
