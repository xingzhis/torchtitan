#!/bin/bash
#SBATCH --job-name=qwen_sweep
#SBATCH --partition=test
#SBATCH --nodelist=misha02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- Environment Setup ---
if [ -f "/diskarray/home/$USER/.bashrc" ]; then
    source "/diskarray/home/$USER/.bashrc"
fi
export PATH="/diskarray/home/$USER/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate torchtitan

# --- Settings ---
export NGPU=2
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE environment variable is not set."
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Config File: $CONFIG_FILE"
echo "GPUs: $NGPU"

# --- CUDA / Memory ---
# Expandable segments reduces fragmentation from variable-size allocations.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Pin GPU memory for faster host<->device transfers (dataloader, checkpointing).
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- NCCL Tuning (PCIe A100s, no NVLink, cross-NUMA SYS topology) ---
# Force shared-memory transport for intra-node (faster than PCIe P2P across NUMA).
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=SYS
# No InfiniBand on this machine.
export NCCL_IB_DISABLE=1
# Use 2 NCCL threads per GPU for better overlap.
export NCCL_NTHREADS=512
export NCCL_BUFFSIZE=8388608

# --- Data Loading ---
# Pin each torchrun worker to its GPU's NUMA node for memory locality.
# GPU0 -> NUMA0 (cores 0-31), GPU1 -> NUMA1 (cores 32-63).
# torchrun spawns 2 processes; OMP threads for each should not over-subscribe.
export OMP_NUM_THREADS=8

# --- Execution ---
./run_train.sh --job.config_file "$CONFIG_FILE"
