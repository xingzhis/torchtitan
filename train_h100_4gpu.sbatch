#!/bin/bash
#SBATCH --job-name=qwen_sweep
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          # 4 GPUs → all 48 CPUs
#SBATCH --mem=0
#SBATCH --gres=gpu:h100:4
#SBATCH --time=2-00:00:00           # cluster max for gpu partition
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- Environment Setup ---
# No micromamba binary on this cluster; prepend the env's bin directly.
# This is equivalent to activation and works without any shell hooks.
export PATH="/gpfs/radev/project/krishnaswamy_smita/xs272/micromamba/envs/torchtitan/bin:$PATH"

# --- Settings ---
export NGPU=4
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE environment variable is not set."
    exit 1
fi

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Config:    $CONFIG_FILE"
echo "GPUs:      $NGPU"

# --- CUDA / Memory ---
# Expandable segments reduces fragmentation from variable-size allocations.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- NCCL Tuning (H100, intra-node) ---
# Let NCCL auto-detect the best transport (NVLink, PCIe, SHM) for each GPU pair.
# No InfiniBand on this node.
export NCCL_IB_DISABLE=1
# Only log warnings/errors (INFO adds overhead).
export NCCL_DEBUG=INFO

# --- Data Loading ---
# 2 sockets × 24 cores (Intel 6442Y). 48 CPUs / 4 GPUs = 12 threads each.
export OMP_NUM_THREADS=12

# --- Execution ---
# run_train.sh already reads $CONFIG_FILE; pass extra overrides via "$@" only.
./run_train.sh
