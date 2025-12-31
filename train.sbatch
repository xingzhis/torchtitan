#!/bin/bash
#SBATCH --job-name=qwen3-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Activate environment
conda activate torchtitan

# Setup
mkdir -p logs
cd /gpfs/radev/home/xs272/project/Primus/third_party/torchtitan

# Run training
NGPU=1 CONFIG_FILE="./test_dispersion_midtraining.toml" ./run_train.sh

