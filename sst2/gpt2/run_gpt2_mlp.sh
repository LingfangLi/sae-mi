#!/bin/bash
#SBATCH --job-name=gpt2-mlp-sst2
#SBATCH --partition=gpu-a100-dacdt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/project/logs
#SBATCH --error=/path/to/project/logs

SCRIPT_DIR="/path/to/project/sst2/GPT2"
PYTHON="python"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$PYTHON "$SCRIPT_DIR/pretrained_gpt2_sae_mlp.py"

echo "Job finished: $(date)"
