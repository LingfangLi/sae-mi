#!/bin/bash
#SBATCH --job-name=qwen25-sae-extract
#SBATCH --partition=gpu-a100-dacdt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/project/logs
#SBATCH --error=/path/to/project/logs

SCRIPT_DIR="/path/to/project/sst2/Qwen2.5 1.5B"
PYTHON="python"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$PYTHON "$SCRIPT_DIR/extract_sae_activations.py"

echo "Job finished: $(date)"
