#!/bin/bash
#SBATCH --job-name=ds_sst2_sae
#SBATCH --partition=gpu-h100,gpu-a100-dacdt,gpu-a100-dacdtbig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/project/logs
#SBATCH --error=/path/to/project/logs

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Start time: $(date)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepseek-sae

SCRIPT_DIR="/path/to/project/sst2/Qwen2.5 1.5B"
python "$SCRIPT_DIR/deepseek_r1_1_5b_pretrained.py"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
