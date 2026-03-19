#!/bin/bash
#SBATCH --job-name=dual-align
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/project/logs
#SBATCH --error=/path/to/project/logs

SCRIPT_DIR="/path/to/project/sst2"
PYTHON="python"

echo "Job started: $(date)"
echo "Node: $(hostname)"

$PYTHON "$SCRIPT_DIR/dual_alignment_analysis.py" --model_a gpt2 --model_b qwen25

echo "Job finished: $(date)"
