#!/bin/bash
#SBATCH --job-name=gpt2-describe
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/project/logs
#SBATCH --error=/path/to/project/logs

SCRIPT_DIR="/path/to/project/sst2/GPT2"
PYTHON="python"

export GROQ_API_KEY="YOUR_GROQ_API_KEY"

echo "Job started: $(date)"
echo "Node: $(hostname)"

$PYTHON "$SCRIPT_DIR/generate_descriptions.py"

echo "Job finished: $(date)"
