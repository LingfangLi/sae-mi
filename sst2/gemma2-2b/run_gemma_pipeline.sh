#!/bin/bash
#SBATCH --job-name=gemma-pipe
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/project/logs
#SBATCH --error=/path/to/project/logs

GEMMA_DIR="/path/to/project/sst2/Gemma 2b"
SST2_DIR="/path/to/project/sst2"
PYTHON="python"

export GROQ_API_KEY="YOUR_GROQ_API_KEY"

echo "Job started: $(date)"
echo "Node: $(hostname)"

echo ""
echo "=== Step 1: Feature Selection ==="
$PYTHON "$GEMMA_DIR/feature_selection.py"

echo ""
echo "=== Step 2: Groq Descriptions ==="
$PYTHON "$GEMMA_DIR/generate_descriptions.py"

echo ""
echo "=== Step 3: Dual Alignment (GPT-2 vs Gemma) ==="
$PYTHON "$SST2_DIR/dual_alignment_analysis.py" --model_a gpt2 --model_b gemma2 --gemma_layer 13

echo ""
echo "=== Step 4: Dual Alignment (Qwen2.5 vs Gemma) ==="
$PYTHON "$SST2_DIR/dual_alignment_analysis.py" --model_a qwen25 --model_b gemma2 --gemma_layer 13

echo ""
echo "Job finished: $(date)"
