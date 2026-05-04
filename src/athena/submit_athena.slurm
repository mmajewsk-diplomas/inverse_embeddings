#!/bin/bash
#SBATCH --job-name=sipit_experiment
#SBATCH --output=logs/sipit_%j.out
#SBATCH --error=logs/sipit_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --mem=32G
#SBATCH -A plgaisec-gpu-a100

cd $SCRATCH/inverse_embeddings

module load Python/3.10.4
module load CUDA/11.8.0
source venv/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Start job on $(hostname)"
echo "GPU info:"
nvidia-smi

python run_athena_experiment.py \
    --model gpt2 \
    --samples 10 \
    --steps 1000 \
    --lr 0.05 \
    --max_len 150 \
    --out_dir $SCRATCH/sipit_results

echo "Job finished"