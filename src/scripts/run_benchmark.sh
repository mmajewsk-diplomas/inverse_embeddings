#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="/basksir/vhome/mslowikowski/scratch/miniconda3"
ENV_NAME="sipit_env"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${SCRIPT_DIR}"

if [ "$#" -eq 0 ]; then
  python benchmark_time.py \
    --model openai-community/gpt2 \
    --samples 10 \
    --prompt_len 15
else
  python benchmark_time.py "$@"
fi
