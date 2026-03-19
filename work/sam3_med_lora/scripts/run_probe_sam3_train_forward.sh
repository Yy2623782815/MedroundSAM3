#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH:-${REPO_ROOT}/conda_envs/sam3_med_lora}"

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd "${PROJECT_ROOT}"
python tools/probe_sam3_train_forward.py
