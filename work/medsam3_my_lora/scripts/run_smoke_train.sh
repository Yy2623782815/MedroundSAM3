#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG_PATH="${PROJECT_ROOT}/configs/my_smoke_lora.yaml"
NUM_GPUS="${NUM_GPUS:-1}"
EXTRA_ARGS=("$@")

# 用法示例：
#   NUM_GPUS=1 bash work/medsam3_my_lora/scripts/run_smoke_train.sh
#   NUM_GPUS=2 bash work/medsam3_my_lora/scripts/run_smoke_train.sh
#   NUM_GPUS=4 bash work/medsam3_my_lora/scripts/run_smoke_train.sh --epochs 10
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[launch] torchrun --nproc_per_node=${NUM_GPUS}"
  torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    "${PROJECT_ROOT}/train_medsam3_my_lora.py" \
    --config "${CONFIG_PATH}" \
    "${EXTRA_ARGS[@]}"
else
  echo "[launch] single GPU python"
  python "${PROJECT_ROOT}/train_medsam3_my_lora.py" \
    --config "${CONFIG_PATH}" \
    "${EXTRA_ARGS[@]}"
fi
