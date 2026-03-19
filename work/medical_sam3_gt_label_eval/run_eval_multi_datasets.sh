#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_ROOT="${SCRIPT_DIR}"

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "${PROJECT_ROOT}/$p"
  fi
}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export PYTHONPATH="${EVAL_ROOT}:${PROJECT_ROOT}/repos/sam3:${PYTHONPATH:-}"

# 可改参数
SPLIT=${SPLIT:-test}             # training / test / all
MAX_SAMPLES=${MAX_SAMPLES:-0}   # >0 表示每个选中 split 最多取多少个；<=0 表示该 split 全部
CHECKPOINT=${CHECKPOINT:-${PROJECT_ROOT}/models/medical-sam3/checkpoint.pt}
SAM3_REPO_ROOT=${SAM3_REPO_ROOT:-${PROJECT_ROOT}/repos/sam3}
DEVICE=${DEVICE:-cuda}
CONF_THRES=${CONF_THRES:-0.5}
OUTPUT_ROOT=${OUTPUT_ROOT:-${EVAL_ROOT}/outputs/multi_datasets_${SPLIT}_${MAX_SAMPLES}}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/SAM3_data}

CHECKPOINT="$(resolve_path "${CHECKPOINT}")"
SAM3_REPO_ROOT="$(resolve_path "${SAM3_REPO_ROOT}")"
OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT}")"
DATA_ROOT="$(resolve_path "${DATA_ROOT}")"

python3 "${EVAL_ROOT}/eval_medical_sam3_gt_label_batch.py" \
  --data_root "${DATA_ROOT}" \
  --datasets AMOS2022 BraTS CHAOS CMRxMotions COVID19 Prostate SegRap2023 \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --output_dir "${OUTPUT_ROOT}" \
  --sam3_repo_root "${SAM3_REPO_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --confidence_threshold "${CONF_THRES}"
