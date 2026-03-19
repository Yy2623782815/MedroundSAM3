#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "${PROJECT_ROOT}/$p"
  fi
}

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/SAM3_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/outputs}"

SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

SAM3_REPO_ROOT="${SAM3_REPO_ROOT:-${PROJECT_ROOT}/repos/MedSAM3}"
MY_LORA_PROJECT_ROOT="${MY_LORA_PROJECT_ROOT:-${PROJECT_ROOT}/work/medsam3_my_lora}"
LORA_CHECKPOINT_PATH="${LORA_CHECKPOINT_PATH:-${PROJECT_ROOT}/work/medsam3_my_lora/outputs/chaos_smoke/checkpoints/best.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/models/sam3_base/sam3.pt}"
BPE_PATH="${BPE_PATH:-${PROJECT_ROOT}/repos/MedSAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"

DEVICE="${DEVICE:-cuda}"
RESOLUTION="${RESOLUTION:-1008}"
DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.5}"
NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.5}"

DATA_ROOT="$(resolve_path "${DATA_ROOT}")"
OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT}")"
SAM3_REPO_ROOT="$(resolve_path "${SAM3_REPO_ROOT}")"
MY_LORA_PROJECT_ROOT="$(resolve_path "${MY_LORA_PROJECT_ROOT}")"
LORA_CHECKPOINT_PATH="$(resolve_path "${LORA_CHECKPOINT_PATH}")"
CHECKPOINT_PATH="$(resolve_path "${CHECKPOINT_PATH}")"
BPE_PATH="$(resolve_path "${BPE_PATH}")"

# 指定评测数据集（空表示交给 Python 默认列表）
# 例如：DATASETS="CHAOS BraTS"
DATASETS="${DATASETS:-}"
# 为 1 时自动评测 data_root 下所有可发现数据集
USE_ALL_DATASETS="${USE_ALL_DATASETS:-0}"

cd "${EVAL_ROOT}"

CMD=(
python eval_medsam3_my_lora_gt_label_batch.py
  --data_root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --output_dir "${OUTPUT_ROOT}" \
  --sam3_repo_root "${SAM3_REPO_ROOT}" \
  --my_lora_project_root "${MY_LORA_PROJECT_ROOT}" \
  --lora_checkpoint_path "${LORA_CHECKPOINT_PATH}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --bpe_path "${BPE_PATH}" \
  --device "${DEVICE}" \
  --resolution "${RESOLUTION}" \
  --detection_threshold "${DETECTION_THRESHOLD}" \
  --nms_iou_threshold "${NMS_IOU_THRESHOLD}"
)

if [ "${USE_ALL_DATASETS}" = "1" ]; then
  CMD+=(--use_all_datasets)
elif [ -n "${DATASETS}" ]; then
  # shellcheck disable=SC2206
  DATASET_ARR=(${DATASETS})
  CMD+=(--datasets "${DATASET_ARR[@]}")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"
