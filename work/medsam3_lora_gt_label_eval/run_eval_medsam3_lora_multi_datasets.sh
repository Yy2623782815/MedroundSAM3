#!/bin/bash
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

# 数据根目录
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/SAM3_data}"
# 输出根目录
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/outputs}"

# 评测 split: training / test / all
SPLIT="${SPLIT:-test}"
# 每个选定 split 最多取多少个样本；<=0 表示全量
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# MedSAM3 配置文件
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/repos/MedSAM3/configs/full_lora_config.yaml}"
# LoRA 权重路径
LORA_WEIGHTS="${LORA_WEIGHTS:-${PROJECT_ROOT}/models/medsam3_lora/best_lora_weights.pt}"
# 本地 SAM3 基础权重
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/models/sam3_base/sam3.pt}"

# 推理设备
DEVICE="${DEVICE:-cuda}"
# 作者推理时使用的输入分辨率
RESOLUTION="${RESOLUTION:-1008}"
# query 分数筛选阈值
DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.5}"
# NMS 的 IoU 阈值
NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.5}"

DATA_ROOT="$(resolve_path "${DATA_ROOT}")"
OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT}")"
CONFIG_PATH="$(resolve_path "${CONFIG_PATH}")"
LORA_WEIGHTS="$(resolve_path "${LORA_WEIGHTS}")"
CHECKPOINT_PATH="$(resolve_path "${CHECKPOINT_PATH}")"

DATASETS=(
  AMOS2022
  BraTS
  CHAOS
  CMRxMotions
  COVID19
  Prostate
  SegRap2023
)

cd "${EVAL_ROOT}"

python eval_medsam3_lora_gt_label_batch.py \
  --data_root "${DATA_ROOT}" \
  --datasets "${DATASETS[@]}" \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --output_dir "${OUTPUT_ROOT}" \
  --config_path "${CONFIG_PATH}" \
  --lora_weights "${LORA_WEIGHTS}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --resolution "${RESOLUTION}" \
  --detection_threshold "${DETECTION_THRESHOLD}" \
  --nms_iou_threshold "${NMS_IOU_THRESHOLD}"
