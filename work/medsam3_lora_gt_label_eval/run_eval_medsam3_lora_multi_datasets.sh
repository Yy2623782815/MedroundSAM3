#!/bin/bash
set -e

# 数据根目录
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data/SAM3_data}"
# 输出根目录
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/work/medsam3_lora_gt_label_eval/outputs}"

# 评测 split: training / test / all
SPLIT="${SPLIT:-test}"
# 每个选定 split 最多取多少个样本；<=0 表示全量
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# MedSAM3 配置文件
CONFIG_PATH="${CONFIG_PATH:-/root/autodl-tmp/repos/MedSAM3/configs/full_lora_config.yaml}"
# LoRA 权重路径
LORA_WEIGHTS="${LORA_WEIGHTS:-/root/autodl-tmp/models/medsam3_lora/best_lora_weights.pt}"
# 本地 SAM3 基础权重
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/root/autodl-tmp/models/sam3_base/sam3.pt}"

# 推理设备
DEVICE="${DEVICE:-cuda}"
# 作者推理时使用的输入分辨率
RESOLUTION="${RESOLUTION:-1008}"
# query 分数筛选阈值
DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.5}"
# NMS 的 IoU 阈值
NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.5}"

DATASETS=(
  AMOS2022
  BraTS
  CHAOS
  CMRxMotions
  COVID19
  Prostate
  SegRap2023
)

cd /root/autodl-tmp/work/medsam3_lora_gt_label_eval

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