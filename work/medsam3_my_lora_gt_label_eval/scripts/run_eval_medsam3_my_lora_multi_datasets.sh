#!/bin/bash
set -e

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data/SAM3_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/work/medsam3_my_lora_gt_label_eval/outputs}"

SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

SAM3_REPO_ROOT="${SAM3_REPO_ROOT:-/root/autodl-tmp/repos/MedSAM3}"
MY_LORA_PROJECT_ROOT="${MY_LORA_PROJECT_ROOT:-/root/autodl-tmp/work/medsam3_my_lora}"
LORA_CHECKPOINT_PATH="${LORA_CHECKPOINT_PATH:-/root/autodl-tmp/work/medsam3_my_lora/outputs/chaos_smoke/checkpoints/best.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/root/autodl-tmp/models/sam3_base/sam3.pt}"
BPE_PATH="${BPE_PATH:-/root/autodl-tmp/repos/MedSAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"

DEVICE="${DEVICE:-cuda}"
RESOLUTION="${RESOLUTION:-1008}"
DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.5}"
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

cd /root/autodl-tmp/work/medsam3_my_lora_gt_label_eval

python eval_medsam3_my_lora_gt_label_batch.py \
  --data_root "${DATA_ROOT}" \
  --datasets "${DATASETS[@]}" \
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
