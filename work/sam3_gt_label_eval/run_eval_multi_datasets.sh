#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export PYTHONPATH=/root/autodl-tmp/work/sam3_gt_label_eval:/root/autodl-tmp/repos/sam3

# 可改参数
SPLIT=${SPLIT:-test}          # training / test / all
MAX_SAMPLES=${MAX_SAMPLES:-0}  # >0 表示每个选中 split 最多取多少个；<=0 表示该 split 全部
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/work/sam3_gt_label_eval/outputs/multi_datasets_${SPLIT}_${MAX_SAMPLES}}

python3 /root/autodl-tmp/work/sam3_gt_label_eval/eval_sam3_gt_label_batch.py \
  --data_root /root/autodl-tmp/data/SAM3_data \
  --datasets AMOS2022 BraTS CHAOS CMRxMotions COVID19 Prostate SegRap2023 \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --output_dir "${OUTPUT_ROOT}"