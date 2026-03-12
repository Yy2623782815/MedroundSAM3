# filename: /root/autodl-tmp/work/medical_sam3_gt_label_eval/run_eval_multi_datasets.sh
#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export PYTHONPATH=/root/autodl-tmp/work/medical_sam3_gt_label_eval:/root/autodl-tmp/repos/sam3

# 可改参数
SPLIT=${SPLIT:-test}             # training / test / all
MAX_SAMPLES=${MAX_SAMPLES:-0}   # >0 表示每个选中 split 最多取多少个；<=0 表示该 split 全部
CHECKPOINT=${CHECKPOINT:-/root/autodl-tmp/models/medical-sam3/checkpoint.pt}
SAM3_REPO_ROOT=${SAM3_REPO_ROOT:-/root/autodl-tmp/repos/sam3}
DEVICE=${DEVICE:-cuda}
CONF_THRES=${CONF_THRES:-0.5}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/work/medical_sam3_gt_label_eval/outputs/multi_datasets_${SPLIT}_${MAX_SAMPLES}}

python3 /root/autodl-tmp/work/medical_sam3_gt_label_eval/eval_medical_sam3_gt_label_batch.py \
  --data_root /root/autodl-tmp/data/SAM3_data \
  --datasets AMOS2022 BraTS CHAOS CMRxMotions COVID19 Prostate SegRap2023 \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --output_dir "${OUTPUT_ROOT}" \
  --sam3_repo_root "${SAM3_REPO_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --confidence_threshold "${CONF_THRES}"