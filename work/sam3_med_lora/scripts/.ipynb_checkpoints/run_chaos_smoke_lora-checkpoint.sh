#!/usr/bin/env bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /root/autodl-tmp/conda_envs/sam3_med_lora

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export PYTHONPATH=/root/autodl-tmp/work/sam3_med_lora:$PYTHONPATH

cd /root/autodl-tmp/work/sam3_med_lora

python train/train_lora_labelname.py \
  --config /root/autodl-tmp/work/sam3_med_lora/configs/chaos_smoke_lora.yaml
