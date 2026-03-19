#!/usr/bin/env bash
set -euo pipefail

# filename: /root/autodl-tmp/work/medsam3_my_lora/scripts/run_smoke_train.sh

export PYTHONPATH="/root/autodl-tmp/work/medsam3_my_lora:${PYTHONPATH:-}"

python /root/autodl-tmp/work/medsam3_my_lora/train_medsam3_my_lora.py \
  --config /root/autodl-tmp/work/medsam3_my_lora/configs/my_smoke_lora.yaml
