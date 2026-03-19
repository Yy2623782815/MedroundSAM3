#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/train_medsam3_my_lora.py" \
  --config "${PROJECT_ROOT}/configs/my_smoke_lora.yaml"
