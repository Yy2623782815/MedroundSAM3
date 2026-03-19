#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

AGENT_EVAL_ROOT="${AGENT_EVAL_ROOT:-${PROJECT_ROOT}/work/sam3_med_agent_eval}"
SAM3_REPO_ROOT="${SAM3_REPO_ROOT:-${PROJECT_ROOT}/repos/sam3}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/SAM3_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${AGENT_EVAL_ROOT}/outputs/AMOS2022_test50}"

PYTHONPATH="${AGENT_EVAL_ROOT}:${SAM3_REPO_ROOT}:${PYTHONPATH:-}" \
python3 "${AGENT_EVAL_ROOT}/eval_medical_agent_batch.py" \
  --data_root "${DATA_ROOT}" \
  --dataset AMOS2022 \
  --split test \
  --max_samples 4 \
  --max_agent_rounds 5 \
  --debug \
  --output_dir "${OUTPUT_DIR}"
