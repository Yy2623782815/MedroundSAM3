#!/usr/bin/env bash

# huggingface cache缓存设置
# 默认放在仓库下 .cache/huggingface，可通过 HF_HOME 覆盖
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
DEFAULT_HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

export HF_HOME="${HF_HOME:-${DEFAULT_HF_HOME}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

#huggingface镜像设置
export HF_ENDPOINT=https://hf-mirror.com
