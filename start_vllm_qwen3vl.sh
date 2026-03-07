#!/usr/bin/env bash
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
#huggingface镜像设置
export HF_ENDPOINT=https://hf-mirror.com

vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --tensor-parallel-size 1 \
  --allowed-local-media-path / \
  --enforce-eager \
  --port 8001