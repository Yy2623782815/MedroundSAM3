#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"

ENV_PATH="${ENV_PATH:-${REPO_ROOT}/conda_envs/sam3_med_lora}"
PYTHON_VERSION="3.12"
SAM3_REPO="${SAM3_REPO:-${REPO_ROOT}/repos/sam3}"

echo "[0/7] remove old env if exists"
if [ -d "${ENV_PATH}" ]; then
    echo "Found existing env at ${ENV_PATH}, removing it..."
    rm -rf "${ENV_PATH}"
fi

echo "[1/7] create conda env at: ${ENV_PATH}"
conda create -y -p "${ENV_PATH}" python="${PYTHON_VERSION}"

echo "[2/7] activate env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "[3/7] upgrade pip/wheel and pin setuptools/numpy"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81" "numpy==1.26.4"

echo "[4/7] install PyTorch (CUDA 12.4 wheels)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[5/7] install common runtime/train deps"
pip install \
    requests \
    psutil \
    pandas \
    matplotlib \
    scikit-image \
    scikit-learn \
    "opencv-python<4.13" \
    pillow \
    pyyaml \
    tqdm \
    hydra-core \
    omegaconf \
    tensorboard \
    ftfy==6.1.1 \
    regex \
    safetensors \
    pycocotools \
    einops \
    decord \
    triton

echo "[6/7] install SAM3 repo"
cd "${SAM3_REPO}"
pip install -e ".[train]"

echo "[7/7] final sanity fix for numpy after sam3 editable install"
python -m pip install "numpy==1.26.4"

echo
echo "===== Environment sanity check ====="
python - << 'PY'
import sys
import torch
import numpy as np
import setuptools
import pkg_resources
import cv2
import psutil
import decord
import einops
import triton
import sam3

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("device name:", torch.cuda.get_device_name(0))

print("numpy:", np.__version__)
print("setuptools:", setuptools.__version__)
print("pkg_resources import: ok")
print("cv2:", cv2.__version__)
print("psutil import: ok")
print("decord import: ok")
print("einops import: ok")
print("triton import: ok")
print("sam3 import: ok")
PY

echo
echo "Done."
echo "Activate with:"
echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "conda activate ${ENV_PATH}"
echo
echo "Recommended runtime env vars:"
echo "unset OMP_NUM_THREADS"
echo "export OMP_NUM_THREADS=4"
echo "export MKL_NUM_THREADS=4"
echo "export OPENBLAS_NUM_THREADS=4"
