# filename: /root/autodl-tmp/work/medsam3_my_lora/utils/med_data_utils.py
import json
import os
import re
from typing import Any, Dict

import numpy as np


def load_dataset_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_name_to_index(dataset_json: Dict[str, Any]) -> Dict[str, int]:
    return {label_name: int(idx_str) for idx_str, label_name in dataset_json["labels"].items()}


def normalize_gt_array_to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def _infer_hw_from_npz_filename(npz_path: str) -> tuple[int, int, int, int]:
    base = os.path.basename(npz_path)
    m = re.search(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)\.npz$", base)
    if m is None:
        raise ValueError(f"Cannot infer shape from filename: {npz_path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def load_gt_mask_from_npz(npz_path: str, channel_idx: int) -> np.ndarray:
    data = np.load(npz_path)

    if "label" in data.files:
        arr = normalize_gt_array_to_chw(data["label"])
        if channel_idx < 0 or channel_idx >= arr.shape[0]:
            raise IndexError(f"channel_idx={channel_idx} out of range: {arr.shape}")
        return (arr[channel_idx] > 0).astype(np.uint8)

    sparse_keys = {"indices", "indptr", "format", "shape", "data"}
    if sparse_keys.issubset(set(data.files)):
        from scipy.sparse import csr_matrix

        indices = data["indices"]
        indptr = data["indptr"]
        values = data["data"]
        sparse_shape = tuple(data["shape"].tolist())
        num_channels, flat_hw = sparse_shape

        sparse_row_idx = channel_idx - 1
        if sparse_row_idx < 0 or sparse_row_idx >= num_channels:
            raise IndexError(f"channel_idx={channel_idx} -> sparse_row={sparse_row_idx} invalid")

        _, h, w, trailing = _infer_hw_from_npz_filename(npz_path)
        if trailing != 1 or h * w != flat_hw:
            raise ValueError(f"Sparse label shape mismatch for {npz_path}")

        csr = csr_matrix((values, indices, indptr), shape=sparse_shape)
        row = csr.getrow(sparse_row_idx).toarray().reshape(h, w)
        return (row > 0).astype(np.uint8)

    if len(data.files) == 1:
        arr = normalize_gt_array_to_chw(data[data.files[0]])
        if channel_idx < 0 or channel_idx >= arr.shape[0]:
            raise IndexError(f"channel_idx={channel_idx} out of range: {arr.shape}")
        return (arr[channel_idx] > 0).astype(np.uint8)

    raise KeyError(f"Unsupported npz format {npz_path}: keys={data.files}")
