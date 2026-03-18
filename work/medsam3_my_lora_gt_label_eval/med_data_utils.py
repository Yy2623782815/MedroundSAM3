import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import pycocotools.mask as mask_utils


def load_dataset_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_json_path(data_root: str, dataset_name: str) -> str:
    return os.path.join(data_root, dataset_name, f"MultiEN_{dataset_name}.json")


def resolve_abs_path(dataset_root: str, relative_path: str) -> str:
    return os.path.join(dataset_root, relative_path)


def build_label_name_to_index(dataset_json: Dict[str, Any]) -> Dict[str, int]:
    raw_labels = dataset_json["labels"]
    label_name_to_idx = {}
    for idx_str, label_name in raw_labels.items():
        label_name_to_idx[label_name] = int(idx_str)
    return label_name_to_idx


def normalize_gt_array_to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        raise ValueError(f"Unsupported 4D label shape: {arr.shape}")

    raise ValueError(f"Unsupported label array shape: {arr.shape}")


def _infer_hw_from_npz_filename(npz_path: str):
    base = os.path.basename(npz_path)
    m = re.search(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)\.npz$", base)
    if m is None:
        raise ValueError(f"Cannot infer (C,H,W,1) from filename: {npz_path}")
    c = int(m.group(1))
    h = int(m.group(2))
    w = int(m.group(3))
    trailing = int(m.group(4))
    return c, h, w, trailing


def load_gt_mask_from_npz(npz_path: str, channel_idx: int) -> np.ndarray:
    data = np.load(npz_path)

    if "label" in data.files:
        arr = normalize_gt_array_to_chw(data["label"])
        if channel_idx < 0 or channel_idx >= arr.shape[0]:
            raise IndexError(
                f"channel_idx={channel_idx} out of range for dense label shape {arr.shape} in {npz_path}"
            )
        return (arr[channel_idx] > 0).astype(np.uint8)

    sparse_keys = {"indices", "indptr", "format", "shape", "data"}
    if sparse_keys.issubset(set(data.files)):
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise ImportError("scipy is required to read sparse CSR-style label npz files.") from e

        indices = data["indices"]
        indptr = data["indptr"]
        sparse_shape = tuple(data["shape"].tolist())
        values = data["data"]

        if len(sparse_shape) != 2:
            raise ValueError(f"Expected sparse 2D shape (C, H*W), got {sparse_shape} from {npz_path}")

        num_channels, flat_hw = sparse_shape
        sparse_row_idx = channel_idx - 1

        if sparse_row_idx < 0 or sparse_row_idx >= num_channels:
            raise IndexError(
                f"channel_idx={channel_idx} maps to sparse_row_idx={sparse_row_idx}, "
                f"which is out of range for sparse shape {sparse_shape} in {npz_path}"
            )

        csr = csr_matrix((values, indices, indptr), shape=sparse_shape)

        c_from_name, h, w, trailing = _infer_hw_from_npz_filename(npz_path)
        if trailing != 1:
            raise ValueError(f"Expected trailing dim 1 in filename, got {trailing} for {npz_path}")
        if c_from_name not in (num_channels, num_channels + 1):
            raise ValueError(
                f"Channel mismatch between sparse shape {sparse_shape} and filename {os.path.basename(npz_path)}"
            )
        if h * w != flat_hw:
            raise ValueError(
                f"Flattened HW mismatch: sparse shape={sparse_shape}, filename gives H={h}, W={w}"
            )

        row = csr.getrow(sparse_row_idx).toarray().reshape(h, w)
        return (row > 0).astype(np.uint8)

    if len(data.files) == 1:
        arr = normalize_gt_array_to_chw(data[data.files[0]])
        if channel_idx < 0 or channel_idx >= arr.shape[0]:
            raise IndexError(
                f"channel_idx={channel_idx} out of range for dense label shape {arr.shape} in {npz_path}"
            )
        return (arr[channel_idx] > 0).astype(np.uint8)

    raise KeyError(f"Unsupported npz format for {npz_path}, available keys: {data.files}")


def decode_pred_rle_mask(rle_obj, height: int, width: int) -> np.ndarray:
    if isinstance(rle_obj, str):
        rle = {"size": [height, width], "counts": rle_obj}
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (mask > 0).astype(np.uint8)

    if isinstance(rle_obj, dict):
        rle = dict(rle_obj)
        if "size" not in rle:
            rle["size"] = [height, width]
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (mask > 0).astype(np.uint8)

    raise TypeError(f"Unsupported RLE type: {type(rle_obj)}")


def union_pred_masks(pred_masks, height: int, width: int) -> np.ndarray:
    if pred_masks is None:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(pred_masks, np.ndarray):
        if pred_masks.ndim == 2:
            return (pred_masks > 0).astype(np.uint8)
        if pred_masks.ndim == 3:
            union_mask = np.zeros((height, width), dtype=np.uint8)
            for i in range(pred_masks.shape[0]):
                union_mask = np.logical_or(union_mask, pred_masks[i] > 0)
            return union_mask.astype(np.uint8)
        raise ValueError(f"Unsupported ndarray pred_masks shape: {pred_masks.shape}")

    if isinstance(pred_masks, list):
        if len(pred_masks) == 0:
            return np.zeros((height, width), dtype=np.uint8)

        first = pred_masks[0]
        if isinstance(first, (str, dict)):
            union_mask = np.zeros((height, width), dtype=np.uint8)
            for rle_obj in pred_masks:
                mask = decode_pred_rle_mask(rle_obj, height, width)
                union_mask = np.logical_or(union_mask, mask)
            return union_mask.astype(np.uint8)

        if isinstance(first, np.ndarray):
            union_mask = np.zeros((height, width), dtype=np.uint8)
            for mask in pred_masks:
                union_mask = np.logical_or(union_mask, mask > 0)
            return union_mask.astype(np.uint8)

    raise TypeError(f"Unsupported pred_masks type: {type(pred_masks)}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_split_items(dataset_json: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    if split not in dataset_json:
        raise KeyError(
            f"Split '{split}' not found in dataset json. Available keys: {list(dataset_json.keys())}"
        )
    return dataset_json[split]


def case_identifier_from_item(item: Dict[str, Any]) -> str:
    image_rel = item["image"]
    return os.path.splitext(os.path.basename(image_rel))[0]
