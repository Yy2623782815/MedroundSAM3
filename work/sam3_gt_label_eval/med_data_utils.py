# filename: /root/autodl-tmp/work/sam3_med_agent_eval/med_data_utils.py
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
    """
    Convert top-level labels mapping like:
        {"0":"background","1":"spleen","8":"aorta"}
    into:
        {"background":0,"spleen":1,"aorta":8}
    """
    raw_labels = dataset_json["labels"]
    label_name_to_idx = {}
    for idx_str, label_name in raw_labels.items():
        label_name_to_idx[label_name] = int(idx_str)
    return label_name_to_idx


def normalize_gt_array_to_chw(arr: np.ndarray) -> np.ndarray:
    """
    Normalize GT label array to shape (C, H, W).

    Supported dense formats:
    - (C, H, W)
    - (C, H, W, 1)
    """
    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        raise ValueError(f"Unsupported 4D label shape: {arr.shape}")

    raise ValueError(f"Unsupported label array shape: {arr.shape}")


def _infer_hw_from_npz_filename(npz_path: str):
    """
    Infer (C, H, W, trailing) from filename pattern like:
        x___amos_0551_0.(15, 468, 576, 1).npz
    """
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
    """
    Load GT mask from npz and return HxW uint8 mask in {0,1}.

    Supported formats:
    1. Dense npz with key "label"
    2. Dense npz with a single array key
    3. Sparse CSR-style npz with keys:
       ['indices', 'indptr', 'format', 'shape', 'data']
       representing a matrix of shape (C, H*W)

    Notes:
    - channel_idx is obtained by reverse-indexing dataset_json["labels"].
    - For sparse CSR storage, each row corresponds to one label channel.
    """
    data = np.load(npz_path)

    # Case 1: dense saved under key "label"
    if "label" in data.files:
        arr = data["label"]
        arr = normalize_gt_array_to_chw(arr)

        if channel_idx < 0 or channel_idx >= arr.shape[0]:
            raise IndexError(
                f"channel_idx={channel_idx} out of range for dense label shape {arr.shape} in {npz_path}"
            )

        mask = arr[channel_idx]
        return (mask > 0).astype(np.uint8)

    # Case 2: sparse CSR-style storage
    sparse_keys = {"indices", "indptr", "format", "shape", "data"}
    if sparse_keys.issubset(set(data.files)):
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise ImportError(
                "scipy is required to read sparse CSR-style label npz files."
            ) from e

        indices = data["indices"]
        indptr = data["indptr"]
        sparse_shape = tuple(data["shape"].tolist())
        values = data["data"]

        if len(sparse_shape) != 2:
            raise ValueError(
                f"Expected sparse 2D shape (C, H*W), got {sparse_shape} from {npz_path}"
            )

        num_channels, flat_hw = sparse_shape
        
        # IMPORTANT:
        # top-level dataset_json["labels"] includes background at id 0,
        # but this sparse GT file stores only foreground channels 1..N.
        # Therefore sparse row index = channel_idx - 1.
        sparse_row_idx = channel_idx - 1
        
        if sparse_row_idx < 0 or sparse_row_idx >= num_channels:
            raise IndexError(
                f"channel_idx={channel_idx} maps to sparse_row_idx={sparse_row_idx}, "
                f"which is out of range for sparse shape {sparse_shape} in {npz_path}"
            )
        
        csr = csr_matrix((values, indices, indptr), shape=sparse_shape)
        
        c_from_name, h, w, trailing = _infer_hw_from_npz_filename(npz_path)
        
        if trailing != 1:
            raise ValueError(
                f"Expected trailing dim 1 in filename, got {trailing} for {npz_path}"
            )
        
        # filename channel count includes background, sparse matrix usually excludes background
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

    # Case 3: dense npz with exactly one array key
    if len(data.files) == 1:
        only_key = data.files[0]
        arr = data[only_key]
        arr = normalize_gt_array_to_chw(arr)

        if channel_idx < 0 or channel_idx >= arr.shape[0]:
            raise IndexError(
                f"channel_idx={channel_idx} out of range for dense label shape {arr.shape} in {npz_path}"
            )

        mask = arr[channel_idx]
        return (mask > 0).astype(np.uint8)

    raise KeyError(
        f"Unsupported npz format for {npz_path}, available keys: {data.files}"
    )


def decode_pred_rle_mask(rle_string: str, height: int, width: int) -> np.ndarray:
    """
    Decode SAM3 pred mask RLE string to HxW uint8 mask.
    """
    rle = {"size": [height, width], "counts": rle_string}
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def union_pred_masks(pred_masks: List[str], height: int, width: int) -> np.ndarray:
    """
    Union multiple RLE-string masks into one HxW uint8 mask.
    """
    if len(pred_masks) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    union_mask = np.zeros((height, width), dtype=np.uint8)
    for rle_string in pred_masks:
        mask = decode_pred_rle_mask(rle_string, height, width)
        union_mask = np.logical_or(union_mask, mask)

    return union_mask.astype(np.uint8)


def build_history_text_from_questions(questions: List[Dict[str, Any]], turn_idx: int) -> str:
    """
    Build FULL history text block for current turn_idx using previous turns only.

    A is fixed as:
        Target {gt_label_name} has been successfully segmented.
    """
    if turn_idx <= 0:
        return ""

    lines = []
    for i in range(turn_idx):
        q = questions[i]["question"].strip()
        gt_label_name = questions[i]["label"].strip()
        a = f"Target {gt_label_name} has been successfully segmented."
        lines.append(f"Turn {i + 1}:")
        lines.append(f"Q: {q}")
        lines.append(f"A: {a}")
        lines.append("")

    return "\n".join(lines).strip()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_split_items(dataset_json: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    if split not in dataset_json:
        raise KeyError(
            f"Split '{split}' not found in dataset json. Available keys: {list(dataset_json.keys())}"
        )
    return dataset_json[split]


def case_identifier_from_item(item: Dict[str, Any]) -> str:
    """
    Use image basename as case id.
    Example:
        image/x/amos_0551_0.png -> amos_0551_0
    """
    image_rel = item["image"]
    return os.path.splitext(os.path.basename(image_rel))[0]