# filename: /root/autodl-tmp/work/sam3_med_lora/datasets/med_labelname_dataset.py
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from med_data_utils import load_gt_mask_from_npz  # noqa: E402


class MedLabelNameDataset(Dataset):
    """
    第一阶段医学 LoRA 训练数据集：
    输入 = 图像 + label_name prompt
    监督 = 对应 HxW 二值 mask
    """

    def __init__(
        self,
        index_jsonl: str,
        image_size: int = 512,
        normalize: bool = True,
    ):
        self.index_jsonl = index_jsonl
        self.image_size = image_size
        self.normalize = normalize
        self.records = self._load_jsonl(index_jsonl)

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self):
        return len(self.records)

    def _load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(
            img,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        img = img.astype(np.float32)

        if self.normalize:
            img = img / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        return img

    def _load_mask(self, label_npz_path: str, channel_idx: int) -> np.ndarray:
        mask = load_gt_mask_from_npz(label_npz_path, channel_idx)  # HxW uint8 in {0,1}
        mask = cv2.resize(
            mask.astype(np.uint8),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        return mask.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        image = self._load_image(rec["image_path"])
        mask = self._load_mask(rec["label_npz_path"], rec["channel_idx"])

        return {
            "image": torch.from_numpy(image).float(),          # [3,H,W]
            "mask": torch.from_numpy(mask).float(),            # [H,W]
            "prompt_text": rec["prompt_text"],                 # str
            "label_name": rec["label_name"],                   # str
            "channel_idx": rec["channel_idx"],                 # int
            "dataset": rec["dataset"],                         # str
            "image_path": rec["image_path"],                   # str
            "label_npz_path": rec["label_npz_path"],           # str
            "question_id": rec.get("question_id", ""),         # str
        }