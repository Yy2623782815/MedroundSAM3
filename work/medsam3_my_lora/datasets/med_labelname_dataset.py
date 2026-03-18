# filename: /root/autodl-tmp/work/medsam3_my_lora/datasets/med_labelname_dataset.py
from typing import Any, Dict, List
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.med_data_utils import load_gt_mask_from_npz


class MedLabelNameDataset(Dataset):
    """Label-name prompt segmentation dataset for MedSAM3 LoRA stage-1."""

    def __init__(self, index_jsonl: str, image_size: int = 1008, normalize: bool = True):
        self.index_jsonl = index_jsonl
        self.image_size = image_size
        self.normalize = normalize
        self.records = self._load_jsonl(index_jsonl)

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        if self.normalize:
            img = img / 255.0
        return np.transpose(img, (2, 0, 1))

    def _load_mask(self, label_npz_path: str, channel_idx: int) -> np.ndarray:
        mask = load_gt_mask_from_npz(label_npz_path, channel_idx)
        mask = cv2.resize(mask.astype(np.uint8), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image = self._load_image(rec["image_path"])
        mask = self._load_mask(rec["label_npz_path"], rec["channel_idx"])
        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
            "prompt_text": rec["prompt_text"],
            "label_name": rec["label_name"],
            "channel_idx": int(rec["channel_idx"]),
            "dataset": rec["dataset"],
            "case_id": rec.get("case_id", ""),
            "sample_id": rec.get("sample_id", f"{rec['dataset']}::{rec.get('image_rel', '')}::{rec['channel_idx']}"),
            "image_path": rec["image_path"],
            "label_npz_path": rec["label_npz_path"],
            "question_id": rec.get("question_id", ""),
        }
