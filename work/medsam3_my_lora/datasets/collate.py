# filename: /root/autodl-tmp/work/medsam3_my_lora/datasets/collate.py
from typing import Any, Dict, List

import torch


def med_labelname_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([x["image"] for x in batch], dim=0).contiguous()
    masks = torch.stack([x["mask"] for x in batch], dim=0).contiguous()
    return {
        "images": images,
        "masks": masks,
        "prompt_texts": [x["prompt_text"] for x in batch],
        "label_names": [x["label_name"] for x in batch],
        "channel_idxs": torch.tensor([x["channel_idx"] for x in batch], dtype=torch.long),
        "datasets": [x["dataset"] for x in batch],
        "case_ids": [x["case_id"] for x in batch],
        "sample_ids": [x["sample_id"] for x in batch],
        "image_paths": [x["image_path"] for x in batch],
        "label_npz_paths": [x["label_npz_path"] for x in batch],
        "question_ids": [x.get("question_id", "") for x in batch],
    }
