from typing import Any, Dict, List

import torch


def med_labelname_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将 MedLabelNameDataset 的样本组织为一个 batch。

    输入单样本字段（当前约定）:
        {
            "image": FloatTensor [3,H,W],
            "mask": FloatTensor [H,W],
            "prompt_text": str,
            "label_name": str,
            "channel_idx": int,
            "dataset": str,
            "image_path": str,
            "label_npz_path": str,
            "question_id": str,
        }

    输出 batch 字段:
        {
            "images": FloatTensor [B,3,H,W],
            "masks": FloatTensor [B,H,W],
            "prompt_texts": List[str],
            "label_names": List[str],
            "channel_idxs": LongTensor [B],
            "datasets": List[str],
            "image_paths": List[str],
            "label_npz_paths": List[str],
            "question_ids": List[str],
        }
    """
    if len(batch) == 0:
        raise ValueError("Empty batch is not allowed.")

    images = []
    masks = []
    prompt_texts = []
    label_names = []
    channel_idxs = []
    datasets = []
    image_paths = []
    label_npz_paths = []
    question_ids = []

    for sample in batch:
        image = sample["image"]
        mask = sample["mask"]

        if not torch.is_tensor(image):
            raise TypeError(f"sample['image'] must be a torch.Tensor, got {type(image)}")
        if not torch.is_tensor(mask):
            raise TypeError(f"sample['mask'] must be a torch.Tensor, got {type(mask)}")

        if image.ndim != 3:
            raise ValueError(f"sample['image'] must have shape [3,H,W], got {tuple(image.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"sample['mask'] must have shape [H,W], got {tuple(mask.shape)}")

        images.append(image)
        masks.append(mask)
        prompt_texts.append(sample["prompt_text"])
        label_names.append(sample["label_name"])
        channel_idxs.append(int(sample["channel_idx"]))
        datasets.append(sample["dataset"])
        image_paths.append(sample["image_path"])
        label_npz_paths.append(sample["label_npz_path"])
        question_ids.append(sample.get("question_id", ""))

    images = torch.stack(images, dim=0).contiguous()   # [B,3,H,W]
    masks = torch.stack(masks, dim=0).contiguous()     # [B,H,W]
    channel_idxs = torch.tensor(channel_idxs, dtype=torch.long)

    return {
        "images": images,
        "masks": masks,
        "prompt_texts": prompt_texts,
        "label_names": label_names,
        "channel_idxs": channel_idxs,
        "datasets": datasets,
        "image_paths": image_paths,
        "label_npz_paths": label_npz_paths,
        "question_ids": question_ids,
    }