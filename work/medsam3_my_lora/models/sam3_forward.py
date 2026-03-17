# filename: /root/autodl-tmp/work/medsam3_my_lora/models/sam3_forward.py
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import torch
import torch.nn.functional as F


def _ensure_sam3_importable(sam3_repo_root: str) -> None:
    repo_root = str(Path(sam3_repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _data_misc_types(sam3_repo_root: str):
    _ensure_sam3_importable(sam3_repo_root)
    from sam3.model.data_misc import BatchedDatapoint, BatchedFindTarget, BatchedInferenceMetadata, FindStage

    return BatchedDatapoint, BatchedFindTarget, BatchedInferenceMetadata, FindStage


def _build_text_batch_and_text_ids(prompt_texts: List[str], device: torch.device) -> Tuple[List[str], torch.Tensor]:
    text_batch: List[str] = []
    text_ids: List[int] = []
    for txt in prompt_texts:
        if txt not in text_batch:
            text_batch.append(txt)
        text_ids.append(text_batch.index(txt))
    return text_batch, torch.as_tensor(text_ids, dtype=torch.long, device=device)


def build_minimal_batched_datapoint(images: torch.Tensor, prompt_texts: List[str], target_masks: torch.Tensor, sam3_repo_root: str):
    BatchedDatapoint, BatchedFindTarget, BatchedInferenceMetadata, FindStage = _data_misc_types(sam3_repo_root)

    device = images.device
    b, _, h, w = images.shape
    find_text_batch, text_ids = _build_text_batch_and_text_ids(prompt_texts, device)
    img_ids = torch.arange(b, device=device, dtype=torch.long)

    find_input = FindStage(
        img_ids=img_ids,
        text_ids=text_ids,
        input_boxes=torch.zeros((0, b, 4), device=device, dtype=torch.float32),
        input_boxes_mask=torch.zeros((b, 0), device=device, dtype=torch.bool),
        input_boxes_label=torch.zeros((0, b), device=device, dtype=torch.long),
        input_points=torch.zeros((0, b, 2), device=device, dtype=torch.float32),
        input_points_mask=torch.zeros((b, 0), device=device, dtype=torch.bool),
        object_ids=[[0] for _ in range(b)],
    )

    find_target = BatchedFindTarget(
        num_boxes=torch.zeros((b,), device=device, dtype=torch.long),
        boxes=torch.zeros((0, 4), device=device, dtype=torch.float32),
        boxes_padded=torch.zeros((b, 0, 4), device=device, dtype=torch.float32),
        repeated_boxes=torch.zeros((0, 4), device=device, dtype=torch.float32),
        segments=torch.zeros((0, h, w), device=device, dtype=torch.bool),
        semantic_segments=target_masks.bool().unsqueeze(1),
        is_valid_segment=torch.zeros((0,), device=device, dtype=torch.bool),
        is_exhaustive=torch.ones((b,), device=device, dtype=torch.bool),
        object_ids=torch.zeros((0,), device=device, dtype=torch.long),
        object_ids_padded=torch.full((b, 0), -1, device=device, dtype=torch.long),
    )

    metadata = BatchedInferenceMetadata(
        coco_image_id=torch.full((b,), -1, device=device, dtype=torch.long),
        original_image_id=torch.full((b,), -1, device=device, dtype=torch.long),
        original_category_id=torch.full((b,), -1, device=device, dtype=torch.int32),
        original_size=torch.tensor([[h, w] for _ in range(b)], device=device, dtype=torch.long),
        object_id=torch.full((b,), -1, device=device, dtype=torch.long),
        frame_index=torch.zeros((b,), device=device, dtype=torch.long),
        is_conditioning_only=[False for _ in range(b)],
    )

    return BatchedDatapoint(
        img_batch=images,
        find_text_batch=find_text_batch,
        find_inputs=[find_input],
        find_targets=[find_target],
        find_metadatas=[metadata],
        raw_images=None,
    )


def sam3_train_forward(model: torch.nn.Module, batch: Dict[str, Any], sam3_repo_root: str) -> torch.Tensor:
    masks = batch["masks"]
    batched_input = build_minimal_batched_datapoint(batch["images"], batch["prompt_texts"], masks, sam3_repo_root)
    outputs = model(batched_input)
    pred = outputs[-1]["pred_masks"] if isinstance(outputs, list) else outputs["pred_masks"]
    if pred.ndim == 4 and pred.shape[1] == masks.shape[0]:
        idx = torch.arange(masks.shape[0], device=pred.device)
        pred = pred[idx, idx].unsqueeze(1)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if pred.shape[-2:] != masks.shape[-2:]:
        pred = F.interpolate(pred, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return pred
