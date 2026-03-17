# filename: /root/autodl-tmp/work/medsam3_my_lora/models/sam3_forward.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def _unwrap_last_stage(outputs: Any) -> Dict[str, Any]:
    if isinstance(outputs, list):
        if len(outputs) == 0:
            raise ValueError("model outputs is empty list")
        stage = outputs[-1]
    else:
        stage = outputs

    if not isinstance(stage, dict):
        raise TypeError(f"Expected dict stage output, got {type(stage)}")
    return stage


def _compute_query_scores(pred_masks: torch.Tensor, pred_logits: Optional[torch.Tensor], mode: str) -> torch.Tensor:
    b, q, _, _ = pred_masks.shape
    if mode == "mask_mean":
        return pred_masks.flatten(2).mean(dim=-1)

    if mode == "logits_max":
        if pred_logits is None:
            raise ValueError("query_select.mode='logits_max' requires pred_logits in model output")
        if pred_logits.ndim != 3 or pred_logits.shape[0] != b or pred_logits.shape[1] != q:
            raise ValueError(
                "pred_logits shape mismatch, expected [B,Q,C] aligned with pred_masks, "
                f"got pred_masks={tuple(pred_masks.shape)} pred_logits={tuple(pred_logits.shape)}"
            )
        return torch.sigmoid(pred_logits).max(dim=-1).values

    raise ValueError(f"Unsupported query_select.mode: {mode}")


def _reduce_topk_masks(topk_masks: torch.Tensor, reduce: str) -> torch.Tensor:
    # topk_masks: [B,K,H,W]
    if reduce == "mean":
        return topk_masks.mean(dim=1, keepdim=True)
    if reduce == "max":
        return topk_masks.max(dim=1, keepdim=True).values
    raise ValueError(f"Unsupported query_select.reduce: {reduce}")


def _select_query_mask(
    pred_masks: torch.Tensor,
    pred_logits: Optional[torch.Tensor],
    query_select_cfg: Optional[Dict[str, Any]],
) -> torch.Tensor:
    """
    将 [B,Q,H,W] 压缩到 [B,1,H,W]，策略可配置：
    - mode: logits_max | mask_mean
    - topk: >=1
    - reduce: mean | max
    """
    if pred_masks.ndim != 4:
        raise ValueError(f"pred_masks must be [B,Q,H,W], got {tuple(pred_masks.shape)}")

    b, q, h, w = pred_masks.shape
    if q == 1:
        return pred_masks

    cfg = query_select_cfg or {}
    mode = str(cfg.get("mode", "logits_max"))
    topk = int(cfg.get("topk", 1))
    reduce = str(cfg.get("reduce", "mean"))

    if topk < 1:
        raise ValueError(f"query_select.topk must be >=1, got {topk}")
    topk = min(topk, q)

    query_scores = _compute_query_scores(pred_masks, pred_logits, mode=mode)

    if topk == 1:
        best_q = query_scores.argmax(dim=1)
        batch_idx = torch.arange(b, device=pred_masks.device)
        return pred_masks[batch_idx, best_q].unsqueeze(1)

    topk_idx = query_scores.topk(k=topk, dim=1, largest=True, sorted=False).indices  # [B,K]
    gather_idx = topk_idx[:, :, None, None].expand(-1, -1, h, w)
    topk_masks = torch.gather(pred_masks, dim=1, index=gather_idx)  # [B,K,H,W]
    return _reduce_topk_masks(topk_masks, reduce=reduce)


def sam3_train_forward(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    sam3_repo_root: str,
    query_select_cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    masks = batch["masks"]
    batched_input = build_minimal_batched_datapoint(batch["images"], batch["prompt_texts"], masks, sam3_repo_root)

    stage = _unwrap_last_stage(model(batched_input))
    if "pred_masks" not in stage:
        raise KeyError(f"model output has no pred_masks, keys={list(stage.keys())}")

    pred_masks = stage["pred_masks"]
    pred_logits = stage.get("pred_logits", None)

    if pred_masks.ndim == 3:
        pred = pred_masks.unsqueeze(1)
    elif pred_masks.ndim == 4:
        pred = _select_query_mask(pred_masks, pred_logits, query_select_cfg=query_select_cfg)
    else:
        raise ValueError(f"Unsupported pred_masks dim: {tuple(pred_masks.shape)}")

    if pred.shape[-2:] != masks.shape[-2:]:
        pred = F.interpolate(pred, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return pred
