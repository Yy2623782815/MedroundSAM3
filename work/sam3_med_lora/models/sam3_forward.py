import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)

_DEBUG_PRINT_ONCE = True


def _build_text_batch_and_text_ids(
    prompt_texts: List[str],
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    """
    官方 collator 的逻辑是：
    - find_text_batch: 去重后的文本列表
    - text_ids: 每个 query 对应到去重文本列表中的索引
    """
    text_batch: List[str] = []
    text_ids: List[int] = []
    for txt in prompt_texts:
        if txt not in text_batch:
            text_batch.append(txt)
        text_ids.append(text_batch.index(txt))
    return text_batch, torch.as_tensor(text_ids, dtype=torch.long, device=device)


def build_minimal_batched_datapoint(
    images: torch.Tensor,
    prompt_texts: List[str],
    target_masks: torch.Tensor,
) -> BatchedDatapoint:
    """
    构造一个最小可用的官方 BatchedDatapoint。

    当前任务：单图 + 单标签名文本 prompt -> 单个二值 mask
    因此这里按“每张图一个 query、单 stage”来组织。

    参数:
        images: [B,3,H,W]
        prompt_texts: List[str], len=B
        target_masks: [B,H,W]

    返回:
        BatchedDatapoint
    """
    if images.ndim != 4:
        raise ValueError(f"images must be [B,3,H,W], got {tuple(images.shape)}")
    if target_masks.ndim != 3:
        raise ValueError(f"target_masks must be [B,H,W], got {tuple(target_masks.shape)}")
    if images.shape[0] != len(prompt_texts) or images.shape[0] != target_masks.shape[0]:
        raise ValueError(
            f"Batch size mismatch: images={images.shape[0]}, "
            f"prompt_texts={len(prompt_texts)}, masks={target_masks.shape[0]}"
        )

    device = images.device
    b, _, h, w = images.shape

    find_text_batch, text_ids = _build_text_batch_and_text_ids(prompt_texts, device=device)

    img_ids = torch.arange(b, device=device, dtype=torch.long)

    # 当前第一版：每张图一个 query，所以 num_queries = B
    num_queries = b
    num_boxes = 0
    num_points = 0

    # 官方 Prompt 约定：
    # box embeddings / labels: sequence-first = [N_boxes, Q, ...]
    # box mask: batch-first = [Q, N_boxes]
    input_boxes = torch.zeros(
        (num_boxes, num_queries, 4),
        device=device,
        dtype=torch.float32,
    )
    input_boxes_label = torch.zeros(
        (num_boxes, num_queries),
        device=device,
        dtype=torch.long,
    )
    input_boxes_mask = torch.zeros(
        (num_queries, num_boxes),
        device=device,
        dtype=torch.bool,
    )

    # points 同理：
    # point embeddings: [N_points, Q, 2]
    # point mask:       [Q, N_points]
    input_points = torch.zeros(
        (num_points, num_queries, 2),
        device=device,
        dtype=torch.float32,
    )
    input_points_mask = torch.zeros(
        (num_queries, num_points),
        device=device,
        dtype=torch.bool,
    )

    object_ids_list = [[0] for _ in range(num_queries)]

    global _DEBUG_PRINT_ONCE
    if _DEBUG_PRINT_ONCE:
        print("[debug] img_ids:", tuple(img_ids.shape))
        print("[debug] text_ids:", tuple(text_ids.shape))
        print("[debug] input_boxes:", tuple(input_boxes.shape))
        print("[debug] input_boxes_label:", tuple(input_boxes_label.shape))
        print("[debug] input_boxes_mask:", tuple(input_boxes_mask.shape))
        print("[debug] input_points:", tuple(input_points.shape))
        print("[debug] input_points_mask:", tuple(input_points_mask.shape))
        print("[debug] num_queries / num_boxes / num_points:", num_queries, num_boxes, num_points)
        _DEBUG_PRINT_ONCE = False

    find_input = FindStage(
        img_ids=img_ids,
        text_ids=text_ids,
        input_boxes=input_boxes,
        input_boxes_mask=input_boxes_mask,
        input_boxes_label=input_boxes_label,
        input_points=input_points,
        input_points_mask=input_points_mask,
        object_ids=object_ids_list,
    )

    # find_targets 按官方字段组织；因为我们当前主要用模型输出再算自定义 loss，
    # 这里只构造最小合法形式，避免 forward 因缺字段报错。
    num_boxes = torch.zeros((b,), device=device, dtype=torch.long)
    boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
    boxes_padded = torch.zeros((b, 0, 4), device=device, dtype=torch.float32)
    repeated_boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)

    # 实例 mask 这里用 packed 形式，当前没有 box 监督，所以给空 packed tensor
    segments = torch.zeros((0, h, w), device=device, dtype=torch.bool)

    # semantic_segments 用每个 query 的目标 mask，形状 [B,H,W]
    semantic_segments = target_masks.bool().unsqueeze(1)   # [B,1,H,W]

    is_valid_segment = torch.zeros((0,), device=device, dtype=torch.bool)
    is_exhaustive = torch.ones((b,), device=device, dtype=torch.bool)

    object_ids = torch.zeros((0,), device=device, dtype=torch.long)
    object_ids_padded = torch.full((b, 0), -1, device=device, dtype=torch.long)

    find_target = BatchedFindTarget(
        num_boxes=num_boxes,
        boxes=boxes,
        boxes_padded=boxes_padded,
        repeated_boxes=repeated_boxes,
        segments=segments,
        semantic_segments=semantic_segments,
        is_valid_segment=is_valid_segment,
        is_exhaustive=is_exhaustive,
        object_ids=object_ids,
        object_ids_padded=object_ids_padded,
    )

    # 元信息字段也补齐最小合法形式
    original_size = torch.tensor([[h, w] for _ in range(b)], device=device, dtype=torch.long)

    metadata = BatchedInferenceMetadata(
        coco_image_id=torch.full((b,), -1, device=device, dtype=torch.long),
        original_image_id=torch.full((b,), -1, device=device, dtype=torch.long),
        original_category_id=torch.full((b,), -1, device=device, dtype=torch.int32),
        original_size=original_size,
        object_id=torch.full((b,), -1, device=device, dtype=torch.long),
        frame_index=torch.zeros((b,), device=device, dtype=torch.long),
        is_conditioning_only=[False for _ in range(b)],
    )

    batched = BatchedDatapoint(
        img_batch=images,
        find_text_batch=find_text_batch,
        find_inputs=[find_input],
        find_targets=[find_target],
        find_metadatas=[metadata],
        raw_images=None,
    )
    return batched


def _search_tensor_dict(obj: Any) -> List[Dict[str, torch.Tensor]]:
    """
    递归搜索包含 pred_masks / pred_logits / semantic_seg 的 dict。
    因为 Sam3Image.forward 返回的是 SAM3Output 容器，不一定是平铺 dict。
    """
    hits = []

    if isinstance(obj, dict):
        if any(k in obj for k in ["pred_masks", "pred_logits", "semantic_seg"]):
            hits.append(obj)
        for v in obj.values():
            hits.extend(_search_tensor_dict(v))
        return hits

    if isinstance(obj, (list, tuple)):
        for v in obj:
            hits.extend(_search_tensor_dict(v))
        return hits

    # SAM3Output 这类自定义容器，尽量从 __dict__ 递归搜
    if hasattr(obj, "__dict__"):
        hits.extend(_search_tensor_dict(vars(obj)))
        return hits

    return hits


def _extract_logits_from_output(outputs: Any) -> torch.Tensor:
    """
    优先取连续值空间图：
    1. pred_masks
    2. semantic_seg
    3. spatial pred_logits
    """
    candidates = _search_tensor_dict(outputs)
    if len(candidates) == 0:
        raise KeyError(
            "Cannot find any nested dict containing pred_masks / pred_logits / semantic_seg."
        )

    for out in reversed(candidates):
        if "pred_masks" in out and torch.is_tensor(out["pred_masks"]):
            x = out["pred_masks"]
            if x.ndim == 4:   # [B,Q,H,W] or [B,1,H,W]
                return x
            if x.ndim == 3:   # [B,H,W]
                return x.unsqueeze(1)

        if "semantic_seg" in out and torch.is_tensor(out["semantic_seg"]):
            x = out["semantic_seg"]
            if x.ndim == 4:
                return x
            if x.ndim == 3:
                return x.unsqueeze(1)

        if "pred_logits" in out and torch.is_tensor(out["pred_logits"]):
            x = out["pred_logits"]
            if x.ndim == 4:
                return x[:, :1]
            if x.ndim == 3:
                return x.unsqueeze(1)

    raise KeyError(
        "Found nested dicts, but none contain spatial pred_masks / semantic_seg / pred_logits."
    )


def _resize_logits_to_target_hw(pred_logits: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if pred_logits.ndim != 4:
        raise ValueError(f"pred_logits must be [B,1,H,W] or [B,C,H,W], got {tuple(pred_logits.shape)}")
    h, w = target_hw
    if pred_logits.shape[-2:] == (h, w):
        return pred_logits
    return F.interpolate(pred_logits, size=(h, w), mode="bilinear", align_corners=False)


def _select_query_aligned_logits(pred_logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    将 [B,Q,H,W] 的多 query 输出对齐成每图一个监督 mask。

    当前我们构造 query 的方式是“每张图一个 query，顺序与 batch 一致”，
    因此当 Q==B 时，应取对角线 pred[i, i] 作为第 i 张图的监督结果。
    """
    if pred_logits.ndim != 4:
        raise ValueError(f"pred_logits must be [B,Q,H,W], got {tuple(pred_logits.shape)}")

    b, q, _, _ = pred_logits.shape
    if b != batch_size:
        raise ValueError(f"Batch mismatch: pred B={b}, batch_size={batch_size}")

    if q == 1:
        return pred_logits

    if q == b:
        idx = torch.arange(b, device=pred_logits.device)
        return pred_logits[idx, idx].unsqueeze(1)

    # 如果 query 数与 batch 不一致，直接报错，避免监督 silently 错位。
    raise ValueError(
        f"Unsupported pred query dim: pred shape={tuple(pred_logits.shape)}, "
        "expect Q==1 or Q==B for current supervision alignment."
    )


# filename: /root/autodl-tmp/work/sam3_med_lora/models/sam3_forward.py

def sam3_train_forward(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: str,
) -> torch.Tensor:
    """
    第一版真实 forward：
    - 构造官方 BatchedDatapoint
    - 用可微的 eval-style forward 跑模型
    - 从输出里提取 pred_masks / semantic_seg / pred_logits
    - resize 到 GT mask 分辨率

    返回:
        pred_logits: [B,1,H,W]
    """
    images = batch["images"]
    masks = batch["masks"]
    prompt_texts = batch["prompt_texts"]

    batched_input = build_minimal_batched_datapoint(
        images=images,
        prompt_texts=prompt_texts,
        target_masks=masks,
    )

    outputs = model(batched_input)

    pred_logits = _extract_logits_from_output(outputs)
    pred_logits = _select_query_aligned_logits(pred_logits, batch_size=masks.shape[0])
    pred_logits = _resize_logits_to_target_hw(pred_logits, target_hw=masks.shape[-2:])
    return pred_logits
