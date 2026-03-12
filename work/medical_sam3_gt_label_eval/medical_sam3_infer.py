# filename: /root/autodl-tmp/work/medical_sam3_gt_label_eval/medical_sam3_infer.py
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.model.sam3_image_processor import Sam3Processor


def _enable_fast_inference(device: str = "cuda"):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Robustly extract a state_dict from different checkpoint layouts.
    """
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "network", "net", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove common prefixes such as:
      - detector.
      - module.
      - model.
    """
    normalized = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("detector.", "module.", "model."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        normalized[new_k] = v
    return normalized


def load_custom_checkpoint(model, checkpoint_path: str, map_location: str = "cpu"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(ckpt)
    state_dict = _normalize_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[Medical-SAM3] Loaded checkpoint: {checkpoint_path}")
    print(f"[Medical-SAM3] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if len(missing) > 0:
        print("[Medical-SAM3] first missing keys:", missing[:20])
    if len(unexpected) > 0:
        print("[Medical-SAM3] first unexpected keys:", unexpected[:20])

    return model


def build_medical_sam3_processor(
    sam3_repo_root: str,
    checkpoint_path: str,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
):
    """
    Build vanilla SAM3 architecture, then load Medical-SAM3 fine-tuned checkpoint.
    This keeps the inference/output interface aligned with your original sam3_gt_label_eval.
    """
    _enable_fast_inference(device=device)

    # Ensure sam3 repo root is importable if caller forgot PYTHONPATH
    if sam3_repo_root not in sys.path:
        sys.path.insert(0, sam3_repo_root)

    bpe_path = os.path.join(
        sam3_repo_root,
        "sam3",
        "assets",
        "bpe_simple_vocab_16e6.txt.gz",
    )
    if not os.path.exists(bpe_path):
        raise FileNotFoundError(f"BPE file not found: {bpe_path}")

    # Important: build empty/base SAM3 model first, then load custom checkpoint.
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=None,
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    model = load_custom_checkpoint(model, checkpoint_path=checkpoint_path, map_location="cpu")
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    return processor


def _binary_mask_to_rle_counts(mask: np.ndarray) -> str:
    """
    Convert HxW binary mask to COCO RLE counts string.
    """
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts


def _ensure_mask_array(mask_like, target_h: int, target_w: int) -> np.ndarray:
    """
    Normalize a predicted mask into HxW uint8 array.
    """
    if isinstance(mask_like, torch.Tensor):
        arr = mask_like.detach().float().cpu().numpy()
    else:
        arr = np.asarray(mask_like)

    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape={arr.shape}")

    if arr.shape != (target_h, target_w):
        raise ValueError(
            f"Pred mask shape mismatch after normalization: got {arr.shape}, "
            f"expected {(target_h, target_w)}"
        )

    # threshold at 0 if logits/prob-like, else >0
    arr = (arr > 0).astype(np.uint8)
    return arr


def medical_sam3_text_inference(processor: Sam3Processor, image_path: str, text_prompt: str):
    """
    Keep output schema compatible with original sam3_text_inference:
      {
        "original_image_path": ...,
        "orig_img_h": ...,
        "orig_img_w": ...,
        "pred_boxes": [...],
        "pred_scores": [...],
        "pred_masks": [rle_counts_str, ...],
      }
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(
        state=inference_state,
        prompt=text_prompt,
    )

    boxes_xyxy = inference_state.get("boxes", None)
    scores = inference_state.get("scores", None)
    masks = inference_state.get("masks", None)

    if masks is None:
        return {
            "original_image_path": image_path,
            "orig_img_h": orig_h,
            "orig_img_w": orig_w,
            "pred_boxes": [],
            "pred_scores": [],
            "pred_masks": [],
        }

    if isinstance(masks, torch.Tensor):
        masks_t = masks.detach()
    else:
        masks_t = torch.as_tensor(masks)

    # Expected common shapes:
    #   [N, 1, H, W]
    #   [N, H, W]
    #   [H, W]
    if masks_t.ndim == 4 and masks_t.shape[1] == 1:
        masks_t = masks_t.squeeze(1)
    elif masks_t.ndim == 2:
        masks_t = masks_t.unsqueeze(0)

    if masks_t.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {tuple(masks_t.shape)}")

    num_masks = int(masks_t.shape[0])

    if num_masks == 0:
        return {
            "original_image_path": image_path,
            "orig_img_h": orig_h,
            "orig_img_w": orig_w,
            "pred_boxes": [],
            "pred_scores": [],
            "pred_masks": [],
        }

    # boxes
    pred_boxes_xywh: List[List[float]] = []
    if boxes_xyxy is not None and isinstance(boxes_xyxy, torch.Tensor) and boxes_xyxy.numel() > 0:
        pred_boxes_xyxy = torch.stack(
            [
                boxes_xyxy[:, 0] / orig_w,
                boxes_xyxy[:, 1] / orig_h,
                boxes_xyxy[:, 2] / orig_w,
                boxes_xyxy[:, 3] / orig_h,
            ],
            dim=-1,
        )
        pred_boxes_xywh = box_xyxy_to_xywh(pred_boxes_xyxy).tolist()
    else:
        pred_boxes_xywh = [[0.0, 0.0, 1.0, 1.0] for _ in range(num_masks)]

    # scores
    if scores is None:
        pred_scores = [1.0 for _ in range(num_masks)]
    elif isinstance(scores, torch.Tensor):
        pred_scores = scores.detach().float().cpu().tolist()
    else:
        pred_scores = list(scores)

    if len(pred_scores) != num_masks:
        if len(pred_scores) == 0:
            pred_scores = [1.0 for _ in range(num_masks)]
        else:
            raise ValueError(
                f"Mismatch between num_masks={num_masks} and len(scores)={len(pred_scores)}"
            )

    pred_masks_rle = []
    for i in range(num_masks):
        mask_np = _ensure_mask_array(masks_t[i], orig_h, orig_w)
        rle_counts = _binary_mask_to_rle_counts(mask_np)
        pred_masks_rle.append(rle_counts)

    score_indices = sorted(
        range(len(pred_scores)),
        key=lambda i: pred_scores[i],
        reverse=True,
    )
    pred_scores = [float(pred_scores[i]) for i in score_indices]
    pred_boxes_xywh = [pred_boxes_xywh[i] for i in score_indices]
    pred_masks_rle = [pred_masks_rle[i] for i in score_indices]

    valid_masks = []
    valid_boxes = []
    valid_scores = []
    for i, rle in enumerate(pred_masks_rle):
        if isinstance(rle, str) and len(rle) > 4:
            valid_masks.append(rle)
            valid_boxes.append(pred_boxes_xywh[i])
            valid_scores.append(pred_scores[i])

    return {
        "original_image_path": image_path,
        "orig_img_h": orig_h,
        "orig_img_w": orig_w,
        "pred_boxes": valid_boxes,
        "pred_scores": valid_scores,
        "pred_masks": valid_masks,
    }