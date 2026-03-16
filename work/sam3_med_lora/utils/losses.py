from typing import Dict

import torch
import torch.nn.functional as F


def _align_pred_target_shapes(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将预测和标签整理成一致 shape，便于做 BCE / Dice。

    支持:
    - pred_logits: [B,1,H,W] or [B,H,W]
    - target_mask: [B,1,H,W] or [B,H,W]

    返回:
    - pred_logits: [B,H,W]
    - target_mask: [B,H,W]
    """
    if pred_logits.ndim == 4 and pred_logits.shape[1] == 1:
        pred_logits = pred_logits[:, 0]
    elif pred_logits.ndim != 3:
        raise ValueError(
            f"pred_logits must be [B,1,H,W] or [B,H,W], got {tuple(pred_logits.shape)}"
        )

    if target_mask.ndim == 4 and target_mask.shape[1] == 1:
        target_mask = target_mask[:, 0]
    elif target_mask.ndim != 3:
        raise ValueError(
            f"target_mask must be [B,1,H,W] or [B,H,W], got {tuple(target_mask.shape)}"
        )

    if pred_logits.shape != target_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred_logits={tuple(pred_logits.shape)}, "
            f"target_mask={tuple(target_mask.shape)}"
        )

    target_mask = target_mask.float()
    return pred_logits, target_mask


def sigmoid_bce_loss(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    pred_logits, target_mask = _align_pred_target_shapes(pred_logits, target_mask)
    return F.binary_cross_entropy_with_logits(pred_logits, target_mask)


def binary_dice_loss(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_logits, target_mask = _align_pred_target_shapes(pred_logits, target_mask)

    pred_prob = torch.sigmoid(pred_logits)

    pred_prob = pred_prob.reshape(pred_prob.shape[0], -1)
    target_mask = target_mask.reshape(target_mask.shape[0], -1)

    intersection = (pred_prob * target_mask).sum(dim=1)
    denominator = pred_prob.sum(dim=1) + target_mask.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth + eps)
    loss = 1.0 - dice
    return loss.mean()


def combined_bce_dice_loss(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if bce_weight < 0 or dice_weight < 0:
        raise ValueError("bce_weight and dice_weight must be >= 0")
    if (bce_weight + dice_weight) <= 0:
        raise ValueError("bce_weight + dice_weight must be > 0")

    bce = sigmoid_bce_loss(pred_logits, target_mask)
    dice = binary_dice_loss(pred_logits, target_mask)

    total = bce_weight * bce + dice_weight * dice

    stats = {
        "loss_total": float(total.detach().item()),
        "loss_bce": float(bce.detach().item()),
        "loss_dice": float(dice.detach().item()),
    }
    return total, stats


@torch.no_grad()
def binary_dice_score_from_logits(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_logits, target_mask = _align_pred_target_shapes(pred_logits, target_mask)

    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob >= threshold).float()

    pred_bin = pred_bin.reshape(pred_bin.shape[0], -1)
    target_mask = target_mask.reshape(target_mask.shape[0], -1)

    intersection = (pred_bin * target_mask).sum(dim=1)
    denominator = pred_bin.sum(dim=1) + target_mask.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth + eps)
    return dice.mean()


@torch.no_grad()
def binary_iou_score_from_logits(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_logits, target_mask = _align_pred_target_shapes(pred_logits, target_mask)

    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob >= threshold).float()

    pred_bin = pred_bin.reshape(pred_bin.shape[0], -1)
    target_mask = target_mask.reshape(target_mask.shape[0], -1)

    intersection = (pred_bin * target_mask).sum(dim=1)
    union = pred_bin.sum(dim=1) + target_mask.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth + eps)
    return iou.mean()