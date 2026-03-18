import numpy as np


def _to_bool_mask(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool)


def dice_score(pred: np.ndarray, gt: np.ndarray, empty_value: float = 1.0) -> float:
    pred_b = _to_bool_mask(pred)
    gt_b = _to_bool_mask(gt)

    pred_sum = pred_b.sum()
    gt_sum = gt_b.sum()

    if pred_sum == 0 and gt_sum == 0:
        return float(empty_value)

    intersection = np.logical_and(pred_b, gt_b).sum()
    denom = pred_sum + gt_sum
    if denom == 0:
        return float(empty_value)

    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray, empty_value: float = 1.0) -> float:
    pred_b = _to_bool_mask(pred)
    gt_b = _to_bool_mask(gt)

    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return float(empty_value)

    intersection = np.logical_and(pred_b, gt_b).sum()
    return float(intersection / union)
