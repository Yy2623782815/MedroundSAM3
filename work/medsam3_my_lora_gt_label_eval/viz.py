import numpy as np
from PIL import Image, ImageDraw, ImageFont

from med_data_utils import union_pred_masks


_DEFAULT_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]


def _normalize_pred_masks_to_list(pred_masks):
    if pred_masks is None:
        return []

    if isinstance(pred_masks, np.ndarray):
        if pred_masks.ndim == 2:
            return [(pred_masks > 0).astype(np.uint8)]
        if pred_masks.ndim == 3:
            return [(pred_masks[i] > 0).astype(np.uint8) for i in range(pred_masks.shape[0])]
        return []

    if isinstance(pred_masks, list) and len(pred_masks) > 0 and isinstance(pred_masks[0], np.ndarray):
        return [(x > 0).astype(np.uint8) for x in pred_masks]

    return []


def _overlay_single_mask(img, mask, color, alpha=0.35):
    overlay = np.zeros_like(img)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    mask_bool = mask.astype(bool)
    img[mask_bool] = img[mask_bool] * (1.0 - alpha) + overlay[mask_bool] * alpha
    return img


def _find_mask_anchor(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x = int(np.mean(xs))
    y = int(np.mean(ys))
    return x, y


def visualize(final_outputs: dict, alpha: float = 0.35):
    image_path = final_outputs["original_image_path"]
    height = int(final_outputs["orig_img_h"])
    width = int(final_outputs["orig_img_w"])
    pred_masks = final_outputs.get("pred_masks", [])
    pred_scores = final_outputs.get("pred_scores", [])

    image = Image.open(image_path).convert("RGB")
    img = np.array(image).astype(np.float32)

    mask_list = _normalize_pred_masks_to_list(pred_masks)

    if len(mask_list) > 0:
        if pred_scores is None:
            pred_scores = []

        indices = list(range(len(mask_list)))
        if len(pred_scores) == len(mask_list):
            indices = sorted(indices, key=lambda i: pred_scores[i], reverse=True)

        ranked_masks = [mask_list[i] for i in indices]

        for rank, mask in enumerate(ranked_masks, start=1):
            color = _DEFAULT_COLORS[(rank - 1) % len(_DEFAULT_COLORS)]
            img = _overlay_single_mask(img, mask, color, alpha=alpha)

        out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(out)
        font = ImageFont.load_default()

        for rank, mask in enumerate(ranked_masks, start=1):
            anchor = _find_mask_anchor(mask)
            if anchor is None:
                continue
            x, y = anchor
            text = str(rank)

            try:
                bbox = draw.textbbox((x, y), text, font=font)
                pad = 2
                draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(255, 255, 255))
            except Exception:
                pass

            draw.text((x, y), text, fill=(0, 0, 0), font=font)

        return out

    union_mask = union_pred_masks(pred_masks, height, width).astype(bool)
    img = _overlay_single_mask(img, union_mask.astype(np.uint8), (0, 255, 0), alpha=alpha)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)
