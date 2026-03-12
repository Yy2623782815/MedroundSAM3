# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from helpers.visualizer import Visualizer
from helpers.zoom_in import render_zoom_in

# 新增到 viz.py 中
def load_image_as_rgb_uint8(img_path: str) -> np.ndarray:
    """
    Robust image loader for medical PNGs.
    Supports grayscale / 16-bit PNG / RGB images.
    Returns an HxWx3 uint8 RGB numpy array.
    """
    pil_img = Image.open(img_path)
    arr = np.array(pil_img)

    # Case 1: grayscale image, shape [H, W]
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        arr = arr.clip(0, 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    # Case 2: [H, W, 1]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0].astype(np.float32)
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        arr = arr.clip(0, 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    # Case 3: RGB/RGBA or other multi-channel
    if arr.ndim == 3:
        # If dtype is not uint8, normalize to uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            arr = arr.clip(0, 255).astype(np.uint8)

        # RGBA -> RGB
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        # If already RGB
        if arr.shape[2] == 3:
            return arr

    raise ValueError(f"Unsupported image format for visualization: {img_path}, shape={arr.shape}, dtype={arr.dtype}")

def visualize(
    input_json: dict,
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
):
    """
    Unified visualization function.

    If zoom_in_index is None:
        - Render all masks in input_json (equivalent to visualize_masks_from_result_json).
        - Returns: PIL.Image

    If zoom_in_index is provided:
        - Returns two PIL.Images:
            1) Output identical to zoom_in_and_visualize(input_json, index).
            2) The same instance rendered via the general overlay using the color
               returned by (1), equivalent to calling visualize_masks_from_result_json
               on a single-mask json_i with color=color_hex.
    """
    # Common fields
    orig_h = int(input_json["orig_img_h"])
    orig_w = int(input_json["orig_img_w"])
    img_path = input_json["original_image_path"]

    # ---------- Mode A: Full-scene render ----------
    if zoom_in_index is None:
        boxes = np.array(input_json["pred_boxes"])
        rle_masks = [
            {"size": (orig_h, orig_w), "counts": rle}
            for rle in input_json["pred_masks"]
        ]
        binary_masks = [mask_utils.decode(rle) for rle in rle_masks]

        img_rgb = load_image_as_rgb_uint8(img_path)

        viz = Visualizer(
            img_rgb,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz.overlay_instances(
            boxes=boxes,
            masks=rle_masks,
            binary_masks=binary_masks,
            assigned_colors=None,
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        pil_all_masks = Image.fromarray(viz.output.get_image())
        return pil_all_masks

    # ---------- Mode B: Zoom-in pair ----------
    else:
        idx = int(zoom_in_index)
        num_masks = len(input_json.get("pred_masks", []))
        if idx < 0 or idx >= num_masks:
            raise ValueError(
                f"zoom_in_index {idx} is out of range (0..{num_masks - 1})."
            )

        # (1) Replicate zoom_in_and_visualize
        object_data = {
            "labels": [{"noun_phrase": f"mask_{idx}"}],
            "segmentation": {
                "counts": input_json["pred_masks"][idx],
                "size": [orig_h, orig_w],
            },
        }
        pil_img = Image.open(img_path)
        pil_mask_i_zoomed, color_hex = render_zoom_in(
            object_data, pil_img, mask_alpha=mask_alpha
        )

        # (2) Single-instance render with the same color
        boxes_i = np.array([input_json["pred_boxes"][idx]])
        rle_i = {"size": (orig_h, orig_w), "counts": input_json["pred_masks"][idx]}
        bin_i = mask_utils.decode(rle_i)

        img_rgb_i = load_image_as_rgb_uint8(img_path)

        viz_i = Visualizer(
            img_rgb_i,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz_i.overlay_instances(
            boxes=boxes_i,
            masks=[rle_i],
            binary_masks=[bin_i],
            assigned_colors=[color_hex],
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        pil_mask_i = Image.fromarray(viz_i.output.get_image())

        return pil_mask_i, pil_mask_i_zoomed
