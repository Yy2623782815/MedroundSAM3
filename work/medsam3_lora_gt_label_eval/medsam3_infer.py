# filename: /root/autodl-tmp/work/medsam3_lora_gt_label_eval/medsam3_infer.py
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision.ops import nms
import yaml

# ---- Make MedSAM3 repo importable ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDSAM3_REPO_ROOT = str(PROJECT_ROOT / "repos" / "MedSAM3")
if MEDSAM3_REPO_ROOT not in sys.path:
    sys.path.insert(0, MEDSAM3_REPO_ROOT)

# ---- Author-style imports from MedSAM3 / SAM3 repo ----
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata,
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


class MedSAM3LoRAInferencer:
    """
    Inference wrapper aligned with MedSAM3 author's infer_sam.py logic:
    - build_sam3_image_model(...)
    - apply LoRA
    - one prompt per forward pass
    - threshold on pred_logits
    - NMS on boxes
    - sigmoid + >0.5 on masks
    - resize masks back to original image size
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        resolution: int = 1008,
        detection_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda",
        load_from_HF: bool = False,
        checkpoint_path: Optional[str] = str(PROJECT_ROOT / "models" / "sam3_base" / "sam3.pt"),
        bpe_path: str = str(PROJECT_ROOT / "repos" / "MedSAM3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

        self.weights_path = weights_path
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print("[MedSAM3] Initializing SAM3 + LoRA")
        print(f"[MedSAM3] device={self.device}")
        print(f"[MedSAM3] resolution={self.resolution}")
        print(f"[MedSAM3] detection_threshold={self.detection_threshold}")
        print(f"[MedSAM3] nms_iou_threshold={self.nms_iou_threshold}")
        print(f"[MedSAM3] load_from_HF={load_from_HF}")

        # Author-style base model construction
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=load_from_HF,
            checkpoint_path=None if load_from_HF else checkpoint_path,
            bpe_path=bpe_path,
            eval_mode=True,
        )

        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )

        self.model = apply_lora_to_model(self.model, lora_config)
        load_lora_weights(self.model, weights_path)

        self.model.to(self.device)
        self.model.eval()

        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def create_datapoint(self, pil_image: PILImage.Image, text_prompts: List[str]) -> Datapoint:
        w, h = pil_image.size
        sam_image = SAMImage(data=pil_image, objects=[], size=[h, w])

        queries = []
        for idx, text_query in enumerate(text_prompts):
            query = FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                ),
            )
            queries.append(query)

        return Datapoint(find_queries=queries, images=[sam_image])

    @torch.no_grad()
    def predict_single_prompt(self, image_path: str, prompt: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_image = PILImage.open(image_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        datapoint = self.create_datapoint(pil_image, [prompt])
        datapoint = self.transform(datapoint)
        batch = collate_fn_api([datapoint], dict_key="input")["input"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        outputs = self.model(batch)
        last_output = outputs[-1]

        pred_logits = last_output["pred_logits"]   # [B, Q, C]
        pred_boxes = last_output["pred_boxes"]     # [B, Q, 4]
        pred_masks = last_output.get("pred_masks", None)  # [B, Q, H, W]

        out_probs = pred_logits.sigmoid()
        scores = out_probs[0, :, :].max(dim=-1)[0]  # [Q]

        keep = scores > self.detection_threshold
        num_keep = int(keep.sum().item())

        if num_keep == 0:
            return {
                "original_image_path": image_path,
                "orig_img_h": orig_h,
                "orig_img_w": orig_w,
                "pred_boxes": [],
                "pred_scores": [],
                "pred_masks": [],
                "num_detections": 0,
            }

        boxes_cxcywh = pred_boxes[0, keep]
        kept_scores = scores[keep]

        cx, cy, w, h = boxes_cxcywh.unbind(-1)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
        boxes_xyxy = boxes_xyxy[keep_nms]
        kept_scores = kept_scores[keep_nms]

        masks_np = []
        if pred_masks is not None:
            # author logic: sigmoid > 0.5, then resize to original image size, then > 0.5
            masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5  # [N, H, W]

            masks_resized = F.interpolate(
                masks_small.unsqueeze(0).float(),  # [1, N, H, W]
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) > 0.5  # [N, orig_h, orig_w]

            masks_np = masks_resized.cpu().numpy().astype(np.uint8)

        pred_boxes_xywh_norm = []
        for box in boxes_xyxy.cpu().numpy():
            x1_, y1_, x2_, y2_ = box.tolist()
            bw = max(0.0, x2_ - x1_)
            bh = max(0.0, y2_ - y1_)
            pred_boxes_xywh_norm.append([
                x1_ / orig_w,
                y1_ / orig_h,
                bw / orig_w,
                bh / orig_h,
            ])

        pred_scores_list = kept_scores.cpu().numpy().tolist()

        # 按 score 从高到低排序，保证：
        # 1号实例 = 最高分
        # 2号实例 = 第二高分
        # ...
        order = sorted(
            range(len(pred_scores_list)),
            key=lambda i: pred_scores_list[i],
            reverse=True,
        )

        pred_scores_list = [pred_scores_list[i] for i in order]
        pred_boxes_xywh_norm = [pred_boxes_xywh_norm[i] for i in order]

        if isinstance(masks_np, np.ndarray) and masks_np.ndim == 3 and len(order) > 0:
            masks_np = masks_np[order]
        elif isinstance(masks_np, list) and len(order) > 0:
            masks_np = [masks_np[i] for i in order]

        print(f"[MedSAM3] prompt={prompt} num_detections={len(pred_scores_list)} top_scores={pred_scores_list[:5]}")
        return {
            "original_image_path": image_path,
            "orig_img_h": orig_h,
            "orig_img_w": orig_w,
            "pred_boxes": pred_boxes_xywh_norm,
            "pred_scores": pred_scores_list,
            "pred_masks": masks_np,
            "num_detections": int(len(pred_scores_list)),
        }