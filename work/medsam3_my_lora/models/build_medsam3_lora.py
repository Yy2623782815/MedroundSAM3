# filename: /root/autodl-tmp/work/medsam3_my_lora/models/build_medsam3_lora.py
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
import sys

import torch

from models.lora import LoRAConfig, inject_lora_into_model, mark_only_lora_as_trainable, print_trainable_param_stats


def _ensure_sam3_importable(sam3_repo_root: str) -> None:
    repo_root = str(Path(sam3_repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def build_medsam3_lora_model(
    *,
    sam3_repo_root: str,
    checkpoint_path: str,
    bpe_path: str,
    device: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    train_bias: bool = False,
    train_norm: bool = False,
    load_from_hf: bool = False,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    _ensure_sam3_importable(sam3_repo_root)
    from sam3 import build_sam3_image_model

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        device=device,
        eval_mode=False,
    )

    lora_cfg = LoRAConfig(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    replaced_modules = inject_lora_into_model(model=model, cfg=lora_cfg, verbose=True)
    mark_only_lora_as_trainable(model=model, train_bias=train_bias, train_norm=train_norm)
    print_trainable_param_stats(model)

    extra = {
        "sam3_repo_root": sam3_repo_root,
        "checkpoint_path": checkpoint_path,
        "bpe_path": bpe_path,
        "load_from_hf": load_from_hf,
        "lora_config": asdict(lora_cfg),
        "num_replaced_modules": len(replaced_modules),
    }
    return model, extra


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out
