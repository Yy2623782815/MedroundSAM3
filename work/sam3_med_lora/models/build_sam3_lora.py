from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from sam3 import build_sam3_image_model

from models.lora import (
    LoRAConfig,
    inject_lora_into_model,
    mark_only_lora_as_trainable,
    print_trainable_param_stats,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BPE_PATH = PROJECT_ROOT / "repos" / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_CKPT_PATH = PROJECT_ROOT / "models" / "sam3_base" / "sam3.pt"


def build_sam3_lora_model(
    checkpoint_path: str = str(DEFAULT_CKPT_PATH),
    bpe_path: str = str(DEFAULT_BPE_PATH),
    device: str = "cuda",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    train_bias: bool = False,
    train_norm: bool = False,
    load_from_hf: bool = False,
    verbose: bool = True,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """
    构建第一版医学 LoRA 训练模型。

    说明：
    - eval_mode=False：后续训练需要保留训练图
    - 先对官方 image model 注入 LoRA
    - 默认只训练 LoRA 参数
    """
    if verbose:
        print("[build] build_sam3_image_model...")
        print(f"[build] checkpoint_path={checkpoint_path}")
        print(f"[build] bpe_path={bpe_path}")
        print(f"[build] device={device}")

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        device=device,
        eval_mode=False,
    )

    lora_cfg = LoRAConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    replaced_modules = inject_lora_into_model(
        model=model,
        cfg=lora_cfg,
        verbose=verbose,
    )

    mark_only_lora_as_trainable(
        model=model,
        train_bias=train_bias,
        train_norm=train_norm,
    )

    if verbose:
        print_trainable_param_stats(model)

    extra = {
        "checkpoint_path": checkpoint_path,
        "bpe_path": bpe_path,
        "device": device,
        "load_from_hf": load_from_hf,
        "lora_config": asdict(lora_cfg),
        "num_replaced_modules": len(replaced_modules),
        "replaced_modules": replaced_modules,
        "train_bias": train_bias,
        "train_norm": train_norm,
    }
    return model, extra


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """
    仅将 tensor 字段搬到 device；字符串/list 元信息保持原样。
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def set_train_mode(model: torch.nn.Module) -> None:
    model.train()


@torch.no_grad()
def set_eval_mode(model: torch.nn.Module) -> None:
    model.eval()


def count_trainable_named_parameters(model: torch.nn.Module) -> list[tuple[str, int]]:
    rows = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            rows.append((name, p.numel()))
    return rows