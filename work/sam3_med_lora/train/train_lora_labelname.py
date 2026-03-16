import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict

PROJECT_ROOT = "/root/autodl-tmp/work/sam3_med_lora"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.collate import med_labelname_collate_fn
from datasets.med_labelname_dataset import MedLabelNameDataset
from models.build_sam3_lora import build_sam3_lora_model, move_batch_to_device
from models.lora import extract_lora_state_dict
from models.sam3_forward import sam3_train_forward
from utils.losses import (
    binary_dice_score_from_logits,
    binary_iou_score_from_logits,
    combined_bce_dice_loss,
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# filename: /root/autodl-tmp/work/sam3_med_lora/train/train_lora_labelname.py

def build_dataloaders(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]

    train_dataset = MedLabelNameDataset(
        index_jsonl=data_cfg["train_index_jsonl"],
        image_size=data_cfg["image_size"],
        normalize=data_cfg.get("normalize", True),
    )
    val_dataset = MedLabelNameDataset(
        index_jsonl=data_cfg["val_index_jsonl"],
        image_size=data_cfg["image_size"],
        normalize=data_cfg.get("normalize", True),
    )

    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", True))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        drop_last=data_cfg.get("drop_last", False),
        collate_fn=med_labelname_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg["val_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        drop_last=False,
        collate_fn=med_labelname_collate_fn,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def _debug_fake_forward(batch: Dict[str, Any], device: str) -> torch.Tensor:
    """
    第一版占位 logits。
    先用一个最小可跑通版本把训练框架搭起来。

    当前返回：
        全 0 logits, shape [B,1,H,W]

    后续要替换成：
        SAM3 图像 + 文本 prompt 的真实训练 forward logits
    """
    masks = batch["masks"].to(device)
    b, h, w = masks.shape
    logits = torch.zeros((b, 1, h, w), device=device, dtype=torch.float32, requires_grad=True)
    return logits


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    train_cfg = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )
    return optimizer


def run_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    cfg: Dict[str, Any],
    is_train: bool,
) -> Dict[str, float]:
    train_cfg = cfg["train"]
    use_amp = bool(train_cfg.get("use_amp", True)) and str(device).startswith("cuda")
    amp_dtype_str = str(train_cfg.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_interval = int(train_cfg.get("log_interval", 20))

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_steps = 0

    if is_train:
        model.train()
    else:
        model.eval()

    pbar = tqdm(loader, desc=f"{'train' if is_train else 'val'} epoch {epoch}", leave=False)

    for step, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred_logits = sam3_train_forward(model, batch, device)
                    loss, loss_stats = combined_bce_dice_loss(
                        pred_logits,
                        batch["masks"],
                        bce_weight=train_cfg.get("bce_weight", 0.5),
                        dice_weight=train_cfg.get("dice_weight", 0.5),
                    )
            else:
                pred_logits = sam3_train_forward(model, batch, device)
                loss, loss_stats = combined_bce_dice_loss(
                    pred_logits,
                    batch["masks"],
                    bce_weight=train_cfg.get("bce_weight", 0.5),
                    dice_weight=train_cfg.get("dice_weight", 0.5),
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        with torch.no_grad():
            dice = binary_dice_score_from_logits(pred_logits, batch["masks"]).item()
            iou = binary_iou_score_from_logits(pred_logits, batch["masks"]).item()

        total_loss += float(loss.item())
        total_bce += float(loss_stats["loss_bce"])
        total_dice_loss += float(loss_stats["loss_dice"])
        total_dice += float(dice)
        total_iou += float(iou)
        num_steps += 1

        if step % log_interval == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{dice:.4f}",
                iou=f"{iou:.4f}",
            )

    if num_steps == 0:
        return {
            "loss": 0.0,
            "loss_bce": 0.0,
            "loss_dice": 0.0,
            "dice": 0.0,
            "iou": 0.0,
        }

    return {
        "loss": total_loss / num_steps,
        "loss_bce": total_bce / num_steps,
        "loss_dice": total_dice_loss / num_steps,
        "dice": total_dice / num_steps,
        "iou": total_iou / num_steps,
    }


def save_checkpoint(
    *,
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_dice: float,
    config: Dict[str, Any],
    extra_info: Dict[str, Any],
) -> None:
    ckpt = {
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "optimizer": optimizer.state_dict(),
        "lora_state_dict": extract_lora_state_dict(model),
        "config": config,
        "extra_info": extra_info,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config path",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    output_dir = cfg["output"]["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "checkpoints"))
    ensure_dir(os.path.join(output_dir, "logs"))

    save_json(cfg, os.path.join(output_dir, "config_resolved.json"))

    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True


    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(cfg)
    print(f"[info] train samples = {len(train_dataset)}")
    print(f"[info] val samples   = {len(val_dataset)}")

    model, model_info = build_sam3_lora_model(
        checkpoint_path=cfg["model"]["checkpoint_path"],
        bpe_path=cfg["model"]["bpe_path"],
        device=device,
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        train_bias=cfg["lora"].get("train_bias", False),
        train_norm=cfg["lora"].get("train_norm", False),
        load_from_hf=cfg["model"].get("load_from_hf", False),
        verbose=True,
    )
    save_json(model_info, os.path.join(output_dir, "model_info.json"))

    optimizer = build_optimizer(cfg, model)

    epochs = int(cfg["train"]["epochs"])
    val_every_n_epochs = int(cfg["train"].get("val_every_n_epochs", 1))
    best_val_dice = -1.0
    history = []

    val_every_n_epochs = int(cfg["train"].get("val_every_n_epochs", 1))


    for epoch in range(1, epochs + 1):
        t0 = time.time()
    
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg,
            is_train=True,
        )
    
        if epoch % val_every_n_epochs == 0:
            val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                cfg=cfg,
                is_train=False,
            )
        else:
            val_metrics = {
                "loss": 0.0,
                "loss_bce": 0.0,
                "loss_dice": 0.0,
                "dice": 0.0,
                "iou": 0.0,
            }
    
        elapsed = time.time() - t0
    
        print("[done] training finished.")
        print(f"[done] output_dir = {output_dir}")
        print(
            "[note] 当前脚本的 forward 仍是占位版 _debug_fake_forward()，"
            "下一步需要替换成真实 SAM3 文本分割训练 forward。"
        )


if __name__ == "__main__":
    main()