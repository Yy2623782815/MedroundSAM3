# filename: /root/autodl-tmp/work/medsam3_my_lora/train_medsam3_my_lora.py
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.collate import med_labelname_collate_fn
from datasets.med_labelname_dataset import MedLabelNameDataset
from models.build_medsam3_lora import build_medsam3_lora_model, move_batch_to_device
from models.lora import extract_lora_state_dict
from models.sam3_forward import sam3_train_forward
from utils.losses import binary_dice_score_from_logits, binary_iou_score_from_logits, combined_bce_dice_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_path(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def build_dataloaders(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    train_dataset = MedLabelNameDataset(data_cfg["train_index_jsonl"], data_cfg["image_size"], data_cfg.get("normalize", True))
    val_dataset = MedLabelNameDataset(data_cfg["val_index_jsonl"], data_cfg["image_size"], data_cfg.get("normalize", True))

    num_workers = int(data_cfg.get("num_workers", 4))
    common = dict(num_workers=num_workers, pin_memory=bool(data_cfg.get("pin_memory", True)), collate_fn=med_labelname_collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=int(data_cfg["batch_size"]), shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_dataset, batch_size=int(data_cfg["val_batch_size"]), shuffle=False, drop_last=False, **common)
    return train_dataset, val_dataset, train_loader, val_loader


def run_one_epoch(model, loader, optimizer, cfg, device: str, is_train: bool) -> Dict[str, float]:
    train_cfg = cfg["train"]
    model.train() if is_train else model.eval()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    steps = 0

    for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            query_select_cfg = cfg.get("model", {}).get("query_select", None)
            if query_select_cfg is None:
                query_select_cfg = train_cfg.get("query_select", None)
            # 向后兼容：若用户还在用旧字段 query_select_mode，则自动映射到新配置
            if query_select_cfg is None and "query_select_mode" in train_cfg:
                query_select_cfg = {
                    "mode": "logits_max" if str(train_cfg["query_select_mode"]) == "best_logit" else "mask_mean",
                    "topk": 1,
                    "reduce": "mean",
                }

            pred_logits = sam3_train_forward(
                model,
                batch,
                cfg["model"]["sam3_repo_root"],
                query_select_cfg=query_select_cfg,
            )
            loss, _ = combined_bce_dice_loss(pred_logits, batch["masks"], train_cfg.get("bce_weight", 0.5), train_cfg.get("dice_weight", 0.5))
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(train_cfg.get("grad_clip", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                optimizer.step()

        with torch.no_grad():
            totals["loss"] += float(loss.item())
            totals["dice"] += float(binary_dice_score_from_logits(pred_logits, batch["masks"]).item())
            totals["iou"] += float(binary_iou_score_from_logits(pred_logits, batch["masks"]).item())
            steps += 1

    return {k: (v / max(steps, 1)) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["data"]["train_index_jsonl"] = resolve_path(cfg["data"]["train_index_jsonl"])
    cfg["data"]["val_index_jsonl"] = resolve_path(cfg["data"]["val_index_jsonl"])
    cfg["model"]["sam3_repo_root"] = resolve_path(cfg["model"]["sam3_repo_root"])
    cfg["model"]["checkpoint_path"] = resolve_path(cfg["model"]["checkpoint_path"])
    cfg["model"]["bpe_path"] = resolve_path(cfg["model"]["bpe_path"])
    cfg["output"]["output_dir"] = resolve_path(cfg["output"]["output_dir"])
    set_seed(int(cfg.get("seed", 42)))

    output_dir = cfg["output"]["output_dir"]
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    save_json(cfg, os.path.join(output_dir, "config_resolved.json"))

    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(cfg)
    print(f"[info] train={len(train_dataset)} val={len(val_dataset)} device={device}")

    model, model_info = build_medsam3_lora_model(
        sam3_repo_root=cfg["model"]["sam3_repo_root"],
        checkpoint_path=cfg["model"]["checkpoint_path"],
        bpe_path=cfg["model"]["bpe_path"],
        device=device,
        lora_r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        train_bias=bool(cfg["lora"].get("train_bias", False)),
        train_norm=bool(cfg["lora"].get("train_norm", False)),
        load_from_hf=bool(cfg["model"].get("load_from_hf", False)),
    )
    save_json(model_info, os.path.join(output_dir, "model_info.json"))

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 1e-2)))
    best_val_dice = -1.0
    history = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        t0 = time.time()
        train_metrics = run_one_epoch(model, train_loader, optimizer, cfg, device, True)
        val_metrics = run_one_epoch(model, val_loader, optimizer, cfg, device, False)
        row = {"epoch": epoch, "time_sec": round(time.time() - t0, 2), "train": train_metrics, "val": val_metrics}
        history.append(row)

        ckpt = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "lora_state_dict": extract_lora_state_dict(model),
            "config": cfg,
            "metrics": row,
        }
        torch.save(ckpt, os.path.join(output_dir, "checkpoints", "latest.pt"))
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            ckpt["best_val_dice"] = best_val_dice
            torch.save(ckpt, os.path.join(output_dir, "checkpoints", "best.pt"))

        print(f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f}")

    save_json({"best_val_dice": best_val_dice, "history": history}, os.path.join(output_dir, "logs", "history.json"))


if __name__ == "__main__":
    main()
