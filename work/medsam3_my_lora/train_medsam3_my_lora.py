import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def setup_distributed() -> Dict[str, Any]:
    rank, world_size, local_rank = get_dist_info()
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA devices.")
        dist.init_process_group(backend="nccl", init_method="env://")

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if distributed else "cuda"
        torch.cuda.set_device(local_rank if distributed else 0)
    else:
        device = "cpu"

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "distributed": distributed,
        "is_main": rank == 0,
        "device": device,
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def maybe_wrap_ddp(model: torch.nn.Module, distributed: bool, local_rank: int) -> torch.nn.Module:
    if not distributed:
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )


def get_per_device_batch_size(data_cfg: Dict[str, Any], world_size: int) -> int:
    if "batch_size_per_device" in data_cfg:
        return int(data_cfg["batch_size_per_device"])
    if "global_batch_size" in data_cfg:
        gbs = int(data_cfg["global_batch_size"])
        if gbs % max(world_size, 1) != 0:
            raise ValueError(f"global_batch_size={gbs} cannot be evenly divided by world_size={world_size}")
        return gbs // max(world_size, 1)
    return int(data_cfg.get("batch_size", 1))


def build_dataloaders(cfg: Dict[str, Any], dist_ctx: Dict[str, Any]):
    data_cfg = cfg["data"]
    train_dataset = MedLabelNameDataset(data_cfg["train_index_jsonl"], data_cfg["image_size"], data_cfg.get("normalize", True))
    val_dataset = MedLabelNameDataset(data_cfg["val_index_jsonl"], data_cfg["image_size"], data_cfg.get("normalize", True))

    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size_per_device = get_per_device_batch_size(data_cfg, int(dist_ctx["world_size"]))
    val_batch_size_per_device = int(data_cfg.get("val_batch_size_per_device", data_cfg.get("val_batch_size", batch_size_per_device)))

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if dist_ctx["distributed"]:
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist_ctx["world_size"], rank=dist_ctx["rank"], shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=dist_ctx["world_size"], rank=dist_ctx["rank"], shuffle=False)

    common = dict(num_workers=num_workers, pin_memory=bool(data_cfg.get("pin_memory", True)), collate_fn=med_labelname_collate_fn)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size_per_device,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    loader_meta = {
        "batch_size_per_device": batch_size_per_device,
        "val_batch_size_per_device": val_batch_size_per_device,
        "global_batch_size": batch_size_per_device * dist_ctx["world_size"],
        "global_val_batch_size": val_batch_size_per_device * dist_ctx["world_size"],
    }
    return train_dataset, val_dataset, train_loader, val_loader, train_sampler, loader_meta


def reduce_scalar(value: float, device: str, distributed: bool) -> float:
    if not distributed:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def run_one_epoch(
    model,
    loader,
    optimizer,
    cfg,
    device: str,
    is_train: bool,
    epoch: int,
    total_epochs: int,
    dist_ctx: Dict[str, Any],
) -> Dict[str, float]:
    train_cfg = cfg["train"]
    model.train() if is_train else model.eval()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    steps = 0
    stage = "train" if is_train else "val"
    lr = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0

    pbar = tqdm(
        loader,
        desc=f"{stage} e{epoch}/{total_epochs}",
        leave=False,
        disable=not dist_ctx["is_main"],
        dynamic_ncols=True,
    )
    step_start = time.time()
    epoch_start = step_start

    for step_idx, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            query_select_cfg = cfg.get("model", {}).get("query_select", None)
            if query_select_cfg is None:
                query_select_cfg = train_cfg.get("query_select", None)
            if query_select_cfg is None and "query_select_mode" in train_cfg:
                query_select_cfg = {
                    "mode": "logits_max" if str(train_cfg["query_select_mode"]) == "best_logit" else "mask_mean",
                    "topk": 1,
                    "reduce": "mean",
                }

            pred_logits = sam3_train_forward(
                model.module if hasattr(model, "module") else model,
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
            dice = float(binary_dice_score_from_logits(pred_logits, batch["masks"]).item())
            iou = float(binary_iou_score_from_logits(pred_logits, batch["masks"]).item())
            loss_item = float(loss.item())
            totals["loss"] += loss_item
            totals["dice"] += dice
            totals["iou"] += iou
            steps += 1

        if dist_ctx["is_main"]:
            now = time.time()
            step_time = now - step_start
            epoch_elapsed = now - epoch_start
            avg_step_time = epoch_elapsed / max(step_idx, 1)
            epoch_eta = avg_step_time * max(len(loader) - step_idx, 0)
            pbar.set_postfix(
                {
                    "loss": f"{loss_item:.4f}",
                    "dice": f"{dice:.4f}",
                    "iou": f"{iou:.4f}",
                    "lr": f"{lr:.2e}",
                    "step_t": f"{step_time:.2f}s",
                    "ep_elapsed": format_seconds(epoch_elapsed),
                    "ep_eta": format_seconds(epoch_eta),
                }
            )
            step_start = now

    total_loss = reduce_scalar(totals["loss"], device, dist_ctx["distributed"])
    total_dice = reduce_scalar(totals["dice"], device, dist_ctx["distributed"])
    total_iou = reduce_scalar(totals["iou"], device, dist_ctx["distributed"])
    total_steps = reduce_scalar(float(steps), device, dist_ctx["distributed"])

    denom = max(total_steps, 1.0)
    return {"loss": total_loss / denom, "dice": total_dice / denom, "iou": total_iou / denom}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs in config")
    args = parser.parse_args()

    dist_ctx = setup_distributed()

    cfg = load_yaml(args.config)
    cfg["data"]["train_index_jsonl"] = resolve_path(cfg["data"]["train_index_jsonl"])
    cfg["data"]["val_index_jsonl"] = resolve_path(cfg["data"]["val_index_jsonl"])
    cfg["model"]["sam3_repo_root"] = resolve_path(cfg["model"]["sam3_repo_root"])
    cfg["model"]["checkpoint_path"] = resolve_path(cfg["model"]["checkpoint_path"])
    cfg["model"]["bpe_path"] = resolve_path(cfg["model"]["bpe_path"])
    cfg["output"]["output_dir"] = resolve_path(cfg["output"]["output_dir"])

    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    set_seed(int(cfg.get("seed", 42)) + dist_ctx["rank"])

    output_dir = cfg["output"]["output_dir"]
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    if dist_ctx["is_main"]:
        save_json(cfg, os.path.join(output_dir, "config_resolved.json"))

    device = cfg["model"].get("device", dist_ctx["device"])
    if dist_ctx["distributed"]:
        device = dist_ctx["device"]

    train_dataset, val_dataset, train_loader, val_loader, train_sampler, loader_meta = build_dataloaders(cfg, dist_ctx)
    if dist_ctx["is_main"]:
        print(
            "[info] "
            f"world_size={dist_ctx['world_size']} "
            f"train={len(train_dataset)} val={len(val_dataset)} device={device} "
            f"batch_size_per_device={loader_meta['batch_size_per_device']} "
            f"global_batch_size={loader_meta['global_batch_size']}"
        )

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
    model = maybe_wrap_ddp(model, dist_ctx["distributed"], dist_ctx["local_rank"])

    if dist_ctx["is_main"]:
        save_json(model_info, os.path.join(output_dir, "model_info.json"))

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-2)),
    )

    total_epochs = int(cfg["train"]["epochs"])
    best_val_dice = -1.0
    history = []
    all_epoch_times = []
    train_start = time.time()

    for epoch in range(1, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_t0 = time.time()
        train_metrics = run_one_epoch(model, train_loader, optimizer, cfg, device, True, epoch, total_epochs, dist_ctx)
        val_metrics = run_one_epoch(model, val_loader, optimizer, cfg, device, False, epoch, total_epochs, dist_ctx)
        epoch_sec = time.time() - epoch_t0
        all_epoch_times.append(epoch_sec)

        elapsed = time.time() - train_start
        avg_epoch = float(np.mean(all_epoch_times)) if all_epoch_times else epoch_sec
        remain_epochs = total_epochs - epoch
        eta = avg_epoch * remain_epochs
        estimated_total = elapsed + eta

        row = {
            "epoch": epoch,
            "time_sec": round(epoch_sec, 2),
            "elapsed_sec": round(elapsed, 2),
            "eta_sec": round(eta, 2),
            "estimated_total_sec": round(estimated_total, 2),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        if dist_ctx["is_main"]:
            ckpt = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "lora_state_dict": extract_lora_state_dict(model.module if hasattr(model, "module") else model),
                "config": cfg,
                "metrics": row,
                "dist": {"world_size": dist_ctx["world_size"]},
            }
            torch.save(ckpt, os.path.join(output_dir, "checkpoints", "latest.pt"))
            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                ckpt["best_val_dice"] = best_val_dice
                torch.save(ckpt, os.path.join(output_dir, "checkpoints", "best.pt"))

            print(
                f"[epoch {epoch}/{total_epochs}] "
                f"train_loss={train_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)} "
                f"estimated_total={format_seconds(estimated_total)}"
            )

    if dist_ctx["is_main"]:
        save_json({"best_val_dice": best_val_dice, "history": history}, os.path.join(output_dir, "logs", "history.json"))

    cleanup_distributed()


if __name__ == "__main__":
    main()
