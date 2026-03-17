# filename: /root/autodl-tmp/work/medsam3_my_lora/tools/build_labelname_samples.py
import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from utils.med_data_utils import build_label_name_to_index, load_dataset_json, load_gt_mask_from_npz


def save_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_samples_for_one_split(dataset_root: str, dataset_name: str, split_key: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    dataset_json = load_dataset_json(os.path.join(dataset_root, f"MultiEN_{dataset_name}.json"))
    label_name_to_idx = build_label_name_to_index(dataset_json)
    items = dataset_json[split_key]
    samples: List[Dict[str, Any]] = []
    stats = {"num_items": len(items), "kept": 0, "skip_empty": 0, "skip_fail": 0, "skip_invalid": 0}

    for item in tqdm(items, desc=f"{dataset_name}-{split_key}", leave=False):
        image_rel = item["image"]
        label_rel = item["label"]
        image_path = os.path.join(dataset_root, image_rel)
        label_npz_path = os.path.join(dataset_root, label_rel)
        case_id = os.path.splitext(os.path.basename(image_rel))[0]

        dedup = set()
        for q in item.get("questions", []):
            label_name = q["label"].strip()
            if label_name not in label_name_to_idx:
                stats["skip_invalid"] += 1
                continue
            channel_idx = label_name_to_idx[label_name]
            if channel_idx == 0:
                continue
            key = (image_rel, label_rel, channel_idx)
            if key in dedup:
                continue
            dedup.add(key)
            try:
                mask = load_gt_mask_from_npz(label_npz_path, channel_idx)
            except Exception:
                stats["skip_fail"] += 1
                continue
            if int((mask > 0).sum()) <= 0:
                stats["skip_empty"] += 1
                continue
            samples.append(
                {
                    "dataset": dataset_name,
                    "split": split_key,
                    "image_path": image_path,
                    "label_npz_path": label_npz_path,
                    "image_rel": image_rel,
                    "label_rel": label_rel,
                    "label_name": label_name,
                    "channel_idx": channel_idx,
                    "prompt_text": label_name,
                    "question_id": q.get("id", ""),
                    "case_id": case_id,
                    "sample_id": f"{dataset_name}::{case_id}::{channel_idx}",
                }
            )
            stats["kept"] += 1
    return samples, stats


def split_train_val(training_samples: List[Dict[str, Any]], val_ratio: float, seed: int):
    samples = training_samples[:]
    random.Random(seed).shuffle(samples)
    n_val = int(len(samples) * val_ratio)
    val = samples[:n_val]
    train = samples[n_val:]
    for x in train:
        x["split"] = "train"
    for x in val:
        x["split"] = "val"
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    train_raw: List[Dict[str, Any]] = []
    test_all: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"datasets": args.datasets, "stats": {}}

    for dataset_name in args.datasets:
        ds_root = os.path.join(args.data_root, dataset_name)
        tr, tr_stat = build_samples_for_one_split(ds_root, dataset_name, "training")
        te, te_stat = build_samples_for_one_split(ds_root, dataset_name, "test")
        train_raw.extend(tr)
        test_all.extend(te)
        summary["stats"][dataset_name] = {"training": tr_stat, "test": te_stat}

    train, val = split_train_val(train_raw, args.val_ratio, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_jsonl(train, os.path.join(args.output_dir, "train_samples.jsonl"))
    save_jsonl(val, os.path.join(args.output_dir, "val_samples.jsonl"))
    save_jsonl(test_all, os.path.join(args.output_dir, "test_samples.jsonl"))
    summary.update({"train": len(train), "val": len(val), "test": len(test_all)})
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
