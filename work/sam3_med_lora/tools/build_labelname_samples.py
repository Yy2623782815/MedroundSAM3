# filename: /root/autodl-tmp/work/sam3_med_lora/tools/build_labelname_samples.py
import os
from pathlib import Path
import sys
import json
import argparse
import random
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from med_data_utils import (  # noqa: E402
    load_dataset_json,
    build_label_name_to_index,
    load_gt_mask_from_npz,
)


def save_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_samples_for_one_split(
    dataset_root: str,
    dataset_name: str,
    json_name: str,
    split_key: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    json_path = os.path.join(dataset_root, json_name)
    dataset_json = load_dataset_json(json_path)
    label_name_to_idx = build_label_name_to_index(dataset_json)

    items = dataset_json[split_key]
    samples: List[Dict[str, Any]] = []

    stats = {
        "num_items": len(items),
        "num_questions_total": 0,
        "num_samples_kept": 0,
        "num_skip_invalid_label": 0,
        "num_skip_background": 0,
        "num_skip_empty": 0,
        "num_skip_load_fail": 0,
        "num_skip_duplicate": 0,
    }

    item_pbar = tqdm(
        items,
        desc=f"{dataset_name} [{split_key}]",
        leave=False,
        dynamic_ncols=True,
    )

    for item in item_pbar:
        image_path = os.path.join(dataset_root, item["image"])
        label_npz_path = os.path.join(dataset_root, item["label"])
        image_rel = item["image"]
        label_rel = item["label"]
        class_ids = item.get("class_ids", [])

        # 同一张图可能有多个 question 指向同一个 label，去重
        seen = set()

        questions = item.get("questions", [])
        stats["num_questions_total"] += len(questions)

        for q in questions:
            label_name = q["label"].strip()

            # 跳过伪标签 / 不存在于 labels 中的标签
            if label_name not in label_name_to_idx:
                stats["num_skip_invalid_label"] += 1
                continue

            channel_idx = label_name_to_idx[label_name]

            # 跳过背景
            if channel_idx == 0:
                stats["num_skip_background"] += 1
                continue

            uniq_key = (image_rel, label_rel, label_name, channel_idx)
            if uniq_key in seen:
                stats["num_skip_duplicate"] += 1
                continue
            seen.add(uniq_key)

            # 用你当前确认无误的读取函数验证该 mask 是否非空
            try:
                mask = load_gt_mask_from_npz(label_npz_path, channel_idx)
            except Exception as e:
                stats["num_skip_load_fail"] += 1
                print(
                    f"[SKIP-LOAD-FAIL] dataset={dataset_name} split={split_key} "
                    f"image={image_rel} label={label_name} ch={channel_idx} err={e}"
                )
                continue

            nonzero = int((mask > 0).sum())
            if nonzero <= 0:
                stats["num_skip_empty"] += 1
                print(
                    f"[SKIP-EMPTY] dataset={dataset_name} split={split_key} "
                    f"image={image_rel} label={label_name} ch={channel_idx}"
                )
                continue

            sample = {
                "dataset": dataset_name,
                "split": split_key,
                "image_path": image_path,
                "label_npz_path": label_npz_path,
                "image_rel": image_rel,
                "label_rel": label_rel,
                "class_ids": class_ids,
                "label_name": label_name,
                "channel_idx": channel_idx,
                "prompt_type": "label_name",
                "prompt_text": label_name,
                "mask_nonzero": nonzero,
                "question_id": q.get("id", ""),
                "question_text": q.get("question", ""),
            }
            samples.append(sample)
            stats["num_samples_kept"] += 1

        item_pbar.set_postfix({
            "kept": stats["num_samples_kept"],
            "empty": stats["num_skip_empty"],
            "fail": stats["num_skip_load_fail"],
        })

    return samples, stats


def split_train_val(
    training_samples: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rnd = random.Random(seed)
    samples = training_samples[:]
    rnd.shuffle(samples)

    n_val = int(len(samples) * val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    for s in train_samples:
        s["split"] = "train"
    for s in val_samples:
        s["split"] = "val"

    return train_samples, val_samples


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dataset = {}
    by_label = {}

    for r in records:
        ds = r["dataset"]
        lb = r["label_name"]
        by_dataset[ds] = by_dataset.get(ds, 0) + 1
        by_label[lb] = by_label.get(lb, 0) + 1

    return {
        "num_samples": len(records),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_label": dict(sorted(by_label.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="例如: data/SAM3_data 或绝对路径")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="如 AMOS2022 或 AMOS2022 BraTS CHAOS")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_training_raw: List[Dict[str, Any]] = []
    all_test: List[Dict[str, Any]] = []

    per_dataset_stats = {}

    dataset_pbar = tqdm(
        args.datasets,
        desc="Datasets",
        leave=True,
        dynamic_ncols=True,
    )

    total_kept_train = 0
    total_kept_test = 0

    for dataset_name in dataset_pbar:
        dataset_root = os.path.join(args.data_root, dataset_name)
        json_name = f"MultiEN_{dataset_name}.json"

        training_raw, training_stats = build_samples_for_one_split(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            json_name=json_name,
            split_key="training",
        )
        test_samples, test_stats = build_samples_for_one_split(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            json_name=json_name,
            split_key="test",
        )

        all_training_raw.extend(training_raw)
        all_test.extend(test_samples)

        total_kept_train += len(training_raw)
        total_kept_test += len(test_samples)

        per_dataset_stats[dataset_name] = {
            "training": training_stats,
            "test": test_stats,
        }

        dataset_pbar.set_postfix({
            "train_kept": total_kept_train,
            "test_kept": total_kept_test,
        })

        print(f"\n[{dataset_name}]")
        print(
            f"  training: items={training_stats['num_items']} "
            f"questions={training_stats['num_questions_total']} "
            f"kept={training_stats['num_samples_kept']} "
            f"empty={training_stats['num_skip_empty']} "
            f"fail={training_stats['num_skip_load_fail']} "
            f"dup={training_stats['num_skip_duplicate']}"
        )
        print(
            f"  test:     items={test_stats['num_items']} "
            f"questions={test_stats['num_questions_total']} "
            f"kept={test_stats['num_samples_kept']} "
            f"empty={test_stats['num_skip_empty']} "
            f"fail={test_stats['num_skip_load_fail']} "
            f"dup={test_stats['num_skip_duplicate']}"
        )

    train_samples, val_samples = split_train_val(
        training_samples=all_training_raw,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_path = os.path.join(args.output_dir, "train_samples.jsonl")
    val_path = os.path.join(args.output_dir, "val_samples.jsonl")
    test_path = os.path.join(args.output_dir, "test_samples.jsonl")

    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)
    save_jsonl(all_test, test_path)

    summary = {
        "datasets": args.datasets,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "per_dataset_stats": per_dataset_stats,
        "train_summary": summarize(train_samples),
        "val_summary": summarize(val_samples),
        "test_summary": summarize(all_test),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] train={len(train_samples)} val={len(val_samples)} test={len(all_test)}")
    print(f"[OUT] {args.output_dir}")
    print(f"[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()