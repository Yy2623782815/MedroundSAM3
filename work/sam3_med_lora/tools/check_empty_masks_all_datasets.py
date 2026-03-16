# filename: /root/autodl-tmp/work/sam3_med_lora/tools/check_empty_masks_all_datasets.py
import os
import sys
import json
import argparse
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = "/root/autodl-tmp/work/sam3_med_agent_eval"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from med_data_utils import (  # noqa: E402
    load_dataset_json,
    build_label_name_to_index,
    load_gt_mask_from_npz,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def init_stats() -> Dict[str, int]:
    return {
        "num_items": 0,
        "num_questions_total": 0,
        "num_questions_valid_label": 0,
        "num_non_empty_masks": 0,
        "num_empty_masks": 0,
        "num_load_fail": 0,
        "num_skipped_invalid_label": 0,
        "num_skipped_background": 0,
    }


def process_one_dataset(dataset_root: str, dataset_name: str) -> Dict[str, Any]:
    json_path = os.path.join(dataset_root, f"MultiEN_{dataset_name}.json")
    dataset_json = load_dataset_json(json_path)
    label_name_to_idx = build_label_name_to_index(dataset_json)

    dataset_stats = {
        "dataset": dataset_name,
        "training": init_stats(),
        "test": init_stats(),
        "overall": init_stats(),
    }

    empty_records: List[Dict[str, Any]] = []
    load_fail_records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []

    for split in ["training", "test"]:
        items = dataset_json.get(split, [])
        dataset_stats[split]["num_items"] = len(items)

        item_pbar = tqdm(
            items,
            desc=f"{dataset_name} [{split}]",
            leave=False,
            dynamic_ncols=True,
        )

        for item in item_pbar:
            image_rel = item["image"]
            label_rel = item["label"]
            image_path = os.path.join(dataset_root, image_rel)
            label_npz_path = os.path.join(dataset_root, label_rel)
            class_ids = item.get("class_ids", [])

            questions = item.get("questions", [])
            dataset_stats[split]["num_questions_total"] += len(questions)

            for q in questions:
                label_name = str(q.get("label", "")).strip()
                qid = q.get("id", "")
                qtext = q.get("question", "")

                if label_name not in label_name_to_idx:
                    dataset_stats[split]["num_skipped_invalid_label"] += 1
                    skipped_records.append({
                        "dataset": dataset_name,
                        "split": split,
                        "image_rel": image_rel,
                        "label_rel": label_rel,
                        "question_id": qid,
                        "label_name": label_name,
                        "reason": "invalid_label_not_in_top_level_labels",
                    })
                    continue

                channel_idx = label_name_to_idx[label_name]

                if channel_idx == 0:
                    dataset_stats[split]["num_skipped_background"] += 1
                    skipped_records.append({
                        "dataset": dataset_name,
                        "split": split,
                        "image_rel": image_rel,
                        "label_rel": label_rel,
                        "question_id": qid,
                        "label_name": label_name,
                        "channel_idx": channel_idx,
                        "reason": "background_label",
                    })
                    continue

                dataset_stats[split]["num_questions_valid_label"] += 1

                try:
                    mask = load_gt_mask_from_npz(label_npz_path, channel_idx)
                except Exception as e:
                    dataset_stats[split]["num_load_fail"] += 1
                    load_fail_records.append({
                        "dataset": dataset_name,
                        "split": split,
                        "image_rel": image_rel,
                        "image_path": image_path,
                        "label_rel": label_rel,
                        "label_npz_path": label_npz_path,
                        "class_ids": class_ids,
                        "question_id": qid,
                        "question_text": qtext,
                        "label_name": label_name,
                        "channel_idx": channel_idx,
                        "error": str(e),
                    })
                    continue

                nonzero = int((mask > 0).sum())
                if nonzero == 0:
                    dataset_stats[split]["num_empty_masks"] += 1
                    empty_records.append({
                        "dataset": dataset_name,
                        "split": split,
                        "image_rel": image_rel,
                        "image_path": image_path,
                        "label_rel": label_rel,
                        "label_npz_path": label_npz_path,
                        "class_ids": class_ids,
                        "question_id": qid,
                        "question_text": qtext,
                        "label_name": label_name,
                        "channel_idx": channel_idx,
                        "mask_nonzero": nonzero,
                    })
                else:
                    dataset_stats[split]["num_non_empty_masks"] += 1

            item_pbar.set_postfix({
                "q_total": dataset_stats[split]["num_questions_total"],
                "valid": dataset_stats[split]["num_questions_valid_label"],
                "empty": dataset_stats[split]["num_empty_masks"],
                "fail": dataset_stats[split]["num_load_fail"],
            })

        for k, v in dataset_stats[split].items():
            if isinstance(v, int):
                dataset_stats["overall"][k] += v

    return {
        "stats": dataset_stats,
        "empty_records": empty_records,
        "load_fail_records": load_fail_records,
        "skipped_records": skipped_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/data/SAM3_data",
        help="数据根目录",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AMOS2022", "BraTS", "CHAOS", "CMRxMotions", "COVID19", "Prostate", "SegRap2023"],
        help="要检查的数据集列表",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/work/sam3_med_lora/empty_mask_check_all",
        help="输出目录",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    all_empty_records: List[Dict[str, Any]] = []
    all_load_fail_records: List[Dict[str, Any]] = []
    all_skipped_records: List[Dict[str, Any]] = []
    summary_per_dataset: Dict[str, Any] = {}

    summary_all = {
        "datasets": args.datasets,
        "overall": init_stats(),
    }

    dataset_pbar = tqdm(
        args.datasets,
        desc="Datasets",
        leave=True,
        dynamic_ncols=True,
    )

    for dataset_name in dataset_pbar:
        dataset_root = os.path.join(args.data_root, dataset_name)
        result = process_one_dataset(dataset_root, dataset_name)

        stats = result["stats"]
        summary_per_dataset[dataset_name] = stats

        all_empty_records.extend(result["empty_records"])
        all_load_fail_records.extend(result["load_fail_records"])
        all_skipped_records.extend(result["skipped_records"])

        for k, v in stats["overall"].items():
            if isinstance(v, int):
                summary_all["overall"][k] += v

        dataset_pbar.set_postfix({
            "items": summary_all["overall"]["num_items"],
            "q_total": summary_all["overall"]["num_questions_total"],
            "empty": summary_all["overall"]["num_empty_masks"],
            "fail": summary_all["overall"]["num_load_fail"],
        })

        print(f"\n[{dataset_name}]")
        print(f"  items={stats['overall']['num_items']}")
        print(f"  questions_total={stats['overall']['num_questions_total']}")
        print(f"  valid_label={stats['overall']['num_questions_valid_label']}")
        print(f"  non_empty={stats['overall']['num_non_empty_masks']}")
        print(f"  empty={stats['overall']['num_empty_masks']}")
        print(f"  load_fail={stats['overall']['num_load_fail']}")
        print(f"  skipped_invalid_label={stats['overall']['num_skipped_invalid_label']}")
        print(f"  skipped_background={stats['overall']['num_skipped_background']}")

    save_json(summary_all, os.path.join(args.output_dir, "summary_all.json"))
    save_json(summary_per_dataset, os.path.join(args.output_dir, "summary_per_dataset.json"))
    save_jsonl(all_empty_records, os.path.join(args.output_dir, "empty_mask_records.jsonl"))
    save_jsonl(all_load_fail_records, os.path.join(args.output_dir, "load_fail_records.jsonl"))
    save_jsonl(all_skipped_records, os.path.join(args.output_dir, "skipped_records.jsonl"))

    print("\n[ALL]")
    print(f"  items={summary_all['overall']['num_items']}")
    print(f"  questions_total={summary_all['overall']['num_questions_total']}")
    print(f"  valid_label={summary_all['overall']['num_questions_valid_label']}")
    print(f"  non_empty={summary_all['overall']['num_non_empty_masks']}")
    print(f"  empty={summary_all['overall']['num_empty_masks']}")
    print(f"  load_fail={summary_all['overall']['num_load_fail']}")
    print(f"  skipped_invalid_label={summary_all['overall']['num_skipped_invalid_label']}")
    print(f"  skipped_background={summary_all['overall']['num_skipped_background']}")
    print(f"\n[OUT] {args.output_dir}")


if __name__ == "__main__":
    main()