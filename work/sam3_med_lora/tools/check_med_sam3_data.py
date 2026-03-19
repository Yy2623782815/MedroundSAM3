# filename: /root/autodl-tmp/work/sam3_med_lora/tools/check_med_sam3_data.py
import os
from pathlib import Path
import sys
import json
import cv2
import argparse
import random
from typing import Any, Dict, List, Tuple

import numpy as np

# 让脚本能找到你的 med_data_utils.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from med_data_utils import (  # noqa: E402
    load_dataset_json,
    build_label_name_to_index,
    load_gt_mask_from_npz,
)


def normalize_to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    image = normalize_to_bgr(image.copy())
    out = image.copy()

    mask = (mask > 0).astype(np.uint8)
    color_layer = np.zeros_like(out, dtype=np.uint8)
    color_layer[mask > 0] = color

    out = cv2.addWeighted(out, 1.0, color_layer, alpha, 0)
    return out


def draw_text(img: np.ndarray, text: str, x: int = 8, y: int = 24) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_color(idx: int) -> Tuple[int, int, int]:
    # 固定调色板，便于看图
    palette = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (128, 0, 255),
        (255, 128, 0),
        (128, 255, 0),
    ]
    return palette[idx % len(palette)]


def build_valid_question_records(
    dataset_json: Dict[str, Any],
    item: Dict[str, Any],
    dataset_root: str,
) -> List[Dict[str, Any]]:
    """
    按你的真实规则，从当前 item 的 questions 中筛出“可用于检查”的标签：
    - 必须在 labels 映射里
    - 不是背景
    - 对应 mask 在当前切片非空
    """
    label_name_to_idx = build_label_name_to_index(dataset_json)

    label_npz_path = os.path.join(dataset_root, item["label"])
    records = []

    for q in item.get("questions", []):
        label_name = q["label"].strip()

        # 跳过伪标签
        if label_name not in label_name_to_idx:
            continue

        channel_idx = label_name_to_idx[label_name]

        # 跳过背景
        if channel_idx == 0:
            continue

        try:
            mask = load_gt_mask_from_npz(label_npz_path, channel_idx)
        except Exception as e:
            records.append({
                "question_id": q.get("id", ""),
                "label_name": label_name,
                "channel_idx": channel_idx,
                "valid": False,
                "reason": f"load_failed: {e}",
            })
            continue

        nonzero = int((mask > 0).sum())
        if nonzero <= 0:
            records.append({
                "question_id": q.get("id", ""),
                "label_name": label_name,
                "channel_idx": channel_idx,
                "valid": False,
                "reason": "empty_mask",
            })
            continue

        records.append({
            "question_id": q.get("id", ""),
            "label_name": label_name,
            "channel_idx": channel_idx,
            "valid": True,
            "reason": "ok",
            "mask": mask,
            "nonzero": nonzero,
            "question": q.get("question", ""),
        })

    return records


def save_one_case_visuals(
    dataset_json: Dict[str, Any],
    item: Dict[str, Any],
    dataset_root: str,
    output_dir: str,
    max_questions: int,
    seed: int,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(dataset_root, item["image"])
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"读取图像失败: {image_path}")
    img = normalize_to_bgr(img)

    records = build_valid_question_records(dataset_json, item, dataset_root)

    valid_records = [r for r in records if r["valid"]]
    invalid_records = [r for r in records if not r["valid"]]

    rng = random.Random(seed)
    rng.shuffle(valid_records)
    picked = valid_records[:max_questions]

    base_name = os.path.splitext(os.path.basename(item["image"]))[0]

    # 保存单标签叠加图
    saved_files = []
    for i, r in enumerate(picked):
        color = make_color(i)
        vis = overlay_mask(img, r["mask"], color=color, alpha=0.45)
        vis = draw_text(
            vis,
            f'label={r["label_name"]} ch={r["channel_idx"]} nz={r["nonzero"]}',
        )

        out_name = f"{base_name}__{r['question_id']}__ch{r['channel_idx']}__{r['label_name']}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, vis)
        saved_files.append(out_path)

    # 保存一个多标签总览图
    if len(picked) > 0:
        merged = img.copy()
        legend_lines = []
        for i, r in enumerate(picked):
            color = make_color(i)
            merged = overlay_mask(merged, r["mask"], color=color, alpha=0.35)
            legend_lines.append(f'{r["label_name"]}(ch={r["channel_idx"]}, nz={r["nonzero"]})')

        y = 24
        for line in legend_lines[:8]:
            cv2.putText(
                merged,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 22

        merged_path = os.path.join(output_dir, f"{base_name}__merged.png")
        cv2.imwrite(merged_path, merged)
        saved_files.append(merged_path)

    summary = {
        "image": item["image"],
        "label": item["label"],
        "class_ids": item.get("class_ids", []),
        "num_questions": len(item.get("questions", [])),
        "num_valid_questions": len(valid_records),
        "num_invalid_questions": len(invalid_records),
        "valid_records": [
            {
                "question_id": r["question_id"],
                "label_name": r["label_name"],
                "channel_idx": r["channel_idx"],
                "nonzero": r["nonzero"],
            }
            for r in valid_records
        ],
        "invalid_records": invalid_records,
        "saved_files": saved_files,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="例如: data/SAM3_data/AMOS2022 或绝对路径")
    parser.add_argument("--json_name", type=str, required=True,
                        help="如 MultiEN_AMOS2022.json")
    parser.add_argument("--split", type=str, default="training", choices=["training", "test"])
    parser.add_argument("--num_cases", type=int, default=20)
    parser.add_argument("--max_questions", type=int, default=3,
                        help="每个 case 最多随机可视化多少个有效标签")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)

    json_path = os.path.join(args.dataset_root, args.json_name)
    dataset_json = load_dataset_json(json_path)
    items = dataset_json[args.split]

    sample_items = random.sample(items, min(args.num_cases, len(items)))
    os.makedirs(args.output_dir, exist_ok=True)

    all_summaries = []
    success = 0

    for idx, item in enumerate(sample_items):
        try:
            summary = save_one_case_visuals(
                dataset_json=dataset_json,
                item=item,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                max_questions=args.max_questions,
                seed=args.seed + idx,
            )
            all_summaries.append(summary)

            print(f'[OK] image={summary["image"]}')
            print(f'     class_ids={summary["class_ids"]}')
            print(f'     valid={summary["num_valid_questions"]} invalid={summary["num_invalid_questions"]}')

            for r in summary["valid_records"][:10]:
                print(
                    f'     [VALID] qid={r["question_id"]} '
                    f'label={r["label_name"]} ch={r["channel_idx"]} nz={r["nonzero"]}'
                )

            for r in summary["invalid_records"][:10]:
                print(
                    f'     [SKIP] qid={r.get("question_id","")} '
                    f'label={r.get("label_name","")} '
                    f'ch={r.get("channel_idx","")} '
                    f'reason={r.get("reason","")}'
                )

            success += 1

        except Exception as e:
            print(f'[ERROR] image={item.get("image")} err={e}')

    summary_path = os.path.join(args.output_dir, "check_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] success={success}/{len(sample_items)}")
    print(f"[VIS DIR] {args.output_dir}")
    print(f"[SUMMARY JSON] {summary_path}")


if __name__ == "__main__":
    main()