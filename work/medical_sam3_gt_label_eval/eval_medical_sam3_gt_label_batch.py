# filename: /root/autodl-tmp/work/medical_sam3_gt_label_eval/eval_medical_sam3_gt_label_batch.py
import argparse
import json
import os
from pathlib import Path
import traceback
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from med_data_utils import (
    build_label_name_to_index,
    case_identifier_from_item,
    ensure_dir,
    get_split_items,
    load_dataset_json,
    load_gt_mask_from_npz,
    resolve_abs_path,
    resolve_dataset_json_path,
    union_pred_masks,
)
from medical_sam3_infer import (
    build_medical_sam3_processor,
    medical_sam3_text_inference,
)
from metrics import dice_score, iou_score
from viz import visualize

def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "work").exists():
            return candidate
    for candidate in [start, *start.parents]:
        if (candidate / "work").exists() and (candidate / "repos").exists():
            return candidate
    return start.parents[2]


PROJECT_ROOT = find_project_root(Path(__file__))
DEFAULT_SAM3_REPO_ROOT = PROJECT_ROOT / "repos" / "sam3"


def resolve_cli_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((PROJECT_ROOT / path).resolve())


def _metric_stats(items, metric_key):
    vals = [x[metric_key] for x in items if x.get("status") == "ok" and x.get(metric_key) is not None]
    if len(vals) == 0:
        return {
            f"mean_{metric_key}": None,
            f"max_{metric_key}": None,
            f"min_{metric_key}": None,
        }
    return {
        f"mean_{metric_key}": float(np.mean(vals)),
        f"max_{metric_key}": float(np.max(vals)),
        f"min_{metric_key}": float(np.min(vals)),
    }


def summarize_results(records):
    summary = {
        "num_records": len(records),
        "mean_dice": None,
        "mean_iou": None,
        "max_dice": None,
        "min_dice": None,
        "max_iou": None,
        "min_iou": None,
        "num_failures": 0,
        "num_no_pred_mask": 0,
        "by_label": {},
        "by_turn": {},
        "failure_reasons": {},
        "split_source_counts": {},
    }

    if not records:
        return summary

    valid = [r for r in records if r.get("status") == "ok"]
    failures = [r for r in records if r.get("status") != "ok"]
    summary["num_failures"] = len(failures)
    summary["num_no_pred_mask"] = sum(1 for r in records if r.get("no_pred_mask") is True)

    split_source_counter = defaultdict(int)
    for r in records:
        split_source_counter[r.get("split_source", "unknown")] += 1
    summary["split_source_counts"] = dict(split_source_counter)

    if valid:
        dice_vals = [r["dice"] for r in valid]
        iou_vals = [r["iou"] for r in valid]
        summary["mean_dice"] = float(np.mean(dice_vals))
        summary["mean_iou"] = float(np.mean(iou_vals))
        summary["max_dice"] = float(np.max(dice_vals))
        summary["min_dice"] = float(np.min(dice_vals))
        summary["max_iou"] = float(np.max(iou_vals))
        summary["min_iou"] = float(np.min(iou_vals))

    by_label = defaultdict(list)
    by_turn = defaultdict(list)

    for r in records:
        by_label[r["gt_label"]].append(r)
        by_turn[str(r["turn_idx"])].append(r)

    for label, items in by_label.items():
        block = {
            "count": len(items),
            "num_valid": sum(1 for x in items if x.get("status") == "ok"),
            "num_errors": sum(1 for x in items if x.get("status") != "ok"),
            "num_no_pred_mask": sum(1 for x in items if x.get("no_pred_mask") is True),
        }
        block.update(_metric_stats(items, "dice"))
        block.update(_metric_stats(items, "iou"))
        summary["by_label"][label] = block

    for turn_idx, items in by_turn.items():
        block = {
            "count": len(items),
            "num_valid": sum(1 for x in items if x.get("status") == "ok"),
            "num_errors": sum(1 for x in items if x.get("status") != "ok"),
            "num_no_pred_mask": sum(1 for x in items if x.get("no_pred_mask") is True),
        }
        block.update(_metric_stats(items, "dice"))
        block.update(_metric_stats(items, "iou"))
        summary["by_turn"][turn_idx] = block

    failure_reasons = defaultdict(int)
    for r in failures:
        failure_reasons[r.get("error_type", "unknown_error")] += 1
    summary["failure_reasons"] = dict(failure_reasons)

    return summary


def build_selected_items(dataset_json, split: str, max_samples: int):
    """
    Returns a list of tuples:
        (split_source, item)

    Rules:
    - split=training: use training only
    - split=test: use test only
    - split=all: use training + test
    - if max_samples <= 0: use all items from selected split(s)
    - if split=all and max_samples > 0: use at most max_samples from training and at most max_samples from test
    """
    selected = []

    if split in ("training", "test"):
        items = get_split_items(dataset_json, split)
        if max_samples > 0:
            items = items[:max_samples]
        selected.extend([(split, x) for x in items])
        return selected

    if split == "all":
        for sp in ("training", "test"):
            items = get_split_items(dataset_json, sp)
            if max_samples > 0:
                items = items[:max_samples]
            selected.extend([(sp, x) for x in items])
        return selected

    raise ValueError(f"Unsupported split: {split}")


def overlay_gt_mask_on_image(
    image_path: str,
    gt_mask: np.ndarray,
    color=(255, 0, 0),
    alpha: float = 0.35,
):
    image = Image.open(image_path).convert("RGB")
    img = np.array(image).astype(np.float32)
    mask = gt_mask.astype(bool)

    overlay = np.zeros_like(img)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    img[mask] = img[mask] * (1.0 - alpha) + overlay[mask] * alpha
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _measure_text(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return draw.textsize(text, font=font)


def make_side_by_side_compare_image(
    pred_img: Image.Image,
    gt_img: Image.Image,
    left_caption: str = "predict",
    right_caption: str = "label",
    gt_label_text: str = "",
    gap: int = 20,
    caption_h: int = 52,
    bg_color=(255, 255, 255),
):
    pred_img = pred_img.convert("RGB")
    gt_img = gt_img.convert("RGB")

    w1, h1 = pred_img.size
    w2, h2 = gt_img.size
    max_h = max(h1, h2)

    canvas_w = w1 + gap + w2
    canvas_h = max_h + caption_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    canvas.paste(pred_img, (0, 0))
    canvas.paste(gt_img, (w1 + gap, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # 第一行：predict / label
    tw1, th1 = _measure_text(draw, left_caption, font)
    tw2, th2 = _measure_text(draw, right_caption, font)

    x1 = max(0, (w1 - tw1) // 2)
    x2 = w1 + gap + max(0, (w2 - tw2) // 2)

    y_line1 = max_h + 4
    draw.text((x1, y_line1), left_caption, fill=(0, 0, 0), font=font)
    draw.text((x2, y_line1), right_caption, fill=(0, 0, 0), font=font)

    # 第二行：显示 gt_label 实际内容
    if gt_label_text:
        label_info = f"gt_label: {gt_label_text}"
        tw3, th3 = _measure_text(draw, label_info, font)
        x3 = max(0, (canvas_w - tw3) // 2)
        y_line2 = y_line1 + th1 + 4
        draw.text((x3, y_line2), label_info, fill=(0, 0, 0), font=font)

    return canvas


def evaluate_one_dataset(
    processor,
    data_root: str,
    dataset_name: str,
    split: str,
    max_samples: int,
    output_dir: str,
):
    dataset_root = os.path.join(data_root, dataset_name)
    dataset_json_path = resolve_dataset_json_path(data_root, dataset_name)
    dataset_json = load_dataset_json(dataset_json_path)
    label_name_to_idx = build_label_name_to_index(dataset_json)

    selected_items = build_selected_items(dataset_json, split, max_samples)

    ensure_dir(output_dir)
    per_case_root = os.path.join(output_dir, "per_case")
    ensure_dir(per_case_root)

    results_jsonl_path = os.path.join(output_dir, "results.jsonl")
    records = []

    with open(results_jsonl_path, "w", encoding="utf-8") as fout:
        for sample_idx, (split_source, item) in enumerate(selected_items):
            case_id_base = case_identifier_from_item(item)
            case_id = f"{split_source}__{case_id_base}"

            image_path = resolve_abs_path(dataset_root, item["image"])
            label_npz_path = resolve_abs_path(dataset_root, item["label"])
            questions = item["questions"]

            case_out_dir = os.path.join(per_case_root, case_id)
            ensure_dir(case_out_dir)

            print("\n" + "=" * 80)
            print(
                f"[{dataset_name}] [Sample {sample_idx + 1}/{len(selected_items)}] "
                f"split_source={split_source} case_id={case_id}"
            )
            print(f"image_path={image_path}")
            print(f"label_npz_path={label_npz_path}")
            print("=" * 80)

            for turn_idx, qobj in enumerate(questions):
                gt_label_name = qobj["label"]
                question_id = qobj.get("id", f"{case_id}_turn_{turn_idx + 1}")
                current_question = qobj["question"]
                prompt = gt_label_name

                if gt_label_name not in label_name_to_idx:
                    record = {
                        "status": "error",
                        "error_type": "gt_label_not_found",
                        "dataset": dataset_name,
                        "split": split,
                        "split_source": split_source,
                        "case_id": case_id,
                        "sample_idx": sample_idx,
                        "turn_idx": turn_idx + 1,
                        "question_id": question_id,
                        "gt_label": gt_label_name,
                        "image_path": image_path,
                        "label_npz_path": label_npz_path,
                        "current_question": current_question,
                        "prompt": prompt,
                        "no_pred_mask": False,
                        "error": f"GT label '{gt_label_name}' not found in top-level labels",
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    records.append(record)
                    continue

                gt_channel_idx = label_name_to_idx[gt_label_name]

                try:
                    final_outputs = medical_sam3_text_inference(
                        processor=processor,
                        image_path=image_path,
                        text_prompt=prompt,
                    )

                    height = int(final_outputs["orig_img_h"])
                    width = int(final_outputs["orig_img_w"])
                    pred_mask = union_pred_masks(
                        final_outputs["pred_masks"], height, width
                    )
                    gt_mask = load_gt_mask_from_npz(label_npz_path, gt_channel_idx)

                    if gt_mask.shape != pred_mask.shape:
                        raise ValueError(
                            f"Shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}, "
                            f"case_id={case_id}, turn_idx={turn_idx + 1}"
                        )

                    dice = dice_score(pred_mask, gt_mask)
                    iou = iou_score(pred_mask, gt_mask)
                    num_pred_masks = len(final_outputs["pred_masks"])
                    no_pred_mask = num_pred_masks == 0

                    record = {
                        "status": "ok",
                        "dataset": dataset_name,
                        "split": split,
                        "split_source": split_source,
                        "case_id": case_id,
                        "sample_idx": sample_idx,
                        "turn_idx": turn_idx + 1,
                        "question_id": question_id,
                        "gt_label": gt_label_name,
                        "gt_channel_idx": gt_channel_idx,
                        "image_path": image_path,
                        "label_npz_path": label_npz_path,
                        "current_question": current_question,
                        "prompt": prompt,
                        "num_pred_masks": num_pred_masks,
                        "no_pred_mask": no_pred_mask,
                        "dice": dice,
                        "iou": iou,
                        "pred_scores": final_outputs.get("pred_scores", []),
                        "pred_boxes": final_outputs.get("pred_boxes", []),
                    }

                    save_outputs = dict(final_outputs)
                    save_outputs.update(
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "split_source": split_source,
                            "case_id": case_id,
                            "question_id": question_id,
                            "turn_idx": turn_idx + 1,
                            "gt_label": gt_label_name,
                            "gt_channel_idx": gt_channel_idx,
                            "current_question": current_question,
                            "prompt": prompt,
                            "num_pred_masks": num_pred_masks,
                            "no_pred_mask": no_pred_mask,
                            "dice": dice,
                            "iou": iou,
                        }
                    )

                    turn_debug_json = os.path.join(
                        case_out_dir, f"turn_{turn_idx + 1:02d}_final_outputs.json"
                    )
                    with open(turn_debug_json, "w", encoding="utf-8") as tf:
                        json.dump(save_outputs, tf, indent=2, ensure_ascii=False)

                    pred_vis = visualize(final_outputs)
                    gt_vis = overlay_gt_mask_on_image(image_path, gt_mask)

                    compare_vis = make_side_by_side_compare_image(
                        pred_vis,
                        gt_vis,
                        left_caption="predict",
                        right_caption="label",
                        gt_label_text=gt_label_name,
                    )

                    turn_compare_png = os.path.join(
                        case_out_dir, f"turn_{turn_idx + 1:02d}_pred_vs_label.png"
                    )
                    compare_vis.save(turn_compare_png)

                except Exception as e:
                    record = {
                        "status": "error",
                        "error_type": "runtime_error",
                        "dataset": dataset_name,
                        "split": split,
                        "split_source": split_source,
                        "case_id": case_id,
                        "sample_idx": sample_idx,
                        "turn_idx": turn_idx + 1,
                        "question_id": question_id,
                        "gt_label": gt_label_name,
                        "gt_channel_idx": gt_channel_idx,
                        "image_path": image_path,
                        "label_npz_path": label_npz_path,
                        "current_question": current_question,
                        "prompt": prompt,
                        "no_pred_mask": False,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                records.append(record)

                if record["status"] == "ok":
                    print(
                        f"[OK] dataset={dataset_name} split_source={split_source} "
                        f"case={case_id} turn={turn_idx + 1} "
                        f"label={gt_label_name} prompt={prompt} "
                        f"dice={record['dice']:.4f} iou={record['iou']:.4f} "
                        f"num_pred_masks={record['num_pred_masks']}"
                    )
                else:
                    print(
                        f"[ERROR] dataset={dataset_name} split_source={split_source} "
                        f"case={case_id} turn={turn_idx + 1} "
                        f"label={gt_label_name} prompt={prompt} "
                        f"err={record['error']}"
                    )

    summary = summarize_results(records)
    summary["dataset"] = dataset_name
    summary["split"] = split
    summary["max_samples"] = max_samples

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{dataset_name}] Evaluation done.")
    print(f"Results saved to: {results_jsonl_path}")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return records, summary


def aggregate_dataset_summaries(dataset_summaries):
    agg = {
        "num_datasets": len(dataset_summaries),
        "datasets": {},
        "overall_mean_dice_macro": None,
        "overall_mean_iou_macro": None,
        "overall_max_dice_macro": None,
        "overall_min_dice_macro": None,
        "overall_max_iou_macro": None,
        "overall_min_iou_macro": None,
        "overall_num_no_pred_mask": 0,
    }

    if not dataset_summaries:
        return agg

    mean_dices = []
    mean_ious = []
    max_dices = []
    min_dices = []
    max_ious = []
    min_ious = []

    for ds_name, ds_summary in dataset_summaries.items():
        agg["datasets"][ds_name] = ds_summary

        if ds_summary.get("mean_dice") is not None:
            mean_dices.append(ds_summary["mean_dice"])
        if ds_summary.get("mean_iou") is not None:
            mean_ious.append(ds_summary["mean_iou"])
        if ds_summary.get("max_dice") is not None:
            max_dices.append(ds_summary["max_dice"])
        if ds_summary.get("min_dice") is not None:
            min_dices.append(ds_summary["min_dice"])
        if ds_summary.get("max_iou") is not None:
            max_ious.append(ds_summary["max_iou"])
        if ds_summary.get("min_iou") is not None:
            min_ious.append(ds_summary["min_iou"])

        agg["overall_num_no_pred_mask"] += int(ds_summary.get("num_no_pred_mask", 0))

    if mean_dices:
        agg["overall_mean_dice_macro"] = float(np.mean(mean_dices))
    if mean_ious:
        agg["overall_mean_iou_macro"] = float(np.mean(mean_ious))
    if max_dices:
        agg["overall_max_dice_macro"] = float(np.mean(max_dices))
    if min_dices:
        agg["overall_min_dice_macro"] = float(np.mean(min_dices))
    if max_ious:
        agg["overall_max_iou_macro"] = float(np.mean(max_ious))
    if min_ious:
        agg["overall_min_iou_macro"] = float(np.mean(min_ious))

    return agg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset names, e.g. AMOS2022 BraTS CHAOS",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["training", "test", "all"],
        help="training / test / all",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="If >0, use at most this many items per selected split. If <=0, use all items.",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--sam3_repo_root",
        type=str,
        default=str(DEFAULT_SAM3_REPO_ROOT),
        help="Root path of original sam3 repo",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Medical-SAM3 checkpoint.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda / cpu",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold passed to Sam3Processor",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root = resolve_cli_path(args.data_root)
    args.output_dir = resolve_cli_path(args.output_dir)
    args.sam3_repo_root = resolve_cli_path(args.sam3_repo_root)
    args.checkpoint = resolve_cli_path(args.checkpoint)
    ensure_dir(args.output_dir)

    processor = build_medical_sam3_processor(
        sam3_repo_root=args.sam3_repo_root,
        checkpoint_path=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    dataset_summaries = {}
    max_tag = "all" if args.max_samples <= 0 else str(args.max_samples)

    for dataset_name in args.datasets:
        dataset_out_dir = os.path.join(
            args.output_dir,
            f"{dataset_name}_{args.split}_{max_tag}",
        )
        ensure_dir(dataset_out_dir)

        _, ds_summary = evaluate_one_dataset(
            processor=processor,
            data_root=args.data_root,
            dataset_name=dataset_name,
            split=args.split,
            max_samples=args.max_samples,
            output_dir=dataset_out_dir,
        )
        dataset_summaries[dataset_name] = ds_summary

    aggregate_summary = aggregate_dataset_summaries(dataset_summaries)
    aggregate_summary["split"] = args.split
    aggregate_summary["max_samples"] = args.max_samples
    aggregate_summary["checkpoint"] = args.checkpoint

    aggregate_summary_path = os.path.join(args.output_dir, "all_datasets_summary.json")
    with open(aggregate_summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2, ensure_ascii=False)

    print("\nAll dataset evaluations done.")
    print(f"Aggregate summary saved to: {aggregate_summary_path}")
    print(json.dumps(aggregate_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
