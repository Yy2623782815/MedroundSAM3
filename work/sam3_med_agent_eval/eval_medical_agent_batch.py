# filename: /root/autodl-tmp/work/sam3_med_agent_eval/eval_medical_agent_batch.py
import argparse
import json
import os
import traceback
from collections import defaultdict
from functools import partial

import numpy as np
import pycocotools.mask as mask_utils  # noqa: F401
import torch

from agent.agent_core import agent_inference
from agent.client_llm import send_generate_request as send_generate_request_orig
from agent.client_sam3 import call_sam_service as call_sam_service_orig

from med_data_utils import (
    build_history_text_from_questions,
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
from metrics import dice_score, iou_score


def build_sam3_processor():
    """
    Build processor using the same route as your working notebook.
    """
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    _ = sam3  # silence linter

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    bpe_path = "/root/autodl-tmp/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    ckpt_path = "/root/autodl-tmp/models/sam3_base/sam3.pt"

    sam3_model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
    )
    processor = Sam3Processor(sam3_model, confidence_threshold=0.5)
    return processor


def build_bound_tools(llm_server_url: str, llm_model: str, llm_api_key: str):
    sam3_processor = build_sam3_processor()

    send_generate_request = partial(
        send_generate_request_orig,
        server_url=llm_server_url,
        model=llm_model,
        api_key=llm_api_key,
    )
    call_sam_service = partial(
        call_sam_service_orig,
        sam3_processor=sam3_processor,
    )
    return send_generate_request, call_sam_service


def summarize_results(records):
    summary = {
        "num_records": len(records),
        "mean_dice": None,
        "mean_iou": None,
        "by_label": {},
        "by_turn": {},
        "num_failures": 0,
        "failure_reasons": {},
    }

    if len(records) == 0:
        return summary

    valid = [r for r in records if r.get("status") == "ok"]
    failures = [r for r in records if r.get("status") != "ok"]
    summary["num_failures"] = len(failures)

    if len(valid) > 0:
        summary["mean_dice"] = float(np.mean([r["dice"] for r in valid]))
        summary["mean_iou"] = float(np.mean([r["iou"] for r in valid]))

        by_label = defaultdict(list)
        by_turn = defaultdict(list)

        for r in valid:
            by_label[r["gt_label"]].append(r)
            by_turn[r["turn_idx"]].append(r)

        for label, items in by_label.items():
            summary["by_label"][label] = {
                "count": len(items),
                "mean_dice": float(np.mean([x["dice"] for x in items])),
                "mean_iou": float(np.mean([x["iou"] for x in items])),
            }

        for turn_idx, items in by_turn.items():
            summary["by_turn"][str(turn_idx)] = {
                "count": len(items),
                "mean_dice": float(np.mean([x["dice"] for x in items])),
                "mean_iou": float(np.mean([x["iou"] for x in items])),
            }

    failure_reasons = defaultdict(int)
    for r in failures:
        failure_reasons[r.get("error_type", "unknown_error")] += 1
    summary["failure_reasons"] = dict(failure_reasons)

    return summary


def evaluate_dataset(
    data_root: str,
    dataset_name: str,
    split: str,
    max_samples: int,
    output_dir: str,
    send_generate_request,
    call_sam_service,
    debug: bool,
    max_agent_rounds: int,
):
    dataset_root = os.path.join(data_root, dataset_name)
    dataset_json_path = resolve_dataset_json_path(data_root, dataset_name)
    dataset_json = load_dataset_json(dataset_json_path)
    label_name_to_idx = build_label_name_to_index(dataset_json)

    split_items = get_split_items(dataset_json, split)
    if max_samples is not None and max_samples > 0:
        split_items = split_items[:max_samples]

    ensure_dir(output_dir)
    results_jsonl_path = os.path.join(output_dir, "results.jsonl")
    debug_case_root = os.path.join(output_dir, "per_case")
    ensure_dir(debug_case_root)

    records = []

    with open(results_jsonl_path, "w", encoding="utf-8") as fout:
        for sample_idx, item in enumerate(split_items):
            case_id = case_identifier_from_item(item)
            image_path = resolve_abs_path(dataset_root, item["image"])
            label_npz_path = resolve_abs_path(dataset_root, item["label"])
            questions = item["questions"]

            case_out_dir = os.path.join(debug_case_root, case_id)
            ensure_dir(case_out_dir)

            print(f"\n{'=' * 80}")
            print(f"[Sample {sample_idx + 1}/{len(split_items)}] case_id={case_id}")
            print(f"image_path={image_path}")
            print(f"label_npz_path={label_npz_path}")
            print(f"{'=' * 80}\n")

            for turn_idx, qobj in enumerate(questions):
                current_question = qobj["question"]
                gt_label_name = qobj["label"]
                question_id = qobj.get("id", f"{case_id}_turn_{turn_idx + 1}")

                history_text = build_history_text_from_questions(questions, turn_idx)

                if gt_label_name not in label_name_to_idx:
                    record = {
                        "status": "error",
                        "error_type": "gt_label_not_found",
                        "error": f"GT label '{gt_label_name}' not found in top-level labels",
                        "dataset": dataset_name,
                        "split": split,
                        "case_id": case_id,
                        "sample_idx": sample_idx,
                        "turn_idx": turn_idx + 1,
                        "question_id": question_id,
                        "gt_label": gt_label_name,
                        "image_path": image_path,
                        "label_npz_path": label_npz_path,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    records.append(record)
                    continue

                gt_channel_idx = label_name_to_idx[gt_label_name]

                try:
                    agent_history, final_outputs, _ = agent_inference(
                        img_path=image_path,
                        initial_text_prompt=current_question,
                        history_text=history_text,
                        debug=debug,
                        send_generate_request=send_generate_request,
                        call_sam_service=call_sam_service,
                        max_generations=max_agent_rounds,   # 关键改动：最多 6 轮
                        output_dir=case_out_dir,
                    )

                    num_pred_masks = len(final_outputs.get("pred_masks", []))

                    # 关键改动：如果最终没有分割对象输出，则判定失败
                    if num_pred_masks == 0:
                        raise RuntimeError(
                            f"No predicted masks after at most {max_agent_rounds} agent rounds"
                        )

                    height = int(final_outputs["orig_img_h"])
                    width = int(final_outputs["orig_img_w"])
                    pred_mask = union_pred_masks(final_outputs["pred_masks"], height, width)
                    gt_mask = load_gt_mask_from_npz(label_npz_path, gt_channel_idx)

                    if gt_mask.shape != pred_mask.shape:
                        raise ValueError(
                            f"Shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}, "
                            f"case_id={case_id}, turn_idx={turn_idx + 1}"
                        )

                    dice = dice_score(pred_mask, gt_mask)
                    iou = iou_score(pred_mask, gt_mask)

                    record = {
                        "status": "ok",
                        "dataset": dataset_name,
                        "split": split,
                        "case_id": case_id,
                        "sample_idx": sample_idx,
                        "turn_idx": turn_idx + 1,
                        "question_id": question_id,
                        "gt_label": gt_label_name,
                        "gt_channel_idx": gt_channel_idx,
                        "image_path": image_path,
                        "label_npz_path": label_npz_path,
                        "current_question": current_question,
                        "history_text": history_text,
                        "num_pred_masks": num_pred_masks,
                        "dice": dice,
                        "iou": iou,
                        "pred_scores": final_outputs.get("pred_scores", []),
                        "pred_boxes": final_outputs.get("pred_boxes", []),
                        "max_agent_rounds": max_agent_rounds,
                    }

                    turn_debug_json = os.path.join(case_out_dir, f"turn_{turn_idx + 1:02d}_final_outputs.json")
                    with open(turn_debug_json, "w", encoding="utf-8") as tf:
                        json.dump(final_outputs, tf, indent=2, ensure_ascii=False)

                    turn_history_json = os.path.join(case_out_dir, f"turn_{turn_idx + 1:02d}_agent_history.json")
                    with open(turn_history_json, "w", encoding="utf-8") as hf:
                        json.dump(agent_history, hf, indent=2, ensure_ascii=False)

                except Exception as e:
                    error_msg = str(e)
                    if "Exceeded maximum number of allowed generation requests" in error_msg:
                        error_type = "max_agent_rounds_exceeded"
                    elif "No predicted masks after at most" in error_msg:
                        error_type = "no_pred_mask"
                    else:
                        error_type = "runtime_error"

                    record = {
                        "status": "error",
                        "error_type": error_type,
                        "dataset": dataset_name,
                        "split": split,
                        "case_id": case_id,
                        "sample_idx": sample_idx,
                        "turn_idx": turn_idx + 1,
                        "question_id": question_id,
                        "gt_label": gt_label_name,
                        "gt_channel_idx": gt_channel_idx,
                        "image_path": image_path,
                        "label_npz_path": label_npz_path,
                        "current_question": current_question,
                        "history_text": history_text,
                        "max_agent_rounds": max_agent_rounds,
                        "error": error_msg,
                        "traceback": traceback.format_exc(),
                    }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                records.append(record)

                if record["status"] == "ok":
                    print(
                        f"[OK] case={case_id} turn={turn_idx + 1} "
                        f"label={gt_label_name} dice={record['dice']:.4f} "
                        f"iou={record['iou']:.4f} num_pred_masks={record['num_pred_masks']}"
                    )
                else:
                    print(
                        f"[ERROR] case={case_id} turn={turn_idx + 1} "
                        f"label={gt_label_name} error_type={record['error_type']} err={record['error']}"
                    )

    summary = summarize_results(records)
    summary["max_agent_rounds"] = max_agent_rounds

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEvaluation done.")
    print(f"Results saved to: {results_jsonl_path}")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return records, summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AMOS2022")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--llm_server_url", type=str, default="http://0.0.0.0:8001/v1")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--llm_api_key", type=str, default="DUMMY_API_KEY")

    parser.add_argument("--max_agent_rounds", type=int, default=6)  # 新增参数
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(args.output_dir)

    send_generate_request, call_sam_service = build_bound_tools(
        llm_server_url=args.llm_server_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
    )

    evaluate_dataset(
        data_root=args.data_root,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        send_generate_request=send_generate_request,
        call_sam_service=call_sam_service,
        debug=args.debug,
        max_agent_rounds=args.max_agent_rounds,
    )


if __name__ == "__main__":
    main()