# filename: compare_sam3_medical_results.py
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ========= 用户可修改路径 =========
MODEL_FILES = {
    "Medical-SAM3": str(PROJECT_ROOT / "work" / "medical_sam3_gt_label_eval" / "outputs" / "multi_datasets_test_0" / "all_datasets_summary.json"),
    "MedSAM3-LoRA": str(PROJECT_ROOT / "work" / "medsam3_lora_gt_label_eval" / "outputs" / "all_datasets_summary.json"),
    "SAM3": str(PROJECT_ROOT / "work" / "sam3_gt_label_eval" / "outputs" / "multi_datasets_test_0" / "all_datasets_summary.json"),
}
OUTPUT_DIR = str(PROJECT_ROOT / "work" / "three_model_anysis")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, key, default=np.nan):
    v = d.get(key, default)
    return np.nan if v is None else v


def clean_model_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def save_fig(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_overall_df(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in results.items():
        total_records = sum(ds.get("num_records", 0) for ds in data["datasets"].values())
        total_failures = sum(ds.get("num_failures", 0) for ds in data["datasets"].values())
        total_no_pred = sum(ds.get("num_no_pred_mask", 0) for ds in data["datasets"].values())

        rows.append(
            {
                "model": model_name,
                "overall_mean_dice_macro": safe_get(data, "overall_mean_dice_macro"),
                "overall_mean_iou_macro": safe_get(data, "overall_mean_iou_macro"),
                "overall_num_no_pred_mask": safe_get(data, "overall_num_no_pred_mask", 0),
                "total_records": total_records,
                "total_failures": total_failures,
                "no_pred_rate": total_no_pred / total_records if total_records else np.nan,
                "failure_rate": total_failures / total_records if total_records else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("overall_mean_dice_macro", ascending=False)


def build_dataset_df(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in results.items():
        for ds_name, ds in data["datasets"].items():
            n = ds.get("num_records", 0)
            rows.append(
                {
                    "model": model_name,
                    "dataset": ds_name,
                    "num_records": n,
                    "mean_dice": safe_get(ds, "mean_dice"),
                    "mean_iou": safe_get(ds, "mean_iou"),
                    "num_failures": ds.get("num_failures", 0),
                    "num_no_pred_mask": ds.get("num_no_pred_mask", 0),
                    "no_pred_rate": ds.get("num_no_pred_mask", 0) / n if n else np.nan,
                    "failure_rate": ds.get("num_failures", 0) / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_turn_df(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in results.items():
        # 先汇总所有 dataset 的同一 turn
        turn_acc = {}
        for _, ds in data["datasets"].items():
            for turn_str, info in ds.get("by_turn", {}).items():
                turn = int(turn_str)
                turn_acc.setdefault(
                    turn,
                    {
                        "count": 0,
                        "num_valid": 0,
                        "num_errors": 0,
                        "num_no_pred_mask": 0,
                        "dice_weighted_sum": 0.0,
                        "iou_weighted_sum": 0.0,
                    },
                )
                acc = turn_acc[turn]
                c = info.get("count", 0)
                v = info.get("num_valid", 0)
                acc["count"] += c
                acc["num_valid"] += v
                acc["num_errors"] += info.get("num_errors", 0)
                acc["num_no_pred_mask"] += info.get("num_no_pred_mask", 0)
                if info.get("mean_dice") is not None:
                    acc["dice_weighted_sum"] += info["mean_dice"] * v
                if info.get("mean_iou") is not None:
                    acc["iou_weighted_sum"] += info["mean_iou"] * v

        for turn, acc in sorted(turn_acc.items()):
            v = acc["num_valid"]
            c = acc["count"]
            rows.append(
                {
                    "model": model_name,
                    "turn": turn,
                    "count": c,
                    "num_valid": v,
                    "num_errors": acc["num_errors"],
                    "num_no_pred_mask": acc["num_no_pred_mask"],
                    "mean_dice_weighted": acc["dice_weighted_sum"] / v if v else np.nan,
                    "mean_iou_weighted": acc["iou_weighted_sum"] / v if v else np.nan,
                    "no_pred_rate": acc["num_no_pred_mask"] / c if c else np.nan,
                    "failure_rate": acc["num_errors"] / c if c else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_label_df(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in results.items():
        for ds_name, ds in data["datasets"].items():
            for label, info in ds.get("by_label", {}).items():
                if label == "单标签追问":
                    continue
                count = info.get("count", 0)
                rows.append(
                    {
                        "model": model_name,
                        "dataset": ds_name,
                        "label": label,
                        "count": count,
                        "num_valid": info.get("num_valid", 0),
                        "num_errors": info.get("num_errors", 0),
                        "num_no_pred_mask": info.get("num_no_pred_mask", 0),
                        "mean_dice": safe_get(info, "mean_dice"),
                        "mean_iou": safe_get(info, "mean_iou"),
                        "no_pred_rate": info.get("num_no_pred_mask", 0) / count if count else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def plot_overall_bars(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df["overall_mean_dice_macro"], width, label="Macro Dice")
    ax.bar(x + width / 2, df["overall_mean_iou_macro"], width, label="Macro IoU")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Overall Macro Dice / IoU Comparison")
    ax.legend()
    for i, v in enumerate(df["overall_mean_dice_macro"]):
        ax.text(i - width / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(df["overall_mean_iou_macro"]):
        ax.text(i + width / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    save_fig(fig, output_dir / "01_overall_macro_dice_iou.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df["no_pred_rate"], width, label="No-pred rate")
    ax.bar(x + width / 2, df["failure_rate"], width, label="Failure rate")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("Rate")
    ax.set_title("Overall No-pred / Failure Rate Comparison")
    ax.legend()
    for i, v in enumerate(df["no_pred_rate"]):
        ax.text(i - width / 2, v + 0.003, f"{v:.2%}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(df["failure_rate"]):
        ax.text(i + width / 2, v + 0.003, f"{v:.2%}", ha="center", va="bottom", fontsize=9)
    save_fig(fig, output_dir / "02_overall_no_pred_failure_rate.png")


def plot_dataset_heatmaps(dataset_df: pd.DataFrame, output_dir: Path):
    for metric, fname, title in [
        ("mean_dice", "03_dataset_mean_dice_heatmap.png", "Dataset-level Mean Dice"),
        ("mean_iou", "04_dataset_mean_iou_heatmap.png", "Dataset-level Mean IoU"),
        ("no_pred_rate", "05_dataset_no_pred_rate_heatmap.png", "Dataset-level No-pred Rate"),
    ]:
        pivot = dataset_df.pivot(index="dataset", columns="model", values=metric)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.65 * len(pivot))))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=20)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric, rotation=270, labelpad=15)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                txt = "nan" if pd.isna(val) else (f"{val:.3f}" if metric != "no_pred_rate" else f"{val:.1%}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        save_fig(fig, output_dir / fname)


def plot_dataset_grouped_bars(dataset_df: pd.DataFrame, output_dir: Path):
    datasets = sorted(dataset_df["dataset"].unique())
    models = list(dataset_df["model"].unique())
    x = np.arange(len(datasets))
    width = 0.24

    for metric, fname, title, ylabel in [
        ("mean_dice", "06_dataset_grouped_mean_dice.png", "Mean Dice by Dataset", "Mean Dice"),
        ("mean_iou", "07_dataset_grouped_mean_iou.png", "Mean IoU by Dataset", "Mean IoU"),
        ("no_pred_rate", "08_dataset_grouped_no_pred_rate.png", "No-pred Rate by Dataset", "No-pred Rate"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for idx, model in enumerate(models):
            vals = []
            for ds in datasets:
                sub = dataset_df[(dataset_df["dataset"] == ds) & (dataset_df["model"] == model)]
                vals.append(sub.iloc[0][metric] if len(sub) else np.nan)
            ax.bar(x + (idx - (len(models)-1)/2) * width, vals, width, label=model)
            for j, v in enumerate(vals):
                if pd.notna(v):
                    label = f"{v:.3f}" if metric != "no_pred_rate" else f"{v:.1%}"
                    ax.text(x[j] + (idx - (len(models)-1)/2) * width, v + (0.005 if metric != "no_pred_rate" else 0.003), label,
                            ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        save_fig(fig, output_dir / fname)


def plot_turn_trends(turn_df: pd.DataFrame, output_dir: Path):
    for metric, fname, title, ylabel in [
        ("mean_dice_weighted", "09_turn_mean_dice_trend.png", "Turn-wise Weighted Mean Dice", "Weighted Mean Dice"),
        ("mean_iou_weighted", "10_turn_mean_iou_trend.png", "Turn-wise Weighted Mean IoU", "Weighted Mean IoU"),
        ("no_pred_rate", "11_turn_no_pred_rate_trend.png", "Turn-wise No-pred Rate", "No-pred Rate"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in turn_df["model"].unique():
            sub = turn_df[turn_df["model"] == model].sort_values("turn")
            ax.plot(sub["turn"], sub[metric], marker="o", label=model)
            for _, row in sub.iterrows():
                val = row[metric]
                if pd.notna(val):
                    txt = f"{val:.3f}" if metric != "no_pred_rate" else f"{val:.1%}"
                    ax.text(row["turn"], val, txt, fontsize=7, ha="center", va="bottom")
        ax.set_xlabel("Turn")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(sorted(turn_df["turn"].unique()))
        save_fig(fig, output_dir / fname)


def plot_label_delta(label_df: pd.DataFrame, output_dir: Path, base_model="SAM3", compare_model="MedSAM3-LoRA", topk=20):
    pivot = label_df.pivot_table(
        index=["dataset", "label", "count"],
        columns="model",
        values="mean_dice",
        aggfunc="mean"
    ).reset_index()

    if base_model not in pivot.columns or compare_model not in pivot.columns:
        return

    pivot["delta"] = pivot[compare_model] - pivot[base_model]
    pivot["dataset_label"] = pivot["dataset"] + " / " + pivot["label"]

    top_gain = pivot.sort_values("delta", ascending=False).head(topk)
    top_drop = pivot.sort_values("delta", ascending=True).head(topk)

    safe_base = base_model.replace("/", "_").replace(" ", "_")
    safe_compare = compare_model.replace("/", "_").replace(" ", "_")

    for df_plot, fname, title in [
        (
            top_gain.sort_values("delta", ascending=True),
            f"12_top_gain_{safe_compare}_vs_{safe_base}.png",
            f"Top {topk} Label Gains: {compare_model} - {base_model}"
        ),
        (
            top_drop.sort_values("delta", ascending=True),
            f"13_top_drop_{safe_compare}_vs_{safe_base}.png",
            f"Top {topk} Label Drops: {compare_model} - {base_model}"
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(df_plot))))
        ax.barh(df_plot["dataset_label"], df_plot["delta"])
        ax.set_xlabel("Dice Delta")
        ax.set_title(title)
        for i, v in enumerate(df_plot["delta"]):
            ax.text(v, i, f"{v:+.3f}", va="center", fontsize=8)
        save_fig(fig, output_dir / fname)


def build_markdown_report(overall_df, dataset_df, turn_df, output_dir: Path):
    best_dice_model = overall_df.sort_values("overall_mean_dice_macro", ascending=False).iloc[0]["model"]
    best_iou_model = overall_df.sort_values("overall_mean_iou_macro", ascending=False).iloc[0]["model"]
    lowest_no_pred_model = overall_df.sort_values("no_pred_rate", ascending=True).iloc[0]["model"]

    dataset_pivot = dataset_df.pivot(index="dataset", columns="model", values="mean_dice")
    dataset_best = dataset_pivot.idxmax(axis=1)

    # 轮次趋势简述：比较首轮和末轮
    trend_lines = []
    for model in turn_df["model"].unique():
        sub = turn_df[turn_df["model"] == model].sort_values("turn")
        if len(sub) >= 2:
            first = sub.iloc[0]["mean_dice_weighted"]
            last = sub.iloc[-1]["mean_dice_weighted"]
            delta = last - first
            trend_lines.append(f"- {model}: 首轮 {first:.3f}，末轮 {last:.3f}，变化 {delta:+.3f}")

    report = f"""# SAM3 医学分割三模型对比报告

## 1. 总体结论
- **总体 Macro Dice 最优**：{best_dice_model}
- **总体 Macro IoU 最优**：{best_iou_model}
- **No-pred rate 最低**：{lowest_no_pred_model}

## 2. 建议重点看的指标
本次对比建议优先看四类指标：
1. **Overall Macro Dice / Macro IoU**：衡量总体分割质量。
2. **Dataset-level Mean Dice / Mean IoU**：衡量跨数据集泛化能力。
3. **No-pred rate / Failure rate**：衡量模型稳定性与“完全没打出来”的风险。
4. **Turn-wise trend**：衡量多轮场景下，随着轮次加深是否明显退化。

## 3. 各数据集 Dice 最优模型
{dataset_best.to_string()}

## 4. 轮次趋势概览
{chr(10).join(trend_lines)}

## 5. 图表说明
- 01/02：总体对比
- 03/04/05：热力图，适合快速看哪一类数据集差距最大
- 06/07/08：分数据集柱状图，适合写论文/汇报
- 09/10/11：轮次趋势图，适合分析多轮推理场景退化
- 12/13：标签级增益/退化最明显的类别，适合做 case study
"""
    report_path = output_dir / "README_compare_report.md"
    report_path.write_text(report, encoding="utf-8")


def main():
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    results = {}
    for model_name, file_path in MODEL_FILES.items():
        results[model_name] = load_json(file_path)

    overall_df = build_overall_df(results)
    dataset_df = build_dataset_df(results)
    turn_df = build_turn_df(results)
    label_df = build_label_df(results)

    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    dataset_df.to_csv(output_dir / "dataset_metrics.csv", index=False, encoding="utf-8-sig")
    turn_df.to_csv(output_dir / "turn_metrics.csv", index=False, encoding="utf-8-sig")
    label_df.to_csv(output_dir / "label_metrics.csv", index=False, encoding="utf-8-sig")

    plot_overall_bars(overall_df, output_dir)
    plot_dataset_heatmaps(dataset_df, output_dir)
    plot_dataset_grouped_bars(dataset_df, output_dir)
    plot_turn_trends(turn_df, output_dir)
    plot_label_delta(label_df, output_dir, base_model="SAM3", compare_model="MedSAM3-LoRA", topk=20)
    plot_label_delta(label_df, output_dir, base_model="SAM3", compare_model="Medical-SAM3", topk=20)

    build_markdown_report(overall_df, dataset_df, turn_df, output_dir)

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
