# medsam3_my_lora_gt_label_eval

面向 `work/medsam3_my_lora` 训练产物的 **gt label name prompt** 评测工程。

该工程参考 `work/medsam3_lora_gt_label_eval` 的目录风格、运行逻辑和结果组织方式，
并把权重加载逻辑适配到了 `train_medsam3_my_lora.py` 产出的 `best.pt`（含 `lora_state_dict`）。

---

## 1. 目录结构

```text
work/medsam3_my_lora_gt_label_eval/
├── README.md
├── eval_medsam3_my_lora_gt_label_batch.py      # 多数据集批量评测主脚本
├── medsam3_my_lora_infer.py                    # SAM3 + my_lora 权重加载与单条 prompt 推理
├── med_data_utils.py                           # 数据路径、GT npz 读取、pred mask union
├── metrics.py                                  # Dice / IoU 指标
├── viz.py                                      # 预测可视化
└── scripts/
    └── run_eval_medsam3_my_lora_multi_datasets.sh  # 一键运行脚本
```

---

## 2. 依赖关系

- Python: 3.10+
- PyTorch / torchvision
- numpy / pillow / pyyaml / tqdm
- pycocotools（RLE 掩码兼容）
- scipy（仅读稀疏 GT npz 时需要）

以及本地项目依赖：
- `repos/MedSAM3`（SAM3 模型与 transform/collator）
- `work/medsam3_my_lora`（LoRA 注入与权重加载实现）

> 说明：默认按本地权重加载（`--load_from_hf` 不开启）。

---

## 3. 运行方式

### 3.1 一键脚本（推荐）

```bash
bash /root/autodl-tmp/work/medsam3_my_lora_gt_label_eval/scripts/run_eval_medsam3_my_lora_multi_datasets.sh
```

可通过环境变量覆盖默认参数，例如：

```bash
DATA_ROOT=/root/autodl-tmp/data/SAM3_data \
OUTPUT_ROOT=/root/autodl-tmp/work/medsam3_my_lora_gt_label_eval/outputs \
SPLIT=test \
MAX_SAMPLES=20 \
DATASETS="CHAOS BraTS" \
LORA_CHECKPOINT_PATH=/root/autodl-tmp/work/medsam3_my_lora/outputs/chaos_smoke/checkpoints/best.pt \
bash /root/autodl-tmp/work/medsam3_my_lora_gt_label_eval/scripts/run_eval_medsam3_my_lora_multi_datasets.sh
```

如果你想自动评测 `data_root` 下全部可发现数据集：

```bash
USE_ALL_DATASETS=1 \
bash /root/autodl-tmp/work/medsam3_my_lora_gt_label_eval/scripts/run_eval_medsam3_my_lora_multi_datasets.sh
```

### 3.2 直接 Python 命令

```bash
cd /root/autodl-tmp/work/medsam3_my_lora_gt_label_eval
python eval_medsam3_my_lora_gt_label_batch.py \
  --data_root /root/autodl-tmp/data/SAM3_data \
  --datasets CHAOS BraTS \
  --split test \
  --max_samples 0 \
  --output_dir /root/autodl-tmp/work/medsam3_my_lora_gt_label_eval/outputs \
  --sam3_repo_root /root/autodl-tmp/repos/MedSAM3 \
  --my_lora_project_root /root/autodl-tmp/work/medsam3_my_lora \
  --lora_checkpoint_path /root/autodl-tmp/work/medsam3_my_lora/outputs/chaos_smoke/checkpoints/best.pt \
  --checkpoint_path /root/autodl-tmp/models/sam3_base/sam3.pt \
  --bpe_path /root/autodl-tmp/repos/MedSAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --device cuda \
  --resolution 1008 \
  --detection_threshold 0.5 \
  --nms_iou_threshold 0.5
```

---

## 4. 关键参数说明

- `--data_root`: 数据集根目录（内部应有 `dataset_name/MultiEN_{dataset_name}.json`）
- `--datasets`: 可一次传多个数据集名，便于横向对比
- `--use_all_datasets`: 自动扫描 `data_root/*/MultiEN_{dataset}.json` 并评测全部数据集
- `--split`: `training` / `test` / `all`
- `--max_samples`: 每个 split 最多样本数，`<=0` 表示全量
- `--sam3_repo_root`: `repos/MedSAM3` 根路径
- `--my_lora_project_root`: `work/medsam3_my_lora` 路径（用于导入 LoRA 代码）
- `--lora_checkpoint_path`: `medsam3_my_lora` 训练产物（默认 best.pt）
- `--checkpoint_path`: SAM3 base 权重路径
- `--bpe_path`: SAM3 tokenizer BPE 路径
- `--detection_threshold`: query score 阈值
- `--nms_iou_threshold`: NMS 阈值
- `--load_from_hf`: 开启后尝试从 HF 加载 base 模型（默认关闭）
- `--lora_r / --lora_alpha / --lora_dropout`: 可覆盖 checkpoint 内 LoRA 配置

---

## 5. 输入输出说明

### 输入
- 每个数据集的 `MultiEN_*.json`
- 每条样本包含 image 路径、label npz 路径、questions 列表
- 当前评测使用 `qobj["label"]` 作为 prompt（即 gt label name prompt）

### 输出（与原评测工程保持相近）
在 `OUTPUT_ROOT` 下生成：

- `all_datasets_summary.json`：多数据集聚合指标（macro）
- 每个数据集一个目录，例如 `CHAOS_test_all/`：
  - `results.jsonl`：逐 turn 记录（含状态、dice、iou、错误信息）
  - `summary.json`：单数据集汇总
  - `per_case/<case_id>/`：
    - `turn_XX_final_outputs.json`：该 turn 推理输出和指标
    - `turn_XX_pred_vs_label.png`：预测 vs GT 可视化对比图

---

## 6. 批量评测建议

- 优先使用 shell 脚本 + 环境变量覆盖路径。
- 若只评测指定数据集，设置 `DATASETS="CHAOS BraTS"`；若全量评测，设置 `USE_ALL_DATASETS=1`。
- 输出根目录按时间戳或实验名拆分，便于后续对比不同 LoRA checkpoint。
