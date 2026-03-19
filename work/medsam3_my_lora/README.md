# medsam3_my_lora

基于 `repos/MedSAM3` 的 LoRA 微调工程（独立目录），用于你的医学图像 + `label_name` prompt 二值分割训练。

> 以下命令默认在仓库根目录执行（`MedroundSAM3/`）。

## 目录
- `datasets/`: 数据集与 collate
- `models/`: LoRA 注入、MedSAM3 本地模型构建、forward 适配
- `utils/med_data_utils.py`: 从旧工程迁移的稳定 GT `.npz` 解析逻辑
- `tools/build_labelname_samples.py`: 从你的数据 JSON 构建 train/val/test JSONL 索引
- `train_medsam3_my_lora.py`: 训练入口（支持单卡/多卡 torchrun）
- `configs/my_smoke_lora.yaml`: 本地模型路径 + 数据路径 + 超参配置（默认 CHAOS）
- `scripts/run_smoke_train.sh`: 一键训练脚本（`NUM_GPUS` 可配）

## 本地模型加载（默认）
通过配置显式传入，不依赖在线下载：
- `model.sam3_repo_root`: MedSAM3 本地仓库路径
- `model.checkpoint_path`: 本地 SAM3 checkpoint
- `model.bpe_path`: 本地 BPE 词表路径
- `model.load_from_hf: false`

> 说明：MedSAM3 源码在导入时会引用 `huggingface_hub`。本工程在 `load_from_hf=false` 时注入本地 stub，确保仅使用本地权重路径，不触发 HF 下载。

## Batch Size 与 GPU 数量说明
训练入口已支持 DDP（`torchrun`），并在日志中打印：
- `world_size`
- `batch_size_per_device`
- `global_batch_size`

建议在 YAML 使用：

```yaml
data:
  batch_size_per_device: 4
  val_batch_size_per_device: 4
```

可选地使用：

```yaml
data:
  global_batch_size: 8
```

当配置 `global_batch_size` 时，训练脚本会按 `global_batch_size / world_size` 自动推导每卡 batch；若不能整除会报错。

兼容旧字段：`batch_size` / `val_batch_size` 仍可用（仅在未配置新字段时回退）。

## 训练启动方式

### 1) 用脚本启动（推荐）
脚本支持通过 `NUM_GPUS` 显式指定卡数。

```bash
# 单卡
NUM_GPUS=1 bash work/medsam3_my_lora/scripts/run_smoke_train.sh

# 2 卡
NUM_GPUS=2 bash work/medsam3_my_lora/scripts/run_smoke_train.sh

# 4 卡
NUM_GPUS=4 bash work/medsam3_my_lora/scripts/run_smoke_train.sh
```

也可透传额外参数给训练入口，例如覆盖 epoch：

```bash
NUM_GPUS=2 bash work/medsam3_my_lora/scripts/run_smoke_train.sh --epochs 20
```

### 2) 直接命令行启动

```bash
# 单卡
python work/medsam3_my_lora/train_medsam3_my_lora.py \
  --config work/medsam3_my_lora/configs/my_smoke_lora.yaml

# 2 卡
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  work/medsam3_my_lora/train_medsam3_my_lora.py \
  --config work/medsam3_my_lora/configs/my_smoke_lora.yaml

# 4 卡
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  work/medsam3_my_lora/train_medsam3_my_lora.py \
  --config work/medsam3_my_lora/configs/my_smoke_lora.yaml
```

> 如需指定特定卡，可配合 `CUDA_VISIBLE_DEVICES`：
>
> `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 ...`

## 多 query mask 选择（可配置）
训练时 SAM3 会输出 `pred_masks=[B,Q,H,W]`，而二值监督 loss 需要单掩码 `[B,1,H,W]`。
可通过 `train.query_select` 配置策略：

```yaml
train:
  query_select:
    mode: logits_max   # logits_max | mask_mean
    topk: 1            # >=1
    reduce: mean       # mean | max
```

- `mode=logits_max`：用 `sigmoid(pred_logits).max(-1)` 作为 query score（更接近 MedSAM3 postprocess 逻辑）
- `mode=mask_mean`：用 `pred_mask` 的空间均值打分
- `topk>1`：先选 top-k query，再按 `reduce` 聚合成 1 张监督掩码

## 训练日志与进度条
训练时进度条（仅主进程打印）会显示：
- 当前阶段（`train`/`val`）
- `epoch` 与 `batch` 进度
- `loss` / `dice` / `iou`
- `lr`
- 当前 step 耗时（`step_t`）
- 当前 epoch 已耗时（`ep_elapsed`）
- 当前 epoch 预估剩余时间（`ep_eta`）

每个 epoch 结束会输出：
- `elapsed`（训练累计耗时）
- `eta`（预计剩余训练时长）
- `estimated_total`（预计总训练时长）

并写入 `logs/history.json` 中的 `elapsed_sec` / `eta_sec` / `estimated_total_sec`。

## 构建索引（CHAOS）
```bash
export PYTHONPATH=work/medsam3_my_lora:$PYTHONPATH
python work/medsam3_my_lora/tools/build_labelname_samples.py \
  --data_root data/SAM3_data \
  --datasets CHAOS \
  --output_dir work/medsam3_my_lora/data_index/chaos_only
```

## 训练输出
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `logs/history.json`
- `model_info.json`

## 后续扩展建议
1. 在 dataset 中增加 history_text / multi-turn 字段。
2. 在 forward 中扩展 query 组织方式（多 query/多目标）。
3. 增加验证集可视化与更细粒度评估。
