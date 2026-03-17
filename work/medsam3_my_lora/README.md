# medsam3_my_lora

基于 `repos/MedSAM3` 的 LoRA 微调工程（独立目录），用于你的医学图像 + `label_name` prompt 二值分割训练。

## 目录
- `datasets/`: 数据集与 collate
- `models/`: LoRA 注入、MedSAM3 本地模型构建、forward 适配
- `utils/med_data_utils.py`: 从旧工程迁移的稳定 GT `.npz` 解析逻辑
- `tools/build_labelname_samples.py`: 从你的数据 JSON 构建 train/val/test JSONL 索引
- `train_medsam3_my_lora.py`: 训练入口
- `configs/my_smoke_lora.yaml`: 本地模型路径 + 数据路径 + 超参配置（默认 CHAOS）
- `scripts/run_smoke_train.sh`: 一键冒烟脚本

## 本地模型加载（默认）
通过配置显式传入，不依赖在线下载：
- `model.sam3_repo_root`: MedSAM3 本地仓库路径
- `model.checkpoint_path`: 本地 SAM3 checkpoint
- `model.bpe_path`: 本地 BPE 词表路径
- `model.load_from_hf: false`

> 说明：MedSAM3 源码在导入时会引用 `huggingface_hub`。本工程在 `load_from_hf=false` 时注入本地 stub，确保仅使用本地权重路径，不触发 HF 下载。

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

## 构建索引（CHAOS）
```bash
export PYTHONPATH=/root/autodl-tmp/work/medsam3_my_lora:$PYTHONPATH
python /root/autodl-tmp/work/medsam3_my_lora/tools/build_labelname_samples.py \
  --data_root /root/autodl-tmp/data/SAM3_data \
  --datasets CHAOS \
  --output_dir /root/autodl-tmp/work/medsam3_my_lora/data_index/chaos_only
```

## 训练（默认 CHAOS）
```bash
bash /root/autodl-tmp/work/medsam3_my_lora/scripts/run_smoke_train.sh
```

输出包括：
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `logs/history.json`
- `model_info.json`

## 后续扩展建议
1. 在 dataset 中增加 history_text / multi-turn 字段。
2. 在 forward 中扩展 query 组织方式（多 query/多目标）。
3. 增加验证集可视化与更细粒度评估。
