# SAM3 医学分割三模型对比报告

## 1. 总体结论
- **总体 Macro Dice 最优**：SAM3
- **总体 Macro IoU 最优**：SAM3
- **No-pred rate 最低**：MedSAM3-LoRA

## 2. 建议重点看的指标
本次对比建议优先看四类指标：
1. **Overall Macro Dice / Macro IoU**：衡量总体分割质量。
2. **Dataset-level Mean Dice / Mean IoU**：衡量跨数据集泛化能力。
3. **No-pred rate / Failure rate**：衡量模型稳定性与“完全没打出来”的风险。
4. **Turn-wise trend**：衡量多轮场景下，随着轮次加深是否明显退化。

## 3. 各数据集 Dice 最优模型
dataset
AMOS2022               SAM3
BraTS          MedSAM3-LoRA
CHAOS                  SAM3
CMRxMotions            SAM3
COVID19        Medical-SAM3
Prostate               SAM3
SegRap2023     MedSAM3-LoRA

## 4. 轮次趋势概览
- Medical-SAM3: 首轮 0.374，末轮 0.000，变化 -0.374
- MedSAM3-LoRA: 首轮 0.320，末轮 0.000，变化 -0.320
- SAM3: 首轮 0.171，末轮 0.000，变化 -0.171

## 5. 图表说明
- 01/02：总体对比
- 03/04/05：热力图，适合快速看哪一类数据集差距最大
- 06/07/08：分数据集柱状图，适合写论文/汇报
- 09/10/11：轮次趋势图，适合分析多轮推理场景退化
- 12/13：标签级增益/退化最明显的类别，适合做 case study
