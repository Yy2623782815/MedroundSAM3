import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_linear_names: Optional[List[str]] = None
    exclude_linear_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {self.alpha}")
        if self.dropout < 0:
            raise ValueError(f"LoRA dropout must be >= 0, got {self.dropout}")

        if self.target_linear_names is None:
            self.target_linear_names = [
                "attn.proj",
                "mlp.fc1",
                "mlp.fc2",
                "prompt_mlp",
                "cross_attend_prompt",
                "dot_prod_scoring",
                "segmentation_head",
                "self_attention",
                "cross_attention",
            ]

        if self.exclude_linear_names is None:
            self.exclude_linear_names = [
                "patch_embed",
                "pos_embed",
                "cls_token",
                "mask_token",
                "out_proj",   # 关键：排除 torch.nn.MultiheadAttention.out_proj
            ]


class LoRALinear(nn.Module):
    """
    用 LoRA 包装一个现有的 nn.Linear。
    输出 = 原始线性层输出 + LoRA 增量
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"base_linear must be nn.Linear, got {type(base_linear)}")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.base = base_linear
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        base_device = base_linear.weight.device
        base_dtype = base_linear.weight.dtype

        # LoRA A: down projection
        self.lora_A = nn.Linear(
            self.in_features,
            r,
            bias=False,
            device=base_device,
            dtype=base_dtype,
        )
        # LoRA B: up projection
        self.lora_B = nn.Linear(
            r,
            self.out_features,
            bias=False,
            device=base_device,
            dtype=base_dtype,
        )

        self.reset_parameters()

        # 冻结 base 参数
        for p in self.base.parameters():
            p.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def _get_parent_module(root_module: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    """
    例如 module_name='a.b.c'，返回:
        parent = root.a.b
        child_name = 'c'
    """
    parts = module_name.split(".")
    parent = root_module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _name_matches(name: str, patterns: Iterable[str]) -> bool:
    name = name.lower()
    for p in patterns:
        if p.lower() in name:
            return True
    return False


def should_replace_linear(module_name: str, cfg: LoRAConfig) -> bool:
    name = module_name.lower()

    if _name_matches(name, cfg.exclude_linear_names):
        return False

    # 额外保险：直接跳过 MultiheadAttention 的 out_proj
    if name.endswith("out_proj") or ".out_proj" in name:
        return False

    return _name_matches(name, cfg.target_linear_names)


def inject_lora_into_model(
    model: nn.Module,
    cfg: LoRAConfig,
    verbose: bool = True,
) -> List[str]:
    """
    遍历模型，将符合条件的 nn.Linear 替换为 LoRALinear。
    返回实际被替换的模块名列表。
    """
    replace_names: List[str] = []

    # 先收集，避免边遍历边修改 named_modules
    linear_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_replace_linear(name, cfg):
            linear_module_names.append(name)

    for name in linear_module_names:
        parent, child_name = _get_parent_module(model, name)
        old_linear = getattr(parent, child_name)
        new_linear = LoRALinear(
            base_linear=old_linear,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )
        new_linear = new_linear.to(
            device=old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )
        setattr(parent, child_name, new_linear)
        replace_names.append(name)

    if verbose:
        print(f"[LoRA] replaced {len(replace_names)} linear layers.")
        for n in replace_names[:50]:
            print(f"  - {n}")
        if len(replace_names) > 50:
            print(f"  ... and {len(replace_names) - 50} more")

    return replace_names


def mark_only_lora_as_trainable(
    model: nn.Module,
    train_bias: bool = False,
    train_norm: bool = False,
) -> None:
    """
    默认冻结所有参数，只训练:
    - LoRALinear.lora_A
    - LoRALinear.lora_B
    可选再开放 bias / norm
    """
    for _, p in model.named_parameters():
        p.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            for p in module.lora_A.parameters():
                p.requires_grad = True
            for p in module.lora_B.parameters():
                p.requires_grad = True

            if train_bias and module.base.bias is not None:
                module.base.bias.requires_grad = True

        if train_norm and isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            for p in module.parameters():
                p.requires_grad = True


def get_trainable_param_stats(model: nn.Module) -> dict:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    ratio = 100.0 * trainable / max(total, 1)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio_percent": float(ratio),
    }


def print_trainable_param_stats(model: nn.Module) -> None:
    stats = get_trainable_param_stats(model)
    print(
        "[LoRA] trainable params: "
        f"{stats['trainable_params']:,} / {stats['total_params']:,} "
        f"({stats['trainable_ratio_percent']:.4f}%)"
    )


def extract_lora_state_dict(model: nn.Module) -> dict:
    """
    只提取 LoRA 权重，方便保存轻量 checkpoint。
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: dict, strict: bool = True) -> None:
    missing = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_a = f"{name}.lora_A.weight"
            key_b = f"{name}.lora_B.weight"

            if key_a in state_dict:
                module.lora_A.weight.data.copy_(state_dict[key_a].to(module.lora_A.weight.device))
            else:
                missing.append(key_a)

            if key_b in state_dict:
                module.lora_B.weight.data.copy_(state_dict[key_b].to(module.lora_B.weight.device))
            else:
                missing.append(key_b)

    if strict and missing:
        raise KeyError(f"Missing LoRA keys: {missing[:20]}")