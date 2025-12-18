# peft_lora_dora.py
# PyTorch ≥ 2.0
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn


@dataclass
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")  # 名称关键字匹配
    lora_bias: str = "none"   # {"none","lora_only","all"}
    dora: bool = False        # True=DoRA, False=LoRA


def _is_target(name: str, cfg: LoraConfig) -> bool:
    return any(t in name for t in cfg.target_modules)


class _LoraDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            return torch.dropout(x, p=self.p, train=True)
        return x


class LoRALinear(nn.Module):
    """
    以“外包裹”的方式给任意 nn.Linear 添加 LoRA；
    基础权重冻结，只训练 A/B（以及可选 bias）。
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float,
                 lora_bias: str = "none"):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        self.r = r
        self.scaling = alpha / r
        self.dropout = _LoraDropout(dropout)

        # 低秩因子：B (out×r), A (r×in)
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        # 按 LoRA 习惯初始化（A 零、B 正态/均匀小值）
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.zeros_(self.A)

        # bias 策略
        self.lora_bias = lora_bias
        if lora_bias == "lora_only":
            self.bias = nn.Parameter(torch.zeros_like(base.bias)) if base.bias is not None else None
        else:
            # 共享 base.bias
            self.bias = base.bias

        # 冻结基础权重
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        # 兼容某些自定义 Linear 的 _linear 实现
        base_out = self.base._linear(x) if hasattr(self.base, "_linear") else self.base(x)
        # ΔW(x) = x A^T B^T
        lora_out = self.dropout(x) @ self.A.t() @ self.B.t()
        return base_out + self.scaling * lora_out

    def merge(self) -> nn.Linear:
        """
        把 ΔW=scale*B@A 写回 base.weight，返回合并后的原生 Linear。
        """
        with torch.no_grad():
            delta = (self.B @ self.A) * self.scaling  # (out×in)
            self.base.weight += delta
            if self.lora_bias == "lora_only" and self.bias is not None:
                if self.base.bias is None:
                    self.base.bias = nn.Parameter(self.bias.detach().clone())
                else:
                    self.base.bias += self.bias
        return self.base


class DoRALinear(nn.Module):
    """
    DoRA：对方向做 LoRA 更新，长度（行范数）保持与预训练相同。
    W_hat = norm(W0) * (W0 + ΔW) / ||W0 + ΔW||
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float,
                 lora_bias: str = "none"):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        self.r = r
        self.scaling = alpha / r
        self.dropout = _LoraDropout(dropout)

        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.zeros_(self.A)

        self.lora_bias = lora_bias
        if lora_bias == "lora_only":
            self.bias = nn.Parameter(torch.zeros_like(base.bias)) if base.bias is not None else None
        else:
            self.bias = base.bias

        # 冻结基础权重
        for p in self.base.parameters():
            p.requires_grad = False

        # 记录每一行的原始范数（长度），作为 buffer，跟随 .to(device) 一起搬
        with torch.no_grad():
            w = self.base.weight.data  # (out×in)
            row_norm = torch.norm(w, dim=1, keepdim=True).clamp_min(1e-12)
        # 注册为 buffer，避免出现在 parameter list 里
        self.register_buffer("row_norm", row_norm)

    def _weight_hat(self) -> torch.Tensor:
        """
        返回 DoRA 重参数化后的权重矩阵（out×in），确保所有张量在同一 device。
        """
        W0 = self.base.weight  # (out×in)
        delta = (self.B @ self.A) * self.scaling  # (out×in)

        W_new = W0 + delta
        dir_norm = torch.norm(W_new, dim=1, keepdim=True).clamp_min(1e-12)

        # 确保 row_norm 在同一 device，避免 cuda / cpu 混用
        row_norm = self.row_norm.to(W_new.device)
        W_dora = W_new / dir_norm * row_norm
        return W_dora

    def forward(self, x):
        # 用 DoRA 重参数化后的权重进行线性映射
        W = self._weight_hat().t()  # (in×out)
        out = x @ W
        if self.bias is not None:
            out = out + self.bias
        return out

    def merge(self) -> nn.Linear:
        """
        把 DoRA 的 W_hat 写回 base.weight，返回合并后的原生 Linear。
        """
        with torch.no_grad():
            self.base.weight.copy_(self._weight_hat())
            if self.lora_bias == "lora_only" and self.bias is not None:
                if self.base.bias is None:
                    self.base.bias = nn.Parameter(self.bias.detach().clone())
                else:
                    self.base.bias += self.bias
        return self.base


def _replace_linear(module: nn.Module, name: str, cfg: LoraConfig):
    child = getattr(module, name)
    if not isinstance(child, nn.Linear):
        return
    if cfg.dora:
        wrapper = DoRALinear(child, cfg.r, cfg.alpha, cfg.dropout, cfg.lora_bias)
    else:
        wrapper = LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout, cfg.lora_bias)
    setattr(module, name, wrapper)


def _walk_and_inject(model: nn.Module, cfg: LoraConfig):
    """
    深度遍历模型，对名字中包含 target_modules 关键字的 nn.Linear 注入 LoRA/DoRA。
    """
    for full_name, m in list(model.named_modules()):
        for child_name, child in list(m.named_children()):
            full_child_name = f"{full_name}.{child_name}" if full_name else child_name
            if isinstance(child, nn.Linear) and _is_target(full_child_name, cfg):
                _replace_linear(m, child_name, cfg)


def mark_only_peft_trainable(model: nn.Module):
    """
    备用：只让 PEFT 相关参数 (A/B/bias) 可训练。
    当前主逻辑在 inject_peft_to_vit 里已经做了冻结/解冻，这个函数可以不用。
    """
    for n, p in model.named_parameters():
        if ("A" in n or "B" in n) or ("lora" in n.lower()):
            p.requires_grad = True
        # 其它保持原状（base.weight/base.bias 在 wrapper 里已冻结）
        # p.requires_grad = p.requires_grad


def inject_peft_to_vit(model: nn.Module, cfg: LoraConfig):
    """
    在 ViT 上注入 LoRA/DoRA，并只训练 A/B (+ 可选 bias)。
    """
    _walk_and_inject(model, cfg)

    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 只训练 LoRA/DoRA wrapper 里的 A/B（以及可选 bias）
    for m in model.modules():
        if isinstance(m, (LoRALinear, DoRALinear)):
            m.A.requires_grad = True
            m.B.requires_grad = True
            if isinstance(m.bias, nn.Parameter):
                m.bias.requires_grad = True


def merge_and_strip_peft(model: nn.Module):
    """
    把所有 LoRA/DoRA wrapper 合并回原生 Linear，并就地替换掉 wrapper。
    导出部署/当做新的预训练模型时用。
    """
    def _merge_in_module(mod: nn.Module):
        for name, child in list(mod.named_children()):
            if isinstance(child, (LoRALinear, DoRALinear)):
                merged = child.merge()
                setattr(mod, name, merged)  # 用合并后的 Linear 替换 wrapper
            else:
                _merge_in_module(child)

    _merge_in_module(model)
    return model
