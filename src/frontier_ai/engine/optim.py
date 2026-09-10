"""Optimizer and learning-rate schedule construction."""

from __future__ import annotations

import math

import torch

from ..config import OptimConfig


def build_optimizer(
    model: torch.nn.Module, cfg: OptimConfig, device_type: str = "cpu"
) -> torch.optim.Optimizer:
    """AdamW, optionally using fused kernels on CUDA (meaningfully faster)."""
    groups = model.parameter_groups(cfg.weight_decay, decay_matrices=cfg.decay_params)
    fused = device_type == "cuda" and _fused_available()
    try:
        return torch.optim.AdamW(
            groups,
            lr=cfg.lr,
            betas=tuple(cfg.betas),  # type: ignore[arg-type]
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,  # overridden per group, kept for clarity
            fused=fused,
        )
    except (TypeError, RuntimeError):  # fused unsupported on this torch/device
        return torch.optim.AdamW(
            groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay
        )


def _fused_available() -> bool:
    return "fused" in torch.optim.AdamW.__init__.__code__.co_varnames


def lr_at(step: int, cfg: OptimConfig, max_steps: int | None = None) -> float:
    """Learning rate for `step` (1-indexed) under warmup + schedule."""
    if step <= 0:
        step = 1
    if step < cfg.warmup_steps and cfg.warmup_steps > 0:
        return cfg.lr * (step / max(1, cfg.warmup_steps))

    if cfg.schedule == "constant":
        return cfg.lr
    if cfg.schedule != "cosine":
        raise ValueError(f"unknown schedule '{cfg.schedule}'")

    total = max(1, max_steps or 0)
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    min_lr = cfg.lr * cfg.min_lr_ratio
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (cfg.lr - min_lr)


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):  # noqa: SLF001 - torch API
    """Manual-float LR scheduler: warmup then cosine (or constant) decay.

    Implemented directly (rather than via `LambdaLR`) so the schedule is explicit
    and easy to unit-test.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: OptimConfig, max_steps: int) -> None:
        self.cfg = cfg
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        return [lr_at(self.last_epoch, self.cfg, self.max_steps) for _ in self.base_lrs]

    # Keep the schedule metadata out of the pickled state: torch >= 2.6 loads
    # checkpoints with weights_only=True, and a custom class would break that.
    def state_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("cfg", "max_steps", "optimizer")}

    def load_state_dict(self, state_dict: dict) -> None:  # type: ignore[override]
        clean = {k: v for k, v in state_dict.items() if k not in ("cfg", "max_steps", "optimizer")}
        super().load_state_dict(clean)

    def current_lr(self) -> float:
        return float(self.get_last_lr()[0])
