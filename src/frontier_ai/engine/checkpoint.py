"""Checkpoint save / load.

A checkpoint directory contains:
    model.pt      - model weights (+ tied-embedding-safe state dict)
    optimizer.pt  - optimizer + scheduler state (for exact resume)
    config.json   - the full ExperimentConfig used for the run
    meta.json     - step, best metric, wall time, torch version
"""

from __future__ import annotations

import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig


def save_checkpoint(
    out_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    cfg: ExperimentConfig | None = None,
    step: int = 0,
    best_val: float = float("inf"),
    extra: dict[str, Any] | None = None,
    tag: str = "last",
) -> Path:
    """Write a checkpoint to `out_dir/tag` (and refresh `out_dir/best` if asked)."""
    ckpt_dir = Path(out_dir) / tag
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    if optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": getattr(scheduler, "state_dict", lambda: None)(),
            },
            ckpt_dir / "optimizer.pt",
        )
    if cfg is not None:
        cfg.save(ckpt_dir / "config.json")

    meta = {
        "step": step,
        "best_val": None if best_val == float("inf") else best_val,
        "timestamp": time.time(),
        "torch": torch.__version__,
        "n_params": sum(p.numel() for p in model.parameters()),
        **(extra or {}),
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return ckpt_dir


def load_checkpoint(
    ckpt_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load weights (and optionally optimizer/scheduler state) in place."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt_dir}")

    state = _torch_load(ckpt_dir / "model.pt", map_location)
    model.load_state_dict(state)

    opt_path = ckpt_dir / "optimizer.pt"
    if optimizer is not None and opt_path.exists():
        blob = _torch_load(opt_path, map_location)
        optimizer.load_state_dict(blob["optimizer"])
        if scheduler is not None and blob.get("scheduler") is not None:
            scheduler.load_state_dict(blob["scheduler"])

    meta_path = ckpt_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return meta


def find_latest(out_dir: str | Path) -> Path | None:
    """Most recent `step-*` directory inside out_dir, else `last`, else None."""
    base = Path(out_dir)
    if not base.exists():
        return None
    steps = sorted(base.glob("step-*"), key=lambda p: int(p.name.split("-")[-1]))
    if steps:
        return steps[-1]
    last = base / "last"
    return last if last.exists() else None


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    """torch.load with weights_only when supported (torch>=2.6 default), else plain.

    Checkpoints written by this repo are plain state dicts; the fallback exists so
    older torch versions (no `weights_only` kwarg) still work.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, pickle.UnpicklingError):  # pragma: no cover - version dependent
        return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_config(cfg: ExperimentConfig, device: torch.device) -> torch.nn.Module:
    """Instantiate a GPT from a config (used by train/eval/generate scripts)."""
    from ..model.gpt import GPT

    model = GPT(cfg.model)
    return model.to(device)
