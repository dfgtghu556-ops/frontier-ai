"""Weights-only model files: EXP-043's ``model_final.pt`` and its fp16 copy (EXP-045).

A weights file is a ``torch.save`` dict with ``model_config`` (the :class:`ModelConfig` fields) and
``state_dict``. EXP-043 writes float32 weights (about 0.76 GB for 190 M parameters);
:func:`export_fp16` writes the same weights in fp16 (about 0.38 GB) for downloading and trying the
model on a laptop. Loading always gives a float32 model (CPUs compute fp16 slowly or not at all);
the fp16 rounding error is far below what changes a sampled continuation in practice, but scores
for reports are always computed from the float32 file.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from ..config import ModelConfig
from .distributed import strip_prefixes

FP16_FORMAT = "frontier-weights-fp16-v1"


def read_weights(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_config" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path}: not a weights file (needs model_config and state_dict)")
    return payload


def model_config_of(payload: dict[str, Any]) -> ModelConfig:
    known = {f.name for f in fields(ModelConfig)}
    unknown = set(payload["model_config"]) - known
    if unknown:
        raise ValueError(f"unknown model_config fields {sorted(unknown)} (newer code needed?)")
    return ModelConfig(**payload["model_config"])


def load_weights(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """A float32 GPT in eval mode from a weights file, plus the file's other fields."""
    from ..model.gpt import GPT

    payload = read_weights(path)
    model = GPT(model_config_of(payload))
    state = {k: v.float() for k, v in strip_prefixes(payload["state_dict"]).items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    info = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, info


def export_fp16(model: torch.nn.Module, path: str | Path, **extra: Any) -> dict[str, Any]:
    """Write ``model``'s weights in fp16; returns ``{file, sha256, bytes}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, torch.Tensor] = {}
    converted: dict[int, torch.Tensor] = {}  # tied weights (embedding = output layer) stay one tensor
    for k, v in strip_prefixes(model.state_dict()).items():
        ptr = v.data_ptr()
        if ptr not in converted:
            converted[ptr] = v.detach().to("cpu", torch.float16)
        state[k] = converted[ptr]
    torch.save({"format": FP16_FORMAT, "model_config": asdict(model.cfg), "state_dict": state, **extra}, path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"file": path.name, "sha256": h.hexdigest(), "bytes": path.stat().st_size, "format": FP16_FORMAT}
