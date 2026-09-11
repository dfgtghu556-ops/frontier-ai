"""Device / dtype resolution.

Nothing here hard-codes CPU: every call inspects what is actually available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSpec:
    """Resolved training device + autocast settings."""

    device: torch.device
    amp: bool  # use torch.autocast
    amp_dtype: torch.dtype  # dtype used inside autocast
    is_cuda: bool

    def describe(self) -> str:
        amp = f"{str(self.amp_dtype).replace('torch.', '')}" if self.amp else "off"
        return f"device={self.device.type} amp={amp}"


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a device from a string preference.

    auto -> cuda > mps > cpu; explicit values ("cpu", "cuda", "cuda:1", "mps") are honoured.
    """
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device '{pref}' requested but CUDA is not available")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device(pref)


def resolve_spec(
    preference: str = "auto",
    precision: str = "auto",
) -> DeviceSpec:
    """Device + mixed-precision plan.

    precision:
      auto  -> bf16 autocast on bf16-capable CUDA, fp16 autocast on older CUDA,
               fp32 (no autocast) on CPU/MPS
      fp32  -> no autocast anywhere
      bf16 / fp16 -> autocast if the device supports it, else fp32 with a warning
    """
    device = resolve_device(preference)
    is_cuda = device.type == "cuda"
    prec = (precision or "auto").lower()

    if prec == "fp32":
        return DeviceSpec(device, False, torch.float32, is_cuda)

    if is_cuda:
        if prec == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif prec == "bf16":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif prec == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"unknown precision '{precision}'")
        return DeviceSpec(device, True, dtype, is_cuda)

    # CPU / MPS: keep master weights in fp32, no autocast.
    return DeviceSpec(device, False, torch.float32, is_cuda)


def threads_for(device: torch.device, requested: int | None = None) -> int:
    """Set (and return) a sensible intra-op thread count.

    Capped low on CPU-only sandboxes so tiny runs stay responsive; left alone on GPU.
    """
    import os

    if requested is not None:
        n = requested
    elif device.type == "cuda":
        n = max(1, (os.cpu_count() or 2) // 2)
    else:
        n = max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(n)
    return n
