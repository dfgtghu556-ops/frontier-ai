"""Data-parallel training on several GPUs of one machine (EXP-044, roadmap step 13).

Why this exists
---------------
Every Kaggle session we ran had two T4s attached and we used one (D-047). With
DistributedDataParallel each process (one per GPU, started by ``torchrun``) takes its own share of
every step's batch, the gradients are averaged across the processes, and every process applies the
same update. With the one-pass sampler (``OnePassDataset.set_shard``) the shares are fixed slices
of the same 32 windows the one-GPU run would read, so the update equals the one-GPU update up to
the order of floating-point additions.

What lives here (small and explicit on purpose)
-----------------------------------------------
* :func:`init_from_env` starts the process group when ``torchrun`` set ``WORLD_SIZE`` > 1 (NCCL on
  CUDA, gloo on CPU for the tests), and does nothing otherwise, so a one-GPU run never touches
  ``torch.distributed``.
* :func:`rank`, :func:`world`, :func:`is_main`: who am I. Without a process group: rank 0 of 1.
* :func:`all_true`: one boolean agreed by every process (used for the non-finite-loss check, so
  all processes stop together instead of one waiting forever in the gradient exchange).
* :func:`run_on_main`: run a function on process 0 only and hand its result, or the exception it
  raised, to every process. Evaluations, reports, checkpoints and stop rules use it, so a stop
  rule that fires on process 0 stops every process.
* :class:`SingleFormat`: reads and writes a DDP model's weights in exactly the format the one-GPU
  run uses, so checkpoints move freely between one and two GPUs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")

PREFIXES = ("_orig_mod.", "module.")


def active() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if active() else 0


def world() -> int:
    return dist.get_world_size() if active() else 1


def is_main() -> bool:
    return rank() == 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def init_from_env(timeout_minutes: float = 30.0) -> bool:
    """Start the process group if launched by torchrun with more than one process.

    Returns True when a group is active. On CUDA each process is pinned to GPU ``LOCAL_RANK``
    before anything is allocated, so ``torch.device("cuda")`` means "my GPU" everywhere else.
    """
    if active():
        return True
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank())
    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo", timeout=timedelta(minutes=timeout_minutes)
    )
    return True


def cleanup() -> None:
    if active():
        dist.destroy_process_group()


def barrier() -> None:
    if active():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _device() -> torch.device:
    if active() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def all_true(flag: bool) -> bool:
    """True only if ``flag`` is True on every process."""
    if not active():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=_device())
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not active():
        return obj
    box = [obj if rank() == src else None]
    dist.broadcast_object_list(box, src=src, device=_device())
    return box[0]


def all_gather_object(obj: Any) -> list[Any]:
    if not active():
        return [obj]
    out: list[Any] = [None] * world()
    dist.all_gather_object(out, obj)
    return out


def run_on_main(fn: Callable[[], T]) -> T:
    """Run ``fn`` on process 0; every process returns its result or raises its exception.

    The exception object itself is sent (pickled), so ``except StopRule`` works the same on every
    process. Without a process group this is just ``fn()``.
    """
    if not active():
        return fn()
    payload: tuple[str, Any] = ("ok", None)
    if is_main():
        try:
            payload = ("ok", fn())
        except BaseException as exc:  # noqa: BLE001 - re-raised on every process below
            payload = ("raise", exc)
    kind, value = broadcast_object(payload)
    if kind == "raise":
        raise value
    return value


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The plain model inside torch.compile and/or DistributedDataParallel wrappers."""
    seen = 0
    while seen < 4:
        inner = getattr(model, "_orig_mod", None) or getattr(model, "module", None)
        if inner is None or not isinstance(inner, torch.nn.Module):
            return model
        model, seen = inner, seen + 1
    return model


def strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        changed = True
        while changed:
            changed = False
            for p in PREFIXES:
                if k.startswith(p):
                    k, changed = k[len(p) :], True
        out[k] = v
    return out


class SingleFormat:
    """Stand-in for the model in checkpoint save/load during multi-process training.

    ``state_dict()`` returns the keys a one-process run with the same ``compiled`` setting writes
    (``_orig_mod.`` + name when compiled, plain names otherwise); ``load_state_dict`` accepts any of
    those formats. So a checkpoint written on two GPUs resumes on one, and the other way round.
    """

    def __init__(self, raw: torch.nn.Module, compiled: bool) -> None:
        self.raw = raw
        self.prefix = "_orig_mod." if compiled else ""

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {self.prefix + k: v for k, v in self.raw.state_dict().items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> Any:
        return self.raw.load_state_dict(strip_prefixes(state))

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.raw.parameters()
