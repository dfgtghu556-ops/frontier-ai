"""Token storage + batch sampling.

The prepared corpus is a flat array of token ids (numpy, memory-mapped so corpora
larger than RAM still work). Batches are built by sampling random offsets and
shifting by one position to form the standard next-token LM target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _dtype_for(vocab_size: int) -> np.dtype:
    if vocab_size <= 2**8:
        return np.dtype("uint8")
    if vocab_size <= 2**16:
        return np.dtype("uint16")
    return np.dtype("int32")


@dataclass
class DataMeta:
    """Side-car metadata written next to the token file."""

    n_tokens: int
    vocab_size: int
    dtype: str
    level: str
    n_train: int
    n_val: int

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DataMeta:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class TokenDataset:
    """Memory-mapped token array with train/val views."""

    def __init__(self, tokens_path: str | Path, meta: DataMeta | None = None) -> None:
        self.path = Path(tokens_path)
        if meta is None:
            meta = DataMeta.load(self.path.with_suffix(".meta.json"))
        self.meta = meta
        self.tokens = np.memmap(self.path, dtype=np.dtype(meta.dtype), mode="r", shape=(meta.n_tokens,))
        self.n_train = meta.n_train
        self.n_val = meta.n_val

    # ------------------------------------------------------------- slicing --
    @property
    def train(self) -> np.memmap:
        return self.tokens[: self.n_train]

    @property
    def val(self) -> np.memmap:
        return self.tokens[self.n_train : self.n_train + self.n_val]

    def split(self, name: str) -> np.memmap:
        if name == "train":
            return self.train
        if name == "val":
            return self.val
        raise ValueError(f"unknown split '{name}' (expected 'train' or 'val')")

    # ---------------------------------------------------------- batching ---
    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample `batch_size` contiguous windows of length `block_size`.

        Uses numpy for the gather (much cheaper than a python loop) and copies the
        result straight onto the target device.
        """
        data = self.split(split)
        if len(data) <= block_size + 1:
            raise ValueError(
                f"split '{split}' has {len(data)} tokens, need more than block_size+1={block_size + 1}. "
                "Prepare more data or lower block_size."
            )
        rng = np.random.default_rng(None if generator is None else int(generator.seed() % (2**32)))
        hi = len(data) - block_size - 1
        offsets = (
            rng.integers(0, hi, size=batch_size) if hi > 0 else np.zeros(batch_size, dtype=np.int64)
        )

        x = np.stack([data[o : o + block_size].astype(np.int64) for o in offsets])
        y = np.stack([data[o + 1 : o + 1 + block_size].astype(np.int64) for o in offsets])

        device_type = device.type if isinstance(device, torch.device) else "cpu"
        non_blocking = device_type == "cuda"
        xt = torch.from_numpy(x).to(device=device, non_blocking=non_blocking)
        yt = torch.from_numpy(y).to(device=device, non_blocking=non_blocking)
        return xt, yt

    def batches_per_epoch(self, batch_size: int, block_size: int, split: str = "train") -> int:
        return max(1, (len(self.split(split)) - block_size - 1) // (batch_size * block_size))


def write_tokens(
    out_path: str | Path,
    ids,
    vocab_size: int,
    level: str,
    val_frac: float = 0.1,
) -> DataMeta:
    """Persist encoded ids + metadata. Returns the metadata object."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = _dtype_for(vocab_size)
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(out_path)
    n_val = int(len(arr) * val_frac)
    meta = DataMeta(
        n_tokens=int(len(arr)),
        vocab_size=int(vocab_size),
        dtype=str(dtype),
        level=level,
        n_train=int(len(arr) - n_val),
        n_val=n_val,
    )
    meta.save(out_path.with_suffix(".meta.json"))
    return meta


def summary(ds: TokenDataset) -> dict[str, int]:
    return {
        "n_tokens": int(ds.meta.n_tokens),
        "vocab_size": int(ds.meta.vocab_size),
        "n_train": int(ds.n_train),
        "n_val": int(ds.n_val),
    }
