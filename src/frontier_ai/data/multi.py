"""Several packed token files read as one corpus (EXP-040: the 13 EXP-037 language files).

Each training sequence comes from one file, chosen with probability proportional to that file's
training tokens, at a uniformly random offset inside it. That is the same as reading the
concatenated corpus uniformly (only windows that would cross a file boundary are excluded), so it
is the corpus *as it is*, not a chosen mixture. Validation batches use the validation token counts
the same way. Per-file (per-language) validation is available through :attr:`parts`.

The object has the interface the :class:`~frontier_ai.engine.trainer.Trainer` uses
(``get_batch``, ``batches_per_epoch``, ``tokens_per_byte`` …), so it trains exactly like a
:class:`~frontier_ai.data.dataset.TokenDataset`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .dataset import TokenDataset


class MultiTokenDataset:
    """Packed token files with natural (size-proportional) sampling across files."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if not paths:
            raise ValueError("MultiTokenDataset needs at least one token file")
        self.parts: dict[str, TokenDataset] = {}
        for p in paths:
            ds = TokenDataset(p)
            name = Path(p).stem
            if name in self.parts:
                raise ValueError(f"duplicate file name {name!r}")
            self.parts[name] = ds
        vocabs = {ds.meta.vocab_size for ds in self.parts.values()}
        if len(vocabs) != 1:
            raise ValueError(f"all files must share one vocabulary, got sizes {sorted(vocabs)}")
        self.vocab_size = vocabs.pop()
        self.names = list(self.parts)
        self.n_train = sum(ds.n_train for ds in self.parts.values())
        self.n_val = sum(ds.n_val for ds in self.parts.values())
        self.path = Path(paths[0]).parent

    # ------------------------------------------------------------- weights --
    def weights(self, split: str) -> np.ndarray:
        """Sampling probability of each file for `split`, proportional to its tokens."""
        counts = np.array(
            [ds.n_train if split == "train" else ds.n_val for ds in self.parts.values()], dtype=np.float64
        )
        if split not in ("train", "val"):
            raise ValueError(f"unknown split '{split}' (expected 'train' or 'val')")
        return counts / counts.sum()

    # ------------------------------------------------- text length (bpb) ---
    def split_lengths(self, name: str) -> tuple[int | None, int | None]:
        """Summed ``(n_bytes, n_chars)``; ``None`` if any file does not record them."""
        out: list[int | None] = []
        for unit in (0, 1):
            vals = [ds.split_lengths(name)[unit] for ds in self.parts.values()]
            out.append(None if any(v is None for v in vals) else int(sum(vals)))  # type: ignore[arg-type]
        return out[0], out[1]

    def tokens_per_byte(self, name: str = "val") -> float | None:
        return self._per_unit(name, 0)

    def tokens_per_char(self, name: str = "val") -> float | None:
        return self._per_unit(name, 1)

    def _per_unit(self, name: str, unit: int) -> float | None:
        denominator = self.split_lengths(name)[unit]
        n_tokens = self.n_train if name == "train" else self.n_val
        if denominator is None or denominator <= 0 or n_tokens <= 0:
            return None
        return n_tokens / denominator

    # ---------------------------------------------------------- batching ---
    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`batch_size` windows; each row's file is drawn in proportion to the split's tokens.

        Deterministic for a seeded `generator` (one integer is drawn from it per batch, as in
        :meth:`TokenDataset.get_batch`).
        """
        seed: int | None = None
        if generator is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        rng = np.random.default_rng(seed)
        files = rng.choice(len(self.names), size=batch_size, p=self.weights(split))
        xs, ys = [], []
        for idx in files:
            data = self.parts[self.names[idx]].split(split)
            hi = len(data) - block_size - 1
            if hi <= 0:
                raise ValueError(f"{self.names[idx]} {split} split is shorter than block_size + 1")
            o = int(rng.integers(0, hi))
            xs.append(data[o : o + block_size].astype(np.int64))
            ys.append(data[o + 1 : o + 1 + block_size].astype(np.int64))
        device_type = device.type if isinstance(device, torch.device) else "cpu"
        non_blocking = device_type == "cuda"
        xt = torch.from_numpy(np.stack(xs)).to(device=device, non_blocking=non_blocking)
        yt = torch.from_numpy(np.stack(ys)).to(device=device, non_blocking=non_blocking)
        return xt, yt

    def batches_per_epoch(self, batch_size: int, block_size: int, split: str = "train") -> int:
        n = self.n_train if split == "train" else self.n_val
        return max(1, (n - block_size - 1) // (batch_size * block_size))


@torch.no_grad()
def full_split_loss(
    model: torch.nn.Module,
    ds: TokenDataset,
    block_size: int,
    batch_size: int,
    device: torch.device,
    split: str = "val",
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, int]:
    """Exact mean loss (nats/token) over a whole split in non-overlapping windows.

    Returns ``(mean_loss, predicted_tokens)``. The tail that does not fill a whole window is not
    scored (at most ``block_size`` tokens). Every window is scored with the same weight, so the
    mean is over all predicted tokens.
    """
    data = ds.split(split)
    n_windows = (len(data) - 1) // block_size
    if n_windows <= 0:
        raise ValueError(f"{split} split is shorter than one window of {block_size + 1} tokens")
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for start in range(0, n_windows, batch_size):
        idx = range(start, min(start + batch_size, n_windows))
        x = np.stack([data[i * block_size : (i + 1) * block_size].astype(np.int64) for i in idx])
        y = np.stack([data[i * block_size + 1 : (i + 1) * block_size + 1].astype(np.int64) for i in idx])
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
        ):
            loss = model(xt, targets=yt).loss
        total += float(loss) * len(idx)
        count += len(idx)
    if was_training:
        model.train()
    return total / count, count * block_size
