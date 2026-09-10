"""Token storage + batch sampling.

The prepared corpus is a flat array of token ids (numpy, memory-mapped so corpora
larger than RAM still work). Batches are built by sampling random offsets and
shifting by one position to form the standard next-token LM target.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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
    """Side-car metadata written next to the token file.

    ``n_bytes_*`` / ``n_chars_*`` are the UTF-8 byte and Unicode character counts of
    the text each split was encoded from. They are optional: corpora prepared before
    bits-per-byte reporting existed (Project 003 Stage 1) simply leave them ``None``,
    and every consumer must then report bits per byte as unknown rather than guess.
    """

    n_tokens: int
    vocab_size: int
    dtype: str
    level: str
    n_train: int
    n_val: int
    n_bytes_train: int | None = None
    n_bytes_val: int | None = None
    n_chars_train: int | None = None
    n_chars_val: int | None = None
    # Provenance of a real (licensed) corpus, copied verbatim from the file written by
    # scripts/fetch_smoke_corpus.py. None for the synthetic corpus and for any corpus
    # prepared without --provenance.
    source_provenance: dict | None = None

    # ------------------------------------------------------------- derived --
    @property
    def n_bytes(self) -> int | None:
        return _sum_or_none(self.n_bytes_train, self.n_bytes_val)

    @property
    def n_chars(self) -> int | None:
        return _sum_or_none(self.n_chars_train, self.n_chars_val)

    @property
    def has_text_lengths(self) -> bool:
        """True when byte/character counts are known for both splits."""
        return None not in (self.n_bytes_train, self.n_bytes_val, self.n_chars_train, self.n_chars_val)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DataMeta:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _sum_or_none(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return int(a) + int(b)


def text_lengths(pieces: Sequence[str]) -> tuple[list[int], list[int]]:
    """Per-piece UTF-8 byte and Unicode character counts.

    ``pieces`` must be the tokenizer's own surface pieces (``Tokenizer.tokenize``)
    so the counts add up to the length of the source text.
    """
    byte_lengths = [len(piece.encode("utf-8")) for piece in pieces]
    char_lengths = [len(piece) for piece in pieces]
    return byte_lengths, char_lengths


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

    # ------------------------------------------------- text length (bpb) ---
    def split_lengths(self, name: str) -> tuple[int | None, int | None]:
        """``(n_bytes, n_chars)`` of the text behind `name`, or ``(None, None)``.

        Unknown is reported as ``None`` — never 0, never an estimate.
        """
        if name == "train":
            return self.meta.n_bytes_train, self.meta.n_chars_train
        if name == "val":
            return self.meta.n_bytes_val, self.meta.n_chars_val
        raise ValueError(f"unknown split '{name}' (expected 'train' or 'val')")

    def tokens_per_byte(self, name: str = "val") -> float | None:
        """Tokens per UTF-8 byte, the factor that turns bits/token into bits/byte."""
        return self._tokens_per_unit(name, unit_index=0)

    def tokens_per_char(self, name: str = "val") -> float | None:
        """Tokens per Unicode character, for bits/character."""
        return self._tokens_per_unit(name, unit_index=1)

    def _tokens_per_unit(self, name: str, unit_index: int) -> float | None:
        denominator = self.split_lengths(name)[unit_index]  # 0 -> bytes, 1 -> chars
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
        # NOTE: torch.Generator.seed() *re-seeds* the generator with a new random value,
        # so calling it here would make sampling irreproducible even with a fixed seed.
        # Drawing an integer from the generator advances it deterministically instead:
        # same initial seed -> same sequence of batches (Project 003 Stage 1).
        seed: int | None = None
        if generator is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        rng = np.random.default_rng(seed)
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
    token_bytes: Sequence[int] | None = None,
    token_chars: Sequence[int] | None = None,
    source_provenance: dict | None = None,
) -> DataMeta:
    """Persist encoded ids + metadata. Returns the metadata object.

    ``token_bytes`` / ``token_chars`` are optional per-token byte and character
    counts (see :func:`text_lengths`). They are summed over each split so that
    loss can be reported per byte and per character, not only per token. Both
    must be supplied together and must align with ``ids``.

    ``source_provenance`` is the record written by ``scripts/fetch_smoke_corpus.py``
    for a real licensed corpus; it is copied into the metadata unchanged so every
    prepared corpus states where it came from and under which licence.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = _dtype_for(vocab_size)
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(out_path)
    n_val = int(len(arr) * val_frac)
    n_train = int(len(arr) - n_val)

    if (token_bytes is None) != (token_chars is None):
        raise ValueError("pass both token_bytes and token_chars, or neither")
    byte_splits: tuple[int | None, int | None] = (None, None)
    char_splits: tuple[int | None, int | None] = (None, None)
    if token_bytes is not None and token_chars is not None:
        if len(token_bytes) != len(arr) or len(token_chars) != len(arr):
            raise ValueError(
                f"length counts must align with the token stream: "
                f"{len(token_bytes)} bytes / {len(token_chars)} chars vs {len(arr)} tokens"
            )
        byte_splits = (sum(int(v) for v in token_bytes[:n_train]), sum(int(v) for v in token_bytes[n_train:]))
        char_splits = (sum(int(v) for v in token_chars[:n_train]), sum(int(v) for v in token_chars[n_train:]))

    meta = DataMeta(
        n_tokens=int(len(arr)),
        vocab_size=int(vocab_size),
        dtype=str(dtype),
        level=level,
        n_train=n_train,
        n_val=n_val,
        n_bytes_train=byte_splits[0],
        n_bytes_val=byte_splits[1],
        n_chars_train=char_splits[0],
        n_chars_val=char_splits[1],
        source_provenance=dict(source_provenance) if source_provenance else None,
    )
    meta.save(out_path.with_suffix(".meta.json"))
    return meta


def summary(ds: TokenDataset) -> dict[str, int | float | None]:
    out: dict[str, int | float | None] = {
        "n_tokens": int(ds.meta.n_tokens),
        "vocab_size": int(ds.meta.vocab_size),
        "n_train": int(ds.n_train),
        "n_val": int(ds.n_val),
    }
    if ds.meta.has_text_lengths:
        out["n_bytes"] = ds.meta.n_bytes
        out["n_chars"] = ds.meta.n_chars
        # corpus-wide averages over both splits
        out["bytes_per_token"] = round(ds.meta.n_bytes / ds.meta.n_tokens, 6)
        out["chars_per_token"] = round(ds.meta.n_chars / ds.meta.n_tokens, 6)
    if ds.meta.source_provenance:
        out["source_id"] = ds.meta.source_provenance.get("id")
        out["source_license"] = ds.meta.source_provenance.get("license_id")
    return out
