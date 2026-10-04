"""EXP-045 test 4: 13-token n-gram overlap between evaluation texts and a packed training stream.

The evaluation texts are tokenized with the training tokenizer; every run of 13 consecutive
tokens becomes a key. The training tokens (uint16 ``.bin`` files, EXP-037 format) are streamed in
chunks; every 13-token window is hashed with a vectorised polynomial hash (uint64, wrapping) and
looked up in the sorted key hashes. **Every hash hit is confirmed token by token**, so a hash
collision can never flag an item; it can only cost a little time.

An n-gram containing ``<|endoftext|>`` cannot match, because evaluation keys never contain it.
Texts shorter than 13 tokens cannot be checked; they are counted, not guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

N = 13
_MULT = np.uint64(0x9E3779B97F4A7C15)  # odd 64-bit multiplier (golden ratio)


def window_hashes(tokens: np.ndarray, n: int = N) -> np.ndarray:
    """uint64 hash of every length-``n`` window of ``tokens`` (len - n + 1 values)."""
    a = np.asarray(tokens).astype(np.uint64) + np.uint64(1)
    m = len(a) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * _MULT + a[j : j + m]
    return h


class NgramIndex:
    """Keys = 13-grams of named evaluation texts; ``refs[i]`` names the texts that contain key ``i``."""

    def __init__(self, n: int = N) -> None:
        self.n = n
        self._grams: dict[tuple[int, ...], set[str]] = {}
        self.too_short: list[str] = []
        self.texts = 0

    def add(self, ref: str, ids: Iterable[int]) -> None:
        ids = list(ids)
        self.texts += 1
        if len(ids) < self.n:
            self.too_short.append(ref)
            return
        for i in range(len(ids) - self.n + 1):
            self._grams.setdefault(tuple(ids[i : i + self.n]), set()).add(ref)

    def freeze(self) -> None:
        self.keys = np.array(list(self._grams), dtype=np.int64).reshape(-1, self.n)
        self.refs = [sorted(self._grams[tuple(k)]) for k in self.keys.tolist()]
        hashes = (
            window_hashes(self.keys.reshape(-1), self.n)[:: self.n]
            if len(self.keys)
            else np.zeros(0, np.uint64)
        )
        # (hashing the flattened keys gives each key's hash at multiples of n)
        self.order = np.argsort(hashes, kind="stable")
        self.sorted_hashes = hashes[self.order]

    def __len__(self) -> int:
        return len(self._grams)


def iter_chunks(tokens: np.ndarray, chunk: int, n: int = N) -> Iterator[tuple[int, np.ndarray]]:
    """``(offset, piece)`` with ``n - 1`` tokens of overlap, so every window is seen exactly once."""
    total = len(tokens)
    start = 0
    while start + n <= total:
        stop = min(total, start + chunk + n - 1)
        yield start, np.asarray(tokens[start:stop])
        start += chunk


def scan(index: NgramIndex, tokens: np.ndarray, chunk: int = 50_000_000) -> dict[str, dict[str, Any]]:
    """Confirmed matches per ref: ``{ref: {"windows": count, "first_offset": int}}``."""
    hits: dict[str, dict[str, Any]] = {}
    if not len(index.sorted_hashes):
        return hits
    sh = index.sorted_hashes
    for offset, piece in iter_chunks(tokens, chunk, index.n):
        h = window_hashes(piece, index.n)
        pos = np.searchsorted(sh, h)
        pos[pos == len(sh)] = len(sh) - 1
        cand = np.flatnonzero(sh[pos] == h)
        for c in cand.tolist():
            window = piece[c : c + index.n].astype(np.int64)
            j = int(pos[c])
            while j < len(sh) and sh[j] == h[c]:  # equal hashes are adjacent; confirm each
                k = int(index.order[j])
                if np.array_equal(index.keys[k], window):
                    for ref in index.refs[k]:
                        e = hits.setdefault(ref, {"windows": 0, "first_offset": offset + c})
                        e["windows"] += 1
                j += 1
    return hits


def train_tokens(bin_path: str | Path, n_train: int) -> np.ndarray:
    """The training part (``train || val`` layout) of an EXP-037 ``.bin`` file, memory-mapped."""
    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    if n_train > len(mm):
        raise ValueError(f"{bin_path}: n_train {n_train} > {len(mm)} tokens")
    return mm[:n_train]
