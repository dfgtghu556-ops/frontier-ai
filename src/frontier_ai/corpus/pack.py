"""EXP-037: turn a built corpus file into training tokens (``.bin`` + ``.meta.json``).

One input file (``<lang>.jsonl.gz`` of FrontierCorpus v2-slice1, EXP-036) becomes one token file
in the layout :class:`frontier_ai.data.dataset.TokenDataset` reads: ``train || val`` as uint16,
with a side-car ``.meta.json``. Every document is encoded with Frontier Tokenizer v2's
``encode_ordinary`` (special-token strings inside text stay text) and followed by exactly one
``<|endoftext|>``.

Validation split: a document goes to validation when ``sha256(record id)`` falls in the lowest
``val_ppm`` parts per million (0.5% by default). The rule depends only on the record id, so it is
deterministic, independent of file order and the same on every machine.

Pre-registered checks (EXP-037), all enforced here; any failure raises :class:`PackError`:

* each document's token count equals its EXP-036 count (``tokens`` field, frontier-tokenizer-v1);
  so v2's ordinary encoding is checked against v1 on every document, not only on samples;
* train tokens + validation tokens - documents == the file's EXP-036 token total;
* in the written file, ``<|endoftext|>`` occurs exactly once per document, both splits end with
  it, and no other special id occurs (an independent scan of the file on disk);
* sampled documents decode back to exactly their text.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from array import array
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from frontier_ai.data.dataset import DataMeta

SCHEMA = "frontier-pack-v1"
VAL_PPM = 5_000  # 0.5% of documents
DTYPE = "uint16"
FLUSH_TOKENS = 1 << 22  # write buffers to disk every ~4M tokens (8 MB)
DECODE_CHECK_FIRST = 100
DECODE_CHECK_EVERY = 5_000
SCAN_BLOCK = 1 << 24


class PackError(RuntimeError):
    """A pre-registered packing check failed; the output must not be used."""


def is_validation(record_id: str, val_ppm: int = VAL_PPM) -> bool:
    """True when the document belongs to the validation split (deterministic, order-free)."""
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 1_000_000 < val_ppm


class OrdinaryEncoder:
    """``encode(text) == tokenizer.encode_ordinary(text)``, with a bounded cache per pre-token.

    Web text repeats the same words constantly; caching the ids of each pre-token avoids
    re-running the BPE merges (the same idea as ``slice_build.TokenCounter``, which counted the
    EXP-036 tokens). The tokenizer is not modified; a test compares against ``encode_ordinary``.
    """

    def __init__(self, tokenizer: Any, cache_limit: int = 500_000) -> None:
        from frontier_ai.tokenization.bpe_python import _PRETOKENIZERS

        self._pre = _PRETOKENIZERS[tokenizer._pretoken]  # noqa: SLF001 - frozen artifact, read only
        self._encode_chunk = tokenizer._encode_chunk  # noqa: SLF001
        self.n_ordinary = 256 + len(tokenizer.merges)
        self.cache: dict[str, tuple[int, ...]] = {}
        self.cache_limit = cache_limit

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        extend = out.extend
        cache = self.cache
        for chunk in self._pre(text):
            ids = cache.get(chunk)
            if ids is None:
                ids = tuple(self._encode_chunk(chunk))
                if len(cache) < self.cache_limit:
                    cache[chunk] = ids
            extend(ids)
        return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_token_file(path: Path, n_train: int, n_tokens: int, eot_id: int) -> dict[str, int]:
    """Independent check of a written file: end-of-text count, other special ids, split ends.

    ``<|endoftext|>`` is the first special id (32768), so any id above it is another special.
    """
    if n_tokens == 0:
        return {
            "tokens_scanned": 0,
            "endoftext": 0,
            "ids_above_endoftext": 0,
            "train_ends_with_endoftext": 1,
            "val_ends_with_endoftext": 1,
        }
    # Plain reads (no memory map), so nothing keeps the file open when the caller renames it
    # (Windows cannot rename a mapped file).
    dtype = np.dtype(DTYPE)
    eot = above = seen = 0
    train_last = last = -1
    with open(path, "rb") as fh:
        while seen < n_tokens:
            block = np.fromfile(fh, dtype=dtype, count=min(SCAN_BLOCK, n_tokens - seen))
            if block.size == 0:
                break
            eot += int(np.count_nonzero(block == eot_id))
            above += int(np.count_nonzero(block > eot_id))
            if seen <= n_train - 1 < seen + block.size:
                train_last = int(block[n_train - 1 - seen])
            seen += block.size
            last = int(block[-1])
    train_ends = int(n_train == 0 or train_last == eot_id)
    val_ends = int(n_tokens == n_train or last == eot_id)
    return {
        "tokens_scanned": seen,
        "endoftext": eot,
        "ids_above_endoftext": above,
        "train_ends_with_endoftext": train_ends,
        "val_ends_with_endoftext": val_ends,
    }


def pack_file(
    in_path: Path,
    out_path: Path,
    *,
    tokenizer: Any,
    encoder: OrdinaryEncoder,
    expected_docs: int | None,
    expected_tokens: int | None,
    val_ppm: int = VAL_PPM,
    max_docs: int | None = None,
    provenance: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Pack one ``<lang>.jsonl.gz`` into ``out_path`` (``.bin``) + ``.meta.json``; returns stats."""
    eot_id = tokenizer.special_token_ids["<|endoftext|>"]
    vocab_size = tokenizer.vocab_size
    if vocab_size > 1 << 16:
        raise PackError(f"vocab {vocab_size} does not fit {DTYPE}")
    if sys.byteorder != "little":  # array("H").tofile writes native order; the format is little-endian
        raise PackError("packing needs a little-endian machine")
    started = time.monotonic()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = {
        "train": out_path.with_name(out_path.name + ".train.part"),
        "val": out_path.with_name(out_path.name + ".val.part"),
    }
    counts = {s: {"docs": 0, "tokens": 0, "bytes": 0, "chars": 0} for s in parts}
    buffers = {s: array("H") for s in parts}
    val_ids: list[str] = []
    seen_ids: set[str] = set()
    source_tokens = 0
    decode_checked = 0
    handles = {s: open(p, "wb") for s, p in parts.items()}
    try:
        with gzip.open(in_path, "rt", encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if max_docs is not None and n >= max_docs:
                    break
                rec = json.loads(line)
                if rec["id"] in seen_ids:  # each record goes to exactly one split, so ids must be unique
                    raise PackError(f"{rec['id']}: record id appears twice")
                seen_ids.add(rec["id"])
                text = rec["text"]
                ids = encoder.encode(text)
                if len(ids) != rec["tokens"]:
                    raise PackError(
                        f"{rec['id']}: {len(ids)} tokens with frontier-tokenizer-v2 (ordinary), "
                        f"{rec['tokens']} recorded by EXP-036 with v1"
                    )
                if ids and max(ids) >= encoder.n_ordinary:
                    raise PackError(f"{rec['id']}: an ordinary encoding produced a special id")
                if n < DECODE_CHECK_FIRST or n % DECODE_CHECK_EVERY == 0:
                    if tokenizer.decode(ids) != text:
                        raise PackError(f"{rec['id']}: decode(encode(text)) != text")
                    decode_checked += 1
                split = "val" if is_validation(rec["id"], val_ppm) else "train"
                if split == "val":
                    val_ids.append(rec["id"])
                c = counts[split]
                c["docs"] += 1
                c["tokens"] += len(ids) + 1
                c["bytes"] += len(text.encode("utf-8"))
                c["chars"] += len(text)
                source_tokens += rec["tokens"]
                buf = buffers[split]
                buf.extend(ids)
                buf.append(eot_id)
                if len(buf) >= FLUSH_TOKENS:
                    buf.tofile(handles[split])
                    buffers[split] = array("H")
                if progress is not None and (n + 1) % 50_000 == 0:
                    progress(
                        f"[pack] {in_path.name}: {n + 1:,} documents ({time.monotonic() - started:.0f} s)"
                    )
        for split, buf in buffers.items():
            buf.tofile(handles[split])
    finally:
        for h in handles.values():
            h.close()

    docs = counts["train"]["docs"] + counts["val"]["docs"]
    n_train, n_val = counts["train"]["tokens"], counts["val"]["tokens"]
    if n_train + n_val - docs != source_tokens:
        raise PackError(f"token identity failed: {n_train} + {n_val} - {docs} != {source_tokens}")
    if expected_docs is not None and docs != expected_docs:
        raise PackError(f"{docs} documents read, the EXP-036 manifest says {expected_docs}")
    if expected_tokens is not None and source_tokens != expected_tokens:
        raise PackError(f"{source_tokens} tokens, the EXP-036 manifest says {expected_tokens}")

    # train || val: append the (small) validation part to the training part, so the disk never
    # holds two copies of a file
    with open(parts["train"], "ab") as dst, open(parts["val"], "rb") as src:
        for block in iter(lambda: src.read(1 << 22), b""):
            dst.write(block)
    parts["val"].unlink()
    tmp = out_path.with_name(out_path.name + ".tmp")
    parts["train"].replace(tmp)
    n_tokens = n_train + n_val
    if tmp.stat().st_size != 2 * n_tokens:
        raise PackError(f"{tmp}: {tmp.stat().st_size} bytes, expected {2 * n_tokens}")
    scan = scan_token_file(tmp, n_train, n_tokens, eot_id)
    if (
        scan["tokens_scanned"] != n_tokens
        or scan["endoftext"] != docs
        or scan["ids_above_endoftext"]
        or not scan["train_ends_with_endoftext"]
        or not scan["val_ends_with_endoftext"]
    ):
        raise PackError(f"{tmp}: file scan failed {scan}")
    tmp.replace(out_path)
    meta = DataMeta(
        n_tokens=n_tokens,
        vocab_size=vocab_size,
        dtype=DTYPE,
        level="bpe",
        n_train=n_train,
        n_val=n_val,
        n_bytes_train=counts["train"]["bytes"],
        n_bytes_val=counts["val"]["bytes"],
        n_chars_train=counts["train"]["chars"],
        n_chars_val=counts["val"]["chars"],
        source_provenance=dict(provenance) if provenance else None,
    )
    meta.save(out_path.with_suffix(".meta.json"))
    val_ids.sort()
    return {
        "input": in_path.name,
        "output": {
            "path": out_path.name,
            "meta": out_path.with_suffix(".meta.json").name,
            "sha256": _sha256_file(out_path),
            "bytes": out_path.stat().st_size,
            "dtype": DTYPE,
            "byteorder": "little",
            "layout": "train || val",
        },
        "docs": docs,
        "source_tokens_v1": source_tokens,
        "tokens": n_tokens,
        "splits": counts,
        "val_ids_sha256": hashlib.sha256("\n".join(val_ids).encode("utf-8")).hexdigest(),
        "checks": {
            "per_document_counts_equal_exp036": True,
            "record_ids_unique": len(seen_ids) == docs,
            "token_identity": f"{n_train} + {n_val} - {docs} == {source_tokens}",
            "file_scan": scan,
            "decode_checked_docs": decode_checked,
        },
        "max_docs": max_docs,
        "seconds": round(time.monotonic() - started, 1),
    }
