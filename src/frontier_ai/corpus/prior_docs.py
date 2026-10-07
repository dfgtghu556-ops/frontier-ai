"""EXP-046 safeguard 2: the documents of an already packed corpus, as digests of their exact text.

Why this exists
---------------
v2-slice2 must not contain a document that is already in v2-slice1. On Kaggle, slice 1 exists only
as its packed token files (EXP-037: uint16 ``train || val``, one ``<|endoftext|>`` after every
document); its text files stay on the founder's PC. Frontier Tokenizer v2 is byte-level and
lossless (EXP-037 checked ``decode(encode(text)) == text``), so the bytes of a document's tokens
*are* the UTF-8 bytes of its text. :func:`document_digests` cuts a token file at its
``<|endoftext|>`` ids and hashes each document's bytes with the same 16-byte blake2b that
``slice_build.text_digest`` uses. Two documents get the same digest exactly when their texts are
identical (up to a 2^-128 collision), which is the same as having identical token sequences.

The bytes are gathered with numpy in batches of whole documents (no Python loop over tokens), so
the 2.78 B slice-1 tokens take minutes, not hours.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

DTYPE = np.dtype("<u2")
DEFAULT_BATCH_TOKENS = 4_000_000


def token_byte_table(tokenizer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(table, offsets, lengths)``: token ``i`` is ``table[offsets[i] : offsets[i] + lengths[i]]``."""
    pieces: list[bytes] = list(tokenizer._id_to_bytes)  # noqa: SLF001 - frozen artifact, read only
    lengths = np.fromiter((len(b) for b in pieces), dtype=np.int64, count=len(pieces))
    offsets = np.zeros(len(pieces), dtype=np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])
    table = np.frombuffer(b"".join(pieces), dtype=np.uint8)
    return table, offsets, lengths


def document_digests(
    bin_path: str | Path, tokenizer: Any, eot_id: int, batch_tokens: int = DEFAULT_BATCH_TOKENS
) -> list[bytes]:
    """16-byte blake2b of every document's text in a packed token file, in file order.

    The file must end with ``<|endoftext|>`` (as every EXP-037 file does); the end-of-text ids
    themselves are not part of any document.
    """
    tokens = np.memmap(bin_path, dtype=DTYPE, mode="r")
    if len(tokens) == 0:
        return []
    if int(tokens[-1]) != eot_id:
        raise ValueError(f"{bin_path}: does not end with <|endoftext|> ({eot_id})")
    table, offsets, lengths = token_byte_table(tokenizer)
    lengths = lengths.copy()
    lengths[eot_id] = 0  # the boundary contributes no bytes
    ends = np.flatnonzero(tokens == eot_id)
    digests: list[bytes] = []
    start, di = 0, 0
    while di < len(ends):
        dj = int(np.searchsorted(ends, start + batch_tokens, side="left"))
        dj = max(dj, di + 1)  # at least one document, however long
        stop = int(ends[dj - 1]) + 1
        ids = np.asarray(tokens[start:stop], dtype=np.int64)
        lens = lengths[ids]
        cum = np.cumsum(lens)
        total = int(cum[-1])
        gather = np.repeat(offsets[ids] - (cum - lens), lens) + np.arange(total, dtype=np.int64)
        buf = memoryview(table[gather].tobytes())
        prev = 0
        for end in cum[ends[di:dj] - start].tolist():
            digests.append(hashlib.blake2b(buf[prev:end], digest_size=16).digest())
            prev = end
        start, di = stop, dj
    return digests
