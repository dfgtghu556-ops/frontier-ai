"""Exact, document-attributed scoring of an evaluation token stream.

Protocol (harness v1):

* The evaluation documents are encoded one by one and concatenated in suite order —
  exactly how the training/validation streams are built (``prepare_exp_b_data.py``); the
  tokenizers carry no special tokens, so there is no separator.
* The stream is cut into consecutive, non-overlapping windows of ``block_size`` tokens.
  Window ``k`` feeds tokens ``[kB, kB+B)`` and is scored on targets ``[kB+1, kB+B+1)``,
  so **every token except the very first one of the stream is scored exactly once**.
* Each scored token's loss is credited to the document the token belongs to. Every
  token belongs to exactly one document, so per-document bits are exact and add up to
  the total.
* The first document supplies the unscored first token, so it is used as *context only*
  and excluded from every statistic. The byte/character denominators are therefore exact
  sums over the remaining documents — no token-length approximation.

Why not score each document in isolation: without a start-of-text token the first token
of each document cannot be predicted and must be skipped; a larger-vocabulary tokenizer's
first token covers more bytes, so isolated scoring would silently favour larger
vocabularies. Stream scoring has no such bias and matches the training distribution.
Tokens near the start of a window see less context than in a sliding-window protocol;
the protocol is identical for every model evaluated, and is recorded in every report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
PROTOCOL = (
    "stream: suite documents encoded individually and concatenated in suite order (no "
    "separator); consecutive non-overlapping windows of block_size; every token except the "
    "first of the stream scored once and credited to its document; the first document is "
    "context-only and excluded from statistics"
)


@dataclass(frozen=True)
class EncodedStream:
    ids: np.ndarray          # (N,) int64 token ids
    doc_of_token: np.ndarray  # (N,) int32 index of the document each token belongs to
    tokens_per_doc: np.ndarray  # (D,) int64


def encode_stream(tokenizer: Any, docs: Sequence[Any]) -> EncodedStream:
    pieces: list[list[int]] = [tokenizer.encode(d.text) for d in docs]
    lengths = np.array([len(p) for p in pieces], dtype=np.int64)
    if len(pieces) < 2 or lengths.sum() < 2:
        raise ValueError("need at least two documents and two tokens to evaluate")
    ids = np.fromiter((t for p in pieces for t in p), dtype=np.int64, count=int(lengths.sum()))
    doc_of_token = np.repeat(np.arange(len(pieces), dtype=np.int32), lengths)
    return EncodedStream(ids=ids, doc_of_token=doc_of_token, tokens_per_doc=lengths)


@torch.no_grad()
def token_nats(
    model: torch.nn.Module,
    ids: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Per-token negative log-likelihood (nats), float64, shape (N,); position 0 is NaN."""
    model.eval()
    n = len(ids)
    out = np.full(n, np.nan, dtype=np.float64)
    stream = torch.as_tensor(ids, dtype=torch.long)
    starts = list(range(0, n - 1, block_size))
    full = [s for s in starts if s + block_size + 1 <= n]
    tail = [s for s in starts if s + block_size + 1 > n]

    def run(batch_starts: list[int], length: int) -> None:
        x = torch.stack([stream[s : s + length] for s in batch_starts]).to(device)
        y = torch.stack([stream[s + 1 : s + length + 1] for s in batch_starts]).to(device)
        logits = model(x).logits.float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        nll = nll.reshape(len(batch_starts), length).double().cpu().numpy()
        for row, s in enumerate(batch_starts):
            out[s + 1 : s + length + 1] = nll[row]

    for i in range(0, len(full), batch_size):
        run(full[i : i + batch_size], block_size)
    for s in tail:  # the last, shorter window
        run([s], n - 1 - s)
    return out


def doc_bits(stream: EncodedStream, nats: np.ndarray) -> np.ndarray:
    """Total bits per document (document 0 includes only its scored tokens)."""
    bits = np.zeros(len(stream.tokens_per_doc), dtype=np.float64)
    scored = ~np.isnan(nats)
    np.add.at(bits, stream.doc_of_token[scored], nats[scored] / LN2)
    return bits
