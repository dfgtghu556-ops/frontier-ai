"""Loss reporting that is comparable across tokenizers.

Per-token loss is the number the training loop optimises, but it is *not*
comparable between tokenizers: a char-level model predicting one byte-sized
piece at a time looks far better per token than a BPE model predicting four
characters at a time, even when the char model is worse at modelling the text.

The fix is to divide by the amount of text each token represents:

    bits_per_token  = nats_per_token / ln 2
    bits_per_byte   = bits_per_token x tokens_per_byte
    bits_per_char   = bits_per_token x tokens_per_char

``tokens_per_byte`` / ``tokens_per_char`` come from the corpus metadata
(:class:`frontier_ai.data.dataset.DataMeta`), which records how many UTF-8 bytes
and Unicode characters each split was encoded from. When a corpus predates that
bookkeeping the ratios are unknown, and every function here returns ``None`` for
the derived quantity instead of inventing a denominator.
"""

from __future__ import annotations

import math

NATS_TO_BITS = 1.0 / math.log(2.0)


def bits_from_nats(nats: float) -> float:
    """Convert a loss in nats per token to bits per token."""
    return nats * NATS_TO_BITS


def perplexity(nats: float, cap: float = 20.0) -> float:
    """Token-level perplexity, capped so a diverged run cannot print inf.

    The cap is a reporting device only: ``exp(20)`` is ~4.9e8, far past any
    useful number, and it keeps the JSON machine-readable.
    """
    return math.exp(min(cap, nats))


def bits_per_unit(nats: float, tokens_per_unit: float | None) -> float | None:
    """Bits per byte (or per character) from nats per token.

    Returns ``None`` when ``tokens_per_unit`` is unknown or non-positive — never
    0, never a guessed value.
    """
    if tokens_per_unit is None or tokens_per_unit <= 0:
        return None
    return bits_from_nats(nats) * tokens_per_unit


def loss_summary(
    nats: float,
    *,
    tokens_per_byte: float | None = None,
    tokens_per_char: float | None = None,
    ppl_cap: float = 20.0,
    decimals: int = 6,
) -> dict[str, float | None]:
    """One comparable loss report.

    Keys are stable and machine-readable; ``bits_per_byte`` / ``bits_per_char``
    are ``None`` (not 0) when the corpus has no byte/character counts.
    """
    def rnd(value: float | None) -> float | None:
        if value is None or value != value:  # None or NaN
            return None
        return round(value, decimals)

    return {
        "val_loss": rnd(nats),  # nats per token - the training objective
        "val_ppl": round(perplexity(nats, ppl_cap), 3),
        "bits_per_token": rnd(bits_from_nats(nats)),
        "bits_per_byte": rnd(bits_per_unit(nats, tokens_per_byte)),
        "bits_per_char": rnd(bits_per_unit(nats, tokens_per_char)),
        "tokens_per_byte": rnd(tokens_per_byte),
        "tokens_per_char": rnd(tokens_per_char),
    }


def bpb_note(byte_counts_known: bool) -> str:
    """Human-readable note explaining why a bits-per-byte field is or is not there."""
    if byte_counts_known:
        return (
            "bits_per_byte converts the per-token loss with this corpus's recorded "
            "byte counts, so it is comparable across tokenizers; the per-token "
            "numbers are not."
        )
    return (
        "bits_per_byte is null: this corpus was prepared without byte/character "
        "counts. Re-run scripts/prepare_data.py to record them."
    )
