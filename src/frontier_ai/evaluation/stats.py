"""Aggregation and uncertainty for document-attributed scores.

Bits-per-byte of a group of documents is a ratio of sums (total bits / total bytes), so
its uncertainty is estimated with a nonparametric **document bootstrap**: resample the
documents with replacement, recompute the ratio, take the 2.5 / 97.5 percentiles. This
measures how much the number would move with a different sample of documents from the
same distribution. It does NOT include seed-to-seed variation of training; comparisons
report that separately (mean ± std over seeds).

A **paired** bootstrap is used for comparisons: both models are scored on the same
documents (same suite), so resampling documents jointly cancels document difficulty and
gives much tighter intervals for the difference than two independent intervals.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ratio(bits: np.ndarray, denom: np.ndarray) -> float:
    total = float(denom.sum())
    return float(bits.sum()) / total if total > 0 else float("nan")


def bootstrap_ratio_ci(
    bits: np.ndarray,
    denom: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of sum(bits)/sum(denom) under document resampling (seeded)."""
    n = len(bits)
    if n < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = bits[idx].sum() / denom[idx].sum()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_delta(
    bits_a: np.ndarray,
    bits_b: np.ndarray,
    denom: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """(B − A) in bits per unit, with a paired document-bootstrap CI. Negative = B better."""
    diff = bits_b - bits_a
    point = ratio(diff, denom)
    lo, hi = bootstrap_ratio_ci(diff, denom, n_boot=n_boot, seed=seed, alpha=alpha)
    if lo > 0:
        verdict = "A better (interval excludes 0)"
    elif hi < 0:
        verdict = "B better (interval excludes 0)"
    else:
        verdict = "no clear difference (interval includes 0)"
    return {"delta_b_minus_a": point, "ci95": [lo, hi], "verdict": verdict}


def group_summary(
    bits: np.ndarray,
    bytes_: np.ndarray,
    chars: np.ndarray,
    tokens: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    bpb = ratio(bits, bytes_)
    lo, hi = bootstrap_ratio_ci(bits, bytes_, n_boot=n_boot, seed=seed)
    return {
        "documents": int(len(bits)),
        "bytes": int(bytes_.sum()),
        "chars": int(chars.sum()),
        "tokens": int(tokens.sum()),
        "bits": float(bits.sum()),
        "bits_per_byte": bpb,
        "bits_per_byte_ci95": [lo, hi],
        "bits_per_char": ratio(bits, chars),
        "bits_per_token": ratio(bits, tokens),
    }
