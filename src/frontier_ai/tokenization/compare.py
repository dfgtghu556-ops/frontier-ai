"""Side-by-side comparison of tokenizer evaluation reports.

The evaluator and the comparator are deliberately separate: reports are plain JSON, so
comparisons can be re-run later over any set of artifacts without touching the evaluator
or re-encoding anything.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from .evaluate import EvalReport, StatBlock

SCHEMA_VERSION = "1.0"

# metric -> ("min" | "max", "what better looks like")
METRICS: dict[str, tuple[str, str]] = {
    "tokens_per_char": ("min", "lower = fewer tokens per character"),
    "chars_per_token": ("max", "higher = more compression"),
    "tokens_per_word": ("min", "lower = fewer tokens per whitespace word"),
    "unk_rate": ("min", "lower = fewer unknown tokens"),
    "round_trip_failures": ("min", "0 required: decode(encode(x)) must equal x"),
    "vocab_size": ("none", "context only, not a quality metric"),
}


def _row(report: EvalReport, label: str) -> dict[str, Any]:
    info = report.tokenizer
    overall = report.overall
    return {
        "label": label,
        "experiment_id": report.experiment_id,
        "impl": info.get("impl"),
        "impl_version": info.get("impl_version"),
        "vocab_size": info.get("vocab_size"),
        "tokens": overall.tokens,
        "chars": overall.chars,
        "words": overall.words,
        "tokens_per_char": overall.tokens_per_char,
        "chars_per_token": overall.chars_per_token,
        "tokens_per_word": overall.tokens_per_word,
        "unk_rate": overall.unk_rate,
        "round_trip_failures": overall.round_trip_failures,
        "utf8_bytes": overall.utf8_bytes,
        "bytes_per_token": round(overall.utf8_bytes / overall.tokens, 6) if overall.tokens else None,
        "lossless": overall.round_trip_failures == 0,
        "special_tokens_ok": bool(report.checks.get("special_tokens", {}).get("all_ok", False)),
        "per_language": {
            lang: {
                "tokens": stats.tokens,
                "chars": stats.chars,
                "tokens_per_char": stats.tokens_per_char,
                "chars_per_token": stats.chars_per_token,
                "unk_rate": stats.unk_rate,
                "round_trip_failures": stats.round_trip_failures,
            }
            for lang, stats in sorted(report.per_language.items())
        },
    }


def _best(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick a winner per metric, considering only lossless tokenizers.

    A tokenizer that cannot round-trip its input (unknown characters/words) can look
    artificially "compressed" while silently destroying text, so it is disqualified from
    winning rather than being silently ranked first.
    """
    eligible = [r for r in rows if r["lossless"]]
    pool = eligible or rows
    best: dict[str, Any] = {}
    for metric, (direction, _meaning) in METRICS.items():
        if direction == "none":
            continue
        values = [(r["label"], r.get(metric)) for r in pool if r.get(metric) is not None]
        if not values:
            continue
        pick = min if direction == "min" else max
        winner = pick(values, key=lambda kv: kv[1])
        best[metric] = {
            "tokenizer": winner[0],
            "value": winner[1],
            "direction": direction,
            "eligible": bool(eligible),
        }
    return best


def build_comparison(
    reports: Sequence[EvalReport],
    labels: Sequence[str] | None = None,
    corpus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable comparison document from evaluation reports."""
    if not reports:
        raise ValueError("no evaluation reports provided")
    if labels is None:
        labels = [r.tokenizer.get("impl", f"tokenizer-{i}") for i, r in enumerate(reports)]
    if len(labels) != len(reports):
        raise ValueError("labels and reports must have the same length")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate comparison labels: {sorted(labels)}")

    corpora = {r.corpus.get("corpus_id") + str(r.corpus.get("corpus_version")) for r in reports}
    if len(corpora) != 1:
        raise ValueError(
            "all reports must be produced against the same corpus version "
            f"(found {sorted(corpora)}) — comparisons across corpora are not meaningful"
        )

    rows = [_row(report, label) for report, label in zip(reports, labels)]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": corpus or reports[0].corpus,
        "metrics": {m: {"direction": d, "meaning": meaning} for m, (d, meaning) in METRICS.items()},
        "rows": rows,
        "best": _best(rows),
        "disqualified": [
            {"tokenizer": r["label"], "reason": f"{r['round_trip_failures']} round-trip failures"}
            for r in rows
            if not r["lossless"]
        ],
        "languages": sorted({lang for row in rows for lang in row["per_language"]}),
    }


def _fmt(value: Any, width: int) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:.3f}".rjust(width)
    return str(value).rjust(width)


def render_comparison(comparison: dict[str, Any]) -> str:
    """Human-readable rendering of a comparison document."""
    rows = comparison["rows"]
    headers = [
        "tokenizer", "impl", "vocab", "tokens", "chars/token", "bytes/token",
        "tok/char", "tok/word", "unk%", "rt_fail",
    ]
    lines: list[str] = []
    lines.append("OVERALL (same corpus, same examples for every row)")
    lines.append("  ".join(h.rjust(10) if h != "tokenizer" else h.ljust(18) for h in headers))
    lines.append("-" * (18 + 10 * (len(headers) - 1) + 2 * (len(headers) - 1)))
    for row in rows:
        unk = row["unk_rate"]
        unk_pct = "n/a" if unk is None else f"{100 * unk:.2f}"
        lines.append(
            "  ".join(
                [
                    str(row["label"]).ljust(18),
                    str(row["impl"]).rjust(10),
                    _fmt(row["vocab_size"], 10),
                    _fmt(row["tokens"], 10),
                    _fmt(row["chars_per_token"], 10),
                    _fmt(row["bytes_per_token"], 10),
                    _fmt(row["tokens_per_char"], 10),
                    _fmt(row["tokens_per_word"], 10),
                    unk_pct.rjust(10),
                    _fmt(row["round_trip_failures"], 10),
                ]
            )
        )

    disqualified = comparison.get("disqualified", [])
    if disqualified:
        lines.append("")
        lines.append("DISQUALIFIED FROM 'BEST' (cannot round-trip the input):")
        for item in disqualified:
            lines.append(f"  {item['tokenizer']:<18} {item['reason']}")

    best = comparison.get("best", {})
    if best:
        lines.append("")
        lines.append("BEST PER METRIC (lossless tokenizers only)")
        for metric, entry in best.items():
            lines.append(f"  {metric:<22} {entry['tokenizer']:<18} {entry['value']}  ({entry['direction']})")

    languages = comparison.get("languages", [])
    if languages:
        lines.append("")
        lines.append("CHARS PER TOKEN BY LANGUAGE (higher = more compression)")
        lines.append("language".ljust(12) + "".join(str(r["label"])[:16].rjust(18) for r in rows))
        lines.append("-" * (12 + 18 * len(rows)))
        for lang in languages:
            cells = []
            for row in rows:
                stats = row["per_language"].get(lang)
                cells.append(_fmt(stats["chars_per_token"] if stats else None, 18))
            lines.append(lang.ljust(12) + "".join(cells))
    return "\n".join(lines)


def render_statblock(stats: StatBlock, title: str) -> str:
    """Render a single StatBlock as text (used by the evaluate CLI)."""
    lines = [title, "-" * len(title)]
    lines.append(f"  examples              : {stats.examples}")
    lines.append(f"  characters (codepoints): {stats.chars}")
    lines.append(f"  utf-8 bytes           : {stats.utf8_bytes}")
    lines.append(f"  words (whitespace)    : {stats.words}")
    lines.append(f"  tokens                : {stats.tokens}")
    lines.append(f"  tokens per character  : {stats.tokens_per_char}")
    lines.append(f"  characters per token  : {stats.chars_per_token}")
    lines.append(f"  tokens per word       : {stats.tokens_per_word}")
    lines.append(f"  unknown tokens        : {stats.unk_count if stats.unk_count is not None else 'n/a'}")
    lines.append(
        f"  unknown rate          : {stats.unk_rate if stats.unk_rate is not None else 'n/a (no UNK token)'}"
    )
    lines.append(f"  round-trip failures   : {stats.round_trip_failures}")
    return "\n".join(lines)
