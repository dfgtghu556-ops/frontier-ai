"""Stage 3: quality filtering.

Why this exists
---------------
Remove text that teaches the model nothing or teaches it wrong — boilerplate,
fragments, garbage characters, degenerate repetition. MASTER_CONTEXT §15 is
explicit about the form this must take: measurable quality signals, one per
named rule, with the contributing values kept — **not** one mysterious
composite "quality score". A document is rejected only when a *reject* rule
violates; a *flag* rule keeps the document but records the rule, so a future
reader can answer "why was this kept/removed" for any document.

Rules (first set — all deterministic, stdlib-only)
--------------------------------------------------
reject:
* ``min_chars``      — document shorter than the floor (floor defaults to 1:
                       the pilot corpus's documents are single lines, many of
                       them verse, so the floor stays deliberately low until
                       per-language calibration in F3).
* ``max_chars``      — document longer than the ceiling (a truncation or a
                       parse accident, not a document).
* ``control_chars``  — invisible control characters above a ratio.
flag:
* ``url_density``    — URL markers per 1,000 characters above a ceiling
                       (navigation residue).
* ``template_residue`` — leftover wiki-template braces ``{{`` / ``}}``; the
                       P004B cleaner removes these, this is defence in depth.
* ``digit_runs``     — runs of 5+ ASCII digits (a phone number, an id, or a
                       scan accident; a printed year is 4 digits and passes).
* ``repetition``     — distinct token-bigram ratio below a floor; skipped for
                       documents too short to have bigrams (recorded as
                       ``insufficient_data`` in the detail, never as pass).

Thresholds live in :class:`QualityPolicy`: one default table plus an optional
per-language override. F3 calibrates the per-language tables on measured
pass rates; until then a threshold is the same for every language, and the
per-language statistics below make any systematic bias visible.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from frontier_ai.corpus.pipeline import PipelineDocument, Removal, StageOutcome

_URL_MARKER = re.compile(r"https?://|www\.")
_BRACES = re.compile(r"\{\{|\}\}")


def _digit_run_hits(text: str, min_length: int) -> int:
    return len(re.findall(r"[0-9]{" + str(int(min_length)) + r",}", text))

REJECT = "reject"
FLAG = "flag"

# Evaluation order; the first rejecting rule in this order is the recorded reason.
RULE_ORDER = (
    "min_chars",
    "max_chars",
    "control_chars",
    "url_density",
    "template_residue",
    "digit_runs",
    "repetition",
)

_ACTION = {
    "min_chars": REJECT,
    "max_chars": REJECT,
    "control_chars": REJECT,
    "url_density": FLAG,
    "template_residue": FLAG,
    "digit_runs": FLAG,
    "repetition": FLAG,
}


@dataclass(frozen=True)
class QualityPolicy:
    """Thresholds for the quality rules.

    ``per_language`` maps language -> rule name -> override value, e.g.
    ``{"hi": {"min_chars": 50}}``. An absent override falls back to the
    default field.
    """

    min_chars: int = 1
    max_chars: int = 20000
    max_control_ratio: float = 0.02
    max_url_density_per_1000: float = 2.0
    template_residue_limit: int = 0
    digit_run_length: int = 5
    min_repetition_ratio: float = 0.3
    repetition_min_tokens: int = 8
    per_language: dict[str, dict[str, Any]] = field(default_factory=dict)

    def threshold(self, language: str, rule: str) -> Any:
        value = self.per_language.get(language, {}).get(rule)
        return value if value is not None else getattr(self, rule)


_WS_CONTROL = frozenset("\t\n\r\f\v")  # handled by normalization, not garbage


def _control_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in _WS_CONTROL)
    return n / len(text)


def _repetition_ratio(text: str) -> float | None:
    """Distinct token-bigram ratio, or ``None`` when the text has too few tokens."""
    tokens = text.split()
    if len(tokens) < 2:
        return None
    bigrams = [tokens[i] + "\x00" + tokens[i + 1] for i in range(len(tokens) - 1)]
    return len(set(bigrams)) / len(bigrams)


def _rule_results(doc: PipelineDocument, policy: QualityPolicy) -> dict[str, tuple[bool, Any]]:
    """rule name -> (violates?, measured value). Measured values are recorded
    for *every* document whether or not the rule fires, so the statistics are
    complete rather than a sample of the guilty."""
    def th(rule: str) -> Any:
        return policy.threshold(doc.language, rule)

    ratio = _repetition_ratio(doc.text)
    control = _control_ratio(doc.text)
    url_hits = len(_URL_MARKER.findall(doc.text))
    url_density = url_hits * 1000.0 / max(doc.chars, 1)
    brace_hits = len(_BRACES.findall(doc.text))
    digit_hits = _digit_run_hits(doc.text, th("digit_run_length"))
    return {
        "min_chars": (doc.chars < th("min_chars"), doc.chars),
        "max_chars": (doc.chars > th("max_chars"), doc.chars),
        "control_chars": (control > th("max_control_ratio"), round(control, 6)),
        "url_density": (url_density > th("max_url_density_per_1000"), round(url_density, 3)),
        "template_residue": (brace_hits > th("template_residue_limit"), brace_hits),
        "digit_runs": (digit_hits > 0, digit_hits),
        "repetition": (
            ratio is not None and ratio < th("min_repetition_ratio"),
            round(ratio, 4) if ratio is not None else "insufficient_data",
        ),
    }


def quality_filter(documents: list[PipelineDocument], policy: QualityPolicy | None = None) -> StageOutcome:
    """Apply every rule to every document. Order is preserved.

    A document is removed on the first *reject* violation in ``RULE_ORDER``
    (the detail records the measured values of *all* rules, not only the
    guilty one); *flag* violations keep it and are recorded under
    ``flagged``.
    """
    policy = policy or QualityPolicy()
    kept: list[PipelineDocument] = []
    removed: list[Removal] = []
    flagged: dict[str, tuple[str, ...]] = {}
    per_rule: dict[str, dict[str, int]] = {r: {"rejects": 0, "flags": 0} for r in RULE_ORDER}
    per_language: dict[str, dict[str, int]] = {}
    for doc in documents:
        stats_l = per_language.setdefault(doc.language, {"in": 0, "kept": 0, "removed": 0})
        stats_l["in"] += 1
        results = _rule_results(doc, policy)
        rejects = [r for r in RULE_ORDER if results[r][0] and _ACTION[r] == REJECT]
        if rejects:
            removed.append(Removal(doc.doc_id, doc.source_id, doc.language, f"quality:{rejects[0]}",
                                   {r: v for r, (_, v) in results.items()}))
            stats_l["removed"] += 1
            for r in RULE_ORDER:
                if results[r][0]:
                    per_rule[r]["rejects" if _ACTION[r] == REJECT else "flags"] += 1
        else:
            flags = tuple(r for r in RULE_ORDER if results[r][0])
            kept.append(doc)
            stats_l["kept"] += 1
            if flags:
                flagged[doc.doc_id] = flags
                for r in flags:
                    per_rule[r]["flags"] += 1
    stats: dict[str, Any] = {
        "policy": {f: getattr(policy, f) for f in
                   ("min_chars", "max_chars", "max_control_ratio", "max_url_density_per_1000",
                    "template_residue_limit", "digit_run_length", "min_repetition_ratio",
                    "repetition_min_tokens")},
        "per_language_overrides": dict(policy.per_language),
        "documents_in": len(documents),
        "documents_out": len(kept),
        "removed": len(removed),
        "flagged_documents": len(flagged),
        "per_rule": per_rule,
        "per_language": per_language,
    }
    return StageOutcome(stage="quality", kept=tuple(kept), removed=tuple(removed),
                        flagged=flagged, stats=stats)
