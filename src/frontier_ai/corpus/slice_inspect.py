"""Inspect one downloaded source file before any filtering decision (EXP-034).

Why this exists
---------------
MASTER_CONTEXT §37: measure before deciding. Before FrontierCorpus v2 chooses filter thresholds
for Sangraha, we need facts about what is actually in the files: how many documents, how long,
which web/OCR/speech ``type`` mix, how much text is in the *wrong script* (the datasets-server
preview already showed a non-Assamese Latin-script document in ``verified/asm``), how many exact
duplicates, how much touches our protected evaluation suite, and how many tokens the frozen v1
tokenizer would produce. This module computes those numbers in one streaming pass; it filters
nothing and changes nothing.

Every measurement reuses the existing pipeline definitions so the numbers mean the same thing
the future build will enforce: ``normalize_text`` (NFC policy), the langid script profile
(``script_profile`` / ``letter_total``, share >= ``DEFAULT_MIN_DECLARED_SHARE``), the quality
rules and thresholds (``QualityPolicy`` defaults) and the protected-suite guard (``SuiteGuard``).

Bounded cost on a 2-core laptop (documented, never hidden):

* the script profile reads the first ``PROFILE_CAP`` characters of a document (the report counts
  how many documents were longer than the cap);
* tokens are measured on a deterministic sample (every ``token_every``-th document, at most
  ``token_char_budget`` characters per file) and the file total is reported as an ESTIMATE with
  its sample size;
* duplicate detection keeps 16 bytes per distinct document for the current file only
  (cross-file duplicates are left to the build's dedup stage).
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any

from frontier_ai.corpus.decontaminate import SuiteGuard
from frontier_ai.corpus.langid import DEFAULT_MIN_DECLARED_SHARE, letter_total, script_profile, top_script
from frontier_ai.corpus.normalize import normalize_text
from frontier_ai.corpus.pipeline import PipelineDocument
from frontier_ai.corpus.quality import RULE_ORDER, QualityPolicy, _rule_results

PROFILE_CAP = 5000
SHARE_BUCKETS = (0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0000001)
QUANTILES = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
SAMPLE_CHARS = 160


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    last = len(ordered) - 1
    out = {f"p{round(q * 100)}": ordered[min(last, int(q * last + 0.5))] for q in QUANTILES}
    out["min"], out["max"] = ordered[0], ordered[-1]
    return out


def _bucket(share: float) -> str:
    lo = 0.0
    for hi in SHARE_BUCKETS:
        if share < hi:
            return f"[{lo:.2f},{min(hi, 1.0):.2f}{']' if hi > 1 else ')'}"
        lo = hi
    return "[1.00,1.00]"


_REDACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://\S+|www\.\S+|\d[\d \-]{4,}\d")


def _snippet(text: str) -> str:
    """A short, single-line excerpt for human review. E-mail addresses, URLs and long digit runs
    (phone numbers, ids) are masked: the report is committed, and D-044 requires PII care."""
    flat = " / ".join(part.strip() for part in text.splitlines() if part.strip())
    flat = _REDACT.sub("[masked]", flat)
    return flat[:SAMPLE_CHARS] + ("..." if len(flat) > SAMPLE_CHARS else "")


def inspect_rows(
    rows: Iterable[tuple[str, str, str]],
    *,
    source_id: str,
    language: str,
    script: str,
    guard: SuiteGuard | None,
    tokenizer: Any | None = None,
    token_every: int = 50,
    token_char_budget: int = 3_000_000,
    samples_per_kind: int = 3,
    max_docs: int | None = None,
    progress: Any = None,
    progress_every: int = 50_000,
) -> dict[str, Any]:
    """One streaming pass over ``(doc_id, type, text)`` rows -> a JSON-ready statistics dict."""
    policy = QualityPolicy()
    started = time.monotonic()
    n = 0
    raw_chars = raw_bytes = norm_chars = words = 0
    empty = nfc_changed = with_newlines = lines_total = profile_capped = 0
    lengths: list[int] = []
    kinds: Counter[str] = Counter()
    kinds_chars: Counter[str] = Counter()
    seen: set[bytes] = set()
    dup_docs = dup_chars = 0
    gate = Counter()
    gate_chars = Counter()
    share_hist: Counter[str] = Counter()
    mismatch_top: Counter[str] = Counter()
    kinds_gate_fail: Counter[str] = Counter()
    rule_hits = {r: 0 for r in RULE_ORDER}
    suite_reasons: Counter[str] = Counter()
    suite_docs: set[str] = set()
    suite_examples: list[dict[str, Any]] = []
    tok_docs = tok_chars = tok_tokens = 0
    samples: dict[str, list[dict[str, str]]] = {"pass": [], "script_mismatch": [], "no_letters": []}

    for doc_id, kind, raw in rows:
        if max_docs is not None and n >= max_docs:
            break
        n += 1
        raw_chars += len(raw)
        raw_bytes += len(raw.encode("utf-8"))
        kinds[kind] += 1
        text = normalize_text(raw)
        if text != raw.strip():
            nfc_changed += 1
        if not text:
            empty += 1
            lengths.append(0)
            continue
        norm_chars += len(text)
        kinds_chars[kind] += len(text)
        lengths.append(len(text))
        words += len(text.split())
        if "\n" in text:
            with_newlines += 1
        lines_total += text.count("\n") + 1

        digest = hashlib.sha256(text.encode("utf-8")).digest()[:16]
        if digest in seen:
            dup_docs += 1
            dup_chars += len(text)
        else:
            seen.add(digest)

        if len(text) > PROFILE_CAP:
            profile_capped += 1
        profile = script_profile(text[:PROFILE_CAP])
        letters = letter_total(profile)
        share = profile.get(script, 0) / letters if letters else 0.0
        if letters == 0:
            verdict = "no_letters"
        elif share < DEFAULT_MIN_DECLARED_SHARE:
            verdict = "script_mismatch"
            mismatch_top[top_script(profile)] += 1
        else:
            verdict = "pass"
        gate[verdict] += 1
        gate_chars[verdict] += len(text)
        if verdict != "pass":
            kinds_gate_fail[kind] += 1
        share_hist[_bucket(share)] += 1
        if len(samples[verdict]) < samples_per_kind:
            samples[verdict].append({"doc_id": doc_id, "type": kind, "chars": str(len(text)),
                                     "share": f"{share:.3f}", "text": _snippet(text)})

        doc = PipelineDocument(doc_id=doc_id, source_id=source_id, language=language, text=text)
        for rule, (violates, _value) in _rule_results(doc, policy).items():
            if violates:
                rule_hits[rule] += 1

        if guard is not None:
            hit = guard.check(text)
            if hit is not None:
                suite_reasons[hit.reason] += 1
                suite_docs.add(hit.suite_doc_id)
                if len(suite_examples) < 10:
                    suite_examples.append({"doc_id": doc_id, "reason": hit.reason,
                                           "suite_doc_id": hit.suite_doc_id,
                                           "matched_ngrams": hit.matched_ngrams})

        if tokenizer is not None and n % token_every == 0 and tok_chars < token_char_budget:
            tok_docs += 1
            tok_chars += len(text)
            tok_tokens += len(tokenizer.encode(text))

        if progress is not None and n % progress_every == 0:
            el = time.monotonic() - started
            progress(f"[inspect] {source_id}: {n:,} docs, {norm_chars / 1e6:,.1f}M chars, "
                     f"{n / max(el, 1e-9):,.0f} docs/s")

    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "source_id": source_id,
        "language": language,
        "declared_script": script,
        "documents": n,
        "documents_empty_after_normalize": empty,
        "normalize_changed_documents": nfc_changed,
        "raw_chars": raw_chars,
        "raw_bytes_utf8": raw_bytes,
        "chars": norm_chars,
        "words_whitespace": words,
        "doc_chars_quantiles": _quantiles(lengths),
        "documents_with_newlines": with_newlines,
        "lines_total": lines_total,
        "type_documents": dict(sorted(kinds.items())),
        "type_chars": dict(sorted(kinds_chars.items())),
        "exact_duplicates_within_file": {"documents": dup_docs, "chars": dup_chars},
        "script_gate": {
            "min_declared_share": DEFAULT_MIN_DECLARED_SHARE,
            "profile_cap_chars": PROFILE_CAP,
            "documents_longer_than_cap": profile_capped,
            "documents": dict(sorted(gate.items())),
            "chars": dict(sorted(gate_chars.items())),
            "declared_share_histogram": dict(sorted(share_hist.items())),
            "mismatch_top_script": dict(mismatch_top.most_common()),
            "failures_by_type": dict(sorted(kinds_gate_fail.items())),
        },
        "quality_rule_hits_default_policy": rule_hits,
        "suite": (
            {"checked": True, "hits_by_reason": dict(sorted(suite_reasons.items())),
             "documents_hit": sum(suite_reasons.values()),
             "distinct_suite_documents_first_hit": len(suite_docs), "examples": suite_examples}
            if guard is not None else {"checked": False}
        ),
        "tokens": (
            {"measured": True, "method": f"frozen v1 tokenizer on every {token_every}th document, "
                                         f"<= {token_char_budget:,} chars per file",
             "sample_documents": tok_docs, "sample_chars": tok_chars, "sample_tokens": tok_tokens,
             "tokens_per_char": (tok_tokens / tok_chars) if tok_chars else None,
             "estimated_file_tokens": round(norm_chars * tok_tokens / tok_chars) if tok_chars else None}
            if tokenizer is not None else {"measured": False}
        ),
        "samples": samples,
        "seconds": round(elapsed, 2),
        "documents_per_second": round(n / elapsed, 1) if elapsed > 0 else None,
        "mb_per_second": round(raw_bytes / 1e6 / elapsed, 3) if elapsed > 0 else None,
    }
    return report


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "-"


def render_text_report(run: dict[str, Any]) -> str:
    """Plain-text summary for NIGHT_REPORT / the founder (tables first, details after)."""
    lines = [
        f"{run['exp_id']} inspection of {run['slice_id']} (revision {run['revision'][:12]})",
        f"Attribution: {run['attribution']}",
        f"Suite check: {run['suite_status']}",
        "",
        "lang  docs        chars(M)  script-pass  wrong-script  no-letters  dup-docs  suite-hits  "
        "est.tokens(M)  docs/s",
    ]
    tot_docs = tot_chars = tot_tokens = 0
    for f in run["files"]:
        g = f["script_gate"]["documents"]
        tokens = f["tokens"].get("estimated_file_tokens")
        tot_docs += f["documents"]
        tot_chars += f["chars"]
        tot_tokens += tokens or 0
        suite = f["suite"]
        lines.append(
            f"{f['language']:<5} {f['documents']:<11,} {f['chars'] / 1e6:<9,.1f} "
            f"{_pct(g.get('pass', 0), f['documents']):<12} "
            f"{_pct(g.get('script_mismatch', 0), f['documents']):<13} "
            f"{_pct(g.get('no_letters', 0), f['documents']):<11} "
            f"{_pct(f['exact_duplicates_within_file']['documents'], f['documents']):<9} "
            f"{(str(suite['documents_hit']) if suite['checked'] else 'n/a'):<11} "
            f"{(f'{tokens / 1e6:,.1f}' if tokens else 'n/a'):<14} {f['documents_per_second'] or '-'}"
        )
    lines.append(f"TOTAL {tot_docs:<11,} {tot_chars / 1e6:<9,.1f} (estimated tokens: "
                 f"{tot_tokens / 1e6:,.1f}M - an ESTIMATE from a per-file sample, not a count)")
    lines.append("")
    lines.append("Document types (share of documents):")
    for f in run["files"]:
        kinds = ", ".join(f"{k or '(blank)'} {_pct(v, f['documents'])}"
                          for k, v in f["type_documents"].items())
        lines.append(f"  {f['language']}: {kinds}")
    lines.append("")
    lines.append("Samples (first " + str(SAMPLE_CHARS) + " characters; source: Sangraha, CC-BY-4.0):")
    for f in run["files"]:
        for verdict in ("pass", "script_mismatch", "no_letters"):
            for s in f["samples"][verdict][:2]:
                lines.append(f"  [{f['language']} {verdict} type={s['type']} share={s['share']}] {s['text']}")
    return "\n".join(lines) + "\n"
