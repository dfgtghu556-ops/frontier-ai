"""Build FrontierCorpus v2 from the pinned Sangraha Verified slice (EXP-036).

Why this exists
---------------
EXP-034 measured the slice and EXP-035 read samples of everything a rule would remove. The founder
approved the resulting ten rules ("approve EXP-036", 2026-09-29). This module applies exactly those
rules, one file at a time, and records **why every removed document or line was removed**, so each
rule's cost can be checked per language before the corpus is accepted.

The rules, in the order they are applied to one document (rule numbers from the EXP-036 table in
EXPERIMENTS.md):

====  =====================  ==================================================================
 #    rule                   what happens
====  =====================  ==================================================================
 1    protected suite        drop the document if it contains a suite text exactly, shares a
                             13-gram with the suite, or contains a suite passage of >= 3 words
                             word for word (checked on the normalized source text)
 2a   exact duplicates       keep the first copy of an identical normalized text (within file)
 --   control characters     v1 rule kept unchanged (> 2% invisible control characters; 0 hits
                             in EXP-034/035)
 3    boilerplate lines      drop lines that occur >= 100 times (exactly) in the file
 4    foreign lines          drop lines that contain letters but no letter of the declared script
 5    script gate            drop if the declared script is < 60% of the letters of the WHOLE
                             cleaned document (or it has no letters)
 6    Urdu check (ur only)   drop Persian-like (no Urdu-only letter among >= 50 letters) and
                             Uyghur-like documents (``slice_calibrate.marker_label``)
 7    long documents         no length limit
 8    wiki markup            drop at >= 2 ``{{``/``}}`` markers
 9    repetition             drop if distinct word-bigram ratio < 0.3 (>= 8 words)
 10   contacts               nothing dropped; e-mail addresses -> ``[email]``, phone numbers ->
                             ``[phone]``
 1    suite, again           the exact output text is checked again (13-gram, exact, short
                             passages); a document that touches the suite is dropped
 2b   near-duplicates        among the documents that passed everything above: keep the first
                             document of each MinHash cluster (estimated Jaccard >= 0.8)
====  =====================  ==================================================================

Near-duplicates are detected last, among documents that passed every other rule, so a cluster
keeps a document that is actually in the corpus instead of one that a later rule removes. The
second suite check exists because removing lines joins the words around them, which could in
principle create a 13-gram that the source text did not have.

Three streaming passes per file, bounded memory (one 8-byte hash per line, one 128 x uint32
signature per surviving document, one 16-byte hash per document):

A. count every exact line (8-byte blake2b hash) -> the set of boilerplate lines;
B. apply the rules, compute the final text, its exact token count with the frozen tokenizer and its
   MinHash signature, and append every candidate to a temporary gzip file;
C. cluster the signatures and copy the candidates that are not near-duplicates to the final
   ``<lang>.jsonl.gz`` (deterministic bytes: gzip ``mtime=0``).

Every output record carries its provenance: source file, row in the parquet file, Sangraha
``doc_id`` (not unique, EXP-035), document type and dataset revision.

Audit samples (masked excerpts, <= 400 characters) are drawn per removal reason and from the kept
documents, with a fixed seed per (file, stratum). Every sampled text passed the suite check on its
full source text first; documents removed for touching the suite are never sampled as documents (a
removed foreign line of such a document can be: it is a part of a text that passed the check).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import unicodedata
from array import array
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_ai.corpus.decontaminate import ShortSuiteIndex, SuiteGuard
from frontier_ai.corpus.langid import _SCRIPT_RANGES, DEFAULT_MIN_DECLARED_SHARE, letter_total, script_profile
from frontier_ai.corpus.normalize import normalize_text
from frontier_ai.corpus.quality import _BRACES, _control_ratio, _repetition_ratio
from frontier_ai.corpus.slice_calibrate import (
    FILE_MARKERS,
    MinHasher,
    Reservoir,
    _line_hash,
    excerpt,
    marker_counts,
    marker_label,
    near_duplicate_clusters,
)

SCHEMA = "frontier-slice-build-v1"
SAMPLE_EXCERPT = 400

# Removal reasons in the order they are tested (the first failing test is the recorded reason).
REASONS = (
    "empty",
    "suite_exact",
    "suite_ngram",
    "suite_short",
    "exact_duplicate",
    "control_chars",
    "empty_after_cleaning",
    "no_letters",
    "script_share",
    "urdu_persian_like",
    "urdu_uyghur_like",
    "wiki_markup",
    "repetition",
    "suite_after_cleaning",
    "near_duplicate",
)
RULE_OF_REASON = {
    "empty": "normalization (v1)",
    "suite_exact": "1 protected suite",
    "suite_ngram": "1 protected suite",
    "suite_short": "1 protected suite",
    "exact_duplicate": "2 duplicates",
    "control_chars": "control chars (v1)",
    "empty_after_cleaning": "3/4 lines",
    "no_letters": "5 script gate",
    "script_share": "5 script gate",
    "urdu_persian_like": "6 Urdu check",
    "urdu_uyghur_like": "6 Urdu check",
    "wiki_markup": "8 wiki markup",
    "repetition": "9 repetition",
    "suite_after_cleaning": "1 protected suite",
    "near_duplicate": "2 duplicates",
}
SUITE_REASONS = frozenset({"suite_exact", "suite_ngram", "suite_short", "suite_after_cleaning"})
NOT_SAMPLED = SUITE_REASONS | {"empty"}

# Pre-registered review bounds (EXPERIMENTS.md, EXP-036). A share of the file's characters above
# the bound does not stop the build; it marks the rule REVIEW in the report so it is read before
# the corpus is accepted. Keys: removal reason, or "lines:<kind>" for removed lines.
REVIEW_BOUNDS = {
    "lines:boilerplate": 0.05,  # EXP-035 max 2.76% (ml)
    "lines:foreign": 0.10,  # not measurable in EXP-035
    "exact_duplicate": 0.05,
    "near_duplicate": 0.05,  # EXP-035 max 1.86% of docs (hi)
    "script_share": 0.08,  # EXP-035 max 5.45% of docs (as)
    "no_letters": 0.02,
    "urdu_persian_like": 0.12,  # EXP-035 ~8.4% of ur chars
    "urdu_uyghur_like": 0.01,
    "wiki_markup": 0.02,
    "repetition": 0.02,
    "empty_after_cleaning": 0.02,
    "control_chars": 0.001,
    "suite_after_cleaning": 0.0,  # any hit is worth reading
}

SAMPLES_PER_REASON = 4
SAMPLES_FOREIGN_LINES = 8
SAMPLES_KEPT = 6
SAMPLES_KEPT_CLEANED = 4
TOP_BOILERPLATE = 25
SHORT_KEPT_CHARS = 200


@dataclass(frozen=True)
class BuildConfig:
    """Every threshold of the build. Its fingerprint is recorded with the output."""

    boilerplate_min_repeats: int = 100
    min_declared_share: float = DEFAULT_MIN_DECLARED_SHARE
    urdu_min_letters: int = 50  # as in slice_calibrate.marker_label (EXP-035 counts)
    wiki_marker_limit: int = 2
    min_repetition_ratio: float = 0.3
    repetition_min_tokens: int = 8
    max_control_ratio: float = 0.02
    near_dup_threshold: float = 0.8
    near_dup_bands: int = 16
    minhash_perm: int = 128
    minhash_shingle: int = 5
    minhash_seed: int = 35
    short_suite_min_words: int = 3
    normalization: str = "nfc"

    def fingerprint(self) -> str:
        blob = json.dumps({"schema": SCHEMA, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ foreign lines --
def _letter_class(script: str) -> re.Pattern[str]:
    """A regex matching one *letter* (Unicode category L*) of ``script``, as ``script_profile`` counts."""
    ranges = dict(_SCRIPT_RANGES)[script]
    runs: list[tuple[int, int]] = []
    for lo, hi in ranges:
        for cp in range(lo, hi + 1):
            if unicodedata.category(chr(cp))[0] != "L":
                continue
            if runs and runs[-1][1] == cp - 1:
                runs[-1] = (runs[-1][0], cp)
            else:
                runs.append((cp, cp))
    body = "".join(
        re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in runs
    )
    return re.compile(f"[{body}]")


_LETTER_CLASS_CACHE: dict[str, re.Pattern[str]] = {}
_MAYBE_LETTER = re.compile(r"[^\W\d_]")  # alphanumeric minus digits: every letter, plus a few numerals


def declared_letter_re(script: str) -> re.Pattern[str]:
    pattern = _LETTER_CLASS_CACHE.get(script)
    if pattern is None:
        pattern = _LETTER_CLASS_CACHE[script] = _letter_class(script)
    return pattern


def is_foreign_line(line: str, script: str) -> bool:
    """True when ``line`` has at least one letter and none of them is in the declared script.

    Letters in blocks the profiler does not know (``unknown``: e.g. CJK, Cyrillic) count as
    letters, so a Chinese line inside a Tamil page is removed. Fast path by regex; the exact
    ``script_profile`` decides only for candidate lines (a test compares with the reference).
    """
    if declared_letter_re(script).search(line):
        return False
    if not _MAYBE_LETTER.search(line):
        return False
    profile = script_profile(line)
    return letter_total(profile) + profile.get("unknown", 0) > 0


def is_foreign_line_reference(line: str, script: str) -> bool:
    """The definition, written directly on the profile (for the equivalence test)."""
    profile = script_profile(line)
    letters = letter_total(profile) + profile.get("unknown", 0)
    return letters > 0 and profile.get(script, 0) == 0


@dataclass
class CleanResult:
    text: str
    boilerplate_lines: int = 0
    boilerplate_chars: int = 0
    foreign_lines: int = 0
    foreign_chars: int = 0
    removed_boilerplate: list[tuple[int, str]] = field(default_factory=list)  # (hash, line)
    removed_foreign: list[str] = field(default_factory=list)

    @property
    def changed_lines(self) -> int:
        return self.boilerplate_lines + self.foreign_lines


def clean_lines(text: str, script: str, boilerplate: frozenset[int] | set[int]) -> CleanResult:
    """Rules 3 and 4 on one normalized document.

    Every line is stripped; a removed line disappears with its line break; runs of blank lines
    collapse to one blank line (paragraph breaks survive); leading/trailing blank lines go.
    """
    out: list[str] = []
    res = CleanResult(text="")
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if boilerplate:
            h = _line_hash(line)
            if h in boilerplate:
                res.boilerplate_lines += 1
                res.boilerplate_chars += len(line)
                res.removed_boilerplate.append((h, line))
                continue
        if is_foreign_line(line, script):
            res.foreign_lines += 1
            res.foreign_chars += len(line)
            res.removed_foreign.append(line)
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    res.text = "\n".join(out)
    return res


# ------------------------------------------------------------------------ contacts --
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Phone numbers: (a) international form "+CC ..." with 8-13 more digits; (b) Indian mobile form
# 98765 43210 / 98765-43210 (first digit 6-9); (c) 10-13 contiguous digits (also 0/91 prefixed
# mobiles). Year lists such as "1991 1992 1993" are not matched: separators are only allowed in
# forms (a) and (b). \d includes Indic digits (e.g. Bengali), which is intended.
_PHONE = re.compile(
    r"(?<![\w+])\+\d{1,3}(?:[ -]?\d){8,13}(?!\d)"
    r"|(?<!\d)[6-9]\d{4}[ -]\d{5}(?!\d)"
    r"|(?<![\d.,])\d{10,13}(?![\d.,])"
)
EMAIL_MASK = "[email]"
PHONE_MASK = "[phone]"


def mask_contacts(text: str) -> tuple[str, int, int]:
    """Rule 10: replace e-mail addresses and phone numbers -> (text, e-mails, phones)."""
    text, emails = _EMAIL.subn(EMAIL_MASK, text)
    text, phones = _PHONE.subn(PHONE_MASK, text)
    return text, emails, phones


# ------------------------------------------------------------------ token counting --
class TokenCounter:
    """Exact token count with a frozen ``PythonBPE``: ``count(t) == len(tokenizer.encode(t))``.

    ``PythonBPE.encode`` has no cache, so on web text most time goes into re-merging the same
    words. This counter uses the tokenizer's own pre-tokenizer and ``_encode_chunk`` and caches
    the token count per pre-token (bounded). The tokenizer itself is not modified; a test compares
    the counts with ``len(encode(...))`` on mixed text, including special-token strings.
    """

    def __init__(self, tokenizer: Any, cache_limit: int = 1_000_000) -> None:
        from frontier_ai.tokenization.base import iter_segments
        from frontier_ai.tokenization.bpe_python import _PRETOKENIZERS

        self.tokenizer = tokenizer
        self._iter_segments = iter_segments
        self._pre = _PRETOKENIZERS[tokenizer._pretoken]  # noqa: SLF001 - frozen artifact, read only
        self._specials = tokenizer._special_tokens  # noqa: SLF001
        self._encode_chunk = tokenizer._encode_chunk  # noqa: SLF001
        self.cache: dict[str, int] = {}
        self.cache_limit = cache_limit

    def count(self, text: str) -> int:
        cache = self.cache
        total = 0
        for segment, is_special in self._iter_segments(text, self._specials):
            if is_special:
                total += 1
                continue
            for chunk in self._pre(segment):
                n = cache.get(chunk)
                if n is None:
                    n = len(self._encode_chunk(chunk))
                    if len(cache) < self.cache_limit:
                        cache[chunk] = n
                total += n
        return total


# ------------------------------------------------------------------ MT artefacts --
def fused_latin_re(script: str) -> re.Pattern[str] | None:
    """Latin letter directly followed by a combining mark of ``script`` ("architectਾਂ").

    Impossible in correctly written text; a machine-translation / OCR artefact (measured only).
    """
    if script in ("latin", "arabic"):
        return None
    ranges = dict(_SCRIPT_RANGES)[script]
    marks = "".join(
        chr(cp) for lo, hi in ranges for cp in range(lo, hi + 1) if unicodedata.category(chr(cp))[0] == "M"
    )
    return re.compile(f"[A-Za-z][{re.escape(marks)}]") if marks else None


# ------------------------------------------------------------------------ the build --
@dataclass
class _Counts:
    docs: Counter[str] = field(default_factory=Counter)
    chars: Counter[str] = field(default_factory=Counter)

    def add(self, key: str, chars: int) -> None:
        self.docs[key] += 1
        self.chars[key] += chars


def _share(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


def pass_a_boilerplate(
    rows: Iterable[tuple[str, str, str]], min_repeats: int, normalization: str = "nfc"
) -> tuple[frozenset[int], dict[str, int]]:
    """Pass A: exact line counts -> (hashes of lines seen >= ``min_repeats`` times, stats)."""
    hashes = array("Q")
    docs = 0
    for _, _, raw in rows:
        docs += 1
        for line in normalize_text(raw, normalization).split("\n"):
            line = line.strip()
            if line:
                hashes.append(_line_hash(line))
    if not hashes:
        return frozenset(), {"docs": docs, "lines": 0, "distinct_lines": 0, "boilerplate_distinct_lines": 0}
    uniq, counts = np.unique(np.frombuffer(hashes, dtype=np.uint64), return_counts=True)
    hot = uniq[counts >= min_repeats]
    stats = {
        "docs": docs,
        "lines": len(hashes),
        "distinct_lines": int(len(uniq)),
        "boilerplate_distinct_lines": int(len(hot)),
        "boilerplate_line_occurrences": int(counts[counts >= min_repeats].sum()),
    }
    count_of = dict(zip(hot.tolist(), counts[counts >= min_repeats].tolist(), strict=True))
    boiler = frozenset(count_of)
    stats["_counts"] = count_of  # removed before publishing; used to rank the top lines
    return boiler, stats


def build_file(
    rows: Callable[[], Iterable[tuple[str, str, str]]],
    *,
    out_path: Path,
    exp_id: str,
    source_id: str,
    language: str,
    script: str,
    revision: str,
    guard: SuiteGuard,
    short_index: ShortSuiteIndex,
    token_counter: TokenCounter,
    config: BuildConfig | None = None,
    expected_rows: int | None = None,
    max_docs: int | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 50_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one file -> (statistics, audit samples). ``rows()`` must yield the same rows each call.

    Writes ``out_path`` (``<lang>.jsonl.gz``) atomically via a ``.tmp`` file; a temporary candidate
    file next to it is deleted at the end.
    """
    cfg = config or BuildConfig()
    if short_index.min_words != cfg.short_suite_min_words:
        raise ValueError(
            f"short-suite index built with min_words={short_index.min_words}, "
            f"config says {cfg.short_suite_min_words}"
        )
    say = progress or (lambda _m: None)
    started = time.monotonic()

    def limited() -> Iterable[tuple[str, str, str]]:
        for i, row in enumerate(rows()):
            if max_docs is not None and i >= max_docs:
                return
            yield row

    # ---- pass A ----
    boiler, line_stats = pass_a_boilerplate(limited(), cfg.boilerplate_min_repeats, cfg.normalization)
    boiler_counts: dict[int, int] = line_stats.pop("_counts", {})
    n_rows = line_stats["docs"]
    if expected_rows is not None and max_docs is None and n_rows != expected_rows:
        raise ValueError(f"{source_id}: read {n_rows} rows, parquet metadata says {expected_rows}")
    say(
        f"[build] {source_id}: pass A done ({n_rows:,} docs, {line_stats['lines']:,} lines, "
        f"{line_stats['boilerplate_distinct_lines']:,} boilerplate lines, "
        f"{(time.monotonic() - started) / 60:.1f} min)"
    )

    # ---- pass B ----
    mh = MinHasher(cfg.minhash_perm, cfg.minhash_shingle, cfg.minhash_seed)
    signatures = np.zeros((max(n_rows, 1), mh.num_perm), dtype=np.uint32)
    valid = np.zeros(max(n_rows, 1), dtype=bool)
    candidate_rows: list[int] = []
    cand_chars: dict[int, tuple[int, int, str]] = {}  # row -> (chars, tokens, type) for near-dup stats
    removed = _Counts()
    input_chars = 0
    input_by_type = _Counts()
    seen_exact: set[bytes] = set()
    line_totals = Counter()
    boiler_text: dict[int, str] = {}
    family = FILE_MARKERS.get(language) if language == "ur" else None
    fused = fused_latin_re(script)
    measured = Counter()
    measured_chars = Counter()
    masks = Counter()
    res = {
        r: Reservoir(SAMPLES_PER_REASON, f"{exp_id}/{source_id}/{r}") for r in REASONS if r not in NOT_SAMPLED
    }
    res_foreign = Reservoir(SAMPLES_FOREIGN_LINES, f"{exp_id}/{source_id}/foreign_line")
    res_kept = Reservoir(SAMPLES_KEPT, f"{exp_id}/{source_id}/kept")
    res_kept_cleaned = Reservoir(SAMPLES_KEPT_CLEANED, f"{exp_id}/{source_id}/kept_lines_removed")
    suite_ids: Counter[str] = Counter()

    def sample(reason: str, row: int, doc_id: str, kind: str, text: str, **extra: Any) -> None:
        res[reason].offer(
            lambda: {
                "exp_id": exp_id,
                "source_id": source_id,
                "language": language,
                "stratum": f"removed:{reason}",
                "row": row,
                "doc_id": doc_id,
                "type": kind,
                "chars": len(text),
                "excerpt": excerpt(text, SAMPLE_EXCERPT),
                **extra,
            }
        )

    tmp_candidates = out_path.with_name(out_path.name + ".candidates.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(tmp_candidates, "wb", compresslevel=1) as cand:
        for row, (doc_id, kind, raw) in enumerate(limited()):
            if row and row % progress_every == 0:
                say(
                    f"[build] {source_id}: pass B {row:,}/{n_rows:,} "
                    f"({(time.monotonic() - started) / 60:.1f} min)"
                )
            text = normalize_text(raw, cfg.normalization)
            chars = len(text)
            input_chars += chars
            input_by_type.add(kind, chars)
            if not text:
                removed.add("empty", 0)
                continue
            # rule 1: protected suite on the normalized source text
            hit = guard.check(text)
            if hit is not None:
                removed.add(hit.reason, chars)
                suite_ids[hit.suite_doc_id] += 1
                continue
            short = short_index.find(text, limit=1)
            if short:
                removed.add("suite_short", chars)
                suite_ids[short[0].suite_doc_id] += 1
                continue
            # rule 2a: exact duplicates
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            if digest in seen_exact:
                removed.add("exact_duplicate", chars)
                sample("exact_duplicate", row, doc_id, kind, text)
                continue
            seen_exact.add(digest)
            if _control_ratio(text) > cfg.max_control_ratio:
                removed.add("control_chars", chars)
                sample("control_chars", row, doc_id, kind, text)
                continue
            # rules 3 + 4: lines
            cleaned = clean_lines(text, script, boiler)
            line_totals["boilerplate_lines"] += cleaned.boilerplate_lines
            line_totals["boilerplate_chars"] += cleaned.boilerplate_chars
            line_totals["foreign_lines"] += cleaned.foreign_lines
            line_totals["foreign_chars"] += cleaned.foreign_chars
            for h, line in cleaned.removed_boilerplate:
                if h not in boiler_text:
                    boiler_text[h] = line
            for line in cleaned.removed_foreign:
                res_foreign.offer(
                    lambda line=line, row=row, doc_id=doc_id: {
                        "exp_id": exp_id,
                        "source_id": source_id,
                        "language": language,
                        "stratum": "removed_line:foreign",
                        "row": row,
                        "doc_id": doc_id,
                        "chars": len(line),
                        "excerpt": excerpt(line, 200),
                    }
                )
            body = cleaned.text
            if not body:
                removed.add("empty_after_cleaning", chars)
                sample("empty_after_cleaning", row, doc_id, kind, text)
                continue
            # rule 5: script gate on the whole cleaned document
            profile = script_profile(body)
            letters = letter_total(profile)
            if letters == 0:
                removed.add("no_letters", chars)
                sample("no_letters", row, doc_id, kind, body)
                continue
            share = profile.get(script, 0) / letters
            if share < cfg.min_declared_share:
                removed.add("script_share", chars)
                sample("script_share", row, doc_id, kind, body, share=round(share, 3))
                continue
            # rule 6: Urdu check
            if family is not None:
                label = marker_label(language, marker_counts(body, family), letters)
                if label == "no_urdu_letters" and letters >= cfg.urdu_min_letters:
                    removed.add("urdu_persian_like", chars)
                    sample("urdu_persian_like", row, doc_id, kind, body)
                    continue
                if label == "uyghur_like":
                    removed.add("urdu_uyghur_like", chars)
                    sample("urdu_uyghur_like", row, doc_id, kind, body)
                    continue
                if label in ("pashto_like", "sindhi_like"):
                    measured[f"kept_{label}"] += 1
                    measured_chars[f"kept_{label}"] += len(body)
            # rule 8: wiki markup
            braces = len(_BRACES.findall(body))
            if braces >= cfg.wiki_marker_limit:
                removed.add("wiki_markup", chars)
                sample("wiki_markup", row, doc_id, kind, body, markers=braces)
                continue
            # rule 9: repetition
            tokens_ws = body.split()
            if len(tokens_ws) >= cfg.repetition_min_tokens:
                ratio = _repetition_ratio(body)
                if ratio is not None and ratio < cfg.min_repetition_ratio:
                    removed.add("repetition", chars)
                    sample("repetition", row, doc_id, kind, body, ratio=round(ratio, 3))
                    continue
            # rule 10: contacts
            final, n_email, n_phone = mask_contacts(body)
            masks["emails"] += n_email
            masks["phones"] += n_phone
            masks["docs_masked"] += 1 if (n_email or n_phone) else 0
            # rule 1 again, on the exact output text
            hit = guard.check(final)
            if hit is None:
                short = short_index.find(final, limit=1)
                if short:
                    suite_ids[short[0].suite_doc_id] += 1
            else:
                suite_ids[hit.suite_doc_id] += 1
            if hit is not None or short:
                removed.add("suite_after_cleaning", chars)
                continue
            tokens = token_counter.count(final)
            sig = mh.signature(final)
            if sig is not None:
                signatures[row] = sig
                valid[row] = True
            record = {
                "id": f"{source_id}#{row}",
                "text": final,
                "lang": language,
                "source": source_id,
                "row": row,
                "doc_id": doc_id,
                "type": kind,
                "revision": revision,
                "tokens": tokens,
            }
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            cand.write(f"{row}\t{line}\n".encode())
            candidate_rows.append(row)
            cand_chars[row] = (len(final), tokens, kind)
            # measured only (nothing removed)
            if cleaned.changed_lines:
                measured["kept_docs_with_lines_removed"] += 1
                res_kept_cleaned.offer(
                    lambda final=final, row=row, doc_id=doc_id, kind=kind, c=cleaned: {
                        "exp_id": exp_id,
                        "source_id": source_id,
                        "language": language,
                        "stratum": "kept:lines_removed",
                        "row": row,
                        "doc_id": doc_id,
                        "type": kind,
                        "chars": len(final),
                        "lines_removed": {"boilerplate": c.boilerplate_lines, "foreign": c.foreign_lines},
                        "excerpt": excerpt(final, SAMPLE_EXCERPT),
                    }
                )
            if len(final) < SHORT_KEPT_CHARS:
                measured[f"kept_under_{SHORT_KEPT_CHARS}_chars"] += 1
                measured_chars[f"kept_under_{SHORT_KEPT_CHARS}_chars"] += len(final)
            if language in ("hi", "mr") and final.count("\u0933") >= 3:
                measured["kept_lla_3plus"] += 1
                measured_chars["kept_lla_3plus"] += len(final)
            if fused is not None:
                n_fused = len(fused.findall(final))
                if n_fused:
                    measured["kept_docs_latin_fused_with_marks"] += 1
                    measured["latin_fused_with_marks_occurrences"] += n_fused
    say(
        f"[build] {source_id}: pass B done ({len(candidate_rows):,} candidates, "
        f"{(time.monotonic() - started) / 60:.1f} min)"
    )

    # ---- pass C: near-duplicates among candidates, then write ----
    roots, pairs = near_duplicate_clusters(
        signatures, valid, bands=cfg.near_dup_bands, threshold=cfg.near_dup_threshold
    )
    cand_set = np.zeros(max(n_rows, 1), dtype=bool)
    cand_set[candidate_rows] = True
    keep = cand_set & (roots == np.arange(len(roots)))
    # a cluster root is the smallest row in its component; all members are candidates (only
    # candidates have valid signatures), so the root is always a kept candidate.
    kept = _Counts()
    kept_tokens = 0
    kept_by_type = _Counts()
    tokens_by_type: Counter[str] = Counter()
    raw_sha = hashlib.sha256()
    tmp_out = out_path.with_name(out_path.name + ".tmp")
    near_examples: list[dict[str, Any]] = []
    with gzip.open(tmp_candidates, "rb") as cand, open(tmp_out, "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, compresslevel=6, mtime=0) as gz:
            for raw_line in cand:
                tab = raw_line.index(b"\t")
                row = int(raw_line[:tab])
                payload = raw_line[tab + 1 :]
                n_chars, tokens, kind = cand_chars[row]
                if not keep[row]:
                    removed.add("near_duplicate", n_chars)
                    if len(near_examples) < 200:
                        near_examples.append({"row": row, "kept_row": int(roots[row])})
                    if "near_duplicate" in res:
                        rec = json.loads(payload)
                        sample(
                            "near_duplicate", row, rec["doc_id"], kind, rec["text"], kept_row=int(roots[row])
                        )
                    continue
                gz.write(payload)
                raw_sha.update(payload)
                kept.add("all", n_chars)
                kept_by_type.add(kind, n_chars)
                tokens_by_type[kind] += tokens
                kept_tokens += tokens
                res_kept.offer(
                    lambda payload=payload, row=row: _kept_sample(exp_id, source_id, language, row, payload)
                )
    tmp_out.replace(out_path)
    tmp_candidates.unlink()
    out_sha = _sha256_path(out_path)

    removed_rows = sum(removed.docs.values())
    kept_docs = kept.docs["all"]
    if removed_rows + kept_docs != n_rows:
        raise AssertionError(f"{source_id}: {removed_rows} removed + {kept_docs} kept != {n_rows} rows")

    reasons = {
        r: {
            "rule": RULE_OF_REASON[r],
            "docs": removed.docs[r],
            "chars": removed.chars[r],
            "share_docs": _share(removed.docs[r], n_rows),
            "share_chars": _share(removed.chars[r], input_chars),
        }
        for r in REASONS
    }
    lines = {
        "boilerplate": {
            "lines": line_totals["boilerplate_lines"],
            "chars": line_totals["boilerplate_chars"],
            "share_chars": _share(line_totals["boilerplate_chars"], input_chars),
        },
        "foreign": {
            "lines": line_totals["foreign_lines"],
            "chars": line_totals["foreign_chars"],
            "share_chars": _share(line_totals["foreign_chars"], input_chars),
        },
    }
    review = []
    for key, bound in REVIEW_BOUNDS.items():
        value = (
            lines[key.split(":", 1)[1]]["share_chars"]
            if key.startswith("lines:")
            else reasons[key]["share_chars"]
        )
        if value > bound:
            review.append({"rule": key, "share_chars": value, "bound": bound})
    top_lines = sorted(
        ((boiler_counts.get(h, 0), t) for h, t in boiler_text.items()), key=lambda x: (-x[0], x[1])
    )[:TOP_BOILERPLATE]

    stats: dict[str, Any] = {
        "source_id": source_id,
        "language": language,
        "script": script,
        "config_fingerprint": cfg.fingerprint(),
        "input": {
            "docs": n_rows,
            "chars": input_chars,
            "by_type": {
                k: {"docs": input_by_type.docs[k], "chars": input_by_type.chars[k]}
                for k in sorted(input_by_type.docs)
            },
        },
        "line_counts": line_stats,
        "removed_docs": reasons,
        "removed_lines": lines,
        "kept": {
            "docs": kept_docs,
            "chars": kept.chars["all"],
            "tokens": kept_tokens,
            "share_docs": _share(kept_docs, n_rows),
            "share_chars_of_input": _share(kept.chars["all"], input_chars),
            "by_type": {
                k: {"docs": kept_by_type.docs[k], "chars": kept_by_type.chars[k], "tokens": tokens_by_type[k]}
                for k in sorted(kept_by_type.docs)
            },
        },
        "masked": dict(masks),
        "measured_only": {"docs": dict(measured), "chars": dict(measured_chars)},
        "near_duplicates": {
            "candidates_with_signature": int(valid.sum()),
            "removed": removed.docs["near_duplicate"],
            "clusters_with_removals": len({int(roots[e["row"]]) for e in near_examples})
            if near_examples
            else 0,
            "clusters_note": "counted over the first 200 removed documents",
        },
        "suite": {
            "touching_docs": sum(removed.docs[r] for r in SUITE_REASONS),
            "suite_documents_touched": len(suite_ids),
            "output_hits": 0,  # every written text passed guard.check and short_index.find
        },
        "top_boilerplate_lines": [{"count": c, "text": excerpt(t, 120)} for c, t in top_lines],
        "review": review,
        "output": {
            "path": out_path.name,
            "sha256_gz": out_sha,
            "sha256_jsonl": raw_sha.hexdigest(),
            "bytes_gz": out_path.stat().st_size,
        },
        "max_docs": max_docs,
        "seconds": round(time.monotonic() - started, 1),
    }
    del pairs
    samples: list[dict[str, Any]] = []
    for r in REASONS:
        if r in res:
            samples.extend(res[r].items)
    samples.extend(res_foreign.items)
    samples.extend(res_kept.items)
    samples.extend(res_kept_cleaned.items)
    return stats, samples


def _kept_sample(exp_id: str, source_id: str, language: str, row: int, payload: bytes) -> dict[str, Any]:
    rec = json.loads(payload)
    return {
        "exp_id": exp_id,
        "source_id": source_id,
        "language": language,
        "stratum": "kept",
        "row": row,
        "doc_id": rec["doc_id"],
        "type": rec["type"],
        "chars": len(rec["text"]),
        "tokens": rec["tokens"],
        "excerpt": excerpt(rec["text"], SAMPLE_EXCERPT),
    }


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iter_output(path: Path | str) -> Iterable[dict[str, Any]]:
    """Read one built ``<lang>.jsonl.gz`` back (one JSON document per line)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


# ------------------------------------------------------------------------ reports --
def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render_text_report(run: dict[str, Any]) -> str:
    """Human-readable SUMMARY.txt (ASCII-friendly layout, numbers from summary.json)."""
    out = [
        f"{run['exp_id']} - FrontierCorpus v2 build on {run['slice_id']} "
        f"({run['dataset']} @ {run['revision'][:12]})",
        f"complete: {run['complete']}   suite: {run['suite_status']}",
        f"tokenizer: {run['tokenizer']['name']}   config: {run['config_fingerprint']}",
        "",
    ]
    files = run["files"]
    if not files:
        out.append("(no file built yet)")
        return "\n".join(out) + "\n"
    out.append("lang | input docs | kept docs | kept chars (share) | tokens (exact) | minutes")
    tot = Counter()
    for f in files:
        k, i = f["kept"], f["input"]
        tot["in_docs"] += i["docs"]
        tot["in_chars"] += i["chars"]
        tot["docs"] += k["docs"]
        tot["chars"] += k["chars"]
        tot["tokens"] += k["tokens"]
        out.append(
            f"{f['language']:>4} | {i['docs']:>10,} | {k['docs']:>9,} | {k['chars']:>14,} "
            f"({_pct(k['share_chars_of_input'])}) | {k['tokens']:>14,} | {f['seconds'] / 60:.1f}"
        )
    out.append(
        f" all | {tot['in_docs']:>10,} | {tot['docs']:>9,} | {tot['chars']:>14,} "
        f"({_pct(tot['chars'] / max(tot['in_chars'], 1))}) | {tot['tokens']:>14,} |"
    )
    out += ["", "Removed documents per rule (docs / share of the file's characters):"]
    for r in REASONS:
        cells = [
            f"{f['language']} {f['removed_docs'][r]['docs']:,}/{_pct(f['removed_docs'][r]['share_chars'])}"
            for f in files
            if f["removed_docs"][r]["docs"]
        ]
        out.append(f"  {r} [{RULE_OF_REASON[r]}]: " + (", ".join(cells) if cells else "0"))
    out += ["", "Removed lines (share of the file's characters):"]
    for kind in ("boilerplate", "foreign"):
        cells = [f"{f['language']} {_pct(f['removed_lines'][kind]['share_chars'])}" for f in files]
        out.append(f"  {kind}: " + ", ".join(cells))
    out += [
        "",
        "Masked (rule 10): "
        + ", ".join(
            f"{f['language']} {f['masked'].get('emails', 0):,} e-mails/"
            f"{f['masked'].get('phones', 0):,} phones"
            for f in files
        ),
    ]
    out += ["", "Measured only (nothing removed):"]
    for f in files:
        if f["measured_only"]["docs"]:
            cells = ", ".join(f"{k} {v:,}" for k, v in sorted(f["measured_only"]["docs"].items()))
            out.append(f"  {f['language']}: {cells}")
    reviews = [(f["language"], r) for f in files for r in f["review"]]
    out += ["", "REVIEW (a rule removed more than its pre-registered bound; read before accepting):"]
    out += [
        f"  {lang}: {r['rule']} {_pct(r['share_chars'])} > {_pct(r['bound'])}" for lang, r in reviews
    ] or ["  none"]
    out += ["", f"Output suite hits: {sum(f['suite']['output_hits'] for f in files)} (must be 0)"]
    return "\n".join(out) + "\n"
