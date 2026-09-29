"""Calibrate the FrontierCorpus v2 build rules on the downloaded Sangraha slice (EXP-035).

Why this exists
---------------
EXP-034 measured the slice and found twelve issues that the build must handle. Examples: 20
documents overlap the protected suite; 1,882 short suite documents are only exact-match
protected; a Uyghur document passed the script gate for Urdu; 21.5% of characters are PDF/OCR
text; the 20,000-character rule would drop long documents. A filter threshold chosen without
looking at the documents it removes is a guess. This module does not guess. In one streaming
pass per file it **removes nothing** and produces:

1. **Random samples for a human read, per question** (who would a rule remove, and are they
   really bad?). Strata: ordinary passing documents; the 0.4-0.6 and 0.6-0.8 script-share bands;
   long documents whose later text fails the script check; PDF (OCR) documents; documents over
   20,000 characters; hits of each default quality rule; near-duplicate pairs; and, where two
   languages share a script, the documents that look like the other language. Selection is
   reservoir sampling with a fixed seed per (file, stratum), so it is reproducible.
2. **Counts that the samples cannot give:**
   - script share by band, split by whether the declared script is still the top script;
   - start/middle/end windows for long documents;
   - language markers in shared scripts: Assamese vs Bengali letters, Urdu vs
     Uyghur/Sindhi/Pashto/Arabic-Persian letters, Marathi's ``ळ`` in the Hindi file;
   - NFC vs whitespace-only normalization changes, and the code points NFC changes;
   - repeated lines (boilerplate);
   - within-file near-duplicates (MinHash, 128 permutations, word 5-grams, 16 bands x 8 rows,
     estimated Jaccard >= 0.8 against a cluster representative);
   - quality-rule hits by document type;
   - short suite documents contained word for word
     (:class:`~frontier_ai.corpus.decontaminate.ShortSuiteIndex`, bucketed by suite document
     length, so the minimum length can be chosen from data).

Samples are committed as evidence, so they are masked: e-mail addresses, URLs and digit runs of
7 or more digits (phone numbers, ids) are replaced. Shorter numbers (years, scores) stay, so the
``digit_runs`` question can be judged. Suite texts are never written out, only suite ids: a
document that touches the protected suite (13-gram guard or short-passage containment) is
counted but never sampled, and its repeated lines are counted without keeping their text.

Bounded cost on the 2-core laptop: signatures are 128 x uint32 per document (about 180 MB for
the largest file). Line statistics keep one 8-byte hash and one length per line. Top repeated
lines are found in a deterministic 1-in-11 sample of lines, then counted exactly. Near-duplicate
examples keep a 100-character excerpt per document.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
import unicodedata
import zlib
from array import array
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from frontier_ai.corpus.decontaminate import ShortSuiteIndex, SuiteGuard
from frontier_ai.corpus.langid import DEFAULT_MIN_DECLARED_SHARE, letter_total, script_profile, top_script
from frontier_ai.corpus.normalize import normalize_text
from frontier_ai.corpus.pipeline import PipelineDocument
from frontier_ai.corpus.quality import RULE_ORDER, QualityPolicy, _rule_results
from frontier_ai.corpus.slice_inspect import PROFILE_CAP, _bucket, _quantiles

SCHEMA = "frontier-slice-calibration-v1"
EXCERPT_CHARS = 400
LONG_DOC_CHARS = 20_000

# ------------------------------------------------------------------------ masking --
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_LONG_NUMBER = re.compile(r"\d(?:[\d \-]*\d){6,}")  # >= 7 digits, spaces/hyphens allowed between


def mask_pii(text: str) -> str:
    """Mask e-mails, URLs and digit runs of 7+ digits; keep short numbers (years, scores)."""
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[url]", text)
    return _LONG_NUMBER.sub(lambda m: f"[{sum(c.isdigit() for c in m.group())} digits]", text)


def excerpt(text: str, limit: int = EXCERPT_CHARS, start: int = 0) -> str:
    """A masked, single-line excerpt (line breaks shown as `` / ``)."""
    window = text[start : start + 4 * limit + 64]  # never flatten/mask a whole long document
    flat = " / ".join(part.strip() for part in window.splitlines() if part.strip())
    flat = mask_pii(flat)
    return flat[:limit] + ("..." if len(flat) > limit else "")


# ---------------------------------------------------------------------- sampling --
class Reservoir:
    """Uniform random sample of fixed size from a stream (Algorithm R), seeded per stratum."""

    def __init__(self, k: int, seed: str) -> None:
        self.k = k
        self.rng = random.Random(seed)
        self.seen = 0
        self.items: list[dict[str, Any]] = []

    def offer(self, make: Callable[[], dict[str, Any]]) -> None:
        """Count one candidate; build its record (``make``) only if it enters the sample."""
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(make())
            return
        j = self.rng.randrange(self.seen)
        if j < self.k:
            self.items[j] = make()


def _line_hash(line: str) -> int:
    return int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "little")


class LineStats:
    """Exact repeated-line statistics for one file from 8-byte line hashes.

    The texts of the most repeated lines come from a deterministic 1-in-``SAMPLE_EVERY`` sample of
    lines (a line repeated hundreds of times is sampled with near certainty); their counts are
    then taken exactly from the hash table.
    """

    SAMPLE_EVERY = 11
    MAX_SAMPLED = 200_000

    def __init__(self) -> None:
        self.hashes = array("Q")
        self.lengths = array("I")
        self.sampled: Counter[str] = Counter()

    def add(self, text: str, sample: bool = True) -> None:
        """``sample=False`` counts the lines but never keeps their text (suite-touching docs)."""
        for raw in text.split("\n"):
            line = raw.strip()
            if not line:
                continue
            self.hashes.append(_line_hash(line))
            self.lengths.append(len(line))
            if sample and len(self.hashes) % self.SAMPLE_EVERY == 0 and len(line) <= 300:
                self.sampled[line] += 1
                if len(self.sampled) > self.MAX_SAMPLED:
                    self.sampled = Counter({k: v for k, v in self.sampled.items() if v > 1})

    def finish(self, top: int = 25) -> dict[str, Any]:
        n = len(self.hashes)
        if n == 0:
            return {"lines": 0}
        h = np.frombuffer(self.hashes, dtype=np.uint64)
        lengths = np.frombuffer(self.lengths, dtype=np.uint32).astype(np.int64)
        uniq, inverse, counts = np.unique(h, return_inverse=True, return_counts=True)
        per_line = counts[inverse]
        out: dict[str, Any] = {
            "lines": int(n),
            "distinct_lines": int(len(uniq)),
            "chars_in_lines": int(lengths.sum()),
        }
        for t in (2, 10, 100):
            mask = per_line >= t
            out[f"lines_repeated_ge_{t}"] = int(mask.sum())
            out[f"chars_in_lines_repeated_ge_{t}"] = int(lengths[mask].sum())
        candidates = [(_line_hash(t), t) for t in self.sampled]
        keys = np.array([k for k, _ in candidates], dtype=np.uint64)
        pos = np.searchsorted(uniq, keys)
        exact = [
            int(counts[p]) if p < len(uniq) and uniq[p] == k else 0
            for p, k in zip(pos.tolist(), keys.tolist(), strict=True)
        ]
        ranked = sorted(
            ((c, t) for c, (_, t) in zip(exact, candidates, strict=True)), key=lambda x: (-x[0], x[1])
        )
        out["top_repeated_lines"] = [{"count": c, "text": excerpt(t, 120)} for c, t in ranked[:top] if c >= 2]
        return out


# -------------------------------------------------------------------- near-dups --
_MERSENNE_MIX = np.uint64(0x9E3779B97F4A7C15)


class MinHasher:
    """MinHash over word 5-gram shingles with multiply-shift hashing (deterministic, numpy)."""

    def __init__(self, num_perm: int = 128, shingle: int = 5, seed: int = 35) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.shingle = shingle
        with np.errstate(over="ignore"):
            self.a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        self.b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
        self.mults = np.array(
            [pow(1_000_003, shingle - 1 - i, 2**64) for i in range(shingle)], dtype=np.uint64
        )

    def signature(self, text: str) -> np.ndarray | None:
        words = text.split()
        if len(words) < self.shingle:
            return None
        wh = np.fromiter((zlib.crc32(w.encode("utf-8")) for w in words), dtype=np.uint64, count=len(words))
        k = self.shingle
        m = len(words) - k + 1
        sh = np.zeros(m, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for i in range(k):
                sh += wh[i : i + m] * self.mults[i]
            sh = np.unique(sh)
            best = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
            for start in range(0, len(sh), 2048):
                block = sh[start : start + 2048, None] * self.a + self.b
                np.minimum(best, block.min(axis=0), out=best)
        return (best >> np.uint64(32)).astype(np.uint32)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def near_duplicate_clusters(
    signatures: np.ndarray, valid: np.ndarray, bands: int = 16, threshold: float = 0.8
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """LSH banding + verification against each bucket's first member -> (root per doc, pairs).

    Documents join a cluster when their estimated Jaccard (share of equal signature values) with
    the bucket representative is >= ``threshold``. Invalid rows (too few words) stay singletons.
    """
    n, num_perm = signatures.shape
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    uf = _UnionFind(n)
    idx = np.flatnonzero(valid)
    pairs: list[tuple[int, int, float]] = []
    if len(idx) < 2:
        return uf.parent.copy(), pairs
    for band in range(bands):
        key = np.zeros(n, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for c in range(band * rows, (band + 1) * rows):
                key = key * _MERSENNE_MIX + signatures[:, c].astype(np.uint64)  # no full-size copy
        keys = key[idx]
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        breaks = np.flatnonzero(np.diff(sorted_keys)) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(order)]))
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            if e - s < 2:
                continue
            members = idx[order[s:e]]
            rep = int(members[0])
            others = members[1:]
            agree = (signatures[others] == signatures[rep]).mean(axis=1)
            for o, a in zip(others.tolist(), agree.tolist(), strict=True):
                if a >= threshold:
                    if uf.find(rep) != uf.find(o) and len(pairs) < 200:
                        pairs.append((rep, o, a))
                    uf.union(rep, o)
    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    return roots, pairs


# ---------------------------------------------------------------- language markers --
MARKERS: dict[str, dict[str, str]] = {
    # Assamese writes ra/wa as U+09F0/U+09F1; Bengali writes ra as U+09B0.
    "bengali_script": {"assamese": "\u09f0\u09f1", "bengali": "\u09b0"},
    # Letters used by Urdu but not by Uyghur/Persian/Arabic; and letters of the other languages.
    "arabic_script": {
        # tteh, ddal, rreh, noon ghunna, yeh barree, heh doachashmee
        "urdu": "\u0679\u0688\u0691\u06ba\u06d2\u06be",
        "uyghur": "\u06d5\u06c7\u06c6\u06c8\u06cb\u06ad",  # ae, u, oe, yu, ve, ng
        "sindhi": "\u0684\u0683\u0687\u068f\u0699\u06aa\u06bb\u067b\u067a\u067d",
        "pashto": "\u067c\u0689\u0693\u0696\u069a\u0681\u0685\u06cd",
        "arabic": "\u0629\u0649",  # teh marbuta, alef maksura
    },
    "devanagari": {"marathi_lla": "\u0933"},
}
FILE_MARKERS = {
    "as": "bengali_script",
    "bn": "bengali_script",
    "ur": "arabic_script",
    "hi": "devanagari",
    "mr": "devanagari",
}


def marker_counts(text: str, family: str) -> dict[str, int]:
    return {name: sum(text.count(ch) for ch in chars) for name, chars in MARKERS[family].items()}


def marker_label(language: str, counts: dict[str, int], letters: int) -> str:
    """A coarse, rule-based label used only to *pick samples and count*; nothing is removed."""
    if language in ("as", "bn"):
        a, b = counts["assamese"], counts["bengali"]
        if a + b < 5:
            return "too_few_markers"
        share = a / (a + b)
        return "assamese_like" if share >= 0.5 else "bengali_like"
    if language == "ur":
        urdu = counts["urdu"]
        for other in ("uyghur", "sindhi", "pashto"):
            if counts[other] >= 3 and counts[other] > urdu:
                return f"{other}_like"
        if urdu == 0 and letters >= 50:
            return "no_urdu_letters"
        return "urdu_like"
    if language in ("hi", "mr"):
        return "has_lla_3plus" if counts["marathi_lla"] >= 3 else "lla_0_2"
    return "n/a"


EXPECTED_LABEL = {"as": "assamese_like", "bn": "bengali_like", "ur": "urdu_like"}


# ------------------------------------------------------------------------ helpers --
def _share(profile: dict[str, int], script: str) -> tuple[float, int]:
    letters = letter_total(profile)
    return (profile.get(script, 0) / letters if letters else 0.0), letters


def _windows(text: str) -> list[str]:
    if len(text) <= PROFILE_CAP:
        return [text]
    mid = len(text) // 2 - PROFILE_CAP // 2
    return [text[:PROFILE_CAP], text[mid : mid + PROFILE_CAP], text[-PROFILE_CAP:]]


def _nfc_removed_codepoints(raw: str, nfc: str) -> Counter[str]:
    return Counter(raw) - Counter(nfc)


def _cp_name(ch: str) -> str:
    return f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}"


def _length_bucket(words: int) -> str:
    if words <= 4:
        return "3-4"
    if words <= 6:
        return "5-6"
    if words <= 9:
        return "7-9"
    return "10-12"


WITHHELD = "[withheld: this document contains protected-suite text]"
SAMPLE_SIZES = {
    "random_pass": 6,
    "band_0.4_0.6": 6,
    "band_0.6_0.8": 4,
    "window_fail_after_start": 3,
    "pdf_pass": 8,
    "long_over_20k": 4,
    "digit_runs": 5,
    "repetition": 4,
    "url_density": 3,
    "template_residue": 3,
    "other_language_marker": 6,
    "hi_with_lla": 4,
}


def calibrate_rows(
    rows: Iterable[tuple[str, str, str]],
    *,
    exp_id: str,
    source_id: str,
    language: str,
    script: str,
    expected_rows: int,
    short_index: ShortSuiteIndex | None,
    guard: SuiteGuard | None = None,
    minhasher: MinHasher | None = None,
    max_docs: int | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 50_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One streaming pass -> (statistics dict, sample records). Removes nothing."""
    policy = QualityPolicy()
    mh = minhasher or MinHasher()
    started = time.monotonic()
    cap = expected_rows if max_docs is None else min(expected_rows, max_docs)
    signatures = np.zeros((max(cap, 1), mh.num_perm), dtype=np.uint32)
    valid = np.zeros(max(cap, 1), dtype=bool)
    types: list[str] = []
    heads: list[str] = []
    doc_ids: list[str] = []
    res = {name: Reservoir(k, f"{exp_id}/{source_id}/{name}") for name, k in SAMPLE_SIZES.items()}
    lines = LineStats()

    n = empty = nfc_changed = ws_only_changed = 0
    nfc_cps: Counter[str] = Counter()
    band_top: Counter[str] = Counter()
    windows = Counter()
    rule_by_type: dict[str, Counter[str]] = {r: Counter() for r in RULE_ORDER}
    long_by_type: Counter[str] = Counter()
    long_chars_by_type: Counter[str] = Counter()
    long_lengths: list[int] = []
    labels: Counter[str] = Counter()
    labels_chars: Counter[str] = Counter()
    short_hits_bucket: Counter[str] = Counter()
    short_hit_docs = guard_hit_docs = withheld_docs = 0
    short_examples: list[dict[str, Any]] = []
    family = FILE_MARKERS.get(language)

    for doc_id, kind, raw in rows:
        if max_docs is not None and n >= max_docs:
            break
        if n >= cap:  # more rows than the parquet metadata declared: refuse to guess
            raise ValueError(f"{source_id}: more rows than the declared {expected_rows}")
        i = n
        n += 1
        types.append(kind)
        doc_ids.append(doc_id)
        if not unicodedata.is_normalized("NFC", raw):
            nfc_changed += 1
            nfc_cps.update(_nfc_removed_codepoints(raw, unicodedata.normalize("NFC", raw)))
        text = normalize_text(raw)
        if text != raw.strip() and unicodedata.is_normalized("NFC", raw):
            ws_only_changed += 1
        if not text:
            heads.append("")
            empty += 1
            continue

        # ---- protected suite: count short-passage containment; never sample a touching doc ----
        withheld = False
        if short_index is not None:
            hits = short_index.find(text, limit=5)
            if hits:
                withheld = True
                short_hit_docs += 1
                for h in hits:
                    short_hits_bucket[_length_bucket(h.words)] += 1
                if len(short_examples) < 20:
                    short_examples.append(
                        {
                            "doc_id": doc_id,
                            "type": kind,
                            "chars": len(text),
                            "suite_hits": [{"suite_doc_id": h.suite_doc_id, "words": h.words} for h in hits],
                        }
                    )
        if guard is not None and guard.check(text) is not None:
            withheld = True
            guard_hit_docs += 1
        if withheld:
            withheld_docs += 1
        heads.append(WITHHELD if withheld else excerpt(text, 100))

        # ---- script share (same rule as the pipeline) + long-document windows ----
        profile = script_profile(text[:PROFILE_CAP])
        share, letters = _share(profile, script)
        passes = letters > 0 and share >= DEFAULT_MIN_DECLARED_SHARE
        top_is_declared = letters > 0 and top_script(profile) == script
        band_top[f"{_bucket(share)} top={'declared' if top_is_declared else 'other'}"] += 1

        def record(
            stratum: str,
            extra: dict[str, Any] | None = None,
            start: int = 0,
            text: str = text,
            doc_id: str = doc_id,
            kind: str = kind,
            share: float = share,
            withheld: bool = withheld,
        ) -> None:
            if withheld:  # contains protected-suite text: counted above, never written out
                return

            def make() -> dict[str, Any]:  # built only if the reservoir keeps it
                item = {
                    "language": language,
                    "stratum": stratum,
                    "doc_id": doc_id,
                    "type": kind,
                    "chars": len(text),
                    "share": round(share, 3),
                    "text": excerpt(text, start=start),
                }
                if extra:
                    item.update({k: (v() if callable(v) else v) for k, v in extra.items()})
                return item

            res[stratum].offer(make)

        if passes:
            record("random_pass")
        if 0.4 <= share < 0.6:
            record("band_0.4_0.6", {"top_is_declared": top_is_declared})
        elif 0.6 <= share < 0.8:
            record("band_0.6_0.8")
        if len(text) > PROFILE_CAP:
            verdicts = []
            for w in _windows(text):
                s, lt = _share(script_profile(w), script)
                verdicts.append(lt > 0 and s >= DEFAULT_MIN_DECLARED_SHARE)
            key = "".join("P" if v else "F" for v in verdicts)
            windows[key] += 1
            if verdicts[0] and not all(verdicts):
                later = len(text) // 2 if not verdicts[1] else max(0, len(text) - PROFILE_CAP)
                record("window_fail_after_start", {"windows": key}, start=later)
        if passes and kind == "pdf":
            record("pdf_pass")

        # ---- length ----
        if len(text) > LONG_DOC_CHARS:
            long_by_type[kind] += 1
            long_chars_by_type[kind] += len(text)
            long_lengths.append(len(text))
            record("long_over_20k", {"middle": lambda t=text: excerpt(t, 300, start=len(t) // 2)})

        # ---- quality rules by type ----
        doc = PipelineDocument(doc_id=doc_id, source_id=source_id, language=language, text=text)
        for rule, (violates, value) in _rule_results(doc, policy).items():
            if violates:
                rule_by_type[rule][kind] += 1
                if rule in res:
                    record(rule, {"measured": value if isinstance(value, (int, float)) else str(value)})

        # ---- language markers in shared scripts ----
        if family is not None and passes:
            counts = marker_counts(text, family)
            label = marker_label(language, counts, letters)
            labels[label] += 1
            labels_chars[label] += len(text)
            expected = EXPECTED_LABEL.get(language)
            if expected and label not in (expected, "too_few_markers"):
                record("other_language_marker", {"marker_label": label, "markers": counts})
            if language == "hi" and label == "has_lla_3plus":
                record("hi_with_lla", {"markers": counts})

        # ---- boilerplate lines, near-dup signature, short suite containment ----
        lines.add(text, sample=not withheld)
        sig = mh.signature(text)
        if sig is not None:
            signatures[i] = sig
            valid[i] = True

        if progress is not None and n % progress_every == 0:
            el = time.monotonic() - started
            progress(f"[calibrate] {source_id}: {n:,} docs, {n / max(el, 1e-9):,.0f} docs/s")

    # ---- near-duplicate clustering (within this file) ----
    signatures, valid = signatures[:n], valid[:n]
    roots, pairs = near_duplicate_clusters(signatures, valid)
    in_valid = roots[valid]
    uniq, counts = np.unique(in_valid, return_counts=True)
    dup_clusters = int((counts > 1).sum())
    removable = int((counts - 1).sum())
    root_size = dict(zip(uniq.tolist(), counts.tolist(), strict=True))
    removable_by_type: Counter[str] = Counter()
    seen_root: set[int] = set()
    for j in np.flatnonzero(valid).tolist():
        r = int(roots[j])
        if root_size.get(r, 1) > 1:
            if r in seen_root:
                removable_by_type[types[j]] += 1
            else:
                seen_root.add(r)
    rng = random.Random(f"{exp_id}/{source_id}/near_dup")
    shown = [p for p in pairs if WITHHELD not in (heads[p[0]], heads[p[1]])]
    chosen = rng.sample(shown, min(4, len(shown))) if shown else []
    near_dup_examples = [
        {
            "language": language,
            "stratum": "near_duplicate_pair",
            "agreement": round(a, 3),
            "a": {"doc_id": doc_ids[x], "type": types[x], "text": heads[x]},
            "b": {"doc_id": doc_ids[y], "type": types[y], "text": heads[y]},
        }
        for x, y, a in chosen
    ]

    elapsed = time.monotonic() - started
    stats: dict[str, Any] = {
        "source_id": source_id,
        "language": language,
        "declared_script": script,
        "documents": n,
        "documents_empty_after_normalize": empty,
        "normalization": {
            "not_nfc_documents": nfc_changed,
            "whitespace_only_changed_documents": ws_only_changed,
            "nfc_removed_codepoints_top": [
                {"codepoint": _cp_name(c), "count": k} for c, k in nfc_cps.most_common(8)
            ],
        },
        "script_share_band_by_top_script": dict(sorted(band_top.items())),
        "long_document_windows": {
            "documents_longer_than_cap": sum(windows.values()),
            "start_middle_end_pattern": dict(sorted(windows.items())),
        },
        "long_documents": {
            "threshold_chars": LONG_DOC_CHARS,
            "documents_by_type": dict(sorted(long_by_type.items())),
            "chars_by_type": dict(sorted(long_chars_by_type.items())),
            "length_quantiles": _quantiles(long_lengths),
        },
        "quality_rule_hits_by_type": {r: dict(sorted(c.items())) for r, c in rule_by_type.items()},
        "language_markers": (
            {
                "family": family,
                "labels_documents": dict(sorted(labels.items())),
                "labels_chars": dict(sorted(labels_chars.items())),
            }
            if family
            else None
        ),
        "repeated_lines": lines.finish(),
        "near_duplicates": {
            "method": "MinHash 128 perms, word 5-gram shingles, LSH 16 bands x 8 rows, estimated "
            "Jaccard >= 0.8 vs bucket representative; within this file only",
            "documents_with_signature": int(valid.sum()),
            "documents_too_short_for_shingles": int(n - valid.sum()),
            "duplicate_clusters": dup_clusters,
            "documents_removable_keep_one_per_cluster": removable,
            "removable_by_type": dict(sorted(removable_by_type.items())),
        },
        "short_suite_containment": (
            {
                "checked": True,
                "index": short_index.describe(),
                "documents_hit": short_hit_docs,
                "hits_by_suite_doc_words": dict(sorted(short_hits_bucket.items())),
                "examples": short_examples,
            }
            if short_index is not None
            else {"checked": False}
        ),
        "suite_ngram_guard": (
            {"checked": True, "documents_hit": guard_hit_docs} if guard is not None else {"checked": False}
        ),
        "documents_withheld_from_samples": withheld_docs,
        "samples_per_stratum": {
            name: {"candidates": r.seen, "kept": len(r.items)} for name, r in res.items()
        },
        "seconds": round(elapsed, 2),
        "documents_per_second": round(n / elapsed, 1) if elapsed > 0 else None,
    }
    samples = [item for r in res.values() for item in r.items] + near_dup_examples
    return stats, samples


# ------------------------------------------------------------------------ report --
def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "-"


def render_text_report(run: dict[str, Any]) -> str:
    """Plain-text summary for NIGHT_REPORT (tables first)."""
    out = [
        f"{run['exp_id']} calibration of {run['slice_id']} (revision {run['revision'][:12]}) "
        "- removes nothing",
        f"Short-suite check: {run['short_suite_status']}",
        "",
        "lang  docs       near-dup-removable  chars-in-lines-seen>=10x  over-20k  not-NFC  "
        "short-suite-hits  13-gram-hits  other-language-marker",
    ]
    for f in run["files"]:
        docs = f["documents"]
        removable = f["near_duplicates"]["documents_removable_keep_one_per_cluster"]
        rl = f["repeated_lines"]
        boiler = _pct(rl.get("chars_in_lines_repeated_ge_10", 0), rl.get("chars_in_lines", 0))
        over = sum(f["long_documents"]["documents_by_type"].values())
        not_nfc = _pct(f["normalization"]["not_nfc_documents"], docs)
        lm = f["language_markers"]
        other = "-"
        if lm and f["language"] in EXPECTED_LABEL:
            skip = (EXPECTED_LABEL[f["language"]], "too_few_markers")
            other = f"{sum(v for k, v in lm['labels_documents'].items() if k not in skip):,}"
        elif lm and f["language"] == "hi":
            other = f"lla>=3: {lm['labels_documents'].get('has_lla_3plus', 0):,}"
        ssc = f["short_suite_containment"]
        hits = str(ssc["documents_hit"]) if ssc["checked"] else "n/a"
        sng = f.get("suite_ngram_guard", {"checked": False})
        ngram = str(sng["documents_hit"]) if sng["checked"] else "n/a"
        near = f"{removable:,} ({_pct(removable, docs)})"
        out.append(
            f"{f['language']:<5} {docs:<10,} {near:<19} {boiler:<25} {over:<9,} {not_nfc:<8} "
            f"{hits:<17} {ngram:<13} {other}"
        )
    out.append("")
    out.append("Top code points changed by NFC (per file):")
    for f in run["files"]:
        top = ", ".join(
            f"{c['codepoint']} x{c['count']:,}" for c in f["normalization"]["nfc_removed_codepoints_top"][:3]
        )
        out.append(f"  {f['language']}: {top or '-'}")
    out.append("")
    out.append("Most repeated lines (first 3 per file; masked):")
    for f in run["files"]:
        for line in f["repeated_lines"].get("top_repeated_lines", [])[:3]:
            out.append(f"  [{f['language']} x{line['count']:,}] {line['text']}")
    out.append("")
    out.append(
        f"Samples for reading: {run['samples_written']:,} records in samples.jsonl "
        "(source: Sangraha, CC-BY-4.0; masked)."
    )
    out.append(f"Samples: {run.get('samples_status', '-')}")
    return "\n".join(out) + "\n"
