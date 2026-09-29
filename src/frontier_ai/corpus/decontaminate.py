"""Protected-suite decontamination for training data at scale (data scale-up phase 1).

Why this exists
---------------
D-042 makes ``frontier-heldout-v1`` protected evaluation data: its documents must never be
trained on. FrontierCorpus v2 will ingest whole external dumps (for example full Wikisource
editions) that *contain* the suite's source pages. The existing guards cannot handle that:

* ``evaluation.suite.find_exact_overlap`` compares whole-document hashes. A dump page is a
  whole page while the suite stores paragraphs and verse lines, so the hashes never match.
* ``evaluation.contamination.contamination_report`` finds 13-gram overlap, but it holds the
  n-grams of *all training documents* in memory. That is fine for the 4.2M-character pilot
  and impossible for hundreds of millions of words.

This module inverts the check: it holds the **suite's** n-grams (small, fixed) in memory and
streams the training documents past them, so memory is bounded by the suite size, not the
corpus size.

Semantics
---------
* The n-gram definition is exactly ``evaluation.contamination``'s: whitespace-split words,
  ``NGRAM`` (13) consecutive words joined by one space. A training document is contaminated
  when it contains the exact text of a suite document (``suite_exact``) or shares at least
  one 13-gram with any suite document (``suite_ngram``). After this stage removes every
  contaminated document, ``contamination_report`` on the kept documents reports zero exact
  and zero 13-gram overlap (covered by a test).
* Unlike the v1 pilot split (where overlap was *reported, not removed*), new external
  sources are **removed** on overlap: the suite is protected. Every removal records the
  suite document it touched and how many n-grams matched, so it is explainable.
* Suite documents shorter than 13 words cannot be checked by n-grams. They are still
  guarded by exact text and counted in ``stats`` (reported, not guessed).
* Pure and deterministic: no RNG; the verdict for a document depends only on its text.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from frontier_ai.corpus.pipeline import PipelineDocument, Removal, StageOutcome, sha256_text
from frontier_ai.evaluation.contamination import NGRAM

REASON_EXACT = "suite_exact"
REASON_NGRAM = "suite_ngram"


@dataclass(frozen=True)
class SuiteHit:
    """Why one training text touches the protected suite."""

    reason: str  # REASON_EXACT or REASON_NGRAM
    suite_doc_id: str  # the first suite document found (deterministic: first matching n-gram)
    matched_ngrams: int  # distinct suite n-grams found in the training text (0 for exact-only)

    def to_detail(self) -> dict[str, Any]:
        return {"suite_doc_id": self.suite_doc_id, "matched_ngrams": self.matched_ngrams}


@dataclass
class SuiteGuard:
    """In-memory index of a protected suite: exact text hashes and all its 13-grams.

    Build it once with :meth:`from_texts` and reuse it for every training document.
    Memory grows with the suite, never with the training corpus.
    """

    n: int
    exact: dict[str, str]  # sha256(text) -> first suite doc_id with that text
    ngrams: dict[str, str]  # n-gram -> first suite doc_id containing it
    suite_documents: int
    too_short_documents: int  # suite documents with fewer than n words (exact-only guard)
    _checked: int = field(default=0, repr=False)

    @classmethod
    def from_texts(cls, documents: Iterable[tuple[str, str]], n: int = NGRAM) -> SuiteGuard:
        """Index ``(doc_id, text)`` pairs. The first doc_id wins for repeated keys."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        exact: dict[str, str] = {}
        ngrams: dict[str, str] = {}
        count = 0
        too_short = 0
        for doc_id, text in documents:
            count += 1
            exact.setdefault(sha256_text(text), doc_id)
            words = text.split()
            if len(words) < n:
                too_short += 1
                continue
            for i in range(len(words) - n + 1):
                ngrams.setdefault(" ".join(words[i : i + n]), doc_id)
        if count == 0:
            raise ValueError("the protected suite is empty; refusing to build a guard that guards nothing")
        return cls(n=n, exact=exact, ngrams=ngrams, suite_documents=count, too_short_documents=too_short)

    def check(self, text: str) -> SuiteHit | None:
        """Return why ``text`` touches the suite, or ``None`` if it is clean."""
        self._checked += 1
        exact_id = self.exact.get(sha256_text(text))
        words = text.split()
        first_id: str | None = None
        seen: set[str] = set()
        for i in range(len(words) - self.n + 1):
            gram = " ".join(words[i : i + self.n])
            suite_id = self.ngrams.get(gram)
            if suite_id is not None and gram not in seen:
                seen.add(gram)
                if first_id is None:
                    first_id = suite_id
        if exact_id is not None:
            return SuiteHit(REASON_EXACT, exact_id, len(seen))
        if first_id is not None:
            return SuiteHit(REASON_NGRAM, first_id, len(seen))
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "ngram_n": self.n,
            "suite_documents": self.suite_documents,
            "suite_unique_texts": len(self.exact),
            "suite_ngrams": len(self.ngrams),
            "suite_too_short_documents": self.too_short_documents,
        }


def iter_decontaminate(
    documents: Iterable[PipelineDocument], guard: SuiteGuard, removals: list[Removal]
) -> Iterator[PipelineDocument]:
    """Streaming form: yield clean documents; append a :class:`Removal` for each dirty one.

    Use this over sharded input so the corpus is never held in memory at once.
    """
    for doc in documents:
        hit = guard.check(doc.text)
        if hit is None:
            yield doc
        else:
            removals.append(Removal(doc.doc_id, doc.source_id, doc.language, hit.reason, hit.to_detail()))


def decontaminate(documents: Sequence[PipelineDocument], guard: SuiteGuard) -> StageOutcome:
    """Pipeline-stage form (``list -> StageOutcome``), same contract as the other stages.

    Input order is preserved in ``kept`` (the verdict is per document, so order cannot
    change which documents are kept).
    """
    removed: list[Removal] = []
    kept = list(iter_decontaminate(documents, guard, removed))
    per_language: dict[str, dict[str, int]] = {}
    for doc in documents:
        per_language.setdefault(doc.language, {"in": 0, "kept": 0, "removed": 0})["in"] += 1
    for doc in kept:
        per_language[doc.language]["kept"] += 1
    by_reason: dict[str, int] = {}
    suite_docs_touched: set[str] = set()
    for r in removed:
        per_language[r.language]["removed"] += 1
        by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
        suite_docs_touched.add(r.detail["suite_doc_id"])
    stats: dict[str, Any] = {
        "documents_in": len(documents),
        "documents_out": len(kept),
        "removed": len(removed),
        "removed_by_reason": dict(sorted(by_reason.items())),
        "suite_documents_first_hit": len(suite_docs_touched),
        "per_language": per_language,
        "guard": guard.describe(),
    }
    return StageOutcome(stage="suite_decontaminate", kept=tuple(kept), removed=tuple(removed),
                        flagged={}, stats=stats)


def guard_from_heldout_shards(shard_paths: Sequence[Path | str], suite: dict[str, Any],
                              n: int = NGRAM) -> SuiteGuard:
    """Build a :class:`SuiteGuard` from the on-disk held-out shard files, verified against the suite.

    ``SUITE.json`` stores fingerprints only (the texts stay out of git). The texts live in the
    git-ignored held-out shards (one document per line, LF). This refuses to build a guard unless
    the shard lines are *exactly* the suite's documents (same multiset of SHA-256 hashes), so a
    stale, partial or edited shard can never silently weaken the protection. Suite doc_ids are
    assigned by hash, so the check does not depend on shard order.
    """
    by_hash: dict[str, list[str]] = {}
    for record in suite["documents"]:
        by_hash.setdefault(record["sha256"], []).append(record["doc_id"])
    pairs: list[tuple[str, str]] = []
    for path in shard_paths:
        # CRLF normalisation: a Windows checkout/copy must not change the verdict.
        text = Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")
        for line in text.split("\n"):
            if line == "":
                continue
            ids = by_hash.get(sha256_text(line))
            if not ids:
                raise ValueError(f"{path}: a line does not match any suite document "
                                 f"(sha256 {sha256_text(line)[:12]}...); wrong or edited shard")
            pairs.append((ids.pop(0), line))
    missing = sum(len(v) for v in by_hash.values())
    if missing:
        raise ValueError(f"held-out shards are missing {missing} of {len(suite['documents'])} "
                         "suite documents")
    return SuiteGuard.from_texts(pairs, n=n)
