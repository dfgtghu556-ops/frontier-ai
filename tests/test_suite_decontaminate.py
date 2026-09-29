"""Protected-suite decontamination at scale (corpus/decontaminate.py)."""

from __future__ import annotations

import random

import pytest

from frontier_ai.corpus.decontaminate import (
    REASON_EXACT,
    REASON_NGRAM,
    SuiteGuard,
    decontaminate,
    iter_decontaminate,
)
from frontier_ai.corpus.pipeline import PipelineDocument, Removal
from frontier_ai.evaluation.contamination import contamination_report


def _words(prefix: str, k: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(k))


SUITE = [
    PipelineDocument("s-000", "suite", "hi", _words("सूत्र", 20)),  # 20 words -> checkable by 13-grams
    PipelineDocument("s-001", "suite", "bn", _words("বাক্য", 15)),
    PipelineDocument("s-002", "suite", "en", "a short refrain"),  # < 13 words: exact-only guard
]


def _guard() -> SuiteGuard:
    return SuiteGuard.from_texts((d.doc_id, d.text) for d in SUITE)


def test_guard_describes_suite_and_counts_too_short_documents():
    g = _guard()
    d = g.describe()
    assert d["ngram_n"] == 13
    assert d["suite_documents"] == 3
    assert d["suite_too_short_documents"] == 1
    assert d["suite_ngrams"] == (20 - 13 + 1) + (15 - 13 + 1)


def test_exact_copy_is_removed_even_when_too_short_for_ngrams():
    hit = _guard().check("a short refrain")
    assert hit is not None and hit.reason == REASON_EXACT and hit.suite_doc_id == "s-002"


def test_suite_paragraph_embedded_in_a_whole_page_is_caught():
    # The dump case: a whole page that contains a suite paragraph plus other text. Its hash
    # differs from every suite document, so only the n-gram guard can catch it.
    page = "शीर्षक पंक्ति " + SUITE[0].text + " और आगे का पाठ जो सूट में नहीं है"
    hit = _guard().check(page)
    assert hit is not None
    assert hit.reason == REASON_NGRAM and hit.suite_doc_id == "s-000"
    assert hit.matched_ngrams == 20 - 13 + 1


def test_a_single_shared_13_gram_is_enough_and_12_is_not():
    thirteen = " ".join(SUITE[1].text.split()[:13])
    twelve = " ".join(SUITE[1].text.split()[:12])
    g = _guard()
    assert g.check(f"x y {thirteen} z").reason == REASON_NGRAM
    assert g.check(f"x y {twelve} z") is None


def test_whitespace_differences_do_not_hide_overlap():
    # Same definition as evaluation.contamination: whitespace-split words.
    spaced = "\n".join(SUITE[0].text.split())
    assert _guard().check(spaced).reason == REASON_NGRAM


def test_clean_text_passes():
    assert _guard().check(_words("साफ", 40)) is None


def test_empty_suite_is_refused():
    with pytest.raises(ValueError):
        SuiteGuard.from_texts([])


def test_stage_removes_with_reasons_and_matches_contamination_report_zero():
    rng = random.Random(7)
    train = [PipelineDocument(f"t-{i:03d}", "src-a" if i % 2 else "src-b", "hi", _words(f"w{i}_", 30))
             for i in range(40)]
    dirty = [
        PipelineDocument("t-900", "src-a", "hi", "पृष्ठ " + SUITE[0].text + " अंत"),
        PipelineDocument("t-901", "src-b", "bn", SUITE[1].text),
        PipelineDocument("t-902", "src-b", "en", "a short refrain"),
    ]
    docs = train + dirty
    rng.shuffle(docs)

    before = contamination_report(SUITE, docs)
    assert before["exact_duplicates"] == 2 and before["ngram_overlap_documents"] == 2

    out = decontaminate(docs, _guard())
    assert {r.doc_id for r in out.removed} == {"t-900", "t-901", "t-902"}
    assert {r.doc_id: r.reason for r in out.removed} == {
        "t-900": REASON_NGRAM, "t-901": REASON_EXACT, "t-902": REASON_EXACT}
    assert out.stats["documents_in"] == 43 and out.stats["documents_out"] == 40
    assert out.stats["removed_by_reason"] == {REASON_EXACT: 2, REASON_NGRAM: 1}
    assert out.stats["per_language"]["bn"] == {"in": 1, "kept": 0, "removed": 1}
    # kept preserves input order
    assert [d.doc_id for d in out.kept] == [d.doc_id for d in docs if d.doc_id not in {"t-900", "t-901", "t-902"}]

    after = contamination_report(SUITE, list(out.kept))
    assert after["exact_duplicates"] == 0 and after["ngram_overlap_documents"] == 0


def test_streaming_form_is_lazy_and_agrees_with_stage():
    docs = [PipelineDocument("t-1", "a", "bn", SUITE[1].text),
            PipelineDocument("t-2", "a", "hi", _words("x", 30))]
    removals: list[Removal] = []
    gen = iter_decontaminate(iter(docs), _guard(), removals)
    assert removals == []  # nothing consumed yet
    kept = list(gen)
    assert [d.doc_id for d in kept] == ["t-2"]
    assert [r.doc_id for r in removals] == ["t-1"]
    assert [d.doc_id for d in decontaminate(docs, _guard()).kept] == ["t-2"]
