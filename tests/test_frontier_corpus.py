"""FrontierCorpus v1 pipeline stages: normalize, langid, quality, exact dedup.

These are unit tests of the stage *contracts* (F1 part 1). They use the
repository's own deterministic probe corpus (``frontier_ai.tokenization.
corpus.EXAMPLE_BANK`` — 14 hand-written languages, sha256-verified upstream)
as the language golden set, plus crafted documents for each quality rule and
deduplication semantics. No network, no third-party text.

The pilot run of the full pipeline on the frozen ``indic-tokenizer/v2``
corpus (real 59 sources) happens on a network-enabled machine and is
recorded as its own experiment (F1 part 2).
"""

from __future__ import annotations

import hashlib
import random
import unicodedata

import pytest

from frontier_ai.corpus import (
    PipelineDocument,
    QualityPolicy,
    exact_dedup,
    langid_documents,
    normalize_documents,
    normalize_text,
    quality_filter,
    script_profile,
    top_script,
)
from frontier_ai.tokenization.corpus import EXAMPLE_BANK

# Declared script (pipeline block name) per probe-bank language. "as" shares
# the Bengali block with "bn"; "hi-en" is romanized (Latin) by construction.
LANGUAGE_SCRIPT = {
    "en": "latin",
    "hi": "devanagari",
    "hi-en": "latin",
    "bn": "bengali",
    "mr": "devanagari",
    "gu": "gujarati",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "pa": "gurmukhi",
    "or": "odia",
    "as": "bengali",
    "ur": "arabic",
}


def _doc(doc_id: str, language: str, text: str, source_id: str = "test") -> PipelineDocument:
    return PipelineDocument(doc_id=doc_id, source_id=source_id, language=language, text=text)


# ---------------------------------------------------------------------------
# A. document model
# ---------------------------------------------------------------------------
def test_document_identity_and_counts_multibyte():
    doc = _doc("s-000001", "hi", "मैं")
    assert doc.chars == 3
    assert doc.bytes == len("मैं".encode())
    assert doc.sha256 == hashlib.sha256("मैं".encode()).hexdigest()


def test_document_identity_changes_with_text():
    a = _doc("s-000001", "hi", "abc")
    b = _doc("s-000002", "hi", "abd")
    assert a.sha256 != b.sha256


# ---------------------------------------------------------------------------
# B. normalization
# ---------------------------------------------------------------------------
def test_normalize_nfc_composes_decomposed_text():
    nfd = "cafe\u0301"  # c a f e + combining acute
    nfc = normalize_text(nfd, "nfc")
    assert unicodedata.is_normalized("NFC", nfc)
    assert nfc == "caf\u00e9"
    assert nfc != nfd


def test_normalize_collapses_whitespace_strips_drops_cr_preserves_newlines():
    # \r dropped, space/tab runs collapsed, ends stripped; \n is content (verse
    # lines) and preserved by design — normalization is not a reformatter.
    assert normalize_text("  a\t\tb   c \r\n d ", "nfc") == "a b c \n d"
    assert normalize_text("line1\nline2\n", "nfc") == "line1\nline2"


def test_normalize_is_idempotent_both_policies():
    for policy in ("nfc", "none"):
        text = "  a\tb  \u00e9  \u0301c "
        once = normalize_text(text, policy)
        assert normalize_text(once, policy) == once


def test_normalize_none_policy_does_not_compose():
    nfd = "cafe\u0301"
    out = normalize_text(nfd, "none")
    assert out == nfd.strip()  # unchanged except whitespace handling


def test_normalize_documents_stats_and_no_removals():
    docs = [_doc("a-000001", "en", "  x  y "), _doc("a-000002", "en", "plain")]
    out = normalize_documents(docs, "nfc")
    assert [d.doc_id for d in out.kept] == ["a-000001", "a-000002"]
    assert out.removed == ()
    assert out.stats["documents_changed"] == 1
    assert out.stats["chars_out"] < out.stats["chars_in"]
    assert out.stats["policy"] == "nfc"


def test_normalize_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unknown normalization policy"):
        normalize_text("x", "nfd")


# ---------------------------------------------------------------------------
# C. language identification (script-profile gate)
# ---------------------------------------------------------------------------
def test_script_profile_counts_letters_by_block_and_common():
    profile = script_profile("hello 123 !")
    assert profile["latin"] == 5
    assert profile["common"] == 4  # 3 digits + '!'
    assert top_script(profile) == "latin"


def test_script_profile_pure_digits_have_no_letters():
    profile = script_profile("1033")
    assert "common" in profile
    assert top_script(profile) == "common"


def test_every_probe_bank_language_passes_its_declared_script():
    """Golden set: all 14 hand-written languages in the repo's own verified
    probe corpus identify as their declared script."""
    docs, langs = [], set()
    i = 0
    for lang, examples in EXAMPLE_BANK.items():
        langs.add(lang)
        for ex in examples:
            i += 1
            docs.append(_doc(f"probe-{i:06d}", lang, ex))
    out = langid_documents(docs, LANGUAGE_SCRIPT)
    assert len(langs) == 14
    assert out.removed == (), [r.to_dict() for r in out.removed]
    assert out.stats["documents_out"] == len(docs)


def test_langid_removes_wrong_script_with_measured_profile():
    docs = [_doc("x-000001", "hi", "This is entirely Latin text.")]
    out = langid_documents(docs, LANGUAGE_SCRIPT)
    assert len(out.removed) == 1
    r = out.removed[0]
    assert r.reason == "langid_mismatch"
    assert r.detail["expected_script"] == "devanagari"
    assert r.detail["top_script"] == "latin"
    assert r.detail["share"] < 0.6


def test_langid_removes_letterless_documents():
    out = langid_documents([_doc("x-000001", "hi", "1033")], LANGUAGE_SCRIPT)
    assert out.removed[0].reason == "langid_no_letters"


def test_langid_keeps_mostly_declared_script_with_minor_code_mix():
    text = EXAMPLE_BANK["hi"][0] + " GPT-2 model"
    out = langid_documents([_doc("x-000001", "hi", text)], LANGUAGE_SCRIPT)
    assert out.kept and out.removed == ()


def test_langid_validation():
    docs = [_doc("x-000001", "hi", "यह")]
    with pytest.raises(ValueError, match="no declared script"):
        langid_documents(docs, {"bn": "bengali"})
    with pytest.raises(ValueError, match="min_declared_share"):
        langid_documents(docs, LANGUAGE_SCRIPT, min_declared_share=0.0)
    with pytest.raises(ValueError, match="min_declared_share"):
        langid_documents(docs, LANGUAGE_SCRIPT, min_declared_share=1.5)


def test_langid_per_language_stats():
    docs = [_doc("a-000001", "hi", "यह वाक्य"), _doc("a-000002", "en", "This is Latin.")]
    out = langid_documents(docs, LANGUAGE_SCRIPT)
    assert out.stats["per_language"]["hi"] == {"in": 1, "kept": 1, "removed": 0}
    assert out.stats["per_language"]["en"] == {"in": 1, "kept": 1, "removed": 0}


# ---------------------------------------------------------------------------
# D. quality filtering
# ---------------------------------------------------------------------------
def _clean_doc(lang: str = "hi") -> PipelineDocument:
    return _doc("q-000001", lang, EXAMPLE_BANK[lang][0])


def test_clean_document_passes_all_rules():
    out = quality_filter([_clean_doc()])
    assert out.kept and out.removed == () and out.flagged == {}
    assert out.stats["per_rule"] == {r: {"rejects": 0, "flags": 0} for r in out.stats["per_rule"]}


def test_control_chars_reject():
    doc = _doc("q-000001", "hi", "abc\x07def\x08ghi\x07jkl")  # 3 control / 16 chars
    out = quality_filter([doc])
    assert out.removed and out.removed[0].reason == "quality:control_chars"
    assert out.removed[0].detail["control_chars"] > 0.02


def test_url_density_flags_but_keeps():
    doc = _doc("q-000001", "en", "see https://a.example and https://b.example and https://c.example ok")
    out = quality_filter([doc])
    assert out.kept and out.flagged["q-000001"] == ("url_density",)


def test_template_residue_flags():
    doc = _doc("q-000001", "hi", "साँचा:GaP {{leftover}}")
    doc2 = _doc("q-000002", "hi", "normal line without braces")
    out = quality_filter([doc, doc2])
    assert out.flagged.get("q-000001") == ("template_residue",)
    assert "q-000002" not in out.flagged


def test_digit_runs_flags_long_runs_not_years():
    flagged = _doc("q-000001", "en", "ref 12345 end")
    year = _doc("q-000002", "en", "published in 1941 end")
    out = quality_filter([flagged, year])
    assert out.flagged.get("q-000001") == ("digit_runs",)
    assert "q-000002" not in out.flagged


def test_repetition_flags_degenerate_text_only():
    rep = _doc("q-000001", "en", "cat dog cat dog cat dog cat dog cat dog cat")
    diverse = _doc("q-000002", "en", "the model trains on a small local corpus every day")
    out = quality_filter([rep, diverse])
    assert "repetition" in out.flagged.get("q-000001", ())
    assert "q-000002" not in out.flagged


def test_short_text_repetition_is_insufficient_data_not_pass():
    out = quality_filter([_doc("q-000001", "en", "cat cat cat")])
    # 3 tokens -> no bigrams below the 8-token floor -> rule not flagged
    assert "q-000001" not in out.flagged


def test_min_chars_per_language_override():
    policy = QualityPolicy(per_language={"hi": {"min_chars": 50}})
    short_hi = _doc("q-000001", "hi", "छोटा")
    short_en = _doc("q-000002", "en", "short")
    out = quality_filter([short_hi, short_en], policy)
    removed_ids = {r.doc_id for r in out.removed}
    assert removed_ids == {"q-000001"}
    assert out.removed[0].reason == "quality:min_chars"
    assert {d.doc_id for d in out.kept} == {"q-000002"}


def test_max_chars_rejects_oversized_documents():
    doc = _doc("q-000001", "en", "word " * 4001)  # > 20000 chars
    out = quality_filter([doc])
    assert out.removed and out.removed[0].reason == "quality:max_chars"
    # detail carries every rule's measured value, not just the guilty one
    assert "min_chars" in out.removed[0].detail and "url_density" in out.removed[0].detail


def test_reject_takes_priority_over_flags_in_rule_order():
    # 18 chars/URL * 1200 = 21600 chars -> over the 20000 max_chars (REJECT),
    # and 1200 URLs / 21600 chars -> ~55 per 1000, far over the url_density
    # limit (FLAG). The reject reason must win; the flag is still counted on
    # the removed doc in per_rule stats.
    doc = _doc("q-000001", "en", "https://a.example " * 1200)
    out = quality_filter([doc])
    assert out.removed[0].reason == "quality:max_chars"
    assert out.stats["per_rule"]["max_chars"]["rejects"] == 1
    assert out.stats["per_rule"]["url_density"]["flags"] >= 1  # counted on the removed doc too


def test_quality_stats_per_language_and_counts():
    docs = [_clean_doc("hi"), _clean_doc("en"), _doc("q-000003", "en", "ref 98765 end")]
    out = quality_filter(docs)
    assert out.stats["per_language"]["hi"] == {"in": 1, "kept": 1, "removed": 0}
    assert out.stats["per_language"]["en"] == {"in": 2, "kept": 2, "removed": 0}
    assert out.stats["flagged_documents"] == 1


# ---------------------------------------------------------------------------
# E. exact deduplication
# ---------------------------------------------------------------------------
def test_within_source_duplicates_collapsed_to_first():
    docs = [
        _doc("a-000002", "hi", "same line", source_id="A"),
        _doc("a-000001", "hi", "same line", source_id="A"),
        _doc("a-000003", "hi", "same line", source_id="A"),
        _doc("a-000004", "hi", "other", source_id="A"),
    ]
    out = exact_dedup(docs)
    assert [d.doc_id for d in out.kept] == ["a-000001", "a-000004"]  # canonical order
    assert out.stats["removed"] == 2
    assert out.stats["duplicate_pairs"] == {"A": 2}


def test_cross_source_duplicate_keeps_first_source():
    docs = [
        _doc("b-000001", "bn", "একই লাইন", source_id="B"),
        _doc("a-000001", "bn", "একই লাইন", source_id="A"),
    ]
    out = exact_dedup(docs)
    assert [d.doc_id for d in out.kept] == ["a-000001"]
    assert out.removed[0].detail == {"first_seen": "a-000001", "first_source": "A"}
    assert out.stats["duplicate_pairs"] == {"A|B": 1}


def test_dedup_is_order_independent():
    docs = [
        _doc("s-000001", "ta", "ஒன்று", source_id="S1"),
        _doc("s-000002", "ta", "ஒன்று", source_id="S2"),
        _doc("s-000003", "ta", "இருந்து", source_id="S1"),
    ]
    base = exact_dedup(docs)
    for seed in (1, 2, 3):
        shuffled = docs[:]
        random.Random(seed).shuffle(shuffled)
        out = exact_dedup(shuffled)
        assert [d.doc_id for d in out.kept] == [d.doc_id for d in base.kept]
        assert out.stats == base.stats


def test_dedup_empty_input():
    out = exact_dedup([])
    assert out.kept == () and out.removed == () and out.stats["documents_in"] == 0


# ---------------------------------------------------------------------------
# F. stage composition (normalize -> langid -> quality -> dedup)
# ---------------------------------------------------------------------------
def test_stages_compose_in_order():
    docs = [
        _doc("m-000001", "hi", "  " + EXAMPLE_BANK["hi"][0] + "  "),
        _doc("m-000002", "hi", EXAMPLE_BANK["hi"][0]),  # exact duplicate of m-000001 after normalization
        _doc("m-000003", "hi", EXAMPLE_BANK["hi"][1]),
    ]
    n = normalize_documents(docs, "nfc")
    lang = langid_documents(list(n.kept), LANGUAGE_SCRIPT)
    q = quality_filter(list(lang.kept))
    d = exact_dedup(q.kept)
    assert d.stats["removed"] == 1  # the whitespace variant collapses into the clean one
    assert len(d.kept) == 2
    assert all(o.removed == () for o in (n, lang, q))
