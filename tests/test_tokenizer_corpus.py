"""Tests for the Project 004 Stage A tokenizer-research corpus foundation."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from frontier_ai.data.corpora import CorpusSource, license_marker_found
from frontier_ai.tokenization.corpus import CORPUS_ID as FIXTURE_CORPUS_ID
from frontier_ai.tokenization.corpus import CORPUS_VERSION as FIXTURE_CORPUS_VERSION
from frontier_ai.tokenization.corpus import (
    LANGUAGES,
    examples_by_category,
    examples_by_language,
    write_corpus,
)
from frontier_ai.tokenization.corpus import (
    load_corpus as load_fixture_corpus,
)
from frontier_ai.tokenization.research_corpus import (
    CORPUS_ID,
    CORPUS_VERSION,
    MAX_NGRAM_DOCS,
    MAX_SOURCE_CHARS,
    STATUS_EVALUATED,
    STATUS_INSUFFICIENT,
    STATUS_NOT_EVALUATED,
    STATUS_UNVERIFIED,
    TRAIN_FILENAME,
    LicenseEvidence,
    TokenizerCorpusManifest,
    build_corpus,
    coverage_report,
    documents_from_text,
    ingest_source,
    language_statistics,
    leakage_report,
    load_corpus,
    preflight_manifest,
    render_preflight,
    split_documents,
    validate_manifest,
)
from tokenizer_corpus_fixtures import (
    BENGALI_TEXT,
    ENGLISH_TEXT,
    HINDI_TEXT,
    LONG_PARAGRAPH,
    MARATHI_TEXT,
    TAMIL_TEXT,
    URDU_TEXT,
    fake_gutenberg_text,
    fake_wikisource_text,
)

REPO_MANIFEST = (
    Path(__file__).resolve().parents[1] / "corpora" / "tokenizer" / "indic-tokenizer-v2" / "sources.json"
)

ATTRIBUTION = "Attribution line for the test corpus source."
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"


def _source(
    source_id: str = "test-source",
    language: str = "hi",
    *,
    license_id: str = "CC-BY-4.0",
    license_url: str = "https://creativecommons.org/licenses/by/4.0/",
    kind: str = "wikitext",
    verified: bool = False,
    sha256: str | None = None,
    max_chars: int = 200_000,
    source_url: str = "https://example.invalid/test.txt",
    license_evidence: dict | None = None,
) -> dict:
    payload = {
        "id": source_id,
        "title": f"Test source {source_id}",
        "language": language,
        "script": "Devanagari" if language in {"hi", "mr"} else "Latin",
        "source_url": source_url,
        "license_id": license_id,
        "license_url": license_url,
        "attribution": ATTRIBUTION,
        "kind": kind,
        "max_chars": max_chars,
        "sha256": sha256,
        "verified": verified,
    }
    if license_evidence is not None:
        payload["license_evidence"] = license_evidence
    return payload


# What hi.wikisource.org's Meta siteinfo endpoint returns for its content licence.
SITEINFO_CC_BY_SA = json.dumps(
    {
        "batchcomplete": True,
        "query": {
            "general": {
                "rightsinfo": {
                    "url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "text": "Creative Commons Attribution-ShareAlike 4.0 License",
                }
            }
        },
    }
)
SITEINFO_OTHER_LICENCE = json.dumps(
    {"query": {"general": {"rightsinfo": {"url": "https://example.org/some-other-licence"}}}}
)


def _manifest(
    tmp_path: Path,
    sources: list[dict],
    *,
    slots: list[dict] | None = None,
    corpus_version: str = CORPUS_VERSION,
    split_seed: int = 1337,
) -> Path:
    payload = {
        "schema_version": "1.0",
        "corpus": {"id": CORPUS_ID, "version": corpus_version, "title": "test"},
        "targets": {"sentences_per_language": 500, "characters_per_language": 200_000},
        "split": {
            "method": "deterministic-hash",
            "seed": split_seed,
            "heldout_fraction": 0.1,
        },
        "normalization_policy": "none",
        "cleaning_policy": "kind-specific cleaner from frontier_ai.data.corpora",
        "language_slots": slots
        if slots is not None
        else [
            {"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": [s["id"] for s in sources]},
            {"code": "bn", "display": "Bengali", "script": "Bengali", "sources": [],
             "reason": "no licensed source verified yet"},
        ],
        "sources": sources,
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# manifest: validation, versioning, licences, hashes
# ---------------------------------------------------------------------------
def test_manifest_validates(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [_source()])
    manifest = TokenizerCorpusManifest.load(path)
    assert validate_manifest(manifest) == []


def test_manifest_rejects_unknown_schema_version(tmp_path: Path) -> None:
    payload = json.loads(_manifest(tmp_path, [_source()]).read_text(encoding="utf-8"))
    payload["schema_version"] = "9.9"
    path = tmp_path / "bad-schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported tokenizer corpus manifest schema"):
        TokenizerCorpusManifest.load(path)


def test_corpus_version_is_checked(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [_source()], corpus_version="nonsense")
    manifest = TokenizerCorpusManifest.load(path)
    assert any("corpus version" in problem for problem in validate_manifest(manifest))


def test_manifest_rejects_licence_outside_the_allow_list(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [_source(license_id="GPL-3.0")])
    manifest = TokenizerCorpusManifest.load(path)
    assert any("is not in" in problem for problem in validate_manifest(manifest))


def test_manifest_rejects_non_https_urls(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [_source(source_url="http://example.invalid/test.txt")])
    manifest = TokenizerCorpusManifest.load(path)
    assert any("must be https" in problem for problem in validate_manifest(manifest))


def test_manifest_rejects_verified_source_without_hash(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [_source(verified=True)])
    manifest = TokenizerCorpusManifest.load(path)
    assert any("verified=true requires a pinned sha256" in p for p in validate_manifest(manifest))


def test_manifest_rejects_malformed_hash(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [_source(sha256="not-a-hash", verified=True)])
    manifest = TokenizerCorpusManifest.load(path)
    assert any("64 lowercase hex" in p for p in validate_manifest(manifest))


def test_manifest_rejects_oversized_max_chars(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [_source(max_chars=MAX_SOURCE_CHARS + 1)])
    manifest = TokenizerCorpusManifest.load(path)
    assert any("max_chars" in p for p in validate_manifest(manifest))
    assert MAX_SOURCE_CHARS > 200_000  # the tokenizer corpus may exceed the smoke cap


def test_manifest_rejects_language_outside_the_research_set(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [_source(language="de")],
        slots=[{"code": "de", "display": "German", "sources": ["test-source"]}],
    )
    manifest = TokenizerCorpusManifest.load(path)
    assert any("is not one of the researched languages" in p for p in validate_manifest(manifest))


def test_manifest_rejects_duplicate_slots_and_unknown_keys(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [_source()],
        slots=[
            {"code": "hi", "display": "Hindi", "sources": ["test-source"]},
            {"code": "hi", "display": "Hindi again", "sources": ["test-source"]},
        ],
    )
    manifest = TokenizerCorpusManifest.load(path)
    assert any("duplicate language slots" in p for p in validate_manifest(manifest))

    bad = _manifest(
        tmp_path,
        [_source()],
        slots=[{"code": "hi", "sources": ["test-source"], "surprise": True}],
    )
    with pytest.raises(ValueError, match="unknown keys in language slot"):
        TokenizerCorpusManifest.load(bad)


def test_repo_manifest_is_usable() -> None:
    """The manifest that ships with the repo must validate and cover all 14 slots."""
    manifest = TokenizerCorpusManifest.load(REPO_MANIFEST)
    assert validate_manifest(manifest) == []
    assert manifest.corpus_id == CORPUS_ID
    assert manifest.corpus_version == CORPUS_VERSION
    assert sorted(slot.code for slot in manifest.language_slots) == sorted(
        [
            "as", "bn", "en", "gu", "hi", "hi-en", "kn", "ml", "mr",
            "or", "pa", "ta", "te", "ur",
        ]
    )
    assert manifest.normalization_policy == "none"
    for source in manifest.sources:
        assert source.verified is False or source.sha256, "verified sources must be pinned"


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def test_ingest_local_text_is_never_auto_verified(tmp_path: Path) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    item = ingest_source(manifest.sources[0], tmp_path / "raw", local_text=MARATHI_TEXT)
    assert item.has_text
    assert item.status == "local_unverified"
    assert item.verified is False  # a local file cannot make a slot EVALUATED
    assert (tmp_path / "raw" / f"{item.source_id}.provenance.json").exists()


def test_ingest_requires_the_licence_marker(tmp_path: Path, monkeypatch) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    source = manifest.sources[0]
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text", lambda *a, **k: "no marker here"
    )
    item = ingest_source(source, tmp_path / "raw", fetch=True)
    assert item.status == "licence_marker_missing"
    assert item.sha256 is None
    assert item.path is None
    assert "refusing to pin a hash" in item.error


def test_ingest_verifies_a_fetch_with_the_licence_marker(tmp_path: Path, monkeypatch) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(MARATHI_TEXT),
    )
    item = ingest_source(manifest.sources[0], tmp_path / "raw", fetch=True)
    assert item.status == "verified"
    assert item.verified is True
    assert len(item.sha256) == 64
    written = Path(item.path).read_text(encoding="utf-8")
    assert item.chars == len(written)
    assert MARATHI_TEXT.strip() in written  # the licence header is not corpus text
    assert license_marker_found(fake_wikisource_text(MARATHI_TEXT), manifest.sources[0])


def test_ingest_reports_a_failed_fetch(tmp_path: Path, monkeypatch) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))

    def _boom(*_args, **_kwargs):
        raise OSError("no network in this sandbox")

    monkeypatch.setattr("frontier_ai.tokenization.research_corpus.fetch_text", _boom)
    item = ingest_source(manifest.sources[0], tmp_path / "raw", fetch=True)
    assert item.status == "fetch_failed"
    assert "no network in this sandbox" in item.error
    assert item.chars == 0


def test_ingest_does_not_fetch_by_default(tmp_path: Path) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    item = ingest_source(manifest.sources[0], tmp_path / "raw")
    assert item.status == "not_attempted"
    assert item.path is None


def test_ingest_reuses_the_existing_gutenberg_cleaner(tmp_path: Path, monkeypatch) -> None:
    manifest = TokenizerCorpusManifest.load(
        _manifest(tmp_path, [_source(kind="gutenberg", license_id="PD-US", license_url=LICENSE_URL)])
    )
    body = fake_gutenberg_text()
    monkeypatch.setattr("frontier_ai.tokenization.research_corpus.fetch_text", lambda *a, **k: body)
    item = ingest_source(manifest.sources[0], tmp_path / "raw", fetch=True)
    assert item.status == "verified"
    text = Path(item.path).read_text(encoding="utf-8")
    assert "Project Gutenberg" not in text
    assert text.startswith("राम ने कहा")


def test_normalization_is_preserved(tmp_path: Path) -> None:
    """Stage A must never change Unicode normalization (that is a later stage)."""
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    composed = "Café " + HINDI_TEXT
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert unicodedata.is_normalized("NFC", composed)
    item = ingest_source(manifest.sources[0], tmp_path / "raw", local_text=decomposed)
    written = Path(item.path).read_text(encoding="utf-8")
    assert written == decomposed.strip()
    assert unicodedata.is_normalized("NFD", written)


# ---------------------------------------------------------------------------
# documents and split
# ---------------------------------------------------------------------------
def test_documents_are_paragraphs_and_never_drop_text() -> None:
    docs = documents_from_text("src", "hi", HINDI_TEXT)
    assert len(docs) == 10
    assert docs[0].doc_id == "src-000000"
    assert sum(doc.chars for doc in docs) == len(HINDI_TEXT) - 9  # newline separators only

    long_docs = documents_from_text("long", "hi", LONG_PARAGRAPH)
    assert len(long_docs) > 1
    joined = "".join(doc.text.replace("\n", " ") for doc in long_docs).replace(" ", "")
    assert joined == LONG_PARAGRAPH.replace(" ", "")
    assert max(doc.chars for doc in long_docs) <= 1_200


def test_split_is_deterministic_and_document_level() -> None:
    docs = documents_from_text("src", "hi", "\n".join(f"वाक्य {i} यहाँ है।" for i in range(500)))
    first = split_documents(docs, seed=1337, held_out_fraction=0.1)
    second = split_documents(docs, seed=1337, held_out_fraction=0.1)
    assert [d.doc_id for d in first[0]] == [d.doc_id for d in second[0]]
    assert [d.doc_id for d in first[1]] == [d.doc_id for d in second[1]]
    # roughly 10%, and always both sides populated for a corpus of this size
    assert 0.02 < len(first[1]) / len(docs) < 0.25
    assert len(first[0]) + len(first[1]) == len(docs)


def test_split_is_independent_of_document_order() -> None:
    docs = documents_from_text("src", "hi", "\n".join(f"वाक्य {i} यहाँ है।" for i in range(300)))
    shuffled = list(reversed(docs))
    held_out_a = {d.doc_id for d in split_documents(docs, seed=7)[1]}
    held_out_b = {d.doc_id for d in split_documents(shuffled, seed=7)[1]}
    assert held_out_a == held_out_b


def test_split_seed_changes_the_split_reproducibly() -> None:
    docs = documents_from_text("src", "hi", "\n".join(f"वाक्य {i} यहाँ है।" for i in range(500)))
    a = {d.doc_id for d in split_documents(docs, seed=1)[1]}
    b = {d.doc_id for d in split_documents(docs, seed=2)[1]}
    assert a != b
    assert a & b  # different seeds may share documents; they are not complementary by design


def test_split_rejects_degenerate_fractions() -> None:
    with pytest.raises(ValueError, match="held_out_fraction"):
        split_documents([], held_out_fraction=0.0)


# ---------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------
def test_leakage_report_finds_no_overlap_for_disjoint_data() -> None:
    train = documents_from_text("a", "hi", HINDI_TEXT)
    held_out = documents_from_text("b", "mr", MARATHI_TEXT)
    report = leakage_report(train, held_out)
    assert report["exact"]["overlap_count"] == 0
    assert report["exact"]["clean"] is True
    assert report["ngram"]["overlap_ratio"] == 0.0


def test_leakage_report_detects_exact_duplicates() -> None:
    train = documents_from_text("a", "hi", HINDI_TEXT)
    duplicate = documents_from_text("b", "hi", HINDI_TEXT)
    report = leakage_report(train, duplicate)
    assert report["exact"]["overlap_count"] == len(duplicate)
    assert report["exact"]["clean"] is False
    assert report["ngram"]["heldout_documents_with_overlap"] == len(duplicate)
    assert report["ngram"]["overlap_ratio"] == 1.0


def test_leakage_report_detects_partial_ngram_overlap() -> None:
    train = documents_from_text("a", "hi", HINDI_TEXT)
    mixed = documents_from_text("b", "hi", HINDI_TEXT.splitlines()[0] + " नया वाक्य जो अलग है।")
    report = leakage_report(train, mixed, n=4)
    assert report["exact"]["overlap_count"] == 0
    assert report["ngram"]["heldout_documents_with_overlap"] == 1
    assert report["ngram"]["train_documents_sampled"] == len(train)


def test_leakage_report_samples_large_train_sets_deterministically() -> None:
    many = documents_from_text("a", "hi", "\n".join([HINDI_TEXT] * 400))
    report = leakage_report(many, documents_from_text("b", "ta", TAMIL_TEXT))
    assert report["ngram"]["sampled"] is True
    assert report["ngram"]["train_documents_sampled"] == 2_000


# ---------------------------------------------------------------------------
# statistics and coverage
# ---------------------------------------------------------------------------
def test_statistics_are_reported_for_every_slot(tmp_path: Path) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    docs = documents_from_text("test-source", "hi", HINDI_TEXT)
    train, held_out = split_documents(docs, seed=manifest.split_seed,
                                      held_out_fraction=manifest.held_out_fraction)
    stats = language_statistics(manifest, train, held_out, [])
    assert {row["language"] for row in stats} == {"hi", "bn"}
    hindi = next(row for row in stats if row["language"] == "hi")
    bengali = next(row for row in stats if row["language"] == "bn")
    assert hindi["examples"] == 10
    assert hindi["chars"] == len(HINDI_TEXT) - 9
    assert hindi["bytes"] == len((HINDI_TEXT.replace("\n", "")).encode("utf-8"))
    assert hindi["train_examples"] + hindi["evaluation_examples"] == 10
    assert hindi["train_chars"] + hindi["evaluation_chars"] == hindi["chars"]
    assert bengali["examples"] == 0
    assert bengali["verification_status"] == "unverified"


def test_coverage_marks_missing_languages_not_evaluated(tmp_path: Path) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    stats = language_statistics(manifest, [], [], [])
    coverage = coverage_report(manifest, stats)
    statuses = {row["language"]: row["status"] for row in coverage["languages"]}
    assert statuses == {"hi": STATUS_UNVERIFIED, "bn": STATUS_NOT_EVALUATED}
    assert coverage["summary"][STATUS_NOT_EVALUATED] == 1
    assert coverage["summary"]["slots"] == 2
    reason = next(row for row in coverage["languages"] if row["language"] == "bn")["reason"]
    assert "no licensed source verified yet" in reason


def test_coverage_marks_verified_but_small_corpora_insufficient(tmp_path: Path, monkeypatch) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(MARATHI_TEXT),
    )
    item = ingest_source(manifest.sources[0], tmp_path / "raw", fetch=True)
    docs = documents_from_text("test-source", "hi", Path(item.path).read_text(encoding="utf-8"))
    train, held_out = split_documents(docs, seed=manifest.split_seed,
                                      held_out_fraction=manifest.held_out_fraction)
    stats = language_statistics(manifest, train, held_out, [item])
    coverage = coverage_report(manifest, stats)
    hindi = next(row for row in coverage["languages"] if row["language"] == "hi")
    assert hindi["status"] == STATUS_INSUFFICIENT
    assert hindi["sufficient"] is False
    assert 0 < hindi["percent_of_target_chars"] < 1
    assert "below target" in hindi["reason"]


def test_coverage_marks_meeting_targets_evaluated(tmp_path: Path) -> None:
    big_body = "\n".join([MARATHI_TEXT] * 4_000)
    manifest = TokenizerCorpusManifest.load(
        _manifest(tmp_path, [_source(max_chars=MAX_SOURCE_CHARS)])
    )
    source = dataclasses.replace(manifest.sources[0], verified=True, sha256="0" * 64)
    item = ingest_source(source, tmp_path / "raw", local_text=big_body)
    item.status = "verified"  # simulate a verified fetch of a large licensed source
    docs = documents_from_text("test-source", "hi", Path(item.path).read_text(encoding="utf-8"))
    train, held_out = split_documents(docs, seed=manifest.split_seed,
                                      held_out_fraction=manifest.held_out_fraction)
    stats = language_statistics(manifest, train, held_out, [item])
    coverage = coverage_report(manifest, stats)
    hindi = next(row for row in coverage["languages"] if row["language"] == "hi")
    assert hindi["status"] == STATUS_EVALUATED
    assert hindi["actual_sentences"] >= 500
    assert hindi["actual_chars"] >= 200_000


def test_coverage_never_counts_unverified_text_as_evaluated(tmp_path: Path) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    big = dataclasses.replace(manifest.sources[0], max_chars=MAX_SOURCE_CHARS)
    item = ingest_source(big, tmp_path / "raw", local_text="\n".join([MARATHI_TEXT] * 4_000))
    docs = documents_from_text("test-source", "hi", Path(item.path).read_text(encoding="utf-8"))
    train, held_out = split_documents(docs, seed=manifest.split_seed,
                                      held_out_fraction=manifest.held_out_fraction)
    stats = language_statistics(manifest, train, held_out, [item])
    coverage = coverage_report(manifest, stats)
    hindi = next(row for row in coverage["languages"] if row["language"] == "hi")
    assert hindi["status"] == STATUS_UNVERIFIED
    assert hindi["actual_chars"] >= 200_000  # honest about size, honest about status


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def test_build_writes_the_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(HINDI_TEXT),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]}],
    )
    result = build_corpus(manifest_path, tmp_path / "build", fetch=True)
    assert result.verified_sources == 1
    for name in ("corpus.json", "train.jsonl", "heldout.jsonl", "stats.json", "coverage.json",
                 "leakage.json"):
        assert (tmp_path / "build" / name).exists()
    assert (tmp_path / "build" / "sources" / "hi-src.txt").exists()
    assert (tmp_path / "build" / "sources" / "hi-src.provenance.json").exists()

    corpus = load_corpus(tmp_path / "build")
    assert corpus["corpus_id"] == CORPUS_ID
    assert corpus["corpus_version"] == CORPUS_VERSION
    assert corpus["split"]["level"] == "document"
    assert corpus["split"]["seed"] == 1337
    assert corpus["normalization_policy"] == "none"
    assert corpus["sources"][0]["ingest"]["status"] == "verified"


def test_build_is_reproducible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(HINDI_TEXT + "\n" + BENGALI_TEXT),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi"), _source("bn-src", "bn")],
        slots=[
            {"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]},
            {"code": "bn", "display": "Bengali", "script": "Bengali", "sources": ["bn-src"]},
        ],
    )
    first = build_corpus(manifest_path, tmp_path / "build-a", fetch=True)
    second = build_corpus(manifest_path, tmp_path / "build-b", fetch=True)
    for name in ("train.jsonl", "heldout.jsonl", "stats.json", "coverage.json", "leakage.json"):
        assert first.files[name]["sha256"] == second.files[name]["sha256"], name


def test_build_round_trips_with_no_exact_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(_distinct_body(400)),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi", max_chars=MAX_SOURCE_CHARS)],
        slots=[{"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]}],
    )
    result = build_corpus(manifest_path, tmp_path / "build", fetch=True)
    train_texts = {line["text"] for line in _read_jsonl(tmp_path / "build" / "train.jsonl")}
    heldout = _read_jsonl(tmp_path / "build" / "heldout.jsonl")
    assert heldout, "a corpus this size must have a held-out set"
    assert all(line["text"] not in train_texts for line in heldout)
    assert result.leakage["exact"]["overlap_count"] == 0
    assert result.results_payload()["exact_overlap_count"] == 0


def test_build_reports_unverified_when_the_fetch_fails(tmp_path: Path, monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise OSError("this sandbox has no route to the source host")

    monkeypatch.setattr("frontier_ai.tokenization.research_corpus.fetch_text", _boom)
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]}],
    )
    result = build_corpus(manifest_path, tmp_path / "build", fetch=True)
    assert result.verified_sources == 0
    assert result.train == [] and result.held_out == []
    assert result.coverage["languages"][0]["status"] == STATUS_UNVERIFIED
    assert "ingest status: fetch_failed" in result.coverage["languages"][0]["reason"]
    # an empty corpus must not report "no leakage" as if it had been checked
    assert result.leakage["note"].startswith("no documents to compare")
    assert result.results_payload()["sources_verified"] == 0
    payload = json.loads((tmp_path / "build" / "corpus.json").read_text(encoding="utf-8"))
    assert payload["sources"][0]["ingest"]["status"] == "fetch_failed"
    assert payload["sources"][0]["ingest"]["error"]


def test_build_excludes_unverified_text_by_default(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]}],
    )
    # no fetch: nothing is ingested, and nothing is silently invented
    result = build_corpus(manifest_path, tmp_path / "build")
    assert result.train == []
    assert result.coverage["languages"][0]["status"] == STATUS_UNVERIFIED


def test_build_rejects_an_invalid_manifest(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, [_source(license_id="GPL-3.0")])
    with pytest.raises(ValueError, match="invalid tokenizer corpus manifest"):
        build_corpus(manifest_path, tmp_path / "build")


def test_build_rejects_unknown_source_ids(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, [_source("hi-src", "hi")],
                              slots=[{"code": "hi", "sources": ["hi-src"]}])
    with pytest.raises(KeyError):
        build_corpus(manifest_path, tmp_path / "build", source_ids=["nope"])


def test_pin_writes_back_verified_hashes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(HINDI_TEXT),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]}],
    )
    result = build_corpus(manifest_path, tmp_path / "build", fetch=True, pin=True)
    pinned = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert pinned["sources"][0]["verified"] is True
    assert pinned["sources"][0]["sha256"] == result.ingested[0].sha256
    assert len(pinned["sources"][0]["sha256"]) == 64
    assert validate_manifest(TokenizerCorpusManifest.load(manifest_path)) == []


def test_load_corpus_detects_a_tampered_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(HINDI_TEXT),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]}],
    )
    build_corpus(manifest_path, tmp_path / "build", fetch=True)
    (tmp_path / "build" / "train.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        load_corpus(tmp_path / "build")


def test_load_corpus_detects_a_missing_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(HINDI_TEXT),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]}],
    )
    build_corpus(manifest_path, tmp_path / "build", fetch=True)
    (tmp_path / "build" / "heldout.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "build")


def _distinct_body(repeats: int) -> str:
    """`repeats` copies of the four fixtures, each line made unique.

    Uniqueness matters: a corpus that literally repeats the same paragraph would put
    byte-identical documents on both sides of the split, which is real duplication the
    diagnostics must flag — not something this no-overlap test may rely on.
    """
    lines = []
    for copy in range(repeats):
        for block in (HINDI_TEXT, ENGLISH_TEXT, TAMIL_TEXT, URDU_TEXT):
            for line in block.splitlines():
                lines.append(f"{copy}: {line}")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_tokenizer_corpus.py"


def _run_cli(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    import os
    import subprocess

    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        check=False,
    )


def test_cli_reports_coverage_without_recording(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [_source("hi-src", "hi")],
                         slots=[{"code": "hi", "sources": ["hi-src"]}])
    out = tmp_path / "out"
    done = _run_cli(["--manifest", str(manifest), "--out", str(out), "--no-record"])
    assert done.returncode == 0, done.stderr
    assert "UNVERIFIED      1  hi" in done.stdout
    assert "sources: 1 ingested, 0 verified" in done.stdout
    assert (out / "coverage.json").exists() and (out / "corpus.json").exists()
    assert json.loads((out / "coverage.json").read_text())["languages"][0]["status"] == (
        STATUS_UNVERIFIED
    )
    assert not (out / "experiment.json").exists()


def test_cli_records_the_build(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path, [_source("hi-src", "hi")],
                         slots=[{"code": "hi", "sources": ["hi-src"]}])
    out = tmp_path / "out"
    records = tmp_path / "out" / "experiments"  # docs/experiments.md convention
    monkeypatch.chdir(tmp_path)
    done = _run_cli(["--manifest", str(manifest), "--out", str(out), "--exp-id", "EXP-008"])
    assert done.returncode == 0, done.stderr
    assert not (out / "experiment.json").exists(), "the record must not pollute the corpus"
    record = json.loads((records / "EXP-008" / "experiment.json").read_text())
    assert record["experiment"]["id"] == "EXP-008"
    assert record["experiment"]["status"] == "success"
    assert "project-004" in record["experiment"]["tags"]
    assert record["data"]["inputs"] == [str(manifest)]
    assert record["results"]["slots_unverified"] == 1
    assert record["results"]["sources_verified"] == 0


def test_cli_defers_to_an_outer_record(tmp_path: Path) -> None:
    """D-032/D-034: a nested build writes no record of its own, and publishes results."""
    manifest = _manifest(tmp_path, [_source("hi-src", "hi")],
                         slots=[{"code": "hi", "sources": ["hi-src"]}])
    out = tmp_path / "out"
    outer = tmp_path / "outer"
    outer.mkdir()
    done = _run_cli(
        ["--manifest", str(manifest), "--out", str(out), "--exp-id", "EXP-008"],
        env={"FRONTIER_AI_EXPERIMENT_DIR": str(outer)},
    )
    assert done.returncode == 0, done.stderr
    assert "the outer run owns the experiment record" in done.stderr
    assert not (out / "experiment.json").exists()
    published = [
        json.loads(line)
        for line in done.stdout.splitlines()
        if line.strip().startswith("{") and "frontier_ai_nested_results" in line
    ]
    assert len(published) == 1
    payload = published[0]["frontier_ai_nested_results"]
    assert payload["schema"] == "1.0"
    assert payload["results"]["slots_unverified"] == 1


def test_cli_rejects_an_invalid_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [_source("hi-src", "hi", license_id="GPL-3.0")],
                         slots=[{"code": "hi", "sources": ["hi-src"]}])
    done = _run_cli(["--manifest", str(manifest), "--out", str(tmp_path / "out")])
    assert done.returncode == 2
    assert "manifest problem" in done.stderr


# ---------------------------------------------------------------------------
# H-1: the statistics must be computed from the split that was actually written
# ---------------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _language_counts(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in _jsonl(path):
        bucket = counts.setdefault(row["language"], {"examples": 0, "chars": 0})
        bucket["examples"] += 1
        bucket["chars"] += int(row["chars"])
    return counts


def _two_language_manifest(tmp_path: Path, seed: int = 1337) -> Path:
    return _manifest(
        tmp_path,
        [
            _source("hi-src", "hi", source_url="https://example.invalid/hi-src.txt"),
            _source("ta-src", "ta", source_url="https://example.invalid/ta-src.txt"),
        ],
        slots=[
            {"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]},
            {"code": "ta", "display": "Tamil", "script": "Tamil", "sources": ["ta-src"]},
        ],
        split_seed=seed,
    )


def _fake_fetch(bodies: dict[str, str], evidence: str | None = None,
                licence_header: bool = False):
    """A ``fetch_text`` stand-in that also answers the licence-evidence endpoint.

    ``evidence=None`` means the evidence host is unreachable; ``licence_header`` adds a
    licence marker to the text payload itself (the Project Gutenberg case).
    """

    def _fetch(url: str, *_args, **_kwargs) -> str:
        if "api.php" in url or "meta=siteinfo" in url:
            if evidence is None:
                raise OSError("no route to the evidence host")
            return evidence
        body = bodies[next(sid for sid in bodies if sid in url)]
        header = (
            "This work is licensed under the Creative Commons "
            "Attribution-ShareAlike 4.0 License.\n"
            if licence_header
            else ""
        )
        return header + body

    return _fetch


AS_SENTENCES = "\n".join(f"অসমীয়া বাক্য {i} ইয়াত আছে।" for i in range(40))


def test_statistics_match_the_artifacts_with_a_non_default_seed(tmp_path: Path, monkeypatch) -> None:
    """H-1: with a non-default seed, stats.json must agree with the JSONL files."""
    manifest_path = _two_language_manifest(tmp_path)
    bodies = {
        "hi-src": "\n".join(f"हिंदी वाक्य संख्या {i} है और यह परीक्षण के लिए है।" for i in range(400)),
        "ta-src": "\n".join(f"தமிழ் வாக்கியம் எண் {i} இங்கே உள்ளது." for i in range(400)),
    }
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fake_fetch(bodies, licence_header=True),
    )
    build_corpus(manifest_path, tmp_path / "build", fetch=True, seed=99)

    stats = json.loads((tmp_path / "build" / "stats.json").read_text())
    train_counts = _language_counts(tmp_path / "build" / "train.jsonl")
    heldout_counts = _language_counts(tmp_path / "build" / "heldout.jsonl")
    assert train_counts and heldout_counts

    for row in stats["languages"]:
        code = row["language"]
        assert row["train_examples"] == train_counts.get(code, {}).get("examples", 0), code
        assert row["evaluation_examples"] == heldout_counts.get(code, {}).get("examples", 0), code
        assert row["train_chars"] == train_counts.get(code, {}).get("chars", 0), code
        assert row["evaluation_chars"] == heldout_counts.get(code, {}).get("chars", 0), code
        assert row["train_examples"] + row["evaluation_examples"] == row["examples"]

    assert stats["split"]["seed"] == 99
    corpus = json.loads((tmp_path / "build" / "corpus.json").read_text())
    assert corpus["split"]["seed"] == 99


def test_stats_json_totals_equal_the_jsonl_files(tmp_path: Path, monkeypatch) -> None:
    bodies = {
        "hi-src": "\n".join(f"हिंदी वाक्य संख्या {i} है और यह परीक्षण के लिए है।" for i in range(400)),
        "ta-src": "\n".join(f"தமிழ் வாக்கியம் எண் {i} இங்கே உள்ளது." for i in range(400)),
    }
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fake_fetch(bodies, licence_header=True),
    )
    build_corpus(_two_language_manifest(tmp_path), tmp_path / "build", fetch=True)
    stats = json.loads((tmp_path / "build" / "stats.json").read_text())
    train = _jsonl(tmp_path / "build" / "train.jsonl")
    heldout = _jsonl(tmp_path / "build" / "heldout.jsonl")
    assert stats["totals"]["train_examples"] == len(train)
    assert stats["totals"]["evaluation_examples"] == len(heldout)
    assert stats["totals"]["examples"] == len(train) + len(heldout)
    assert stats["totals"]["chars"] == sum(row["chars"] for row in train + heldout)
    assert stats["totals"]["bytes"] == sum(row["bytes"] for row in train + heldout)


def test_cli_honours_the_manifest_seed_when_no_seed_is_given(tmp_path: Path) -> None:
    manifest_path = _two_language_manifest(tmp_path, seed=4242)
    out = tmp_path / "out"
    done = _run_cli(["--manifest", str(manifest_path), "--out", str(out), "--no-record"])
    assert done.returncode == 0, done.stderr
    corpus = json.loads((out / "corpus.json").read_text())
    assert corpus["split"]["seed"] == 4242, "the manifest seed must be the default"
    stats = json.loads((out / "stats.json").read_text())
    assert stats["split"]["seed"] == 4242


def test_cli_seed_overrides_the_manifest_seed(tmp_path: Path, monkeypatch) -> None:
    """An explicit --seed must change both the split and the reported seed."""
    manifest_path = _two_language_manifest(tmp_path, seed=4242)
    bodies = {
        "hi-src": "\n".join(f"हिंदी वाक्य संख्या {i} है और यह परीक्षण के लिए है।" for i in range(400)),
        "ta-src": "\n".join(f"தமிழ் வாக்கியம் எண் {i} இங்கே உள்ளது." for i in range(400)),
    }
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fake_fetch(bodies, licence_header=True),
    )
    default = build_corpus(manifest_path, tmp_path / "default", fetch=True)
    overridden = build_corpus(manifest_path, tmp_path / "seeded", fetch=True, seed=7)
    assert default.files[TRAIN_FILENAME]["sha256"] != overridden.files[TRAIN_FILENAME]["sha256"]
    assert json.loads((tmp_path / "seeded" / "corpus.json").read_text())["split"]["seed"] == 7


def test_reported_split_seed_is_the_one_used_for_statistics(tmp_path: Path, monkeypatch) -> None:
    """The seed in stats.json must be the seed the per-language counts came from."""
    manifest_path = _two_language_manifest(tmp_path)
    bodies = {
        "hi-src": "\n".join(f"हिंदी वाक्य संख्या {i} है और यह परीक्षण के लिए है।" for i in range(400)),
        "ta-src": "\n".join(f"தமிழ் வாக்கியம் எண் {i} இங்கே உள்ளது." for i in range(400)),
    }
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fake_fetch(bodies, licence_header=True),
    )
    result = build_corpus(manifest_path, tmp_path / "build", fetch=True, seed=99)
    stats = json.loads((tmp_path / "build" / "stats.json").read_text())
    train_ids = {row["doc_id"] for row in _jsonl(tmp_path / "build" / "train.jsonl")}
    recomputed = {
        document.doc_id
        for document in split_documents(result.train + result.held_out, seed=99)[0]
    }
    assert train_ids == recomputed
    assert stats["split"]["seed"] == 99


# ---------------------------------------------------------------------------
# M-1: identical documents must not cross the split
# ---------------------------------------------------------------------------
def test_identical_documents_land_on_the_same_side() -> None:
    repeated = "\n".join(["यह वाक्य कई बार दोहराया गया है।"] * 12)
    docs = documents_from_text("src", "hi", repeated)
    assert len({d.text_sha256 for d in docs}) == 1
    train, held_out = split_documents(docs, seed=1337)
    assert len(train) + len(held_out) == len(docs)
    assert not ({d.text_sha256 for d in train} & {d.text_sha256 for d in held_out})


def test_duplicate_documents_cannot_straddle_the_split(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: a source full of repeated paragraphs yields disjoint artifacts."""
    refrain = "राम ने कहा कि काम पूरा हो गया है और अब हम घर जा सकते हैं।"
    body = "\n".join([refrain] * 60 + [f"अलग अनुच्छेद संख्या {i} यहाँ लिखा है।" for i in range(200)])
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(body),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi", max_chars=MAX_SOURCE_CHARS)],
        slots=[{"code": "hi", "display": "Hindi", "script": "Devanagari", "sources": ["hi-src"]}],
    )
    result = build_corpus(manifest_path, tmp_path / "build", fetch=True)

    train_texts = {row["text"] for row in _jsonl(tmp_path / "build" / "train.jsonl")}
    heldout_texts = {row["text"] for row in _jsonl(tmp_path / "build" / "heldout.jsonl")}
    assert refrain in train_texts or refrain in heldout_texts
    assert not (train_texts & heldout_texts), "identical text appeared on both sides"
    assert result.leakage["exact"]["overlap_count"] == 0
    assert result.leakage["exact"]["clean"] is True

    stats = json.loads((tmp_path / "build" / "stats.json").read_text())
    hindi = next(row for row in stats["languages"] if row["language"] == "hi")
    assert hindi["duplicate_documents"] > 0
    assert hindi["unique_texts"] + hindi["duplicate_documents"] == hindi["examples"]


# ---------------------------------------------------------------------------
# M-2: language-aware n-gram sampling
# ---------------------------------------------------------------------------
def test_leakage_sampling_is_language_aware() -> None:
    hi = documents_from_text("hi-src", "hi", "\n".join(f"हिंदी वाक्य {i} है।" for i in range(2_500)))
    ta = documents_from_text("ta-src", "ta", "\n".join(f"தமிழ் வாக்கியம் {i}." for i in range(2_500)))
    report = leakage_report(hi + ta, documents_from_text("bn-src", "bn", BENGALI_TEXT))
    per_language = report["ngram"]["per_language_sampled"]
    assert per_language.get("hi", 0) > 0
    assert per_language.get("ta", 0) > 0
    assert report["ngram"]["train_documents_sampled"] == MAX_NGRAM_DOCS
    assert report["ngram"]["sampling_rule"].startswith("stratified")


def test_leakage_sampling_is_deterministic_and_order_independent() -> None:
    hi = documents_from_text("hi-src", "hi", "\n".join(f"हिंदी वाक्य {i} है।" for i in range(2_500)))
    ta = documents_from_text("ta-src", "ta", "\n".join(f"தமிழ் வாக்கியம் {i}." for i in range(2_500)))
    held_out = documents_from_text("bn-src", "bn", BENGALI_TEXT)
    first = leakage_report(hi + ta, held_out)
    second = leakage_report(list(reversed(ta)) + list(reversed(hi)), held_out)
    assert first["ngram"]["per_language_sampled"] == second["ngram"]["per_language_sampled"]
    assert first["ngram"]["overlap_ratio"] == second["ngram"]["overlap_ratio"]


def test_leakage_sampling_gives_every_language_a_share() -> None:
    """A language with few documents must not be squeezed out by a big one."""
    tiny = documents_from_text("as-src", "as", AS_SENTENCES)
    huge = documents_from_text("hi-src", "hi", "\n".join(f"हिंदी वाक्य {i} है।" for i in range(2_500)))
    report = leakage_report(huge + tiny, documents_from_text("bn-src", "bn", BENGALI_TEXT))
    assert report["ngram"]["per_language_sampled"].get("as", 0) == len(tiny)


# ---------------------------------------------------------------------------
# M-5: manifest reference validation
# ---------------------------------------------------------------------------
def test_manifest_rejects_unknown_source_reference(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "sources": ["hi-typo"]}],
    )
    manifest = TokenizerCorpusManifest.load(path)
    problems = validate_manifest(manifest)
    assert any("references unknown source id 'hi-typo'" in p for p in problems)
    with pytest.raises(ValueError, match="unknown source id"):
        build_corpus(path, tmp_path / "build")


def test_manifest_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [_source("hi-src", "hi"), _source("hi-src", "hi")],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    manifest = TokenizerCorpusManifest.load(path)
    assert any("duplicate source id 'hi-src'" in p for p in validate_manifest(manifest))


def test_manifest_rejects_orphan_sources(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [_source("hi-src", "hi"), _source("unused-src", "bn")],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    manifest = TokenizerCorpusManifest.load(path)
    assert any("not referenced by any language slot" in p for p in validate_manifest(manifest))


def test_repo_manifest_has_no_dangling_references() -> None:
    manifest = TokenizerCorpusManifest.load(REPO_MANIFEST)
    assert validate_manifest(manifest) == []
    declared = {source.id for source in manifest.sources}
    referenced = {sid for slot in manifest.language_slots for sid in slot.sources}
    assert declared == referenced
    for source_id in manifest.license_evidence:
        assert source_id in declared
        assert manifest.license_evidence[source_id].url.startswith("https://")


# ---------------------------------------------------------------------------
# M-6: truncation must be visible
# ---------------------------------------------------------------------------
def test_ingestion_records_truncation(tmp_path: Path) -> None:
    body = "\n".join(f"यह अनुच्छेद संख्या {i} है और काफी लंबा है।" for i in range(2_000))
    manifest = TokenizerCorpusManifest.load(
        _manifest(tmp_path, [_source(max_chars=1_000)])
    )
    item = ingest_source(manifest.sources[0], tmp_path / "raw", local_text=body)
    assert item.truncated is True
    assert item.chars < item.chars_before_trim
    assert item.chars <= 1_000

    provenance = json.loads(Path(item.provenance_path).read_text())
    assert provenance["truncated"] is True
    assert provenance["chars_before_trim"] == item.chars_before_trim
    assert provenance["max_chars"] == 1_000

    untrimmed = ingest_source(
        dataclasses.replace(manifest.sources[0], max_chars=MAX_SOURCE_CHARS),
        tmp_path / "raw",
        local_text=body,
    )
    assert untrimmed.truncated is False
    assert untrimmed.chars == untrimmed.chars_before_trim


def test_build_reports_truncation_per_language(tmp_path: Path) -> None:
    body = "\n".join(f"यह अनुच्छेद संख्या {i} है और काफी लंबा है।" for i in range(2_000))
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi", max_chars=1_000)],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    (tmp_path / "local.txt").write_text(body, encoding="utf-8")
    done = _run_cli(
        [
            "--manifest", str(manifest_path),
            "--local-file", f"hi-src={tmp_path / 'local.txt'}",
            "--include-unverified",
            "--out", str(tmp_path / "out"),
            "--no-record",
        ]
    )
    assert done.returncode == 0, done.stderr
    stats = json.loads((tmp_path / "out" / "stats.json").read_text())
    hindi = next(row for row in stats["languages"] if row["language"] == "hi")
    assert hindi["truncated_sources"] == 1
    corpus = json.loads((tmp_path / "out" / "corpus.json").read_text())
    assert corpus["sources"][0]["ingest"]["truncated"] is True


# ---------------------------------------------------------------------------
# M-7: --pin must not touch the manifest when it pins nothing
# ---------------------------------------------------------------------------
def test_pin_leaves_the_manifest_untouched_when_nothing_is_pinned(tmp_path: Path, monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise OSError("no route to the source host")

    monkeypatch.setattr("frontier_ai.tokenization.research_corpus.fetch_text", _boom)
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    before = manifest_path.read_bytes()
    build_corpus(manifest_path, tmp_path / "build", fetch=True, pin=True)
    assert manifest_path.read_bytes() == before, "--pin with zero pinned sources must be a no-op"


def test_pin_does_not_add_noise_fields(tmp_path: Path, monkeypatch) -> None:
    """When --pin does write, it must not invent empty candidates/reason keys."""
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: fake_wikisource_text(MARATHI_TEXT),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    build_corpus(manifest_path, tmp_path / "build", fetch=True, pin=True)
    written = json.loads(manifest_path.read_text())
    assert written["sources"][0]["verified"] is True
    for slot in written["language_slots"]:
        if not slot.get("candidates"):
            assert "candidates" not in slot
        if not slot.get("reason"):
            assert "reason" not in slot


# ---------------------------------------------------------------------------
# H-3: local file ingestion
# ---------------------------------------------------------------------------
def test_local_file_ingestion_is_local_unverified(tmp_path: Path) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source("hi-src", "hi")]))
    item = ingest_source(
        manifest.sources[0], tmp_path / "raw", local_text=MARATHI_TEXT,
        local_origin="/human/supplied/godan.txt",
    )
    assert item.status == "local_unverified"
    assert item.verified is False
    assert item.local is True
    assert item.local_origin == "/human/supplied/godan.txt"
    assert item.licence_proof == ""
    assert len(item.sha256) == 64  # a content hash, computed but not licence proof

    provenance = json.loads(Path(item.provenance_path).read_text())
    assert provenance["verification"] == "local_unverified"
    assert provenance["local_source"] is True
    assert provenance["local_origin_path"] == "/human/supplied/godan.txt"
    assert provenance["hash_is_licence_proof"] is False
    assert provenance["licence_proof"] == "none"


def test_local_text_never_becomes_evaluated(tmp_path: Path) -> None:
    big = "\n".join([MARATHI_TEXT] * 4_000)
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi", max_chars=MAX_SOURCE_CHARS)],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    (tmp_path / "local.txt").write_text(big, encoding="utf-8")
    done = _run_cli(
        [
            "--manifest", str(manifest_path),
            "--local-file", f"hi-src={tmp_path / 'local.txt'}",
            "--include-unverified",
            "--out", str(tmp_path / "out"),
            "--no-record",
        ]
    )
    assert done.returncode == 0, done.stderr

    corpus = json.loads((tmp_path / "out" / "corpus.json").read_text())
    assert corpus["sources"][0]["ingest"]["status"] == "local_unverified"
    assert corpus["sources"][0]["verified"] is False
    # the text really is in the corpus, and it still is not EVALUATED
    assert len(_jsonl(tmp_path / "out" / "train.jsonl")) > 500
    coverage = json.loads((tmp_path / "out" / "coverage.json").read_text())
    hindi = next(row for row in coverage["languages"] if row["language"] == "hi")
    assert hindi["status"] == STATUS_UNVERIFIED
    assert hindi["actual_chars"] >= 200_000
    assert hindi["sufficient"] is False


def test_local_text_is_excluded_until_include_unverified(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    (tmp_path / "local.txt").write_text(MARATHI_TEXT, encoding="utf-8")
    out = tmp_path / "out"
    done = _run_cli(
        ["--manifest", str(manifest_path), "--local-file", f"hi-src={tmp_path / 'local.txt'}",
         "--out", str(out), "--no-record"]
    )
    assert done.returncode == 0, done.stderr
    corpus = json.loads((out / "corpus.json").read_text())
    assert corpus["sources"][0]["ingest"]["status"] == "local_unverified"
    assert corpus["totals"]["documents"] == 0, "local text stays out unless asked for"


def test_local_dir_ingestion(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi"), _source("bn-src", "bn")],
        slots=[{"code": "hi", "sources": ["hi-src"]}, {"code": "bn", "sources": ["bn-src"]}],
    )
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "hi-src.txt").write_text(HINDI_TEXT, encoding="utf-8")
    (local_dir / "bn-src.txt").write_text(BENGALI_TEXT, encoding="utf-8")
    out = tmp_path / "out"
    done = _run_cli(
        ["--manifest", str(manifest_path), "--local-dir", str(local_dir),
         "--include-unverified", "--out", str(out), "--no-record"]
    )
    assert done.returncode == 0, done.stderr
    rows = _jsonl(out / "train.jsonl") + _jsonl(out / "heldout.jsonl")
    assert {row["language"] for row in rows} == {"hi", "bn"}


def test_local_dir_rejects_unknown_files(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "not-a-source.txt").write_text(HINDI_TEXT, encoding="utf-8")
    done = _run_cli(
        ["--manifest", str(manifest_path), "--local-dir", str(local_dir),
         "--out", str(tmp_path / "out"), "--no-record"]
    )
    assert done.returncode == 2
    assert "matches no declared source id" in done.stderr


def test_local_file_rejects_unknown_source(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    (tmp_path / "local.txt").write_text(HINDI_TEXT, encoding="utf-8")
    done = _run_cli(
        ["--manifest", str(manifest_path), "--local-file", f"nope={tmp_path / 'local.txt'}",
         "--out", str(tmp_path / "out"), "--no-record"]
    )
    assert done.returncode == 2
    assert "no declared source with id" in done.stderr


# ---------------------------------------------------------------------------
# H-2: licence evidence
# ---------------------------------------------------------------------------
def _evidence_manifest(tmp_path: Path) -> Path:
    evidence = {
        "url": "https://example.invalid/api.php?action=query&meta=siteinfo&siprop=rightsinfo",
        "kind": "mediawiki-api",
        "marker": "https://creativecommons.org/licenses/by-sa/4.0/",
        "scope": "site",
        "note": "site-level rights declaration",
    }
    return _manifest(
        tmp_path,
        [_source("hi-src", "hi", license_id="CC-BY-SA-4.0",
                 license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                 source_url="https://example.invalid/hi-src.txt",
                 license_evidence=evidence)],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )


def test_licence_evidence_verifies_a_payload_without_a_marker(tmp_path: Path, monkeypatch) -> None:
    """The Wikisource case: bare wikitext + a separate rights endpoint."""
    manifest_path = _evidence_manifest(tmp_path)
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fake_fetch({"hi-src": HINDI_TEXT}, evidence=SITEINFO_CC_BY_SA),
    )
    manifest = TokenizerCorpusManifest.load(manifest_path)
    item = ingest_source(
        manifest.sources[0],
        tmp_path / "raw",
        fetch=True,
        license_evidence=manifest.evidence_for("hi-src"),
    )
    assert item.status == "verified"
    assert item.licence_proof == "licence-evidence"
    assert item.license_evidence["status"] == "ok"
    assert item.license_evidence["marker_found"] is True
    assert item.license_evidence["scope"] == "site"
    assert item.license_evidence["url"].startswith("https://")
    assert item.license_evidence["checked_at"]

    provenance = json.loads(Path(item.provenance_path).read_text())
    assert provenance["licence_proof"] == "licence-evidence"
    assert provenance["license_evidence"]["status"] == "ok"
    assert provenance["license_evidence"]["scope"] == "site"


def test_licence_evidence_failure_keeps_the_source_unverified(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _evidence_manifest(tmp_path)
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fake_fetch({"hi-src": HINDI_TEXT}, evidence=None),  # evidence host unreachable
    )
    manifest = TokenizerCorpusManifest.load(manifest_path)
    item = ingest_source(
        manifest.sources[0],
        tmp_path / "raw",
        fetch=True,
        license_evidence=manifest.evidence_for("hi-src"),
    )
    assert item.status == "licence_marker_missing"
    assert item.verified is False
    assert item.sha256 is None
    assert item.license_evidence["status"] == "fetch_failed"
    assert "did not confirm it" in item.error


def test_licence_evidence_with_the_wrong_licence_is_rejected(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _evidence_manifest(tmp_path)
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fake_fetch({"hi-src": HINDI_TEXT}, evidence=SITEINFO_OTHER_LICENCE),
    )
    manifest = TokenizerCorpusManifest.load(manifest_path)
    item = ingest_source(
        manifest.sources[0],
        tmp_path / "raw",
        fetch=True,
        license_evidence=manifest.evidence_for("hi-src"),
    )
    assert item.status == "licence_marker_missing"
    assert item.license_evidence["status"] == "marker_not_found"
    assert item.license_evidence["accepted"] is False


def test_licence_evidence_ignores_malformed_payloads(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _evidence_manifest(tmp_path)
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fake_fetch({"hi-src": HINDI_TEXT}, evidence="<html>not json</html>"),
    )
    manifest = TokenizerCorpusManifest.load(manifest_path)
    item = ingest_source(
        manifest.sources[0],
        tmp_path / "raw",
        fetch=True,
        license_evidence=manifest.evidence_for("hi-src"),
    )
    assert item.status == "licence_marker_missing"
    assert item.license_evidence["status"] == "malformed"


def test_no_evidence_and_no_marker_stays_unverified(tmp_path: Path, monkeypatch) -> None:
    """Without a declared evidence endpoint, an unmarked payload cannot verify."""
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi")],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        lambda *a, **k: HINDI_TEXT,  # no licence marker, no evidence configured
    )
    manifest = TokenizerCorpusManifest.load(manifest_path)
    item = ingest_source(manifest.sources[0], tmp_path / "raw", fetch=True)
    assert item.status == "licence_marker_missing"
    assert item.license_evidence is None
    assert "no licence evidence endpoint is declared" in item.error


def test_manifest_rejects_invalid_licence_evidence(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [_source("hi-src", "hi", license_evidence={
            "url": "http://example.invalid/api.php",  # not https
            "kind": "made-up-kind",
            "scope": "everywhere",
        })],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    manifest = TokenizerCorpusManifest.load(path)
    problems = validate_manifest(manifest)
    assert any("licence evidence url must be https" in p for p in problems)
    assert any("licence evidence kind must be one of" in p for p in problems)
    assert any("licence evidence scope must be one of" in p for p in problems)


def test_licence_evidence_defaults_to_declaring_the_licence_url() -> None:
    """With no explicit marker, the evidence must confirm the declared licence URL."""
    evidence = LicenseEvidence(url="https://example.invalid/api.php", kind="mediawiki-api")
    source = CorpusSource.from_dict(_source("hi-src", "hi"))
    assert evidence.effective_marker(source) == source.license_url


# ---------------------------------------------------------------------------
# preflight (read-only probe: no acquisition, no verification, no pinning)
# ---------------------------------------------------------------------------
ALICE_SAMPLE = (
    "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***\n"
    + "\n".join(f"Alice was beginning to get very tired of sitting by her sister {i}." for i in range(400))
)
INDEX_PAGE_SAMPLE = "{{header}}\n" + "\n".join(
    f"[[गोदान/अध्याय {i}|अध्याय {i}]]" for i in range(1, 40)
)
WORK_PAGE_SAMPLE = "\n".join(
    f"होरी का मन आज बहुत उदास था, क्योंकि गाँव में सब कुछ बदल गया था। ({i})" for i in range(400)
)


def _probe(text: str, *, ok: bool = True, status: int = 200, content_type: str = "text/plain"):
    """A stand-in for ``probe_url``: no sockets, no files, deterministic payloads."""

    def _fake(url: str, max_bytes: int = 65_536, timeout: float = 30.0) -> dict:
        if not ok:
            return {
                "url": url, "ok": False, "http_status": None, "content_type": "",
                "bytes_read": 0, "elapsed_s": 0.01, "text": "",
                "error": "URLError: <urlopen error TLS/SSL connection has been closed (EOF)>",
            }
        payload = text(url) if callable(text) else text
        raw = payload.encode("utf-8")[:max_bytes]
        return {
            "url": url, "ok": True, "http_status": status, "content_type": content_type,
            "bytes_read": len(raw), "elapsed_s": 0.02,
            "text": raw.decode("utf-8", errors="replace"), "error": "",
        }

    return _fake


def test_preflight_reports_reachability_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.probe_url", _probe(ALICE_SAMPLE)
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("en-src", "en", kind="gutenberg", license_id="PD-US",
                 license_url=LICENSE_URL)],
        slots=[{"code": "en", "sources": ["en-src"]}],
    )
    manifest = TokenizerCorpusManifest.load(manifest_path)
    report = preflight_manifest(manifest)

    row = report["sources"][0]
    assert row["ok"] is True and row["http_status"] == 200
    assert row["payload_licence_marker_seen"] is True
    assert row["gutenberg_marker_seen"] is True
    assert row["sample_lines"] > 100
    assert row["looks_like_index_page"] is False
    assert report["summary"] == {"sources": 1, "endpoints": 1, "reachable": 1, "unreachable": 0}

    # nothing was stored, verified or pinned
    assert not (tmp_path / "build").exists()
    assert manifest_path.read_bytes() == _manifest_bytes(manifest_path)
    rendered = render_preflight(report)
    assert "nothing is stored, verified or pinned" in rendered
    assert "read-only probe" in rendered


def _manifest_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()


def test_preflight_reports_unreachable_endpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.probe_url", _probe("", ok=False)
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi", license_evidence={
            "url": "https://example.invalid/api.php?action=query&meta=siteinfo",
            "kind": "mediawiki-api", "marker": "https://creativecommons.org/licenses/by-sa/4.0/",
            "scope": "site"})],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    report = preflight_manifest(TokenizerCorpusManifest.load(manifest_path))
    row = report["sources"][0]
    assert row["ok"] is False and row["bytes_read"] == 0
    assert "unreachable" in row["notes"][0]
    assert row["evidence"]["ok"] is False
    assert report["summary"]["unreachable"] == 2  # source endpoint + evidence endpoint


def test_preflight_probes_the_evidence_endpoint_separately(tmp_path: Path, monkeypatch) -> None:
    """Reachable text + reachable evidence, but preflight must still verify nothing."""
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.probe_url",
        _probe(lambda url: SITEINFO_CC_BY_SA if "api.php" in url else WORK_PAGE_SAMPLE),
    )
    manifest_path = _manifest(
        tmp_path,
        [_source("hi-src", "hi", license_evidence={
            "url": "https://example.invalid/api.php?action=query&meta=siteinfo",
            "kind": "mediawiki-api", "marker": "https://creativecommons.org/licenses/by-sa/4.0/",
            "scope": "site"})],
        slots=[{"code": "hi", "sources": ["hi-src"]}],
    )
    manifest = TokenizerCorpusManifest.load(manifest_path)
    report = preflight_manifest(manifest)
    evidence = report["sources"][0]["evidence"]

    assert evidence["ok"] is True
    assert evidence["marker_seen_in_sample"] is True
    assert evidence["scope"] == "site"
    assert "never verifies" in evidence["note"]
    # the source itself is still unverified, and the manifest is untouched
    assert manifest.sources[0].verified is False and manifest.sources[0].sha256 is None
    assert manifest_path.read_bytes() == _manifest_bytes(manifest_path)


def test_preflight_flags_an_index_like_wikisource_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.probe_url", _probe(INDEX_PAGE_SAMPLE)
    )
    manifest = TokenizerCorpusManifest.load(
        _manifest(tmp_path, [_source("hi-src", "hi")],
                  slots=[{"code": "hi", "sources": ["hi-src"]}])
    )
    row = preflight_manifest(manifest)["sources"][0]
    assert row["looks_like_index_page"] is True
    assert row["link_line_ratio"] > 0.5
    assert any("contents/index page" in note for note in row["notes"])


def test_preflight_never_fetches_through_the_corpus_path(tmp_path: Path, monkeypatch) -> None:
    """Preflight must not call fetch_text, which is the acquisition path."""
    manifest = TokenizerCorpusManifest.load(
        _manifest(tmp_path, [_source("hi-src", "hi")],
                  slots=[{"code": "hi", "sources": ["hi-src"]}])
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("preflight must not call fetch_text")

    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.probe_url", _probe(WORK_PAGE_SAMPLE)
    )
    monkeypatch.setattr("frontier_ai.tokenization.research_corpus.fetch_text", _boom)
    report = preflight_manifest(manifest)
    assert report["sources"][0]["ok"] is True


def test_cli_preflight_reports_unreachable_endpoints_without_writing(tmp_path: Path) -> None:
    """example.invalid never resolves, so this holds on any machine with or without web access."""
    out = tmp_path / "out"
    done = _run_cli(["--manifest", str(_two_language_manifest(tmp_path)), "--preflight",
                     "--timeout", "5", "--out", str(out)])
    assert done.returncode == 1, done.stdout
    assert "[preflight] nothing is stored, verified or pinned" in done.stdout
    assert "unreachable" in done.stdout
    assert "summary: 2 endpoints, 0 reachable, 2 unreachable" in done.stdout
    assert not out.exists(), "preflight must not create an output directory"


def test_cli_preflight_print_json_is_read_only(tmp_path: Path) -> None:
    out = tmp_path / "out"
    done = _run_cli(["--manifest", str(_two_language_manifest(tmp_path)), "--preflight",
                     "--print-json", "--timeout", "5", "--out", str(out)])
    assert done.returncode == 1
    report = json.loads(done.stdout)
    assert report["summary"]["endpoints"] == 2
    assert {row["source_id"] for row in report["sources"]} == {"hi-src", "ta-src"}
    assert "No text was stored, no source was verified" in report["note"]
    assert all(row["evidence"] is None for row in report["sources"])
    assert not out.exists()


def test_cli_preflight_rejects_an_invalid_manifest(tmp_path: Path) -> None:
    done = _run_cli(["--manifest",
                     str(_manifest(tmp_path, [_source("hi-src", "hi", license_id="GPL-3.0")],
                                   slots=[{"code": "hi", "sources": ["hi-src"]}])),
                     "--preflight"])
    assert done.returncode == 2
    assert "manifest problem" in done.stderr


# ---------------------------------------------------------------------------
# regression: the Project 002 fixture corpus must be untouched
# ---------------------------------------------------------------------------
def test_indic_eval_v1_fixture_is_unchanged(tmp_path: Path) -> None:
    """The Project 002 fixture corpus must still build, load and hash as it always did."""
    manifest = write_corpus(tmp_path / "indic-eval-v1", seed=0)
    examples, loaded = load_fixture_corpus(tmp_path / "indic-eval-v1")
    assert manifest.corpus_id == loaded.corpus_id == FIXTURE_CORPUS_ID == "indic-eval"
    assert manifest.corpus_version == loaded.corpus_version == FIXTURE_CORPUS_VERSION == "v1"
    assert len(examples) == 179
    assert {example.lang for example in examples} == set(LANGUAGES) | {"mixed"}
    assert len(examples_by_category(examples)["unicode_marks"]) >= 8
    assert loaded.train_sha256 == manifest.train_sha256
    assert len(loaded.train_sha256) == 64
    # the fixture is a probe set, not a corpus: every language is a handful of sentences
    by_language = examples_by_language(examples)
    assert max(len(group) for lang, group in by_language.items() if lang != "mixed") <= 8
    assert len(by_language["mixed"]) == 70  # the non-language orthography probes
