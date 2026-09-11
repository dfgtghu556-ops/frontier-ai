"""Tests for the Project 004 Stage A tokenizer-research corpus foundation."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from frontier_ai.data.corpora import license_marker_found
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
    MAX_SOURCE_CHARS,
    STATUS_EVALUATED,
    STATUS_INSUFFICIENT,
    STATUS_NOT_EVALUATED,
    STATUS_UNVERIFIED,
    TokenizerCorpusManifest,
    build_corpus,
    coverage_report,
    documents_from_text,
    ingest_source,
    language_statistics,
    leakage_report,
    load_corpus,
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
) -> dict:
    return {
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


def _manifest(
    tmp_path: Path,
    sources: list[dict],
    *,
    slots: list[dict] | None = None,
    corpus_version: str = CORPUS_VERSION,
) -> Path:
    payload = {
        "schema_version": "1.0",
        "corpus": {"id": CORPUS_ID, "version": corpus_version, "title": "test"},
        "targets": {"sentences_per_language": 500, "characters_per_language": 200_000},
        "split": {"method": "deterministic-hash", "seed": 1337, "heldout_fraction": 0.1},
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
    stats = language_statistics(manifest, docs, [])
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
    stats = language_statistics(manifest, [], [])
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
    stats = language_statistics(manifest, docs, [item])
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
    stats = language_statistics(manifest, docs, [item])
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
    stats = language_statistics(manifest, docs, [item])
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
