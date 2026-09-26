"""FrontierCorpus v1 part 2: registry, split, shards, manifest, and the builder script.

The builder e2e test runs the real ``scripts/build_frontier_corpus.py`` end to end
on a tiny *fake* frozen corpus built under ``tmp_path`` (two sources, one per
language, with real hash pins and a real FREEZE.json) — no network, no real
corpus text. The pilot run on the actual frozen 59-source corpus happens on a
network-enabled machine (EXP-027).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.corpus import (
    PipelineDocument,
    check_shards,
    language_scripts,
    load_frontier_registry,
    pack,
    seeded_shuffle,
    train_holdout,
)
from frontier_ai.corpus.manifest import build_manifest, write_manifest
from frontier_ai.corpus.registry import RegistryError
from frontier_ai.corpus.shards import write_shards

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_MANIFEST = REPO_ROOT / "corpora/tokenizer/indic-tokenizer-v2/sources.json"
V2_FREEZE = REPO_ROOT / "corpora/tokenizer/indic-tokenizer-v2/FREEZE.json"


def _doc(doc_id: str, language: str, text: str, source_id: str = "s") -> PipelineDocument:
    return PipelineDocument(doc_id=doc_id, source_id=source_id, language=language, text=text)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_registry_imports_the_real_frozen_corpus():
    registry = load_frontier_registry(V2_MANIFEST, V2_FREEZE)
    assert len(registry) == 59
    assert {s.domain for s in registry} == {"literature"}
    scripts = language_scripts(registry)
    assert scripts["hi"] == "devanagari"
    assert scripts["mr"] == "devanagari"  # same-script: the gate cannot tell them apart
    assert scripts["bn"] == "bengali"
    assert scripts["as"] == "bengali"
    assert scripts["ur"] == "arabic"
    assert "hi-en" not in scripts  # the slot has no sources


def test_registry_refuses_a_tampered_manifest(tmp_path):
    manifest = V2_MANIFEST.read_bytes()
    tampered = tmp_path / "sources.json"
    tampered.write_bytes(manifest[:-20] + b"~" * 20)
    with pytest.raises(RegistryError, match="does not match the freeze identity"):
        load_frontier_registry(tampered, V2_FREEZE)


def test_registry_refuses_an_unfrozen_record(tmp_path):
    freeze = json.loads(V2_FREEZE.read_text(encoding="utf-8"))
    freeze["status"] = "draft"
    path = tmp_path / "FREEZE.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(RegistryError, match="not a frozen record"):
        load_frontier_registry(V2_MANIFEST, path)


# ---------------------------------------------------------------------------
# split (reuse of the P004B content-hash split)
# ---------------------------------------------------------------------------
def test_split_keeps_identical_text_on_one_side():
    docs = [
        _doc("a-000001", "hi", "एक ही वाक्य है।", source_id="A"),
        _doc("b-000001", "hi", "एक ही वाक्य है।", source_id="B"),  # same content, other source
        _doc("a-000002", "hi", "दूसरा वाक्य।", source_id="A"),
    ] * 4  # 12 documents, several exact duplicates
    train_out, held = train_holdout(docs)
    train_ids = {d.doc_id for d in train_out.kept}
    held_ids = {d.doc_id for d in held}
    assert train_ids.isdisjoint(held_ids)
    # content-keying: every copy of the shared text landed on the same side
    shared = [d for d in docs if d.sha256 == docs[0].sha256]
    assert len(shared) == 8  # 2 docs of the same text x 4 repeats
    sides = {("train" if d.doc_id in train_ids else "heldout") for d in shared}
    assert len(sides) == 1  # the duplicates cannot straddle the split


def test_split_is_reproducible_and_roughly_fractional():
    docs = [_doc(f"s-{i:06d}", "en", f"sentence number {i} here") for i in range(200)]
    (o1, h1), (o2, h2) = train_holdout(docs), train_holdout(docs)
    assert [d.doc_id for d in o1.kept] == [d.doc_id for d in o2.kept]
    assert [d.doc_id for d in h1] == [d.doc_id for d in h2]
    held_fraction = len(h1) / 200
    assert 0.03 < held_fraction < 0.17  # 10% with sampling noise


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------
def test_pack_respects_budget_and_keeps_documents_atomic():
    docs = [_doc(f"s-{i:06d}", "en", "x" * 40) for i in range(10)]  # 40 chars each
    shards = pack(docs, 100, "train")
    total = sum(len(sh.documents) for sh in shards)
    assert total == 10
    for sh in shards:
        assert sh.chars <= 100 or len(sh.documents) == 1  # budget or a lone oversized doc
    # exact accounting: shard chars = document chars + the joining newlines
    assert sum(sh.chars for sh in shards) == sum(d.chars for d in docs) + sum(
        len(sh.documents) - 1 for sh in shards
    )


def test_pack_oversized_document_gets_own_shard():
    docs = [_doc("s-000001", "en", "y" * 500), _doc("s-000002", "en", "z" * 10)]
    shards = pack(docs, 100, "train")
    assert len(shards) == 2
    assert [d.doc_id for d in shards[0].documents] == ["s-000001"]


def test_pack_empty_input():
    assert pack([], 100, "train") == ()


def test_seeded_shuffle_is_deterministic_and_moves_documents():
    docs = [_doc(f"s-{i:06d}", "en", f"t{i}") for i in range(50)]
    a = seeded_shuffle(docs, 1337)
    b = seeded_shuffle(docs, 1337)
    c = seeded_shuffle(docs, 1338)
    assert [d.doc_id for d in a] == [d.doc_id for d in b]
    assert [d.doc_id for d in a] != [d.doc_id for d in c]
    assert {d.doc_id for d in a} == {d.doc_id for d in docs}


def test_write_and_check_shards(tmp_path):
    docs = [_doc(f"s-{i:06d}", "hi", "पंक्ति नं " + "क" * (i % 7)) for i in range(9)]
    shards = pack(seeded_shuffle(docs, 7), 200, "train")
    write_shards(shards, tmp_path)
    assert check_shards(tmp_path, {sh.name: sh.sha256 for sh in shards}) == []
    # tamper one byte -> detected
    first = sorted((tmp_path).glob("*.txt"))[0]
    data = first.read_bytes()
    first.write_bytes(data[:-1] + (b"\x00" if data[-1:] != b"\x00" else b"\x01"))
    problems = check_shards(tmp_path, {sh.name: sh.sha256 for sh in shards})
    assert any("hash mismatch" in p for p in problems)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def _mini_registry():
    from frontier_ai.corpus import FrontierSource

    return (
        FrontierSource(
            source_id="a", title="A", language="hi", script="Devanagari", domain="literature",
            kind="mediawiki-parse", license_id="CC-BY-SA-4.0", source_url="https://x/a",
            sha256="0" * 64, retrieved_at="2026-01-01T00:00:00Z",
        ),
    )


def test_manifest_content_identity_is_deterministic(tmp_path):
    kwargs = dict(
        git_sha="abc123",
        registry=_mini_registry(),
        source_registry_path="corpora/tokenizer/indic-tokenizer-v2/sources.json",
        freeze_sha256="0" * 64,
        seed=1337,
        held_out_fraction=0.1,
        normalization_policy="nfc",
        policy_version="1",
        split_stats={"method": "deterministic-content-hash", "seed": 1337, "held_out_fraction": 0.1,
                     "level": "document", "train": {}, "held_out": {}},
        stages=(),
        train_shards=pack([_doc("a-000001", "hi", "पहली पंक्ति")], 1000, "train"),
        heldout_shards=pack([_doc("a-000002", "hi", "दूसरी पंक्ति")], 1000, "heldout"),
        train_per_language={"hi": {"documents": 1, "chars": 9}},
        heldout_per_language={"hi": {"documents": 1, "chars": 11}},
    )
    m1 = build_manifest(created_at="2026-09-26T00:00:00+00:00", **kwargs)
    m2 = build_manifest(created_at="2030-01-01T00:00:00+00:00", **kwargs)
    assert m1["created_at"] != m2["created_at"]
    assert m1["content_sha256"] == m2["content_sha256"]  # identity ignores the timestamp

    p1 = tmp_path / "m1.json"
    p2 = tmp_path / "m2.json"
    s1 = write_manifest(m1, p1)
    s2 = write_manifest(m2, p2)
    assert s1 == hashlib.sha256(p1.read_bytes()).hexdigest()
    assert s2 == hashlib.sha256(p2.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# builder script e2e on a tiny fake frozen corpus (offline)
# ---------------------------------------------------------------------------
def _make_fake_frozen_corpus(root: Path) -> dict:
    """Two sources (hi + en) with real pins and a real FREEZE.json, under ``root``."""
    hi_text = "मेरी जान तुम हो।\nतुम्हारे बिना मैं कुछ नहीं।\n" * 3 + "अलग लाइन यहाँ लिखी गई है।\n"
    # two separate lines -> two distinct English documents
    en_text = "The fox jumps over the lazy dog.\nA separate English line here.\n"
    sources_dir = root / "corpus" / "sources"
    sources_dir.mkdir(parents=True)
    files = {
        "hi-fake-book": (hi_text, "hi", "Devanagari"),
        "en-fake-book": (en_text, "en", "Latin"),
    }
    sources = []
    for source_id, (text, language, script) in files.items():
        path = sources_dir / f"{source_id}.txt"
        path.write_text(text, encoding="utf-8", newline="\n")
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(
            {
                "id": source_id, "title": f"Fake {source_id}", "language": language, "script": script,
                "source_url": f"https://example.org/{source_id}", "license_id": "CC-BY-SA-4.0",
                "attribution": "test fixture", "max_chars": 100000, "kind": "mediawiki-parse",
                "sha256": pin, "verified": True, "retrieved_at": "2026-09-26T00:00:00Z", "notes": "test",
            }
        )
    manifest_path = root / "sources.json"
    manifest_bytes = (json.dumps({"schema_version": "1.0", "sources": sources}, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    freeze_path = root / "FREEZE.json"
    freeze_path.write_text(
        json.dumps(
            {"schema_version": "1.0", "corpus_id": "indic-tokenizer", "corpus_version": "v2",
             "status": "frozen", "manifest": {"path": str(manifest_path),
                                              "sha256": hashlib.sha256(manifest_bytes).hexdigest()}},
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "freeze": freeze_path, "corpus_dir": root / "corpus"}


def _run_builder(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/build_frontier_corpus.py"), *args],
        capture_output=True, text=True, timeout=300,
    )


def test_builder_e2e_on_fake_frozen_corpus(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path)
    out = tmp_path / "out"
    proc = _run_builder(
        ["--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]), "--out", str(out), "--no-record"]
    )
    assert proc.returncode == 0, proc.stderr

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    # 7 hi lines (3 unique) + 2 en lines = 9 in; dedup removes the 4 duplicate hi lines -> 5
    assert manifest["sides"]["train"]["documents"] + manifest["sides"]["held_out"]["documents"] == 5
    # the split preserved the per-language document counts (no document lost to the split)
    for lang, expected in (("hi", 3), ("en", 2)):
        total = (
            manifest["sides"]["train"]["per_language"].get(lang, {}).get("documents", 0)
            + manifest["sides"]["held_out"]["per_language"].get(lang, {}).get("documents", 0)
        )
        assert total == expected, (lang, total)
    assert manifest["identity"]["split"]["seed"] == 1337
    dedup = [s for s in manifest["stages"] if s["stage"] == "exact_dedup"][0]
    assert dedup["stats"]["removed"] == 4
    # per-language breakdown — the shape the 396-duplicate cross-check consumes
    assert dedup["stats"]["per_language"]["hi"] == {"in": 7, "kept": 3, "removed": 4}
    assert dedup["stats"]["per_language"]["en"] == {"in": 2, "kept": 2, "removed": 0}

    # --check passes on the fresh build
    check = _run_builder(["--out", str(out), "--check"])
    assert check.returncode == 0, check.stdout + check.stderr
    # tamper a shard -> --check fails
    shard = sorted((out / "shards/train").glob("*.txt") or (out / "shards/heldout").glob("*.txt"))[0]
    data = shard.read_bytes()
    shard.write_bytes(data[:-1] + (b"~" if data[-1:] != b"~" else b"~"))
    check2 = _run_builder(["--out", str(out), "--check"])
    assert check2.returncode == 1

    # a second build at the same code state has the same content identity
    out2 = tmp_path / "out2"
    proc2 = _run_builder(
        ["--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]), "--out", str(out2), "--no-record"]
    )
    assert proc2.returncode == 0, proc2.stderr
    m2 = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["content_sha256"] == m2["content_sha256"]
    for side in ("train", "held_out"):
        a = {s["name"]: s["sha256"] for s in manifest["sides"][side]["shards"]}
        b = {s["name"]: s["sha256"] for s in m2["sides"][side]["shards"]}
        assert a == b  # byte-identical shards


def test_builder_accepts_windows_newlines_for_frozen_text(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path)
    for path in (fake["corpus_dir"] / "sources").glob("*.txt"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    proc = _run_builder(
        ["--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]), "--out", str(tmp_path / "out"), "--no-record"]
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_builder_refuses_drifted_corpus_text(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path)
    out = tmp_path / "out"
    # drift: change the on-disk text after the pin was set
    drifted = fake["corpus_dir"] / "sources/en-fake-book.txt"
    drifted.write_text(drifted.read_text(encoding="utf-8") + " tampered\n", encoding="utf-8", newline="\n")
    proc = _run_builder(
        ["--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]), "--out", str(out), "--no-record"]
    )
    assert proc.returncode == 2
    assert "pinned" in proc.stderr


def test_builder_refuses_tampered_freeze_identity(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path)
    freeze = json.loads(fake["freeze"].read_text(encoding="utf-8"))
    freeze["manifest"]["sha256"] = "f" * 64
    fake["freeze"].write_text(json.dumps(freeze), encoding="utf-8")
    proc = _run_builder(
        ["--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]), "--out", str(tmp_path / "out"), "--no-record"]
    )
    assert proc.returncode == 2
    assert "freeze identity" in proc.stderr
