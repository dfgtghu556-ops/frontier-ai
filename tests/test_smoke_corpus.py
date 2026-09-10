"""Real, clearly licensed smoke-test corpora.

The corpus itself cannot be fetched from a sandbox that only reaches PyPI, so these
tests are split deliberately:

* **Always run** — the manifest, the licence allow-list, the documentation, the text
  cleaners, hash verification, the provenance record, the CLI, and the full
  prepare-data → tokenize path driven by ``tests/fixtures/pipeline-sample.txt`` (text
  authored in this repository, so it is ours to use and needs no licence).
* **Run only if a corpus is present** — the checks that need the real files. They skip
  with a clear message, because a skipped test is honest and a fabricated corpus is not.

See ``corpora/smoke/README.md`` for sources, licences and acquisition.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.data.corpora import (
    ALLOWED_LICENSES,
    DEFAULT_MAX_CHARS,
    SCHEMA_VERSION,
    CorpusSource,
    SourceManifest,
    clean_gutenberg_text,
    clean_wikitext,
    license_marker_found,
    load_sources,
    prepare_source_text,
    sha256_file,
    sha256_text,
    trim_text,
    validate_source,
    verify_file,
    write_provenance,
)
from frontier_ai.data.dataset import DataMeta, TokenDataset
from frontier_ai.data.synthetic import generate_corpus
from frontier_ai.data.tokenizer import WordTokenizer

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "corpora" / "smoke" / "sources.json"
CORPORA_README = REPO / "corpora" / "smoke" / "README.md"
SAMPLE = Path(__file__).resolve().parent / "fixtures" / "pipeline-sample.txt"
RAW_DIR = REPO / "data" / "raw" / "smoke"

# Pinned 2026-09-10 on this branch: guards the synthetic generator against accidental
# change (requirement: existing synthetic corpus behaviour must stay unchanged).
SYNTHETIC_SHA256 = "c5e7caf2e89f6a1ae637acd2a57a1ec05eb918c0221333ba03f33fb5c990f3b4"


def run_cli(*args: str, expect: int = 0) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert proc.returncode == expect, f"{args} exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc


def prepare(text_path: Path, out_prefix: Path, provenance: Path | None = None,
            level: str = "char") -> DataMeta:
    cmd = [str(REPO / "scripts" / "prepare_data.py"), "--source", str(text_path),
           "--out", str(out_prefix), "--level", level]
    if provenance is not None:
        cmd += ["--provenance", str(provenance)]
    run_cli(*cmd)
    return DataMeta.load(out_prefix.with_suffix(".meta.json"))


# ------------------------------------------------------------------ manifest --


def test_manifest_loads_and_every_entry_validates():
    manifest = SourceManifest.load(MANIFEST)
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.sources, "manifest has no sources"
    ids = [source.id for source in manifest.sources]
    assert len(ids) == len(set(ids)), f"duplicate source ids: {ids}"
    for source in manifest.sources:
        assert validate_source(source) == [], f"{source.id}: {validate_source(source)}"


def test_manifest_licences_are_allow_listed_and_attributed():
    for source in load_sources(MANIFEST):
        assert source.license_id in ALLOWED_LICENSES, (
            f"{source.id}: licence {source.license_id!r} is outside the allow-list"
        )
        assert source.license_url.startswith("https://")
        assert len(source.attribution.strip()) > 20, f"{source.id}: attribution is too terse to use"
        assert "wikimedia" not in source.source_url.lower() or "wikisource" in source.source_url.lower()


def test_manifest_entries_stay_smoke_sized_and_https():
    for source in load_sources(MANIFEST):
        assert source.source_url.startswith("https://"), f"{source.id}: only https sources"
        assert 0 < source.max_chars <= DEFAULT_MAX_CHARS, f"{source.id}: not a smoke-sized corpus"
        assert source.kind in {"gutenberg", "wikitext", "plain"}


def test_no_hash_is_ever_asserted_without_verification():
    """A pinned hash implies verified=true; we never store a hash we did not measure."""
    for source in load_sources(MANIFEST):
        if source.sha256 is not None:
            assert source.verified, f"{source.id}: has a sha256 but is not marked verified"
            assert len(source.sha256) == 64
        else:
            assert source.verified is False
            assert source.pinned is False


def test_readme_documents_every_source_and_the_licence_rules():
    text = CORPORA_README.read_text(encoding="utf-8")
    for source in load_sources(MANIFEST):
        assert source.id in text, f"{source.id} is not documented in corpora/smoke/README.md"
        assert source.license_id in text, f"{source.license_id} is not documented"
    for marker in ("fetch_smoke_corpus.py", "--provenance", "--check", "CC BY-SA 4.0", "PD-US"):
        assert marker in text, f"README does not mention {marker!r}"


def test_manifest_rejects_a_questionable_entry(tmp_path):
    bad = SourceManifest(sources=[
        CorpusSource(
            id="mystery",
            title="Some text found somewhere",
            language="xx",
            script="Latin",
            source_url="http://example.com/text.txt",  # http, unclear provenance
            license_id="unclear",
            license_url="http://example.com/licence",
            attribution="?",
        )
    ])
    problems = validate_source(bad.sources[0])
    assert any("licence" in problem for problem in problems)
    assert any("https" in problem for problem in problems)
    assert any("attribution" in problem for problem in problems)


def test_manifest_round_trips(tmp_path):
    manifest = SourceManifest.load(MANIFEST)
    path = tmp_path / "sources.json"
    manifest.save(path)
    assert SourceManifest.load(path).to_dict() == manifest.to_dict()
    other = SourceManifest.load(path)
    other.sources[0] = CorpusSource(**{**other.sources[0].to_dict(), "sha256": "not-a-hash"})
    assert any("sha256" in problem for problem in validate_source(other.sources[0]))


# ------------------------------------------------------------------ cleaning --


def test_clean_gutenberg_text_strips_the_licence_wrapper():
    raw = (
        "Some header prose about Project Gutenberg.\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK ALICE ***\n"
        "Alice was beginning to get very tired.\n"
        "So she sat down.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK ALICE ***\n"
        "End of the Project Gutenberg licence."
    )
    assert clean_gutenberg_text(raw) == "Alice was beginning to get very tired.\nSo she sat down."


def test_clean_gutenberg_text_without_markers_is_left_alone():
    raw = "plain text with no markers\r\nsecond line"
    assert clean_gutenberg_text(raw) == "plain text with no markers\nsecond line"


def test_clean_wikitext_removes_markup():
    raw = (
        "<!-- a comment -->\n"
        "{{header|title=Godaan}}\n"
        "== Chapter One ==\n"
        "'''Godaan''' is a [[novel|novel]] by [[Munshi Premchand]].\n"
        "It was published in 1936.<ref>Some reference</ref>\n"
        "<br />\n"
        "Line with <ref name=\"x\" /> and '''bold'''."
    )
    cleaned = clean_wikitext(raw)
    assert "<!--" not in cleaned and "{{" not in cleaned and "==" not in cleaned
    assert "[[novel|novel]]" not in cleaned and "novel" in cleaned
    assert "<ref" not in cleaned and "<br" not in cleaned
    assert "'''" not in cleaned
    assert "Godaan" in cleaned and "Munshi Premchand" in cleaned


def test_trim_text_is_line_aligned_and_deterministic():
    text = "\n".join(f"line {i}" for i in range(1000))
    trimmed = trim_text(text, 100)
    assert trimmed == trim_text(text, 100)
    assert trimmed.endswith("line") is False  # no half-cut lines
    assert len(trimmed) <= 100
    assert trimmed.startswith("line 0")
    assert trim_text("short", 1000) == "short"


def test_prepare_source_text_cleans_and_caps():
    source = CorpusSource(
        id="t", title="t", language="en", script="Latin",
        source_url="https://example.org/t.txt", license_id="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="t", kind="wikitext", max_chars=50,
    )
    raw = "{{template}}\n" + ("word " * 200)
    out = prepare_source_text(raw, source)
    assert "{{" not in out
    assert len(out) <= 50


# --------------------------------------------------------------- provenance ---


def test_license_marker_detection_is_conservative():
    gutenberg = CorpusSource(
        id="g", title="g", language="en", script="Latin", source_url="https://example.org/g",
        license_id="PD-US", license_url="https://www.gutenberg.org/policy/license.html",
        attribution="g", kind="gutenberg",
    )
    assert license_marker_found("*** START OF THE PROJECT GUTENBERG EBOOK X ***", gutenberg)
    assert not license_marker_found("just some text", gutenberg)

    cc = CorpusSource(
        id="c", title="c", language="hi", script="Devanagari", source_url="https://example.org/c",
        license_id="CC-BY-SA-4.0", license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="c", kind="wikitext",
    )
    assert license_marker_found("{{CC BY-SA 4.0}} some text", cc)
    assert not license_marker_found("{{PD-old}} some text", cc)


def test_verify_file_rejects_unpinned_and_mismatched(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("hello\n", encoding="utf-8")
    digest = sha256_file(path)
    assert verify_file(path, digest) is True
    assert verify_file(path, "0" * 64) is False
    assert verify_file(path, None) is False, "an unmeasured hash must never verify"


def test_write_provenance_records_source_and_bytes(tmp_path):
    source = CorpusSource(
        id="en-alice-pd", title="Alice", language="en", script="Latin",
        source_url="https://www.gutenberg.org/cache/epub/11/pg11.txt",
        license_id="PD-US", license_url="https://www.gutenberg.org/policy/license.html",
        attribution="Lewis Carroll, via Project Gutenberg", kind="gutenberg",
    )
    content = "Alice was beginning to get very tired of sitting by her sister.\n"
    path = tmp_path / "out" / "en-alice-pd.provenance.json"
    record = write_provenance(path, source, content, retrieved_at="2026-09-10T00:00:00Z")

    assert record["sha256"] == sha256_text(content)
    assert record["n_bytes"] == len(content.encode("utf-8"))
    assert record["n_chars"] == len(content)
    for key in ("id", "title", "language", "script", "source_url", "license_id",
                "license_url", "attribution", "retrieved_at", "verified"):
        assert key in record, f"provenance is missing {key}"
    assert record["verified"] is False, "verification only comes from a checked fetch"

    # deterministic: same input, same timestamp -> identical record
    again = write_provenance(tmp_path / "out" / "again.json", source, content,
                             retrieved_at="2026-09-10T00:00:00Z")
    assert again == record
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_provenance_marks_a_matching_hash_as_verified(tmp_path):
    source = CorpusSource(
        id="x", title="x", language="en", script="Latin", source_url="https://example.org/x",
        license_id="CC0-1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="x", kind="plain", sha256="0" * 64, verified=True,
    )
    record = write_provenance(tmp_path / "p.json", source, "text\n",
                              retrieved_at="2026-09-10T00:00:00Z")
    assert record["verified"] is False, "a mismatching pinned hash must not be reported as verified"


# --------------------------------------------------------------------- CLI ---


def test_fetch_script_lists_the_manifest():
    proc = run_cli(str(REPO / "scripts" / "fetch_smoke_corpus.py"), "--list")
    for source in load_sources(MANIFEST):
        assert source.id in proc.stdout
        assert source.license_id in proc.stdout


def test_fetch_script_check_reports_what_is_missing(tmp_path):
    proc = run_cli(str(REPO / "scripts" / "fetch_smoke_corpus.py"), "--check",
                   "--out-dir", str(tmp_path / "empty"))
    assert "not fetched" in proc.stdout
    assert "MISMATCH" not in proc.stdout


def test_fetch_script_does_nothing_without_an_action():
    proc = run_cli(str(REPO / "scripts" / "fetch_smoke_corpus.py"))
    assert "nothing to do" in proc.stdout


# ------------------------------------------------------- pipeline end-to-end --


def test_prepare_data_with_provenance_records_the_licence(tmp_path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    source = CorpusSource(
        id="pipeline-sample", title="Pipeline sample", language="en", script="Latin",
        source_url="https://example.invalid/pipeline-sample.txt", license_id="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="authored in this repository", kind="plain",
    )
    content = text_path.read_text(encoding="utf-8")
    prov_path = tmp_path / "sample.provenance.json"
    provenance = write_provenance(prov_path, source, content, retrieved_at="2026-09-10T00:00:00Z")

    meta = prepare(text_path, tmp_path / "out" / "sample", provenance=prov_path)

    # expected files
    assert (tmp_path / "out" / "sample.bin").exists()
    assert (tmp_path / "out" / "sample.tokenizer.json").exists()
    assert (tmp_path / "out" / "sample.meta.json").exists()
    # metadata
    assert meta.source_provenance is not None
    assert meta.source_provenance["license_id"] == "CC0-1.0"
    assert meta.source_provenance["id"] == "pipeline-sample"
    assert meta.source_provenance["sha256"] == provenance["sha256"]
    assert meta.has_text_lengths is True
    assert meta.n_bytes == len(content.encode("utf-8"))
    assert meta.n_chars == len(content)

    # deterministic: preparing again gives byte-identical artefacts
    meta2 = prepare(text_path, tmp_path / "out" / "sample2", provenance=prov_path)
    first = sha256_file(tmp_path / "out" / "sample.bin")
    second = sha256_file(tmp_path / "out" / "sample2.bin")
    assert first == second
    assert meta.source_provenance == meta2.source_provenance
    assert meta.n_bytes_train == meta2.n_bytes_train

    # and the dataset reads back with the corpus-level ratios
    ds = TokenDataset(tmp_path / "out" / "sample.bin", meta)
    assert ds.tokens_per_byte("val") == pytest.approx(meta.n_val / meta.n_bytes_val)


def test_prepare_data_without_provenance_records_none(tmp_path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    meta = prepare(text_path, tmp_path / "sample")
    assert meta.source_provenance is None
    assert meta.has_text_lengths is True  # byte/char counting is independent of provenance


def test_synthetic_corpus_behaviour_is_unchanged(tmp_path):
    text = generate_corpus(target_chars=20_000, seed=1337)
    assert sha256_text(text) == SYNTHETIC_SHA256

    text_path = tmp_path / "synthetic.txt"
    text_path.write_text(text, encoding="utf-8")
    meta = prepare(text_path, tmp_path / "synthetic")
    assert meta.source_provenance is None
    assert meta.n_bytes == meta.n_tokens, "the synthetic corpus is ASCII"
    assert meta.n_chars == meta.n_tokens


def test_sample_corpus_is_multi_byte_and_small():
    text = SAMPLE.read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) > len(text), "the sample must exercise multi-byte UTF-8"
    assert len(text) < 20_000, "the sample is a fixture, not a corpus"


def test_tokenizer_smoke_on_the_sample_corpus(tmp_path):
    """End-to-end: prepare the sample text, then train two tokenizers on it."""
    from frontier_ai.tokenization.artifact import load_artifact

    text_path = tmp_path / "corpus" / "train.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    prepare(text_path, tmp_path / "prepared" / "sample", level="char")

    artifacts = {}
    for impl, args in (("char", []), ("bpe_python", ["--vocab-size", "300"])):
        out = tmp_path / f"tok-{impl}"
        run_cli(str(REPO / "scripts" / "tokenizer_train.py"), "--corpus", str(text_path),
                "--impl", impl, *args, "--out", str(out), "--exp-id", "EXP-000")
        tokenizer, manifest = load_artifact(out)
        artifacts[impl] = (tokenizer, manifest)
        assert (out / "manifest.json").exists()

    text = text_path.read_text(encoding="utf-8")
    char_ids = artifacts["char"][0].encode(text)
    bpe_ids = artifacts["bpe_python"][0].encode(text)
    assert len(char_ids) == len(text)  # char tokenizer: one token per character
    assert artifacts["bpe_python"][1].vocab_size <= 300
    # no compression claim here: on a 2.5 KB sample with 300 merges and three scripts,
    # BPE does not beat characters (measured: 2,663 vs 2,525 tokens). What must hold is
    # that both tokenizers cover the text losslessly, so the pipeline is exercised.
    assert artifacts["char"][0].decode(char_ids) == text
    assert artifacts["bpe_python"][0].decode(bpe_ids) == text
    assert len(bpe_ids) != len(char_ids), "the two tokenizations should differ"


@pytest.mark.skipif(not RAW_DIR.exists() or not any(RAW_DIR.glob("*.txt")),
                    reason="no real corpus fetched yet (run scripts/fetch_smoke_corpus.py --fetch)")
def test_real_corpus_prepares_deterministically_with_provenance(tmp_path):
    """Skipped until a licensed corpus has actually been fetched."""
    for text_path in sorted(RAW_DIR.glob("*.txt")):
        prov_path = text_path.with_suffix(".provenance.json")
        assert prov_path.exists(), f"{text_path.name} has no provenance file"
        provenance = json.loads(prov_path.read_text(encoding="utf-8"))
        assert provenance["license_id"] in ALLOWED_LICENSES
        assert provenance["attribution"]
        if provenance["sha256"]:
            assert verify_file(text_path, provenance["sha256"]), f"{text_path.name} hash mismatch"

        meta = prepare(text_path, tmp_path / f"{text_path.stem}-a", provenance=prov_path)
        meta2 = prepare(text_path, tmp_path / f"{text_path.stem}-b", provenance=prov_path)
        assert sha256_file(tmp_path / f"{text_path.stem}-a.bin") == sha256_file(
            tmp_path / f"{text_path.stem}-b.bin"
        )
        assert meta.source_provenance == meta2.source_provenance
        assert meta.source_provenance["id"] == provenance["id"]


def test_word_level_preparation_of_the_sample(tmp_path):
    text = SAMPLE.read_text(encoding="utf-8")
    text_path = tmp_path / "sample.txt"
    text_path.write_text(text, encoding="utf-8")
    meta = prepare(text_path, tmp_path / "sample-word", level="word")
    tok = WordTokenizer.fit(text)
    assert meta.vocab_size == tok.vocab_size
    assert meta.n_bytes == len(text.encode("utf-8"))
    assert meta.n_tokens < meta.n_chars, "word tokens are longer than characters"
