"""EXP-A sweep: grid shape, the losslessness gate, and an end-to-end smoke run.

The e2e test builds a tiny *fake* frozen corpus (unique lines per language,
real hash pins, a real FREEZE.json — the same pattern the builder e2e test
uses), builds a Frontier dataset from it with ``build_frontier_dataset``, and
then runs the real ``scripts/run_tokenizer_sweep.py`` end to end on top. No
network, no real corpus text.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.corpus import build_frontier_dataset
from frontier_ai.corpus.pipeline import PipelineDocument
from frontier_ai.tokenization.base import SubwordTokenizer, TokenizerError
from frontier_ai.tokenization.sweep import (
    HEADLINE_METRIC,
    PRETOKEN_GPT2_STYLE,
    PRETOKEN_MARK_AWARE,
    SweepConfig,
    docs_to_examples,
    gate_losslessness,
    measure_docs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

VOCAB_SIZES = (2048, 4096, 8192, 16384, 32768)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def test_grid_is_the_approved_15_configurations():
    configs = SweepConfig.grid(VOCAB_SIZES)
    assert len(configs) == 15
    names = {c.name for c in configs}
    for vocab in VOCAB_SIZES:
        assert f"py-{PRETOKEN_MARK_AWARE}-{vocab}" in names
        assert f"py-{PRETOKEN_GPT2_STYLE}-{vocab}" in names
        assert f"hf-byte_level-{vocab}" in names
    # configuration names must be sweep-safe (they become directory names)
    from frontier_ai.experiments.sweep import CONFIGURATION_NAME_RE

    assert all(CONFIGURATION_NAME_RE.match(c.name) for c in configs)


# ---------------------------------------------------------------------------
# gate + measure
# ---------------------------------------------------------------------------
class _ByteStub(SubwordTokenizer):
    """One token per UTF-8 byte: lossless by construction, easy to reason about."""

    name = "stub_bytes"

    def train(self, corpus_path, vocab_size, **kwargs) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(ids).decode("utf-8")

    @property
    def vocab_size(self) -> int:
        return 256

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out

    @classmethod
    def load(cls, artifact_dir) -> SubwordTokenizer:
        return cls()

    def id_to_token(self, index: int) -> str | None:
        return bytes([index]).decode("utf-8", errors="replace")

    def impl_version(self) -> str:
        return "stub"


def _doc(doc_id: str, language: str, text: str) -> PipelineDocument:
    return PipelineDocument(doc_id=doc_id, source_id="s", language=language, text=text)


def test_gate_detects_round_trip_failure():
    docs = [_doc("a-000001", "hi", "नमस्ते"), _doc("a-000002", "en", "hello")]
    assert gate_losslessness(_ByteStub(), docs)["lossless"] is True

    class _BrokenStub(_ByteStub):
        def decode(self, ids) -> str:
            return "X" * len(ids)  # never the inverse of encode

    result = gate_losslessness(_BrokenStub(), docs)
    assert result["lossless"] is False
    assert result["failures"] == 2
    assert result["sample"][0]["doc_id"] == "a-000001"


def test_measure_docs_reports_per_language_and_overall():
    docs = [
        _doc("a-000001", "hi", "नमस्ते दुनिया"),
        _doc("a-000002", "hi", "दूसरा वाक्य है"),
        _doc("b-000001", "en", "hello world"),
    ]
    block = measure_docs(_ByteStub(), docs)
    assert set(block["per_language"]) == {"en", "hi"}
    assert block["per_language"]["hi"]["examples"] == 2
    assert block["overall"]["examples"] == 3
    # one token per byte: chars_per_token must equal chars/bytes overall
    overall = block["overall"]
    assert abs(overall["chars_per_token"] - overall["chars"] / overall["utf8_bytes"]) < 1e-5
    examples = docs_to_examples(docs)
    assert all(ex.category == "document" for ex in examples)


def test_run_one_config_rejects_hf_with_non_bytelevel_pretoken(tmp_path):
    from frontier_ai.tokenization.sweep import IMPL_BPE_HF, run_one_config

    with pytest.raises(TokenizerError, match="only supports the built-in"):
        run_one_config(
            SweepConfig(IMPL_BPE_HF, PRETOKEN_MARK_AWARE, 300),
            train_file=tmp_path / "nope.txt",
            train_documents=[],
            heldout_documents=[],
            out_dir=tmp_path,
        )


def test_headline_metric_path_is_numeric_in_real_results(tmp_path):
    from frontier_ai.tokenization.sweep import run_one_config

    train_file = tmp_path / "train.txt"
    train_file.write_text(("मेरी जान तुम हो। " * 50) + "hello world " * 40, encoding="utf-8")
    docs = [
        _doc("t-000001", "hi", "मेरी जान तुम हो।"),
        _doc("t-000002", "en", "hello world"),
    ]
    results = run_one_config(
        SweepConfig("bpe_python", PRETOKEN_MARK_AWARE, 300),
        train_file=train_file,
        train_documents=docs,
        heldout_documents=docs,
        out_dir=tmp_path,
    )
    node = results
    for part in HEADLINE_METRIC.split("."):
        assert part in node, f"headline metric path {HEADLINE_METRIC} broken at {part}"
        node = node[part]
    assert isinstance(node, (int, float)) and node > 0
    assert results["gate"]["lossless"] is True


# ---------------------------------------------------------------------------
# e2e smoke: fake frozen corpus -> frontier dataset -> the real sweep script
# ---------------------------------------------------------------------------
def _make_fake_frozen_corpus(root: Path, n_hi: int = 60, n_en: int = 20) -> dict:
    hi_words = ["मेरी", "जान", "तुम", "हो", "यह", "किताब", "पढ़ो", "हम", "लिखते", "हैं", "शहर", "मैं"]
    hi_lines = [
        " ".join(hi_words[(i * (j + 1) + j) % len(hi_words)] for j in range(7)) + f" पंक्ति {i}।"
        for i in range(n_hi)
    ]
    en_words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "near", "river"]
    en_lines = [
        " ".join(en_words[(i * (j + 1) + j) % len(en_words)] for j in range(8)) + f" sentence {i}."
        for i in range(n_en)
    ]
    files = {
        "hi-fake-book": ("\n".join(hi_lines) + "\n", "hi", "Devanagari"),
        "en-fake-book": ("\n".join(en_lines) + "\n", "en", "Latin"),
    }
    sources_dir = root / "corpus" / "sources"
    sources_dir.mkdir(parents=True)
    sources = []
    for source_id, (text, language, script) in files.items():
        path = sources_dir / f"{source_id}.txt"
        path.write_text(text, encoding="utf-8", newline="\n")
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(
            {
                "id": source_id, "title": f"Fake {source_id}", "language": language,
                "script": script, "source_url": f"https://example.org/{source_id}",
                "license_id": "CC-BY-SA-4.0", "attribution": "test fixture",
                "max_chars": 100000, "kind": "mediawiki-parse", "sha256": pin, "verified": True,
                "retrieved_at": "2026-09-26T00:00:00Z", "notes": "test",
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


def _build_fake_frontier(fake: dict, out: Path) -> Path:
    build = build_frontier_dataset(
        manifest_path=fake["manifest"],
        freeze_path=fake["freeze"],
        corpus_dir=fake["corpus_dir"],
        out_dir=out,
        seed=1337,
        held_out_fraction=0.1,
        max_shard_chars=4000,
    )
    assert build.manifest["sides"]["train"]["documents"] > 0
    assert build.manifest["sides"]["held_out"]["documents"] > 0, "the split must leave a held-out side"
    return out


def _run_sweep_script(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/run_tokenizer_sweep.py"), *args],
        capture_output=True, text=True, timeout=900,
    )


def test_sweep_e2e_smoke_on_fake_corpus(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path)
    frontier = _build_fake_frontier(fake, tmp_path / "frontier")

    out = tmp_path / "sweep_out"
    proc = _run_sweep_script(
        ["--exp-id", "EXP-999", "--frontier-dir", str(frontier),
         "--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]),
         "--vocab-sizes", "512,768", "--max-train-chars", "4000",
         "--out", str(out), "--no-record"]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    sweep = json.loads((out / "sweep.json").read_text(encoding="utf-8"))
    assert sweep["sweep"]["status"] == "success"
    assert sweep["sweep"]["runs_successful"] == 6  # 2 vocabs x (py-mark, py-gpt2, hf)
    for run in sweep["runs"]:
        assert run["status"] == "success", run
        assert isinstance(run["metric_value"], (int, float))

    # every run record: gate passed on both sides, per-language metrics present
    for run in sweep["runs"]:
        record = json.loads((out / run["record"]).read_text(encoding="utf-8"))
        results = record["results"]
        assert results["gate"]["lossless"] is True
        assert results["gate"]["train"]["failures"] == 0
        assert results["gate"]["heldout"]["failures"] == 0
        assert results["heldout"]["overall"]["round_trip_failures"] == 0

    report = (out / "report.txt").read_text(encoding="utf-8")
    assert "EXP-A tokenizer sweep" in report
    # both languages must appear in the per-language table (each has held-out docs)
    assert "hi" in report and "en" in report
    # the shared input audit trail was written
    assert (out / "_shared" / "train.txt").is_file()
    assert (out / "_shared" / "heldout_docs.jsonl").is_file()


def test_sweep_e2e_full_records_mode_smoke(tmp_path):
    """Without --no-record the outer script record is written too (D-032)."""
    fake = _make_fake_frozen_corpus(tmp_path, n_hi=30, n_en=10)
    frontier = _build_fake_frontier(fake, tmp_path / "frontier")
    out = tmp_path / "sweep_out"
    proc = _run_sweep_script(
        ["--exp-id", "EXP-999", "--frontier-dir", str(frontier),
         "--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]),
         "--vocab-sizes", "512", "--max-train-chars", "3000",
         "--out", str(out)]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "experiment.json").is_file()
    outer = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    # the outer record embeds the sweep record as its results (D-032)
    assert outer["results"]["record_type"] == "frontier-ai.experiment-sweep"
    assert outer["results"]["sweep"]["runs_successful"] == 3
    assert (out / "sweep.json").is_file()


def test_sweep_refuses_tampered_shard(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path)
    frontier = _build_fake_frontier(fake, tmp_path / "frontier")
    shard = sorted((frontier / "shards" / "train").glob("*.txt"))[0]
    data = shard.read_bytes()
    shard.write_bytes(data[:-1] + (b"~" if data[-1:] != b"~" else b"~"))
    proc = _run_sweep_script(
        ["--exp-id", "EXP-999", "--frontier-dir", str(frontier),
         "--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]), "--vocab-sizes", "512",
         "--out", str(tmp_path / "sweep_out"), "--no-record"]
    )
    assert proc.returncode == 2
    assert "INPUT GATE FAILED" in proc.stderr


def test_sweep_refuses_manifest_count_mismatch(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path)
    frontier = _build_fake_frontier(fake, tmp_path / "frontier")
    manifest_path = frontier / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sides"]["train"]["per_language"]["hi"]["documents"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    proc = _run_sweep_script(
        ["--exp-id", "EXP-999", "--frontier-dir", str(frontier),
         "--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]), "--vocab-sizes", "512",
         "--out", str(tmp_path / "sweep_out"), "--no-record"]
    )
    assert proc.returncode == 2
    assert "re-deriv" in proc.stderr or "manifest" in proc.stderr


def test_sweep_refuses_missing_frontier_dir(tmp_path):
    proc = _run_sweep_script(
        ["--exp-id", "EXP-999", "--frontier-dir", str(tmp_path / "nope"),
         "--vocab-sizes", "512", "--out", str(tmp_path / "sweep_out"), "--no-record"]
    )
    assert proc.returncode in (2, 127, 1)
    assert proc.returncode != 0
    assert "INPUT GATE" in proc.stderr or "no dataset manifest" in (proc.stderr + proc.stdout)
