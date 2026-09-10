"""Tests for the Project 002 tokenizer research subsystem.

Covers: the pluggable registry, corpus determinism, both subword implementations,
the Project 001 adapters, evaluator arithmetic, comparison logic, invalid input
handling, and a small end-to-end CLI workflow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.tokenization import available, create, is_available
from frontier_ai.tokenization.adapters import CharTokenizerAdapter, WordTokenizerAdapter
from frontier_ai.tokenization.artifact import load_artifact, load_manifest, save_artifact
from frontier_ai.tokenization.base import TokenizerError, iter_segments
from frontier_ai.tokenization.bpe_python import PythonBPE, pretokenize
from frontier_ai.tokenization.compare import build_comparison, render_comparison
from frontier_ai.tokenization.corpus import (
    CATEGORY_BANK,
    EXAMPLE_BANK,
    LANGUAGES,
    WORD_BANK,
    CorpusManifest,
    Example,
    build_examples,
    examples_by_language,
    generate_train_text,
    load_corpus,
    sha256_text,
    write_corpus,
)
from frontier_ai.tokenization.evaluate import count_words, evaluate_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# A probe that stresses scripts, digits, combining marks, emoji and punctuation.
MULTILINGUAL_PROBE = [
    "मैं स्कूल जा रहा हूँ।",
    "আজ আবহাওয়া খুব ভালো।",
    "நான் பள்ளிக்கு செல்கிறேன்.",
    "నేను బడికి వెళ్తున్నాను.",
    "ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
    "ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്.",
    "ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ।",
    "ଆଜି ପାଗ ବହୁତ ଭଲ।",
    "আজি বতৰ বৰ ভাল।",
    "آج موسم بہت اچھا ہے۔",
    "હું શાળાએ જાઉં છું.",
    "मी शाळेत जात आहे.",
    "Main aaj office nahi ja raha.",
    "https://example.com/docs?token=42",
    "x^2 + y^2 = z^2",
    "2026-09-10 and 3.14159 and ₹1,299.50",
    "👨‍👩‍👧 family emoji and नमस्ते 🙏",
    "क्षमा and हिन्दी with combining marks",
    "Hello, world! (fine, thanks) — really?",
]

SPECIALS = ["<pad>", "<bos>", "<eos>"]


def write_tiny_corpus(path: Path, text: str | None = None) -> Path:
    """A small deterministic training file (multilingual + tech + numbers)."""
    if text is None:
        text = "\n".join(
            [
                " ".join(WORD_BANK[lang][:6]) + " " + " ".join(WORD_BANK[lang][:3]) + "."
                for lang in WORD_BANK
            ]
            * 12
            + ["def train(model): optimizer.step()", "2026-09-10", "3.14159", "https://example.com"]
            + EXAMPLE_BANK["en"]
        )
    path.write_text(text, encoding="utf-8")
    return path


def train_python_bpe(tmp_path: Path, vocab_size: int = 320, **kwargs) -> PythonBPE:
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    tok = PythonBPE()
    tok.train(corpus, vocab_size=vocab_size, special_tokens=SPECIALS, **kwargs)
    return tok


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
def test_corpus_generation_is_deterministic():
    a = generate_train_text(seed=7, sentences_per_language=3, english_chars=500)
    b = generate_train_text(seed=7, sentences_per_language=3, english_chars=500)
    assert a == b
    assert a != generate_train_text(seed=8, sentences_per_language=3, english_chars=500)


def test_write_and_load_corpus_round_trip(tmp_path):
    manifest = write_corpus(tmp_path / "corpus", seed=3, sentences_per_language=4, english_chars=800)
    examples, loaded = load_corpus(tmp_path / "corpus")
    assert isinstance(loaded, CorpusManifest)
    assert loaded.seed == 3 and len(examples) == manifest.n_examples == loaded.n_examples
    assert (tmp_path / "corpus" / "train.txt").exists()
    assert (tmp_path / "corpus" / "eval.jsonl").exists()
    assert sha256_text((tmp_path / "corpus" / "train.txt").read_text(encoding="utf-8")) == (
        manifest.train_sha256
    )


def test_corpus_covers_every_language_and_category():
    examples = build_examples()
    by_lang = examples_by_language(examples)
    for lang in LANGUAGES:
        assert lang in by_lang, f"no evaluation examples for language '{lang}'"
        assert len(by_lang[lang]) >= 5
        assert all(ex.text.strip() for ex in by_lang[lang])
    for category in CATEGORY_BANK:
        assert any(ex.category == category for ex in examples)
    ids = [ex.id for ex in examples]
    assert len(ids) == len(set(ids)), "evaluation example ids must be unique"


def test_load_corpus_detects_modified_fixture(tmp_path):
    write_corpus(tmp_path / "corpus", seed=1, sentences_per_language=2, english_chars=200)
    (tmp_path / "corpus" / "eval.jsonl").write_text('{"id": "x", "lang": "en", "category": "sentence", "text": "tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_corpus(tmp_path / "corpus")


def test_load_corpus_requires_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_registry_contains_baselines():
    impls = available()
    assert {"char", "word", "bpe_python"} <= set(impls)
    assert is_available("bpe_python")


def test_create_unknown_implementation_raises():
    with pytest.raises(KeyError, match="unknown tokenizer implementation"):
        create("definitely_not_a_tokenizer")


# ---------------------------------------------------------------------------
# our own BPE
# ---------------------------------------------------------------------------
def test_python_bpe_round_trips_multilingual_and_unicode(tmp_path):
    tok = train_python_bpe(tmp_path, vocab_size=400)
    for text in MULTILINGUAL_PROBE:
        assert tok.decode(tok.encode(text)) == text, f"round trip failed for {text!r}"
    assert all(0 <= i < tok.vocab_size for i in tok.encode("".join(MULTILINGUAL_PROBE)))


def test_python_bpe_vocab_size_and_ranges(tmp_path):
    tok = train_python_bpe(tmp_path, vocab_size=320)
    assert tok.vocab_size == 320
    assert len(tok.merges) == 320 - 256 - len(SPECIALS)
    with pytest.raises(TokenizerError, match="out of range"):
        tok.decode([tok.vocab_size + 5])


def test_python_bpe_training_is_deterministic(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    a, b = PythonBPE(), PythonBPE()
    a.train(corpus, vocab_size=300, special_tokens=SPECIALS)
    b.train(corpus, vocab_size=300, special_tokens=SPECIALS)
    assert a.merges == b.merges
    assert a.encode("मैं स्कूल जाता हूँ") == b.encode("मैं स्कूल जाता हूँ")


def test_python_bpe_save_load_preserves_encoding(tmp_path):
    tok = train_python_bpe(tmp_path, vocab_size=340)
    directory = tok.save(tmp_path / "artifact")
    reloaded = PythonBPE.load(directory)
    assert reloaded.vocab_size == tok.vocab_size
    for text in MULTILINGUAL_PROBE[:8]:
        assert reloaded.encode(text) == tok.encode(text)
        assert reloaded.decode(tok.encode(text)) == text


def test_python_bpe_special_tokens(tmp_path):
    tok = train_python_bpe(tmp_path, vocab_size=320)
    ids = tok.special_token_ids
    assert set(ids) == set(SPECIALS)
    for token in SPECIALS:
        assert tok.encode(token) == [ids[token]]
        assert ids[token] in tok.encode(f"hello{token}world")
        assert tok.decode([ids[token]]) == token


def test_python_bpe_rejects_impossible_vocab_and_missing_corpus(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    with pytest.raises(TokenizerError, match="too small"):
        PythonBPE().train(corpus, vocab_size=100, special_tokens=SPECIALS)
    with pytest.raises(FileNotFoundError):
        PythonBPE().train(tmp_path / "nope.txt", vocab_size=300)
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(TokenizerError, match="empty"):
        PythonBPE().train(empty, vocab_size=300)


def test_python_bpe_compresses_text_seen_in_training(tmp_path):
    """Repeated frequent text must cost fewer tokens than raw bytes."""
    corpus = tmp_path / "repeated.txt"
    corpus.write_text("hello world " * 200, encoding="utf-8")
    tok = PythonBPE()
    tok.train(corpus, vocab_size=300)
    text = "hello world hello world"
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8")), "learned merges should beat raw bytes"


def test_pretokenize_keeps_indic_words_whole():
    chunks = pretokenize("मैं स्कूल जाता हूँ। 42!")
    assert "मैं" in chunks
    assert "42" in chunks
    assert "!" in chunks
    # combining marks (matras, virama, chandrabindu) stay attached to their base character
    for word in ["क्षमा", "हिन्दी", "আমি", "ਅੱਜ"]:
        assert word in pretokenize(f"{word} test"), f"{word} was split into {pretokenize(word)}"


def test_iter_segments_prefers_longest_special_token():
    segments = iter_segments("a<unk>b<unkx>c", ["<unk>", "<unkx>"])
    assert segments == [("a", False), ("<unk>", True), ("b", False), ("<unkx>", True), ("c", False)]


# ---------------------------------------------------------------------------
# Project 001 adapters
# ---------------------------------------------------------------------------
def test_char_adapter_round_trips_in_vocab_text(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    tok = CharTokenizerAdapter()
    tok.train(corpus, special_tokens=SPECIALS)
    text = "abcdef"
    assert tok.decode(tok.encode(text)) == text
    assert set(tok.special_token_ids) == set(SPECIALS)


def test_char_adapter_maps_unseen_characters_to_unk(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt", text="abc def")
    tok = CharTokenizerAdapter()
    tok.train(corpus)
    unk_id = tok.unk_token_id
    assert unk_id is not None
    ids = tok.encode("abcζ")
    assert ids[-1] == unk_id
    assert tok.decode(ids) != "abcζ"  # unknown characters are destroyed, by design


def test_word_adapter_unk_and_special_tokens(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt", text="alpha beta alpha")
    tok = WordTokenizerAdapter()
    tok.train(corpus, special_tokens=SPECIALS)
    assert tok.unk_token == "<unk>"
    assert tok.unk_token_id == 0
    assert tok.unk_token_id in tok.encode("alpha unseenword")
    assert tok.encode("<pad>") == [tok.special_token_ids["<pad>"]]


def test_adapters_save_and_load(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    for adapter in (CharTokenizerAdapter(), WordTokenizerAdapter()):
        adapter.train(corpus, special_tokens=SPECIALS)
        directory = adapter.save(tmp_path / adapter.name)
        reloaded = type(adapter).load(directory)
        assert reloaded.vocab_size == adapter.vocab_size
        assert reloaded.encode("test 42 <pad>") == adapter.encode("test 42 <pad>")


def test_adapter_requires_training_before_use():
    with pytest.raises(TokenizerError, match="not trained"):
        CharTokenizerAdapter().encode("anything")


# ---------------------------------------------------------------------------
# optional HuggingFace baseline
# ---------------------------------------------------------------------------
def test_huggingface_bpe_round_trips_and_saves(tmp_path):
    pytest.importorskip("tokenizers")
    from frontier_ai.tokenization.bpe_hf import HuggingFaceBPE

    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    tok = HuggingFaceBPE()
    tok.train(corpus, vocab_size=400, special_tokens=SPECIALS)
    for text in MULTILINGUAL_PROBE:
        assert tok.decode(tok.encode(text)) == text, f"HF round trip failed for {text!r}"

    directory = tok.save(tmp_path / "hf")
    reloaded = HuggingFaceBPE.load(directory)
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.encode("मैं स्कूल जाता हूँ <pad>") == tok.encode("मैं स्कूल जाता हूँ <pad>")
    assert reloaded.impl_version().startswith("tokenizers-")


def test_all_implementations_round_trip_text_containing_special_tokens(tmp_path):
    """decode(encode(x)) must equal x even when x contains a special token."""
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    texts = ["hello <pad> world", "<bos>मैं स्कूल<eos>", "a<unk>b 42"]

    tok = PythonBPE()
    tok.train(corpus, vocab_size=320, special_tokens=SPECIALS)
    for text in texts:
        assert tok.decode(tok.encode(text)) == text

    pytest.importorskip("tokenizers")
    from frontier_ai.tokenization.bpe_hf import HuggingFaceBPE

    hf = HuggingFaceBPE()
    hf.train(corpus, vocab_size=400, special_tokens=SPECIALS)
    for text in texts:
        assert hf.decode(hf.encode(text)) == text, f"HF round trip failed for {text!r}"


def test_huggingface_bpe_special_tokens(tmp_path):
    pytest.importorskip("tokenizers")
    from frontier_ai.tokenization.bpe_hf import HuggingFaceBPE

    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    tok = HuggingFaceBPE()
    tok.train(corpus, vocab_size=400, special_tokens=SPECIALS)
    ids = tok.special_token_ids
    for token in SPECIALS:
        assert tok.encode(token) == [ids[token]]
        assert ids[token] in tok.encode(f"a{token}b")


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------
def test_evaluator_arithmetic_on_known_input(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt", text="ab cd x")
    tok = CharTokenizerAdapter()
    tok.train(corpus)
    examples = [
        Example(id="1", lang="en", category="sentence", text="ab cd"),   # 5 chars, 5 tokens, 2 words
        Example(id="2", lang="en", category="sentence", text="x"),       # 1 char, 1 token, 1 word
    ]
    manifest = CorpusManifest(
        corpus_id="unit", corpus_version="test", schema_version="1.0", seed=0,
        n_examples=2, languages=["en"], categories=["sentence"], train_chars=7, train_lines=1,
        train_sha256=sha256_text("ab cd x"), eval_sha256=sha256_text("ab cd x"),
    )
    report = evaluate_tokenizer(tok, examples, manifest, experiment_id="EXP-TEST")

    overall = report.overall
    assert overall.examples == 2
    assert overall.chars == 6
    assert overall.tokens == 6
    assert overall.words == 3
    assert overall.tokens_per_char == pytest.approx(1.0)
    assert overall.chars_per_token == pytest.approx(1.0)
    assert overall.tokens_per_word == pytest.approx(2.0)
    assert overall.round_trip_failures == 0
    assert report.checks["utf8_round_trip_ok"] is True
    assert "en" in report.per_language
    assert "sentence" in report.per_category


def test_evaluator_counts_round_trip_failures_and_unks(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt", text="abc")
    tok = CharTokenizerAdapter()
    tok.train(corpus)
    examples = [Example(id="1", lang="en", category="sentence", text="abcζ")]
    manifest = CorpusManifest(
        corpus_id="unit", corpus_version="test", schema_version="1.0", seed=0, n_examples=1,
        languages=["en"], categories=["sentence"], train_chars=3, train_lines=1,
        train_sha256=sha256_text("abc"), eval_sha256=sha256_text("abcζ"),
    )
    report = evaluate_tokenizer(tok, examples, manifest)
    assert report.overall.round_trip_failures == 1
    assert report.checks["utf8_round_trip_ok"] is False
    assert report.overall.unk_count == 1
    assert report.overall.unk_rate == pytest.approx(1 / 4)
    assert report.failing_examples and report.failing_examples[0]["id"] == "1"


def test_evaluator_rejects_empty_example_set(tmp_path):
    manifest = CorpusManifest(
        corpus_id="unit", corpus_version="test", schema_version="1.0", seed=0, n_examples=0,
        languages=[], categories=[], train_chars=0, train_lines=0,
        train_sha256="0", eval_sha256="0",
    )
    tok = train_python_bpe(tmp_path, vocab_size=300)
    with pytest.raises(ValueError, match="no evaluation examples"):
        evaluate_tokenizer(tok, [], manifest)


def test_evaluator_reports_no_unk_for_byte_level(tmp_path):
    tok = train_python_bpe(tmp_path, vocab_size=320)
    examples = [Example(id="1", lang="hi", category="sentence", text="मैं स्कूल जाता हूँ")]
    manifest = CorpusManifest(
        corpus_id="unit", corpus_version="test", schema_version="1.0", seed=0, n_examples=1,
        languages=["hi"], categories=["sentence"], train_chars=10, train_lines=1,
        train_sha256="0", eval_sha256="0",
    )
    report = evaluate_tokenizer(tok, examples, manifest)
    assert report.overall.unk_count is None  # byte-level: no unknown tokens exist
    assert report.overall.round_trip_failures == 0


def test_count_words_ignores_punctuation_only_tokens():
    assert count_words("hello world  !!! 42") == 3
    assert count_words("!!!") == 0


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def _report_for(tok, examples, manifest, exp_id="EXP-X"):
    return evaluate_tokenizer(tok, examples, manifest, experiment_id=exp_id)


def test_comparison_ranks_only_lossless_tokenizers(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt", text="abc def ghi")
    examples = [Example(id="1", lang="en", category="sentence", text="abc def"),
                Example(id="2", lang="en", category="sentence", text="abc ζ")]
    manifest = CorpusManifest(
        corpus_id="unit", corpus_version="v1", schema_version="1.0", seed=0, n_examples=2,
        languages=["en"], categories=["sentence"], train_chars=11, train_lines=1,
        train_sha256="0", eval_sha256="0",
    )
    char_tok = CharTokenizerAdapter()
    char_tok.train(corpus)
    bpe = PythonBPE()
    bpe.train(corpus, vocab_size=300)

    comparison = build_comparison(
        [_report_for(char_tok, examples, manifest), _report_for(bpe, examples, manifest)],
        labels=["char", "bpe"],
    )
    assert [d["tokenizer"] for d in comparison["disqualified"]] == ["char"]
    assert comparison["best"]["chars_per_token"]["tokenizer"] == "bpe"
    assert comparison["best"]["round_trip_failures"]["tokenizer"] == "bpe"
    rendered = render_comparison(comparison)
    assert "DISQUALIFIED" in rendered and "bpe" in rendered


def test_comparison_rejects_mismatched_corpus_or_labels(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    bpe = PythonBPE()
    bpe.train(corpus, vocab_size=300)
    examples = [Example(id="1", lang="en", category="sentence", text="abc")]
    base = dict(schema_version="1.0", seed=0, n_examples=1, languages=["en"], categories=["sentence"],
                train_chars=3, train_lines=1, train_sha256="0", eval_sha256="0")
    m1 = CorpusManifest(corpus_id="a", corpus_version="v1", **base)
    m2 = CorpusManifest(corpus_id="a", corpus_version="v2", **base)

    with pytest.raises(ValueError, match="same corpus version"):
        build_comparison([_report_for(bpe, examples, m1), _report_for(bpe, examples, m2)], labels=["a", "b"])
    with pytest.raises(ValueError, match="duplicate"):
        build_comparison([_report_for(bpe, examples, m1), _report_for(bpe, examples, m1)], labels=["a", "a"])
    with pytest.raises(ValueError, match="labels and reports"):
        build_comparison([_report_for(bpe, examples, m1)], labels=["a", "b"])
    with pytest.raises(ValueError, match="no evaluation reports"):
        build_comparison([])


def test_comparison_rows_match_source_reports(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    bpe = PythonBPE()
    bpe.train(corpus, vocab_size=320)
    examples = [Example(id="1", lang="en", category="sentence", text="hello world")]
    manifest = CorpusManifest(
        corpus_id="unit", corpus_version="v1", schema_version="1.0", seed=0, n_examples=1,
        languages=["en"], categories=["sentence"], train_chars=11, train_lines=1,
        train_sha256="0", eval_sha256="0",
    )
    report = _report_for(bpe, examples, manifest)
    comparison = build_comparison([report], labels=["bpe"])
    row = comparison["rows"][0]
    assert row["tokens"] == report.overall.tokens
    assert row["chars"] == report.overall.chars
    assert row["bytes_per_token"] == pytest.approx(report.overall.utf8_bytes / report.overall.tokens)
    assert row["per_language"]["en"]["tokens"] == report.overall.tokens


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def test_artifact_manifest_records_provenance(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    tok = PythonBPE()
    tok.train(corpus, vocab_size=320, special_tokens=SPECIALS)
    manifest = save_artifact(
        tok,
        tmp_path / "artifact",
        experiment_id="EXP-002",
        corpus={"corpus_id": "unit", "corpus_version": "v1", "train_sha256": sha256_text("tiny")},
        train_params={"vocab_size": 320},
        seed=1337,
    )
    assert manifest.impl == "bpe_python" and manifest.vocab_size == 320
    assert manifest.experiment_id == "EXP-002" and manifest.seed == 1337
    assert (tmp_path / "artifact" / "manifest.json").exists()
    assert (tmp_path / "artifact" / "vocab.json").exists()

    loaded_tok, loaded_manifest = load_artifact(tmp_path / "artifact")
    assert loaded_manifest.impl == manifest.impl
    assert loaded_tok.encode("test <pad>") == tok.encode("test <pad>")
    assert load_manifest(tmp_path / "artifact").experiment_id == "EXP-002"


def test_artifact_detects_corrupted_vocab_size(tmp_path):
    corpus = write_tiny_corpus(tmp_path / "tiny.txt")
    tok = PythonBPE()
    tok.train(corpus, vocab_size=320)
    save_artifact(tok, tmp_path / "artifact", experiment_id="EXP", corpus={}, train_params={})
    manifest_path = tmp_path / "artifact" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["vocab_size"] = 999
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="corrupted"):
        load_artifact(tmp_path / "artifact")


def test_load_artifact_requires_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="no tokenizer manifest"):
        load_artifact(tmp_path / "empty")


# ---------------------------------------------------------------------------
# end-to-end CLI
# ---------------------------------------------------------------------------
def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_cli_workflow_prepare_train_evaluate_compare(tmp_path):
    corpus = tmp_path / "corpus"
    result = _run("tokenizer_prepare_corpus.py", "--out", str(corpus), "--sentences-per-language", "3",
                  "--english-chars", "1500")
    assert result.returncode == 0, result.stderr
    assert (corpus / "eval.jsonl").exists() and (corpus / "manifest.json").exists()

    artifact = tmp_path / "bpe"
    result = _run("tokenizer_train.py", "--corpus", str(corpus), "--impl", "bpe_python",
                  "--vocab-size", "300", "--out", str(artifact), "--exp-id", "EXP-TEST")
    assert result.returncode == 0, result.stderr
    assert (artifact / "manifest.json").exists()

    eval_out = tmp_path / "eval.json"
    result = _run("tokenizer_evaluate.py", "--corpus", str(corpus), "--tokenizer", str(artifact),
                  "--out", str(eval_out), "--exp-id", "EXP-TEST")
    assert result.returncode == 0, result.stderr
    report = json.loads(eval_out.read_text())
    assert report["overall"]["examples"] > 0
    assert report["checks"]["utf8_round_trip_ok"] is True
    assert report["tokenizer"]["impl"] == "bpe_python"

    second = tmp_path / "char"
    assert _run("tokenizer_train.py", "--corpus", str(corpus), "--impl", "char",
                "--out", str(second)).returncode == 0

    compare_out = tmp_path / "compare.json"
    result = _run("tokenizer_compare.py", "--corpus", str(corpus),
                  "--tokenizer", str(artifact), str(second), "--out", str(compare_out))
    assert result.returncode == 0, result.stderr
    comparison = json.loads(compare_out.read_text())
    assert len(comparison["rows"]) == 2
    assert "best" in comparison and "OVERALL" in result.stdout


def test_cli_rejects_unknown_implementation(tmp_path):
    corpus = tmp_path / "corpus"
    _run("tokenizer_prepare_corpus.py", "--out", str(corpus), "--sentences-per-language", "2",
         "--english-chars", "500")
    result = _run("tokenizer_train.py", "--corpus", str(corpus), "--impl", "nope",
                  "--out", str(tmp_path / "x"))
    assert result.returncode != 0
    assert "unknown implementation" in (result.stderr + result.stdout)


def test_cli_evaluate_requires_existing_corpus(tmp_path):
    result = _run("tokenizer_evaluate.py", "--corpus", str(tmp_path / "missing"),
                  "--tokenizer", str(tmp_path / "nope"), "--out", str(tmp_path / "o.json"))
    assert result.returncode != 0
