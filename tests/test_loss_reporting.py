"""Bits-per-byte / per-character loss reporting (Project 003, Stage 1 item 2).

Per-token loss is not comparable across tokenizers: it depends on how much text a
token happens to carry. These tests pin down the conversion, the "unknown is
null, never 0" rule, the exactness of the byte/character bookkeeping, and the
end-to-end CLI behaviour.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.config import DataConfig, ExperimentConfig, ModelConfig, OptimConfig, TrainConfig
from frontier_ai.data.dataset import DataMeta, TokenDataset, text_lengths, write_tokens
from frontier_ai.data.synthetic import generate_corpus
from frontier_ai.data.tokenizer import CharTokenizer, WordTokenizer, fit_tokenizer
from frontier_ai.engine.metrics import (
    bits_from_nats,
    bits_per_unit,
    bpb_note,
    loss_summary,
    perplexity,
)
from frontier_ai.engine.trainer import Trainer
from frontier_ai.utils.seed import set_seed

REPO = Path(__file__).resolve().parents[1]
MULTILINGUAL = "the cat sat. \u0a2a\u0a70\u0a3c\u0a3e\u0a2c \u0915\u094d\u092f\u093e \u0986\u09ae\u09bf 42! \u00e9\u00e8\n"


# --------------------------------------------------------------- conversions --


def test_nats_convert_to_bits():
    assert bits_from_nats(math.log(2)) == pytest.approx(1.0)
    assert bits_from_nats(0.0) == 0.0
    assert bits_from_nats(1.0) == pytest.approx(1.4426950408889634)


def test_perplexity_is_capped_so_it_stays_json_serialisable():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(100.0) == pytest.approx(math.exp(20.0))
    assert perplexity(100.0, cap=5.0) == pytest.approx(math.exp(5.0))


def test_bits_per_unit_scales_by_the_token_ratio():
    nats = math.log(2)  # 1 bit per token
    assert bits_per_unit(nats, 1.0) == pytest.approx(1.0)  # 1 token per byte
    assert bits_per_unit(nats, 0.25) == pytest.approx(0.25)  # 4 bytes per token
    assert bits_per_unit(nats, 4.0) == pytest.approx(4.0)  # 4 tokens per byte


@pytest.mark.parametrize("ratio", [None, 0.0, -1.0])
def test_bits_per_unit_is_none_not_zero_when_the_ratio_is_unknown(ratio):
    assert bits_per_unit(1.0, ratio) is None


def test_loss_summary_with_known_ratios():
    nats = math.log(2)
    report = loss_summary(nats, tokens_per_byte=0.5, tokens_per_char=2.0)
    assert report["val_loss"] == pytest.approx(nats)
    assert report["bits_per_token"] == pytest.approx(1.0)
    assert report["bits_per_byte"] == pytest.approx(0.5)
    assert report["bits_per_char"] == pytest.approx(2.0)
    assert report["tokens_per_byte"] == pytest.approx(0.5)


def test_loss_summary_reports_null_when_counts_are_missing():
    report = loss_summary(1.0)
    assert report["bits_per_byte"] is None
    assert report["bits_per_char"] is None
    assert report["tokens_per_byte"] is None
    # the per-token numbers are still reported: they are just not comparable
    assert report["bits_per_token"] == pytest.approx(1.442695)


def test_bpb_note_explains_both_cases():
    assert "null" in bpb_note(False)
    assert "comparable" in bpb_note(True)


# ------------------------------------------------------------ tokenization ---


@pytest.mark.parametrize("level", ["char", "word"])
def test_tokenize_pieces_reconstruct_the_source_text(level):
    text = MULTILINGUAL * 20
    tok = fit_tokenizer(text, level=level)
    pieces = tok.tokenize(text)
    assert "".join(pieces) == text, "pieces must partition the text for byte counts to be exact"
    assert len(pieces) == len(tok.encode(text))


def test_text_lengths_match_the_source_text_for_both_levels():
    text = MULTILINGUAL * 20
    expected_bytes = len(text.encode("utf-8"))
    expected_chars = len(text)
    for level in ("char", "word"):
        pieces = fit_tokenizer(text, level=level).tokenize(text)
        token_bytes, token_chars = text_lengths(pieces)
        assert sum(token_bytes) == expected_bytes
        assert sum(token_chars) == expected_chars


def test_multibyte_text_counts_bytes_not_characters():
    text = "\u00e9\u00e8\u0915\u094d\u092f\u093e"  # 6 chars, 14 UTF-8 bytes
    token_bytes, token_chars = text_lengths(fit_tokenizer(text, level="char").tokenize(text))
    assert sum(token_chars) == 6
    assert sum(token_bytes) == len(text.encode("utf-8"))
    assert sum(token_bytes) > sum(token_chars)


# ------------------------------------------------------------------ dataset --


def _prepare(tmp_path: Path, text: str, level: str, name: str) -> tuple[TokenDataset, DataMeta]:
    tok = fit_tokenizer(text, level=level)
    tok.save(tmp_path / f"{name}.tokenizer.json")
    token_bytes, token_chars = text_lengths(tok.tokenize(text))
    meta = write_tokens(
        tmp_path / f"{name}.bin",
        tok.encode(text),
        tok.vocab_size,
        level,
        val_frac=0.1,
        token_bytes=token_bytes,
        token_chars=token_chars,
    )
    return TokenDataset(tmp_path / f"{name}.bin", meta), meta


def test_write_tokens_records_split_byte_and_char_counts(tmp_path):
    text = MULTILINGUAL * 50
    ds, meta = _prepare(tmp_path, text, "word", "word")
    assert meta.has_text_lengths is True
    assert meta.n_bytes_train + meta.n_bytes_val == len(text.encode("utf-8"))
    assert meta.n_chars_train + meta.n_chars_val == len(text)
    # splits are contiguous, so the sums must be strictly positive and ordered
    assert meta.n_bytes_train > 0 and meta.n_bytes_val > 0
    assert meta.n_bytes_train > meta.n_bytes_val  # 90/10 split
    assert ds.tokens_per_byte("val") == pytest.approx(ds.n_val / meta.n_bytes_val)
    assert ds.tokens_per_char("val") == pytest.approx(ds.n_val / meta.n_chars_val)


def test_length_counts_must_align_with_the_token_stream(tmp_path):
    ids = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="align"):
        write_tokens(tmp_path / "a.bin", ids, 10, "char", token_bytes=[1, 1], token_chars=[1, 1])
    with pytest.raises(ValueError, match="both"):
        write_tokens(tmp_path / "b.bin", ids, 10, "char", token_bytes=[1, 1, 1, 1])


def test_corpus_without_counts_stays_unknown_rather_than_zero(tmp_path):
    meta = write_tokens(tmp_path / "legacy.bin", [0, 1, 2, 3] * 10, 10, "char")
    assert meta.n_bytes is None and meta.n_chars is None
    ds = TokenDataset(tmp_path / "legacy.bin", meta)
    assert ds.tokens_per_byte("val") is None
    assert ds.tokens_per_char("val") is None
    assert loss_summary(1.0, tokens_per_byte=ds.tokens_per_byte())["bits_per_byte"] is None


def test_pre_bpb_meta_files_still_load(tmp_path):
    legacy = {
        "n_tokens": 100,
        "vocab_size": 10,
        "dtype": "uint8",
        "level": "char",
        "n_train": 90,
        "n_val": 10,
    }
    path = tmp_path / "legacy.meta.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    meta = DataMeta.load(path)
    assert meta.has_text_lengths is False
    assert meta.n_bytes_val is None


def test_same_text_different_tokenizers_agree_on_bytes_not_tokens(tmp_path):
    """The whole point: byte counts are tokenizer-independent, token counts are not."""
    text = generate_corpus(target_chars=20_000, seed=3)
    _, char_meta = _prepare(tmp_path, text, "char", "char")
    _, word_meta = _prepare(tmp_path, text, "word", "word")

    assert char_meta.n_bytes == word_meta.n_bytes == len(text.encode("utf-8"))
    assert char_meta.n_chars == word_meta.n_chars == len(text)
    assert char_meta.n_tokens > word_meta.n_tokens  # more, smaller tokens

    # identical per-token loss therefore means a very different bits per byte
    nats = 1.0
    char_bpb = bits_per_unit(nats, char_meta.n_tokens / char_meta.n_bytes)
    word_bpb = bits_per_unit(nats, word_meta.n_tokens / word_meta.n_bytes)
    assert word_bpb < char_bpb  # a word token carries more text, so fewer bits/byte


# ------------------------------------------------------------------ trainer --


def _tiny_cfg(tmp_path: Path, data_path: Path, vocab_size: int, **overrides) -> ExperimentConfig:
    train = dict(
        out_dir=str(tmp_path / "out"),
        max_steps=20,
        eval_interval=10,
        eval_iters=4,
        log_interval=5,
        device="cpu",
        precision="fp32",
        seed=0,
        accum_steps=1,
    )
    train.update(overrides)
    return ExperimentConfig(
        model=ModelConfig(vocab_size=vocab_size, n_layer=2, n_head=2, n_embd=32, block_size=32),
        data=DataConfig(path=str(data_path), batch_size=4),
        optim=OptimConfig(lr=0.01, warmup_steps=5),
        train=TrainConfig(**train),
    )


def test_trainer_logs_bits_per_byte_and_returns_it(tmp_path):
    set_seed(0)
    text = generate_corpus(target_chars=20_000, seed=5)
    ds, meta = _prepare(tmp_path, text, "char", "chars")
    cfg = _tiny_cfg(tmp_path, tmp_path / "chars.bin", meta.vocab_size)
    trainer = Trainer(cfg, ds)
    result = trainer.fit()

    assert result["best_bpb"] is not None
    expected = result["best_val"] / math.log(2) * ds.tokens_per_byte("val")
    assert result["best_bpb"] == pytest.approx(expected, abs=1e-6)

    events = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    eval_events = [e for e in events if e["event"] == "eval"]
    assert eval_events, "no eval events were logged"
    for event in eval_events:
        assert event["bits_per_byte"] == pytest.approx(event["val_loss"] / math.log(2), abs=1e-5)
    assert events[-1]["event"] == "run.end"


def test_trainer_without_byte_counts_logs_no_bits_per_byte(tmp_path):
    set_seed(0)
    text = generate_corpus(target_chars=20_000, seed=5)
    tok = CharTokenizer.fit(text)
    meta = write_tokens(tmp_path / "legacy.bin", tok.encode(text), tok.vocab_size, "char")
    ds = TokenDataset(tmp_path / "legacy.bin", meta)
    cfg = _tiny_cfg(tmp_path, tmp_path / "legacy.bin", meta.vocab_size, max_steps=10, eval_interval=10)
    result = Trainer(cfg, ds).fit()

    assert result["best_bpb"] is None
    events = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    for event in events:
        assert "bits_per_byte" not in event


def test_evaluate_cli_reports_bits_per_byte_for_both_levels(tmp_path):
    """End to end: prepare -> train -> scripts/evaluate.py prints comparable losses."""
    set_seed(0)
    text = generate_corpus(target_chars=20_000, seed=9)
    reports = {}
    for level in ("char", "word"):
        level_dir = tmp_path / level
        level_dir.mkdir()
        ds, meta = _prepare(level_dir, text, level, level)
        cfg = _tiny_cfg(level_dir, level_dir / f"{level}.bin", meta.vocab_size, max_steps=10)
        model_trainer = Trainer(cfg, ds)
        model_trainer.fit()
        out = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "evaluate.py"),
                "--ckpt",
                str(level_dir / "out" / "best"),
                "--data",
                str(level_dir / f"{level}.bin"),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        payload = json.loads(out.stdout)
        reports[level] = payload
        assert payload["bits_per_byte"] is not None
        assert payload["bits_per_char"] is not None
        assert payload["corpus"]["level"] == level
        assert payload["corpus"]["n_val_bytes"] == meta.n_bytes_val

    # the two runs score the same text, so the byte counts must line up even
    # though the token vocabularies and token counts do not
    bytes_per_token_char = 1 / reports["char"]["tokens_per_byte"]
    bytes_per_token_word = 1 / reports["word"]["tokens_per_byte"]
    assert bytes_per_token_word > bytes_per_token_char


def test_prepare_data_cli_records_byte_counts(tmp_path):
    out_prefix = tmp_path / "synth" / "corpus"
    out_prefix.parent.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "prepare_data.py"),
            "--source",
            "synthetic",
            "--target-chars",
            "20000",
            "--out",
            str(out_prefix),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    meta = DataMeta.load(out_prefix.with_suffix(".meta.json"))
    assert meta.has_text_lengths is True
    assert meta.n_bytes == meta.n_tokens  # the synthetic corpus is ASCII
    assert "bytes/token" in proc.stdout


def test_word_level_pieces_and_vocab_still_round_trip():
    text = generate_corpus(target_chars=5_000, seed=13)
    tok = WordTokenizer.fit(text, min_count=1)
    assert tok.decode(tok.encode(text)) == text
    assert tok.tokenize("the cat sat") == ["the", " ", "cat", " ", "sat"]
