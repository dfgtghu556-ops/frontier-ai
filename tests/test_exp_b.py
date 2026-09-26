"""EXP-B harness: document-aligned data preparation and the 2-tokenizer x N-seed matrix.

End-to-end on the tiny *fake* frozen corpus (same pattern as the EXP-A e2e test):
build the fake corpus, build a Frontier dataset, train two small tokenizers via
the sweep's ``run_one_config``, then run the real ``prepare_exp_b_data.py`` and
``run_exp_b.py`` scripts. A repeated cell must reproduce the identical
bits-per-byte (deterministic mode). No network, no real corpus text.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.corpus import build_frontier_dataset
from frontier_ai.data.dataset import TokenDataset
from frontier_ai.tokenization.sweep import PRETOKEN_MARK_AWARE, SweepConfig, run_one_config
from test_tokenizer_sweep import _make_fake_frozen_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(path: str, args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / path), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _build_frontier(fake: dict, out: Path):
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
    assert build.manifest["sides"]["held_out"]["documents"] > 0
    return build


def _train_two_tokenizers(frontier_dir: Path, tmp_path: Path) -> list[tuple[str, Path]]:
    """Two small mark-aware BPE artifacts via the sweep's per-config trainer."""
    # re-derive the documents the same way the scripts do (fake pins, no network)
    from frontier_ai.corpus import derive_frontier_documents

    manifest = json.loads((frontier_dir / "manifest.json").read_text(encoding="utf-8"))
    split = manifest["identity"]["split"]
    derivation = derive_frontier_documents(
        manifest_path=frontier_dir.parent / "fake" / "sources.json",
        freeze_path=frontier_dir.parent / "fake" / "FREEZE.json",
        corpus_dir=frontier_dir.parent / "fake" / "corpus",
        seed=int(split["seed"]),
        held_out_fraction=float(split["held_out_fraction"]),
    )
    train_docs = list(derivation.train)
    held_docs = list(derivation.held_out)
    train_file = tmp_path / "train.txt"
    train_file.write_text("\n".join(d.text for d in train_docs), encoding="utf-8")

    artifacts: list[tuple[str, Path]] = []
    for name, vocab in (("t-small", 300), ("t-big", 400)):
        results = run_one_config(
            SweepConfig("bpe_python", PRETOKEN_MARK_AWARE, vocab),
            train_file=train_file,
            train_documents=train_docs,
            heldout_documents=held_docs,
            out_dir=tmp_path / f"tok-{name}",
        )
        assert results["gate"]["lossless"] is True
        artifacts.append((name, Path(results["artifact"])))
    return artifacts


# ---------------------------------------------------------------------------
# write_split_tokens
# ---------------------------------------------------------------------------
def test_write_split_tokens_round_trip(tmp_path):
    from frontier_ai.data.dataset import write_split_tokens

    out = tmp_path / "toks.bin"
    meta = write_split_tokens(
        out, [1, 2, 3, 4], [5, 6], vocab_size=10, level="bpe",
        train_bytes=8, train_chars=4, val_bytes=2, val_chars=2,
    )
    assert (meta.n_train, meta.n_val) == (4, 2)
    assert (meta.n_bytes_train, meta.n_bytes_val) == (8, 2)
    assert (meta.n_chars_train, meta.n_chars_val) == (4, 2)
    ds = TokenDataset(out)
    assert ds.meta.n_tokens == 6
    assert list(ds.train) == [1, 2, 3, 4]
    assert list(ds.val) == [5, 6]
    # bits-per-byte needs the exact byte counts: 6 tokens over 10 val bytes...
    assert abs(ds.tokens_per_byte("val") - 2 / 2) < 1e-9
    assert abs(ds.tokens_per_byte("train") - 4 / 8) < 1e-9


# ---------------------------------------------------------------------------
# prepare_exp_b_data.py e2e
# ---------------------------------------------------------------------------
def test_prepare_exp_b_e2e_on_fake_corpus(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path / "fake", n_hi=30, n_en=12)
    build = _build_frontier(fake, tmp_path / "frontier")
    artifacts = _train_two_tokenizers(tmp_path / "frontier", tmp_path)

    out_dir = tmp_path / "expb_data"
    proc = _run_script(
        "prepare_exp_b_data.py",
        ["--exp-id", "EXP-099", "--frontier-dir", str(tmp_path / "frontier"),
         "--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]),
         "--tokens", ",".join(f"{n}={a}" for n, a in artifacts),
         "--out", str(out_dir), "--no-record"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [t["name"] for t in data["tokenizers"]] == [n for n, _ in artifacts]
    assert data["frontier_manifest_content_sha256"] == build.manifest["content_sha256"]

    train_docs, held_docs = build.train_docs, build.held_docs
    expect_train_bytes = sum(len(d.text.encode("utf-8")) for d in train_docs)
    expect_held_bytes = sum(len(d.text.encode("utf-8")) for d in held_docs)
    for entry in data["tokenizers"]:
        assert (out_dir / entry["file"]).is_file()
        ds = TokenDataset(out_dir / entry["file"])
        assert ds.meta.vocab_size == entry["vocab_size"]
        assert ds.meta.n_train == entry["n_tokens_train"] > 0
        assert ds.meta.n_val == entry["n_tokens_val"] > 0
        # exact byte/char totals (bits-per-byte correctness depends on these)
        assert ds.meta.n_bytes_train == expect_train_bytes == entry["n_bytes_train"]
        assert ds.meta.n_bytes_val == expect_held_bytes == entry["n_bytes_val"]
        assert ds.meta.n_chars_train == sum(d.chars for d in train_docs)
        assert ds.meta.n_chars_val == sum(d.chars for d in held_docs)


# ---------------------------------------------------------------------------
# run_exp_b.py e2e + determinism
# ---------------------------------------------------------------------------
def _tiny_config(tmp_path: Path) -> Path:
    cfg = {
        "model": {"n_layer": 1, "n_head": 2, "n_embd": 32, "block_size": 32, "dropout": 0.0,
                  "norm": "rmsnorm", "ffn": "swiglu", "pos": "learned", "tie_embeddings": True},
        "data": {"path": "placeholder", "batch_size": 4, "num_workers": 0, "seed": 1337},
        "optim": {"lr": 0.003, "min_lr_ratio": 0.1, "weight_decay": 0.1,
                  "betas": [0.9, 0.95], "grad_clip": 1.0, "warmup_steps": 10,
                  "schedule": "cosine"},
        "train": {"out_dir": "out/x", "max_steps": 30, "accum_steps": 1,
                  "eval_interval": 30, "eval_iters": 5, "log_interval": 30,
                  "save_interval": 0, "save_best": True, "device": "cpu",
                  "precision": "fp32", "seed": 1337, "deterministic": True, "num_threads": 2},
        "gen": {"max_new_tokens": 10, "temperature": 0.8, "top_k": 20, "top_p": 1.0},
    }
    path = tmp_path / "tiny_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_run_exp_b_matrix_and_determinism(tmp_path):
    fake = _make_fake_frozen_corpus(tmp_path / "fake", n_hi=30, n_en=12)
    _build_frontier(fake, tmp_path / "frontier")
    artifacts = _train_two_tokenizers(tmp_path / "frontier", tmp_path)

    data_dir = tmp_path / "expb_data"
    proc = _run_script(
        "prepare_exp_b_data.py",
        ["--exp-id", "EXP-099", "--frontier-dir", str(tmp_path / "frontier"),
         "--manifest", str(fake["manifest"]), "--freeze", str(fake["freeze"]),
         "--corpus-dir", str(fake["corpus_dir"]),
         "--tokens", ",".join(f"{n}={a}" for n, a in artifacts),
         "--out", str(data_dir), "--no-record"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    cfg = _tiny_config(tmp_path)
    proc = _run_script(
        "run_exp_b.py",
        ["--exp-id", "EXP-099", "--config", str(cfg), "--data-dir", str(data_dir),
         "--seeds", "1337,1338", "--max-steps", "30", "--no-record"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = (data_dir / "runs" / "report.txt").read_text(encoding="utf-8")
    assert "DECISION (pre-registered rule" in report
    assert "t-small" in report and "t-big" in report
    # every cell ran (2 tokenizers x 2 seeds)
    for name, _ in artifacts:
        for seed in (1337, 1338):
            record = data_dir / "runs" / name / f"seed-{seed}" / "experiment.json"
            data = json.loads(record.read_text(encoding="utf-8"))
            assert data["execution"]["status"] == "success"
            assert data["results"]["best_bpb"] is not None

    # determinism: the same cell (same seed, fresh out dir) must reproduce
    # the identical bits-per-byte
    name = artifacts[0][0]
    first = json.loads(
        (data_dir / "runs" / name / "seed-1337" / "experiment.json").read_text(encoding="utf-8")
    )["results"]["best_bpb"]
    repeat_dir = tmp_path / f"repeat-{name}"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/train.py"),
         "--config", str(cfg),
         "--set", f"data.path={data_dir / (name + '.bin')}",
         "--set", "data.seed=1337", "--set", "train.seed=1337",
         "--set", f"train.out_dir={repeat_dir}", "--set", "train.max_steps=30",
         "--exp-id", "EXP-0995"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    repeated = json.loads(
        (repeat_dir / "experiment.json").read_text(encoding="utf-8")
    )["results"]["best_bpb"]
    assert repeated == pytest.approx(first, abs=1e-12), (
        f"deterministic mode must reproduce the cell exactly: {repeated} vs {first}"
    )


def test_run_exp_b_refuses_missing_data(tmp_path):
    proc = _run_script(
        "run_exp_b.py",
        ["--exp-id", "EXP-099", "--data-dir", str(tmp_path / "nope"),
         "--max-steps", "30", "--no-record"],
    )
    assert proc.returncode == 2
    assert "no prepared data" in proc.stderr
