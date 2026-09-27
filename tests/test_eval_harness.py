"""Evaluation harness v1 (step 8): suite, exact scoring, statistics, contamination, reports.

End-to-end on the tiny fake frozen corpus: build a Frontier dataset, train small
tokenizers, prepare EXP-B-style data, save tiny (randomly initialised, seeded) GPT
checkpoints, build the protected suite, then run the real scripts. The core numbers are
cross-checked against an independent computation through the model's own loss.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from frontier_ai.config import ExperimentConfig
from frontier_ai.engine import checkpoint as ckpt
from frontier_ai.evaluation.contamination import contamination_report
from frontier_ai.evaluation.scoring import doc_bits, encode_stream, token_nats
from frontier_ai.evaluation.stats import bootstrap_ratio_ci, paired_delta
from frontier_ai.evaluation.suite import (
    SuiteError,
    build_suite,
    find_exact_overlap,
    load_suite,
    verify_docs_against_suite,
    write_suite,
)
from test_exp_b import _build_frontier, _run_script, _train_two_tokenizers
from test_tokenizer_sweep import _make_fake_frozen_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
LN2 = math.log(2.0)


def _doc(i: int, text: str, lang: str = "hi") -> SimpleNamespace:
    return SimpleNamespace(doc_id=f"src-{i:06d}", source_id="src", language=lang, text=text)


def _save_ckpt(out: Path, vocab: int, data_path: Path, seed: int) -> Path:
    cfg = ExperimentConfig.from_dict({
        "model": {"n_layer": 1, "n_head": 2, "n_embd": 32, "block_size": 32, "dropout": 0.0,
                  "norm": "rmsnorm", "ffn": "swiglu", "pos": "learned", "tie_embeddings": True,
                  "vocab_size": vocab},
        "data": {"path": str(data_path), "batch_size": 4, "num_workers": 0, "seed": seed},
        "train": {"device": "cpu", "precision": "fp32", "seed": seed},
    })
    torch.manual_seed(seed)
    model = ckpt.build_model_from_config(cfg, torch.device("cpu"))
    return ckpt.save_checkpoint(out, model, cfg=cfg, step=7, tag="best")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("evalh")
    fake = _make_fake_frozen_corpus(tmp / "fake", n_hi=30, n_en=12)
    _build_frontier(fake, tmp / "frontier")
    (small, small_art), (big, big_art) = _train_two_tokenizers(tmp / "frontier", tmp)
    data_dir = tmp / "data"
    proc = _run_script("prepare_exp_b_data.py", [
        "--frontier-dir", str(tmp / "frontier"), "--manifest", str(fake["manifest"]),
        "--freeze", str(fake["freeze"]), "--corpus-dir", str(fake["corpus_dir"]),
        "--tokens", f"{small}={small_art},{big}={big_art}", "--out", str(data_dir), "--no-record"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    suite_path = tmp / "suite" / "SUITE.json"
    corpus_args = ["--frontier-dir", str(tmp / "frontier"), "--manifest", str(fake["manifest"]),
                   "--freeze", str(fake["freeze"]), "--corpus-dir", str(fake["corpus_dir"])]
    proc = _run_script("build_eval_suite.py", ["--out", str(suite_path), *corpus_args])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    c1 = _save_ckpt(tmp / "runs" / "small" / "seed-1", 300, data_dir / f"{small}.bin", 1)
    c2 = _save_ckpt(tmp / "runs" / "small" / "seed-2", 300, data_dir / f"{small}.bin", 2)
    return SimpleNamespace(tmp=tmp, fake=fake, small_art=small_art, big_art=big_art, suite=suite_path,
                           corpus_args=corpus_args, c1=c1, c2=c2, data_dir=data_dir, small=small)


def _report(env, ckpt_dir: Path, out: Path, *extra: str, tokenizer: Path | None = None):
    return _run_script("eval_report.py", [
        "--ckpt", str(ckpt_dir), "--tokenizer", str(tokenizer or env.small_art), "--suite", str(env.suite),
        *env.corpus_args, "--out", str(out), "--bootstrap", "200", "--no-record", *extra])


# ------------------------------------------------------------------ unit level --

def test_token_nats_scores_every_token_but_the_first_once_and_matches_model_loss():
    torch.manual_seed(0)
    cfg = ExperimentConfig.from_dict({"model": {"n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8,
                                                "dropout": 0.0, "vocab_size": 50}})
    model = ckpt.build_model_from_config(cfg, torch.device("cpu"))
    ids = np.random.default_rng(0).integers(0, 50, 45).astype(np.int64)  # 44 targets: 5 full windows + tail 4
    nats = token_nats(model, ids, block_size=8, batch_size=2, device=torch.device("cpu"))
    assert np.isnan(nats[0]) and not np.isnan(nats[1:]).any()
    # independent: the model's own mean loss per window, times window length
    total = 0.0
    t = torch.as_tensor(ids)
    with torch.no_grad():
        for s in range(0, 44, 8):
            n = min(8, 44 - s)
            out = model(t[s:s + n][None], targets=t[s + 1:s + n + 1][None])
            total += float(out.loss) * n
    assert nats[1:].sum() == pytest.approx(total, rel=1e-6)


def test_doc_bits_attribution_is_exact():
    class Tok:
        def encode(self, text):
            return [ord(c) % 7 for c in text]
    docs = [_doc(0, "abc"), _doc(1, "de"), _doc(2, "fghi")]
    stream = encode_stream(Tok(), docs)
    assert stream.tokens_per_doc.tolist() == [3, 2, 4]
    nats = np.array([np.nan, 1, 1, 2, 2, 3, 3, 3, 3], dtype=np.float64) * LN2
    assert doc_bits(stream, nats).tolist() == pytest.approx([2.0, 4.0, 12.0])


def test_bootstrap_is_seeded_and_brackets_the_point():
    rng = np.random.default_rng(1)
    bits, denom = rng.uniform(1, 5, 300), rng.uniform(10, 50, 300)
    ci1 = bootstrap_ratio_ci(bits, denom, n_boot=300, seed=3)
    assert ci1 == bootstrap_ratio_ci(bits, denom, n_boot=300, seed=3)
    assert ci1[0] < bits.sum() / denom.sum() < ci1[1]
    same = paired_delta(bits, bits, denom, n_boot=100)
    assert same["delta_b_minus_a"] == 0 and same["ci95"] == [0.0, 0.0]
    worse = paired_delta(bits, bits + 0.5, denom, n_boot=100)
    assert worse["verdict"].startswith("A better")


def test_contamination_finds_planted_exact_and_ngram_overlap():
    long_line = " ".join(f"w{i}" for i in range(20))
    train = [_doc(1, "unrelated training text"), _doc(2, "exact copy here"), _doc(3, "prefix " + long_line)]
    evald = [_doc(10, "exact copy here"), _doc(11, long_line + " suffix"), _doc(12, "short clean line")]
    rep = contamination_report(evald, train)
    assert rep["exact_duplicates"] == 1 and rep["exact_duplicate_doc_ids"] == ["src-000010"]
    assert rep["ngram_overlap_documents"] == 1 and rep["ngram_eligible_documents"] == 1
    assert rep["ngram_too_short_documents"] == 2


def test_suite_roundtrip_verification_and_training_guard(tmp_path):
    docs = [_doc(0, "पहला"), _doc(1, "second"), _doc(2, "তৃতীয়", "bn")]
    suite = build_suite(docs, "t-suite", {"content_sha256": "x"})
    path = tmp_path / "SUITE.json"
    write_suite(path, suite)
    assert b"\r\n" not in path.read_bytes()
    loaded = load_suite(path)
    verify_docs_against_suite(docs, loaded)
    with pytest.raises(SuiteError):
        verify_docs_against_suite([docs[1], docs[0], docs[2]], loaded)  # order matters
    with pytest.raises(SuiteError):
        verify_docs_against_suite([docs[0], docs[1], _doc(2, "changed", "bn")], loaded)
    assert find_exact_overlap(["nothing", "second"], loaded) == ["src-000001"]
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["documents"][0]["language"] = "mr"
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(SuiteError, match="fingerprint"):
        load_suite(path)


# ------------------------------------------------------------------ end to end --

def test_eval_report_e2e_exact_and_reproducible(env):
    out1, out2 = env.tmp / "r1", env.tmp / "r2"
    proc = _report(env, env.c1, out1)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _report(env, env.c1, out2).returncode == 0
    r1 = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
    r2 = json.loads((out2 / "report.json").read_text(encoding="utf-8"))

    # reproducible: bit-identical per-document scores
    assert r1["reproducibility"]["scores_sha256"] == r2["reproducibility"]["scores_sha256"]
    assert (out1 / "per_document.jsonl").read_bytes() == (out2 / "per_document.jsonl").read_bytes()
    # provenance + checks
    assert r1["data_identity"]["status"] == "PASS"
    assert r1["tokenizer"]["matches_frozen_v1"] in (False, None)
    assert r1["checkpoint"]["step"] == 7
    cov = r1["coverage"]
    assert cov["scored_tokens"] == cov["stream_tokens"] - 1
    suite = load_suite(env.suite)
    assert r1["suite"]["fingerprint"] == suite["fingerprint"]

    # exactness: per-document bits sum to the total from an independent model-loss computation
    docs = [json.loads(x) for x in (out1 / "per_document.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(d["tokens"] for d in docs) == cov["stream_tokens"]
    from frontier_ai.data.dataset import TokenDataset
    val = torch.as_tensor(np.asarray(TokenDataset(env.data_dir / f"{env.small}.bin").val, dtype=np.int64))
    cfg = ExperimentConfig.load(env.c1 / "config.json")
    model = ckpt.build_model_from_config(cfg, torch.device("cpu"))
    ckpt.load_checkpoint(env.c1, model, map_location="cpu")
    model.eval()
    total_nats, n = 0.0, len(val)
    with torch.no_grad():
        for s in range(0, n - 1, 32):
            k = min(32, n - 1 - s)
            total_nats += float(model(val[s:s + k][None], targets=val[s + 1:s + k + 1][None]).loss) * k
    assert sum(float(d["bits"]) for d in docs) == pytest.approx(total_nats / LN2, rel=1e-6)

    # aggregation: overall excludes the context-only doc; languages add up to overall
    ov = r1["results"]["overall"]
    scored = [d for d in docs if not d["context_only"]]
    assert ov["bytes"] == sum(d["bytes"] for d in scored)
    assert ov["bits_per_byte"] == pytest.approx(sum(float(d["bits"]) for d in scored) / ov["bytes"])
    langs = r1["results"]["per_language"]
    assert sum(g["bits"] for g in langs.values()) == pytest.approx(ov["bits"])
    assert sum(g["documents"] for g in langs.values()) == ov["documents"]
    lo, hi = ov["bits_per_byte_ci95"]
    assert lo <= ov["bits_per_byte"] <= hi
    # contamination is reported (fake corpus: numbers present, not asserted to be zero)
    assert r1["contamination"]["status"] == "CHECKED"
    assert r1["contamination"]["evaluation_documents"] == ov["documents"]
    assert "domain" in r1["results"] and "not available" in r1["results"]["domain"]
    assert (out1 / "report.txt").read_text(encoding="utf-8").startswith("EVALUATION REPORT")


def test_eval_report_refuses_wrong_tokenizer(env):
    proc = _report(env, env.c1, env.tmp / "bad-tok", tokenizer=env.big_art)
    assert proc.returncode == 2
    assert "vocab" in proc.stderr


def test_eval_report_refuses_suite_mismatch(env, tmp_path):
    suite = load_suite(env.suite)
    suite["documents"] = suite["documents"][:-1]
    from frontier_ai.evaluation.suite import records_fingerprint
    suite["fingerprint"] = records_fingerprint(suite["documents"])
    bad = tmp_path / "SUITE.json"
    write_suite(bad, suite)
    proc = _run_script("eval_report.py", [
        "--ckpt", str(env.c1), "--tokenizer", str(env.small_art), "--suite", str(bad), *env.corpus_args,
        "--out", str(tmp_path / "o"), "--no-record"])
    assert proc.returncode == 2
    assert "protected suite" in proc.stderr


def test_eval_report_flags_data_identity_failure(env, tmp_path):
    other = env.data_dir / f"{env.small}.bin"
    # a dataset whose val split differs: re-point --data at the other tokenizer's .bin
    big_bin = next(p for p in env.data_dir.glob("*.bin") if p != other)
    proc = _report(env, env.c1, tmp_path / "o", "--data", str(big_bin))
    assert proc.returncode == 1
    rep = json.loads((tmp_path / "o" / "report.json").read_text(encoding="utf-8"))
    assert rep["data_identity"]["status"] == "FAIL"


def test_build_eval_suite_is_idempotent_and_never_replaces(env):
    proc = _run_script("build_eval_suite.py", ["--out", str(env.suite), *env.corpus_args])
    assert proc.returncode == 0 and "already up to date" in proc.stdout
    suite = load_suite(env.suite)
    suite["source"]["content_sha256"] = "different"
    alt = env.tmp / "alt" / "SUITE.json"
    write_suite(alt, suite)
    proc = _run_script("build_eval_suite.py", ["--out", str(alt), *env.corpus_args])
    assert proc.returncode == 2 and "never replaced" in proc.stderr


def test_eval_compare_groups(env):
    ra, rb = env.tmp / "cmp-a", env.tmp / "cmp-b"
    assert _report(env, env.c1, ra).returncode == 0
    assert _report(env, env.c2, rb).returncode == 0
    out = env.tmp / "cmp"
    proc = _run_script("eval_compare.py", ["--a", str(ra), "--b", str(rb), "--label-a", "s1",
                                           "--label-b", "s2", "--bootstrap", "200", "--out", str(out)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    res = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    a = json.loads((ra / "report.json").read_text(encoding="utf-8"))["results"]["overall"]["bits_per_byte"]
    b = json.loads((rb / "report.json").read_text(encoding="utf-8"))["results"]["overall"]["bits_per_byte"]
    assert res["overall"]["delta_b_minus_a"] == pytest.approx(b - a)
    lo, hi = res["overall"]["ci95"]
    assert lo <= res["overall"]["delta_b_minus_a"] <= hi
    # self-comparison: zero difference
    proc = _run_script("eval_compare.py", ["--a", str(ra), "--b", str(ra), "--out", str(env.tmp / "self")])
    same = json.loads((env.tmp / "self" / "compare.json").read_text(encoding="utf-8"))
    assert same["overall"]["delta_b_minus_a"] == 0


def test_eval_report_recorded_mode(env, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    out = tmp_path / "rec"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval_report.py"), "--ckpt", str(env.c1),
         "--tokenizer", str(env.small_art), "--suite", str(env.suite), *env.corpus_args,
         "--out", str(out), "--bootstrap", "100", "--exp-id", "EXP-097"],
        capture_output=True, text=True, timeout=900, cwd=work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert record["execution"]["status"] == "success"
    assert record["results"]["data_identity"] == "PASS"
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert record["results"]["scores_sha256"] == rep["reproducibility"]["scores_sha256"]


def test_publish_eval_results_copies_small_files_only(tmp_path):
    src = tmp_path / "out" / "eval" / "EXP-096"
    (src / "cell-a").mkdir(parents=True)
    (src / "cell-a" / "report.json").write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
    (src / "cell-a" / "report.txt").write_text("r", encoding="utf-8")
    (src / "cell-a" / "per_document.jsonl").write_text("big", encoding="utf-8")
    (src / "compare").mkdir()
    (src / "compare" / "compare.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "evals" / "results" / "EXP-096"
    args = ["--exp-id", "EXP-096", "--src", str(src), "--dest", str(dest)]
    proc = _run_script("publish_eval_results.py", args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / "cell-a" / "report.json").read_bytes() == b'{\n  "x": 1\n}\n'
    assert (dest / "compare" / "compare.json").is_file()
    assert not (dest / "cell-a" / "per_document.jsonl").exists()
    assert _run_script("publish_eval_results.py", args).returncode == 0  # identical: fine
    (src / "cell-a" / "report.txt").write_text("changed", encoding="utf-8")
    proc = _run_script("publish_eval_results.py", args)
    assert proc.returncode == 2 and "REFUSED" in proc.stderr


def test_scripts_survive_a_windows_cp1252_console(env, tmp_path):
    """EXP-031 regression: eval_compare crashed printing U+2212 on a cp1252 console/pipe."""
    import os
    ra = tmp_path / "a"
    assert _report(env, env.c1, ra).returncode == 0
    cp1252 = {**os.environ, "PYTHONIOENCODING": "cp1252"}  # strict errors, like a Windows pipe
    runs = {
        "eval_compare.py": ["--a", str(ra), "--b", str(ra), "--out", str(tmp_path / "cmp")],
        "eval_report.py": ["--ckpt", str(env.c1), "--tokenizer", str(env.small_art), "--suite", str(env.suite),
                           *env.corpus_args, "--out", str(tmp_path / "r"), "--bootstrap", "50", "--no-record"],
        "build_eval_suite.py": ["--out", str(env.suite), *env.corpus_args],
        "publish_eval_results.py": ["--exp-id", "EXP-095", "--src", str(ra), "--dest", str(tmp_path / "pub")],
    }
    for script, args in runs.items():
        for extra in ([], ["--help"]):
            proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script), *args, *extra],
                                  capture_output=True, timeout=900, env=cp1252)
            assert proc.returncode == 0, (script, extra, proc.stderr.decode("cp1252", "replace")[-800:])
