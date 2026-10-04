"""EXP-043: the step-12 training runner (CPU smoke): learning-rate check, the main run carried over
three "sessions" through folders that stand in for Kaggle's outputs, the sha256 refusal and the
stop rules."""

# ruff: noqa: F811  (pytest fixtures named like the imported `langs` fixture)
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import gpu_pretrain as gp  # noqa: E402
from test_gpu_lr_arch import TEXT, langs  # noqa: E402, F401  (the 3-language fixture)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(out: Path, n: int) -> dict:
    return json.loads((out / f"session-{n}" / "summary.json").read_text(encoding="utf-8"))


def _session(langs: Path, res: Path, base: Path, chain_in: Path, chain_out: Path, *extra: str) -> int:
    return gp.main(
        [
            "--smoke",
            "--skip-tests",
            "--data-dir",
            str(langs),
            "--manifest",
            str(langs / "manifest.json"),
            "--out",
            str(res),
            "--prev-dir",
            str(res),
            "--chain-in",
            str(chain_in),
            "--chain-out",
            str(chain_out / "exp043_chain"),
            "--final-dir",
            str(base / "final"),
            "--scratch",
            str(base / "scratch"),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def chain(langs, tmp_path_factory):
    """Session 1 = learning-rate check; session 2 = first 100 main-run steps."""
    base = tmp_path_factory.mktemp("chain")
    res = base / "res"
    assert _session(langs, res, base, base / "nothing", base / "c1", "--part", "lrcheck") == 0
    assert _session(langs, res, base, base / "c1", base / "c2", "--session-steps", "100") == 0
    return base


def test_learning_rate_rule_follows_the_preregistration():
    setup = gp.SETUP_FULL

    def r(lr, bpb, status="done", failed=False):
        return {"lr": lr, "bpb_mean": bpb, "status": status, "failed": failed, "attempted": True}

    assert gp.choose_lr([r(5e-4, 0.9)], setup)["next"] == [2.5e-4, 1e-3]
    # clear winner in the middle
    d = gp.choose_lr([r(2.5e-4, 0.95), r(5e-4, 0.90), r(1e-3, 0.93)], setup)
    assert d["decided"] and d["lr"] == 5e-4
    # a lower lr within the noise (0.0174) is chosen instead of the best
    d = gp.choose_lr([r(2.5e-4, 0.95), r(5e-4, 0.91), r(1e-3, 0.90)], setup)
    assert d["decided"] and d["lr"] == 5e-4 and d["best_lr"] == 1e-3
    # chosen at the upper edge -> one extension run at 2e-3, then the rule on all four
    runs = [r(2.5e-4, 0.99), r(5e-4, 0.95), r(1e-3, 0.90)]
    d = gp.choose_lr(runs, setup)
    assert not d["decided"] and d["next"] == [2e-3]
    d = gp.choose_lr(runs + [r(2e-3, 0.85)], setup)
    assert d["decided"] and d["lr"] == 2e-3  # at the edge again, but at most one extension
    # lower edge (also via the noise rule) -> extension at 1.25e-4
    d = gp.choose_lr([r(2.5e-4, 0.905), r(5e-4, 0.90), r(1e-3, 0.95)], setup)
    assert d["next"] == [1.25e-4]
    # failed runs (NaN/inf or >5% skipped) cannot be chosen
    d = gp.choose_lr([r(2.5e-4, 0.95), r(5e-4, 0.80, failed=True), r(1e-3, 0.93)], setup)
    assert d["next"] == [2e-3]  # 1e-3 is the best eligible run, at the edge
    d = gp.choose_lr(
        [r(2.5e-4, 0.97), r(5e-4, 0.94), r(1e-3, 0.80, failed=True), r(2e-3, 0.7, failed=True)], setup
    )
    assert d["decided"] and d["lr"] == 5e-4
    d = gp.choose_lr([r(x, float("nan"), status="failed: x", failed=True) for x in setup["lr_grid"]], setup)
    assert d.get("stop") and not d["decided"]


def test_full_setup_is_the_preregistered_one():
    s = gp.SETUP_FULL
    assert s["model"] == {
        "n_layer": 14,
        "n_head": 14,
        "n_embd": 896,
        "block_size": 512,
        "pos": "rope",
        "n_kv_head": 2,
    }
    assert s["lr_grid"] == [2.5e-4, 5e-4, 1e-3] and s["lr_extension"] == {"low": 1.25e-4, "high": 2e-3}
    assert s["lr_check_steps"] * s["windows_per_step"] * 512 == 100_007_936  # 100 M tokens
    assert s["expected_steps"] == s["expected_windows"] // s["windows_per_step"] == 169_911
    assert (s["warmup"], s["min_lr_ratio"], s["eval_every"], s["save_every"]) == (1000, 0.1, 1000, 5000)
    assert (s["skip_window"], s["max_skipped"], s["diverge_nats"], s["diverge_evals"]) == (2000, 0.05, 0.1, 3)
    assert s["micro_options"] == [[16, 2], [8, 4]] and s["mem_limit_gb"] == 14.5
    assert s["total_cap_hours"] == 100.0 and s["noise"] == 0.0174
    assert (s["reference_bpb"], s["predicted_bpb"], s["consistent_band"]) == (0.7736, 0.683, 0.03)
    # the window count follows from the committed EXP-037 manifest (per file (n_train - 1) // 512)
    m = json.loads((ROOT / "evals/results/EXP-037/manifest.json").read_text(encoding="utf-8"))
    assert sum((f["n_train"] - 1) // 512 for f in m["files"]) == s["expected_windows"]
    # the C3-s3 reference run exists in the committed EXP-042 results
    ref = gp.reference_bpb(s)
    assert len(ref) == 13 and abs(sum(ref.values()) / 13 - 0.7736) < 5e-5


def test_three_sessions_carry_the_run_exactly(langs, chain, tmp_path):
    base, res = chain, chain / "res"
    s1, s2 = _summary(res, 1), _summary(res, 2)
    # session 1: the learning-rate check only (smoke: CPU, so no fp64 device comparison)
    assert s1["schema"] == "frontier-pretrain-v1" and s1["smoke"] is True and not s1.get("stopped")
    assert s1["part0"]["files_ok"] == 3 and s1["part0"]["one_step_check"]["max_loss_diff"] == 0.0
    runs = s1["lr_runs"]
    assert [r["lr"] for r in runs][:3] == [2.5e-4, 5e-4, 1e-3] and 3 <= len(runs) <= 4
    assert all(r["status"] == "done" and r["tokens"] == 10 * 4 * 16 and r["session"] == 1 for r in runs)
    assert s1["lr_choice"]["decided"] and "main" not in s1 and s1["checkpoint"] is None
    marker = json.loads((base / "c1/exp043_chain/chain.json").read_text(encoding="utf-8"))
    assert marker == {"exp_id": "EXP-043", "written_by_session": 1, "checkpoint": None}
    # session 2: finds session 1's output, starts the main run, stops at its budget with a checkpoint
    assert s2["chain_check"]["ok"] and "mounted" in s2["chain_check"]["result"]
    assert s2["order"]["fingerprint"] == s1["order"]["fingerprint"]
    total = s2["order"]["steps_per_pass"]
    m2 = s2["main"]
    assert (m2["start_step"], m2["end_step"], m2["total_steps"]) == (0, 100, total)
    assert m2["lr"] == s1["lr_choice"]["lr"] and m2["sampler_position"] == 100 * 4
    assert [c["step"] for c in m2["val_curve"]] == list(range(10, 101, 10))
    assert m2["periodic_saves"] == [25, 50, 75, 100] and set(m2["session_eval"]["bpb"]) == set(TEXT)
    ck = s2["checkpoint"]
    assert ck["session"] == 2 and ck["step"] == 100 and set(ck["files"]) == set(gp.CKPT_FILES)
    assert ck["files"]["model.pt"] == _sha(base / "c2/exp043_chain/last/model.pt")
    assert sorted(p.name for p in (base / "c2/exp043_chain").iterdir()) == [
        "chain.json",
        "last",
        "train.jsonl",
    ]
    # session 3 (in a copy, so the module fixture stays reusable): verifies, resumes, finishes the pass
    res3 = tmp_path / "res"
    shutil.copytree(res, res3)
    assert _session(langs, res3, tmp_path, base / "c2", tmp_path / "c3") == 0
    s3 = _summary(res3, 3)
    assert s3["chain_check"]["ok"] and "verified" in s3["chain_check"]["result"]
    m3 = s3["main"]
    assert (m3["start_step"], m3["end_step"], m3["session_end"]) == (100, total, "one pass complete")
    assert m3["tokens_seen"] == total * 4 * 16 and m3["sampler_position"] == total * 4
    assert s3["complete"] is True and set(s3["final"]["scores"]["bpb"]) == set(TEXT)
    assert set(s3["final"]["checks"]) >= {"check1_gate", "check2_prediction", "check3_languages"}
    samples = (res3 / "session-3" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(samples) == s3["final"]["samples"] == 3 * 3 * 2
    assert {json.loads(x)["language"] for x in samples} == set(TEXT)
    info = s3["final"]["model_final"]
    assert info["sha256"] == _sha(tmp_path / "final" / "model_final.pt")
    # the same run in ONE session gives bit-identical final weights: the chain is exact
    res1 = tmp_path / "single"
    res1.mkdir()
    shutil.copytree(res / "session-1", res1 / "session-1")
    assert _session(langs, res1, tmp_path / "s", base / "c1", tmp_path / "s" / "c") == 0
    one = _summary(res1, 2)
    assert one["main"]["end_step"] == total and one["final"]["model_final"]["sha256"] == info["sha256"]
    assert one["final"]["scores"]["bpb_mean"] == s3["final"]["scores"]["bpb_mean"]
    # a further launch does nothing
    assert _session(langs, res3, tmp_path, tmp_path / "c3", tmp_path / "c4") == 0
    assert "already completed" in _summary(res3, 4)["stopped"]


def test_a_checkpoint_with_other_sha256_is_refused(langs, chain, tmp_path):
    res = tmp_path / "res"
    shutil.copytree(chain / "res", res)
    bad = tmp_path / "in"
    shutil.copytree(chain / "c2", bad)
    opt = bad / "exp043_chain/last/optimizer.pt"
    opt.write_bytes(opt.read_bytes()[:-1] + bytes([opt.read_bytes()[-1] ^ 1]))
    assert _session(langs, res, tmp_path, bad, tmp_path / "c3") == 1
    s3 = _summary(res, 3)
    assert "REFUSED" in s3["stopped"] and "optimizer.pt" in s3["stopped"] and "main" not in s3
    # the committed checkpoint record is unchanged (the good copy lives in session 2's output)
    assert s3["checkpoint"] == _summary(res, 2)["checkpoint"]
    # nothing mounted at all: stops at Part 0 too
    res_b = tmp_path / "res_b"
    shutil.copytree(chain / "res", res_b)
    assert _session(langs, res_b, tmp_path / "b", tmp_path / "empty", tmp_path / "b" / "c3") == 1
    assert "was not found" in _summary(res_b, 3)["stopped"]


def test_divergence_stops_and_keeps_the_last_good_checkpoint(langs, chain, tmp_path, monkeypatch):
    res = tmp_path / "res"
    res.mkdir()
    shutil.copytree(chain / "res" / "session-1", res / "session-1")
    # a sampled validation loss that only rises: best at step 10, then 3 evaluations above best + 0.1
    monkeypatch.setattr(gp.Trainer, "evaluate", lambda self: 5.0 + 0.02 * self.state.step)
    assert _session(langs, res, tmp_path, chain / "c1", tmp_path / "c2") == 1
    s2 = _summary(res, 2)
    assert "0.1 nats above the best" in s2["stop_rule"] and s2["main"]["session_end"] == "stop rule"
    assert [c["step"] for c in s2["main"]["val_curve"]] == [10, 20, 30, 40]
    assert s2["main"]["periodic_saves"] == [25] and s2["checkpoint"]["step"] == 25  # not the step-40 state
    assert "session_eval" not in s2["main"] and not s2.get("complete")
    # going on needs a decision: the next launch refuses
    monkeypatch.undo()
    assert _session(langs, res, tmp_path, tmp_path / "c2", tmp_path / "c3") == 1
    assert "stop rule fired in session 2" in _summary(res, 3)["stopped"]


def test_sessions_made_with_another_plan_are_refused(langs, chain, tmp_path):
    res = tmp_path / "res"
    shutil.copytree(chain / "res", res)
    f = res / "session-2" / "summary.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    d["setup"]["warmup"] = 3
    f.write_text(json.dumps(d), encoding="utf-8")
    assert _session(langs, res, tmp_path, chain / "c2", tmp_path / "c3") == 1
    assert not (res / "session-3").exists()
