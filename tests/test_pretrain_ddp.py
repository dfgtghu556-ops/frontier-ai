"""D-050: the EXP-043 main run on two processes (CPU smoke, gloo, GPUs hidden).

The module fixture of tests/test_pretrain.py gives session 1 (learning-rate check) and session 2
(first 100 steps, one process). Here session 3 runs with ``--gpus 2``: it must finish the same pass
with the same windows per step, write the same kinds of records as one process, and match the
one-process result up to float reduction order. A worker that fails before training falls back to
one process (bit-identical to a one-process session); one that fails during training keeps the
newest checkpoint; a stop rule on process 0 stops both processes.
"""

# ruff: noqa: F811  (pytest fixtures named like the imported fixtures)
from __future__ import annotations

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
from test_pretrain import _session, _sha, _summary, chain  # noqa: E402, F401


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("FRONTIER_TEST_DDP_FAIL", raising=False)


def _copy_res(chain: Path, dst: Path) -> Path:
    res = dst / "res"
    shutil.copytree(chain / "res", res)
    return res


@pytest.fixture(scope="module")
def one_proc(langs, chain, tmp_path_factory):
    """Session 3 on one process: the reference."""
    base = tmp_path_factory.mktemp("one")
    res = _copy_res(chain, base)
    assert _session(langs, res, base, chain / "c2", base / "c3") == 0
    return base, _summary(res, 3)


@pytest.fixture(scope="module")
def two_proc(langs, chain, tmp_path_factory):
    """Session 3 on two processes."""
    mp = pytest.MonkeyPatch()
    mp.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.setenv("OMP_NUM_THREADS", "1")
    try:
        base = tmp_path_factory.mktemp("two")
        res = _copy_res(chain, base)
        assert _session(langs, res, base, chain / "c2", base / "c3", "--gpus", "2") == 0
    finally:
        mp.undo()
    return base, _summary(res, 3)


def test_two_processes_finish_the_pass_like_one(one_proc, two_proc):
    (b1, s1), (b2, s2) = one_proc, two_proc
    m1, m2 = s1["main"], s2["main"]
    total = s2["order"]["steps_per_pass"]
    assert not s2.get("stopped") and not s2.get("two_gpu_fallback") and not s2.get("two_gpu_failure")
    assert "two CPU processes" in s2["gpus_plan"] and "D-050" in s2["precision"]
    assert (m1["gpus"], m1["micro_per_gpu"]) == (1, [2, 2])
    assert (m2["gpus"], m2["micro_per_gpu"]) == (2, [2, 1])  # the same 4 windows per step
    assert len(m2["peak_mem_gb_per_gpu"]) == 2
    # the same steps, tokens, order position, evaluation points and checkpoints
    for k in ("start_step", "end_step", "session_end", "tokens_seen", "sampler_position", "periodic_saves"):
        assert m2[k] == m1[k], k
    assert (m2["start_step"], m2["end_step"]) == (100, total)
    assert m2["tokens_seen"] == total * 4 * 16 and m2["sampler_position"] == total * 4
    assert [c["step"] for c in m2["val_curve"]] == [c["step"] for c in m1["val_curve"]]
    # only float reduction order differs: the numbers agree closely
    for c1, c2 in zip(m1["val_curve"], m2["val_curve"]):
        assert abs(c1["val_loss"] - c2["val_loss"]) < 1e-3
    f1, f2 = s1["final"]["scores"], s2["final"]["scores"]
    assert abs(f1["bpb_mean"] - f2["bpb_mean"]) < 1e-3
    assert set(f2["bpb"]) == set(TEXT)
    # process 0 wrote the final files and the summary; the summary text shows the GPU count
    assert s2["complete"] is True and set(s2["final"]["checks"]) == set(s1["final"]["checks"])
    samples = (b2 / "res" / "session-3" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(samples) == s2["final"]["samples"] == 3 * 3 * 2
    info = s2["final"]["model_final"]
    assert (
        info["sha256"] == _sha(b2 / "final" / "model_final.pt")
        and info["n_params"] == (s1["final"]["model_final"]["n_params"])
    )
    text = (b2 / "res" / "session-3" / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "trained on 2 GPU(s): 2 x 1 per GPU" in text
    assert "trained on 1 GPU(s): 2 x 2 per GPU" in (b1 / "res" / "session-3" / "SUMMARY.txt").read_text(
        encoding="utf-8"
    )
    # nothing extra lands in the chain output: the same layout as one process
    assert sorted(p.name for p in (b2 / "c3/exp043_chain").iterdir()) == sorted(
        p.name for p in (b1 / "c3/exp043_chain").iterdir()
    )


def test_a_two_process_checkpoint_resumes_on_one_process(langs, chain, one_proc, tmp_path):
    """Session 3 on two processes stops after 50 steps; session 4 (one process) finishes the pass."""
    res = _copy_res(chain, tmp_path)
    assert (
        _session(langs, res, tmp_path, chain / "c2", tmp_path / "c3", "--gpus", "2", "--session-steps", "50")
        == 0
    )
    s3 = _summary(res, 3)
    assert s3["main"]["gpus"] == 2 and s3["main"]["end_step"] == 150
    assert s3["checkpoint"]["step"] == 150 and s3["checkpoint"]["session"] == 3
    assert s3["checkpoint"]["files"]["model.pt"] == _sha(tmp_path / "c3/exp043_chain/last/model.pt")
    assert _session(langs, res, tmp_path, tmp_path / "c3", tmp_path / "c4") == 0
    s4 = _summary(res, 4)
    assert s4["chain_check"]["ok"] and s4["main"]["gpus"] == 1 and s4["main"]["start_step"] == 150
    assert s4["main"]["sampler_position"] == one_proc[1]["main"]["sampler_position"]
    assert s4["complete"] is True
    assert abs(s4["final"]["scores"]["bpb_mean"] - one_proc[1]["final"]["scores"]["bpb_mean"]) < 1e-3


def test_a_failed_two_process_start_falls_back_to_one_process(langs, chain, one_proc, tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTIER_TEST_DDP_FAIL", "start")
    res = _copy_res(chain, tmp_path)
    assert _session(langs, res, tmp_path, chain / "c2", tmp_path / "c3", "--gpus", "2") == 0
    s3 = _summary(res, 3)
    fb = s3["two_gpu_fallback"]
    assert fb["exit_code"] != 0 and "before the first training step" in fb["note"]
    assert set(fb["errors"]) == {"error_rank0", "error_rank1"}
    assert all(v == "RuntimeError: test: the two-GPU start fails" for v in fb["errors"].values())
    assert "FALLBACK" in s3["gpus_plan"] and not s3.get("stopped")
    assert s3["main"]["gpus"] == 1 and s3["complete"] is True
    # bit-identical to a session that ran on one process from the start
    ref = one_proc[1]
    assert s3["final"]["model_final"]["sha256"] == ref["final"]["model_final"]["sha256"]
    assert s3["final"]["scores"]["bpb_mean"] == ref["final"]["scores"]["bpb_mean"]
    for f in ("model.pt", "optimizer.pt", "trainer_state.pt"):  # meta/config hold wall time and paths
        assert s3["checkpoint"]["files"][f] == ref["checkpoint"]["files"][f], f
    assert "TWO-GPU START FAILED" in (res / "session-3" / "SUMMARY.txt").read_text(encoding="utf-8")


def test_a_two_process_failure_during_training_keeps_the_checkpoint(langs, chain, tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTIER_TEST_DDP_FAIL", "step:130")
    res = _copy_res(chain, tmp_path)
    assert _session(langs, res, tmp_path, chain / "c2", tmp_path / "c3", "--gpus", "2") == 1
    s3 = _summary(res, 3)
    assert "two-GPU training failed after it started" in s3["stopped"]
    errs = s3["two_gpu_failure"]["errors"]
    assert errs and all("test: the two-GPU worker fails at step 130" in v for v in errs.values())
    assert s3["main"]["periodic_saves"] == [125] and s3.get("complete") is not True
    ck = s3["checkpoint"]
    assert ck["step"] == 125 and ck["session"] == 3
    assert ck["files"]["model.pt"] == _sha(tmp_path / "c3/exp043_chain/last/model.pt")
    assert "TWO-GPU RUN FAILED" in (res / "session-3" / "SUMMARY.txt").read_text(encoding="utf-8")


def test_a_stop_rule_on_process_0_stops_both_processes(langs, chain, tmp_path):
    res = _copy_res(chain, tmp_path)
    rc = _session(langs, res, tmp_path, chain / "c2", tmp_path / "c3", "--gpus", "2", "--test-stop-at", "120")
    s3 = _summary(res, 3)
    assert s3["stop_rule"] == "test stop rule at step 120" and s3["main"]["session_end"] == "stop rule"
    assert s3["main"]["gpus"] == 2 and s3["main"]["end_step"] == 120
    assert not s3.get("two_gpu_failure") and "session_eval" not in s3["main"]
    assert rc == 1 or "STOP RULE FIRED" in (res / "session-3" / "SUMMARY.txt").read_text(encoding="utf-8")


def test_one_gpu_is_the_default_and_no_setup_key_was_added():
    """Earlier sessions are refused when their setup differs: the GPU count must not be a setup key."""
    assert "gpus" not in gp.SETUP_FULL and "gpus" not in gp.SETUP_SMOKE
    assert gp.main.__defaults__ == (None,)
    args = type("A", (), {"gpus": 1, "smoke": False})()
    assert gp.gpus_for_main(args, {"checkpoint": {"step": 1}}) == (1, "one GPU (--gpus 1)")
    # D-050: no main-run checkpoint yet (the first main-run session) -> one GPU even with --gpus 2
    two = type("A", (), {"gpus": 2, "smoke": True})()
    assert gp.gpus_for_main(two, {"checkpoint": None})[0] == 1
    assert gp.gpus_for_main(two, {"checkpoint": {"step": 1}})[0] == 2


def test_the_first_main_run_session_stays_on_one_gpu_with_gpus_2(langs, chain, tmp_path):
    """Session 2 (no main-run checkpoint yet) with --gpus 2 is the one-GPU session, bit for bit."""
    res = tmp_path / "res"
    res.mkdir()
    shutil.copytree(chain / "res" / "session-1", res / "session-1")
    assert (
        _session(langs, res, tmp_path, chain / "c1", tmp_path / "c2", "--gpus", "2", "--session-steps", "100")
        == 0
    )
    s2, ref = _summary(res, 2), _summary(chain / "res", 2)
    assert s2["main"]["gpus"] == 1 and "no checkpoint yet" in s2["gpus_plan"]
    assert not (tmp_path / "scratch" / gp.DDP_DIR).exists()  # no worker was started
    for f in ("model.pt", "optimizer.pt", "trainer_state.pt"):
        assert s2["checkpoint"]["files"][f] == ref["checkpoint"]["files"][f], f
