"""EXP-044 (step 13): two-process data-parallel training on CPU (gloo), checked before any GPU time.

1. The sampler: two processes together read exactly the windows one process reads.
2. Two processes vs one with the same global batch, in float64 (the D-047 method): every step's loss
   and the final weights agree to 1e-10.
3. Checkpoints cross: saved on two processes, resumed on one (and the other way round), the result
   equals the uninterrupted run.
4. A stop rule raised on process 0 stops both processes; nothing is saved.
5. The whole check (``scripts/gpu_ddp_check.py --smoke``) runs and writes its verdict.
"""

# ruff: noqa: F811  (pytest fixtures named like the imported `langs` fixture)
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import gpu_ddp_check as dc  # noqa: E402
from frontier_ai.data.multi import MultiTokenDataset, OnePassDataset  # noqa: E402
from frontier_ai.engine import distributed as dd  # noqa: E402
from test_gpu_lr_arch import langs  # noqa: E402, F401  (the 3-language fixture)

TOL = 1e-10


def _run(langs: Path, out: Path, procs: int, micro: int, accum: int, *extra: str) -> dict:
    """One worker run (1 process, or `procs` processes via torchrun); returns its worker.json."""
    import argparse

    args = argparse.Namespace(data_dir=str(langs), manifest=str(langs / "manifest.json"), smoke=True)
    cmd = dc.worker_cmd(args, procs, micro, accum, out) + list(extra)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}  # CPU + gloo, also on Kaggle's GPU machine (Part 0)
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300, env=env)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return json.loads((out / "worker.json").read_text(encoding="utf-8"))


def _weights(out: Path) -> dict[str, torch.Tensor]:
    return torch.load(out / "weights.pt", map_location="cpu", weights_only=True)


def _max_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    assert set(a) == set(b)
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


F64 = ("--float64", "--save-weights", "--no-eval", "--steps", "12")


def test_two_shards_read_exactly_the_one_process_windows(langs):
    base = MultiTokenDataset(sorted(langs.glob("*.bin")))
    one = OnePassDataset(base, 16, seed=1)
    shards = [OnePassDataset(base, 16, seed=1) for _ in range(2)]
    for r, s in enumerate(shards):
        s.set_shard(r, 2)
    dev = torch.device("cpu")
    for _ in range(5):
        x1, y1 = one.next_train_batch(4, 16, dev)
        parts = [s.next_train_batch(2, 16, dev) for s in shards]
        assert torch.equal(x1, torch.cat([p[0] for p in parts])) and torch.equal(
            y1, torch.cat([p[1] for p in parts])
        )
        assert one.position == shards[0].position == shards[1].position
    with pytest.raises(ValueError):
        one.set_shard(2, 2)


def test_single_format_keeps_the_one_gpu_checkpoint_keys():
    lin = torch.nn.Linear(2, 2)
    plain = dd.SingleFormat(lin, compiled=False).state_dict()
    comp = dd.SingleFormat(lin, compiled=True).state_dict()
    assert set(plain) == {"weight", "bias"} and set(comp) == {"_orig_mod.weight", "_orig_mod.bias"}
    other = torch.nn.Linear(2, 2)
    dd.SingleFormat(other, compiled=False).load_state_dict(
        {"_orig_mod.module." + k: v for k, v in plain.items()}
    )
    assert torch.equal(other.weight, lin.weight)
    assert dd.strip_prefixes({"module._orig_mod.a": 1}) == {"a": 1}
    # without a process group everything is one process
    assert (dd.rank(), dd.world(), dd.is_main(), dd.all_true(True), dd.run_on_main(lambda: 7)) == (
        0,
        1,
        True,
        True,
        7,
    )


@pytest.fixture(scope="module")
def single12(langs, tmp_path_factory):
    out = tmp_path_factory.mktemp("single12")
    return out, _run(langs, out, 1, 2, 2, *F64)


def test_two_processes_equal_one_process_in_float64(langs, single12, tmp_path):
    out1, one = single12
    two = _run(langs, tmp_path, 2, 2, 1, *F64)
    assert one["world"] == 1 and two["world"] == 2 and one["steps"] == two["steps"] == 12
    assert len(one["losses"]) == len(two["losses"]) == 12
    assert max(abs(a - b) for a, b in zip(one["losses"], two["losses"])) <= TOL
    assert _max_diff(_weights(out1), _weights(tmp_path)) <= TOL
    assert one["tokens_seen"] == two["tokens_seen"] == 12 * 4 * 16
    assert one["sampler_position"] == two["sampler_position"] == 12 * 4
    # also with accumulation on each of the two processes (1 x 2 per process)
    acc = _run(langs, tmp_path / "acc", 2, 1, 2, *F64)
    assert acc["world"] == 2 and acc["accum"] == 2
    assert _max_diff(_weights(out1), _weights(tmp_path / "acc")) <= TOL


@pytest.mark.parametrize("first,second", [(2, 1), (1, 2)])
def test_checkpoints_cross_between_one_and_two_processes(langs, single12, tmp_path, first, second):
    out1, _ = single12
    a, b = tmp_path / "a", tmp_path / "b"
    shape = {1: (2, 2), 2: (2, 1)}
    r1 = _run(langs, a, first, *shape[first], *F64, "--session-steps", "6")
    assert r1["steps"] == 6 and r1["end"] == "session step limit (test)"
    keys = set(torch.load(a / "ckpt" / "last" / "model.pt", map_location="cpu", weights_only=True))
    assert not any(k.startswith(("module.", "_orig_mod.")) for k in keys)  # the one-process CPU format
    r2 = _run(langs, b, second, *shape[second], *F64, "--resume", str(a / "ckpt"))
    assert r2["start_step"] == 6 and r2["steps"] == 12 and r2["sampler_position"] == 12 * 4
    assert _max_diff(_weights(out1), _weights(b)) <= TOL


def test_a_stop_rule_on_process_zero_stops_both(langs, tmp_path):
    r = _run(langs, tmp_path, 2, 2, 1, "--no-eval", "--steps", "12", "--raise-at", "5")
    assert r["stop_rule"] == "test stop rule at step 5" and r["steps"] == 5
    assert not (tmp_path / "ckpt" / "last").exists()  # a stop rule ends without saving


def test_the_whole_check_runs_on_cpu(langs, tmp_path):
    code = dc.main(
        [
            "--smoke",
            "--skip-tests",
            "--data-dir",
            str(langs),
            "--manifest",
            str(langs / "manifest.json"),
            "--out",
            str(tmp_path / "res"),
            "--scratch",
            str(tmp_path / "scratch"),
        ]
    )
    d = json.loads((tmp_path / "res" / "summary.json").read_text(encoding="utf-8"))
    assert code == 0 and d["complete"] and d["schema"] == "frontier-ddp-check-v1" and d["smoke"]
    assert (d["one"]["world"], d["one"]["micro"], d["one"]["accum"]) == (1, 2, 2)
    assert (d["two"]["world"], d["two"]["micro"], d["two"]["accum"]) == (2, 2, 1)
    v = d["verdict"]
    assert v["agreement_pass"] and v["result"].startswith("SMOKE (no verdict)")
    assert abs(d["one"]["scores"]["bpb_mean"] - d["two"]["scores"]["bpb_mean"]) < 1e-4
    assert np.allclose(d["one"]["losses"], d["two"]["losses"], atol=1e-4)
    assert d["one"]["tokens_per_s"] > 0 and d["two"]["tokens_per_s"] > 0
    text = (tmp_path / "res" / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "Pre-registered rules" in text and "VERDICT" in text


def test_full_setup_is_the_preregistered_one():
    s = dc.SETUP_FULL
    assert s["model"]["n_layer"] == 14 and s["model"]["n_embd"] == 896 and s["model"]["n_kv_head"] == 2
    assert (s["steps"], s["speed_from_step"], s["windows_per_step"]) == (300, 100, 32)
    assert (s["agree_bpb"], s["min_speedup"], s["mem_limit_gb"], s["max_skipped"]) == (0.01, 1.4, 14.5, 0.05)
    assert s["micro_options_one"] == [[16, 2], [8, 4]] and s["gpus"] == 2 and s["order_seed"] == 1
    one = {"scores": {"bpb_mean": 1.000}, "skipped_steps": 0, "tokens_per_s": 100.0, "peak_mem_gb": [10.0]}
    good = {
        "scores": {"bpb_mean": 1.009},
        "skipped_steps": 3,
        "tokens_per_s": 141.0,
        "peak_mem_gb": [10.1, 10.0],
    }
    assert dc.verdict(one, good, s, smoke=False)["pass"]
    slow = {**good, "tokens_per_s": 139.0}
    assert not dc.verdict(one, slow, s, smoke=False)["pass"]
    apart = {**good, "scores": {"bpb_mean": 1.011}}
    assert not dc.verdict(one, apart, s, smoke=False)["agreement_pass"]
    big = {**good, "peak_mem_gb": [14.6, 10.0]}
    assert not dc.verdict(one, big, s, smoke=False)["memory_pass"]
    skips = {**good, "skipped_steps": 16}
    assert not dc.verdict(one, skips, s, smoke=False)["agreement_pass"]
