"""Step 9 architecture ablation runner: spec validation, reuse checks, e2e train → grade → decide."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from test_exp_b import _build_frontier, _run_script, _train_two_tokenizers
from test_tokenizer_sweep import _make_fake_frozen_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_arch_ablation", REPO_ROOT / "scripts" / "run_arch_ablation.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


def test_real_exp032_spec_is_valid_and_plans_the_approved_grid():
    spec = ra.load_spec(REPO_ROOT / "configs" / "ablations" / "EXP-032.json")
    cells = ra.plan_cells(spec, Path("/tmp/x"))
    groups = [c["group"] for c in cells]
    assert groups.count("baseline") == 3
    assert [v["name"] for v in spec["variants"]] == ["rope", "gelu", "layernorm", "gqa2"]
    assert all(groups.count(v) == 3 for v in ("rope", "gelu", "layernorm", "gqa2"))
    assert "lr-0.0015" in groups and "lr-0.006" in groups and groups.count("repro-baseline") == 1
    assert len(cells) == 18 and len({c["id"] for c in cells}) == 18
    assert spec["checkpoint"] == "last" and spec["max_steps"] == 150
    gqa = next(v for v in spec["variants"] if v["name"] == "gqa2")
    assert gqa["rule"] == "non_inferiority" and gqa["margin_bpb"] == 0.010


def test_gelu_variant_is_exactly_parameter_matched():
    from frontier_ai.config import ExperimentConfig
    from frontier_ai.engine import checkpoint as ckpt

    spec = ra.load_spec(REPO_ROOT / "configs" / "ablations" / "EXP-032.json")
    base = ExperimentConfig.load(REPO_ROOT / spec["base_config"]).with_overrides(["model.vocab_size=32768"])
    gelu = base.with_overrides(ra._overrides(spec["variants"][1]["overrides"]))
    n = [sum(p.numel() for p in ckpt.build_model_from_config(c, torch.device("cpu")).parameters())
         for c in (base, gelu)]
    assert n[0] == n[1] == 5_260_416


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s["variants"][0].update(rule="bogus"), "rule"),
    (lambda s: s["variants"][0].update(name="baseline"), "reserved"),
    (lambda s: s["variants"][0].update(overrides={"model.nope": 1}), "invalid override"),
    (lambda s: s["variants"][1].update(name=s["variants"][0]["name"]), "duplicate"),
    (lambda s: s.update(checkpoint="best"), "last"),
    (lambda s: s["variants"][3].pop("margin_bpb"), "margin"),
    (lambda s: s["lr_check"].update(seed=99), "seed"),
])
def test_spec_validation_rejects_bad_specs(tmp_path, mutate, message):
    spec = json.loads((REPO_ROOT / "configs" / "ablations" / "EXP-032.json").read_text(encoding="utf-8"))
    mutate(spec)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ra.SpecError, match=message):
        ra.load_spec(p)


def test_superiority_rule():
    assert ra.superiority([-0.03, -0.01], [1.50, 1.51, 1.52], [1.47, 1.48, 1.49]) == "BETTER"
    assert ra.superiority([0.01, 0.03], [1.47, 1.48, 1.49], [1.50, 1.51, 1.52]) == "WORSE"
    # CI excludes 0 but the seeds overlap -> not enough
    assert ra.superiority([-0.03, -0.01], [1.48, 1.51, 1.52], [1.47, 1.49, 1.50]).startswith("NO DETECTABLE")
    # seeds separate but the CI includes 0 -> not enough
    assert ra.superiority([-0.03, 0.01], [1.50, 1.51, 1.52], [1.47, 1.48, 1.49]).startswith("NO DETECTABLE")


def test_weights_compare(tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    for d in (a, b, c):
        d.mkdir()
    sd = {"w": torch.arange(6.0).reshape(2, 3), "b": torch.zeros(3)}
    torch.save(sd, a / "model.pt")
    torch.save({k: v.clone() for k, v in sd.items()}, b / "model.pt")
    torch.save({"w": sd["w"] + 1e-6, "b": sd["b"]}, c / "model.pt")
    assert ra.weights_compare(a, b)["status"] == "IDENTICAL"
    diff = ra.weights_compare(a, c)
    assert diff["status"] == "DIFFERENT" and diff["unequal_tensors"] == 1 and diff["max_abs_diff"] > 0


# ------------------------------------------------------------------------ e2e --

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("arch")
    fake = _make_fake_frozen_corpus(tmp / "fake", n_hi=30, n_en=12)
    _build_frontier(fake, tmp / "frontier")
    (small, small_art), (big, big_art) = _train_two_tokenizers(tmp / "frontier", tmp)
    data_dir = tmp / "data"
    corpus_args = ["--frontier-dir", str(tmp / "frontier"), "--manifest", str(fake["manifest"]),
                   "--freeze", str(fake["freeze"]), "--corpus-dir", str(fake["corpus_dir"])]
    proc = _run_script("prepare_exp_b_data.py", [*corpus_args, "--tokens", f"{small}={small_art}",
                                                 "--out", str(data_dir), "--no-record"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    suite = tmp / "suite" / "SUITE.json"
    proc = _run_script("build_eval_suite.py", ["--out", str(suite), *corpus_args])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    base_cfg = json.loads((REPO_ROOT / "configs" / "exp_b.json").read_text(encoding="utf-8"))
    base_cfg["model"].update(n_layer=1, n_head=2, n_embd=32, block_size=32)
    base_cfg["data"].update(batch_size=4)
    base_cfg["optim"].update(warmup_steps=2)
    base_cfg["train"].update(max_steps=6, accum_steps=1, eval_interval=3, eval_iters=2, log_interval=3,
                             device="cpu", num_threads=1)
    cfg_path = tmp / "tiny_exp_b.json"
    cfg_path.write_text(json.dumps(base_cfg), encoding="utf-8")
    spec = {
        "schema": "frontier-arch-ablation-v1", "exp_id": "EXP-094", "base_config": str(cfg_path),
        "data": str(data_dir / f"{small}.bin"), "tokenizer": str(small_art), "max_steps": 6,
        "seeds": [1, 2], "checkpoint": "last",
        "baseline": {"name": "baseline", "overrides": {}},
        "variants": [
            {"name": "rope", "overrides": {"model.pos": "rope"}, "rule": "superiority"},
            {"name": "mqa", "overrides": {"model.n_kv_head": 1}, "rule": "non_inferiority", "margin_bpb": 0.5},
        ],
        "lr_check": {"seed": 1, "values": [0.0015]},
        "repro_check": {"seed": 1},
        "eval_args": ["--no-contamination", "--bootstrap", "50", "--suite", str(suite), *corpus_args],
    }
    return type("Env", (), {"tmp": tmp, "spec": spec})


def _write_spec(path: Path, spec: dict) -> Path:
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _ablate(spec_path: Path, out: Path, *extra: str, cwd: Path | None = None):
    return subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_arch_ablation.py"),
                           "--spec", str(spec_path), "--out", str(out), *extra],
                          capture_output=True, text=True, timeout=1500, cwd=cwd)


def test_e2e_train_grade_decide_restart_and_reuse(env):
    work = env.tmp / "work"
    work.mkdir()
    spec1 = _write_spec(env.tmp / "spec1.json", env.spec)
    out1 = env.tmp / "out1"
    proc = _ablate(spec1, out1, cwd=work)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    s = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))
    groups = {r["group"]: r for r in s["rows"]}
    assert set(groups) == {"baseline", "rope", "mqa", "lr-0.0015", "repro-baseline"}
    assert all(v is not None for r in s["rows"] for v in r["bpb"])
    assert groups["rope"]["body_params"] == groups["baseline"]["body_params"]
    assert groups["mqa"]["kv_cache_values_per_token"] == groups["baseline"]["kv_cache_values_per_token"] // 2
    for name in ("rope", "mqa"):
        v = s["variants"][name]
        assert v["verdict"] != "INCOMPLETE" and len(v["ci95"]) == 2
        # the verdict's delta is the difference of the group means
        assert v["delta_bpb"] == pytest.approx(groups[name]["mean"] - groups["baseline"]["mean"], abs=1e-9)
    assert s["variants"]["mqa"]["verdict"] in ("ACCEPTABLE", "NOT SHOWN")
    assert s["lr_check"]["verdict"] in ("LR-CONFOUNDED", "NOT CONFOUNDED")
    # deterministic CPU training: retraining the baseline reproduces its weights exactly
    assert s["repro_check"]["status"] == "IDENTICAL", s["repro_check"]
    assert s["repro_check"]["bpb_reference"] == s["repro_check"]["bpb_retrained"]
    # the final checkpoint was graded (not 'best')
    rep = json.loads((out1 / "eval" / "rope-seed-1" / "report.json").read_text(encoding="utf-8"))
    assert rep["checkpoint"]["dir"].replace("\\", "/").endswith("rope/seed-1/last")
    assert json.loads((out1 / "experiment.json").read_text(encoding="utf-8"))["execution"]["status"] == "success"
    assert (out1 / "SUMMARY.txt").read_text(encoding="utf-8").startswith("ARCHITECTURE ABLATION EXP-094")

    # restart: nothing is retrained or regraded, and the summary is unchanged
    proc = _ablate(spec1, out1, "--no-record", cwd=work)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "TRAIN " not in proc.stdout and "GRADE " not in proc.stdout
    assert proc.stdout.count("already trained, skipping") == 8
    assert json.loads((out1 / "summary.json").read_text(encoding="utf-8"))["variants"] == s["variants"]

    # reuse: a new experiment reuses the verified baseline runs instead of retraining them
    spec2 = dict(env.spec, exp_id="EXP-093",
                 baseline={"name": "baseline", "overrides": {}, "reuse_runs_dir": str(out1 / "runs" / "baseline")},
                 variants=[env.spec["variants"][0]], lr_check=None, repro_check=None)
    out2 = env.tmp / "out2"
    proc = _ablate(_write_spec(env.tmp / "spec2.json", spec2), out2, "--no-record", cwd=work)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert proc.stdout.count("config verified") == 2
    s2 = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))
    b2 = next(r for r in s2["rows"] if r["group"] == "baseline")
    assert b2["bpb"] == groups["baseline"]["bpb"] and b2["reused_from"]

    # a reuse candidate with a different budget is refused and would be retrained (dry run)
    spec3 = dict(spec2, exp_id="EXP-092", max_steps=5)
    proc = _ablate(_write_spec(env.tmp / "spec3.json", spec3), env.tmp / "out3", "--dry-run", cwd=work)
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.count("NOT reused") == 2 and "max_steps" in proc.stdout
    assert not (env.tmp / "out3" / "runs").exists()


def test_missing_data_is_an_input_error(env, tmp_path):
    spec = dict(env.spec, data=str(tmp_path / "nope.bin"))
    proc = _ablate(_write_spec(tmp_path / "s.json", spec), tmp_path / "o", "--dry-run")
    assert proc.returncode == 2 and "data file not found" in proc.stderr
