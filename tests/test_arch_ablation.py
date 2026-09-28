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


def test_spec_fingerprint_ignores_windows_line_endings(tmp_path):
    raw = (REPO_ROOT / "configs" / "ablations" / "EXP-032.json").read_bytes().replace(b"\r\n", b"\n")
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(raw)
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert ra.spec_sha256(lf) == ra.spec_sha256(crlf)
    # EXP-032: the PC recorded the CRLF raw-byte hash of the committed spec
    import hashlib
    assert hashlib.sha256(crlf.read_bytes()).hexdigest().startswith("2a7048f290ff2221")


# ------------------------------------------------------------------ v2 (EXP-033) --

def test_real_exp033_spec_is_valid_and_plans_the_approved_grid():
    from frontier_ai.config import ExperimentConfig
    from frontier_ai.engine import checkpoint as ckpt

    spec = ra.load_spec(REPO_ROOT / "configs" / "ablations" / "EXP-033.json")
    assert spec["schema"] == "frontier-arch-ablation-v2" and spec["lrs"] == [0.006, 0.01]
    cells = ra.plan_cells_v2(spec, Path("/tmp/x"))
    assert len(cells) == 18 and len({c["id"] for c in cells}) == 18
    groups = [c["group"] for c in cells]
    for arm in ("baseline", "rope-gqa2", "rope-gqa2-gelu"):
        for lr in ("0.006", "0.01"):
            assert groups.count(f"{arm}-lr-{lr}") == 3
    assert groups[:3] == ["baseline-lr-0.006"] * 3  # the first learning rate is completed first
    assert all(c["overrides"][-1] == f"optim.lr={c['lr']}" for c in cells)
    assert [(c["a"], c["b"]) for c in spec["comparisons"]] == [("baseline", "rope-gqa2"),
                                                               ("rope-gqa2", "rope-gqa2-gelu")]
    assert spec["reuse"] == [dict(spec["reuse"][0], arm="baseline", lr=0.006, seed=1337,
                                  run_dir="out/arch/EXP-032/runs/lr-0.006/seed-1337")]
    # GELU (ffn_mult 6) is exactly parameter-matched to its SwiGLU reference arm
    base = ExperimentConfig.load(REPO_ROOT / spec["base_config"]).with_overrides(["model.vocab_size=32768"])
    n = {a["name"]: sum(p.numel() for p in ckpt.build_model_from_config(
        base.with_overrides(ra._overrides(a["overrides"])), torch.device("cpu")).parameters())
        for a in spec["arms"]}
    assert n == {"baseline": 5_260_416, "rope-gqa2": 5_178_496, "rope-gqa2-gelu": 5_178_496}


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s["arms"][1]["overrides"].update({"optim.lr": 0.1}), "optim.lr"),
    (lambda s: s["arms"][1].update(name=s["arms"][0]["name"]), "duplicate"),
    (lambda s: s["arms"][1].update(overrides={"model.nope": 1}), "invalid override"),
    (lambda s: s["comparisons"][0].update(b="nope"), "two different arms"),
    (lambda s: s["comparisons"][0].update(b="baseline"), "two different arms"),
    (lambda s: s["comparisons"][0].update(rule="non_inferiority"), "superiority"),
    (lambda s: s.update(lrs=[0.006, 0.006]), "lrs"),
    (lambda s: s.update(lrs=[]), "lrs"),
    (lambda s: s["reuse"][0].update(lr=0.003), "reuse"),
    (lambda s: s.update(checkpoint="best"), "last"),
    (lambda s: s.pop("comparisons"), "comparisons"),
])
def test_v2_spec_validation_rejects_bad_specs(tmp_path, mutate, message):
    spec = json.loads((REPO_ROOT / "configs" / "ablations" / "EXP-033.json").read_text(encoding="utf-8"))
    mutate(spec)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ra.SpecError, match=message):
        ra.load_spec(p)


def test_per_learning_rate_rule_and_divergence():
    ok3, inf = ["ok"] * 3, float("inf")
    a, b = [1.40, 1.41, 1.42], [1.30, 1.31, 1.32]
    assert ra.decide_at_lr(ok3, ok3, a, b, [-0.1, -0.05])[0] == "BETTER"
    assert ra.decide_at_lr(ok3, ok3, b, a, [0.05, 0.1])[0] == "WORSE"
    assert ra.decide_at_lr(ok3, ok3, a, [1.30, 1.31, 1.45], [-0.1, -0.01])[0].startswith("NO DETECTABLE")
    div = ["ok", "DIVERGED", "ok"]
    assert ra.decide_at_lr(ok3, div, a, [1.3, inf, 1.3], None)[0] == "WORSE"
    assert ra.decide_at_lr(div, ok3, [1.4, inf, 1.4], b, None)[0] == "BETTER"
    assert ra.decide_at_lr(div, div, a, b, None)[0] == "UNSTABLE"
    assert ra.decide_at_lr(ok3, ["ok", "MISSING", "ok"], a, b, None)[0] == "INCOMPLETE"
    assert ra.decide_at_lr(ok3, ok3, a, b, None)[0] == "INCOMPLETE"


@pytest.mark.parametrize("verdicts, decision, basis", [
    (["BETTER", "BETTER"], "ADOPT", "every"),
    (["BETTER", "NO DETECTABLE DIFFERENCE AT THIS SCALE"], "NOT ADOPTED", ""),
    (["BETTER", "WORSE"], "NOT ADOPTED", ""),
    (["BETTER", "UNSTABLE"], "ADOPT", "weaker evidence"),
    (["UNSTABLE", "UNSTABLE"], "NOT ADOPTED", "unstable"),
    (["BETTER", "INCOMPLETE"], "INCOMPLETE", ""),
])
def test_adoption_needs_better_at_every_learning_rate(verdicts, decision, basis):
    per_lr = {str(i): {"verdict": v} for i, v in enumerate(verdicts)}
    got = ra.adoption(per_lr, len(verdicts))
    assert got[0] == decision and basis in got[1]


def test_v2_e2e_divergence_is_a_result_restart_and_reuse(env):
    work = env.tmp / "work_v2"
    work.mkdir()
    spec = {k: env.spec[k] for k in ("base_config", "data", "tokenizer", "max_steps", "seeds", "eval_args")}
    spec.update(schema="frontier-arch-ablation-v2", exp_id="EXP-089", checkpoint="last",
                lrs=[0.003, 1e30],  # 1e30 makes every arm blow up -> UNSTABLE at that learning rate
                arms=[{"name": "baseline", "overrides": {}}, {"name": "rope", "overrides": {"model.pos": "rope"}}],
                comparisons=[{"name": "rope-vs-baseline", "a": "baseline", "b": "rope", "rule": "superiority"}])
    out = env.tmp / "out_v2"
    spec_path = _write_spec(env.tmp / "spec_v2.json", spec)
    proc = _ablate(spec_path, out, cwd=work)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.stdout.count("DIVERGED baseline-lr-1e+30") == 2 and proc.stdout.count("DIVERGED rope-lr-1e+30") == 2
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["failures"] == []
    rows = {r["group"]: r for r in s["rows"]}
    assert rows["rope-lr-1e+30"]["status"] == ["DIVERGED", "DIVERGED"] and rows["rope-lr-1e+30"]["mean"] is None
    assert all(v is not None for v in rows["rope-lr-0.003"]["bpb"])
    c = s["comparisons"]["rope-vs-baseline"]
    assert c["per_lr"]["1e+30"]["verdict"] == "UNSTABLE"
    assert c["per_lr"]["0.003"]["verdict"] in ("BETTER", "WORSE", "NO DETECTABLE DIFFERENCE AT THIS SCALE")
    assert c["per_lr"]["0.003"]["delta_bpb"] == pytest.approx(
        rows["rope-lr-0.003"]["mean"] - rows["baseline-lr-0.003"]["mean"], abs=1e-9)
    assert c["decision"] == ("ADOPT" if c["per_lr"]["0.003"]["verdict"] == "BETTER" else "NOT ADOPTED")
    d = json.loads((out / "runs" / "rope-lr-1e+30" / "seed-1" / "DIVERGED.json").read_text(encoding="utf-8"))
    assert d["step"] >= 1
    assert json.loads((out / "experiment.json").read_text(encoding="utf-8"))["execution"]["status"] == "success"
    text = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "DECISION:" in text and "UNSTABLE" in text and "DIVERGED" in text

    # restart: nothing retrained (diverged cells included) or regraded; same decision
    proc = _ablate(spec_path, out, "--no-record", cwd=work)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "TRAIN " not in proc.stdout and "GRADE " not in proc.stdout
    assert proc.stdout.count("DIVERGED earlier, not retrained") == 4
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["comparisons"] == s["comparisons"]

    # reuse of a single verified cell from another experiment (dry run)
    spec2 = dict(spec, exp_id="EXP-088", lrs=[0.003],
                 reuse=[{"arm": "baseline", "lr": 0.003, "seed": 1,
                         "run_dir": str(out / "runs" / "baseline-lr-0.003" / "seed-1")}])
    proc = _ablate(_write_spec(env.tmp / "spec_v2b.json", spec2), env.tmp / "out_v2b", "--dry-run", cwd=work)
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.count("config verified") == 1 and "3 to train" in proc.stdout


def test_unattended_night_script_static_safety():
    """The PC runs this script with no agent watching it, so check its safety properties statically."""
    import re

    raw = (REPO_ROOT / "scripts" / "run_night_unattended.ps1").read_bytes()
    raw.decode("ascii")  # Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page
    text = raw.decode("ascii")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert '$branch = "arena/01a0dc16-frontier-ai"' in code
    assert "--force" not in code and " -f " not in code and "reset --hard" not in code
    assert "Remove-Item" not in code and "git clean" not in code and "git checkout" not in code
    # every git / python call goes through the stderr-safe cmd.exe helper (EXP-029 lesson)
    for line in code.splitlines():
        if "Stop-Night" in line or "Add-Report" in line:  # messages only
            continue
        if re.search(r"\bgit (status|add|commit|push|reset|diff|rev-parse)\b|\$python -u", line):
            assert "Invoke-Logged" in line, line
    assert '& cmd.exe /d /c "$cmdline 2>&1"' in code
    assert "git push origin $branch" in code and code.count("Invoke-Logged \"git push") == 1
    assert "git commit -q -F $msgFile" in code  # no quoted -m through cmd.exe
    assert 'git add $resultsPrefix' in code and '$resultsPrefix = "evals/results/$Exp/"' in code
