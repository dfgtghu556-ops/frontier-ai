"""EXP-042: the IsoFLOP fits (synthetic data with a known answer) and the multi-session ladder (CPU smoke)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gpu_lr_arch.py"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ladder_fit  # noqa: E402
from test_gpu_lr_arch import langs  # noqa: E402,F401  (the shared 3-language fixture)

# ---------------------------------------------------------------- fits on synthetic data


def _parabola(n_opt: float, k: float = 0.2, floor: float = 1.0):
    return lambda n: floor + k * (math.log10(n) - math.log10(n_opt)) ** 2


def test_fit_budget_finds_a_known_minimum_and_flat_region():
    f = _parabola(3e6)
    pts = [(n, f(n)) for n in (1e6, 2e6, 5e6, 1e7)]
    r = ladder_fit.fit_budget(pts, noise=0.0174)
    assert r["bracketed"] is True
    assert r["n_opt"] == pytest.approx(3e6, rel=1e-6)
    assert r["loss_at_opt"] == pytest.approx(1.0, abs=1e-9)
    half = math.sqrt(0.0174 / 0.2)
    lo, hi = r["flat_region"]
    assert lo == pytest.approx(3e6 / 10**half, rel=1e-6) and hi == pytest.approx(3e6 * 10**half, rel=1e-6)
    assert set(r["flat_sizes"]) == {n for n, y in pts if y <= 1.0 + 0.0174}


def test_fit_budget_refuses_edges_and_too_few_points():
    f = _parabola(3e6)
    edge = ladder_fit.fit_budget([(n, f(n)) for n in (3e6, 5e6, 1e7, 2e7)], 0.0174)
    assert edge["bracketed"] is False and "edge" in edge["reason"]
    outside = ladder_fit.fit_budget([(n, f(n)) for n in (1e7, 2e7, 4e7)], 0.0174)
    assert outside["bracketed"] is False
    two = ladder_fit.fit_budget([(1e6, 1.2), (2e6, 1.1)], 0.0174)
    assert two["bracketed"] is False and "need 3" in two["reason"]
    down = ladder_fit.fit_budget([(1e6, 1.0), (2e6, 1.2), (4e6, 1.0)], 0.0174)
    assert down["bracketed"] is False and "downwards" in down["reason"]


def test_fit_power_recovers_the_exponent():
    cs = [1e16, 3e16, 1e17]
    r = ladder_fit.fit_power(cs, [2.0 * c**0.5 for c in cs])
    assert r["exponent"] == pytest.approx(0.5, abs=1e-9)
    assert ladder_fit.fit_power([1e16], [1.0])["computable"] is False


def test_growth_law_projection_matches_the_closed_form():
    # N_opt = c^0.5 / 20 and D_opt = C / (6 N_opt): D_opt = 20/6 * C^0.5; tokens T -> C* = (6T/20)^2
    budgets = []
    for name, c in (("C1", 1e16), ("C2", 3e16), ("C3", 1e17)):
        n = c**0.5 / 20
        budgets.append({"name": name, "C": c, "bracketed": True, "n_opt": n, "d_opt": c / (6 * n)})
    t = 2.78e9
    g = ladder_fit.growth_law(budgets, t)
    assert g["computable"] and g["a"] == pytest.approx(0.5) and g["b"] == pytest.approx(0.5)
    pj = g["projection"]
    assert pj["C_star"] == pytest.approx((6 * t / 20) ** 2, rel=1e-6)
    assert pj["n_opt_non_embedding"] == pytest.approx(pj["C_star"] ** 0.5 / 20, rel=1e-6)
    assert pj["orders_of_magnitude_beyond_largest_budget"] == pytest.approx(math.log10(pj["C_star"] / 1e17))
    lo, hi = pj["leave_one_out_n_opt"]
    assert lo == pytest.approx(hi, rel=1e-6)  # exact data: every subset gives the same answer
    one = ladder_fit.growth_law(budgets[:1] + [dict(budgets[1], bracketed=False)], t)
    assert one["computable"] is False and "at least 2" in one["reason"]


def test_parametric_fit_recovers_known_constants():
    e, a, al, b, be = 1.5, 400.0, 0.34, 4000.0, 0.28
    pts = []
    for n in (5e5, 4e6, 1.3e7, 3e7, 6e7):
        for d in (2e8, 6e8, 1.2e9):
            pts.append((n, d, e + a / n**al + b / d**be))
    r = ladder_fit.fit_parametric(pts)
    assert r["converged"] is True
    assert r["alpha"] == pytest.approx(al, abs=0.011) and r["beta"] == pytest.approx(be, abs=0.011)
    assert r["E"] == pytest.approx(e, abs=0.05) and r["rmse"] < 1e-3
    assert ladder_fit.fit_parametric(pts[:4])["converged"] is False


def test_nnls_never_returns_negative_coefficients():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.random((20, 3))
    y = x @ np.array([1.0, -2.0, 0.5])
    coef = ladder_fit._nnls3(x, y)
    assert (coef >= 0).all()


# ---------------------------------------------------------------- the ladder plan


def test_full_setup_matches_the_pre_registered_table():
    import gpu_lr_arch as g

    s = g.ladder_setup(False)
    want = {  # EXP-042 proposal: total and non-embedding parameters
        "s1": (4_735_616, 524_928),
        "s2": (12_355_840, 3_934_464),
        "s3": (25_613_184, 12_981_120),
        "s4": (47_260_160, 30_417_408),
        "s5": (80_049_280, 58_995_840),
    }
    assert {k: (v["n_params"], v["n_non_embedding"]) for k, v in s["sizes"].items()} == want
    assert s["tokens_per_step"] == 16_384
    assert s["sizes"]["s5"]["micro_batch"] * s["sizes"]["s5"]["accum"] * 512 == 16_384
    assert all(v["model"]["n_kv_head"] == 2 and v["model"]["pos"] == "rope" for v in s["sizes"].values())
    plan = g.ladder_plan([], s)
    assert len(plan) == 20 and sum(p["lr"] is not None for p in plan) == 12  # C2/C3 wait for C1
    # the longest run: s2 at C3, about 1.24 B tokens
    c3s2 = next(p for p in plan if p["name"] == "C3-s2")
    assert c3s2["steps"] * 16_384 == pytest.approx(1.24e9, rel=0.01)


def test_best_lr_waits_for_c1_and_blocks_a_size_with_no_good_run():
    import gpu_lr_arch as g

    s = g.ladder_setup(True)
    runs = []
    for lr in g.LADDER_LRS:
        runs.append(
            {
                "name": g.ladder_run_name("C1", "s1", lr),
                "budget": "C1",
                "size": "s1",
                "lr": lr,
                "status": "done",
                "failed": False,
                "bpb_mean": {1e-3: 2.0, 2e-3: 1.9}.get(lr, 2.1),
                "attempted": True,
            }
        )
        runs.append(
            {
                "name": g.ladder_run_name("C1", "s2", lr),
                "budget": "C1",
                "size": "s2",
                "lr": lr,
                "status": "failed: non-finite loss",
                "failed": True,
                "attempted": True,
            }
        )
    best = g.ladder_best_lrs(runs, s)
    assert best["s1"] == 2e-3 and best["s2"] == "none" and best["s3"] is None
    plan = {p["name"]: p for p in g.ladder_plan(runs, s)}
    assert plan["C2-s1-lr0.002"]["lr"] == 2e-3 and plan["C2-s2"]["blocked"] is True
    assert plan["C2-s3"]["lr"] is None and plan["C2-s3"]["blocked"] is False
    assert plan["C3-s5"]["lr"] is None  # s5 follows s4, which has not finished C1


def test_analysis_on_synthetic_runs_finds_the_planted_optimum():
    import gpu_lr_arch as g

    s = g.ladder_setup(False)
    opt = {"C1": 2.0e6, "C2": 5.0e6, "C3": 1.5e7}  # planted N_opt (non-embedding) per budget
    runs = []
    for budget, c, window in s["budgets"]:
        for size in window:
            sz = s["sizes"][size]
            lrs = g.LADDER_LRS if budget == "C1" else [1e-3]
            for lr in lrs:
                steps = max(1, round(c / sz["flops_per_token"] / s["tokens_per_step"]))
                bpb = (
                    1.0
                    + 0.2 * (math.log10(sz["n_non_embedding"] / opt[budget])) ** 2
                    + (0 if lr == 1e-3 else 0.05)
                )
                runs.append({
                    "name": g.ladder_run_name(budget, size, lr), "budget": budget, "size": size, "lr": lr,
                    "status": "done", "failed": False, "attempted": True, "bpb_mean": bpb,
                    "tokens": steps * s["tokens_per_step"], "tokens_per_s": 1.0, "minutes": 1.0,
                })  # fmt: skip
    plan = g.ladder_plan(runs, s)
    assert all(p["lr"] == 1e-3 for p in plan if p["budget"] != "C1")
    assert next(p for p in plan if p["size"] == "s5")["lr"] == 1e-3  # from s4
    a = g.ladder_analysis(runs, s)
    assert all(v == {"lr": 1e-3, "edge_of_range": False} for v in a["best_lr"].values())
    for b in a["budgets"]:
        assert b["bracketed"] is True and b["n_opt"] == pytest.approx(opt[b["name"]], rel=1e-6)
        assert (
            b["n_opt"] < b["n_opt_total"]
            and b["tokens_per_total_param"] < b["tokens_per_non_embedding_param"]
        )
        assert b["d_opt"] > 0
    gl = a["growth_law"]
    assert gl["computable"] and gl["bracketed_budgets"] == ["C1", "C2", "C3"]
    assert gl["projection"]["computable"] and "leave_one_out_n_opt" in gl["projection"]
    text = g.render_ladder({"exp_id": "EXP-042", "session": 3, "complete": True, "started_at": "-",
                            "runs": runs, "analysis": a, "setup": s})  # fmt: skip
    assert "EXTRAPOLATION" in text and "per parameter counting embeddings" in text


# ---------------------------------------------------------------- CPU smoke over two sessions


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=1500)


def test_ladder_smoke_continues_across_sessions(langs, tmp_path):  # noqa: F811
    out = tmp_path / "EXP-042"
    common = [
        "--smoke", "--skip-tests", "--part", "ladder", "--exp-id", "EXP-042",
        "--data-dir", str(langs), "--manifest", str(langs / "manifest.json"),
        "--out", str(out), "--prev-dir", str(out),
    ]  # fmt: skip
    res = _run([*common, "--scratch", str(tmp_path / "a"), "--max-runs", "5"])
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    s1 = json.loads((out / "session-1" / "summary.json").read_text(encoding="utf-8"))
    assert s1["schema"] == "frontier-ladder-v1" and s1["session"] == 1 and s1["complete"] is False
    assert s1["part0"]["files_ok"] == 3 and s1["part0"]["one_step_check"]["pass"] is None  # CPU = CPU
    assert len(s1["runs"]) == 5 and len(s1["remaining"]) == 15 and "analysis" not in s1
    assert "Still to run" in (out / "session-1" / "SUMMARY.txt").read_text(encoding="utf-8")

    res = _run([*common, "--scratch", str(tmp_path / "b")])
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    s2 = json.loads((out / "session-2" / "summary.json").read_text(encoding="utf-8"))
    assert s2["session"] == 2 and s2["complete"] is True and s2["remaining"] == []
    assert s2["hours_before"] == pytest.approx(s1["elapsed_min"] / 60)
    runs = s2["runs"]
    assert len(runs) == 20 and len({r["name"] for r in runs}) == 20  # no run twice
    assert sum(r["session"] == 1 for r in runs) == 5
    assert [r for r in runs if r["session"] == 1] == s1["runs"]  # earlier runs carried over unchanged
    for r in runs:
        assert r["status"] == "done", r
        assert r["tokens"] == r["steps"] * s2["setup"]["tokens_per_step"]
        assert set(r["bpb"]) == {"hi", "en", "ta"} and math.isfinite(r["bpb_mean"])
    s5 = next(r for r in runs if r["size"] == "s5")
    best_s4 = s2["analysis"]["best_lr"]["s4"]["lr"]
    assert s5["lr"] == best_s4  # pre-registered: s5 uses s4's best learning rate
    a = s2["analysis"]
    assert [b["name"] for b in a["budgets"]] == ["C1", "C2", "C3"]
    assert all(len(b["sizes_used"]) == 4 for b in a["budgets"])
    assert "growth_law" in a and "parametric" in a and a["failed_runs"] == []
    text = (out / "session-2" / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "SMOKE TEST" in text and "Pre-registered analysis" in text

    # a third launch does nothing: the ladder is complete
    res = _run([*common, "--scratch", str(tmp_path / "c")])
    s3 = json.loads((out / "session-3" / "summary.json").read_text(encoding="utf-8"))
    assert res.returncode == 0 and s3["complete"] is True and "already completed" in s3["stopped"]
    assert s3.get("runs", []) == []


def test_ladder_stops_when_the_total_cap_is_used(langs, tmp_path):  # noqa: F811
    out = tmp_path / "EXP-042"
    common = [
        "--smoke", "--skip-tests", "--part", "ladder", "--exp-id", "EXP-042",
        "--data-dir", str(langs), "--manifest", str(langs / "manifest.json"),
        "--out", str(out), "--prev-dir", str(out), "--scratch", str(tmp_path / "s"),
    ]  # fmt: skip
    res = _run([*common, "--total-cap-hours", "0"])
    assert res.returncode == 1
    s = json.loads((out / "session-1" / "summary.json").read_text(encoding="utf-8"))
    assert "cap" in s["stopped"] and s["complete"] is False and s.get("runs", []) == []


def test_ladder_refuses_sessions_from_another_plan(langs, tmp_path):  # noqa: F811
    out = tmp_path / "EXP-042"
    (out / "session-1").mkdir(parents=True)
    (out / "session-1" / "summary.json").write_text(
        json.dumps({"schema": "frontier-ladder-v1", "smoke": True, "session": 1, "setup": {"other": 1}}),
        encoding="utf-8",
    )
    res = _run(
        [
            "--smoke",
            "--skip-tests",
            "--part",
            "ladder",
            "--data-dir",
            str(langs),
            "--manifest",
            str(langs / "manifest.json"),
            "--out",
            str(out),
            "--prev-dir",
            str(out),
            "--scratch",
            str(tmp_path / "s"),
        ]  # fmt: skip
    )
    assert res.returncode == 1
    assert "different ladder plan" in res.stdout
