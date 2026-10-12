"""EXP-047 section 4 refit (approved 2026-10-12, "approve refit"): CPU only, no training.

Checks the committed ``evals/results/EXP-047/refit/`` against the code and the recorded inputs: the fit
without EXP-043 is exactly EXP-042's recorded fit, the primary refit is reproduced, the post-hoc calibration
hits EXP-043 exactly, the FLOP-match size is the one EXP-047 section 1 used, and REFIT.txt is rendered from
refit.json. (The full recomputation, with leave-one-out, is ``python scripts/refit_exp047.py --check``.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ladder_fit  # noqa: E402
import plot_ladder  # noqa: E402
import refit_exp047 as rf  # noqa: E402

FIT_KEYS = ("E", "A", "B", "alpha", "beta", "rmse")


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads((rf.OUT_DIR / "refit.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def points() -> list:
    return [(n, d, y) for n, d, y, _ in rf.ladder_points(plot_ladder.final_summary(rf.EXP042_DIR))]


def test_inputs(committed, points):
    assert len(points) == 20 == committed["inputs"]["n_ladder_points"]
    e = rf.exp043_point()
    assert e == committed["inputs"]["exp043"]
    assert e["n_params"] == 190_064_000 and e["n_non_embedding"] == 160_589_184
    assert (
        e["tokens"] == 2_783_821_824
        and round(e["bpb_mean"], 4) == 0.6283
        and e["predicted_by_exp042"] == 0.683
    )


def test_without_exp043_is_exp042s_recorded_fit(committed, points):
    f = ladder_fit.fit_parametric(points)
    rec = plot_ladder.final_summary(rf.EXP042_DIR)["analysis"]["parametric"]
    for k in FIT_KEYS:
        assert f[k] == pytest.approx(rec[k], rel=1e-12)
        assert committed["fit_without_exp043"][k] == pytest.approx(rec[k], rel=1e-12)


def test_primary_refit_reproduced(committed, points):
    e = rf.exp043_point()
    f = ladder_fit.fit_parametric(points + [(e["n_non_embedding"], e["tokens"], e["bpb_mean"])])
    for k in FIT_KEYS:
        assert committed["fit_refit_primary"][k] == pytest.approx(f[k], rel=1e-12)
    c = committed["exp043_check"]
    assert c["pred_refit"] == pytest.approx(rf._pred(f, e["n_non_embedding"], e["tokens"]))
    assert c["refit_miss"] == pytest.approx(e["bpb_mean"] - c["pred_refit"])
    assert c["refit_reproduces_within_noise"] == (abs(c["refit_miss"]) <= 0.0174)


def test_rows_consistent(committed):
    fit = committed["fit_refit_primary"]
    miss = committed["exp043_check"]["refit_miss"]
    gate = committed["gate"]["threshold"]
    assert gate == pytest.approx(committed["inputs"]["exp043"]["bpb_mean"] - 0.0174)
    assert round(gate, 4) == 0.6109
    sp = committed["speed_basis_exp043"]
    assert sp == rf.exp043_speed()
    f896 = plot_ladder.family(896)["flops_per_token"]
    assert [r["width"] for r in committed["rows"]] == list(rf.WIDTHS)
    for r in committed["rows"]:
        p = rf._pred(fit, r["n_non_embedding"], rf.TOKENS_EXP047)
        assert r["pred_refit"] == pytest.approx(p)
        assert r["pred_calibrated_posthoc"] == pytest.approx(p + miss)
        lo, hi = r["pred_refit_loo_range"]
        assert lo <= p <= hi
        assert r["beats_gate_refit"] == (p < gate)
        assert r["t4"] == pytest.approx(rf.t4_hours(r["flops_per_token"] / f896, sp))
    # the calibration reproduces EXP-043 at its own size and tokens, by construction
    e = committed["inputs"]["exp043"]
    assert rf._pred(fit, e["n_non_embedding"], e["tokens"]) + miss == pytest.approx(e["bpb_mean"])
    assert len(committed["sensitivity"]["leave_one_out"]) == 20


def test_flop_match_unchanged(committed):
    fm = committed["flop_match_unchanged"]
    assert fm["C_star"] == pytest.approx(3.07e19, rel=0.01)  # EXP-047 section 1
    assert fm["n_opt_total"] == pytest.approx(568e6, rel=0.01)
    assert (fm["nearest_below"], fm["nearest_above"]) == ("20 x 1280", "22 x 1408")
    above = [r["size"] for r in committed["rows"] if r["above_flop_match_size"]]
    assert above == ["22 x 1408"]


def test_report_rendered_from_json(committed):
    assert (rf.OUT_DIR / "REFIT.txt").read_text(encoding="utf-8") == rf.render(committed)
