"""EXP-047 section 4 step 1: refit the size laws with EXP-043 added, then recompute the section 1 table.

Approved by the founder 2026-10-12 ("approve refit"). CPU only, no training, no GPU, no data reads beyond the
committed result files.

What it does (pre-registered in EXP-047 section 4: "the laws are refit with EXP-043 added as a fourth point,
and the table above is recomputed before choosing"):

1. **Primary refit.** The same secondary fit as EXP-042 (``ladder_fit.fit_parametric``: loss = E + A/N^alpha
   + B/D^beta, same grid, same NNLS) on the same 20 successful EXP-042 runs, plus EXP-043's final result as
   one more (non-embedding N, tokens, bpb) point. Both use the same score: the mean over 13 languages of the
   full validation split's bits per byte.
2. **What cannot change.** The growth laws (N_opt and D_opt against compute) come from three bracketed
   IsoFLOP budgets. EXP-043 is one run at one size, so it brings no new optimum; the FLOP-match size at
   8.16 B tokens is recomputed only to show it is unchanged.
3. **Sensitivity (reported, not pre-registered):** the fit without EXP-043 (must reproduce EXP-042's recorded
   fit exactly), leave-one-out over the 20 ladder runs, only the best learning rate per (budget, size), and a
   post-hoc *calibrated* prediction (the primary refit shifted by its own miss at EXP-043).
4. **Cost columns:** 2xT4 session-hours from EXP-043's measured whole-run speed (sessions 1-6), scaled by
   FLOPs per token; H100 hours from stated assumptions (NOT measured).

Output: ``evals/results/EXP-047/refit/refit.json`` and ``REFIT.txt``.

    python scripts/refit_exp047.py            # write the files
    python scripts/refit_exp047.py --check    # recompute and compare with the committed refit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

import ladder_fit  # noqa: E402
import plot_ladder  # noqa: E402

EXP042_DIR = ROOT / "evals" / "results" / "EXP-042"
EXP043_DIR = ROOT / "evals" / "results" / "EXP-043"
OUT_DIR = ROOT / "evals" / "results" / "EXP-047" / "refit"

TOKENS_EXP047 = 8_159_265_692  # v2-slice1 + v2-slice2, one pass (EXP-047 section 5)
STEPS_EXP047 = 498_001  # pretrain_exp047.EXPECTED_STEPS (32 windows x 512 tokens per step)
TOKENS_PER_STEP = 32 * 512
WIDTHS = (896, 1024, 1152, 1280, 1408)  # the EXP-047 section 1 family
GATE_NOISE = 0.0174  # EXP-043 seed noise; gate = EXP-043 final - noise (EXP-047 note 2026-10-12)
OPTIONS = {  # EXP-047 section 3 (free quota weeks NOT VERIFIED)
    896: "A (free)",
    1024: "A (free)",
    1152: "A (free)",
    1280: "B (free + sharded optimizer, new code) or C (rented H100)",
    1408: "C (rented H100) only",
}
STATE_GB = {1024: 4.4, 1152: 6.0, 1280: 8.1, 1408: 10.6}  # EXP-047 section 2: 16 bytes/param (estimate)
FREE_HOURS_PER_WEEK = 30.0  # Kaggle GPU quota, NOT VERIFIED
H100 = {
    "peak_bf16_flops": 989e12,
    "mfu_range": (0.30, 0.50),
    "usd_per_hour": (0.90, 3.99),
    "inr_per_usd": 88.0,
}


def _pred(f: dict[str, Any], n: float, d: float) -> float:
    return f["E"] + f["A"] / n ** f["alpha"] + f["B"] / d ** f["beta"]


def _fit_view(f: dict[str, Any]) -> dict[str, Any]:
    return {k: f.get(k) for k in ("E", "A", "B", "alpha", "beta", "rmse", "implied_a", "converged")}


def ladder_points(s: dict[str, Any]) -> list[tuple[float, float, float, str]]:
    """(non-embedding N, tokens, bpb_mean, run name): exactly the runs EXP-042's analysis used."""
    sizes = s["setup"]["sizes"]
    good = [r for r in s["runs"] if r.get("status") == "done" and not r.get("failed")]
    return [(sizes[r["size"]]["n_non_embedding"], r["tokens"], r["bpb_mean"], r["name"]) for r in good]


def exp043_point() -> dict[str, Any]:
    s6 = json.loads((EXP043_DIR / "session-6" / "summary.json").read_text(encoding="utf-8"))
    fam = plot_ladder.family(s6["setup"]["model"]["n_embd"])
    n_total = s6["final"]["model_final"]["n_params"]
    if n_total != fam["n_params"]:  # EXP-043's model is the family member of its width
        raise SystemExit(f"EXP-043 has {n_total} parameters, family says {fam['n_params']}")
    return {
        "source": "evals/results/EXP-043/session-6/summary.json",
        "n_params": n_total,
        "n_non_embedding": fam["n_non_embedding"],
        "tokens": s6["main"]["tokens_seen"],
        "bpb_mean": s6["final"]["scores"]["bpb_mean"],
        "predicted_by_exp042": s6["setup"]["predicted_bpb"],
    }


def exp043_speed() -> dict[str, Any]:
    """Measured EXP-043 session timings (190 M model): the basis of the 2xT4 hours column."""
    s = {
        n: json.loads((EXP043_DIR / f"session-{n}" / "summary.json").read_text(encoding="utf-8"))
        for n in range(1, 7)
    }
    tok = mins = wall = 0.0
    for n in (3, 4, 5, 6):  # two-GPU sessions (D-050)
        m = s[n]["main"]
        tok += (m["end_step"] - m["start_step"]) * TOKENS_PER_STEP
        mins += m["train_minutes"]
        wall += s[n]["elapsed_min"]
    m2 = s[2]["main"]
    return {
        "lr_check_session_min": s[1]["elapsed_min"],  # 3 x 6,104 steps on one GPU, evaluations included
        "one_gpu_session_min": s[2]["elapsed_min"],
        "one_gpu_train_min": m2["train_minutes"],
        "one_gpu_tokens_per_s": m2["tokens_per_s"],
        "two_gpu_tokens_per_s": tok / (mins * 60),
        "two_gpu_wall_over_train": wall / mins,
        "total_session_hours": sum(x["elapsed_min"] for x in s.values()) / 60,
    }


def t4_hours(ratio: float, sp: dict[str, Any]) -> dict[str, float]:
    """Session-hours on Kaggle 2xT4 for one EXP-047 run at FLOPs/token = ratio x EXP-043's.

    Same flow as EXP-043: learning-rate check (one GPU), first main session on one GPU, the rest on two.
    Assumes speed scales as 1/ratio (same hardware utilisation as the 190 M model) - NOT VERIFIED for bigger
    models (memory may force smaller micro-batches)."""
    lr_h = sp["lr_check_session_min"] * ratio / 60
    first_tokens = sp["one_gpu_tokens_per_s"] / ratio * sp["one_gpu_train_min"] * 60
    rest_tokens = STEPS_EXP047 * TOKENS_PER_STEP - first_tokens
    rest_h = rest_tokens / (sp["two_gpu_tokens_per_s"] / ratio) / 3600 * sp["two_gpu_wall_over_train"]
    total = lr_h + sp["one_gpu_session_min"] / 60 + rest_h
    return {
        "lr_check_h": lr_h,
        "main_h": total - lr_h,
        "total_h": total,
        "weeks_free": total / FREE_HOURS_PER_WEEK,
    }


def h100(train_flops: float, lr_flops: float) -> dict[str, Any]:
    lo_mfu, hi_mfu = H100["mfu_range"]
    hours = [(train_flops + lr_flops) / (m * H100["peak_bf16_flops"]) / 3600 for m in (hi_mfu, lo_mfu)]
    usd = [hours[0] * H100["usd_per_hour"][0], hours[1] * H100["usd_per_hour"][1]]
    return {"hours": hours, "usd": usd, "inr": [u * H100["inr_per_usd"] for u in usd]}


def compute() -> dict[str, Any]:
    s = plot_ladder.final_summary(EXP042_DIR)
    lp = ladder_points(s)
    pts = [(n, d, y) for n, d, y, _ in lp]
    e43 = exp043_point()
    p43 = (e43["n_non_embedding"], e43["tokens"], e43["bpb_mean"])
    n43, d43, y43 = p43

    recorded = s["analysis"]["parametric"]
    orig = ladder_fit.fit_parametric(pts)
    refit = ladder_fit.fit_parametric(pts + [p43])
    miss = y43 - _pred(refit, n43, d43)

    # best learning rate per (budget, size), plus EXP-043 (sensitivity)
    best: dict[tuple[str, str], tuple[float, float, float, str]] = {}
    for r, p in zip([r for r in s["runs"] if r.get("status") == "done" and not r.get("failed")], lp):
        k = (r["budget"], r["size"])
        if k not in best or p[2] < best[k][2]:
            best[k] = p
    best_fit = ladder_fit.fit_parametric([(n, d, y) for n, d, y, _ in best.values()] + [p43])

    # leave-one-out over the 20 ladder runs (EXP-043 always kept)
    loo = []
    for i in range(len(pts)):
        f = ladder_fit.fit_parametric([p for j, p in enumerate(pts) if j != i] + [p43])
        loo.append({"dropped": lp[i][3], "fit": _fit_view(f), "exp043_pred": _pred(f, n43, d43)})

    # FLOP-match size at 8.16 B tokens: growth laws only (unchanged by EXP-043)
    ph = plot_ladder.posthoc({**s, "tokens_available": TOKENS_EXP047})
    gl = ph["total_parameter_growth_law"]
    fm = ph["family_at_C_star"]

    sp = exp043_speed()
    gate = y43 - GATE_NOISE
    fam896 = plot_ladder.family(896)
    rows = []
    for w in WIDTHS:
        f = plot_ladder.family(w)
        n, fpt = f["n_non_embedding"], f["flops_per_token"]
        prim = _pred(refit, n, TOKENS_EXP047)
        loo_p = [_pred(x["fit"], n, TOKENS_EXP047) for x in loo]
        train_flops = fpt * TOKENS_EXP047
        lr_flops = 3 * 6_104 * TOKENS_PER_STEP * fpt
        rows.append(
            {
                "size": f"{w // 64} x {w}",
                "width": w,
                "layers": w // 64,
                "n_params": f["n_params"],
                "n_non_embedding": n,
                "flops_per_token": fpt,
                "train_flops": train_flops,
                "tokens_per_param": TOKENS_EXP047 / f["n_params"],
                "pred_exp042_fit": _pred(orig, n, TOKENS_EXP047),
                "pred_refit": prim,
                "pred_refit_loo_range": [min(loo_p), max(loo_p)],
                "pred_best_lr_refit": _pred(best_fit, n, TOKENS_EXP047),
                "pred_calibrated_posthoc": prim + miss,
                "beats_gate_refit": prim < gate,
                "beats_gate_calibrated_posthoc": prim + miss < gate,
                "above_flop_match_size": fpt > fm["flops_per_token_needed"],
                "optimizer_states_gb_est": STATE_GB.get(w),
                "t4": t4_hours(fpt / fam896["flops_per_token"], sp),
                "h100_est": h100(train_flops, lr_flops),
                "option": OPTIONS[w],
            }
        )

    return {
        "experiment": "EXP-047",
        "what": "section 4 step 1 refit (approved 2026-10-12, 'approve refit'); CPU only, no training",
        "inputs": {
            "exp042_summary": "evals/results/EXP-042/session-3/summary.json",
            "n_ladder_points": len(pts),
            "exp043": e43,
            "metric": "mean over 13 languages of full-validation bits per byte (same in EXP-042 and EXP-043)",
            "tokens_exp047": TOKENS_EXP047,
        },
        "fit_recorded_exp042": _fit_view(recorded),
        "fit_without_exp043": _fit_view(orig),
        "fit_refit_primary": _fit_view(refit),
        "exp043_check": {
            "actual": y43,
            "pred_exp042_fit": _pred(orig, n43, d43),
            "pred_refit": _pred(refit, n43, d43),
            "refit_miss": miss,
            "noise": GATE_NOISE,
            "refit_reproduces_within_noise": abs(miss) <= GATE_NOISE,
        },
        "same_model_more_data": {
            "size": "14 x 896 (EXP-043's model)",
            "already_reached_on_2.78B": y43,
            "pred_refit_on_8.16B": _pred(refit, n43, TOKENS_EXP047),
            "pred_calibrated_on_8.16B": _pred(refit, n43, TOKENS_EXP047) + miss,
        },
        "sensitivity": {
            "best_lr_only_plus_exp043": {
                **_fit_view(best_fit),
                "n_runs": len(best),
                "exp043_pred": _pred(best_fit, n43, d43),
            },
            "leave_one_out": loo,
            "calibrated_posthoc_note": "primary refit + its miss at EXP-043 (a constant shift); "
            "post-hoc, not pre-registered; assumes the miss is the same at every size",
        },
        "flop_match_unchanged": {
            "note": "growth laws come from 3 bracketed IsoFLOP budgets; "
            "EXP-043 (one run, one size) adds no optimum",
            "C_star": gl["C_star"],
            "n_opt_total": gl["n_opt_total_at_C_star"],
            "leave_one_out_n_opt_total": gl["leave_one_out_n_opt_total"],
            "nearest_below": f"{fm['nearest_below']['layers']} x {fm['nearest_below']['width']}",
            "nearest_above": f"{fm['nearest_above']['layers']} x {fm['nearest_above']['width']}",
        },
        "gate": {
            "threshold": gate,
            "rule": "final slice-1 mean < EXP-043 final - 0.0174; no language worse by > 0.0174",
        },
        "speed_basis_exp043": sp,
        "assumptions": {
            "free_hours_per_week": f"{FREE_HOURS_PER_WEEK} (NOT VERIFIED)",
            "t4_speed": "scales as 1 / FLOPs-per-token from EXP-043's measured runs "
            "(NOT VERIFIED for larger sizes)",
            "h100": {
                **H100,
                "note": "NOT measured; prices seen 2026-10, NOT VERIFIED now; bf16 path not built",
            },
        },
        "rows": rows,
    }


def _r(x: float, k: int = 4) -> float:
    return round(x, k)


def render(d: dict[str, Any]) -> str:
    c = d["exp043_check"]
    fo, fr = d["fit_without_exp043"], d["fit_refit_primary"]
    m = d["same_model_more_data"]
    fmz = d["flop_match_unchanged"]
    lo_n, hi_n = fmz["leave_one_out_n_opt_total"]
    lines = [
        "EXP-047 section 4 refit (approved 2026-10-12; CPU only, no training)",
        "",
        "1. The fit: loss = E + A/N^alpha + B/D^beta (EXP-042's secondary fit, same grid, same runs)",
        f"   EXP-042's 20 runs:          E {fo['E']:.4f}  A {fo['A']:.2f}  B {fo['B']:.1f}  "
        f"alpha {fo['alpha']:.2f}  "
        f"beta {fo['beta']:.2f}  rmse {fo['rmse']:.4f}",
        f"   20 runs + EXP-043 (primary): E {fr['E']:.4f}  A {fr['A']:.2f}  B {fr['B']:.1f}  "
        f"alpha {fr['alpha']:.2f}  "
        f"beta {fr['beta']:.2f}  rmse {fr['rmse']:.4f}",
        "",
        f"2. Check at EXP-043 (actual {c['actual']:.4f}): old fit {c['pred_exp042_fit']:.4f}, refit "
        f"{c['pred_refit']:.4f} -> miss {c['refit_miss']:+.4f} "
        f"({'within' if c['refit_reproduces_within_noise'] else 'OUTSIDE'} seed noise {c['noise']})",
        f"   Same 14 x 896 model on 8.16 B tokens: refit {m['pred_refit_on_8.16B']:.4f} "
        f"(vs {m['already_reached_on_2.78B']:.4f} already reached on 2.78 B), calibrated "
        f"{m['pred_calibrated_on_8.16B']:.4f}",
        "",
        f"3. FLOP-match size at 8.16 B tokens (growth laws, unchanged): C* {fmz['C_star']:.3g}, N_opt "
        f"{fmz['n_opt_total'] / 1e6:.0f} M total (leave-one-out {lo_n / 1e6:.0f} M - {hi_n / 1e9:.1f} B); "
        f"between {fmz['nearest_below']} and {fmz['nearest_above']}",
        "",
        f"4. Table at 8,159,265,692 tokens (gate: below {d['gate']['threshold']:.4f})",
        f"   {'size':<10}{'params':>8}{'old':>8}{'refit':>8}{'LOO range':>16}{'best-lr':>9}{'calib.':>8}"
        f"{'2xT4 h':>8}{'weeks':>7}{'H100 h':>10}  option",
    ]
    for r in d["rows"]:
        lo, hi = r["pred_refit_loo_range"]
        hh = r["h100_est"]["hours"]
        flag = " (above FLOP-match)" if r["above_flop_match_size"] else ""
        lines.append(
            f"   {r['size']:<10}{r['n_params'] / 1e6:>7.0f}M"
            f"{r['pred_exp042_fit']:>8.4f}{r['pred_refit']:>8.4f}"
            f"{lo:>9.4f}-{hi:.4f}{r['pred_best_lr_refit']:>9.4f}{r['pred_calibrated_posthoc']:>8.4f}"
            f"{r['t4']['total_h']:>8.0f}{r['t4']['weeks_free']:>7.1f}{hh[0]:>5.0f}-{hh[1]:.0f}"
            f"  {r['option']}{flag}"
        )
    lines += [
        "",
        "Honest limits: one point (EXP-043) at 10-40x the ladder's largest budget carries the refit;",
        "every row is an extrapolation. 'calib.' is post-hoc (refit shifted by its miss at EXP-043),",
        "not pre-registered.",
        "Free quota 30 h/week, T4 speed scaling, memory and H100 numbers are NOT VERIFIED estimates.",
    ]
    return "\n".join(lines) + "\n"


def _dump(d: dict[str, Any]) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check", action="store_true", help="recompute and compare with the committed refit.json"
    )
    args = ap.parse_args(argv)
    d = compute()
    if args.check:
        old = json.loads((OUT_DIR / "refit.json").read_text(encoding="utf-8"))
        if json.loads(_dump(d)) != old:
            print("refit.json differs from a fresh recomputation")
            return 1
        print("refit.json OK")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "refit.json").write_text(_dump(d), encoding="utf-8")
    (OUT_DIR / "REFIT.txt").write_text(render(d), encoding="utf-8")
    print(render(d), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
