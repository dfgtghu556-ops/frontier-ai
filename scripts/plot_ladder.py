"""EXP-042 rule 6: the IsoFLOP plot, plus the post-hoc checks written down in the Results block.

Reads the final committed session summary (the one with ``"complete": true`` and the analysis) and
writes, next to the session folders:

* ``isoflop.svg``: left, validation bits per byte against non-embedding parameters, one curve per
  budget (the pre-registered parabola fits, the measured points, N_opt marked); right, N_opt and
  D_opt against compute with the pre-registered power laws, and the extrapolated point at the
  budget where D_opt equals our 2.78 B training tokens (dashed: an extrapolation).
* ``posthoc.json``: checks that were NOT pre-registered and are labelled as such: the growth law
  with total parameters instead of non-embedding parameters, the model of the same family whose
  FLOPs per token match the extrapolated budget, the secondary parametric fit evaluated at 2.78 B
  tokens, and the T4 hours implied by the measured speed.

Plain SVG text (no plotting library), numpy for the fits. Usage:

    python scripts/plot_ladder.py [--dir evals/results/EXP-042]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import ladder_fit  # noqa: E402

COLORS = {"C1": "#1f77b4", "C2": "#2ca02c", "C3": "#d62728"}
VOCAB = 32896
HEAD = 64
CONTEXT = 512


def final_summary(d: Path) -> dict[str, Any]:
    done = []
    for f in d.glob("session-*/summary.json"):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("complete") and s.get("analysis"):
            done.append(s)
    if len(done) != 1:
        raise SystemExit(f"expected exactly one complete session with the analysis in {d}, found {len(done)}")
    return done[0]


def best_points(s: dict[str, Any]) -> dict[str, list[tuple[float, float, str]]]:
    """Per budget: (non-embedding N, best bits per byte, size) over the successful runs."""
    sizes = s["setup"]["sizes"]
    out: dict[str, list[tuple[float, float, str]]] = {}
    for budget, _c, window in s["setup"]["budgets"]:
        pts = []
        for size in window:
            rs = [
                r
                for r in s["runs"]
                if r["budget"] == budget
                and r["size"] == size
                and r["status"] == "done"
                and not r.get("failed")
            ]
            if rs:
                pts.append((sizes[size]["n_non_embedding"], min(r["bpb_mean"] for r in rs), size))
        out[budget] = pts
    return out


def family(width: int) -> dict[str, float]:
    """Parameters and FLOPs per token of the EXP-042 family at this width (depth = width / 64)."""
    import gpu_bringup as gb
    from frontier_ai.config import ModelConfig
    from frontier_ai.model.gpt import GPT

    mc = ModelConfig(
        vocab_size=VOCAB,
        n_layer=width // HEAD,
        n_head=width // HEAD,
        n_embd=width,
        block_size=CONTEXT,
        pos="rope",
        n_kv_head=2,
        dropout=0.0,
        norm="rmsnorm",
        ffn="swiglu",
        tie_embeddings=True,
    )
    import torch

    with torch.device("meta"):  # only the parameter count is needed
        n = GPT(mc).n_params()
    return {
        "width": width,
        "layers": width // HEAD,
        "n_params": n,
        "n_non_embedding": n - VOCAB * width,
        "flops_per_token": gb.flops_per_token(mc, n),
    }


def posthoc(s: dict[str, Any]) -> dict[str, Any]:
    a = s["analysis"]
    br = [b for b in a["budgets"] if b.get("bracketed")]
    tokens = s.get("tokens_available", 2_783_830_088)
    out: dict[str, Any] = {
        "label": "POST-HOC checks, not pre-registered (EXP-042 Results block); the pre-registered "
        "analysis is in the final session's summary.json",
    }
    # 1. growth law with TOTAL parameters (Pearce & Song 2024; Porian et al. 2024)
    cs = [b["C"] for b in br]
    tot = ladder_fit.fit_power(cs, [b["n_opt_total"] for b in br])
    dl = ladder_fit.fit_power(cs, [b["d_opt"] for b in br])
    c_star = (tokens / dl["k"]) ** (1 / dl["exponent"])
    loo = []
    for drop in range(len(br)):
        keep = [b for i, b in enumerate(br) if i != drop]
        t = ladder_fit.fit_power([b["C"] for b in keep], [b["n_opt_total"] for b in keep])
        d = ladder_fit.fit_power([b["C"] for b in keep], [b["d_opt"] for b in keep])
        c = (tokens / d["k"]) ** (1 / d["exponent"])
        loo.append(t["k"] * c ** t["exponent"])
    out["total_parameter_growth_law"] = {
        "a_total": tot["exponent"],
        "b": dl["exponent"],
        "a_total_plus_b": tot["exponent"] + dl["exponent"],
        "a_non_embedding_plus_b": a["growth_law"]["a"] + a["growth_law"]["b"],
        "residuals_log10": tot.get("residuals_log10"),
        "C_star": c_star,
        "n_opt_total_at_C_star": tot["k"] * c_star ** tot["exponent"],
        "leave_one_out_n_opt_total": [min(loo), max(loo)],
        "tokens_per_total_param_at_C_star": tokens / (tot["k"] * c_star ** tot["exponent"]),
    }
    # 2. which model of the same family spends C* on 2.78 B tokens
    fpt = c_star / tokens
    fam = [family(w) for w in range(640, 2049, 128)]
    below = max((f for f in fam if f["flops_per_token"] <= fpt), key=lambda f: f["flops_per_token"])
    above = min((f for f in fam if f["flops_per_token"] >= fpt), key=lambda f: f["flops_per_token"])
    out["family_at_C_star"] = {"flops_per_token_needed": fpt, "nearest_below": below, "nearest_above": above}
    # 3. the secondary parametric fit at D = 2.78 B tokens (an extrapolation of a secondary fit)
    pf = a["parametric"]
    if pf.get("converged"):
        rows = []
        for w in (640, 768, 896, 1024, 1152, 1280):
            f = family(w)
            loss = pf["E"] + pf["A"] / f["n_non_embedding"] ** pf["alpha"] + pf["B"] / tokens ** pf["beta"]
            rows.append({**f, "predicted_bpb_mean": loss, "train_flops": f["flops_per_token"] * tokens})
        out["parametric_at_all_tokens"] = rows
    # 4. T4 hours from the measured speed (largest size, evaluations included)
    s5 = next(r for r in s["runs"] if r["size"] == max(s["setup"]["sizes"]) and r["status"] == "done")
    rate = s5["C_actual"] / (s5["minutes"] * 60)
    out["t4_flops_per_s_measured"] = {"run": s5["name"], "flops_per_s": rate}
    out["t4_hours_for_C_star"] = c_star / rate / 3600
    return out


# ------------------------------------------------------------------------------------- SVG --


def _svg(s: dict[str, Any], ph: dict[str, Any]) -> str:
    a = s["analysis"]
    pts = best_points(s)
    w, h = 1100, 490
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w / 2}" y="22" text-anchor="middle" font-size="15" font-weight="bold">'
        "EXP-042: IsoFLOP ladder on 13 Indian languages (one Kaggle T4, seed 1)</text>",
    ]
    # ---- left panel: isoFLOP profiles
    x0, y0, pw, ph_ = 70, 50, 440, 360
    lx = (math.log10(3e5), math.log10(1e8))
    ly = (0.75, 1.05)

    def px(n: float) -> float:
        return x0 + (math.log10(n) - lx[0]) / (lx[1] - lx[0]) * pw

    def py(v: float) -> float:
        return y0 + ph_ - (v - ly[0]) / (ly[1] - ly[0]) * ph_

    parts.append(_axes(x0, y0, pw, ph_))
    for e in range(6, 9):
        for m in (1, 3) if e < 8 else (1,):
            n = m * 10**e
            if lx[0] <= math.log10(n) <= lx[1]:
                parts.append(_xtick(px(n), y0 + ph_, f"{n / 1e6:g} M"))
    for v in np.arange(0.75, 1.0501, 0.05):
        parts.append(_ytick(x0, py(v), f"{v:.2f}"))
    parts.append(
        f'<text x="{x0 + pw / 2}" y="{y0 + ph_ + 36}" text-anchor="middle">'
        "non-embedding parameters (log scale)</text>"
    )
    parts.append(
        f'<text transform="translate({x0 - 48},{y0 + ph_ / 2}) rotate(-90)" text-anchor="middle">'
        "validation bits per byte (mean of 13 languages)</text>"
    )
    for b in a["budgets"]:
        col = COLORS.get(b["name"], "black")
        p = pts[b["name"]]
        if b.get("coef"):
            xs = np.linspace(math.log10(p[0][0]), math.log10(p[-1][0]), 60)
            ys = np.polyval(b["coef"], xs)
            path = " ".join(
                f"{'M' if i == 0 else 'L'}{px(10**x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys))
            )
            parts.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        for n, v, size in p:
            parts.append(f'<circle cx="{px(n):.1f}" cy="{py(v):.1f}" r="4" fill="{col}"/>')
            parts.append(
                f'<text x="{px(n) + 6:.1f}" y="{py(v) - 6:.1f}" font-size="10" fill="{col}">{size}</text>'
            )
        if b.get("n_opt"):
            parts.append(
                f'<path d="M{px(b["n_opt"]):.1f},{py(b["loss_at_opt"]) - 9:.1f} l-6,-10 h12 z" fill="{col}"/>'
            )
    ly0 = y0 + ph_ - 70  # bottom-left corner: empty in this plot
    lgx = x0 + 12
    for i, b in enumerate(a["budgets"]):
        col = COLORS.get(b["name"], "black")
        y = ly0 + 18 * i
        txt = f"{b['name']} ({b['C']:.0e}): N_opt {b['n_opt'] / 1e6:.1f} M" if b.get("n_opt") else b["name"]
        parts.append(f'<line x1="{lgx}" y1="{y}" x2="{lgx + 20}" y2="{y}" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{lgx + 25}" y="{y + 4}">{txt}</text>')
    parts.append(
        f'<text x="{lgx}" y="{ly0 + 58}" font-size="10" fill="#555">'
        "lines: parabola fits; triangles: N_opt</text>"
    )

    # ---- right panel: growth laws
    x1 = 640
    gl = a["growth_law"]
    pj = gl["projection"]
    lcx = (15.8, math.log10(pj["C_star"]) + 0.3)
    lcy = (6.0, 10.0)

    def qx(c: float) -> float:
        return x1 + (math.log10(c) - lcx[0]) / (lcx[1] - lcx[0]) * pw

    def qy(v: float) -> float:
        return y0 + ph_ - (math.log10(v) - lcy[0]) / (lcy[1] - lcy[0]) * ph_

    parts.append(_axes(x1, y0, pw, ph_))
    for e in range(16, int(lcx[1]) + 1):
        parts.append(_xtick(qx(10**e), y0 + ph_, f"1e{e}"))
    for e in range(6, 11):
        parts.append(_ytick(x1, qy(10**e), f"1e{e}"))
    parts.append(
        f'<text x="{x1 + pw / 2}" y="{y0 + ph_ + 36}" text-anchor="middle">'
        "training compute C (FLOPs, log scale)</text>"
    )
    parts.append(
        f'<text transform="translate({x1 - 48},{y0 + ph_ / 2}) rotate(-90)" text-anchor="middle">'
        "parameters or tokens (log scale)</text>"
    )
    br = [b for b in a["budgets"] if b.get("bracketed")]
    series = [
        ("N_opt, non-embedding (pre-registered)", "n_opt", gl["n_law"], "#9467bd"),
        ("D_opt, tokens (pre-registered)", "d_opt", gl["d_law"], "#ff7f0e"),
    ]
    for i, (label, key, law, col) in enumerate(series):
        c_lo, c_hi = 10 ** lcx[0], pj["C_star"]
        c_meas = max(b["C"] for b in br)
        for seg, dash in (((c_lo, c_meas), ""), ((c_meas, c_hi), ' stroke-dasharray="6,4"')):
            ya, yb = law["k"] * seg[0] ** law["exponent"], law["k"] * seg[1] ** law["exponent"]
            parts.append(
                f'<line x1="{qx(seg[0]):.1f}" y1="{qy(ya):.1f}" x2="{qx(seg[1]):.1f}" y2="{qy(yb):.1f}" '
                f'stroke="{col}" stroke-width="1.8"{dash}/>'
            )
        for b in br:
            parts.append(f'<circle cx="{qx(b["C"]):.1f}" cy="{qy(b[key]):.1f}" r="4" fill="{col}"/>')
        y_end = law["k"] * c_hi ** law["exponent"]
        parts.append(
            f'<circle cx="{qx(c_hi):.1f}" cy="{qy(y_end):.1f}" r="5" fill="white" '
            f'stroke="{col}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{x1 + 12}" y1="{y0 + 14 + 18 * i}" x2="{x1 + 32}" y2="{y0 + 14 + 18 * i}" '
            f'stroke="{col}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x1 + 37}" y="{y0 + 18 + 18 * i}">{label}: exponent {law["exponent"]:.2f}</text>'
        )
    tg = ph["total_parameter_growth_law"]
    parts.append(
        f'<text x="{x1 + 12}" y="{y0 + 62}" font-size="11">'
        "open circles: EXTRAPOLATION to D_opt = 2.78 B tokens "
        f"(C = {pj['C_star']:.1e})</text>"
    )
    parts.append(
        f'<text x="{x1 + 12}" y="{y0 + 78}" font-size="11">'
        f"there: N_opt {pj['n_opt_non_embedding'] / 1e6:.0f} M non-embedding"
        f" (leave-one-out {pj['leave_one_out_n_opt'][0] / 1e6:.0f} M"
        f" - {pj['leave_one_out_n_opt'][1] / 1e9:.1f} B)</text>"
    )
    parts.append(
        f'<text x="{x1 + 12}" y="{y0 + 94}" font-size="11" fill="#555">post-hoc, total parameters: exponent '
        f"{tg['a_total']:.2f}, N_opt {tg['n_opt_total_at_C_star'] / 1e6:.0f} M total</text>"
    )
    parts.append(
        f'<text x="{w / 2}" y="{h - 12}" text-anchor="middle" font-size="10" fill="#555">'
        "20 runs, 16,384 tokens per step, context 512, RoPE + GQA-2 (D-048)."
        f" Code {s['environment']['code_commit'][:7]}. "
        "Smaller is better on the left panel.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _axes(x: float, y: float, w: float, h: float) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="black"/>'


def _xtick(x: float, y: float, label: str) -> str:
    return (
        f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + 5}" stroke="black"/>'
        f'<text x="{x:.1f}" y="{y + 18}" text-anchor="middle">{label}</text>'
    )


def _ytick(x: float, y: float, label: str) -> str:
    return (
        f'<line x1="{x - 5}" y1="{y:.1f}" x2="{x}" y2="{y:.1f}" stroke="black"/>'
        f'<text x="{x - 8}" y="{y + 4:.1f}" text-anchor="end">{label}</text>'
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="evals/results/EXP-042")
    args = p.parse_args(argv)
    d = Path(args.dir) if Path(args.dir).is_absolute() else ROOT / args.dir
    s = final_summary(d)
    ph = posthoc(s)
    (d / "posthoc.json").write_text(json.dumps(ph, indent=2) + "\n", encoding="utf-8", newline="\n")
    (d / "isoflop.svg").write_text(_svg(s, ph), encoding="utf-8", newline="\n")
    print(json.dumps(ph, indent=2))
    print(f"wrote {d / 'isoflop.svg'} and {d / 'posthoc.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
