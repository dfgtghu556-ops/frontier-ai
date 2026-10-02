"""EXP-042 analysis: IsoFLOP fits with numpy only (pre-registered in EXPERIMENTS.md, EXP-042).

Pure functions on plain numbers, so they can be tested on synthetic data with a known answer.

* :func:`fit_budget`: one budget. A parabola in log10(N) through (N, loss); its minimum is
  N_opt(C). "Bracketed" only if the minimum lies strictly inside the window AND the lowest measured
  point is not at an edge. The "flat region" is where the parabola is within `noise` of its minimum.
* :func:`fit_power`: y = k * C**e by a straight line in log-log; residuals if there are > 2 points.
* :func:`growth_law`: rule 2 + rule 3 (N_opt ~ C^a, D_opt ~ C^b, the budget at which D_opt equals the
  available tokens, with a leave-one-budget-out range).
* :func:`fit_parametric`: loss = E + A / N**alpha + B / D**beta (Chinchilla approach 3), by a grid over
  (alpha, beta) with a non-negative least-squares solve for (E, A, B) at each grid point.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def fit_budget(points: Sequence[tuple[float, float]], noise: float) -> dict[str, Any]:
    """Parabola in log10(N); `points` = (N, loss) for one budget, any order."""
    pts = sorted((float(n), float(y)) for n, y in points)
    if len(pts) < 3:
        return {
            "bracketed": False,
            "reason": f"only {len(pts)} sizes succeeded (need 3)",
            "n_points": len(pts),
        }
    x = np.log10([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    a, b, c = np.polyfit(x, y, 2)
    lowest = int(np.argmin(y))
    out: dict[str, Any] = {"n_points": len(pts), "coef": [float(a), float(b), float(c)]}
    out["lowest_measured_n"] = pts[lowest][0]
    if a <= 0:
        out.update(bracketed=False, reason="the parabola opens downwards (no minimum)")
        return out
    xs = -b / (2 * a)
    fmin = float(c - b * b / (4 * a))
    half = math.sqrt(noise / a)
    out.update(
        n_opt=float(10**xs),
        loss_at_opt=fmin,
        flat_region=[float(10 ** (xs - half)), float(10 ** (xs + half))],
        flat_sizes=[p[0] for p, f in zip(pts, np.polyval([a, b, c], x)) if f <= fmin + noise],
    )
    inside = x[0] < xs < x[-1]
    edge = lowest in (0, len(pts) - 1)
    out["bracketed"] = bool(inside and not edge)
    if not out["bracketed"]:
        out["reason"] = (
            "the lowest measured point is at the edge of the window"
            if edge
            else "the minimum is outside the window"
        )
    return out


def fit_power(cs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    """y = k * C**e; exact through 2 points, least squares (with residuals) through more."""
    if len(cs) < 2:
        return {"computable": False, "reason": "needs at least 2 points"}
    lx, ly = np.log10(cs), np.log10(ys)
    e, logk = np.polyfit(lx, ly, 1)
    out = {"computable": True, "exponent": float(e), "k": float(10**logk), "n_points": len(cs)}
    if len(cs) > 2:
        out["residuals_log10"] = [float(r) for r in ly - (e * lx + logk)]
    return out


def _crossing(n_law: dict, d_law: dict, tokens: float) -> tuple[float, float]:
    """The budget C* at which D_opt(C*) = tokens, and N_opt(C*)."""
    c_star = (tokens / d_law["k"]) ** (1.0 / d_law["exponent"])
    return float(c_star), float(n_law["k"] * c_star ** n_law["exponent"])


def growth_law(budgets: Sequence[dict[str, Any]], tokens_available: float) -> dict[str, Any]:
    """Rules 2 and 3 over the bracketed budgets (each: {"C", "n_opt", "d_opt"})."""
    br = [b for b in budgets if b.get("bracketed")]
    out: dict[str, Any] = {"bracketed_budgets": [b["name"] for b in br]}
    if len(br) < 2:
        out.update(computable=False, reason=f"{len(br)} bracketed budget(s); at least 2 are needed")
        return out
    cs = [b["C"] for b in br]
    n_law = fit_power(cs, [b["n_opt"] for b in br])
    d_law = fit_power(cs, [b["d_opt"] for b in br])
    out.update(computable=True, a=n_law["exponent"], b=d_law["exponent"], n_law=n_law, d_law=d_law)
    if d_law["exponent"] <= 0:
        out["projection"] = {"computable": False, "reason": "D_opt does not grow with compute"}
        return out
    c_star, n_star = _crossing(n_law, d_law, tokens_available)
    proj: dict[str, Any] = {
        "computable": True,
        "tokens": tokens_available,
        "C_star": c_star,
        "n_opt_non_embedding": n_star,
        "orders_of_magnitude_beyond_largest_budget": float(math.log10(c_star / max(cs))),
        "label": "EXTRAPOLATION beyond the measured budgets",
    }
    if len(br) >= 3:
        lo_hi = []
        for drop in range(len(br)):
            keep = [b for i, b in enumerate(br) if i != drop]
            nl = fit_power([b["C"] for b in keep], [b["n_opt"] for b in keep])
            dl = fit_power([b["C"] for b in keep], [b["d_opt"] for b in keep])
            if dl["exponent"] > 0:
                lo_hi.append(_crossing(nl, dl, tokens_available))
        if lo_hi:
            proj["leave_one_out_C_star"] = [min(c for c, _ in lo_hi), max(c for c, _ in lo_hi)]
            proj["leave_one_out_n_opt"] = [min(n for _, n in lo_hi), max(n for _, n in lo_hi)]
    else:
        proj["leave_one_out"] = "needs 3 bracketed budgets"
    out["projection"] = proj
    return out


def fit_parametric(
    points: Sequence[tuple[float, float, float]],
    alphas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
) -> dict[str, Any]:
    """loss = E + A/N^alpha + B/D^beta over (N, D, loss) points; E, A, B >= 0."""
    if len(points) < 5:
        return {"converged": False, "reason": "needs at least 5 runs"}
    alphas = list(alphas if alphas is not None else np.round(np.arange(0.05, 1.501, 0.01), 2))
    betas = list(betas if betas is not None else np.round(np.arange(0.05, 1.501, 0.01), 2))
    n = np.array([p[0] for p in points], dtype=np.float64)
    d = np.array([p[1] for p in points], dtype=np.float64)
    y = np.array([p[2] for p in points], dtype=np.float64)
    best: tuple[float, float, float, np.ndarray] | None = None
    for al, be in itertools.product(alphas, betas):
        x = np.stack([np.ones_like(n), n**-al, d**-be], axis=1)
        coef = _nnls3(x, y)
        sse = float(((x @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, al, be, coef)
    assert best is not None
    sse, al, be, coef = best
    at_edge = al in (alphas[0], alphas[-1]) or be in (betas[0], betas[-1])
    return {
        "converged": bool(not at_edge and coef[1] > 0 and coef[2] > 0),
        "reason": "best (alpha, beta) at the edge of the grid" if at_edge else None,
        "E": float(coef[0]),
        "A": float(coef[1]),
        "B": float(coef[2]),
        "alpha": float(al),
        "beta": float(be),
        "rmse": math.sqrt(sse / len(y)),
        "implied_a": float(be / (al + be)),  # N_opt ~ C^(beta/(alpha+beta)) when C ~ N*D
    }


def _nnls3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares for 3 columns: try every subset of active columns."""
    best, best_sse = np.zeros(3), float("inf")
    for mask in itertools.product([True, False], repeat=3):
        cols = [i for i in range(3) if mask[i]]
        if not cols:
            continue
        sol, *_ = np.linalg.lstsq(x[:, cols], y, rcond=None)
        if (sol < 0).any():
            continue
        coef = np.zeros(3)
        coef[cols] = sol
        sse = float(((x @ coef - y) ** 2).sum())
        if sse < best_sse:
            best, best_sse = coef, sse
    return best
