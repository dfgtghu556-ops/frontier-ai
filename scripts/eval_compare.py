#!/usr/bin/env python3
"""Compare two groups of evaluation reports under identical conditions.

    python scripts/eval_compare.py \\
        --a out/eval/EXP-031/mark_aware-32768-seed-1337 ... --label-a mark_aware-32768 \\
        --b out/eval/EXP-031/mark_aware-16384-seed-1337 ... --label-b mark_aware-16384

Each group is one or more report directories (e.g. one per training seed). Refuses to
compare reports made with different harness versions, suites or scoring protocols.

For overall and each language it reports:
  * per group: mean and sample std of bits-per-byte across the group's reports (seed
    variation), and the individual values;
  * the difference B - A of the group means with a **paired document-bootstrap** 95%
    interval (per-document bits averaged over each group's reports; documents resampled
    jointly), and a verdict (interval excludes 0 or not).

Writes compare.json + compare.txt into --out (default out/eval/compare-<A>-vs-<B>).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from frontier_ai.evaluation import make_console_safe  # noqa: E402
from frontier_ai.evaluation.stats import paired_delta  # noqa: E402


def _load(dirpath: str) -> tuple[dict, list[dict]]:
    d = Path(dirpath)
    report = json.loads((d / "report.json").read_text(encoding="utf-8"))
    docs = [json.loads(line) for line in (d / "per_document.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return report, docs


def _check_same_conditions(reports: list[dict]) -> None:
    ref = reports[0]
    for r in reports[1:]:
        for key, a, b in (
            ("harness_version", ref["harness_version"], r["harness_version"]),
            ("suite fingerprint", ref["suite"]["fingerprint"], r["suite"]["fingerprint"]),
            ("protocol", ref["protocol"]["description"], r["protocol"]["description"]),
        ):
            if a != b:
                raise SystemExit(f"[compare] REFUSED: {key} differs between reports ({a!r} vs {b!r})")


def _matrix(docs_list: list[list[dict]]) -> tuple[list[dict], np.ndarray]:
    """(document meta for scored docs, bits matrix [reports x docs])."""
    ref = [d for d in docs_list[0] if not d["context_only"]]
    ids = [d["doc_id"] for d in ref]
    rows = []
    for docs in docs_list:
        scored = [d for d in docs if not d["context_only"]]
        if [d["doc_id"] for d in scored] != ids:
            raise SystemExit("[compare] REFUSED: reports cover different documents")
        rows.append([float(d["bits"]) for d in scored])
    return ref, np.array(rows, dtype=np.float64)


def main() -> int:
    make_console_safe()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", nargs="+", required=True, help="report dirs of group A")
    p.add_argument("--b", nargs="+", required=True, help="report dirs of group B")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    loaded_a = [_load(x) for x in args.a]
    loaded_b = [_load(x) for x in args.b]
    reports = [r for r, _ in loaded_a + loaded_b]
    _check_same_conditions(reports)
    meta, bits_a = _matrix([d for _, d in loaded_a])
    meta_b, bits_b = _matrix([d for _, d in loaded_b])
    if [m["doc_id"] for m in meta] != [m["doc_id"] for m in meta_b]:
        raise SystemExit("[compare] REFUSED: groups cover different documents")
    bytes_ = np.array([m["bytes"] for m in meta], dtype=np.float64)
    langs = [m["language"] for m in meta]

    def section(mask: np.ndarray) -> dict:
        denom = bytes_[mask]
        per_a = [float(row[mask].sum() / denom.sum()) for row in bits_a]
        per_b = [float(row[mask].sum() / denom.sum()) for row in bits_b]
        delta = paired_delta(bits_a[:, mask].mean(axis=0), bits_b[:, mask].mean(axis=0), denom,
                             n_boot=args.bootstrap, seed=args.seed)
        return {
            "documents": int(mask.sum()),
            "a": {"mean_bpb": statistics.fmean(per_a),
                  "std_bpb": statistics.stdev(per_a) if len(per_a) > 1 else None, "values": per_a},
            "b": {"mean_bpb": statistics.fmean(per_b),
                  "std_bpb": statistics.stdev(per_b) if len(per_b) > 1 else None, "values": per_b},
            **delta,
        }

    result = {
        "schema": "frontier-eval-compare-v1",
        "harness_version": reports[0]["harness_version"],
        "suite_fingerprint": reports[0]["suite"]["fingerprint"],
        "label_a": args.label_a, "label_b": args.label_b,
        "reports_a": args.a, "reports_b": args.b,
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "paired": True},
        "note": "delta = mean bpb(B) - mean bpb(A); negative means B is better. The paired interval "
                "covers document sampling; seed variation is shown by the per-group std.",
        "overall": section(np.ones(len(meta), dtype=bool)),
        "per_language": {lang: section(np.array([x == lang for x in langs]))
                         for lang in sorted(set(langs))},
    }

    def fmt_group(g: dict) -> str:
        std = f" ± {g['std_bpb']:.4f}" if g["std_bpb"] is not None else ""
        return f"{g['mean_bpb']:.4f}{std}"

    header = (f"{'':10}{'docs':>6}  {'A mean ± std':>18}  {'B mean ± std':>18}  "
              f"{'delta':>8}  {'95% CI':20} verdict")
    lines = [f"COMPARISON (harness v{result['harness_version']}) — "
             f"A = {args.label_a} ({len(args.a)} reports), B = {args.label_b} ({len(args.b)} reports)",
             "delta = B - A in bits-per-byte (negative = B better); "
             "95% paired document-bootstrap interval",
             "", header]
    for name, sec in [("OVERALL", result["overall"]), *result["per_language"].items()]:
        lo, hi = sec["ci95"]
        lines.append(f"{name:10}{sec['documents']:>6}  {fmt_group(sec['a']):>18}  {fmt_group(sec['b']):>18}  "
                     f"{sec['delta_b_minus_a']:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {sec['verdict']}")
    text = "\n".join(lines)

    out = Path(args.out or f"out/eval/compare-{args.label_a}-vs-{args.label_b}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "compare.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
