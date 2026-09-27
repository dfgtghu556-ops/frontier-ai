#!/usr/bin/env python3
"""EXP-B: tokenizer-vs-tokenizer small-model comparison (MASTER_CONTEXT §37 step 6).

Trains the approved fixed small model (configs/exp_b.json) on each prepared
EXP-B data split with >= 3 seeds and compares the held-out bits-per-byte.
Each cell is a self-recording run of scripts/train.py (D-032), so every run
has its own experiment record; this script writes the aggregate table,
applies the PRE-REGISTERED decision rule (see EXPERIMENTS.md, EXP-029), and
writes its own outer record.

Pre-registered decision rule (must match EXPERIMENTS.md EXP-029):
* winner = the tokenizer with the LOWER mean held-out bits-per-byte across
  its seeds;
* if |mean_A - mean_B| < the average of the two within-tokenizer standard
  deviations, the result is a TIE and the smaller-vocabulary tokenizer wins
  (deployment economy at equal quality);
* the selection is recorded as a new DECISIONS.md entry only after this
  run is reviewed.

Exit codes: 0 = all cells ran and the table is complete; 1 = at least one
cell failed; 2 = bad input (missing prepared data).

Examples
--------
    # timing smoke: 1 tokenizer x 1 seed x 300 steps; prints the estimated
    # wall time for the full budget, so the budget can be set from the PC's
    # real CPU speed before committing to the 6-run matrix
    python scripts/run_exp_b.py --smoke

    # the full matrix (after the budget is settled; the same --max-steps for
    # every cell)
    python scripts/run_exp_b.py --seeds 1337,1338,1339 --max-steps 1000
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.experiments import ExperimentSpec  # noqa: E402
from frontier_ai.experiments.autowire import run_self_recorded  # noqa: E402

DEFAULT_CONFIG = "configs/exp_b.json"
DEFAULT_SEEDS = "1337,1338,1339"
SMOKE_STEPS = 300


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="EXP-029", help="outer experiment id (default: %(default)s)")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="model/training config shared by ALL cells (default: %(default)s)")
    p.add_argument("--data-dir", default=None,
                   help="prepared EXP-B data dir (default: out/exp_b/<exp-id>)")
    p.add_argument("--seeds", default=DEFAULT_SEEDS,
                   help="comma-separated seeds, one full set per tokenizer (default: %(default)s)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="step budget for EVERY cell (default: the config's train.max_steps)")
    p.add_argument("--smoke", action="store_true",
                   help="timing smoke only: 1 tokenizer x 1 seed x "
                        f"{SMOKE_STEPS} steps, then estimate the full budget and exit")
    p.add_argument("--seed", type=int, default=1337,
                   help="outer record seed (the cell seeds come from --seeds)")
    p.add_argument("--notes", default="", help="notes for the experiment record")
    p.add_argument("--no-record", action="store_true", help="do not write the outer record")
    return p


def _parse_seeds(raw: str) -> list[int]:
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                seeds.append(int(part))
            except ValueError:
                raise SystemExit(f"[exp-b] bad --seeds entry {part!r} (expected integers)") from None
    return sorted(set(seeds))


def _load_data_manifest(data_dir: Path) -> dict:
    path = data_dir / "manifest.json"
    if not path.is_file():
        print(f"[exp-b] no prepared data at {path} — run scripts/prepare_exp_b_data.py first",
              file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_record(out_dir: Path) -> dict:
    path = out_dir / "experiment.json"
    if not path.is_file():
        raise RuntimeError(f"no experiment record at {path} — the run crashed before recording")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cell(
    *,
    cell_exp_id: str,
    config: str,
    data_file: str,
    seed: int,
    out_dir: Path,
    max_steps: int | None,
) -> dict:
    """One self-recording train.py run; returns the recorded results."""
    overrides = [
        f"data.path={data_file}",
        f"data.seed={seed}",
        f"train.seed={seed}",
        f"train.out_dir={out_dir}",
    ]
    if max_steps is not None:
        overrides.append(f"train.max_steps={max_steps}")
    set_args: list[str] = []
    for override in overrides:
        set_args.extend(["--set", override])
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/train.py"),
        "--config", config,
        *set_args,
        "--exp-id", cell_exp_id,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"cell {cell_exp_id} exited {proc.returncode}")
    record = _load_run_record(out_dir)
    results = record.get("results") or {}
    if (
        record.get("execution", {}).get("status") != "success"
        or results.get("best_bpb") is None
    ):
        raise RuntimeError(f"cell {cell_exp_id} did not record a usable bits-per-byte")
    return {
        "exp_id": cell_exp_id,
        "seed": seed,
        "best_bpb": float(results["best_bpb"]),
        "best_val": float(results["best_val"]),
        "steps": int(results["steps"]),
        "n_params": int(results.get("n_params", 0)),
        "tokens_seen": int(results.get("tokens_seen", 0)),
        "wall_seconds": round(wall, 1),
        "record": str(out_dir / "experiment.json"),
    }


def decide(rows: list[dict]) -> tuple[str, str]:
    """The pre-registered rule (see the module docstring and EXPERIMENTS.md).

    ``rows``: one entry per tokenizer: {name, vocab_size, values:[bpb per seed]}.
    Returns (winner_name, reason).
    """
    for row in rows:
        row["mean"] = sum(row["values"]) / len(row["values"])
        if len(row["values"]) > 1:
            var = sum((v - row["mean"]) ** 2 for v in row["values"]) / (len(row["values"]) - 1)
            row["std"] = math.sqrt(var)
        else:
            row["std"] = 0.0
    if len(rows) == 2:
        gap = abs(rows[0]["mean"] - rows[1]["mean"])
        avg_std = (rows[0]["std"] + rows[1]["std"]) / 2
        if avg_std > 0 and gap < avg_std:
            smaller = min(rows, key=lambda r: r["vocab_size"])
            return smaller["name"], (
                f"TIE (gap {gap:.4f} bpb < avg seed-std {avg_std:.4f}): pre-registered "
                f"tie-break -> the smaller-vocabulary tokenizer"
            )
    best = min(rows, key=lambda r: r["mean"])
    return best["name"], f"lower mean bits-per-byte ({best['mean']:.4f} bpb)"


def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir or f"out/exp_b/{args.exp_id}")
    out_root = data_dir / "runs"
    data = _load_data_manifest(data_dir)
    tokenizers = data["tokenizers"]
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("[exp-b] --seeds must name at least one seed")

    # ---------------------------------------------------------------- smoke --
    if args.smoke:
        tok = tokenizers[0]
        seed = seeds[0]
        out_dir = out_root / f"{tok['name']}" / f"seed-{seed}"
        print(f"[exp-b] SMOKE: {tok['name']} seed {seed}, {SMOKE_STEPS} steps ...")
        cell = _run_cell(
            cell_exp_id=f"{args.exp_id}00",
            config=args.config,
            data_file=str(data_dir / tok["file"]),
            seed=seed,
            out_dir=out_dir,
            max_steps=SMOKE_STEPS,
        )
        per_step = cell["wall_seconds"] / SMOKE_STEPS
        budget = args.max_steps
        print(f"[exp-b] smoke done: {cell['wall_seconds']:.0f}s for {SMOKE_STEPS} steps "
              f"({cell['best_bpb']:.4f} bpb at this budget)")
        if budget is None:
            budget = 1000
        total = per_step * budget * len(tokenizers) * len(seeds)
        print(f"[exp-b] estimate for the full matrix ({len(tokenizers)} tokenizers x "
              f"{len(seeds)} seeds x {budget} steps): ~{total / 60:.0f} min on this CPU")
        print(f"[exp-b] if that fits, re-run without --smoke "
              f"(add --max-steps {budget} to pin the budget)")
        return 0

    # ------------------------------------------------------------ the matrix --
    def run_matrix() -> dict:
        t0 = time.monotonic()
        rows: list[dict] = []
        failed = 0
        for i, tok in enumerate(tokenizers):
            row = {"name": tok["name"], "vocab_size": tok["vocab_size"], "values": []}
            rows.append(row)
            for j, seed in enumerate(seeds):
                cell_exp_id = f"{args.exp_id}{i * len(seeds) + j + 1:02d}"
                out_dir = out_root / tok["name"] / f"seed-{seed}"
                print(f"[exp-b] {tok['name']} seed {seed} ({cell_exp_id}) ...", flush=True)
                try:
                    cell = _run_cell(
                        cell_exp_id=cell_exp_id,
                        config=args.config,
                        data_file=str(data_dir / tok["file"]),
                        seed=seed,
                        out_dir=out_dir,
                        max_steps=args.max_steps,
                    )
                except (RuntimeError, OSError) as exc:
                    failed += 1
                    print(f"[exp-b] FAILED: {tok['name']} seed {seed}: {exc}", file=sys.stderr)
                    continue
                row["values"].append(cell["best_bpb"])
                row.setdefault("cells", []).append(cell)
                print(f"[exp-b]   {tok['name']} seed {seed}: bpb={cell['best_bpb']:.4f} "
                      f"val_loss={cell['best_val']:.4f} ({cell['wall_seconds']:.0f}s)", flush=True)

        winner, reason = decide(rows) if any(r["values"] for r in rows) else ("", "no completed cells")

        lines = [
            "EXP-B tokenizer comparison — results",
            f"generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            f"config: {args.config} (fixed for all cells) | seeds: {', '.join(map(str, seeds))}",
            "",
            f"{'tokenizer':<28} " + "".join(f"{'seed ' + str(s):>14}" for s in seeds) +
            f" {'mean bpb':>12} {'std':>10} {'tokens':>12}",
        ]
        for row in rows:
            padded = list(row["values"]) + [""] * (len(seeds) - len(row["values"]))
            lines.append(
                f"{row['name']:<28} "
                + "".join(f"{v:>14.4f}" if v != "" else f"{'FAIL':>14}" for v in padded)
                + (f" {row.get('mean', float('nan')):>12.4f} {row.get('std', float('nan')):>10.4f} "
                   if any(v for v in row["values"]) else f" {'—':>12} {'—':>10}")
                + f" {sum(c['tokens_seen'] for c in row.get('cells', [])):>12,}"
            )
        lines += [
            "",
            f"DECISION (pre-registered rule, see EXPERIMENTS.md EXP-029): {winner} — {reason}",
            "",
            "The selection is recorded as a new DECISIONS.md entry only after founder review.",
        ]
        report = "\n".join(lines) + "\n"
        (out_root / "report.txt").write_text(report, encoding="utf-8", newline="\n")
        print()
        print(report)
        print(f"[exp-b] report: {out_root / 'report.txt'}")
        if failed:
            print(f"[exp-b] exit 1: {failed} cell(s) failed", file=sys.stderr)
        return {
            "config": args.config,
            "seeds": seeds,
            "matrix_wall_seconds": round(time.monotonic() - t0, 1),
            "rows": [
                {"name": r["name"], "vocab_size": r["vocab_size"],
                 "bpb_per_seed": r["values"],
                 "mean_bpb": round(r.get("mean", float("nan")), 6),
                 "std_bpb": round(r.get("std", float("nan")), 6),
                 "cells": [
                     {k: v for k, v in c.items() if k != "record"} for c in r.get("cells", [])
                 ],
                 "failed_seeds": len(seeds) - len(r["values"])}
                for r in rows
            ],
            "winner": winner,
            "decision_reason": reason,
            "failed_cells": failed,
        }

    # The cells must run with NO active outer experiment: while the outer
    # record is active, train.py runs in nested mode and skips its own
    # per-cell experiment.json, which every cell check then reports as a
    # failure. Run the matrix first; write the outer aggregate record after,
    # carrying the matrix results (no subprocesses inside the record body).
    result = run_matrix()

    if args.no_record:
        return 1 if result["failed_cells"] else 0

    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            seed=args.seed,
            name="EXP-B tokenizer-vs-tokenizer small-model comparison",
            output_dir=str(out_root),
            params={
                "config": args.config,
                "seeds": seeds,
                "max_steps": args.max_steps,
                "tokenizers": [t["name"] for t in tokenizers],
                "frontier_manifest_content_sha256": data["frontier_manifest_content_sha256"],
            },
            data_paths=[str(data_dir / "manifest.json")],
            command=list(sys.argv),
            tags=["tokenizer", "exp-b", "frontier-corpus-v1"],
            notes=args.notes,
        )

    recorded = run_self_recorded(build_spec, lambda: result)
    return 0 if not recorded.failed and result["failed_cells"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
