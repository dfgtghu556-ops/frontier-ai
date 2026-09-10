#!/usr/bin/env python3
"""Run one experiment across seeds (and optionally across named configurations) and aggregate
the result as mean ± spread.

Every configuration × seed produces its own normal experiment record
(``[<config>/]seed-<seed>/experiment.json`` with git, data, environment and configuration
provenance); the sweep then writes ``sweep.json`` with the aggregate. Follows the
Project 001/002/003 CLI conventions (argparse, ``--exp-id``, ``--out``, ``--set``), and
everything after ``--`` is the command to run per configuration × seed.

Examples
--------
    # wrap an existing Project 001 training run, one run per seed
    python scripts/experiment_sweep.py --exp-id EXP-004 --seeds 1,2,3,4,5 \\
        --metric best_val_loss --config configs/cpu_smoke.json --data data/synthetic.bin \\
        --out out/sweeps/EXP-004-train --tag project-001 -- \\
        python scripts/train.py --config configs/cpu_smoke.json --max-steps 50 \\
            --set train.seed={seed}

    # the metric is read from the run record's results section; a nested field is fine
    python scripts/experiment_sweep.py --exp-id EXP-004 --seeds 7,8 \\
        --metric eval.val_loss --out out/sweeps/EXP-004 -- \\
        python scripts/my_experiment.py --seed {seed}

    # several named configurations, each across every seed ({config} and {seed})
    python scripts/experiment_sweep.py --exp-id EXP-005 --seeds 1,2,3 \\
        --configs "lr_high:params.lr=0.02" "lr_low:params.lr=0.005" \\
        --metric final_val_loss --out out/sweeps/EXP-005 -- \\
        python scripts/my_experiment.py --lr-config {config} --seed {seed}

Use ``{seed}`` in the command where the seed must reach the experiment, and ``{config}``
where the configuration name must reach it. A command without ``{seed}`` runs identically
for every seed, and one without ``{config}`` runs identically for every configuration;
the recorder prints a warning in both cases.

With ``--configs`` each configuration is aggregated separately (mean ± sample standard
deviation per configuration); results from different configurations are never mixed.

Exit codes: 0 = every seed produced a usable metric, 2 = partial (some seeds failed),
1 = no seed produced a usable metric. The sweep record is written in all three cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.experiments import (  # noqa: E402
    ExperimentSpec,
    parse_configuration_list,
    parse_seed_list,
    run_command_sweep,
)

EXIT_SUCCESS = 0
EXIT_ALL_FAILED = 1
EXIT_PARTIAL = 2


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split recorder flags from the recorded command at the first ``--``."""
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="experiment_sweep.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--exp-id", required=True, help="experiment id, e.g. EXP-004")
    p.add_argument("--seeds", nargs="+", required=True, metavar="SEED",
                   help="seeds to run: '1,2,3' or '1 2 3' (order does not affect the result)")
    p.add_argument("--configs", nargs="+", action="extend", default=[], metavar="NAME:key=value",
                   help="named configurations, each run across every seed, e.g. "
                        "'lr_high:params.lr=0.02' (repeat the flag; keys: params.<name>, "
                        "config_path, name, notes; order does not affect the result)")
    p.add_argument("--metric", required=True,
                   help="result field to aggregate from each run (dotted path allowed)")
    p.add_argument("--name", default="", help="short human-readable sweep name")
    p.add_argument("--out", default=None, help="output directory (default out/sweeps/<exp-id>)")
    p.add_argument("--config", dest="config_path", default="", help="config file to embed verbatim")
    p.add_argument("--data", nargs="*", default=[], help="input files/dirs to hash for provenance")
    p.add_argument("--set", nargs="+", action="extend", default=[], metavar="key=value",
                   help="spec overrides: output_dir, name, notes, params.<name>")
    p.add_argument("--tag", nargs="*", default=[], help="free-form tags")
    p.add_argument("--notes", default="", help="free-form notes recorded with the sweep")
    p.add_argument("--deterministic", action="store_true",
                   help="request torch deterministic algorithms (best effort, see docs)")
    p.add_argument("--timeout", type=float, default=None,
                   help="timeout in seconds for each command")
    p.add_argument("--quiet", action="store_true", help="do not print the sweep summary")
    return p.parse_args(argv)


def main() -> int:
    recorder_argv, command = split_argv(sys.argv[1:])
    args = parse_args(recorder_argv)

    seeds = parse_seed_list(args.seeds)
    configs = parse_configuration_list(args.configs) if args.configs else None
    out_dir = args.out or f"out/sweeps/{args.exp_id}"

    spec = ExperimentSpec(
        experiment_id=args.exp_id,
        seed=seeds[0],
        name=args.name,
        output_dir=out_dir,
        data_paths=list(args.data),
        config_path=args.config_path,
        command=command,
        tags=list(args.tag),
        notes=args.notes,
        deterministic_mode=args.deterministic,
    ).with_overrides(args.set)

    try:
        outcome = run_command_sweep(
            spec, seeds, args.metric, command, output_dir=args.out, timeout=args.timeout,
            configurations=configs,
        )
    except SystemExit:  # pragma: no cover - defensive
        raise
    except BaseException as exc:  # noqa: BLE001 - report, then fail loudly
        print(f"[sweep] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[sweep] partial results (if any) under {out_dir}", file=sys.stderr)
        return getattr(exc, "returncode", 1) or 1

    if not command_has_seed(command):
        print("[sweep] WARNING: the command contains no '{seed}' placeholder, so every seed "
              "runs the same command.", file=sys.stderr)
    if configs and not command_has_config(command):
        print("[sweep] WARNING: --configs was given but the command contains no '{config}' "
              "placeholder, so every configuration runs the same command.", file=sys.stderr)

    status = outcome.record.sweep["status"]
    if not args.quiet:
        print()
        print(outcome.rendered)
        print(f"\n[sweep] record: {outcome.path}")
    print(json.dumps({
        "experiment_id": spec.experiment_id,
        "status": status,
        "sweep": str(outcome.path),
        "metric": args.metric,
        "mean": outcome.record.statistics.get("mean"),
        "spread": outcome.record.statistics.get("spread"),
        "successful_seeds": outcome.record.sweep["successful_seeds"],
        "failed_seeds": outcome.record.sweep["failed_seeds"],
        "configurations": [
            {"name": cfg["name"], "status": cfg["status"],
             "mean": cfg["statistics"]["mean"], "spread": cfg["statistics"]["spread"]}
            for cfg in outcome.record.configurations
        ],
        "fingerprint": outcome.record.content_fingerprint(),
    }))

    if status == "success":
        return EXIT_SUCCESS
    return EXIT_ALL_FAILED if status == "failed" else EXIT_PARTIAL


def command_has_seed(command: list[str]) -> bool:
    return any("{seed}" in arg for arg in command)


def command_has_config(command: list[str]) -> bool:
    return any("{config}" in arg for arg in command)


if __name__ == "__main__":
    raise SystemExit(main())
