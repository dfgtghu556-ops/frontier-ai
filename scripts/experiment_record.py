#!/usr/bin/env python3
"""Record provenance for an experiment, optionally wrapping a command.

Follows the Project 001/002 CLI conventions (argparse, ``--exp-id``, ``--out``, ``--set``)
and captures, for every run: git provenance, input data digests, configuration, seed
handling, environment, the exact command and its exit status.

Examples
--------
    # wrap an existing Project 001 training run
    python scripts/experiment_record.py --exp-id EXP-003 --seed 1337 \\
        --config configs/cpu_smoke.json --data data/synthetic.bin \\
        --out out/experiments/EXP-003-train --tag project-001 -- \\
        python scripts/train.py --config configs/cpu_smoke.json --max-steps 50

    # wrap a tokenizer training run (Project 002)
    python scripts/experiment_record.py --exp-id EXP-003 --seed 1337 \\
        --data data/tokenizer/indic-v1/train.txt --out out/experiments/EXP-003-tok -- \\
        python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 \\
            --impl bpe_hf --vocab-size 1024 --out artifacts/tokenizers/bpe_hf_1024

    # provenance only (no command): describe the current code + data state
    python scripts/experiment_record.py --exp-id EXP-003 --data data/synthetic.bin \\
        --out out/experiments/inspect

Use ``--`` to separate the recorded command from the recorder's own flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.experiments import ExperimentSpec, run_command  # noqa: E402


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split recorder flags from the recorded command at the first ``--``."""
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="experiment_record.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--exp-id", required=True, help="experiment id, e.g. EXP-003")
    p.add_argument("--seed", type=int, default=1337, help="master seed recorded for the run")
    p.add_argument("--name", default="", help="short human-readable experiment name")
    p.add_argument("--out", default=None, help="output directory (default out/experiments/<exp-id>)")
    p.add_argument("--config", dest="config_path", default="", help="config file to embed verbatim")
    p.add_argument("--data", nargs="*", default=[], help="input files/dirs to hash for provenance")
    p.add_argument("--set", nargs="+", action="extend", default=[], metavar="key=value",
                   help="spec overrides: seed, output_dir, name, notes, params.<name>")
    p.add_argument("--tag", nargs="*", default=[], help="free-form tags")
    p.add_argument("--notes", default="", help="free-form notes recorded with the experiment")
    p.add_argument("--deterministic", action="store_true",
                   help="request torch deterministic algorithms (best effort, see docs)")
    p.add_argument("--timeout", type=float, default=None, help="timeout in seconds for the command")
    p.add_argument("--quiet", action="store_true", help="do not print the record summary")
    return p.parse_args(argv)


def main() -> int:
    recorder_argv, command = split_argv(sys.argv[1:])
    args = parse_args(recorder_argv)

    spec = ExperimentSpec(
        experiment_id=args.exp_id,
        seed=args.seed,
        name=args.name,
        output_dir=args.out or f"out/experiments/{args.exp_id}",
        data_paths=list(args.data),
        config_path=args.config_path,
        command=command,
        tags=list(args.tag),
        notes=args.notes,
        deterministic_mode=args.deterministic,
    ).with_overrides(args.set)

    try:
        outcome = run_command(spec, command, output_dir=args.out, timeout=args.timeout)
    except SystemExit:  # pragma: no cover - defensive
        raise
    except BaseException as exc:  # noqa: BLE001 - report, then fail with the original code
        print(f"[experiment] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[experiment] partial record (if written) under {spec.output_dir}", file=sys.stderr)
        return getattr(exc, "returncode", 1) or 1

    if not args.quiet:
        print()
        print(outcome.rendered)
        print(f"\n[experiment] record: {outcome.path}")
        print(f"[experiment] command: {' '.join(command) if command else '(none — provenance only)'}")
    print(json.dumps({"experiment_id": spec.experiment_id,
                      "record": str(outcome.path),
                      "fingerprint": outcome.record.content_fingerprint()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
