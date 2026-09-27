#!/usr/bin/env python3
"""Copy the small, reviewable evaluation outputs of one experiment into the repo.

    python scripts/publish_eval_results.py --exp-id EXP-031

Copies, for every report directory under ``out/eval/<exp-id>/`` (recursively), the files
``report.json``, ``report.txt``, ``compare.json``, ``compare.txt`` and ``experiment.json``
into ``evals/results/<exp-id>/`` with the same relative layout, normalised to LF line
endings. ``per_document.jsonl`` (large, regenerable by re-running the report) stays in
``out/``. Refuses to overwrite a different existing file (exit 2): published results are
evidence and are not silently replaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PUBLISHED = ("report.json", "report.txt", "compare.json", "compare.txt", "experiment.json")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a legacy console code page
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--src", default=None, help="default out/eval/<exp-id>")
    p.add_argument("--dest", default=None, help="default evals/results/<exp-id>")
    args = p.parse_args()
    src = Path(args.src or f"out/eval/{args.exp_id}")
    dest = Path(args.dest or f"evals/results/{args.exp_id}")
    if not src.is_dir():
        print(f"[publish] source not found: {src}", file=sys.stderr)
        return 2
    files = sorted(f for f in src.rglob("*") if f.is_file() and f.name in PUBLISHED)
    if not files:
        print(f"[publish] nothing to publish under {src}", file=sys.stderr)
        return 2
    planned = []
    for f in files:
        target = dest / f.relative_to(src)
        data = f.read_bytes().replace(b"\r\n", b"\n")
        if target.exists() and target.read_bytes() != data:
            print(f"[publish] REFUSED: {target} exists with different content", file=sys.stderr)
            return 2
        planned.append((target, data))
    for target, data in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"[publish] {len(planned)} files -> {dest}")
    for target, _ in planned:
        print(f"  {target.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
