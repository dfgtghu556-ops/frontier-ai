#!/usr/bin/env python3
"""Write the Lab OS dashboard snapshot from the repository's own records.

    python scripts/export_lab_state.py            # (re)write apps/lab-os/src/data/lab_state.json
    python scripts/export_lab_state.py --check    # exit 1 if the committed snapshot is out of date

Read-only on every record; see ``frontier_ai.lab_state`` for what is read and why. Run it after
changing EXPERIMENTS.md, DECISIONS.md, lab/registry.json or evaluation results, then commit the
snapshot together with the change (a test fails while the snapshot is stale).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_ai.lab_state import DEFAULT_OUTPUT, LabStateError, build_state, render


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--out", default=None, help=f"default: <root>/{DEFAULT_OUTPUT}")
    p.add_argument("--check", action="store_true", help="verify the snapshot is current; write nothing")
    args = p.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / DEFAULT_OUTPUT
    try:
        text = render(build_state(root))
    except LabStateError as exc:
        print(f"[lab-state] {exc}", file=sys.stderr)
        return 2
    if args.check:
        current = out.read_text(encoding="utf-8") if out.is_file() else None
        if current != text:
            print(
                f"[lab-state] {out} is out of date; run: python scripts/export_lab_state.py", file=sys.stderr
            )
            return 1
        print(f"[lab-state] {out} is current")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"[lab-state] wrote {out} ({len(text.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
