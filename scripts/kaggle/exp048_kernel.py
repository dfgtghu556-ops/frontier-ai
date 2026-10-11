"""EXP-048 Kaggle kernel: measure FineWeb-2 for our 12 Indian languages (CPU session, no GPU).

Do not run this file directly. ``scripts/run_kaggle_exp048.ps1`` copies it to ``run.py``, fills in
the commit, and pushes it as the private Kaggle script kernel ``<username>/frontier-exp048``: CPU
only, internet on, four private inputs:

* the v2-slice1 token dataset (EXP-037) and the two v2-slice2 token datasets (EXP-046 A and B),
  used only to find FineWeb-2 documents we already have (exact text);
* the held-out texts (``heldout-v1.jsonl``), so the protected-suite check runs as in a real build.

On Kaggle it clones the repository at that commit and runs ``scripts/measure_fineweb2.py``. That
script pins the FineWeb-2 revision, downloads ONE train file per language into ``/tmp`` (checked by
SHA-256, deleted after use) plus the 13 pinned Belebele files, and writes only small result files:

* ``EXP-048/``: summary.json, SUMMARY.txt, samples.jsonl (the runner downloads only this folder and
  publishes it to ``evals/results/EXP-048/``).

No FineWeb-2 corpus is kept. No credentials are needed or used.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
BELEBELE = Path("/tmp/belebele")
SCRATCH = Path("/tmp/fw2")
HELDOUT_NAME = "heldout-v1.jsonl"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp048.ps1")
    held = sorted(INPUT.rglob(HELDOUT_NAME))
    if len(held) != 1:
        raise SystemExit(f"expected one {HELDOUT_NAME} under {INPUT} (the held-out input), found {held}")
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "measure_fineweb2.py"),
        "--input-dir",
        str(INPUT),
        "--heldout-jsonl",
        str(held[0]),
        "--belebele-dir",
        str(BELEBELE),
        "--work",
        str(SCRATCH),
        "--out",
        str(WORKING / "EXP-048"),
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"measure_fineweb2.py exit code {code} (summary.json says what was measured)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the outcome


if __name__ == "__main__":
    sys.exit(main())
