"""EXP-046 Kaggle kernel: check the dataset frontier-v2-slice2-text against the rewrite (CPU session).

Approved 2026-10-09 (founder "DO ALL", backup item b). Do not run this file directly.
``scripts/run_kaggle_exp046_text.ps1 -Verify`` copies it to ``run.py``, replaces the commit placeholder
and pushes it as a private Kaggle script kernel (CPU only, no GPU quota). Mounted under
``/kaggle/input``: the founder's private dataset ``frontier-v2-slice2-text``.

On Kaggle it clones the repository at that commit and runs ``scripts/slice2_text_backup.py verify``:
every text file must have the unzipped SHA-256 and line count recorded in the committed
``evals/results/EXP-046/text-rewrite/summary.json``. It only reads and hashes. Written to
``/kaggle/working/EXP-046/text-verify/summary.json`` and ``SUMMARY.txt``. No credentials are used.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-046/text-verify")
INPUT = Path("/kaggle/input")
WORKERS = 4


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp046_text.ps1")
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    print("mounted inputs:", sorted(str(p) for p in INPUT.iterdir()), flush=True)
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "slice2_text_backup.py"),
        "verify",
        "--input-dir",
        str(INPUT),
        "--out",
        str(OUT),
        "--workers",
        str(WORKERS),
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"slice2_text_backup.py exit code {code} (0 = PASS; summary.json says what was found)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the verdict


if __name__ == "__main__":
    sys.exit(main())
