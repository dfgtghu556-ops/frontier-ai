"""EXP-046 Kaggle kernel: rewrite the slice-2 text files as ONE gzip member each (CPU session).

Approved 2026-10-09 (founder "DO ALL", backup item b). Do not run this file directly.
``scripts/run_kaggle_exp046_text.ps1`` copies it to ``run.py``, replaces the commit placeholder with the
commit being tested and pushes it as a private Kaggle script kernel (CPU only, internet on, no GPU
quota). Mounted read only under ``/kaggle/input``: the outputs of the build kernels
``frontier-exp046-build-a`` and ``-b`` (``kernel_sources``; the only full copy of the text).

On Kaggle it clones the repository at that commit and runs ``scripts/slice2_text_backup.py rewrite``.
Written to ``/kaggle/working`` (the kernel output): the 13 new ``slice2-text/<lang>.jsonl.gz`` files
(about 5.6 GB; they stay on Kaggle and become the founder's dataset ``frontier-v2-slice2-text``) and
the small reports ``EXP-046/text-rewrite/summary.json`` and ``SUMMARY.txt``. No credentials are used.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-046/text-rewrite")
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
        "rewrite",
        "--input-dir",
        str(INPUT),
        "--out",
        str(OUT),
        "--text-out",
        "/kaggle/working/slice2-text",
        "--workers",
        str(WORKERS),
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"slice2_text_backup.py exit code {code} (0 = PASS; summary.json says what was found)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the verdict


if __name__ == "__main__":
    sys.exit(main())
