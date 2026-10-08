"""EXP-046 Kaggle kernel: check the slice-2 backup datasets against the build manifests (CPU session).

Approved 2026-10-09 ("approve A B C", item B). Do not run this file directly.
``scripts/run_kaggle_exp046_check.ps1`` copies it to ``run.py``, replaces the commit placeholder with
the commit being tested and pushes it as a private Kaggle script kernel (CPU only, internet on, no GPU
quota). Four inputs are mounted under ``/kaggle/input``: the private datasets ``frontier-v2-slice2-a``
and ``-b`` (the backup made by the founder on 2026-10-09) and, for comparison, the outputs of the build
kernels ``frontier-exp046-build-a`` and ``-b`` (the originals; ``kernel_sources``, read only).

On Kaggle it clones the repository at that commit and runs ``scripts/check_slice2_backup.py``: every
token file's size and SHA-256 against ``evals/results/EXP-046/build-{A,B}/manifest.json``, the meta
files, and every text file (unzipped by Kaggle) against the original ``.jsonl.gz`` unzipped. It only
reads and hashes. Written to ``/kaggle/working`` (the kernel output): ``EXP-046/backup-check/summary.json``
and ``SUMMARY.txt``. No credentials are needed or used.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-046/backup-check")
INPUT = Path("/kaggle/input")
WORKERS = 4


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp046_check.ps1")
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    print("mounted inputs:", sorted(str(p) for p in INPUT.iterdir()), flush=True)
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "check_slice2_backup.py"),
        "--input-dir",
        str(INPUT),
        "--out",
        str(OUT),
        "--workers",
        str(WORKERS),
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"check_slice2_backup.py exit code {code} (0 = PASS; summary.json says what was found)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the verdict


if __name__ == "__main__":
    sys.exit(main())
