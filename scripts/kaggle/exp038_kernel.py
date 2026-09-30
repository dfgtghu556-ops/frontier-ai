"""EXP-038 Kaggle kernel: GPU training bring-up at one pinned commit of the public repository.

Do not run this file directly. ``scripts/run_kaggle_exp038.ps1`` copies it to ``run.py``, replaces
the commit placeholder with the commit being tested and pushes it as a private Kaggle script kernel
(one T4 GPU, internet on, the private Hindi token dataset attached).

On Kaggle it: clones the repository at that commit, finds ``hi.bin`` under ``/kaggle/input``, runs
``scripts/gpu_bringup.py`` (which checks the file's sha256 and runs the tests first) and leaves the
three small result files in ``/kaggle/working/EXP-038``. Checkpoints stay in ``/tmp`` and vanish
with the session. No credentials are needed or used: the repository is public and nothing is pushed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-038")
SCRATCH = Path("/tmp/exp038")
MAX_HOURS = "5.5"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp038.ps1")
    sh(["nvidia-smi"])
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    try:
        import pytest  # noqa: F401
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "--quiet", "pytest"])
    found = sorted(Path("/kaggle/input").rglob("hi.bin"))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one hi.bin under /kaggle/input, found {[str(f) for f in found]}")
    data = found[0]
    if not data.with_suffix(".meta.json").exists():
        raise SystemExit(f"{data.with_suffix('.meta.json')} is missing")
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "gpu_bringup.py"),
        "--data",
        str(data),
        "--out",
        str(OUT),
        "--scratch",
        str(SCRATCH),
        "--exp-id",
        "EXP-038",
        "--max-hours",
        MAX_HOURS,
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"gpu_bringup.py exit code {code} (summary.json in {OUT} says what completed)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the verdict


if __name__ == "__main__":
    sys.exit(main())
