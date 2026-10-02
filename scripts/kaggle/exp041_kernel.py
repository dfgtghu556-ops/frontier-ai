"""EXP-041 Kaggle kernel: one-step float64 device check, then the RoPE + GQA-2 runs (EXP-040 follow-up).

Do not run this file directly. ``scripts/run_kaggle_exp041.ps1`` copies it to ``run.py``, replaces
the commit placeholder with the commit being tested and pushes it as a private Kaggle script kernel
(one T4 GPU, internet on, the private 13-language token dataset attached).

On Kaggle it: clones the repository at that commit, finds the folder under ``/kaggle/input`` that
holds all 13 EXP-037 token files (the dataset uploaded for EXP-040; no new upload), runs
``scripts/gpu_lr_arch.py --part followup`` (sha256 checks, tests, Part A: the one-step float64
check for both architectures; Part B: the 6 RoPE + GQA-2 runs and one baseline control run, only
if Part A passes; EXP-040's baseline runs are read from the committed
``evals/results/EXP-040/summary.json``) and leaves the two small result files in
``/kaggle/working/EXP-041``. Checkpoints stay in ``/tmp`` and vanish with the session. No
credentials are needed or used: the repository is public and nothing is pushed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-041")
SCRATCH = Path("/tmp/exp041")
MAX_HOURS = "5.0"
MANIFEST = "evals/results/EXP-037/manifest.json"
PREVIOUS = "evals/results/EXP-040/summary.json"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp041.ps1")
    sh(["nvidia-smi"])
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    try:
        import pytest  # noqa: F401
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "--quiet", "pytest"])
    names = [f["path"] for f in json.loads((SRC / MANIFEST).read_text(encoding="utf-8"))["files"]]
    dirs = sorted({p.parent for p in Path("/kaggle/input").rglob("*.bin")})
    found = [d for d in dirs if all((d / n).exists() for n in names)]
    if len(found) != 1:
        raise SystemExit(
            f"expected one folder with all {len(names)} token files under /kaggle/input, found {found}"
        )
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "gpu_lr_arch.py"),
        "--data-dir",
        str(found[0]),
        "--out",
        str(OUT),
        "--scratch",
        str(SCRATCH),
        "--exp-id",
        "EXP-041",
        "--part",
        "followup",
        "--prev",
        PREVIOUS,
        "--max-hours",
        MAX_HOURS,
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"gpu_lr_arch.py exit code {code} (summary.json in {OUT} says what completed)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the verdict


if __name__ == "__main__":
    sys.exit(main())
