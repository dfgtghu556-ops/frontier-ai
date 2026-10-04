"""EXP-044 Kaggle kernel: the two-GPU data-parallel check (step 13), about 1.5 GPU-hours.

Do not run this file directly. ``scripts/run_kaggle_exp044.ps1`` copies it to ``run.py``, replaces
the commit placeholder with the commit being tested and pushes it as a private Kaggle script kernel
(T4 x2, internet on, the private 13-language token dataset attached). It reads nothing from EXP-043
(no ``kernel_sources``) and is started only between EXP-043 sessions.

On Kaggle it: clones the repository at that commit, finds the folder under ``/kaggle/input`` that
holds all 13 EXP-037 token files, and runs ``scripts/gpu_ddp_check.py``: Part 0 (sha256 checks and
the model/trainer/sampler/DDP tests), then 300 steps of the real 190 M model on one GPU and the same
300 steps on two GPUs (``torchrun``), then the three pre-registered rules. Written to
``/kaggle/working`` (the kernel output): ``EXP-044/summary.json`` and ``SUMMARY.txt`` (published),
and the two runs' logs in ``exp044_logs`` (for diagnosis only). No checkpoint is kept. No
credentials are needed or used: the repository is public and nothing is pushed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-044")
LOGS = Path("/kaggle/working/exp044_logs")
INPUT = Path("/kaggle/input")
SCRATCH = Path("/tmp/exp044")
MANIFEST = "evals/results/EXP-037/manifest.json"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp044.ps1")
    sh(["nvidia-smi"])
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    try:
        import pytest  # noqa: F401
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "--quiet", "pytest"])
    names = [f["path"] for f in json.loads((SRC / MANIFEST).read_text(encoding="utf-8"))["files"]]
    dirs = sorted({p.parent for p in INPUT.rglob("*.bin")})
    found = [d for d in dirs if all((d / n).exists() for n in names)]
    if len(found) != 1:
        raise SystemExit(
            f"expected one folder with all {len(names)} token files under {INPUT}, found {found}"
        )
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "gpu_ddp_check.py"),
        "--data-dir",
        str(found[0]),
        "--out",
        str(OUT),
        "--scratch",
        str(SCRATCH),
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    LOGS.mkdir(parents=True, exist_ok=True)
    for log in SCRATCH.glob("*/worker.log"):
        shutil.copy(log, LOGS / f"{log.parent.name}.log")
    print(f"gpu_ddp_check.py exit code {code} (summary.json in {OUT} says what happened)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the verdict


if __name__ == "__main__":
    sys.exit(main())
