"""EXP-043 Kaggle kernel: ONE session of training the step-12 model (D-049, 14 x 896).

Do not run this file directly. ``scripts/run_kaggle_exp043.ps1`` copies it to ``run.py``, replaces
the commit placeholder with the commit being tested and pushes it as a private Kaggle script kernel
(one T4 GPU, internet on, the private 13-language token dataset attached). Two kernels,
``frontier-exp043-a`` and ``frontier-exp043-b``, take turns; each gets the other's latest output
mounted under ``/kaggle/input`` (``kernel_sources``), which is how the checkpoint travels from one
session to the next.

On Kaggle it: clones the repository at that commit, finds the folder under ``/kaggle/input`` that
holds all 13 EXP-037 token files, and runs ``scripts/gpu_pretrain.py --part auto``: Part 0 (sha256
checks, tests, the one-step float64 device check, the checkpoint sha256 check against the last
committed ``evals/results/EXP-043/session-*/summary.json``), then the learning-rate check or the
next stretch of the main run, until the 9-hour budget of this session. Written to
``/kaggle/working`` (the kernel output): the two small result files in ``EXP-043/session-<n>``
(plus ``samples.jsonl`` at the end), the checkpoint for the next session in ``exp043_chain``
(about 2.3 GB) and, in the last session only, ``exp043_final/model_final.pt`` (about 760 MB).
No credentials are needed or used: the repository is public and nothing is pushed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-043")
CHAIN_OUT = Path("/kaggle/working/exp043_chain")
FINAL_DIR = Path("/kaggle/working/exp043_final")
CHAIN_IN = Path("/kaggle/input")
SCRATCH = Path("/tmp/exp043")
MAX_HOURS = "9.0"  # one session; Kaggle stops a session at 12 hours
MANIFEST = "evals/results/EXP-037/manifest.json"
PREVIOUS_DIR = "evals/results/EXP-043"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp043.ps1")
    sh(["nvidia-smi"])
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    try:
        import pytest  # noqa: F401
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "--quiet", "pytest"])
    names = [f["path"] for f in json.loads((SRC / MANIFEST).read_text(encoding="utf-8"))["files"]]
    dirs = sorted({p.parent for p in CHAIN_IN.rglob("*.bin")})
    found = [d for d in dirs if all((d / n).exists() for n in names)]
    if len(found) != 1:
        raise SystemExit(
            f"expected one folder with all {len(names)} token files under {CHAIN_IN}, found {found}"
        )
    mounted = sorted(str(p.parent) for p in CHAIN_IN.rglob("chain.json"))
    print(f"chain.json files mounted under {CHAIN_IN}: {mounted or 'none'}", flush=True)
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "gpu_pretrain.py"),
        "--data-dir",
        str(found[0]),
        "--part",
        "auto",
        "--out",
        str(OUT),
        "--prev-dir",
        PREVIOUS_DIR,
        "--chain-in",
        str(CHAIN_IN),
        "--chain-out",
        str(CHAIN_OUT),
        "--final-dir",
        str(FINAL_DIR),
        "--scratch",
        str(SCRATCH),
        "--exp-id",
        "EXP-043",
        "--max-hours",
        MAX_HOURS,
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(
        f"gpu_pretrain.py exit code {code} (summary.json in {OUT}/session-<n> says what happened)", flush=True
    )
    return 0  # the kernel itself succeeded; the result files carry the verdict


if __name__ == "__main__":
    sys.exit(main())
