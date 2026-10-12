"""EXP-047 Kaggle kernel: ONE session of training the next base model (D-052: 20 x 1280).

Do not run this file directly. ``scripts/run_kaggle_exp047.ps1`` copies it to ``run.py``, replaces
the commit placeholder with the commit being tested and pushes it as a private Kaggle script kernel
(Kaggle's T4 machine, which has two T4 GPUs; internet on). Three private datasets are attached: the
13-language EXP-037 token files (v2-slice1) and the two v2-slice2 backups ``frontier-v2-slice2-a`` and
``-b`` (D-051; their token files matched the manifests in the EXP-046 backup check). Two kernels,
``frontier-exp047-a`` and ``frontier-exp047-b``, take turns; each gets the other's latest output
mounted under ``/kaggle/input`` (``kernel_sources``), which is how the checkpoint travels from one
session to the next (as in EXP-043).

On Kaggle it: clones the repository at that commit and runs ``scripts/pretrain_exp047.py --part auto``
with the approved values typed in (D-052: ``--size 20x1280``; ``--cap-hours 380``, the estimate of
314 session-hours plus 20%, D-052's revisit trigger). Part 0 finds each slice under ``/kaggle/input``
and checks every manifest and token file (size, meta counts, SHA-256), runs the tests, the one-step
float64 device check and the checkpoint sha256 check against the last committed
``evals/results/EXP-047/session-*/summary.json``. **Session 1 is short (about 30-45 minutes):** Part 0
and the memory probe only (``--max-runs 0``: no learning-rate run starts), so whether 20 x 1280 fits on
a T4 is known before the long sessions (EXP-047 note 2026-10-13). From session 2 on: the
learning-rate check (one GPU) or the next stretch of the main run (two GPUs after the first main-run
session, D-050), until the 9-hour budget of this session. Written to ``/kaggle/working`` (the kernel
output): the small result files in ``EXP-047/session-<n>`` (plus ``samples.jsonl`` at the end), the
checkpoint for the next session in ``exp047_chain`` (about 6.1 GB at 20 x 1280: fp32 weights and the
AdamW moments) and, in the last session only, ``exp047_final/model_final.pt`` (about 2 GB).
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
OUT = Path("/kaggle/working/EXP-047")
CHAIN_OUT = Path("/kaggle/working/exp047_chain")
FINAL_DIR = Path("/kaggle/working/exp047_final")
INPUT = Path("/kaggle/input")  # the three datasets and the other kernel's output (the checkpoint)
SCRATCH = Path("/tmp/exp047")
MAX_HOURS = "9.0"  # one session; Kaggle stops a session at 12 hours (longer only for the LR check, below)
LR_SESSION_MAX = 11.0  # hours: below Kaggle's 12-hour limit (clone, install and upload need minutes)
PART0_MARGIN = 0.5  # hours: Part 0 (22 GB of SHA-256, tests, one-step check) before a learning-rate run
SIZE = "20x1280"  # D-052 (EXP-047 section 4 rule after the 2026-10-12 refit)
CAP_HOURS = "380"  # D-052: about 314 estimated session-hours + 20% (its revisit trigger)
PREVIOUS_DIR = "evals/results/EXP-047"
PROBE_ONLY_FIRST = ["--max-runs", "0"]  # session 1: Part 0 + memory probe, then stop (no learning-rate run)


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def session_hours(prev: Path) -> str:
    """--max-hours of this session (EXP-047 note 2026-10-13).

    The runner starts the next learning-rate run only if 1.1 x the longest finished run still fits in the
    session. At 20 x 1280 a run is estimated at about 7 hours (NOT VERIFIED); a 9-hour session would refuse
    every further run if one took more than about 7.9 hours. So while the check is not decided, the
    session is made long enough for that rule plus Part 0, never shorter than 9 and never longer than 11
    hours. Otherwise 9 hours (three sessions fit in a week of about 30 free hours, NOT VERIFIED)."""
    sessions = [
        json.loads(f.read_text(encoding="utf-8")) for f in sorted(prev.glob("session-*/summary.json"))
    ]
    if any((d.get("lr_choice") or {}).get("decided") for d in sessions):
        return MAX_HOURS
    minutes = [
        r["minutes"]
        for d in sessions
        for r in d.get("lr_runs", [])
        if r.get("session") == d.get("session") and r.get("status") == "done" and r.get("minutes")
    ]
    if not minutes:
        return MAX_HOURS
    need = 1.1 * max(minutes) / 60 + PART0_MARGIN
    if need <= float(MAX_HOURS):
        return MAX_HOURS
    return f"{min(LR_SESSION_MAX, need):.2f}"


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp047.ps1")
    sh(["nvidia-smi"])
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    try:
        import pytest  # noqa: F401
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "--quiet", "pytest"])
    # what is mounted (the slices are found and checked by pretrain_exp047.py itself)
    for p in sorted({p.parent for p in INPUT.rglob("*.meta.json")}):
        print(f"token folder mounted: {p}", flush=True)
    mounted = sorted(str(p.parent) for p in INPUT.rglob("chain.json"))
    print(f"chain.json files mounted under {INPUT}: {mounted or 'none'}", flush=True)
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "pretrain_exp047.py"),
        "--size",
        SIZE,
        "--cap-hours",
        CAP_HOURS,
        "--data-dir",
        str(INPUT),
        "--part",
        "auto",
        "--out",
        str(OUT),
        "--prev-dir",
        PREVIOUS_DIR,
        "--chain-in",
        str(INPUT),
        "--chain-out",
        str(CHAIN_OUT),
        "--final-dir",
        str(FINAL_DIR),
        "--scratch",
        str(SCRATCH),
        "--exp-id",
        "EXP-047",
        "--max-hours",
        session_hours(SRC / PREVIOUS_DIR),
        "--gpus",
        "2",
    ]
    if not list((SRC / PREVIOUS_DIR).glob("session-*/summary.json")):
        cmd += PROBE_ONLY_FIRST
        print("session 1: Part 0 and the memory probe only (no learning-rate run)", flush=True)
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(
        f"pretrain_exp047.py exit code {code} (summary.json in {OUT}/session-<n> says what happened)",
        flush=True,
    )
    return 0  # the kernel itself succeeded; the result files carry the verdict


if __name__ == "__main__":
    sys.exit(main())
