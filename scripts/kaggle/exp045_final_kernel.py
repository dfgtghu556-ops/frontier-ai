"""EXP-045 Kaggle kernel: the final evaluation of the EXP-043 model (tests 1, 2, 3 and 5) on a GPU.

Approved with the EXP-045 plan on 2026-10-04: tests 1-3 and 5 run in ONE short GPU session after
EXP-043 has finished (estimated under 0.5 GPU-hours, NOT VERIFIED; cap 1). Test 4 and the coverage
re-scan ran earlier on CPU sessions (``exp045_kernel.py``); their committed files are inputs here and
are never touched.

Do not run this file directly. ``scripts/run_kaggle_exp045_final.ps1`` copies it to ``run.py``,
replaces the commit placeholder with the commit being tested and pushes it as a private Kaggle script
kernel (Kaggle's T4 machine, internet on). Two inputs are mounted under ``/kaggle/input``:
- the output of the EXP-043 kernel that ran the LAST session (``kernel_sources``; the runner picks
  ``frontier-exp043-a`` or ``-b`` from the session number), which holds
  ``exp043_final/model_final.pt``;
- the private dataset ``frontier-heldout-v1-text`` (the protected suite's texts, for test 1 only).

On Kaggle it: clones the repository at that commit; reads the expected sha256 of ``model_final.pt``
from the ONE committed ``evals/results/EXP-043/session-*/summary.json`` that says ``complete``
(it refuses if EXP-043 is not complete); finds the mounted ``model_final.pt`` whose sha256 matches
(it refuses if none does); finds ``heldout-v1.jsonl`` (without it, test 1 is reported as NOT RUN,
never estimated); then runs ``scripts/eval_exp045.py`` with test 4's ``contamination.json`` and the
coverage re-scan's ``coverage.json`` from the repository, so Belebele is reported with and without
the flagged questions. Belebele is downloaded into ``/tmp`` by pinned revision and checked by hash;
it never goes into the kernel output (ShareAlike).

Written to ``/kaggle/working`` (the kernel output): ``EXP-045/final/summary.json``, ``SUMMARY.txt``,
``samples.jsonl`` and ``belebele_items.jsonl`` (keys and choices only, no Belebele text), and the
weights-only fp16 copy ``exp045_fp16/model_fp16.pt`` (about 0.38 GB) for the founder to download.
No credentials are needed or used: the repository is public and nothing is pushed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-045/final")
FP16 = Path("/kaggle/working/exp045_fp16/model_fp16.pt")
INPUT = Path("/kaggle/input")
BELEBELE = Path("/tmp/belebele")
EXP043_DIR = "evals/results/EXP-043"
CONTAMINATION = "evals/results/EXP-045/contamination.json"
COVERAGE = "evals/results/EXP-045/coverage/coverage.json"
WEIGHTS_NAME = "model_final.pt"
HELDOUT_NAME = "heldout-v1.jsonl"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def final_record(results_dir: Path) -> dict:
    """The committed EXP-043 session that completed the run: ``{session, sha256, step}``."""
    done = []
    for f in sorted(results_dir.glob("session-*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("complete"):
            done.append((f.parent.name, s))
    if len(done) != 1:
        raise SystemExit(
            f"expected exactly one complete EXP-043 session under {results_dir}, found {len(done)}: "
            "the final evaluation runs only after EXP-043 has finished"
        )
    name, s = done[0]
    mf = (s.get("final") or {}).get("model_final") or {}
    if len(str(mf.get("sha256", ""))) != 64:
        raise SystemExit(f"{name}/summary.json is complete but records no model_final.pt sha256")
    return {"session": name, "sha256": mf["sha256"], "step": mf.get("step")}


def find_weights(input_dir: Path, sha256: str) -> Path:
    """The mounted ``model_final.pt`` whose sha256 is the committed one (refuses otherwise)."""
    candidates = sorted(input_dir.rglob(WEIGHTS_NAME))
    seen = []
    for p in candidates:
        h = sha256_file(p)
        seen.append(f"{p} {h[:12]}")
        if h == sha256:
            return p
    raise SystemExit(
        f"no mounted {WEIGHTS_NAME} has the committed sha256 {sha256[:12]}...; found: {seen or 'none'}"
    )


def find_heldout(input_dir: Path) -> Path | None:
    """The protected texts for test 1, or None (then test 1 is reported as NOT RUN)."""
    found = sorted(input_dir.rglob(HELDOUT_NAME))
    if len(found) > 1:
        raise SystemExit(f"more than one {HELDOUT_NAME} under {input_dir}: {found}")
    return found[0] if found else None


def eval_command(src: Path, weights: Path, heldout: Path | None) -> list[str]:
    cmd = [
        sys.executable,
        str(src / "scripts" / "eval_exp045.py"),
        "--weights",
        str(weights),
        "--out",
        str(OUT),
        "--belebele-dir",
        str(BELEBELE),
        "--contamination",
        str(src / CONTAMINATION),
        "--coverage",
        str(src / COVERAGE),
        "--export-fp16",
        str(FP16),
    ]
    if heldout is not None:
        cmd += ["--heldout-jsonl", str(heldout)]
    return cmd


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp045_final.ps1")
    sh(["nvidia-smi"])
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    rec = final_record(SRC / EXP043_DIR)
    print(f"EXP-043 finished in {rec['session']} (step {rec['step']}); sha256 {rec['sha256']}", flush=True)
    weights = find_weights(INPUT, rec["sha256"])
    print(f"weights: {weights} (sha256 matches)", flush=True)
    heldout = find_heldout(INPUT)
    print(f"held-out texts: {heldout or 'NOT FOUND - test 1 will be reported as NOT RUN'}", flush=True)
    cmd = eval_command(SRC, weights, heldout)
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"eval_exp045.py exit code {code} (no summary.json means it failed)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the outcome


if __name__ == "__main__":
    sys.exit(main())
