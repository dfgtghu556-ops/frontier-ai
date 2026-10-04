"""EXP-045 Kaggle kernel: test 4, the Belebele contamination check (CPU session, no GPU).

Do not run this file directly. ``scripts/run_kaggle_exp045.ps1`` copies it to ``run.py``, replaces
the commit placeholder with the commit being tested and pushes it as a private Kaggle script kernel
(CPU only, internet on, the private 13-language token dataset attached). It reads nothing from
EXP-043 (no ``kernel_sources``) and uses no GPU quota.

On Kaggle it: clones the repository at that commit, finds the folder under ``/kaggle/input`` that
holds all 13 EXP-037 token files, and runs ``scripts/belebele_contamination.py``: sha256 checks of
the token files, download of the 13 pinned Belebele files into ``/tmp`` (verified by hash; never
saved to the output, because Belebele is ShareAlike and must not be republished from here), then
the 13-token overlap scan over all 2.78 B training tokens. Written to ``/kaggle/working`` (the
kernel output): ``EXP-045/summary.json``, ``SUMMARY.txt`` and ``contamination.json`` (question keys,
counts and offsets only; no Belebele text). No credentials are needed or used.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-045")
INPUT = Path("/kaggle/input")
BELEBELE = Path("/tmp/belebele")
MANIFEST = "evals/results/EXP-037/manifest.json"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp045.ps1")
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    names = [f["path"] for f in json.loads((SRC / MANIFEST).read_text(encoding="utf-8"))["files"]]
    dirs = sorted({p.parent for p in INPUT.rglob("*.bin")})
    found = [d for d in dirs if all((d / n).exists() for n in names)]
    if len(found) != 1:
        raise SystemExit(
            f"expected one folder with all {len(names)} token files under {INPUT}, found {found}"
        )
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "belebele_contamination.py"),
        "--data-dir",
        str(found[0]),
        "--out",
        str(OUT),
        "--belebele-dir",
        str(BELEBELE),
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"belebele_contamination.py exit code {code} (no summary.json means it failed)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the outcome


if __name__ == "__main__":
    sys.exit(main())
