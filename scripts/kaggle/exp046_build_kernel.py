"""EXP-046 Kaggle kernel: build FrontierCorpus v2-slice2, part A or B (CPU session, no GPU).

Do not run this file directly. ``scripts/run_kaggle_exp046_build.ps1 -Part A`` (or ``B``) copies it
to ``run.py``, fills in the commit and the part, and pushes it as the private Kaggle script kernel
``<username>/frontier-exp046-build-a`` (or ``-b``): CPU only, internet on, two private inputs:

* the EXP-037 token dataset (v2-slice1; needed to remove slice-1 documents, safeguard 2);
* the held-out texts (``heldout-v1.jsonl`` from ``scripts/export_heldout_text.py``; the build
  refuses to run without the protected suite, and checks the texts against SUITE.json).

On Kaggle it clones the repository at that commit and runs ``scripts/build_slice2.py``. That
script downloads the part's pinned Sangraha files and the 13 pinned Belebele files into ``/tmp``
(verified by hash; Belebele never goes into the output), builds and packs every language with 4
worker processes, and stops cleanly before Kaggle's 12-hour limit. Kernel output:

* ``EXP-046/build-<part>/``: summary.json, SUMMARY.txt, manifest.json, samples.jsonl (small; the
  runner downloads only this folder and publishes it to ``evals/results/EXP-046/build-<part>/``);
* ``slice2-<part>/``: ``<lang>.jsonl.gz`` (text) + ``<lang>.bin`` / ``<lang>.meta.json`` (tokens),
  manifest.json and ATTRIBUTION.txt. This stays on Kaggle; a later training kernel mounts it.

No credentials are needed or used.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
PART = "__BUILD_PART__"
SRC = Path("/tmp/frontier-ai")
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
BELEBELE = Path("/tmp/belebele")
SCRATCH = Path("/tmp/slice2")
MANIFEST = "evals/results/EXP-037/manifest.json"
HELDOUT_NAME = "heldout-v1.jsonl"
WORKERS = 4


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp046_build.ps1")
    if PART not in ("A", "B"):
        raise SystemExit("the part placeholder was not filled in; use scripts/run_kaggle_exp046_build.ps1")
    sh(["git", "clone", "--quiet", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
    names = [f["path"] for f in json.loads((SRC / MANIFEST).read_text(encoding="utf-8"))["files"]]
    dirs = sorted({p.parent for p in INPUT.rglob("*.bin")})
    found = [d for d in dirs if all((d / n).exists() for n in names)]
    if len(found) != 1:
        raise SystemExit(f"expected one folder with all {len(names)} token files in {INPUT}: {found}")
    held = sorted(INPUT.rglob(HELDOUT_NAME))
    if len(held) != 1:
        raise SystemExit(f"expected one {HELDOUT_NAME} under {INPUT} (the held-out input), found {held}")
    cmd = [
        sys.executable,
        str(SRC / "scripts" / "build_slice2.py"),
        "--part",
        PART,
        "--tokens-dir",
        str(found[0]),
        "--heldout-jsonl",
        str(held[0]),
        "--belebele-dir",
        str(BELEBELE),
        "--work",
        str(SCRATCH),
        "--data-out",
        str(WORKING / f"slice2-{PART}"),
        "--out",
        str(WORKING / "EXP-046" / f"build-{PART}"),
        "--workers",
        str(WORKERS),
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=SRC).returncode
    print(f"build_slice2.py exit code {code} (summary.json says what was done)", flush=True)
    return 0  # the kernel itself succeeded; the result files carry the outcome


if __name__ == "__main__":
    sys.exit(main())
