"""Repository hygiene guards.

Why this file exists: an unanchored `data/` rule in `.gitignore` (intended for prepared
corpora at the repo root) also matched `src/frontier_ai/data/`, so the entire data
package was silently excluded from version control. Everything passed locally, and only a
fresh checkout revealed that the pushed branch could not import `frontier_ai.data` at all.

These tests fail loudly if a source file is ever git-ignored or untracked again.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "tests")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )


def _in_git_repo() -> bool:
    return _git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


pytestmark = pytest.mark.skipif(not _in_git_repo(), reason="not a git working tree")


def _source_files():
    files = []
    for directory in SOURCE_DIRS:
        files.extend(sorted((REPO_ROOT / directory).rglob("*.py")))
    return [f for f in files if "__pycache__" not in f.parts]


def test_every_source_file_is_tracked_by_git():
    tracked = set(_git("ls-files").stdout.splitlines())
    untracked = [
        f.relative_to(REPO_ROOT).as_posix()
        for f in _source_files()
        if f.relative_to(REPO_ROOT).as_posix() not in tracked
    ]
    assert not untracked, (
        "source files are not tracked by git (an overly broad .gitignore rule is the usual "
        f"cause; see D-023): {untracked}"
    )


def test_no_source_file_is_git_ignored():
    ignored: list[str] = []
    for path in _source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        result = _git("check-ignore", "-q", rel)
        if result.returncode == 0:
            ignored.append(rel)
    assert not ignored, (
        f"source files are excluded by .gitignore: {ignored}. Anchor project-artifact rules "
        "with a leading '/' (e.g. /data/) so they only match the repository root."
    )


def test_artifact_ignore_rules_are_anchored_to_the_repo_root():
    """`data/` would also match `src/frontier_ai/data/`; `/data/` only matches the root."""
    lines = [
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for rule in ("data/", "out/", "artifacts/"):
        if rule in lines:
            pytest.fail(f".gitignore contains the unanchored rule '{rule}'; use '/{rule}' instead")


def test_package_imports_from_a_clean_tree():
    """The failure this file guards against: a tracked-but-incomplete package."""
    import os
    import sys

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c", "import frontier_ai.data.dataset, frontier_ai.data.tokenizer, "
                               "frontier_ai.data.synthetic; print('ok')"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"the data package is not importable: {result.stderr}"
