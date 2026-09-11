"""Git provenance: which exact source revision produced a result.

Rules that matter here:

* **Never invent a SHA.** If git is unavailable, or the directory is not a repository, the
  result is an explicit ``available=False`` with a human-readable ``reason``.
* **Dirty is sticky information.** A run from a dirty working tree can never be reproduced
  from its SHA alone, so it is recorded and surfaced loudly rather than silently ignored.
* Detached HEAD (common in CI) yields an empty branch, which is recorded as ``null``
  rather than guessed.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

GIT_TIMEOUT_SECONDS = 10


@dataclass
class GitInfo:
    """Source-control state at the time an experiment ran."""

    available: bool
    commit: str | None = None
    branch: str | None = None
    dirty: bool = False
    dirty_files: list[str] = field(default_factory=list)
    commit_subject: str = ""
    remote: str | None = None
    reason: str = ""  # populated when available is False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def unavailable(cls, reason: str) -> GitInfo:
        return cls(available=False, reason=reason)

    @property
    def reproducible_from_commit(self) -> bool:
        """True only if the run can be reproduced from the recorded commit alone."""
        return bool(self.available and self.commit and not self.dirty)

    def describe(self) -> str:
        if not self.available:
            return f"unavailable ({self.reason})"
        short = (self.commit or "")[:8]
        state = "dirty" if self.dirty else "clean"
        branch = self.branch or "detached HEAD"
        return f"{short} on {branch} ({state})"


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )


def capture_git_info(repo_path: str | Path = ".") -> GitInfo:
    """Capture commit / branch / dirty state for ``repo_path``.

    Returns :meth:`GitInfo.unavailable` (never raises) when git cannot answer: not a
    repository, git missing, or the call times out.
    """
    repo = Path(repo_path).resolve()
    if not repo.exists():
        return GitInfo.unavailable(f"path does not exist: {repo}")

    try:
        inside = _run_git(repo, "rev-parse", "--is-inside-work-tree")
    except (OSError, subprocess.SubprocessError) as exc:  # git missing / killed
        return GitInfo.unavailable(f"git call failed: {exc}")

    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return GitInfo.unavailable("not a git working tree")

    try:
        commit = _run_git(repo, "rev-parse", "HEAD")
        if commit.returncode != 0:  # repository with no commits yet
            return GitInfo.unavailable("repository has no commits")

        sha = commit.stdout.strip()
        branch_proc = _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        branch = branch_proc.stdout.strip() or None
        if branch == "HEAD":  # detached HEAD
            branch = None

        status = _run_git(repo, "status", "--porcelain")
        dirty_files = [line[3:].strip() for line in status.stdout.splitlines() if line.strip()]

        subject_proc = _run_git(repo, "log", "-1", "--pretty=%s")
        subject = subject_proc.stdout.strip() if subject_proc.returncode == 0 else ""

        remote_proc = _run_git(repo, "config", "--get", "remote.origin.url")
        remote = remote_proc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - defensive
        return GitInfo.unavailable(f"git call failed: {exc}")

    return GitInfo(
        available=True,
        commit=sha,
        branch=branch,
        dirty=bool(dirty_files),
        dirty_files=dirty_files[:50],  # bounded: a dirty tree can have thousands of files
        commit_subject=subject,
        remote=remote,
    )


def git_section(info: GitInfo) -> dict[str, Any]:
    """Dict form recorded under the ``code`` section of an experiment record."""
    data = info.to_dict()
    data["reproducible_from_commit"] = info.reproducible_from_commit
    return data
