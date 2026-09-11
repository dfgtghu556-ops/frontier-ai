"""Deterministic hashing of experiment inputs.

Single source of truth: the byte-level SHA-256 helpers live in
:mod:`frontier_ai.tokenization.corpus` (Project 002 introduced them and its corpus
manifests already depend on them). This module **re-exports** them and adds the
multi-file / directory hashing that experiment records need, so there is exactly one
hashing implementation in the repository (D-024).

Documented rules
----------------
* **Algorithm**: SHA-256 over raw file bytes, hex digest.
* **One file**: the digest *is* the file's content digest — independent of where the file
  lives, so moving or renaming it does not change the experiment's data identity.
* **Several files or directories**: the digest is computed over a **manifest** of
  ``relative path + size + content digest`` lines. Names and structure are therefore part
  of the identity (renaming, adding or removing a file changes it).
* **Ordering**: manifest lines are sorted by relative POSIX path, so passing the same
  inputs in a different order yields the same digest and filesystem enumeration order is
  irrelevant.
* **Relative to what**: ``root`` if given, otherwise the **common ancestor directory** of
  all inputs, so digests do not depend on absolute machine paths. If inputs share no
  common ancestor, absolute paths are used (visible in the manifest, documented here).
* **Directories**: expanded recursively to files only; empty directories contribute
  nothing.
* **Symlinks**: ignored (never followed), reported under ``skipped``.
* **Records are not data** (D-033): expanding a *directory* skips the experiment record
  files an earlier recorded run wrote (``experiment.json`` / ``experiment.txt``). They are
  generated output, not source data, and they carry timestamps - hashing them would make
  every downstream fingerprint depend on *when* the upstream run happened. A record named
  explicitly in ``data_paths`` is still hashed: that is declared intent.
* **Missing input**: raises :class:`FileNotFoundError` naming the path. We never silently
  hash an incomplete dataset.
* **Empty input list**: raises :class:`ValueError` — an experiment with no inputs must say
  so explicitly rather than hashing "nothing".
* **Generated output**: is never hashed as a stand-in for source data.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Canonical implementations (Project 002). Re-exported, not reimplemented.
from ..tokenization.corpus import sha256_file, sha256_text

__all__ = [
    "HASH_ALGORITHM",
    "DataDigest",
    "sha256_file",
    "sha256_text",
    "hash_paths",
    "digest_paths",
]


HASH_ALGORITHM = "sha256"


@dataclass
class DataDigest:
    """Result of hashing a set of experiment inputs."""

    algorithm: str
    digest: str
    file_count: int
    total_bytes: int
    files: list[dict]
    skipped: list[str]
    root: str | None = None

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": self.files,
            "skipped": self.skipped,
            "root": self.root,
        }


@lru_cache(maxsize=1)
def _record_file_names() -> frozenset[str]:
    """Names of the files a recorded run writes (D-033).

    Imported lazily because :mod:`frontier_ai.experiments.record` imports this module, so
    importing it at module scope here would be circular.
    """
    from .record import RECORD_FILENAME

    return frozenset({RECORD_FILENAME, Path(RECORD_FILENAME).with_suffix(".txt").name})


def _collect(paths: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """Expand inputs to a de-duplicated list of files (symlinks skipped)."""
    found: dict[str, Path] = {}
    skipped: list[str] = []
    record_names = _record_file_names()

    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(
                f"experiment input not found: {path}. Refusing to hash an incomplete dataset — "
                "check the data_paths section of the experiment spec."
            )
        if path.is_symlink():
            skipped.append(str(path))
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.name in record_names and child.is_file():
                    continue                      # generated output, not source data
                if child.is_symlink():
                    skipped.append(str(child))
                elif child.is_file():
                    found.setdefault(str(child.resolve()), child.resolve())
        elif path.is_file():
            found.setdefault(str(path.resolve()), path.resolve())

    return [found[key] for key in sorted(found)], skipped


def _relative_root(files: Sequence[Path], root: str | Path | None) -> Path | None:
    """Root used to make recorded paths machine-independent."""
    if root:
        return Path(root).resolve()
    parents = [str(f.parent) for f in files]
    try:
        common = os.path.commonpath(parents)
    except ValueError:  # inputs share no common ancestor (e.g. different drives)
        return None
    return Path(common)


def hash_paths(
    paths: Iterable[str | Path],
    root: str | Path | None = None,
    max_files: int = 100_000,
) -> DataDigest:
    """Hash the inputs under ``paths`` and return a :class:`DataDigest`."""
    items = list(paths)
    if not items:
        raise ValueError(
            "no data paths provided: an experiment must declare its inputs explicitly "
            "(or hash an empty placeholder file if it genuinely has none)"
        )

    files, skipped = _collect(items)
    if not files:
        raise ValueError(f"no regular files found under: {items}")
    if len(files) > max_files:
        raise ValueError(f"too many input files ({len(files)} > {max_files}); raise max_files if intended")

    # A single input is identified by its content alone (path-independent).
    if len(files) == 1:
        digest = sha256_file(files[0])
        return DataDigest(
            algorithm=HASH_ALGORITHM,
            digest=digest,
            file_count=1,
            total_bytes=files[0].stat().st_size,
            files=[{"path": files[0].name, "bytes": files[0].stat().st_size, "sha256": digest}],
            skipped=skipped,
            root=str(files[0].parent),
        )

    base = _relative_root(files, root)
    entries: list[dict] = []
    total_bytes = 0
    for file_path in files:
        rel = str(file_path.relative_to(base).as_posix()) if base else str(file_path.as_posix())
        entries.append({"path": rel, "bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)})
        total_bytes += entries[-1]["bytes"]

    combined = hashlib.sha256()
    for entry in sorted(entries, key=lambda e: e["path"]):
        combined.update(f"{entry['path']}\t{entry['bytes']}\t{entry['sha256']}\n".encode())

    return DataDigest(
        algorithm=HASH_ALGORITHM,
        digest=combined.hexdigest(),
        file_count=len(entries),
        total_bytes=total_bytes,
        files=sorted(entries, key=lambda e: e["path"]),
        skipped=skipped,
        root=str(base) if base else None,
    )


def digest_paths(paths: Iterable[str | Path], root: str | Path | None = None) -> str:
    """Convenience wrapper returning only the combined digest."""
    return hash_paths(paths, root=root).digest
