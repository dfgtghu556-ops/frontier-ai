"""Stages 7–8: seeded shuffle, character-based packing into text shards.

The pipeline's physical output is **pre-tokenization text** (D-037): one shard
is one UTF-8 text file holding whole documents, one document per line. Tokens
belong to the tokenizer experiments (MASTER_CONTEXT §37 steps 5–7) — this
module never looks at a tokenizer.

Determinism
-----------
* ``seeded_shuffle`` uses a fresh ``random.Random(seed)`` — no global RNG, the
  seed is recorded in the manifest, and the same input + seed gives the same
  order every run.
* ``pack`` fills shards by **character budget**: documents are appended in the
  order given (post-shuffle) until the next document would push the shard past
  ``max_chars_per_shard``; a document never moves across shards or is split
  (documents are atomic). Two builds therefore write byte-identical shard
  files.
* Files are written with explicit ``newline="\\n"`` so the bytes do not depend
  on the operating system (the pilot runs on a Windows machine).

Shard identity
--------------
Each shard carries the SHA-256 of its exact bytes. The dataset manifest records
one hash per shard, and ``check_shards`` re-hashes the files on disk — this is
the ``--check`` mode of the builder: a reader can verify an existing build
without re-running anything.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from frontier_ai.corpus.pipeline import PipelineDocument, sha256_text


@dataclass(frozen=True)
class Shard:
    """One output file: whole documents, one per line."""

    name: str  # file stem, e.g. "train-000002"
    documents: tuple[PipelineDocument, ...]

    @property
    def text(self) -> str:
        # documents are single-line (the corpus document model), so re-splitting
        # the file on newlines reproduces the document sequence exactly
        return "\n".join(d.text for d in self.documents)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "documents": len(self.documents),
            "chars": self.chars,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


def seeded_shuffle(documents: Sequence[PipelineDocument], seed: int) -> list[PipelineDocument]:
    """Deterministic shuffle under ``seed`` (fresh generator, no global state)."""
    docs = list(documents)
    random.Random(seed).shuffle(docs)
    return docs


def pack(documents: Sequence[PipelineDocument], max_chars_per_shard: int, prefix: str) -> tuple[Shard, ...]:
    """Pack documents (in order) into shards of at most ``max_chars_per_shard``.

    The budget counts the joining newlines. A single document larger than the
    budget still gets its own shard — documents are atomic (in the pilot the
    quality ceiling of 20,000 chars is far below any shard budget, so this is
    unreachable there).
    """
    if max_chars_per_shard < 1:
        raise ValueError("max_chars_per_shard must be >= 1")
    shards: list[Shard] = []
    current: list[PipelineDocument] = []
    current_chars = 0
    for doc in documents:
        add = doc.chars + (1 if current else 0)  # the joining newline, if any
        if current and current_chars + add > max_chars_per_shard:
            shards.append(Shard(name=f"{prefix}-{len(shards):06d}", documents=tuple(current)))
            current, current_chars = [], 0
            add = doc.chars
        current.append(doc)
        current_chars += add
    if current:
        shards.append(Shard(name=f"{prefix}-{len(shards):06d}", documents=tuple(current)))
    return tuple(shards)


def write_shards(shards: Sequence[Shard], directory: str | Path) -> tuple[Path, ...]:
    """Write the shards as ``<directory>/<name>.txt`` (UTF-8, LF newlines)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for shard in shards:
        path = directory / f"{shard.name}.txt"
        path.write_text(shard.text, encoding="utf-8", newline="\n")
        paths.append(path)
    return tuple(paths)


def shard_hashes(directory: str | Path) -> dict[str, str]:
    """On-disk shard name -> SHA-256 of the file bytes (for ``--check``)."""
    out: dict[str, str] = {}
    for path in sorted(Path(directory).glob("*.txt")):
        out[path.stem] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def check_shards(directory: str | Path, expected: dict[str, str]) -> list[str]:
    """Compare on-disk shard files against ``{name: sha256}``.

    Returns a list of problems (empty = the build on disk matches the
    manifest). Files are never modified.
    """
    problems: list[str] = []
    disk = shard_hashes(directory)
    for name in sorted(expected):
        if name not in disk:
            problems.append(f"missing shard file {name}.txt")
    for name in sorted(disk):
        if name not in expected:
            problems.append(f"unexpected shard file {name}.txt (not in manifest)")
        elif disk[name] != expected[name]:
            problems.append(
                f"shard {name}.txt hash mismatch:\n  manifest {expected[name]}\n  on disk  {disk[name]}"
            )
    return problems
