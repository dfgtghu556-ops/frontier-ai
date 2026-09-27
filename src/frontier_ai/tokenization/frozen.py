"""Frozen, hash-verified production tokenizers (EXP-030, D-040 / D-041).

A *frozen* tokenizer lives in a tracked directory ``tokenizers/<id>-<version>/``:

    tokenizers/frontier-tokenizer-v1/
        FREEZE.json                 identity record: hashes, lineage, gates, golden samples
        tokenizer/bpe_python.json   the artifact, byte-identical to the EXP-028 cell

Downstream code must load it through :func:`load_frontier_tokenizer` (or
:func:`load_frozen_tokenizer`), never through a loose path: the loader refuses to
return a tokenizer whose bytes, structure or behaviour differ from the freeze
record. Git must store the artifact byte-exactly (``.gitattributes``:
``tokenizers/** -text``) because the file written on Windows has CRLF line
endings and every hash below is over the raw bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base import TokenizerError
from .bpe_python import PythonBPE

REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_ROOT = REPO_ROOT / "tokenizers"
FRONTIER_TOKENIZER_V1 = "frontier-tokenizer-v1"
FREEZE_NAME = "FREEZE.json"
ARTIFACT_SUBDIR = "tokenizer"
ARTIFACT_FILE = "bpe_python.json"
FREEZE_SCHEMA = "frontier-tokenizer-freeze-v1"


class FrozenTokenizerError(TokenizerError):
    """The frozen tokenizer on disk does not match its freeze record."""


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_dir_sha256(artifact_dir: str | Path) -> str:
    """Fingerprint of an artifact directory: sorted (file name, file bytes), top level only.

    Byte-for-byte the algorithm of ``scripts/prepare_exp_b_data.py::_artifact_sha256``,
    which recorded the fingerprint of each EXP-029 tokenizer — so a frozen copy can be
    proven to be the exact artifact that experiment used (test-locked).
    """
    directory = Path(artifact_dir)
    h = hashlib.sha256()
    for name in sorted(p.name for p in directory.iterdir() if p.is_file()):
        h.update(name.encode("utf-8"))
        h.update(directory.joinpath(name).read_bytes())
    return h.hexdigest()


def read_freeze(freeze_dir: str | Path) -> dict[str, Any]:
    path = Path(freeze_dir) / FREEZE_NAME
    if not path.is_file():
        raise FrozenTokenizerError(f"no {FREEZE_NAME} in {freeze_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_structure(tokenizer: PythonBPE, expected: dict[str, Any]) -> list[str]:
    """Differences between a loaded tokenizer and the expected structure (empty = match)."""
    observed = {
        "vocab_size": tokenizer.vocab_size,
        "merges": len(tokenizer.merges),
        "pretoken": tokenizer._pretoken,  # noqa: SLF001 - no public accessor
        "special_tokens": list(tokenizer.special_token_ids),
    }
    return [
        f"{key}: expected {expected[key]!r}, found {observed[key]!r}"
        for key in ("vocab_size", "merges", "pretoken", "special_tokens")
        if key in expected and expected[key] != observed[key]
    ]


def load_frozen_tokenizer(freeze_dir: str | Path) -> tuple[PythonBPE, dict[str, Any]]:
    """Load a frozen tokenizer after checking bytes, structure and golden samples.

    Raises :class:`FrozenTokenizerError` on any mismatch; returns
    ``(tokenizer, freeze_record)`` otherwise.
    """
    freeze_dir = Path(freeze_dir)
    freeze = read_freeze(freeze_dir)
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("status") != "frozen":
        raise FrozenTokenizerError(
            f"{freeze_dir / FREEZE_NAME}: expected schema {FREEZE_SCHEMA!r} with status 'frozen'"
        )
    art = freeze["artifact"]
    artifact_dir = freeze_dir / ARTIFACT_SUBDIR

    present = sorted(p.name for p in artifact_dir.iterdir() if p.is_file()) if artifact_dir.is_dir() else []
    if present != sorted(art["files"]):
        raise FrozenTokenizerError(f"{artifact_dir}: files {present} != frozen {sorted(art['files'])}")
    for name, digest in art["files"].items():
        observed = file_sha256(artifact_dir / name)
        if observed != digest:
            raise FrozenTokenizerError(
                f"{artifact_dir / name}: sha256 {observed[:16]}… != frozen {digest[:16]}… "
                "(modified file, or git converted line endings — see .gitattributes)"
            )
    if artifact_dir_sha256(artifact_dir) != art["dir_sha256"]:
        raise FrozenTokenizerError(f"{artifact_dir}: directory fingerprint differs from the freeze record")

    tokenizer = PythonBPE.load(artifact_dir)
    problems = verify_structure(tokenizer, art)
    if problems:
        raise FrozenTokenizerError(f"{artifact_dir}: " + "; ".join(problems))
    for sample in freeze.get("golden_samples", []):
        if tokenizer.encode(sample["text"]) != sample["ids"]:
            raise FrozenTokenizerError(
                f"golden sample {sample['id']!r}: encoding differs from the freeze record"
            )
    return tokenizer, freeze


def frontier_tokenizer_v1_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else FROZEN_ROOT / FRONTIER_TOKENIZER_V1


def load_frontier_tokenizer(root: str | Path | None = None) -> PythonBPE:
    """Frontier Tokenizer v1 (D-040), verified against its freeze record."""
    tokenizer, _freeze = load_frozen_tokenizer(frontier_tokenizer_v1_dir(root))
    return tokenizer
