"""Shared CLI helpers for the tokenizer scripts.

Kept in the library (not in ``scripts/``) so the behaviour is unit-testable and identical
across the prepare/train/evaluate/compare commands.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .artifact import ArtifactManifest, load_artifact
from .base import SubwordTokenizer
from .corpus import CorpusManifest, sha256_file


def resolve_corpus_dir(path: str | Path) -> Path:
    """Accept either a corpus directory or a ``train.txt`` path; return the directory."""
    p = Path(path)
    if p.is_dir():
        return p
    if p.is_file():
        return p.parent
    raise FileNotFoundError(
        f"corpus path not found: {p}. Run scripts/tokenizer_prepare_corpus.py first."
    )


def corpus_meta(corpus_dir: str | Path) -> dict[str, object]:
    """Provenance block for manifests and reports."""
    base = Path(corpus_dir)
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        manifest = CorpusManifest.from_dict(
            __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        )
        return {"dir": str(base), **manifest.to_dict()}
    train = base / "train.txt"
    return {
        "dir": str(base),
        "corpus_id": base.name,
        "corpus_version": "unknown",
        "train_path": str(train),
        "train_sha256": sha256_file(train) if train.exists() else None,
    }


def train_path_for(corpus_dir: str | Path) -> Path:
    base = Path(corpus_dir)
    candidate = base / "train.txt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no train.txt in {base}. Run scripts/tokenizer_prepare_corpus.py first.")


def load_tokenizer_artifacts(
    artifact_dirs: Sequence[str | Path], labels: Sequence[str] | None = None
) -> list[tuple[str, SubwordTokenizer, ArtifactManifest]]:
    """Load ``(label, tokenizer, manifest)`` triples, labelling by directory name."""
    if labels and len(labels) != len(artifact_dirs):
        raise ValueError("--label must be provided once per --tokenizer")
    out: list[tuple[str, SubwordTokenizer, ArtifactManifest]] = []
    seen: dict[str, str] = {}
    for i, directory in enumerate(artifact_dirs):
        tokenizer, manifest = load_artifact(directory)
        label = labels[i] if labels else Path(directory).name
        if label in seen:
            raise ValueError(f"duplicate label '{label}' for artifacts {seen[label]} and {directory}")
        seen[label] = str(directory)
        out.append((label, tokenizer, manifest))
    return out


def manifest_summary(manifest: ArtifactManifest) -> str:
    return (
        f"impl={manifest.impl} version={manifest.impl_version} vocab={manifest.vocab_size} "
        f"specials={manifest.special_tokens}"
    )
