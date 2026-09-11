"""Tokenizer artifacts: the tokenizer files plus a provenance manifest.

An artifact directory always contains the implementation's own files
(``tokenizer.json`` / ``bpe_python.json``) **and** a ``manifest.json`` recording what was
trained, on what, with which versions. Without the manifest, a tokenizer is just a blob
nobody can reproduce six months later.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .base import SubwordTokenizer
from .registry import create

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = "1.0"


@dataclass
class ArtifactManifest:
    """Provenance for one trained tokenizer."""

    schema_version: str
    experiment_id: str
    created_at: str
    impl: str
    impl_version: str
    vocab_size: int
    special_tokens: list[str] = field(default_factory=list)
    train_params: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    library_versions: dict[str, str] = field(default_factory=dict)
    artifact_dir: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ArtifactManifest:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _library_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:  # optional dependency, only used by one implementation
        import tokenizers as _tk

        versions["tokenizers"] = _tk.__version__
    except Exception:  # pragma: no cover - optional
        versions["tokenizers"] = "not-installed"
    return versions


def save_artifact(
    tokenizer: SubwordTokenizer,
    out_dir: str | Path,
    experiment_id: str,
    corpus: dict[str, Any],
    train_params: dict[str, Any],
    seed: int | None = None,
    created_at: str | None = None,
) -> ArtifactManifest:
    """Write the tokenizer files + ``manifest.json`` and return the manifest."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(directory)
    tokenizer.info()  # sanity: touches vocab_size, fails loudly on an untrained tokenizer

    manifest = ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        created_at=created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        impl=tokenizer.name,
        impl_version=tokenizer.impl_version(),
        vocab_size=tokenizer.vocab_size,
        special_tokens=list(tokenizer.special_tokens),
        train_params=dict(train_params),
        corpus=dict(corpus),
        seed=seed,
        library_versions=_library_versions(),
        artifact_dir=str(directory),
    )
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _export_vocab(tokenizer, directory)
    return manifest


def _export_vocab(tokenizer: SubwordTokenizer, directory: Path) -> None:
    """Best-effort id -> token dump, handy when inspecting what a tokenizer learned."""
    try:
        entries = {str(i): tokenizer.id_to_token(i) for i in range(tokenizer.vocab_size)}
    except Exception:  # pragma: no cover - implementation may not support introspection
        return
    (directory / "vocab.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=0), encoding="utf-8"
    )


def load_manifest(artifact_dir: str | Path) -> ArtifactManifest:
    path = Path(artifact_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no tokenizer manifest at {path}. Artifacts must be written by "
            "scripts/tokenizer_train.py (or tokenization.artifact.save_artifact)."
        )
    return ArtifactManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_artifact(artifact_dir: str | Path) -> tuple[SubwordTokenizer, ArtifactManifest]:
    """Load a tokenizer from an artifact directory, using the manifest to pick the impl."""
    manifest = load_manifest(artifact_dir)
    tokenizer = create(manifest.impl)
    loaded = tokenizer.load(artifact_dir)
    if loaded.vocab_size != manifest.vocab_size:
        raise ValueError(
            f"artifact {artifact_dir}: manifest vocab_size={manifest.vocab_size} but the loaded "
            f"tokenizer reports {loaded.vocab_size}; the artifact may be corrupted"
        )
    return loaded, manifest
