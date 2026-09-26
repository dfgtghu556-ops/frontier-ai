"""Stage 9: the dataset manifest — what a build is, recorded once.

The manifest is the answer to "which corpus is this build of, and what did the
pipeline do to it?" (MASTER_CONTEXT §12–13, D-037):

* **identity** — corpus id/version, git SHA, the frozen source-registry hash
  (the D-035 freeze identity), the split seed/fraction, the normalization
  policy and version (D-036);
* **stages** — every stage's :meth:`StageOutcome.to_dict` (per-rule and
  per-language stats, every removal counted) so the build's behaviour is
  auditable without re-running it;
* **sides** — train and held_out, each with totals, per-language counts, and
  one SHA-256 per shard.

Reproducibility
---------------
Everything in the manifest is a deterministic function of the frozen corpus
text, the code, and the recorded parameters — **except ``created_at``**, which
is kept as the single non-deterministic field. ``content_sha256`` hashes the
manifest with ``created_at`` removed, so two builds of the same corpus at the
same code commit compare by ``content_sha256``: the files may differ by
timestamp only, and the reader can say so explicitly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from frontier_ai.corpus.pipeline import StageOutcome
from frontier_ai.corpus.registry import (
    FRONTIER_CORPUS_ID,
    FRONTIER_CORPUS_VERSION,
    FrontierSource,
)
from frontier_ai.corpus.shards import Shard

MANIFEST_SCHEMA_VERSION = "1.0"


def _side(shards: Sequence[Shard], per_language: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "documents": sum(len(s.documents) for s in shards),
        "chars": sum(s.chars for s in shards),
        "bytes": sum(s.bytes for s in shards),
        "per_language": per_language,
        "shards": [s.to_dict() for s in shards],
    }


def build_manifest(
    *,
    git_sha: str,
    registry: Sequence[FrontierSource],
    source_registry_path: str,
    freeze_sha256: str,
    seed: int,
    held_out_fraction: float,
    normalization_policy: str,
    policy_version: str,
    split_stats: dict[str, Any],
    stages: Sequence[StageOutcome],
    train_shards: Sequence[Shard],
    heldout_shards: Sequence[Shard],
    train_per_language: dict[str, dict[str, int]],
    heldout_per_language: dict[str, dict[str, int]],
    created_at: str,
) -> dict[str, Any]:
    """Assemble the manifest dict (deterministic apart from ``created_at``)."""
    domains = sorted({s.domain for s in registry})
    content: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus": {"id": FRONTIER_CORPUS_ID, "version": FRONTIER_CORPUS_VERSION, "domains": domains},
        "identity": {
            "git_sha": git_sha,
            "source_registry": {
                "path": source_registry_path,
                "corpus": "indic-tokenizer/v2 (frozen, D-035)",
                "sha256": freeze_sha256,
                "sources": len(registry),
                "licenses": sorted({s.license_id for s in registry}),
            },
            "split": {
                "method": split_stats["method"],
                "seed": seed,
                "held_out_fraction": held_out_fraction,
                "level": "document",
            },
            "normalization": {"policy": normalization_policy, "version": policy_version},
            "sources": [asdict(s) for s in sorted(registry, key=lambda s: s.source_id)],
        },
        "stages": [s.to_dict() for s in stages],
        "sides": {
            "train": _side(train_shards, train_per_language),
            "held_out": _side(heldout_shards, heldout_per_language),
        },
    }
    content["content_sha256"] = _content_sha256(content)
    return {"created_at": created_at, **content}


def _content_sha256(content: dict[str, Any]) -> str:
    """Hash of the canonical JSON of ``content`` minus its own ``content_sha256``."""
    payload = {k: v for k, v in content.items() if k != "content_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, fixed indent, non-ASCII preserved, LF newlines."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write_manifest(manifest: dict[str, Any], path: str | Path) -> str:
    """Write the manifest; returns its on-disk SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(manifest)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
