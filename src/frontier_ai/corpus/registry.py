"""Frontier v1 source registry: the frozen ``indic-tokenizer/v2`` manifest, verified.

The pilot FrontierCorpus (D-037) has exactly one source inventory: the frozen
tokenizer corpus (D-035). This module imports that inventory under the
FrontierCorpus contract:

* the manifest bytes must hash to the identity recorded in
  ``FREEZE.json`` — anything else is a *different corpus*, and the build
  refuses to run on it (a frozen corpus that silently changes is the failure
  D-035 exists to prevent);
* every source carries the pinned ``sha256`` of its cleaned text, the licence
  record, and its retrieval time — no new acquisition, no new licensing
  surface (the pilot's hard scope boundary);
* each source gets a ``domain`` — the pilot corpus is books and verse, so the
  domain rule is "every current kind is literature", and an unknown kind fails
  loudly instead of inheriting a label.

Nothing here fetches. The builder verifies the on-disk text against the pins.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FRONTIER_CORPUS_ID = "frontier-corpus"
FRONTIER_CORPUS_VERSION = "1.0.0-pilot"


class RegistryError(ValueError):
    """The registry input is not what the frozen corpus says it is."""


@dataclass(frozen=True)
class FrontierSource:
    """One source in the Frontier v1 registry (a frozen v2 source + domain)."""

    source_id: str
    title: str
    language: str
    script: str  # manifest display name, e.g. "Devanagari"
    domain: str
    kind: str
    license_id: str
    source_url: str
    sha256: str  # pinned hash of the cleaned text (the builder's verification key)
    retrieved_at: str


# Manifest display script name -> pipeline script-block name (corpus.langid).
# "Bengali-Assamese" maps to the Bengali block: Assamese is written in Bengali
# script; the langid stage gates on the block, not the language (documented
# limitation — bn/as are not distinguished by the gate).
SCRIPT_PIPELINE_NAMES: dict[str, str] = {
    "Latin": "latin",
    "Devanagari": "devanagari",
    "Bengali": "bengali",
    "Bengali-Assamese": "bengali",
    "Gurmukhi": "gurmukhi",
    "Gujarati": "gujarati",
    "Odia": "odia",
    "Tamil": "tamil",
    "Telugu": "telugu",
    "Kannada": "kannada",
    "Malayalam": "malayalam",
    "Perso-Arabic": "arabic",
}


def language_scripts(registry: Sequence[FrontierSource]) -> dict[str, str]:
    """``language -> pipeline script block`` for the langid gate.

    Derived from the manifest's own script declarations (one per language in
    the pilot; a conflict or an unknown script is a caller/corpus bug — fails
    loudly).
    """
    out: dict[str, str] = {}
    for s in registry:
        pipeline = SCRIPT_PIPELINE_NAMES.get(s.script)
        if pipeline is None:
            raise RegistryError(f"source {s.source_id!r}: unknown manifest script {s.script!r}")
        if s.language in out and out[s.language] != pipeline:
            raise RegistryError(
                f"language {s.language!r}: conflicting scripts {out[s.language]!r} vs {pipeline!r}"
            )
        out[s.language] = pipeline
    return out


def _domain_for(kind: str, source_id: str) -> str:
    # Pilot rule: every current source kind is a book or verse (Gutenberg
    # e-books; Wikisource chapter texts). A new kind is a scope change and
    # must be reviewed, not labelled by default.
    if kind in ("gutenberg", "mediawiki-parse"):
        return "literature"
    raise RegistryError(f"source {source_id!r}: unknown kind {kind!r} — domain assignment needs review")


def load_frontier_registry(manifest_path: str | Path, freeze_path: str | Path) -> tuple[FrontierSource, ...]:
    """Import the frozen v2 manifest as the Frontier v1 registry.

    Fails with :class:`RegistryError` when the manifest bytes do not match
    the freeze record — the pilot builds on exactly the frozen corpus.
    """
    manifest_path = Path(manifest_path)
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        raise RegistryError(f"{freeze_path}: not a frozen record (status={freeze.get('status')!r})")
    expected = freeze.get("manifest", {}).get("sha256")
    if not expected:
        raise RegistryError(f"{freeze_path}: no manifest.sha256 recorded")
    manifest_bytes = manifest_path.read_bytes()
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != expected:
        raise RegistryError(
            f"{manifest_path} does not match the freeze identity:\n"
            f"  freeze {freeze_path} records  {expected}\n"
            f"  manifest bytes hash to       {actual}\n"
            "The frozen corpus (D-035) may not be edited in place; the pilot builds on the pinned bytes."
        )
    manifest: dict[str, Any] = json.loads(manifest_bytes.decode("utf-8"))
    sources: list[FrontierSource] = []
    for s in manifest["sources"]:
        sources.append(
            FrontierSource(
                source_id=s["id"],
                title=s["title"],
                language=s["language"],
                script=s["script"],
                domain=_domain_for(s["kind"], s["id"]),
                kind=s["kind"],
                license_id=s["license_id"],
                source_url=s["source_url"],
                sha256=s["sha256"],
                retrieved_at=s["retrieved_at"],
            )
        )
    if not sources:
        raise RegistryError(f"{manifest_path}: no sources")
    return tuple(sources)
