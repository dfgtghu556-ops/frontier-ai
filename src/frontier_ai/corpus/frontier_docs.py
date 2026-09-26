"""Shared derivation of FrontierCorpus v1 documents from the frozen v2 corpus.

Two consumers need the exact same pipeline path, so it lives here instead of in
one of them:

* ``scripts/build_frontier_corpus.py`` — builds (and re-builds) the dataset:
  shards on disk plus the dataset manifest.
* ``scripts/run_tokenizer_sweep.py`` (EXP-A) — re-derives the per-language
  train/held-out documents to train and score tokenizers, then verifies the
  derivation against the frozen shards (counts + document-text multiset).

The pipeline is deterministic (NFC normalization, script gate, quality, exact
dedup, content-hash split with the seed and fraction the dataset manifest
records), so a re-derivation on the same frozen inputs reproduces exactly the
documents the frozen shards contain — that is the identity the sweep relies on
and asserts.

Nothing here writes to the frozen corpus or to an existing dataset; the build
path writes only under an explicit ``out_dir``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tokenization.research_corpus import documents_from_text
from .dedup import exact_dedup
from .langid import langid_documents
from .manifest import build_manifest, write_manifest
from .normalize import POLICY_VERSION, normalize_documents
from .pipeline import PipelineDocument, StageOutcome
from .quality import quality_filter
from .registry import FrontierSource, RegistryError, language_scripts, load_frontier_registry
from .shards import Shard, check_shards, pack, seeded_shuffle, write_shards
from .split import train_holdout


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown (not a git checkout)"


def verify_frozen_inputs(
    manifest_path: str | Path, freeze_path: str | Path, corpus_dir: str | Path
) -> tuple[tuple[FrontierSource, ...], str]:
    """Load the freeze-verified registry and verify every on-disk text file.

    Returns ``(registry, freeze_sha256)``. Raises :class:`RegistryError` on any
    input problem (missing files, freeze mismatch, drifted text — the pilot
    builds on exactly the frozen corpus, never on approximations of it).
    """
    manifest_path = Path(manifest_path)
    freeze_path = Path(freeze_path)
    corpus_dir = Path(corpus_dir)
    registry = tuple(load_frontier_registry(manifest_path, freeze_path))
    freeze_sha256 = json.loads(freeze_path.read_text(encoding="utf-8"))["manifest"]["sha256"]
    problems: list[str] = []
    for source in registry:
        text_path = corpus_dir / "sources" / f"{source.source_id}.txt"
        if not text_path.is_file():
            problems.append(f"missing text file {text_path}")
            continue
        # The pin is a text hash (sha256 of the decoded UTF-8 text), so validate
        # the decoded text — never raw file bytes (CRLF vs LF would be a false
        # alarm; EXP-027).
        text = text_path.read_text(encoding="utf-8")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != source.sha256:
            problems.append(
                f"{source.source_id}: text hash {actual} != pinned {source.sha256} "
                "(stale or drifted file — rebuild the v2 corpus first)"
            )
    if problems:
        raise RegistryError(
            "the v2 corpus text under "
            f"{corpus_dir} is not the frozen corpus ({len(problems)} problem(s)); the pilot "
            f"refuses to build on it:\n  " + "\n  ".join(problems)
        )
    return registry, freeze_sha256


def derive_documents(
    registry: Sequence[FrontierSource], corpus_dir: str | Path
) -> list[PipelineDocument]:
    """The frozen corpus documents: the same document derivation the v2 build used."""
    corpus_dir = Path(corpus_dir)
    documents: list[PipelineDocument] = []
    for source in sorted(registry, key=lambda s: s.source_id):
        text = (corpus_dir / "sources" / f"{source.source_id}.txt").read_text(encoding="utf-8")
        for d in documents_from_text(source.source_id, source.language, text):
            documents.append(PipelineDocument(d.doc_id, d.source_id, d.language, d.text))
    return documents


def run_stages(
    documents: list[PipelineDocument], scripts_map: dict[str, str]
) -> tuple[StageOutcome, StageOutcome, StageOutcome, StageOutcome]:
    """normalize -> langid -> quality -> exact dedup.

    The pilot build's exact stage order and parameters; the four outcomes carry
    the per-stage statistics (the stage summary every report renders).
    """
    n = normalize_documents(documents, "nfc")
    lang = langid_documents(list(n.kept), scripts_map)
    q = quality_filter(list(lang.kept))
    d = exact_dedup(list(q.kept))
    return n, lang, q, d


@dataclass(frozen=True)
class FrontierDerivation:
    """The deterministic re-derivation of one frozen dataset (no files written)."""

    registry: tuple[FrontierSource, ...]
    freeze_sha256: str
    stages: tuple[StageOutcome, StageOutcome, StageOutcome, StageOutcome]
    split: StageOutcome
    held_out: tuple[PipelineDocument, ...]

    @property
    def train(self) -> tuple[PipelineDocument, ...]:
        return self.split.kept


def derive_frontier_documents(
    *,
    manifest_path: str | Path,
    freeze_path: str | Path,
    corpus_dir: str | Path,
    seed: int,
    held_out_fraction: float,
) -> FrontierDerivation:
    """Verify the frozen inputs and re-derive the train/held-out documents.

    Read-only: it hashes and reads inputs, writes nothing.
    """
    registry, freeze_sha256 = verify_frozen_inputs(manifest_path, freeze_path, corpus_dir)
    documents = derive_documents(registry, corpus_dir)
    stages = run_stages(documents, language_scripts(registry))
    split_out, held = train_holdout(
        list(stages[3].kept), seed=seed, held_out_fraction=held_out_fraction
    )
    return FrontierDerivation(
        registry=registry,
        freeze_sha256=freeze_sha256,
        stages=stages,
        split=split_out,
        held_out=tuple(held),
    )


@dataclass(frozen=True)
class FrontierBuild:
    """A full pilot dataset build: derivation plus shards and the manifest."""

    derivation: FrontierDerivation
    train_docs: tuple[PipelineDocument, ...]
    held_docs: tuple[PipelineDocument, ...]
    train_shards: tuple[Shard, ...]
    held_shards: tuple[Shard, ...]
    manifest: dict[str, Any]
    manifest_sha256: str


def build_frontier_dataset(
    *,
    manifest_path: str | Path,
    freeze_path: str | Path,
    corpus_dir: str | Path,
    out_dir: str | Path,
    seed: int,
    held_out_fraction: float,
    max_shard_chars: int,
    source_registry_path: str | None = None,
    created_at: str | None = None,
) -> FrontierBuild:
    """Derive, shuffle, pack, and write shards + ``manifest.json`` under ``out_dir``.

    Byte-identical output for identical inputs (the builder's reproducibility
    promise): the shuffle, packing and manifest are all deterministic given the
    frozen inputs, the seed, the fraction and the shard budget.
    """
    out_dir = Path(out_dir)
    derivation = derive_frontier_documents(
        manifest_path=manifest_path,
        freeze_path=freeze_path,
        corpus_dir=corpus_dir,
        seed=seed,
        held_out_fraction=held_out_fraction,
    )
    train_docs = tuple(seeded_shuffle(list(derivation.train), seed))
    held_docs = tuple(seeded_shuffle(list(derivation.held_out), seed + 1))
    train_shards = pack(train_docs, max_shard_chars, "train")
    held_shards = pack(held_docs, max_shard_chars, "heldout")
    write_shards(train_shards, out_dir / "shards" / "train")
    write_shards(held_shards, out_dir / "shards" / "heldout")

    split_stats = derivation.split.stats
    manifest = build_manifest(
        git_sha=_git_sha(),
        registry=derivation.registry,
        source_registry_path=str(source_registry_path or manifest_path),
        freeze_sha256=derivation.freeze_sha256,
        seed=seed,
        held_out_fraction=held_out_fraction,
        normalization_policy="nfc",
        policy_version=POLICY_VERSION,
        split_stats=split_stats,
        stages=derivation.stages,
        train_shards=train_shards,
        heldout_shards=held_shards,
        train_per_language=split_stats["train"]["per_language"],
        heldout_per_language=split_stats["held_out"]["per_language"],
        created_at=created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    manifest_sha = write_manifest(manifest, out_dir / "manifest.json")
    return FrontierBuild(
        derivation=derivation,
        train_docs=train_docs,
        held_docs=held_docs,
        train_shards=train_shards,
        held_shards=held_shards,
        manifest=manifest,
        manifest_sha256=manifest_sha,
    )


# ---------------------------------------------------------------------------
# input identity gates (shared by every consumer of a frozen dataset)
# ---------------------------------------------------------------------------
def _load_frontier(frontier_dir: Path) -> dict:
    manifest_file = frontier_dir / "manifest.json"
    if not manifest_file.is_file():
        print(
            f"[frontier] INPUT GATE FAILED: no dataset manifest at {manifest_file} — build the "
            "pilot first (scripts/build_frontier_corpus.py) or point --frontier-dir at the "
            "existing build (on the PC: corpora/frontier/v1)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return json.loads(manifest_file.read_text(encoding="utf-8"))


def _gate_shards_match_manifest(frontier_dir: Path, manifest: dict) -> None:
    problems: list[str] = []
    for side, sub in (("train", "train"), ("held_out", "heldout")):
        expected = {s["name"]: s["sha256"] for s in manifest["sides"][side]["shards"]}
        problems.extend(f"[{side}] {p}" for p in check_shards(frontier_dir / "shards" / sub, expected))
    if problems:
        print(
            "[frontier] INPUT GATE FAILED: on-disk shards do not match the frozen dataset "
            f"manifest ({len(problems)} problem(s)):\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        raise SystemExit(2)


def _gate_derivation_matches_shards(
    frontier_dir: Path, manifest: dict, derivation: FrontierDerivation
) -> None:
    """The re-derived documents must reproduce the frozen shards exactly."""
    split = derivation.split.stats
    problems: list[str] = []

    # (a) counts: totals and per-language, side by side, against the manifest
    for side in ("train", "held_out"):
        manifest_side = manifest["sides"][side]
        derived_side = split[side]
        if derived_side["documents"] != manifest_side["documents"]:
            problems.append(
                f"[{side}] derived {derived_side['documents']} documents, "
                f"manifest has {manifest_side['documents']}"
            )
        for lang in sorted(set(manifest_side["per_language"]) | set(derived_side["per_language"])):
            m = manifest_side["per_language"].get(lang, {"documents": 0, "chars": 0})
            d = derived_side["per_language"].get(lang, {"documents": 0, "chars": 0})
            if (m["documents"], m["chars"]) != (d["documents"], d["chars"]):
                problems.append(
                    f"[{side}/{lang}] derived {d['documents']} docs / {d['chars']} chars, "
                    f"manifest has {m['documents']} docs / {m['chars']} chars"
                )

    # (b) exact sequence: the shard lines, in shard order, side by side
    for side, sub, docs in (
        ("train", "train", list(derivation.train)),
        ("held_out", "heldout", list(derivation.held_out)),
    ):
        shard_text: list[str] = []
        for shard in manifest["sides"][side]["shards"]:
            shard_text.append((frontier_dir / "shards" / sub / f"{shard['name']}.txt")
                              .read_text(encoding="utf-8"))
        shard_lines = [line for text in shard_text for line in text.split("\n")]
        seed = split["seed"]
        shuffled = seeded_shuffle(docs, seed) if side == "train" else seeded_shuffle(docs, seed + 1)
        derived_lines = [d.text for d in shuffled]
        if shard_lines != derived_lines:
            n_shard, n_derived = len(shard_lines), len(derived_lines)
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(shard_lines, derived_lines)) if a != b),
                min(n_shard, n_derived),
            )
            problems.append(
                f"[{side}] shard line sequence differs from the deterministic re-derivation "
                f"({n_shard} vs {n_derived} lines; first difference at line {first_diff}) — "
                "the frozen build and the pipeline on this machine disagree"
            )
    if problems:
        print(
            "[frontier] INPUT GATE FAILED: the re-derived documents do not reproduce the frozen "
            f"shards ({len(problems)} problem(s)):\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        raise SystemExit(2)


def verify_and_derive_frontier(
    frontier_dir: str | Path,
    manifest_path: str | Path,
    freeze_path: str | Path,
    corpus_dir: str | Path,
) -> tuple[list[PipelineDocument], list[PipelineDocument], dict]:
    """The frozen-input identity gates + deterministic re-derivation (one call).

    On-disk shards must hash to their manifest; the manifest's split parameters
    drive a deterministic re-derivation of the per-language documents; the
    re-derivation must reproduce the frozen shards exactly (per-language counts
    and the full shard line sequence). Any failure: message on stderr,
    ``SystemExit(2)``. On success returns ``(train_docs, held_out_docs, manifest)``.

    Shared by ``scripts/run_tokenizer_sweep.py`` (EXP-A) and
    ``scripts/prepare_exp_b_data.py`` (EXP-B) so every consumer of the frozen
    dataset proves its identity the same way before touching the documents.
    """
    frontier_dir = Path(frontier_dir)
    manifest = _load_frontier(frontier_dir)
    _gate_shards_match_manifest(frontier_dir, manifest)

    split_identity = manifest["identity"]["split"]
    split_seed = int(split_identity["seed"])
    held_out_fraction = float(split_identity["held_out_fraction"])

    if not Path(manifest_path).is_file() or not Path(freeze_path).is_file():
        print(
            f"[frontier] INPUT GATE FAILED: missing frozen input: {manifest_path} or {freeze_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        derivation = derive_frontier_documents(
            manifest_path=Path(manifest_path),
            freeze_path=Path(freeze_path),
            corpus_dir=Path(corpus_dir),
            seed=split_seed,
            held_out_fraction=held_out_fraction,
        )
    except RegistryError as exc:
        print(f"[frontier] INPUT GATE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    _gate_derivation_matches_shards(frontier_dir, manifest, derivation)

    train_docs = list(derivation.train)
    held_docs = list(derivation.held_out)
    print(
        f"[frontier] input verified: {manifest['content_sha256'][:16]}… (frozen shards match; "
        f"re-derivation reproduces them exactly)"
    )
    print(
        f"[frontier] documents: train {len(train_docs)} / {sum(d.chars for d in train_docs):,} chars; "
        f"held_out {len(held_docs)} / {sum(d.chars for d in held_docs):,} chars "
        f"(split seed {split_seed}, held_out {held_out_fraction})"
    )
    return train_docs, held_docs, manifest
