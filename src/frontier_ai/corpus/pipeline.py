"""Common document model and stage conventions for the FrontierCorpus pipeline.

FrontierCorpus v1 (MASTER_CONTEXT §12–13, §37 step 3) is the reproducible
foundation-model data pipeline. It extends — it does not fork — the P004A/B
corpus infrastructure: acquisition, licensing, provenance, splitting and
hashing already live in ``frontier_ai.data.corpora`` and
``frontier_ai.tokenization.research_corpus``. This package adds the stages
that transform a frozen, verified source registry into a training-ready,
versioned dataset:

    normalize -> langid -> quality -> exact dedup -> (mix -> split -> shuffle
    -> pack -> shard -> manifest; later stages, F1 part 2 / F2 / F3)

Every stage is a pure function with one contract so that stages compose,
test independently, and the build can report exactly what each stage kept,
removed, and flagged (MASTER_CONTEXT §15: keep intermediate statistics so we
can understand what the pipeline removed). No stage has a global RNG and no
stage depends on input order beyond what it records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from frontier_ai.data.corpora import sha256_text


@dataclass(frozen=True)
class PipelineDocument:
    """One unit the pipeline processes: a single document.

    In the pilot corpus (the frozen ``indic-tokenizer/v2`` documents) a
    document is one non-empty line — a paragraph in prose, one verse line in
    poetry. Stages treat documents as atomic: none of the current stages
    splits or merges them.

    ``text`` is always the document as the *current* stage sees it; after the
    normalization stage it is the normalized form. ``doc_id`` follows the
    P004B convention (``<source_id>-<index:06d>``) so pipeline documents map
    one-to-one onto the frozen corpus documents.
    """

    doc_id: str
    source_id: str
    language: str
    text: str

    # ------------------------------------------------------------- derived --
    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def sha256(self) -> str:
        """SHA-256 of the document's current UTF-8 text.

        This is the identity exact deduplication keys on. It is recomputed
        from the text rather than stored, so a document whose text changed in
        an earlier stage automatically changes identity — the same rule that
        makes the frozen corpus's split honest.
        """
        return sha256_text(self.text)


@dataclass(frozen=True)
class Removal:
    """One document a stage removed, with the machine-readable reason.

    ``reason`` is a stable rule/stage id (``langid_mismatch``,
    ``quality:control_chars``, ``exact_duplicate`` …); ``detail`` carries the
    measured values so a future reader can answer *why* without re-running
    the stage.
    """

    doc_id: str
    source_id: str
    language: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "language": self.language,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class StageOutcome:
    """The result of one pipeline stage.

    * ``kept`` — the documents that go on, in the stage's documented order.
    * ``removed`` — every dropped document with its reason (nothing is ever
      dropped silently; MASTER_CONTEXT §15).
    * ``flagged`` — doc_id -> tuple of rule names for kept documents that a
      rule *flagged* (kept, but recorded for review).
    * ``stats`` — the stage's summary numbers, per language where relevant.
    """

    stage: str
    kept: tuple[PipelineDocument, ...]
    removed: tuple[Removal, ...]
    flagged: dict[str, tuple[str, ...]]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "kept": len(self.kept),
            "removed": len(self.removed),
            "flagged": {k: list(v) for k, v in sorted(self.flagged.items())},
            "stats": dict(self.stats),
        }
