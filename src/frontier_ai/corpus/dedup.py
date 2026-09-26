"""Stage 4: exact deduplication.

Why this exists
---------------
Duplicate text inflates effective data volume and trains the model to
memorize. The frozen corpus's own inspection reports count identical
documents *within* a language (e.g. 218 in the Bengali book, 396 in total
across the 13 evaluated slots — reports/EXP-023-inspection.txt), but the
pipeline must remove them, not just count them. This stage does that across
the whole registry, including *across* sources (the same text re-hosted on
two wikis), which is the case MASTER_CONTEXT §14 singles out: "also detect
cross-source duplication."

Key and semantics
-----------------
The key is the SHA-256 of the document's *current* text — i.e. after the
normalization stage — so NFC variants of the same text deduplicate together.
Documents are processed in deterministic order ``sorted(source_id, doc_id)``;
the first occurrence wins, every later one is removed with
``reason="exact_duplicate"`` and a ``first_seen``/``first_source`` pointer,
so the removal is explainable and order-independent (shuffling the input
changes nothing — covered by a test).

Scope and limits
----------------
This is *exact* deduplication at document granularity (paragraph/verse-line
in the pilot corpus). Near-duplicate detection (MinHash/LSH, paragraph-level
fuzzy matches) is F3 — deliberately not approximated here, because a silent
"almost" would corrupt the honest exact count that the frozen corpus's
reports can be cross-checked against.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from frontier_ai.corpus.pipeline import PipelineDocument, Removal, StageOutcome


def exact_dedup(documents: Sequence[PipelineDocument]) -> StageOutcome:
    """Remove later occurrences of identical (normalized) documents.

    The kept set is independent of input order; the kept tuple is returned in
    the canonical ``(source_id, doc_id)`` order so downstream stages always
    see a canonical sequence.
    """
    ordered = sorted(documents, key=lambda d: (d.source_id, d.doc_id))
    first_seen: dict[str, PipelineDocument] = {}
    kept: list[PipelineDocument] = []
    removed: list[Removal] = []
    per_source: Counter[str] = Counter()
    cross_pairs: Counter[str] = Counter()
    for doc in ordered:
        key = doc.sha256
        first = first_seen.get(key)
        if first is None:
            first_seen[key] = doc
            kept.append(doc)
        else:
            removed.append(Removal(doc.doc_id, doc.source_id, doc.language, "exact_duplicate",
                                   {"first_seen": first.doc_id, "first_source": first.source_id}))
            per_source[doc.source_id] += 1
            pair = first.source_id if first.source_id == doc.source_id \
                else "|".join(sorted((first.source_id, doc.source_id)))
            cross_pairs[pair] += 1
    stats: dict[str, Any] = {
        "documents_in": len(documents),
        "documents_out": len(kept),
        "unique_keys": len(first_seen),
        "removed": len(removed),
        "removed_per_source": dict(sorted(per_source.items())),
        # "a|b" = duplicates of a document first seen in a, found in b;
        # "a" = within-source duplicates (both copies in a).
        "duplicate_pairs": dict(sorted(cross_pairs.items())),
    }
    return StageOutcome(
        stage="exact_dedup", kept=tuple(kept), removed=tuple(removed), flagged={}, stats=stats
    )
