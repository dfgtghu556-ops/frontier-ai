"""Stage 5: the train/holdout split — reusing the P004B content-hash split.

The split is *reused, not re-implemented* (D-037: extend, don't fork): it is
``frontier_ai.tokenization.research_corpus.split_documents`` itself — the same
deterministic rule the frozen v2 corpus was split with, ``sha256("<seed>:<content
sha256>") < held_out_fraction``, no global RNG, no order dependence. Two things
make it correct to apply here rather than to the v2 ``train.jsonl``/
``heldout.jsonl`` files directly:

* the split keys on the **content** hash, so it must be computed on the text as
  the pipeline finally stores it — i.e. *after* normalization and dedup (the v2
  split was computed on un-normalized text, D-035's ``normalization_policy:
  "none"``);
* content-keying guarantees the property evaluation depends on: byte-identical
  text can never appear on both sides, even across sources.

The holdout side is not a removal — it is the second, equally real output of
this stage, and the builder writes it as its own shard set.
"""

from __future__ import annotations

from collections.abc import Sequence

from frontier_ai.corpus.pipeline import PipelineDocument, StageOutcome
from frontier_ai.tokenization.research_corpus import (
    DEFAULT_HELD_OUT_FRACTION,
    DEFAULT_SPLIT_SEED,
    CorpusDocument,
    split_documents,
)

SPLIT_METHOD = "deterministic-content-hash"


def _to_corpus(doc: PipelineDocument) -> CorpusDocument:
    # same fields, same derived identity (sha256 of the text)
    return CorpusDocument(doc_id=doc.doc_id, source_id=doc.source_id, language=doc.language, text=doc.text)


def train_holdout(
    documents: Sequence[PipelineDocument],
    seed: int = DEFAULT_SPLIT_SEED,
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
) -> tuple[StageOutcome, tuple[PipelineDocument, ...]]:
    """Split into ``(train_outcome, held_out)``.

    ``train_outcome.kept`` is the train side (input order preserved — shuffling
    is the next stage); ``held_out`` is the evaluation side. ``stats`` records
    the method, seed, fraction and per-side/per-language counts, so the manifest
    can reproduce exactly how the dataset was divided.
    """
    corpus_docs = [_to_corpus(d) for d in documents]
    train, held = split_documents(corpus_docs, seed=seed, held_out_fraction=held_out_fraction)

    def per_side(docs: list[CorpusDocument]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for d in docs:
            c = out.setdefault(d.language, {"documents": 0, "chars": 0})
            c["documents"] += 1
            c["chars"] += len(d.text)
        return out

    stats = {
        "method": SPLIT_METHOD,
        "seed": seed,
        "held_out_fraction": held_out_fraction,
        "level": "document",
        "train": {
            "documents": len(train),
            "chars": sum(d.chars for d in train),
            "per_language": per_side(train),
        },
        "held_out": {
            "documents": len(held),
            "chars": sum(d.chars for d in held),
            "per_language": per_side(held),
        },
    }
    outcome = StageOutcome(
        stage="split",
        kept=tuple(
            PipelineDocument(d.doc_id, d.source_id, d.language, d.text) for d in train
        ),
        removed=(),
        flagged={},
        stats=stats,
    )
    held_out = tuple(PipelineDocument(d.doc_id, d.source_id, d.language, d.text) for d in held)
    return outcome, held_out
