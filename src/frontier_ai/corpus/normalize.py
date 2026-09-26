"""Stage 1: normalization.

Why this exists
---------------
The frozen corpus (``indic-tokenizer/v2``, D-035) stored text *without*
Unicode normalization on purpose: Stage A refused to normalize so that a
normalization experiment would stay reviewable. FrontierCorpus v1 is that
decision point. The pipeline normalizes once, here, at the front of the line,
so every later stage (identity, dedup, quality, tokenizer experiments) works
on one canonical form — and the policy is recorded, so a later ablation
(NFD, none) is a new pipeline version, not a silent re-run (D-036).

What it does
------------
Policy ``nfc`` (the default, D-036):

* NFC composition (``unicode.normalize``) — "the same word written two
  technically different ways" becomes one form;
* ``\\r`` removed, runs of spaces/tabs collapsed to one space, ends stripped;
* document text is otherwise untouched — no case folding, no character
  rewrites, no script changes.

Policy ``none``: identity, kept so normalization ablations (Q-11) can run
through the same pipeline with the policy recorded.

What it records
---------------
``stats`` carries the policy, the policy version, how many documents' bytes
actually changed, and the in/out character and byte totals, so a reader can
see the normalization's effect without re-running it. No document is ever
removed here: normalization transforms, it does not filter.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from frontier_ai.corpus.pipeline import PipelineDocument, StageOutcome

POLICY_VERSION = "1"
SUPPORTED_POLICIES = ("nfc", "none")

_WS_RUN = re.compile(r"[ \t]{2,}")


def normalize_text(text: str, policy: str = "nfc") -> str:
    """Normalize one document's text under ``policy``.

    Deterministic and idempotent: ``normalize(normalize(x)) == normalize(x)``
    for both policies (covered by a test).
    """
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unknown normalization policy {policy!r}; expected one of {SUPPORTED_POLICIES}")
    if policy == "nfc":
        text = unicodedata.normalize("NFC", text)
    text = text.replace("\r", "")
    text = _WS_RUN.sub(" ", text)
    return text.strip()


def normalize_documents(
    documents: list[PipelineDocument], policy: str = "nfc"
) -> StageOutcome:
    """Apply the policy to every document. Order is preserved."""
    kept: list[PipelineDocument] = []
    changed = 0
    for doc in documents:
        new_text = normalize_text(doc.text, policy)
        if new_text != doc.text:
            changed += 1
        kept.append(PipelineDocument(doc.doc_id, doc.source_id, doc.language, new_text))
    stats: dict[str, Any] = {
        "policy": policy,
        "policy_version": POLICY_VERSION,
        "documents_in": len(documents),
        "documents_out": len(kept),
        "documents_changed": changed,
        "chars_in": sum(d.chars for d in documents),
        "chars_out": sum(d.chars for d in kept),
        "bytes_in": sum(d.bytes for d in documents),
        "bytes_out": sum(d.bytes for d in kept),
    }
    return StageOutcome(stage="normalize", kept=tuple(kept), removed=(), flagged={}, stats=stats)
