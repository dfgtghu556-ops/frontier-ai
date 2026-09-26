"""Stage 2: language identification (script-profile gate).

Why this exists
---------------
A declared language is a claim. Devanagari carries Hindi, Marathi, Sanskrit
and Nepali; Bengali script carries Bengali and Assamese; Perso-Arabic carries
Urdu. A mis-declared source would silently corrupt every per-language number
downstream, so every document is checked against its declared script before
anything else runs on it. MASTER_CONTEXT §13 lists language identification as
a pipeline stage; this is the deterministic first pass of it.

What this stage is — and is not
-------------------------------
It is a **script-profile gate**, not a language classifier. It counts the
letter characters of each document by Unicode script block, and removes a
document only when its declared script is absent or a clear minority
(default: less than 60% of the document's letters). It cannot tell Hindi
apart from Marathi (same script); that needs a word-level classifier and is
deliberately deferred (F3, optional stage). For the current corpus — one
declared script per language slot — a script gate is exactly the right
strength, and it has no new dependency: pure stdlib, deterministic, fast.

Document classes
----------------
* *identified* — the declared script holds at or above the threshold; kept.
* *mismatch*   — the declared script is below threshold; removed, with the
  full profile recorded so a future reader sees exactly what the document
  actually was.
* *no letters* — the document has no letter characters at all (e.g. a line of
  bare digits); it cannot be assigned to any language; removed, recorded.

Nothing is removed silently: every removal carries the measured profile.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import Any

from frontier_ai.corpus.pipeline import PipelineDocument, Removal, StageOutcome

# Letter ranges per script block. Whitespace is ignored by the profiler;
# digits and punctuation count as "common" (script-neutral), mirroring how
# the P004B inspection report expresses "DEVANAGARI 100.0%".
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("latin", ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF), (0x0100, 0x024F), (0x0250, 0x02A8),
               (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF))),
    ("devanagari", ((0x0900, 0x097F),)),
    ("bengali", ((0x0980, 0x09FF),)),
    ("gurmukhi", ((0x0A00, 0x0A7F),)),
    ("gujarati", ((0x0A80, 0x0AFF),)),
    ("odia", ((0x0B00, 0x0B7F),)),
    ("tamil", ((0x0B80, 0x0BFF),)),
    ("telugu", ((0x0C00, 0x0C7F),)),
    ("kannada", ((0x0C80, 0x0CFF),)),
    ("malayalam", ((0x0D00, 0x0D7F),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
)

DEFAULT_MIN_DECLARED_SHARE = 0.6


def _script_of(char: str) -> str | None:
    cp = ord(char)
    for name, ranges in _SCRIPT_RANGES:
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def script_profile(text: str) -> dict[str, int]:
    """Count non-whitespace characters of ``text`` by script.

    Letter characters (Unicode category ``L*``) count under their script
    block; anything else that is not whitespace (digits, punctuation,
    symbols) counts under ``"common"``. Whitespace counts nowhere. A letter
    outside the known blocks counts under ``"unknown"`` so it can never
    vanish silently.
    """
    profile: dict[str, int] = {}
    for ch in text:
        if ch.isspace():
            continue
        script = _script_of(ch) if unicodedata.category(ch)[0] == "L" else "common"
        if script is None:
            script = "unknown"
        profile[script] = profile.get(script, 0) + 1
    return profile


def letter_total(profile: dict[str, int]) -> int:
    """Number of letter characters in a profile (excludes ``common``/``unknown``)."""
    return sum(n for s, n in profile.items() if s not in ("common", "unknown"))


def top_script(profile: dict[str, int]) -> str:
    """The dominant *letter* script, or ``"common"`` when the text has no letters."""
    best, best_n = "common", 0
    for script, n in profile.items():
        if script in ("common", "unknown"):
            continue
        if n > best_n:
            best, best_n = script, n
    return best


def langid_documents(
    documents: list[PipelineDocument],
    language_scripts: Mapping[str, str],
    min_declared_share: float = DEFAULT_MIN_DECLARED_SHARE,
) -> StageOutcome:
    """Gate every document on its declared script. Order is preserved.

    ``language_scripts`` maps each declared language to the pipeline script
    name expected for it (e.g. ``{"hi": "devanagari", "as": "bengali",
    "ur": "arabic"}``). A language absent from the mapping is a caller bug
    (fail loudly) — a document can only be checked against a declared
    expectation.
    """
    if not 0.0 < min_declared_share <= 1.0:
        raise ValueError("min_declared_share must be in (0, 1]")
    kept: list[PipelineDocument] = []
    removed: list[Removal] = []
    per_language: dict[str, dict[str, int]] = {}
    for doc in documents:
        expected = language_scripts.get(doc.language)
        if expected is None:
            raise ValueError(f"document {doc.doc_id!r}: no declared script for language {doc.language!r}")
        stats_l = per_language.setdefault(doc.language, {"in": 0, "kept": 0, "removed": 0})
        stats_l["in"] += 1
        profile = script_profile(doc.text)
        letters = letter_total(profile)
        share = (profile.get(expected, 0) / letters) if letters else 0.0
        if letters == 0:
            removed.append(Removal(doc.doc_id, doc.source_id, doc.language, "langid_no_letters",
                                   {"profile": dict(profile)}))
            stats_l["removed"] += 1
        elif share < min_declared_share:
            removed.append(Removal(doc.doc_id, doc.source_id, doc.language, "langid_mismatch",
                                   {"expected_script": expected, "share": round(share, 6),
                                    "top_script": top_script(profile), "profile": dict(profile)}))
            stats_l["removed"] += 1
        else:
            kept.append(doc)
            stats_l["kept"] += 1
    stats: dict[str, Any] = {
        "method": "script-profile",
        "min_declared_share": min_declared_share,
        "documents_in": len(documents),
        "documents_out": len(kept),
        "removed": len(removed),
        "per_language": per_language,
        "note": "script gate, not a language classifier: same-script languages "
                "(hi/mr in Devanagari, bn/as in Bengali) are not distinguished here",
    }
    return StageOutcome(stage="langid", kept=tuple(kept), removed=tuple(removed), flagged={}, stats=stats)
