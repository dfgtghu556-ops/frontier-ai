"""Protected evaluation suites: a fingerprint-only definition of the evaluation documents.

A suite file (``evals/suites/<id>/SUITE.json``) lists every evaluation document in
evaluation order with its language, source and the SHA-256 of its text — never the text
itself (the text is re-derived from the frozen corpus at evaluation time and must match
these hashes exactly). The same hashes let any future training-data builder prove that it
contains no evaluation document (:func:`find_exact_overlap`): evaluation data must never be
inserted into pretraining (MASTER_CONTEXT §19, D-042).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SUITES_ROOT = REPO_ROOT / "evals" / "suites"
SUITE_SCHEMA = "frontier-eval-suite-v1"
FRONTIER_HELDOUT_V1 = "frontier-heldout-v1"

# Script of each corpus language (for per-script aggregation).
LANGUAGE_SCRIPT: dict[str, str] = {
    "en": "Latin",
    "hi": "Devanagari",
    "mr": "Devanagari",
    "bn": "Bengali-Assamese",
    "as": "Bengali-Assamese",
    "gu": "Gujarati",
    "pa": "Gurmukhi",
    "or": "Odia",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ur": "Perso-Arabic",
}


class SuiteError(ValueError):
    """The evaluation documents do not match the protected suite definition."""


def script_of(language: str) -> str:
    return LANGUAGE_SCRIPT.get(language, "other")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_records(docs: Sequence[Any]) -> list[dict[str, Any]]:
    """One fingerprint record per document, in the given (evaluation) order."""
    return [
        {
            "doc_id": d.doc_id,
            "language": d.language,
            "source_id": d.source_id,
            "sha256": text_sha256(d.text),
            "bytes": len(d.text.encode("utf-8")),
            "chars": len(d.text),
        }
        for d in docs
    ]


def records_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    """Order-sensitive fingerprint of the document records."""
    h = hashlib.sha256()
    for r in records:
        line = json.dumps(
            [r["doc_id"], r["language"], r["source_id"], r["sha256"], r["bytes"], r["chars"]],
            ensure_ascii=False,
        )
        h.update(line.encode("utf-8") + b"\n")
    return h.hexdigest()


def build_suite(docs: Sequence[Any], suite_id: str, source: dict[str, Any]) -> dict[str, Any]:
    records = doc_records(docs)
    per_language: dict[str, dict[str, int]] = {}
    for r in records:
        agg = per_language.setdefault(r["language"], {"documents": 0, "bytes": 0, "chars": 0})
        agg["documents"] += 1
        agg["bytes"] += r["bytes"]
        agg["chars"] += r["chars"]
    return {
        "schema": SUITE_SCHEMA,
        "suite_id": suite_id,
        "status": "protected",
        "source": source,
        "order": "evaluation order = the frozen corpus side in its derived order "
                 "(the same order EXP-B encoded its validation stream)",
        "totals": {
            "documents": len(records),
            "bytes": sum(r["bytes"] for r in records),
            "chars": sum(r["chars"] for r in records),
        },
        "per_language": dict(sorted(per_language.items())),
        "fingerprint": records_fingerprint(records),
        "policy": "Protected evaluation data: never train on these documents. Every future "
                  "training-data builder must check its documents against these hashes "
                  "(frontier_ai.evaluation.suite.find_exact_overlap) (D-042).",
        "documents": records,
    }


def write_suite(path: str | Path, suite: dict[str, Any]) -> None:
    """Write with one document per line (readable diffs), LF line endings everywhere."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in suite.items() if k != "documents"}
    lines = json.dumps(head, ensure_ascii=False, indent=2)[:-2]  # drop the closing "\n}"
    doc_lines = ",\n".join("    " + json.dumps(r, ensure_ascii=False) for r in suite["documents"])
    text = f'{lines},\n  "documents": [\n{doc_lines}\n  ]\n}}\n'
    json.loads(text)  # self-check: the hand-assembled layout must be valid JSON
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def load_suite(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise SuiteError(f"suite file not found: {path}")
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite.get("schema") != SUITE_SCHEMA:
        raise SuiteError(f"{path}: expected schema {SUITE_SCHEMA!r}")
    if records_fingerprint(suite["documents"]) != suite["fingerprint"]:
        raise SuiteError(f"{path}: document list does not match its own fingerprint (edited?)")
    return suite


def verify_docs_against_suite(docs: Sequence[Any], suite: dict[str, Any]) -> None:
    """Raise :class:`SuiteError` unless ``docs`` are exactly the suite's documents, in order."""
    records = doc_records(docs)
    expected = suite["documents"]
    if len(records) != len(expected):
        raise SuiteError(f"{len(records)} documents, suite has {len(expected)}")
    for i, (got, want) in enumerate(zip(records, expected)):
        if got != want:
            raise SuiteError(
                f"document #{i} differs from the suite: got {got['doc_id']} "
                f"({got['sha256'][:12]}…), suite {want['doc_id']} ({want['sha256'][:12]}…)"
            )


def find_exact_overlap(texts: Iterable[str], suite: dict[str, Any]) -> list[str]:
    """Suite doc_ids whose exact text appears among ``texts`` (the training-data guard)."""
    by_hash: dict[str, list[str]] = {}
    for r in suite["documents"]:
        by_hash.setdefault(r["sha256"], []).append(r["doc_id"])
    hits: list[str] = []
    for text in texts:
        ids = by_hash.pop(text_sha256(text), None)
        if ids:
            hits.extend(ids)
    return sorted(hits)
