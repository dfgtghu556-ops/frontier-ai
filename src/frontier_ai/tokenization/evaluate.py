"""Tokenizer evaluation: machine-readable metrics over a probe corpus.

What is measured
----------------
Per tokenizer, overall and split by language and by probe category:

* ``tokens`` — ids produced
* ``chars`` — Unicode code points (``len(text)``)
* ``utf8_bytes`` — bytes when encoded as UTF-8
* ``words`` — whitespace-separated tokens containing at least one alphanumeric
* ``tokens_per_char`` — fertility: lower is better
* ``chars_per_token`` — compression: higher is better
* ``tokens_per_word`` — meaningful where whitespace word segmentation applies
* ``unk_count`` / ``unk_rate`` — unknown tokens, if the tokenizer has an UNK id
* ``round_trip_failures`` — examples where ``decode(encode(text)) != text``

Plus behavioural checks: UTF-8 round-trip correctness and special-token handling.

Metric caveats
--------------
* Compression measured on this fixture is **optimistic** (see ``corpus.py``): train and
  eval text share small word lists, and the fixture is tiny.
* Numbers are comparable **between tokenizers measured on the same fixture only**. They
  are not comparable to published figures computed on real corpora.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .base import SubwordTokenizer
from .corpus import CorpusManifest, Example, examples_by_category, examples_by_language

SCHEMA_VERSION = "1.0"


@dataclass
class StatBlock:
    """Metrics over a set of examples."""

    examples: int = 0
    chars: int = 0
    utf8_bytes: int = 0
    words: int = 0
    tokens: int = 0
    tokens_per_char: float = 0.0
    chars_per_token: float = 0.0
    tokens_per_word: float | None = None
    unk_count: int | None = None
    unk_rate: float | None = None
    round_trip_failures: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalReport:
    """One tokenizer evaluated against one corpus."""

    schema_version: str
    experiment_id: str
    created_at: str
    tokenizer: dict[str, Any]
    corpus: dict[str, Any]
    overall: StatBlock
    per_language: dict[str, StatBlock] = field(default_factory=dict)
    per_category: dict[str, StatBlock] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    failing_examples: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "tokenizer": self.tokenizer,
            "corpus": self.corpus,
            "overall": self.overall.to_dict(),
            "per_language": {k: v.to_dict() for k, v in self.per_language.items()},
            "per_category": {k: v.to_dict() for k, v in self.per_category.items()},
            "checks": self.checks,
            "failing_examples": self.failing_examples,
            "notes": self.notes,
        }


def count_words(text: str) -> int:
    """Whitespace-delimited words containing at least one alphanumeric character.

    A heuristic: it is meaningless for scripts without whitespace word separation, which
    is why ``tokens_per_word`` is reported per language and can be ``None``.
    """
    return sum(1 for part in text.split() if any(ch.isalnum() for ch in part))


def _measure(tokenizer: SubwordTokenizer, examples: Sequence[Example]) -> tuple[StatBlock, list[Example]]:
    unk_id = tokenizer.unk_token_id
    block = StatBlock()
    failures: list[Example] = []
    unk_total = 0

    for ex in examples:
        ids = tokenizer.encode(ex.text)
        decoded = tokenizer.decode(ids)
        block.examples += 1
        block.chars += len(ex.text)
        block.utf8_bytes += len(ex.text.encode("utf-8"))
        block.words += count_words(ex.text)
        block.tokens += len(ids)
        if unk_id is not None:
            unk_total += sum(1 for i in ids if i == unk_id)
        if decoded != ex.text:
            block.round_trip_failures += 1
            failures.append(ex)

    block.tokens_per_char = round(block.tokens / block.chars, 6) if block.chars else 0.0
    block.chars_per_token = round(block.chars / block.tokens, 6) if block.tokens else 0.0
    block.tokens_per_word = round(block.tokens / block.words, 6) if block.words else None
    if unk_id is not None:
        block.unk_count = unk_total
        block.unk_rate = round(unk_total / block.tokens, 6) if block.tokens else 0.0
    return block, failures


def check_special_tokens(tokenizer: SubwordTokenizer) -> dict[str, Any]:
    """Verify every declared special token keeps its own id, alone and inside text."""
    results: dict[str, Any] = {}
    special_ids = tokenizer.special_token_ids
    for token in tokenizer.special_tokens:
        token_id = special_ids.get(token)
        entry: dict[str, Any] = {"id": token_id}
        if token_id is None:
            entry["ok"] = False
            entry["error"] = "token is not in the vocabulary"
            results[token] = entry
            continue
        alone = tokenizer.encode(token)
        in_context = tokenizer.encode(f"a{token}b")
        entry["encodes_alone_to_single_id"] = alone == [token_id]
        entry["preserved_in_context"] = token_id in in_context
        entry["decodes_to_itself"] = tokenizer.decode([token_id]) == token
        entry["ok"] = (
            entry["encodes_alone_to_single_id"]
            and entry["preserved_in_context"]
            and entry["decodes_to_itself"]
        )
        results[token] = entry
    return {
        "declared": list(tokenizer.special_tokens),
        "per_token": results,
        "all_ok": all(v.get("ok", False) for v in results.values()) if results else True,
    }


def evaluate_tokenizer(
    tokenizer: SubwordTokenizer,
    examples: Sequence[Example],
    corpus_manifest: CorpusManifest,
    experiment_id: str = "EXP-000",
    tokenizer_meta: dict[str, Any] | None = None,
    max_failing_examples: int = 20,
) -> EvalReport:
    """Evaluate one tokenizer over the whole probe corpus."""
    if not examples:
        raise ValueError("no evaluation examples provided")

    overall, failures = _measure(tokenizer, examples)
    per_language = {
        lang: _measure(tokenizer, group)[0] for lang, group in sorted(examples_by_language(examples).items())
    }
    per_category = {
        cat: _measure(tokenizer, group)[0] for cat, group in sorted(examples_by_category(examples).items())
    }

    info = tokenizer.info()
    meta = dict(info.to_dict())
    meta.update(tokenizer_meta or {})

    return EvalReport(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tokenizer=meta,
        corpus=corpus_manifest.to_dict(),
        overall=overall,
        per_language=per_language,
        per_category=per_category,
        checks={
            "utf8_round_trip_ok": overall.round_trip_failures == 0,
            "special_tokens": check_special_tokens(tokenizer),
            "unknown_token": tokenizer.unk_token,
        },
        failing_examples=[
            {"id": ex.id, "lang": ex.lang, "category": ex.category, "text": ex.text}
            for ex in failures[:max_failing_examples]
        ],
        notes=[
            "Metrics are computed on a hand-written evaluation fixture, not on real corpora.",
            "Compare tokenizers measured on the SAME corpus only.",
        ],
    )


def report_to_json(report: EvalReport) -> str:
    import json

    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
