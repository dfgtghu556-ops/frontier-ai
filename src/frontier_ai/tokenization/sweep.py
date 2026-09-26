"""EXP-A: the production tokenizer sweep — configuration grid, gate and scoring.

Scope (approved 2026-09-26; D-038): 15 configurations —
``bpe_python`` x {mark_aware, gpt2_style} x 5 vocab sizes, plus ``bpe_hf``
(Built-in ByteLevel pre-tokenization, the GPT-2-style variant) x 5 vocab sizes —
each trained on the FrontierCorpus v1 train side, gated on losslessness, and
scored model-free on the held-out side. No tokenizer is *selected* here; the
top-2 hand-off to EXP-B (small model, >= 3 seeds) happens after the sweep.

Per-configuration metrics (EXP-A is model-free by design):

* **losslessness gate** — ``decode(encode(doc)) == doc`` for EVERY document on
  both sides; 100 % is required, a single failure fails the configuration;
* **held-out token density** — ``tokens_per_char`` / ``chars_per_token`` /
  ``tokens_per_word`` overall and per language, via the evaluator's ``_measure``
  with documents mapped to :class:`~frontier_ai.tokenization.corpus.Example`;
* **training time** and final vocab size.

Bits-per-character (which needs a language model) is deliberately NOT an
EXP-A metric — that comparison belongs to EXP-B.

The five ``bpe_hf`` + mark-aware combinations are out of the grid: they would
need a custom HuggingFace ``PreTokenizer``, which cannot be tested until the
PC run; they are recorded as a documented limitation, not silently skipped.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..corpus import PipelineDocument
from . import create
from .base import SubwordTokenizer, TokenizerError
from .corpus import Example
from .evaluate import StatBlock, _measure

PRETOKEN_MARK_AWARE = "mark_aware"
PRETOKEN_GPT2_STYLE = "gpt2_style"
PRETOKEN_HF_BYTE_LEVEL = "byte_level"

IMPL_BPE_PYTHON = "bpe_python"
IMPL_BPE_HF = "bpe_hf"

# The headline number the sweep aggregate tracks (held-out chars per token;
# higher = denser compression). Dotted path into each run's results dict.
HEADLINE_METRIC = "heldout.overall.chars_per_token"


@dataclass(frozen=True)
class SweepConfig:
    """One grid cell: tokenizer implementation, pre-tokenization, vocab size."""

    impl: str
    pretoken: str
    vocab_size: int

    @property
    def name(self) -> str:
        impl = "py" if self.impl == IMPL_BPE_PYTHON else "hf"
        return f"{impl}-{self.pretoken}-{self.vocab_size}"

    @staticmethod
    def grid(vocab_sizes: Sequence[int]) -> tuple[SweepConfig, ...]:
        """The EXP-A grid: 2 pre-tokens for bpe_python + ByteLevel for bpe_hf."""
        configs = [
            SweepConfig(IMPL_BPE_PYTHON, pretoken, vocab)
            for vocab in vocab_sizes
            for pretoken in (PRETOKEN_MARK_AWARE, PRETOKEN_GPT2_STYLE)
        ]
        configs += [
            SweepConfig(IMPL_BPE_HF, PRETOKEN_HF_BYTE_LEVEL, vocab) for vocab in vocab_sizes
        ]
        return tuple(configs)


def docs_to_examples(documents: Sequence[PipelineDocument]) -> list[Example]:
    """Map corpus documents onto the evaluator's example model (category: document)."""
    return [
        Example(id=d.doc_id, lang=d.language, category="document", text=d.text)
        for d in documents
    ]


def gate_losslessness(
    tokenizer: SubwordTokenizer,
    documents: Sequence[PipelineDocument],
    max_failures_recorded: int = 5,
) -> dict[str, Any]:
    """Exact round-trip of every document; 100 % required to pass the gate."""
    failures: list[dict[str, str]] = []
    for doc in documents:
        if tokenizer.decode(tokenizer.encode(doc.text)) != doc.text:
            failures.append({"doc_id": doc.doc_id, "excerpt": doc.text[:80]})
            if len(failures) >= max_failures_recorded:
                break
    return {
        "checked": len(documents),
        "failures": min(len(failures), max_failures_recorded),
        "truncated": len(failures) > max_failures_recorded,
        "lossless": not failures,
        "sample": failures,
    }


def measure_docs(
    tokenizer: SubwordTokenizer,
    documents: Sequence[PipelineDocument],
) -> dict[str, dict[str, Any]]:
    """Evaluator metrics (``_measure``) over documents: overall + per language."""
    examples = docs_to_examples(documents)
    if not examples:
        raise ValueError("no documents to measure")
    overall, _failures = _measure(tokenizer, examples)
    per_language: dict[str, dict[str, Any]] = {}
    by_lang: dict[str, list[Example]] = {}
    for ex in examples:
        by_lang.setdefault(ex.lang, []).append(ex)
    for lang in sorted(by_lang):
        block: StatBlock = _measure(tokenizer, by_lang[lang])[0]
        per_language[lang] = block.to_dict()
    return {"overall": overall.to_dict(), "per_language": per_language}


def run_one_config(
    config: SweepConfig,
    *,
    train_file: str | Path,
    train_documents: Sequence[PipelineDocument],
    heldout_documents: Sequence[PipelineDocument],
    out_dir: str | Path,
    special_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    """Train one grid cell, gate it, score it, and save its artifact.

    Returns the results dict recorded by the sweep (flat enough for the
    ``headline metric`` extraction, detailed enough for the report). Raises on
    training errors; a failed losslessness gate is a *result* (the configuration
    is reported as failed), not an exception.
    """
    if config.impl == IMPL_BPE_PYTHON:
        kwargs: dict[str, Any] = {"pretoken": config.pretoken}
    elif config.impl == IMPL_BPE_HF:
        if config.pretoken != PRETOKEN_HF_BYTE_LEVEL:
            raise TokenizerError(
                f"{config.impl} only supports the built-in {PRETOKEN_HF_BYTE_LEVEL} "
                "pre-tokenization in EXP-A (a mark-aware PreTokenizer is not part of the grid)"
            )
        kwargs = {}
    else:
        raise TokenizerError(f"unknown tokenizer implementation {config.impl!r}")

    started = time.monotonic()
    tokenizer = create(config.impl)
    tokenizer.train(train_file, vocab_size=config.vocab_size, special_tokens=special_tokens, **kwargs)
    train_seconds = round(time.monotonic() - started, 3)

    out_dir = Path(out_dir)
    artifact_dir = tokenizer.save(out_dir / "tokenizer")

    train_file = Path(train_file)
    train_file_chars = len(train_file.read_text(encoding="utf-8"))

    gate_train = gate_losslessness(tokenizer, train_documents)
    gate_held = gate_losslessness(tokenizer, heldout_documents)
    heldout = measure_docs(tokenizer, heldout_documents) if heldout_documents else {
        "overall": {},
        "per_language": {},
    }

    return {
        "config": {"impl": config.impl, "pretoken": config.pretoken, "vocab_size": config.vocab_size},
        "train": {
            "seconds": train_seconds,
            "documents": len(train_documents),
            "chars": sum(d.chars for d in train_documents),
            # the file actually trained on (differs from chars only in smoke mode,
            # where --max-train-chars truncates it)
            "file_chars": train_file_chars,
            "file_truncated": train_file_chars < sum(d.chars for d in train_documents),
        },
        "vocab_size": tokenizer.vocab_size,
        "gate": {
            "train": gate_train,
            "heldout": gate_held,
            "lossless": gate_train["lossless"] and gate_held["lossless"],
        },
        "heldout": {
            "documents": len(heldout_documents),
            "chars": sum(d.chars for d in heldout_documents),
            **heldout,
        },
        "artifact": str(artifact_dir),
    }
