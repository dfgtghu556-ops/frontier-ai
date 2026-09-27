"""Training/evaluation contamination checks.

Two measurements between the evaluation documents and a set of training documents:

* **exact**: evaluation documents whose exact text (SHA-256) occurs in the training set;
* **13-gram**: evaluation documents sharing at least one whitespace-delimited 13-word
  sequence with the training set (the long-n-gram overlap criterion used in large-scale
  LM contamination studies). Documents shorter than 13 words cannot be checked this way
  and are counted separately — reported, not guessed.

Overlap is *reported*, not silently removed: repeated formulae, verse refrains or
boilerplate can legitimately occur on both sides of a split, and the reader decides with
the numbers in front of them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

NGRAM = 13


def _ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def contamination_report(
    eval_docs: Sequence[Any], train_docs: Sequence[Any], n: int = NGRAM
) -> dict[str, Any]:
    train_hashes = {hashlib.sha256(d.text.encode("utf-8")).hexdigest() for d in train_docs}
    train_grams: set[str] = set()
    for d in train_docs:
        train_grams |= _ngrams(d.text, n)

    exact: list[str] = []
    overlap: list[str] = []
    eligible = 0
    for d in eval_docs:
        if hashlib.sha256(d.text.encode("utf-8")).hexdigest() in train_hashes:
            exact.append(d.doc_id)
        grams = _ngrams(d.text, n)
        if grams:
            eligible += 1
            if grams & train_grams:
                overlap.append(d.doc_id)
    return {
        "training_documents": len(train_docs),
        "evaluation_documents": len(eval_docs),
        "exact_duplicates": len(exact),
        "exact_duplicate_doc_ids": exact[:20],
        "ngram_n": n,
        "ngram_eligible_documents": eligible,
        "ngram_too_short_documents": len(eval_docs) - eligible,
        "ngram_overlap_documents": len(overlap),
        "ngram_overlap_fraction": (len(overlap) / eligible) if eligible else None,
        "ngram_overlap_doc_ids_sample": overlap[:20],
    }
