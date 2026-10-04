#!/usr/bin/env python3
"""EXP-045 test 1 input: write the protected suite's texts to one JSON-lines file (founder's PC).

    python scripts/export_heldout_text.py            # -> out/eval/heldout-v1.jsonl

The protected suite ``frontier-heldout-v1`` (D-042) stores only hashes; its texts are re-derived
from the frozen FrontierCorpus v1 on the PC (the same identity gates as ``scripts/eval_report.py``).
This writes them, in suite order, with ``doc_id``, ``language``, ``source_id`` and ``text``, after
checking every document against the suite. The file (about 1 MB) is only an evaluation input: it
goes into ``out/`` (git-ignored) and, for the EXP-045 GPU session, into a private Kaggle dataset.
It must never be used for training (D-042). Takes seconds; no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_ai.corpus import verify_and_derive_frontier  # noqa: E402
from frontier_ai.evaluation.suite import (  # noqa: E402
    FRONTIER_HELDOUT_V1,
    SUITES_ROOT,
    load_suite,
    verify_docs_against_suite,  # noqa: E402
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default=str(SUITES_ROOT / FRONTIER_HELDOUT_V1 / "SUITE.json"))
    p.add_argument("--frontier-dir", default="corpora/frontier/v1")
    p.add_argument("--manifest", default="corpora/tokenizer/indic-tokenizer-v2/sources.json")
    p.add_argument("--freeze", default="corpora/tokenizer/indic-tokenizer-v2/FREEZE.json")
    p.add_argument("--corpus-dir", default="data/tokenizer/indic-tokenizer-v2")
    p.add_argument("--out", default="out/eval/heldout-v1.jsonl")
    args = p.parse_args(argv)

    suite = load_suite(args.suite)
    _train, held, manifest = verify_and_derive_frontier(
        args.frontier_dir, Path(args.manifest), Path(args.freeze), Path(args.corpus_dir)
    )
    if suite["source"].get("content_sha256") != manifest["content_sha256"]:
        raise SystemExit("the suite was built from a different corpus version than the one on disk")
    verify_docs_against_suite(held, suite)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"doc_id": d.doc_id, "language": d.language, "source_id": d.source_id, "text": d.text} for d in held
    ]
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    print(f"[heldout] {len(rows)} documents (all match {suite['suite_id']}) -> {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
