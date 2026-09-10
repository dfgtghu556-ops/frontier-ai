#!/usr/bin/env python3
"""Compare tokenizer artifacts against exactly the same evaluation corpus.

The evaluator is never modified to add a tokenizer: point this command at any number of
artifact directories and it produces a machine-readable comparison plus a summary table.

Example
-------
    python scripts/tokenizer_compare.py --corpus data/tokenizer/indic-v1 \
        --tokenizer artifacts/tokenizers/char artifacts/tokenizers/word \
                    artifacts/tokenizers/bpe_hf_1k artifacts/tokenizers/bpe_py_1k \
        --out out/tokenizer/compare.json --exp-id EXP-006
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.tokenization.cli import (  # noqa: E402
    corpus_meta,
    load_tokenizer_artifacts,
    resolve_corpus_dir,
)
from frontier_ai.tokenization.compare import build_comparison, render_comparison  # noqa: E402
from frontier_ai.tokenization.corpus import load_corpus  # noqa: E402
from frontier_ai.tokenization.evaluate import evaluate_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="corpus directory written by the prepare CLI")
    p.add_argument("--tokenizer", nargs="+", required=True, help="two or more artifact directories")
    p.add_argument("--label", nargs="*", default=None, help="optional display names per tokenizer")
    p.add_argument("--out", default="out/tokenizer/compare.json", help="output JSON path")
    p.add_argument("--exp-id", default="EXP-000", help="experiment id recorded in each report")
    p.add_argument("--quiet", action="store_true", help="write the JSON without printing the table")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.tokenizer) < 2:
        raise SystemExit("[compare] provide at least two --tokenizer artifacts")

    corpus_dir = resolve_corpus_dir(args.corpus)
    examples, manifest = load_corpus(corpus_dir)
    loaded = load_tokenizer_artifacts(args.tokenizer, args.label)
    labels = [label for label, _, _ in loaded]

    reports = [
        evaluate_tokenizer(
            tokenizer,
            examples,
            manifest,
            experiment_id=args.exp_id,
            tokenizer_meta={
                "artifact_dir": str(art.artifact_dir or ""),
                "train_params": art.train_params,
                "impl_version_recorded": art.impl_version,
            },
        )
        for label, tokenizer, art in loaded
    ]

    comparison = build_comparison(reports, labels=labels, corpus=corpus_meta(corpus_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(render_comparison(comparison))
        print()
    print(f"[compare] corpus={manifest.corpus_id}-{manifest.corpus_version} examples={manifest.n_examples}")
    print(f"[compare] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
