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

from frontier_ai.experiments import ExperimentSpec  # noqa: E402
from frontier_ai.experiments.autowire import run_self_recorded, stable_results  # noqa: E402
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
    p.add_argument("--no-record", action="store_true",
                   help="do not write an experiment record for this run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.tokenizer) < 2:
        raise SystemExit("[compare] provide at least two --tokenizer artifacts")

    corpus_dir = resolve_corpus_dir(args.corpus)

    def body() -> dict:
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
        out_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        if not args.quiet:
            print(render_comparison(comparison))
            print()
        print(
            f"[compare] corpus={manifest.corpus_id}-{manifest.corpus_version} "
            f"examples={manifest.n_examples}"
        )
        print(f"[compare] wrote {out_path}")
        stable = stable_results(comparison)
        return {
            "corpus_id": manifest.corpus_id,
            "corpus_version": manifest.corpus_version,
            "n_examples": manifest.n_examples,
            "labels": labels,
            "rows": stable["rows"],
            "best": stable.get("best", {}),
            "disqualified": stable.get("disqualified", []),
            "comparison_path": str(out_path),
        }

    if args.no_record:
        body()
        return 0

    # a comparison is a measurement, not just an artifact: record it with the corpus and
    # every tokenizer artifact it read as inputs.
    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            seed=1337,
            name="tokenizer comparison",
            output_dir=str(Path(args.out).parent or "."),
            params={"corpus": args.corpus, "tokenizers": list(args.tokenizer)},
            data_paths=[str(corpus_dir), *[str(t) for t in args.tokenizer]],
            command=list(sys.argv),
            tags=["project-002", "comparison"],
        )
    return run_self_recorded(build_spec, body).exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
