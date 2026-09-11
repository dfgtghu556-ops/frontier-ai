#!/usr/bin/env python3
"""Build (or report on) the Project 004 tokenizer research corpus.

Examples
--------
    # report only: validate the manifest and show what is declared
    python scripts/build_tokenizer_corpus.py

    # attempt a real fetch of the declared sources (needs network to the source hosts)
    python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-008

    # print a previously built corpus manifest
    python scripts/build_tokenizer_corpus.py --print-json \
        --out data/tokenizer/indic-tokenizer-v2

Stage A only: this ingests declared sources, splits them, and reports coverage and
leakage. It never invents data — a source that cannot be fetched stays ``unverified``
and its language slot is reported as ``UNVERIFIED``.

Experiment records (D-032): the outer run owns the record. ``run_self_recorded`` handles
that, so this script records itself when run directly and publishes a single
``frontier_ai_nested_results`` line when it is nested (D-034).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.experiments import ExperimentSpec  # noqa: E402
from frontier_ai.experiments.autowire import run_self_recorded, stable_results  # noqa: E402
from frontier_ai.tokenization.research_corpus import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    BuildResult,
    TokenizerCorpusManifest,
    build_corpus,
    load_corpus,
    validate_manifest,
)

STATUS_ORDER = ["EVALUATED", "INSUFFICIENT", "UNVERIFIED", "NOT_EVALUATED"]
DEFAULT_OUT = "data/tokenizer/indic-tokenizer-v2"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="tokenizer corpus manifest (default: %(default)s)",
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: %(default)s)")
    p.add_argument("--fetch", action="store_true", help="fetch declared sources over https")
    p.add_argument("--timeout", type=float, default=30.0, help="per-source fetch timeout")
    p.add_argument("--source", action="append", dest="source_ids", help="ingest only this source id")
    p.add_argument(
        "--include-unverified",
        action="store_true",
        help="also include locally supplied (unverified) text in the split",
    )
    p.add_argument(
        "--pin",
        action="store_true",
        help="write verified sha256 values back into the manifest (never offline)",
    )
    p.add_argument("--seed", type=int, default=1337, help="split seed override (manifest default 1337)")
    p.add_argument("--exp-id", default="EXP-000", help="experiment id (default: %(default)s)")
    p.add_argument("--notes", default="", help="notes for the experiment record")
    p.add_argument("--print-json", action="store_true", help="print a built corpus.json and exit")
    p.add_argument("--no-record", action="store_true", help="do not write an experiment record")
    return p


def _render(result: BuildResult) -> str:
    lines = [
        f"[corpus] {result.manifest.corpus_id}/{result.manifest.corpus_version}",
        f"[corpus] output: {result.out_dir}",
        f"[corpus] sources: {len(result.ingested)} ingested, {result.verified_sources} verified",
        f"[corpus] documents: {len(result.train)} train / {len(result.held_out)} held out",
        "[corpus] coverage (per language slot)",
    ]
    by_status: dict[str, list[str]] = {status: [] for status in STATUS_ORDER}
    for row in result.coverage["languages"]:
        by_status.setdefault(row["status"], []).append(row["language"])
    for status in STATUS_ORDER:
        codes = by_status.get(status) or []
        lines.append(f"[corpus]   {status:<14} {len(codes):>2}  {' '.join(codes)}".rstrip())
    lines.append(
        "[corpus] leakage: exact_overlap={exact} ngram_{n}_overlap_ratio={ratio}".format(
            exact=result.leakage["exact"]["overlap_count"],
            n=result.leakage["ngram"]["n"],
            ratio=result.leakage["ngram"]["overlap_ratio"],
        )
    )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()

    if args.print_json:
        print(json.dumps(load_corpus(args.out), indent=2, ensure_ascii=False))
        return 0

    problems = validate_manifest(TokenizerCorpusManifest.load(args.manifest))
    if problems:
        for problem in problems:
            print(f"[corpus] manifest problem: {problem}", file=sys.stderr)
        return 2

    def body() -> dict:
        result = build_corpus(
            args.manifest,
            args.out,
            seed=args.seed,
            held_out_fraction=None,
            fetch=args.fetch,
            source_ids=args.source_ids,
            include_unverified=args.include_unverified,
            timeout=args.timeout,
            pin=args.pin,
        )
        print(_render(result))
        return stable_results(result.results_payload())

    if args.no_record:
        body()
        return 0

    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            seed=args.seed,
            name="tokenizer research corpus",
            # The record follows the documented convention (docs/experiments.md) and
            # stays out of the corpus directory, which is a data artifact.
            output_dir=str(Path("out/experiments") / args.exp_id),
            params={
                "corpus_id": TokenizerCorpusManifest.load(args.manifest).corpus_id,
                "corpus_version": TokenizerCorpusManifest.load(args.manifest).corpus_version,
                "fetch": bool(args.fetch),
                "pin": bool(args.pin),
                "include_unverified": bool(args.include_unverified),
                "source_ids": sorted(args.source_ids or []),
            },
            data_paths=[str(Path(args.manifest))],
            command=list(sys.argv),
            tags=["project-004", "corpus", "stage-a"],
            notes=args.notes,
        )

    return run_self_recorded(build_spec, body).exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
