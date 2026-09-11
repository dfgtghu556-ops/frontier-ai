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
        "--local-file",
        action="append",
        dest="local_files",
        metavar="SOURCE_ID=PATH",
        help="ingest lawfully obtained local text for a declared source "
             "(repeatable). Recorded as local_unverified: it is never EVALUATED.",
    )
    p.add_argument(
        "--local-dir",
        metavar="DIR",
        help="directory of <source_id>.txt files to ingest as local_unverified text",
    )
    p.add_argument(
        "--include-unverified",
        action="store_true",
        help="include local_unverified text in the split (off by default; local text is "
             "ingested and reported either way, and never counts as EVALUATED)",
    )
    p.add_argument(
        "--pin",
        action="store_true",
        help="write verified sha256 values back into the manifest (only genuinely "
             "verified sources; the manifest is left untouched when nothing was pinned)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the manifest split seed (default: the manifest's own seed)",
    )
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


def _parse_local_files(
    manifest: TokenizerCorpusManifest, entries: list[str] | None, directory: str | None
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve ``--local-file ID=PATH`` / ``--local-dir DIR`` into text + origin paths.

    Raises ``ValueError`` on anything ambiguous: an unknown source id or a file in
    ``--local-dir`` that matches no declared source must be a hard error, never a
    silently ignored input.
    """
    declared = {source.id for source in manifest.sources}
    wanted: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"--local-file expects SOURCE_ID=PATH, got {entry!r}")
        source_id, _, path = entry.partition("=")
        if source_id not in declared:
            raise ValueError(f"--local-file: no declared source with id {source_id!r}")
        wanted[source_id] = path

    if directory:
        for path in sorted(Path(directory).glob("*.txt")):
            source_id = path.stem
            if source_id not in declared:
                raise ValueError(
                    f"--local-dir: {path.name} matches no declared source id "
                    f"(declared: {', '.join(sorted(declared))})"
                )
            wanted.setdefault(source_id, str(path))

    texts: dict[str, str] = {}
    origins: dict[str, str] = {}
    for source_id, path in wanted.items():
        target = Path(path)
        if not target.is_file():
            raise ValueError(f"--local-file: no such file for {source_id!r}: {path}")
        texts[source_id] = target.read_text(encoding="utf-8")
        origins[source_id] = str(target)
    return texts, origins


def main() -> int:
    args = build_parser().parse_args()

    if args.print_json:
        print(json.dumps(load_corpus(args.out), indent=2, ensure_ascii=False))
        return 0

    manifest = TokenizerCorpusManifest.load(args.manifest)
    problems = validate_manifest(manifest)
    if problems:
        for problem in problems:
            print(f"[corpus] manifest problem: {problem}", file=sys.stderr)
        return 2

    try:
        local_texts, local_origins = _parse_local_files(
            manifest, args.local_files, args.local_dir
        )
    except ValueError as exc:
        print(f"[corpus] {exc}", file=sys.stderr)
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
            local_texts=local_texts,
            local_origins=local_origins,
        )
        print(_render(result))
        return stable_results(result.results_payload())

    if args.no_record:
        body()
        return 0

    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            # args.seed is None when the manifest seed applies; the record then stores
            # the seed that was actually used.
            seed=args.seed if args.seed is not None else manifest.split_seed,
            name="tokenizer research corpus",
            # The record follows the documented convention (docs/experiments.md) and
            # stays out of the corpus directory, which is a data artifact.
            output_dir=str(Path("out/experiments") / args.exp_id),
            params={
                "corpus_id": manifest.corpus_id,
                "corpus_version": manifest.corpus_version,
                "fetch": bool(args.fetch),
                "pin": bool(args.pin),
                "include_unverified": bool(args.include_unverified),
                "source_ids": sorted(args.source_ids or []),
                "local_sources": sorted(local_texts),
                "split_seed": args.seed if args.seed is not None else manifest.split_seed,
            },
            data_paths=[str(Path(args.manifest))],
            command=list(sys.argv),
            tags=["project-004", "corpus", "stage-a"],
            notes=args.notes,
        )

    return run_self_recorded(build_spec, body).exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
