#!/usr/bin/env python3
"""EXP-035: calibrate the v2 build rules on the downloaded Sangraha slice (removes nothing).

    python scripts/calibrate_sangraha_slice.py                        # all 13 files
    python scripts/calibrate_sangraha_slice.py --only ur --max-docs 20000   # quick look

Read-only on the data. For every pinned file it re-verifies size + SHA-256 against the pin, then
streams the parquet file once (``frontier_ai.corpus.slice_calibrate``) and writes:

* ``summary.json`` / ``SUMMARY.txt`` with the counts: near-duplicates, repeated lines, language
  markers where scripts are shared, long-document windows, NFC causes, quality-rule hits by type,
  and short protected-suite passages contained word for word;
* ``samples.jsonl``, random, masked excerpts per question, for a human read before any
  threshold is chosen.

The short-suite check needs the v1 held-out shard on this machine (verified against SUITE.json,
exactly as in EXP-034). Without it the report says NOT CHECKED and the run is not complete.
Outputs are rewritten after every file, so an interrupted night keeps what it measured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.corpus.decontaminate import ShortSuiteIndex, SuiteGuard, heldout_pairs
from frontier_ai.corpus.sangraha import SliceError, iter_rows, load_slice_pins, parquet_row_count, sha256_file
from frontier_ai.corpus.slice_calibrate import SCHEMA, MinHasher, calibrate_rows, render_text_report
from frontier_ai.evaluation.suite import SuiteError, load_suite

DEFAULT_PINS = "corpora/frontier/v2/sangraha_slice1.json"
DEFAULT_ROOT = "data/sangraha"
DEFAULT_SUITE = "evals/suites/frontier-heldout-v1/SUITE.json"
DEFAULT_HELDOUT = "corpora/frontier/v1/shards/heldout"
DEFAULT_OUT = "out/data/EXP-035"
SHORT_MIN_WORDS = 3  # measure from 3 words up; the build's minimum is chosen from these counts


def _write(out: Path, run: dict, samples: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    run["samples_written"] = len(samples)
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(out / "summary.json")
    with open(out / "samples.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for item in samples:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "SUMMARY.txt").write_text(render_text_report(run), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="EXP-035")
    p.add_argument("--pins", default=DEFAULT_PINS)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--suite", default=DEFAULT_SUITE)
    p.add_argument("--heldout-dir", default=DEFAULT_HELDOUT)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--only", default="", help="comma-separated language codes (default: all)")
    p.add_argument("--max-docs", type=int, default=None, help="at most N documents per file (not complete)")
    args = p.parse_args(argv)

    try:
        header, files = load_slice_pins(args.pins)
    except (SliceError, OSError, KeyError, TypeError) as exc:
        print(f"[calibrate] bad pin file: {exc}", file=sys.stderr)
        return 2
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        files = tuple(f for f in files if f.language in wanted)
        if not files:
            print(f"[calibrate] --only {args.only!r} matches no pinned file", file=sys.stderr)
            return 2
    root = Path(args.root) / header["revision"][:12]

    try:
        suite = load_suite(args.suite)
    except SuiteError as exc:
        print(f"[calibrate] {exc}", file=sys.stderr)
        return 2
    shards = sorted(Path(args.heldout_dir).glob("*.txt"))
    short_index = guard = None
    if shards:
        try:
            pairs = heldout_pairs(shards, suite)
            short_index = ShortSuiteIndex.from_texts(pairs, min_words=SHORT_MIN_WORDS)
            guard = SuiteGuard.from_texts(pairs)
        except ValueError as exc:
            print(
                f"[calibrate] STOP: held-out shard does not match the protected suite: {exc}", file=sys.stderr
            )
            return 2
        d = short_index.describe()
        status = (
            f"CHECKED against {suite['suite_id']}: {d['indexed_documents']} suite documents of "
            f"{SHORT_MIN_WORDS}-12 words indexed ({d['not_indexed_too_short']} shorter; shard verified "
            "against SUITE.json)"
        )
    else:
        status = f"NOT CHECKED - no held-out shard in {args.heldout_dir} on this machine"
    print(f"[calibrate] short-suite: {status}", flush=True)

    run = {
        "schema": SCHEMA,
        "exp_id": args.exp_id,
        "slice_id": header["slice_id"],
        "dataset": header["dataset"],
        "revision": header["revision"],
        "license_id": header["license_id"],
        "attribution": header["attribution"],
        "short_suite_status": status,
        "samples_status": (
            "screened: documents touching the protected suite (exact, 13-gram, short passage) are not sampled"
            if shards
            else "WITHHELD: no held-out shard, so samples could not be screened against the suite"
        ),
        "max_docs_per_file": args.max_docs,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [],
        "complete": False,
    }
    samples: list[dict] = []
    out = Path(args.out)
    minhasher = MinHasher()
    for i, pf in enumerate(files, 1):
        path = pf.local_path(root)
        print(f"[calibrate] ({i}/{len(files)}) {pf.source_id}: verifying {path}", flush=True)
        if not path.is_file():
            print(
                f"[calibrate] STOP: {path} is missing (run scripts/fetch_sangraha_slice.py first)",
                file=sys.stderr,
            )
            return 1
        if path.stat().st_size != pf.size or sha256_file(path) != pf.sha256:
            print(f"[calibrate] STOP: {path} does not match its pin (size/sha256)", file=sys.stderr)
            return 1
        stats, file_samples = calibrate_rows(
            iter_rows(path),
            exp_id=args.exp_id,
            source_id=pf.source_id,
            language=pf.language,
            script=pf.script,
            expected_rows=parquet_row_count(path),
            short_index=short_index,
            guard=guard,
            minhasher=minhasher,
            max_docs=args.max_docs,
            progress=lambda m: print(m, flush=True),
        )
        stats["pinned_sha256_verified"] = True
        run["files"].append(stats)
        if guard is not None:  # samples are published only after screening against the suite
            samples.extend(file_samples)
        _write(out, run, samples)
        print(
            f"[calibrate] {pf.source_id}: {stats['documents']:,} docs, {stats['seconds'] / 60:.1f} min",
            flush=True,
        )

    run["complete"] = args.max_docs is None and not args.only and short_index is not None
    run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(out, run, samples)
    print(f"[calibrate] done -> {out / 'summary.json'} (complete: {run['complete']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
