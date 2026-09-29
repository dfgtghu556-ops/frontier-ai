#!/usr/bin/env python3
"""EXP-034: inspect the downloaded Sangraha Verified slice before any filtering decision.

    python scripts/inspect_sangraha_slice.py                 # all 13 files
    python scripts/inspect_sangraha_slice.py --only hi --max-docs 20000   # quick look

Read-only on the data. For every pinned file it (1) re-verifies size + SHA-256 against the pin,
(2) streams the parquet file once and measures what is inside (``frontier_ai.corpus.slice_inspect``:
documents, characters, length spread, web/OCR/speech ``type`` mix, wrong-script share under the
pipeline's own langid rule, exact duplicates, quality-rule hits, overlap with the protected
evaluation suite, and a sample-based token estimate with the frozen v1 tokenizer).

The protected-suite check needs the held-out shard of FrontierCorpus v1 on this machine
(``corpora/frontier/v1/shards/heldout/``, git-ignored). The shard is verified line by line against
``evals/suites/frontier-heldout-v1/SUITE.json`` before use. If the shard is absent, the report says
"NOT CHECKED" in so many words (never a silent zero); a shard that does not match the suite is an
error (exit 2).

Writes ``summary.json`` + ``SUMMARY.txt`` to ``--out`` (default ``out/data/EXP-034``) after every
file, so an interrupted night keeps what it measured. ``summary.json`` ends with
``"complete": true`` only when every requested file was inspected.
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

from frontier_ai.corpus.decontaminate import guard_from_heldout_shards
from frontier_ai.corpus.sangraha import SliceError, iter_rows, load_slice_pins, parquet_row_count, sha256_file
from frontier_ai.corpus.slice_inspect import inspect_rows as _inspect_rows
from frontier_ai.corpus.slice_inspect import render_text_report
from frontier_ai.evaluation.suite import SuiteError, load_suite

DEFAULT_PINS = "corpora/frontier/v2/sangraha_slice1.json"
DEFAULT_ROOT = "data/sangraha"
DEFAULT_SUITE = "evals/suites/frontier-heldout-v1/SUITE.json"
DEFAULT_HELDOUT = "corpora/frontier/v1/shards/heldout"
DEFAULT_OUT = "out/data/EXP-034"


def _write(out: Path, run: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(out / "summary.json")
    (out / "SUMMARY.txt").write_text(render_text_report(run), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="EXP-034")
    p.add_argument("--pins", default=DEFAULT_PINS)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--suite", default=DEFAULT_SUITE)
    p.add_argument("--heldout-dir", default=DEFAULT_HELDOUT)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--only", default="", help="comma-separated language codes (default: all)")
    p.add_argument("--max-docs", type=int, default=None, help="inspect at most N documents per file")
    p.add_argument("--no-tokens", action="store_true", help="skip the token estimate")
    p.add_argument("--token-every", type=int, default=50)
    p.add_argument("--token-char-budget", type=int, default=3_000_000)
    args = p.parse_args(argv)

    try:
        header, files = load_slice_pins(args.pins)
    except (SliceError, OSError, KeyError, TypeError) as exc:
        print(f"[inspect] bad pin file: {exc}", file=sys.stderr)
        return 2
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        files = tuple(f for f in files if f.language in wanted)
        if not files:
            print(f"[inspect] --only {args.only!r} matches no pinned file", file=sys.stderr)
            return 2
    root = Path(args.root) / header["revision"][:12]

    # ---- protected suite guard ------------------------------------------------------
    guard = None
    try:
        suite = load_suite(args.suite)
    except SuiteError as exc:
        print(f"[inspect] {exc}", file=sys.stderr)
        return 2
    shards = sorted(Path(args.heldout_dir).glob("*.txt"))
    if shards:
        try:
            guard = guard_from_heldout_shards(shards, suite)
        except ValueError as exc:
            print(f"[inspect] STOP: held-out shard does not match the protected suite: {exc}",
                  file=sys.stderr)
            return 2
        suite_status = (f"CHECKED against {suite['suite_id']} ({guard.suite_documents} documents, "
                        f"{len(guard.ngrams):,} 13-grams; shard verified against SUITE.json)")
    else:
        suite_status = (f"NOT CHECKED - no held-out shard in {args.heldout_dir} on this machine "
                        "(the build stage will require it)")
    print(f"[inspect] suite: {suite_status}", flush=True)

    tokenizer = None
    token_info = {"measured": False}
    if not args.no_tokens:
        from frontier_ai.tokenization.frozen import load_frontier_tokenizer

        tokenizer = load_frontier_tokenizer()
        token_info = {"measured": True, "tokenizer": "frontier-tokenizer-v1 (frozen, D-041)",
                      "vocab_size": tokenizer.vocab_size}

    run = {
        "schema": "frontier-slice-inspection-v1",
        "exp_id": args.exp_id,
        "slice_id": header["slice_id"],
        "dataset": header["dataset"],
        "revision": header["revision"],
        "license_id": header["license_id"],
        "attribution": header["attribution"],
        "suite_status": suite_status,
        "suite_guard": guard.describe() if guard is not None else None,
        "tokenizer": token_info,
        "max_docs_per_file": args.max_docs,
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "cpu_count": os.cpu_count()},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [],
        "complete": False,
    }
    out = Path(args.out)
    for i, pf in enumerate(files, 1):
        path = pf.local_path(root)
        print(f"[inspect] ({i}/{len(files)}) {pf.source_id}: verifying {path}", flush=True)
        if not path.is_file():
            print(f"[inspect] STOP: {path} is missing (run scripts/fetch_sangraha_slice.py first)",
                  file=sys.stderr)
            return 1
        if path.stat().st_size != pf.size or sha256_file(path) != pf.sha256:
            print(f"[inspect] STOP: {path} does not match its pin (size/sha256)", file=sys.stderr)
            return 1
        rows_declared = parquet_row_count(path)
        stats = _inspect_rows(
            iter_rows(path), source_id=pf.source_id, language=pf.language, script=pf.script,
            guard=guard, tokenizer=tokenizer, token_every=args.token_every,
            token_char_budget=args.token_char_budget, max_docs=args.max_docs,
            progress=lambda m: print(m, flush=True),
        )
        stats["pinned_sha256_verified"] = True
        stats["parquet_rows_declared"] = rows_declared
        run["files"].append(stats)
        _write(out, run)
        print(f"[inspect] {pf.source_id}: {stats['documents']:,} docs, {stats['chars'] / 1e6:,.1f}M chars, "
              f"{stats['seconds'] / 60:.1f} min", flush=True)

    run["complete"] = True
    run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(out, run)
    print(f"[inspect] done -> {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
