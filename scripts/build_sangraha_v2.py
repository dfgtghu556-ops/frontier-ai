#!/usr/bin/env python3
"""EXP-036: build FrontierCorpus v2 from the pinned Sangraha Verified slice (the approved 10 rules).

    python scripts/build_sangraha_v2.py                                   # all 13 files
    python scripts/build_sangraha_v2.py --only ur --max-docs 20000        # quick look (not complete)

For every pinned file it re-verifies size + SHA-256 against the pin, then builds the file with
``frontier_ai.corpus.slice_build`` (rules, reasons, exact token count with the frozen tokenizer,
suite check on the exact output text). Writes:

* the corpus: ``<data-out>/<lang>.jsonl.gz`` (one JSON document per line with provenance),
  ``<data-out>/ATTRIBUTION.txt`` and ``<data-out>/manifest.json``. It stays on this machine
  (``data/`` is git-ignored);
* the evidence, small and committable: ``<out>/summary.json``, ``SUMMARY.txt``, ``manifest.json``
  and ``samples.jsonl`` (masked excerpts per removal reason and of kept documents).

The protected-suite check is mandatory: without the v1 held-out shard (verified against
SUITE.json) the build STOPs before reading any data. So does a free-space shortfall.

Outputs are rewritten after every file. A finished file is recorded in ``<data-out>/<lang>.stats.json``;
running the same command again skips files whose output still matches (same code, rules, pin,
suite and tokenizer) and builds only the rest, so an interrupted night loses at most the files
that were in progress.

``--workers 2`` (default) builds two files at a time: the founder's laptop has two cores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.corpus.decontaminate import ShortSuiteIndex, SuiteGuard, heldout_pairs  # noqa: E402
from frontier_ai.corpus.sangraha import (  # noqa: E402
    PinnedFile,
    SliceError,
    iter_rows,
    load_slice_pins,
    parquet_row_count,
    sha256_file,
)
from frontier_ai.corpus.slice_build import (  # noqa: E402
    SCHEMA,
    BuildConfig,
    TokenCounter,
    build_file,
    render_text_report,
)
from frontier_ai.evaluation.suite import SuiteError, load_suite  # noqa: E402

DEFAULT_PINS = "corpora/frontier/v2/sangraha_slice1.json"
DEFAULT_ROOT = "data/sangraha"
DEFAULT_SUITE = "evals/suites/frontier-heldout-v1/SUITE.json"
DEFAULT_HELDOUT = "corpora/frontier/v1/shards/heldout"
DEFAULT_OUT = "out/data/EXP-036"
DEFAULT_DATA_OUT = "data/frontier_v2/sangraha-slice1-v2"
CORPUS_ID = "frontier-v2-sangraha-slice1"
MANIFEST_SCHEMA = "frontier-corpus-manifest-v1"
TOKENIZER_NAME = "frontier-tokenizer-v1 (frozen, D-041)"
# Output (gzip JSONL) is about the size of the parquet input or smaller (NOT VERIFIED before the
# first build); the temporary candidate file of a file in progress adds up to one more copy.
FREE_SPACE_FACTOR = 1.5
FREE_SPACE_MARGIN = 2 * 1024**3

RECORD_FIELDS = {
    "id": "<source_id>#<row>: unique in the corpus",
    "text": "the document after the EXP-036 rules (NFC, cleaned lines, masked contacts)",
    "lang": "declared language code of the source file",
    "source": "pinned source file id (corpora/frontier/v2/sangraha_slice1.json)",
    "row": "0-based row in the source parquet file",
    "doc_id": "Sangraha doc_id (NOT unique, EXP-035)",
    "type": "Sangraha document type (web, pdf, ...)",
    "revision": "Hugging Face dataset revision",
    "tokens": "exact token count with frontier-tokenizer-v1",
}

_WORKER: dict[str, Any] = {}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _module_sha() -> str:
    import frontier_ai.corpus.slice_build as sb

    # CRLF-insensitive, so a Windows checkout and a Linux checkout agree.
    return _sha256_bytes(Path(sb.__file__).read_bytes().replace(b"\r\n", b"\n"))[:16]


def build_key(
    pf: PinnedFile, cfg: BuildConfig, suite_fp: str, tokenizer_fp: str, max_docs: int | None
) -> str:
    blob = json.dumps(
        {
            "schema": SCHEMA,
            "config": cfg.fingerprint(),
            "code": _module_sha(),
            "pin": pf.sha256,
            "suite": suite_fp,
            "tokenizer": tokenizer_fp,
            "max_docs": max_docs,
        },
        sort_keys=True,
    )
    return _sha256_bytes(blob.encode("utf-8"))[:24]


def _init_worker(pairs: list[tuple[str, str]], short_min_words: int) -> None:
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer

    _WORKER["guard"] = SuiteGuard.from_texts(pairs)
    _WORKER["short"] = ShortSuiteIndex.from_texts(pairs, min_words=short_min_words)
    tokenizer = load_frontier_tokenizer()
    _WORKER["tokens"] = TokenCounter(tokenizer)


def _build_task(task: dict[str, Any]) -> dict[str, Any]:
    """Verify one pinned file and build it (runs in a worker process, or in-process)."""
    pf: PinnedFile = task["pf"]
    path = Path(task["path"])

    def say(msg: str) -> None:
        print(msg, flush=True)

    say(f"[build] {pf.source_id}: verifying {path}")
    if not path.is_file():
        return {"error": f"{path} is missing (run scripts/fetch_sangraha_slice.py first)", "pf": pf}
    if path.stat().st_size != pf.size or sha256_file(path) != pf.sha256:
        return {"error": f"{path} does not match its pin (size/sha256)", "pf": pf}
    _WORKER["tokens"].cache.clear()  # bound memory: each file starts with an empty cache
    stats, samples = build_file(
        lambda: iter_rows(path),
        out_path=Path(task["out_path"]),
        exp_id=task["exp_id"],
        source_id=pf.source_id,
        language=pf.language,
        script=pf.script,
        revision=task["revision"],
        guard=_WORKER["guard"],
        short_index=_WORKER["short"],
        token_counter=_WORKER["tokens"],
        config=task["config"],
        expected_rows=parquet_row_count(path),
        max_docs=task["max_docs"],
        progress=say,
    )
    stats["pinned_sha256_verified"] = True
    stats["build_key"] = task["key"]
    say(
        f"[build] {pf.source_id}: kept {stats['kept']['docs']:,} of {stats['input']['docs']:,} docs, "
        f"{stats['kept']['tokens']:,} tokens, {stats['seconds'] / 60:.1f} min"
    )
    return {"stats": stats, "samples": samples, "pf": pf}


def _stats_path(data_out: Path, pf: PinnedFile) -> Path:
    return data_out / f"{pf.language}.stats.json"


def _reusable(data_out: Path, pf: PinnedFile, key: str) -> dict[str, Any] | None:
    sp = _stats_path(data_out, pf)
    if not sp.is_file():
        return None
    try:
        saved = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stats = saved.get("stats", {})
    out = data_out / stats.get("output", {}).get("path", "?")
    if stats.get("build_key") != key or not out.is_file():
        return None
    if sha256_file(out) != stats["output"]["sha256_gz"]:
        return None
    return saved


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def _manifest(run: dict[str, Any], header: dict[str, Any], pins_path: str) -> dict[str, Any]:
    files = []
    tot = {"docs": 0, "chars": 0, "tokens": 0}
    for f in run["files"]:
        k = f["kept"]
        files.append(
            {
                "language": f["language"],
                "source_id": f["source_id"],
                "path": f["output"]["path"],
                "docs": k["docs"],
                "chars": k["chars"],
                "tokens": k["tokens"],
                "sha256_gz": f["output"]["sha256_gz"],
                "sha256_jsonl": f["output"]["sha256_jsonl"],
                "bytes_gz": f["output"]["bytes_gz"],
            }
        )
        for key in tot:
            tot[key] += k[key]
    return {
        "schema": MANIFEST_SCHEMA,
        "corpus_id": CORPUS_ID,
        "exp_id": run["exp_id"],
        "complete": run["complete"],
        "built_at": run.get("finished_at") or run["started_at"],
        "source": {
            "dataset": header["dataset"],
            "subset": header.get("subset"),
            "revision": header["revision"],
            "slice_id": header["slice_id"],
            "pins": pins_path,
            "license_id": header["license_id"],
            "attribution": header["attribution"],
        },
        "rules": "EXPERIMENTS.md EXP-036 (approved 2026-09-29); code: src/frontier_ai/corpus/slice_build.py",
        "config": run["config"],
        "config_fingerprint": run["config_fingerprint"],
        "tokenizer": run["tokenizer"],
        "suite": run["suite"],
        "record_fields": RECORD_FIELDS,
        "files": files,
        "totals": tot,
        "token_count_status": "exact count with the frozen tokenizer over every kept document",
    }


def _attribution(header: dict[str, Any]) -> str:
    return (
        f"{CORPUS_ID} is derived from {header['dataset']} ({header.get('subset', '')}), revision "
        f"{header['revision']}.\n\n"
        f"Source attribution: {header['attribution']}\n"
        f"License of the source: {header['license_id']}.\n\n"
        "Changes made: Unicode NFC normalization and whitespace cleanup; documents removed for\n"
        "protected-evaluation overlap, duplication, wrong language/script, wiki markup, repetition\n"
        "and control characters; repeated boilerplate lines and lines without a letter of the\n"
        "declared script removed; e-mail addresses and phone numbers replaced by [email]/[phone].\n"
        "Rules: EXPERIMENTS.md EXP-036 in the frontier-ai repository.\n"
    )


def _write_outputs(
    out: Path,
    data_out: Path,
    run: dict[str, Any],
    samples: list[dict[str, Any]],
    header: dict[str, Any],
    pins_path: str,
) -> None:
    order = {sid: i for i, sid in enumerate(run["_order"])}
    run["files"].sort(key=lambda f: order[f["source_id"]])
    samples.sort(key=lambda s: (order[s["source_id"]], s["stratum"], s["row"]))
    public = {k: v for k, v in run.items() if not k.startswith("_")}
    public["samples_written"] = len(samples)
    _write_json(out / "summary.json", public)
    with open(out / "samples.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for item in samples:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "SUMMARY.txt").write_text(render_text_report(public), encoding="utf-8", newline="\n")
    manifest = _manifest(public, header, pins_path)
    _write_json(out / "manifest.json", manifest)
    _write_json(data_out / "manifest.json", manifest)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="EXP-036")
    p.add_argument("--pins", default=DEFAULT_PINS)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--suite", default=DEFAULT_SUITE)
    p.add_argument("--heldout-dir", default=DEFAULT_HELDOUT)
    p.add_argument("--out", default=DEFAULT_OUT, help="small reports (published)")
    p.add_argument("--data-out", default=DEFAULT_DATA_OUT, help="the corpus (stays on this machine)")
    p.add_argument("--only", default="", help="comma-separated language codes (default: all)")
    p.add_argument("--max-docs", type=int, default=None, help="at most N documents per file (not complete)")
    p.add_argument("--workers", type=int, default=2, help="files built in parallel (default 2)")
    args = p.parse_args(argv)

    try:
        header, files = load_slice_pins(args.pins)
    except (SliceError, OSError, KeyError, TypeError) as exc:
        print(f"[build] bad pin file: {exc}", file=sys.stderr)
        return 2
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        files = tuple(f for f in files if f.language in wanted)
        if not files:
            print(f"[build] --only {args.only!r} matches no pinned file", file=sys.stderr)
            return 2
    root = Path(args.root) / header["revision"][:12]
    out = Path(args.out)
    data_out = Path(args.data_out)
    cfg = BuildConfig()

    # ---- protected suite: mandatory ----
    try:
        suite = load_suite(args.suite)
    except SuiteError as exc:
        print(f"[build] {exc}", file=sys.stderr)
        return 2
    shards = sorted(Path(args.heldout_dir).glob("*.txt"))
    if not shards:
        print(
            f"[build] STOP: no held-out shard in {args.heldout_dir}. The build must check every document "
            "against the protected suite, so it does not run without it.",
            file=sys.stderr,
        )
        return 2
    try:
        pairs = heldout_pairs(shards, suite)
        guard = SuiteGuard.from_texts(pairs)
        short_index = ShortSuiteIndex.from_texts(pairs, min_words=cfg.short_suite_min_words)
    except ValueError as exc:
        print(f"[build] STOP: held-out shard does not match the protected suite: {exc}", file=sys.stderr)
        return 2
    suite_fp = _sha256_bytes(Path(args.suite).read_bytes().replace(b"\r\n", b"\n"))[:16]
    suite_status = (
        f"CHECKED against {suite['suite_id']} ({guard.suite_documents} documents; 13-grams + exact + "
        f"passages of {cfg.short_suite_min_words}-12 words; shard verified against SUITE.json)"
    )
    print(f"[build] suite: {suite_status}", flush=True)

    # ---- frozen tokenizer ----
    from frontier_ai.tokenization.frozen import FRONTIER_TOKENIZER_V1, FROZEN_ROOT, load_frozen_tokenizer

    tokenizer, freeze = load_frozen_tokenizer(FROZEN_ROOT / FRONTIER_TOKENIZER_V1)
    tokenizer_fp = freeze["artifact"]["dir_sha256"][:16]
    del tokenizer

    # ---- resume + free space ----
    keys = {pf.source_id: build_key(pf, cfg, suite_fp, tokenizer_fp, args.max_docs) for pf in files}
    reused: dict[str, dict[str, Any]] = {}
    for pf in files:
        saved = _reusable(data_out, pf, keys[pf.source_id])
        if saved is not None:
            reused[pf.source_id] = saved
    todo = [pf for pf in files if pf.source_id not in reused]
    data_out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_out).free
    need = int(sum(pf.size for pf in todo) * FREE_SPACE_FACTOR) + FREE_SPACE_MARGIN if todo else 0
    if args.max_docs is None and free < need:
        print(
            f"[build] STOP: not enough free disk space in {data_out.resolve()}: {free / 1e9:.1f} GB free, "
            f"need about {need / 1e9:.1f} GB. Free some space (e.g. empty the Recycle Bin) and run again.",
            file=sys.stderr,
        )
        return 2
    print(f"[build] disk: {free / 1e9:.1f} GB free, about {need / 1e9:.1f} GB needed", flush=True)
    for sid in reused:
        print(f"[build] {sid}: already built with the same code, rules, pin, suite and tokenizer; reused")

    run: dict[str, Any] = {
        "schema": SCHEMA,
        "exp_id": args.exp_id,
        "slice_id": header["slice_id"],
        "dataset": header["dataset"],
        "revision": header["revision"],
        "license_id": header["license_id"],
        "attribution": header["attribution"],
        "config": asdict(cfg),
        "config_fingerprint": cfg.fingerprint(),
        "code_sha256_16": _module_sha(),
        "tokenizer": {"name": TOKENIZER_NAME, "dir_sha256_16": tokenizer_fp, "exact_count": True},
        "suite": {
            "status": suite_status,
            "suite_id": suite["suite_id"],
            "suite_sha256_16": suite_fp,
            **guard.describe(),
            "short_index": short_index.describe(),
        },
        "suite_status": suite_status,
        "samples_status": "screened: every sampled document passed the suite check on its full text; "
        "suite-touching documents are never sampled",
        "max_docs_per_file": args.max_docs,
        "workers": args.workers,
        "data_out": str(data_out),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [saved["stats"] for saved in reused.values()],
        "reused_files": sorted(reused),
        "complete": False,
        "_order": [pf.source_id for pf in files],
    }
    samples: list[dict[str, Any]] = [s for saved in reused.values() for s in saved["samples"]]
    _write_outputs(out, data_out, run, samples, header, args.pins)

    tasks = [
        {
            "pf": pf,
            "path": str(pf.local_path(root)),
            "out_path": str(data_out / f"{pf.language}.jsonl.gz"),
            "exp_id": args.exp_id,
            "revision": header["revision"],
            "config": cfg,
            "max_docs": args.max_docs,
            "key": keys[pf.source_id],
        }
        for pf in sorted(todo, key=lambda f: -f.size)  # largest first: better balance over 2 workers
    ]

    def record(result: dict[str, Any]) -> bool:
        if "error" in result:
            print(f"[build] STOP: {result['error']}", file=sys.stderr)
            return False
        pf = result["pf"]
        _write_json(_stats_path(data_out, pf), {"stats": result["stats"], "samples": result["samples"]})
        run["files"].append(result["stats"])
        samples.extend(result["samples"])
        _write_outputs(out, data_out, run, samples, header, args.pins)
        return True

    ok = True
    if tasks:
        if args.workers <= 1:
            _init_worker(pairs, cfg.short_suite_min_words)
            for task in tasks:
                if not record(_build_task(task)):
                    ok = False
                    break
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")  # same behaviour on Windows (the PC) and Linux (tests)
            with ctx.Pool(
                processes=min(args.workers, len(tasks)),
                initializer=_init_worker,
                initargs=(pairs, cfg.short_suite_min_words),
                maxtasksperchild=1,
            ) as pool:
                for result in pool.imap_unordered(_build_task, tasks):
                    if not record(result):
                        ok = False
                        pool.terminate()
                        break
    if not ok:
        return 1

    (data_out / "ATTRIBUTION.txt").write_text(_attribution(header), encoding="utf-8", newline="\n")
    run["complete"] = args.max_docs is None and not args.only and len(run["files"]) == len(files)
    run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_outputs(out, data_out, run, samples, header, args.pins)
    tokens = sum(f["kept"]["tokens"] for f in run["files"])
    print(f"[build] done -> {out / 'summary.json'} (complete: {run['complete']}; {tokens:,} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
