#!/usr/bin/env python3
"""EXP-037 part 2: pack FrontierCorpus v2-slice1 into training token files (Frontier Tokenizer v2).

    python scripts/pack_sangraha_v2.py                                    # all 13 files
    python scripts/pack_sangraha_v2.py --only ur --max-docs 20000         # quick look (not complete)

Input: the 13 ``<lang>.jsonl.gz`` files built by EXP-036 (accepted as v2-slice1, D-045). The
committed manifest ``evals/results/EXP-036/manifest.json`` must have the sha256 recorded in D-045,
and every input file must match the sha256 listed in it; otherwise the script STOPs before
packing. For every file it writes ``<data-out>/<lang>.bin`` + ``<lang>.meta.json`` (uint16,
``train || val``, one ``<|endoftext|>`` after every document; ``frontier_ai.corpus.pack``), which
stay on this machine (``data/`` is git-ignored), and the small evidence under ``--out``:
``summary.json``, ``SUMMARY.txt`` and ``manifest.json`` (file names, counts, sha256; no text).

Resumable: a finished file is recorded in ``<data-out>/<lang>.pack.json``; running the same
command again reuses files whose output still matches (same code, tokenizer, input and split) and
packs only the rest. ``--workers 2`` (default) packs two files at a time (the laptop has two
cores). The script checks free disk space first.
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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_ai.corpus.pack import SCHEMA, VAL_PPM, OrdinaryEncoder, PackError, pack_file  # noqa: E402

D045_MANIFEST_SHA256 = "734874358cddf20df98fa812f291e104211cb2d60fb8fae163a0afcb144caf39"
DEFAULT_MANIFEST = "evals/results/EXP-036/manifest.json"
DEFAULT_PINS = "corpora/frontier/v2/sangraha_slice1.json"
DEFAULT_DATA_IN = "data/frontier_v2/sangraha-slice1-v2"
DEFAULT_DATA_OUT = "data/frontier_v2/sangraha-slice1-v2-tok2"
DEFAULT_OUT = "out/data/EXP-037"
PACKED_ID = "frontier-v2-sangraha-slice1@frontier-tokenizer-v2"
MANIFEST_SCHEMA = "frontier-packed-manifest-v1"
FREE_SPACE_FACTOR = 1.02  # output is exactly 2 bytes per token; parts are appended in place
FREE_SPACE_MARGIN = 1024**3

_WORKER: dict[str, Any] = {}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _module_sha() -> str:
    import frontier_ai.corpus.pack as pk

    return _sha256_bytes(Path(pk.__file__).read_bytes().replace(b"\r\n", b"\n"))[:16]


def _init_worker(tokenizer_root: str | None) -> None:
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    tok = load_frontier_tokenizer_v2(tokenizer_root)
    _WORKER["tok"] = tok
    _WORKER["enc"] = OrdinaryEncoder(tok)


def _pack_task(task: dict[str, Any]) -> dict[str, Any]:
    in_path = Path(task["in_path"])

    def say(msg: str) -> None:
        print(msg, flush=True)

    entry = task["entry"]
    say(f"[pack] {entry['language']}: verifying {in_path}")
    if not in_path.is_file():
        return {"error": f"{in_path} is missing (the EXP-036 corpus must be on this machine)", "entry": entry}
    if _sha256_file(in_path) != entry["sha256_gz"]:
        return {"error": f"{in_path} does not match its sha256 in the EXP-036 manifest", "entry": entry}
    _WORKER["enc"].cache.clear()
    try:
        stats = pack_file(
            in_path,
            Path(task["out_path"]),
            tokenizer=_WORKER["tok"],
            encoder=_WORKER["enc"],
            expected_docs=None if task["max_docs"] else entry["docs"],
            expected_tokens=None if task["max_docs"] else entry["tokens"],
            val_ppm=task["val_ppm"],
            max_docs=task["max_docs"],
            provenance=task["provenance"]
            | {"language": entry["language"], "input_sha256_gz": entry["sha256_gz"]},
            progress=say,
        )
    except PackError as exc:
        return {"error": f"{entry['language']}: CHECK FAILED: {exc}", "entry": entry}
    stats["language"] = entry["language"]
    stats["source_id"] = entry["source_id"]
    stats["input_sha256_verified"] = True
    stats["pack_key"] = task["key"]
    say(
        f"[pack] {entry['language']}: {stats['docs']:,} docs, {stats['tokens']:,} tokens "
        f"(val {stats['splits']['val']['docs']:,} docs), {stats['seconds'] / 60:.1f} min"
    )
    return {"stats": stats, "entry": entry}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def _reusable(data_out: Path, lang: str, key: str) -> dict[str, Any] | None:
    sp = data_out / f"{lang}.pack.json"
    if not sp.is_file():
        return None
    try:
        stats = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    out = data_out / stats.get("output", {}).get("path", "?")
    if stats.get("pack_key") != key or not out.is_file() or _sha256_file(out) != stats["output"]["sha256"]:
        return None
    return stats


def render_text_report(run: dict[str, Any]) -> str:
    lines = [
        f"{run['exp_id']}: FrontierCorpus v2-slice1 packed for training with {run['tokenizer']['name']}",
        f"complete: {run['complete']}   workers: {run['workers']}   started {run['started_at']}"
        f"   finished {run.get('finished_at', '-')}",
        f"input: {run['input']['corpus_id']} (manifest sha256 {run['input']['manifest_sha256'][:16]}..., "
        f"D-045 match: {run['input']['manifest_is_d045']})",
        f"split: validation = sha256(record id) in the lowest {run['val_ppm']} ppm "
        f"({run['val_ppm'] / 10_000:.2f}% of documents)",
        "",
        f"{'lang':<5}{'docs':>11}{'val docs':>10}{'train tokens':>16}{'val tokens':>13}{'EXP-036 tokens':>16}"
        f"{'check':>7}{'min':>7}",
    ]
    tot = {"docs": 0, "vdocs": 0, "train": 0, "val": 0, "src": 0}
    for f in run["files"]:
        s = f["splits"]
        ok = f["splits"]["train"]["tokens"] + s["val"]["tokens"] - f["docs"] == f["source_tokens_v1"]
        lines.append(
            f"{f['language']:<5}{f['docs']:>11,}{s['val']['docs']:>10,}{s['train']['tokens']:>16,}"
            f"{s['val']['tokens']:>13,}{f['source_tokens_v1']:>16,}{'OK' if ok else 'FAIL':>7}"
            f"{f['seconds'] / 60:>7.1f}"
        )
        tot["docs"] += f["docs"]
        tot["vdocs"] += s["val"]["docs"]
        tot["train"] += s["train"]["tokens"]
        tot["val"] += s["val"]["tokens"]
        tot["src"] += f["source_tokens_v1"]
    lines.append(
        f"{'all':<5}{tot['docs']:>11,}{tot['vdocs']:>10,}{tot['train']:>16,}{tot['val']:>13,}{tot['src']:>16,}"
    )
    lines += [
        "",
        "check = train tokens + val tokens - documents == EXP-036 exact count",
        "(one <|endoftext|> per document).",
        "Every document's count was also compared with its EXP-036 count, every file was re-read from disk",
        "(one <|endoftext|> per document, no other special id, both splits end with <|endoftext|>), and",
        "sampled documents were decoded back to their exact text. Token files stay on this machine.",
    ]
    return "\n".join(lines) + "\n"


def _manifest(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "packed_id": PACKED_ID,
        "exp_id": run["exp_id"],
        "complete": run["complete"],
        "source_corpus": run["input"],
        "tokenizer": run["tokenizer"],
        "val_ppm": run["val_ppm"],
        "format": {
            "dtype": "uint16",
            "byteorder": "little",
            "layout": "train || val",
            "document_end": "<|endoftext|> (id 32768) after every document",
            "reader": "frontier_ai.data.dataset.TokenDataset",
        },
        "files": [
            {
                "language": f["language"],
                "path": f["output"]["path"],
                "meta": f["output"]["meta"],
                "sha256": f["output"]["sha256"],
                "bytes": f["output"]["bytes"],
                "docs": f["docs"],
                "n_train": f["splits"]["train"]["tokens"],
                "n_val": f["splits"]["val"]["tokens"],
                "val_docs": f["splits"]["val"]["docs"],
                "val_ids_sha256": f["val_ids_sha256"],
            }
            for f in run["files"]
        ],
    }


def _write_outputs(out: Path, data_out: Path, run: dict[str, Any]) -> None:
    order = {lang: i for i, lang in enumerate(run["_order"])}
    run["files"].sort(key=lambda f: order[f["language"]])
    public = {k: v for k, v in run.items() if not k.startswith("_")}
    tot = {"docs": 0, "tokens": 0, "train_tokens": 0, "val_tokens": 0, "val_docs": 0, "source_tokens_v1": 0}
    for f in run["files"]:
        tot["docs"] += f["docs"]
        tot["tokens"] += f["tokens"]
        tot["train_tokens"] += f["splits"]["train"]["tokens"]
        tot["val_tokens"] += f["splits"]["val"]["tokens"]
        tot["val_docs"] += f["splits"]["val"]["docs"]
        tot["source_tokens_v1"] += f["source_tokens_v1"]
    public["totals"] = tot
    _write_json(out / "summary.json", public)
    (out / "SUMMARY.txt").write_text(render_text_report(public), encoding="utf-8", newline="\n")
    manifest = _manifest(public)
    _write_json(out / "manifest.json", manifest)
    _write_json(data_out / "manifest.json", manifest)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="EXP-037")
    p.add_argument("--pins", default=DEFAULT_PINS, help="the Sangraha pin file (its revision must match)")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="the EXP-036 corpus manifest")
    p.add_argument("--expect-manifest-sha256", default=D045_MANIFEST_SHA256, help=argparse.SUPPRESS)
    p.add_argument("--data-in", default=DEFAULT_DATA_IN, help="the v2-slice1 corpus (EXP-036 output)")
    p.add_argument("--data-out", default=DEFAULT_DATA_OUT, help="token files (stay on this machine)")
    p.add_argument("--out", default=DEFAULT_OUT, help="small reports (published)")
    p.add_argument("--tokenizer-root", default=None, help=argparse.SUPPRESS)
    p.add_argument("--only", default="", help="comma-separated language codes (default: all)")
    p.add_argument("--max-docs", type=int, default=None, help="at most N documents per file (not complete)")
    p.add_argument("--workers", type=int, default=2, help="files packed in parallel (default 2)")
    args = p.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[pack] STOP: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest_bytes = manifest_path.read_bytes().replace(b"\r\n", b"\n")
    manifest_sha = _sha256_bytes(manifest_bytes)
    if manifest_sha != args.expect_manifest_sha256:
        print(
            f"[pack] STOP: {manifest_path} has sha256 {manifest_sha[:16]}..., not the one accepted in D-045 "
            f"({args.expect_manifest_sha256[:16]}...). Only the accepted corpus is packed.",
            file=sys.stderr,
        )
        return 2
    manifest = json.loads(manifest_bytes)
    if not manifest.get("complete"):
        print("[pack] STOP: the EXP-036 manifest is not complete", file=sys.stderr)
        return 2
    pins_path = Path(args.pins)
    if pins_path.is_file():
        pins_rev = json.loads(pins_path.read_text(encoding="utf-8")).get("revision")
        if pins_rev and pins_rev != manifest["source"]["revision"]:
            print(f"[pack] STOP: pin revision {pins_rev} != manifest revision", file=sys.stderr)
            return 2
    entries = list(manifest["files"])
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        entries = [e for e in entries if e["language"] in wanted]
        if not entries:
            print(f"[pack] --only {args.only!r} matches no file in the manifest", file=sys.stderr)
            return 2
    data_in, data_out, out = Path(args.data_in), Path(args.data_out), Path(args.out)

    from frontier_ai.tokenization.frozen import frontier_tokenizer_v2_dir, load_frozen_tokenizer

    try:
        tok, freeze = load_frozen_tokenizer(frontier_tokenizer_v2_dir(args.tokenizer_root))
    except Exception as exc:  # noqa: BLE001 - any failure to verify stops the run
        print(f"[pack] STOP: frontier-tokenizer-v2 does not verify: {exc}", file=sys.stderr)
        return 2
    tok_fp = freeze["artifact"]["dir_sha256"][:16]
    tokenizer_info = {
        "name": "frontier-tokenizer-v2 (frozen, EXP-037)",
        "dir_sha256_16": tok_fp,
        "vocab_size": tok.vocab_size,
        "endoftext_id": tok.special_token_ids["<|endoftext|>"],
        "encoding": "encode_ordinary (special-token strings in text stay text)",
    }
    del tok

    def key_for(entry: dict[str, Any]) -> str:
        blob = json.dumps(
            {
                "schema": SCHEMA,
                "code": _module_sha(),
                "tokenizer": tok_fp,
                "input": entry["sha256_gz"],
                "val_ppm": VAL_PPM,
                "max_docs": args.max_docs,
            },
            sort_keys=True,
        )
        return _sha256_bytes(blob.encode("utf-8"))[:24]

    keys = {e["language"]: key_for(e) for e in entries}
    reused = {
        e["language"]: s for e in entries if (s := _reusable(data_out, e["language"], keys[e["language"]]))
    }
    todo = [e for e in entries if e["language"] not in reused]
    data_out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_out).free
    need = (
        int(sum(2 * (e["tokens"] + e["docs"]) for e in todo) * FREE_SPACE_FACTOR) + FREE_SPACE_MARGIN
        if todo
        else 0
    )
    if args.max_docs is None and free < need:
        print(
            f"[pack] STOP: not enough free disk space in {data_out.resolve()}: {free / 1e9:.1f} GB free, "
            f"need about {need / 1e9:.1f} GB. Free some space (e.g. empty the Recycle Bin) and run again.",
            file=sys.stderr,
        )
        return 2
    print(f"[pack] disk: {free / 1e9:.1f} GB free, about {need / 1e9:.1f} GB needed", flush=True)
    for lang in reused:
        print(f"[pack] {lang}: already packed with the same code, tokenizer, input and split; reused")

    provenance = {
        "packed_id": PACKED_ID,
        "exp_id": args.exp_id,
        "source_corpus": manifest["corpus_id"],
        "source_manifest_sha256": manifest_sha,
        "tokenizer": "frontier-tokenizer-v2",
        "tokenizer_dir_sha256_16": tok_fp,
        "endoftext_id": tokenizer_info["endoftext_id"],
        "val_rule": f"sha256(record id) first 8 bytes mod 1e6 < {VAL_PPM}",
        "license_id": manifest["source"]["license_id"],
        "attribution": manifest["source"]["attribution"],
    }
    run: dict[str, Any] = {
        "schema": SCHEMA,
        "exp_id": args.exp_id,
        "input": {
            "corpus_id": manifest["corpus_id"],
            "manifest": manifest_path.as_posix(),
            "manifest_sha256": manifest_sha,
            "manifest_is_d045": manifest_sha == D045_MANIFEST_SHA256,
            "revision": manifest["source"]["revision"],
            "license_id": manifest["source"]["license_id"],
            "attribution": manifest["source"]["attribution"],
        },
        "tokenizer": tokenizer_info,
        "val_ppm": VAL_PPM,
        "code_sha256_16": _module_sha(),
        "max_docs_per_file": args.max_docs,
        "workers": args.workers,
        "data_out": str(data_out),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": list(reused.values()),
        "reused_files": sorted(reused),
        "complete": False,
        "_order": [e["language"] for e in entries],
    }
    _write_outputs(out, data_out, run)

    tasks = [
        {
            "entry": e,
            "in_path": str(data_in / e["path"]),
            "out_path": str(data_out / f"{e['language']}.bin"),
            "val_ppm": VAL_PPM,
            "max_docs": args.max_docs,
            "provenance": provenance,
            "key": keys[e["language"]],
        }
        for e in sorted(todo, key=lambda e: -e["tokens"])  # largest first: better balance over 2 workers
    ]

    def record(result: dict[str, Any]) -> bool:
        if "error" in result:
            print(f"[pack] STOP: {result['error']}", file=sys.stderr)
            return False
        stats = result["stats"]
        _write_json(data_out / f"{stats['language']}.pack.json", stats)
        run["files"].append(stats)
        _write_outputs(out, data_out, run)
        return True

    ok = True
    if tasks:
        if args.workers <= 1:
            _init_worker(args.tokenizer_root)
            for task in tasks:
                if not record(_pack_task(task)):
                    ok = False
                    break
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")  # same behaviour on Windows (the PC) and Linux (tests)
            with ctx.Pool(
                processes=min(args.workers, len(tasks)),
                initializer=_init_worker,
                initargs=(args.tokenizer_root,),
                maxtasksperchild=1,
            ) as pool:
                for result in pool.imap_unordered(_pack_task, tasks):
                    if not record(result):
                        ok = False
                        pool.terminate()
                        break
    if not ok:
        return 1
    run["complete"] = (
        args.max_docs is None
        and not args.only
        and len(run["files"]) == len(manifest["files"])
        and run["input"]["manifest_is_d045"]
    )
    run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_outputs(out, data_out, run)
    tokens = sum(f["tokens"] for f in run["files"])
    print(
        f"[pack] done -> {out / 'summary.json'} (complete: {run['complete']}; {tokens:,} tokens incl. "
        "one <|endoftext|> per document)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
