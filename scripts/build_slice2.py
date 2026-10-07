#!/usr/bin/env python3
"""EXP-046: build FrontierCorpus v2-slice2 (part A or B) on a Kaggle CPU session, then pack it.

    python scripts/build_slice2.py --part A --tokens-dir <EXP-037 token files> \\
        --heldout-jsonl <heldout-v1.jsonl> --out <reports> --data-out <corpus> --work /tmp/slice2

The approved plan (EXPERIMENTS.md EXP-046): files ``data-1`` and ``data-2`` of each language
(``corpora/frontier/v2/sangraha_slice2.json``), the unchanged v2 build rules (EXP-036, D-045's
configuration), plus four safeguards:

1. the same rules: ``frontier_ai.corpus.slice_build.build_file`` with the default ``BuildConfig``;
2. no document from v2-slice1 twice: every slice-1 document's exact text is recovered from the
   EXP-037 token files (``frontier_ai.corpus.prior_docs``); a slice-2 document with the same final
   text is removed (``slice1_duplicate``). The two files of a language also share rule 2a, so a
   text from ``data-1`` is not kept again from ``data-2`` (``cross_file_duplicate``);
3. Belebele protected: its passages, questions and options (all 13 languages, pinned files,
   NFC-normalized) join the protected suite's 13-word-run and exact-text guard. They are NOT added
   to the short-passage index (3-12 words), which would remove every document containing a short
   common option such as a name;
4. read before accepting: ``samples.jsonl`` holds masked excerpts of kept and removed documents per
   rule, as in EXP-036. Acceptance is a separate decision.

Part A = en ur as bn gu hi, part B = kn ml mr or pa ta te (EXP-046 probe, 2026-10-06). Each language
is one task: download its two pinned parquet files (verified by size and SHA-256), build them in
order, join the two outputs into ``<data-out>/<lang>.jsonl.gz`` and pack that with Frontier
Tokenizer v2 into ``<lang>.bin`` + ``<lang>.meta.json`` (``frontier_ai.corpus.pack``, the EXP-037
format: uint16, ``train || val``, one ``<|endoftext|>`` per document, the same validation rule).
Parquet files are deleted after use; Sangraha text goes only into the (private) data output.

Reports (small, published): ``<out>/summary.json``, ``SUMMARY.txt``, ``manifest.json`` (EXP-037's
packed-manifest format plus the text files) and ``samples.jsonl``. They are rewritten after every
finished language. ``--deadline-hours`` stops cleanly before Kaggle's 12-hour limit: finished
languages are kept and reported, unfinished ones are removed and listed.
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
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_ai.corpus.decontaminate import ShortSuiteIndex, SuiteGuard, heldout_pairs  # noqa: E402
from frontier_ai.corpus.normalize import normalize_text  # noqa: E402
from frontier_ai.corpus.sangraha import (  # noqa: E402
    PinnedFile,
    SliceError,
    download_verified,
    iter_rows,
    load_slice_pins,
    parquet_row_count,
    sha256_file,
)
from frontier_ai.corpus.slice_build import (  # noqa: E402
    REASONS,
    BuildConfig,
    TokenCounter,
    build_file,
    render_text_report,
)
from frontier_ai.evaluation import belebele as bb  # noqa: E402
from frontier_ai.evaluation.suite import SuiteError, load_suite  # noqa: E402

SCHEMA = "frontier-slice2-build-v1"
PARTS = {"A": ("en", "ur", "as", "bn", "gu", "hi"), "B": ("kn", "ml", "mr", "or", "pa", "ta", "te")}
DEFAULT_PINS = "corpora/frontier/v2/sangraha_slice2.json"
DEFAULT_SUITE = "evals/suites/frontier-heldout-v1/SUITE.json"
SLICE1_MANIFEST = "evals/results/EXP-037/manifest.json"
CORPUS_ID = "frontier-v2-sangraha-slice2"
MANIFEST_SCHEMA = "frontier-packed-manifest-v1"
# probe estimate: text + tokens ~1.68 x the parquet bytes; one language's two parts on top
FREE_SPACE_FACTOR = 1.9
FREE_SPACE_MARGIN = 1024**3

_WORKER: dict[str, Any] = {}


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _code_shas() -> dict[str, str]:
    import frontier_ai.corpus.pack as pk
    import frontier_ai.corpus.prior_docs as pd
    import frontier_ai.corpus.slice_build as sb

    return {
        "slice_build": _sha16(Path(sb.__file__)),
        "pack": _sha16(Path(pk.__file__)),
        "prior_docs": _sha16(Path(pd.__file__)),
        "build_slice2": _sha16(Path(__file__)),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


# ----------------------------------------------------------------------------- protected texts --
def heldout_from_jsonl(path: Path, suite: dict[str, Any], work: Path) -> list[tuple[str, str]]:
    """The protected suite's texts from ``export_heldout_text.py``'s file, verified like the shards."""
    texts = [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any("\n" in t or "\r" in t for t in texts):
        raise ValueError(f"{path}: a held-out text contains a line break (the shards hold one per line)")
    shard = work / "heldout" / "heldout-000000.txt"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text("\n".join(texts) + "\n", encoding="utf-8", newline="\n")
    return heldout_pairs([shard], suite)  # raises unless the texts are exactly the suite's documents


def belebele_pairs(directory: Path, verify: bool) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """``("belebele/<lang>/<part>/<key>", NFC text)`` for every passage, question and option."""
    pairs: list[tuple[str, str]] = []
    per_lang: dict[str, int] = {}
    for lang in bb.LANGUAGES:
        if verify:
            bb.download(lang, directory)
        items = bb.load_language(
            directory, lang, verify=verify, expected=bb.QUESTIONS_PER_LANGUAGE if verify else None
        )
        before = len(pairs)
        for key, text in bb.passages(items).items():
            pairs.append((f"belebele/{lang}/passage/{key}", normalize_text(text, "nfc")))
        for it in items:
            pairs.append((f"belebele/{lang}/question/{it.key}", normalize_text(it.question, "nfc")))
            for j, option in enumerate(it.options, 1):
                pairs.append((f"belebele/{lang}/option{j}/{it.key}", normalize_text(option, "nfc")))
        per_lang[lang] = len(pairs) - before
    info = {
        "dataset": bb.REPO,
        "revision": bb.REVISION,
        "files_verified": verify,
        "texts": len(pairs),
        "texts_per_language": per_lang,
        "guarded_by": "13-word runs and exact text (not the 3-12-word short-passage index)",
    }
    return pairs, info


# ---------------------------------------------------------------------------------- workers --
def _init_worker(
    guard_pairs: list[tuple[str, str]], short_pairs: list[tuple[str, str]], short_min: int
) -> None:
    from frontier_ai.corpus.pack import OrdinaryEncoder
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer, load_frontier_tokenizer_v2

    _WORKER["guard"] = SuiteGuard.from_texts(guard_pairs)
    _WORKER["short"] = ShortSuiteIndex.from_texts(short_pairs, min_words=short_min)
    _WORKER["count"] = TokenCounter(load_frontier_tokenizer())
    tok2 = load_frontier_tokenizer_v2()
    _WORKER["tok2"] = tok2
    _WORKER["enc2"] = OrdinaryEncoder(tok2)


def _language_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        return _language(task)
    except Exception:  # noqa: BLE001 - reported to the main process, which stops the run
        return {"language": task["language"], "error": traceback.format_exc()}


def _language(task: dict[str, Any]) -> dict[str, Any]:
    from frontier_ai.corpus.pack import VAL_PPM, pack_file
    from frontier_ai.corpus.prior_docs import document_digests

    lang: str = task["language"]
    pfs: list[PinnedFile] = task["files"]
    data_out, work = Path(task["data_out"]), Path(task["work"])
    started = time.monotonic()

    def say(msg: str) -> None:
        print(msg, flush=True)

    # safeguard 2: the exact texts of every slice-1 document of this language
    s1 = task["slice1"]
    s1_path = Path(task["tokens_dir"]) / s1["path"]
    if sha256_file(s1_path) != s1["sha256"]:
        raise SliceError(f"{s1_path} does not match the EXP-037 manifest sha256")
    tok2 = _WORKER["tok2"]
    digests = document_digests(s1_path, tok2, tok2.special_token_ids["<|endoftext|>"])
    if len(digests) != s1["docs"]:
        raise SliceError(f"{s1_path}: {len(digests)} documents, the EXP-037 manifest says {s1['docs']}")
    slice1 = set(digests)
    say(f"[slice2] {lang}: {len(digests):,} slice-1 documents read ({len(slice1):,} distinct texts)")

    shared: set[bytes] = set()
    built: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    parts: list[Path] = []
    for pf in pfs:
        what = download_verified(pf, work / "parquet", log=say)
        path = pf.local_path(work / "parquet")
        say(f"[slice2] {pf.source_id}: {what}; building")
        part = data_out / "parts" / f"{pf.source_id}.jsonl.gz"
        _WORKER["count"].cache.clear()
        stats, smp = build_file(
            lambda path=path: iter_rows(path),
            out_path=part,
            exp_id=task["exp_id"],
            source_id=pf.source_id,
            language=pf.language,
            script=pf.script,
            revision=task["revision"],
            guard=_WORKER["guard"],
            short_index=_WORKER["short"],
            token_counter=_WORKER["count"],
            config=task["config"],
            expected_rows=parquet_row_count(path),
            max_docs=task["max_docs"],
            progress=say,
            shared_exact=shared,
            exclude_final=slice1,
        )
        stats["pinned_sha256_verified"] = True
        built.append(stats)
        samples.extend(smp)
        parts.append(part)
        if not task["keep_parquet"]:
            path.unlink()
        say(
            f"[slice2] {pf.source_id}: kept {stats['kept']['docs']:,} of {stats['input']['docs']:,} docs, "
            f"{stats['kept']['tokens']:,} tokens ({stats['seconds'] / 60:.1f} min)"
        )
    # join the two outputs (gzip members concatenate into one valid gzip file)
    text_path = data_out / f"{lang}.jsonl.gz"
    tmp = text_path.with_name(text_path.name + ".tmp")
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 20)
    tmp.replace(text_path)
    for part in parts:
        part.unlink()
    docs = sum(s["kept"]["docs"] for s in built)
    tokens = sum(s["kept"]["tokens"] for s in built)
    text_info = {
        "language": lang,
        "path": text_path.name,
        "sources": [pf.source_id for pf in pfs],
        "docs": docs,
        "tokens": tokens,
        "bytes": text_path.stat().st_size,
        "sha256": sha256_file(text_path),
    }
    _WORKER["enc2"].cache.clear()
    packed = pack_file(
        text_path,
        data_out / f"{lang}.bin",
        tokenizer=tok2,
        encoder=_WORKER["enc2"],
        expected_docs=docs,
        expected_tokens=tokens,
        val_ppm=VAL_PPM,
        provenance=task["provenance"] | {"language": lang, "input_sha256_gz": text_info["sha256"]},
        progress=say,
    )
    packed["language"] = lang
    packed["sources"] = text_info["sources"]
    say(
        f"[slice2] {lang}: packed {packed['docs']:,} docs, {packed['tokens']:,} tokens "
        f"({(time.monotonic() - started) / 60:.1f} min for the language)"
    )
    return {
        "language": lang,
        "files": built,
        "samples": samples,
        "text": text_info,
        "packed": packed,
        "slice1": {"docs": len(digests), "distinct_texts": len(slice1), "sha256_verified": True},
        "seconds": round(time.monotonic() - started, 1),
    }


# ---------------------------------------------------------------------------------- reports --
def render(run: dict[str, Any]) -> str:
    lines = [
        f"{run['exp_id']} build {run['part']} - FrontierCorpus v2-slice2 ({', '.join(run['languages'])})",
        f"code commit {run['environment'].get('code_commit')}   seconds {run.get('seconds')}   "
        f"stopped: {run.get('stopped') or 'no'}",
        "",
    ]
    lines.append(render_text_report(run).rstrip("\n"))
    lines += [
        "",
        "Safeguards (EXP-046):",
        "lang | slice-1 docs read | slice1_duplicate | cross_file_duplicate | touching Belebele",
    ]
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for f in run["files"]:
        by_lang.setdefault(f["language"], []).append(f)
    for lang, fs in by_lang.items():
        s1 = run["slice1"]["per_language"].get(lang, {}).get("docs", 0)
        dup = sum(f["removed_docs"]["slice1_duplicate"]["docs"] for f in fs)
        cross = sum(f["removed_docs"]["cross_file_duplicate"]["docs"] for f in fs)
        bele = sum(f["suite"].get("belebele_touching_docs", 0) for f in fs)
        lines.append(f"{lang:>4} | {s1:>17,} | {dup:>16,} | {cross:>20,} | {bele:>17,}")
    lines += [
        "",
        "Packed with frontier-tokenizer-v2 (train || val, one <|endoftext|> per document):",
        "lang | docs | val docs | train tokens | val tokens | check | text GB | tokens GB",
    ]
    for p in run["packed"]:
        s = p["splits"]
        ok = s["train"]["tokens"] + s["val"]["tokens"] - p["docs"] == p["source_tokens_v1"]
        text = next(t for t in run["text_files"] if t["language"] == p["language"])
        lines.append(
            f"{p['language']:>4} | {p['docs']:,} | {s['val']['docs']:,} | {s['train']['tokens']:,} | "
            f"{s['val']['tokens']:,} | {'OK' if ok else 'FAIL'} | {text['bytes'] / 1e9:.2f} | "
            f"{p['output']['bytes'] / 1e9:.2f}"
        )
    t = run["totals"]
    lines += [
        f" all | {t['docs']:,} | {t['val_docs']:,} | {t['train_tokens']:,} | {t['val_tokens']:,}",
        "",
        f"languages done: {', '.join(run['done']) or '-'}; not done: {', '.join(run['not_done']) or '-'}",
        f"data output: {run['data_out']} ({run['data_bytes'] / 1e9:.2f} GB, stays on Kaggle)",
        "Samples of kept and removed documents: samples.jsonl (read before accepting; acceptance is a",
        "separate decision, as D-045 was for slice 1).",
    ]
    if run.get("max_docs_per_file"):
        lines.append(f"SMOKE RUN: at most {run['max_docs_per_file']} documents per file; not a result.")
    return "\n".join(lines) + "\n"


def manifest(run: dict[str, Any]) -> dict[str, Any]:
    texts = {t["language"]: t for t in run["text_files"]}
    files = []
    for p in run["packed"]:
        files.append(
            {
                "language": p["language"],
                "path": p["output"]["path"],
                "meta": p["output"]["meta"],
                "sha256": p["output"]["sha256"],
                "bytes": p["output"]["bytes"],
                "docs": p["docs"],
                "n_train": p["splits"]["train"]["tokens"],
                "n_val": p["splits"]["val"]["tokens"],
                "val_docs": p["splits"]["val"]["docs"],
                "val_ids_sha256": p["val_ids_sha256"],
                "sources": p["sources"],
                "text": {k: texts[p["language"]][k] for k in ("path", "sha256", "bytes", "docs", "tokens")},
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "packed_id": f"{CORPUS_ID}-{run['part']}@frontier-tokenizer-v2",
        "corpus_id": CORPUS_ID,
        "part": run["part"],
        "exp_id": run["exp_id"],
        "complete": run["complete"],
        "source": {
            "dataset": run["dataset"],
            "revision": run["revision"],
            "slice_id": run["slice_id"],
            "pins": run["pins"],
            "license_id": run["license_id"],
            "attribution": run["attribution"],
        },
        "rules": "EXPERIMENTS.md EXP-036 (D-045 configuration) + EXP-046 safeguards; "
        "code: src/frontier_ai/corpus/slice_build.py, scripts/build_slice2.py",
        "config_fingerprint": run["config_fingerprint"],
        "tokenizer": run["tokenizer"],
        "val_ppm": run["val_ppm"],
        "suite": run["suite_status"],
        "format": {
            "dtype": "uint16",
            "byteorder": "little",
            "layout": "train || val",
            "document_end": "<|endoftext|> (id 32768) after every document",
            "reader": "frontier_ai.data.dataset.TokenDataset",
        },
        "files": files,
        "totals": run["totals"],
    }


def attribution(run: dict[str, Any]) -> str:
    return (
        f"{CORPUS_ID} (part {run['part']}) is derived from {run['dataset']} (verified), revision "
        f"{run['revision']}.\n\nSource attribution: {run['attribution']}\nLicense of the source: "
        f"{run['license_id']}.\n\nChanges made: as FrontierCorpus v2-slice1 (EXPERIMENTS.md EXP-036): NFC\n"
        "normalization and whitespace cleanup; documents removed for protected-evaluation overlap\n"
        "(including Belebele), duplication (also against v2-slice1), wrong language/script, wiki markup,\n"
        "repetition and control characters; repeated boilerplate lines and lines without a letter of the\n"
        "declared script removed; e-mail addresses and phone numbers replaced by [email]/[phone].\n"
        "Rules: EXPERIMENTS.md EXP-036 and EXP-046 in the frontier-ai repository.\n"
    )


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_outputs(out: Path, data_out: Path, run: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    order = {lang: i for i, lang in enumerate(run["languages"])}
    run["files"].sort(key=lambda f: (order[f["language"]], f["source_id"]))
    run["packed"].sort(key=lambda p: order[p["language"]])
    run["text_files"].sort(key=lambda t: order[t["language"]])
    samples.sort(key=lambda s: (order[s["language"]], s["source_id"], s["stratum"], s["row"]))
    run["done"] = [lang for lang in run["languages"] if lang in {p["language"] for p in run["packed"]}]
    run["not_done"] = [lang for lang in run["languages"] if lang not in run["done"]]
    run["totals"] = {
        "docs": sum(p["docs"] for p in run["packed"]),
        "tokens": sum(p["tokens"] for p in run["packed"]),
        "train_tokens": sum(p["splits"]["train"]["tokens"] for p in run["packed"]),
        "val_tokens": sum(p["splits"]["val"]["tokens"] for p in run["packed"]),
        "val_docs": sum(p["splits"]["val"]["docs"] for p in run["packed"]),
        "text_bytes": sum(t["bytes"] for t in run["text_files"]),
    }
    run["data_bytes"] = sum(f.stat().st_size for f in data_out.rglob("*") if f.is_file())
    run["samples_written"] = len(samples)
    _write_json(out / "summary.json", run)
    with open(out / "samples.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for item in samples:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "SUMMARY.txt").write_text(render(run), encoding="utf-8", newline="\n")
    man = manifest(run)
    _write_json(out / "manifest.json", man)
    _write_json(data_out / "manifest.json", man)


# ------------------------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", required=True, choices=sorted(PARTS))
    p.add_argument("--exp-id", default="EXP-046")
    p.add_argument("--pins", default=DEFAULT_PINS)
    p.add_argument("--suite", default=DEFAULT_SUITE)
    p.add_argument("--heldout-jsonl", required=True, help="scripts/export_heldout_text.py output")
    p.add_argument("--tokens-dir", required=True, help="the EXP-037 token files (v2-slice1)")
    p.add_argument("--slice1-manifest", default=SLICE1_MANIFEST)
    p.add_argument("--belebele-dir", required=True)
    p.add_argument("--no-verify", action="store_true", help="tests only: do not download/verify Belebele")
    p.add_argument("--work", required=True, help="scratch: parquet downloads (deleted after use)")
    p.add_argument("--data-out", required=True, help="the corpus: text + token files")
    p.add_argument("--out", required=True, help="small reports (published)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--languages", default="", help="tests only: a subset of the part's languages")
    p.add_argument("--max-docs", type=int, default=None, help="smoke runs: at most N documents per file")
    p.add_argument("--keep-parquet", action="store_true", help="tests only")
    p.add_argument("--deadline-hours", type=float, default=11.0)
    args = p.parse_args(argv)
    t0 = time.monotonic()

    langs = list(PARTS[args.part])
    if args.languages:
        wanted = [x.strip() for x in args.languages.split(",") if x.strip()]
        if not set(wanted) <= set(langs):
            print(f"[slice2] --languages {wanted} are not all in part {args.part}", file=sys.stderr)
            return 2
        langs = [lang for lang in langs if lang in wanted]
    header, pins = load_slice_pins(args.pins)
    files = {lang: [pf for pf in pins if pf.language == lang] for lang in langs}
    for lang, fs in files.items():
        if len(fs) != 2:
            print(
                f"[slice2] STOP: {lang} has {len(fs)} pinned files, expected data-1 and data-2",
                file=sys.stderr,
            )
            return 2
    out, data_out, work = Path(args.out), Path(args.data_out), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cfg = BuildConfig()

    # protected suite (mandatory) + Belebele
    try:
        suite = load_suite(args.suite)
        held = heldout_from_jsonl(Path(args.heldout_jsonl), suite, work)
    except (SuiteError, ValueError, OSError, KeyError) as exc:
        print(f"[slice2] STOP: the protected suite cannot be checked: {exc}", file=sys.stderr)
        return 2
    bele, bele_info = belebele_pairs(Path(args.belebele_dir), verify=not args.no_verify)
    guard = SuiteGuard.from_texts(held + bele)
    short = ShortSuiteIndex.from_texts(held, min_words=cfg.short_suite_min_words)
    suite_status = (
        f"CHECKED against {suite['suite_id']} ({len(held)} documents; 13-grams + exact + passages of "
        f"{cfg.short_suite_min_words}-12 words; verified against SUITE.json) and Belebele "
        f"({bele_info['texts']:,} texts; 13-grams + exact)"
    )
    print(f"[slice2] suite: {suite_status}", flush=True)

    # slice 1 (safeguard 2) and the tokenizers
    s1_bytes = Path(args.slice1_manifest).read_bytes().replace(b"\r\n", b"\n")
    s1_manifest = json.loads(s1_bytes)
    s1_files = {f["language"]: f for f in s1_manifest["files"]}
    from frontier_ai.corpus.pack import VAL_PPM
    from frontier_ai.tokenization.frozen import (
        FRONTIER_TOKENIZER_V1,
        FROZEN_ROOT,
        frontier_tokenizer_v2_dir,
        load_frozen_tokenizer,
    )

    _t1, freeze1 = load_frozen_tokenizer(FROZEN_ROOT / FRONTIER_TOKENIZER_V1)
    tok2, freeze2 = load_frozen_tokenizer(frontier_tokenizer_v2_dir(None))
    tokenizer = {
        "name": "frontier-tokenizer-v1 counts (D-041), frontier-tokenizer-v2 packs (EXP-037)",
        "v1_dir_sha256_16": freeze1["artifact"]["dir_sha256"][:16],
        "v2_dir_sha256_16": freeze2["artifact"]["dir_sha256"][:16],
        "vocab_size": tok2.vocab_size,
        "endoftext_id": tok2.special_token_ids["<|endoftext|>"],
        "exact_count": True,
    }
    del _t1, tok2

    # disk
    data_out.mkdir(parents=True, exist_ok=True)
    need = int(sum(pf.size for fs in files.values() for pf in fs) * FREE_SPACE_FACTOR) + FREE_SPACE_MARGIN
    free = shutil.disk_usage(data_out).free
    disk = {"data_out_free": free, "data_out_need_estimate": need, "work_free": shutil.disk_usage(work).free}
    if args.max_docs is None and free < need:
        print(
            f"[slice2] STOP: {free / 1e9:.1f} GB free in {data_out}, need about {need / 1e9:.1f} GB",
            file=sys.stderr,
        )
        return 2

    run: dict[str, Any] = {
        "schema": SCHEMA,
        "exp_id": args.exp_id,
        "part": args.part,
        "languages": langs,
        "slice_id": header["slice_id"],
        "dataset": header["dataset"],
        "revision": header["revision"],
        "license_id": header["license_id"],
        "attribution": header["attribution"],
        "pins": args.pins,
        "config": asdict(cfg),
        "config_fingerprint": cfg.fingerprint(),
        "code_sha256_16": _code_shas(),
        "tokenizer": tokenizer,
        "val_ppm": VAL_PPM,
        "suite": {
            "suite_id": suite["suite_id"],
            "suite_sha256_16": _sha16(Path(args.suite)),
            **guard.describe(),
            "short_index": short.describe(),
            "belebele": bele_info,
        },
        "suite_status": suite_status,
        "slice1": {
            "manifest": args.slice1_manifest,
            "manifest_sha256": hashlib.sha256(s1_bytes).hexdigest(),
            "packed_id": s1_manifest.get("packed_id"),
            "rule": "a slice-2 document whose final text equals a slice-1 document's text is removed",
            "per_language": {},
        },
        "reasons": list(REASONS),
        "max_docs_per_file": args.max_docs,
        "workers": args.workers,
        "data_out": str(data_out),
        "disk": disk,
        "environment": {
            "code_commit": _git_commit(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [],
        "packed": [],
        "text_files": [],
        "complete": False,
        "stopped": None,
    }
    samples: list[dict[str, Any]] = []
    write_outputs(out, data_out, run, samples)

    provenance = {
        "packed_id": f"{CORPUS_ID}-{args.part}@frontier-tokenizer-v2",
        "exp_id": args.exp_id,
        "source_corpus": CORPUS_ID,
        "tokenizer": "frontier-tokenizer-v2",
        "tokenizer_dir_sha256_16": tokenizer["v2_dir_sha256_16"],
        "endoftext_id": tokenizer["endoftext_id"],
        "val_rule": f"sha256(record id) first 8 bytes mod 1e6 < {VAL_PPM}",
        "license_id": header["license_id"],
        "attribution": header["attribution"],
    }
    tasks = [
        {
            "language": lang,
            "files": files[lang],
            "slice1": s1_files[lang],
            "tokens_dir": args.tokens_dir,
            "data_out": str(data_out),
            "work": str(work),
            "exp_id": args.exp_id,
            "revision": header["revision"],
            "config": cfg,
            "max_docs": args.max_docs,
            "keep_parquet": args.keep_parquet,
            "provenance": provenance,
        }
        for lang in sorted(langs, key=lambda x: -sum(pf.size for pf in files[x]))  # largest first
    ]

    def record(res: dict[str, Any]) -> bool:
        if "error" in res:
            print(f"[slice2] STOP: {res['language']} failed:\n{res['error']}", file=sys.stderr)
            run["stopped"] = f"{res['language']} failed: {res['error'].strip().splitlines()[-1]}"
            return False
        run["files"].extend(res["files"])
        run["packed"].append(res["packed"])
        run["text_files"].append(res["text"])
        run["slice1"]["per_language"][res["language"]] = res["slice1"]
        samples.extend(res["samples"])
        run["seconds"] = round(time.monotonic() - t0, 1)
        write_outputs(out, data_out, run, samples)
        return True

    deadline = t0 + args.deadline_hours * 3600
    init = (held + bele, held, cfg.short_suite_min_words)
    ok = True
    if args.workers <= 1:
        _init_worker(*init)
        for task in tasks:
            if time.monotonic() > deadline:
                run["stopped"] = f"deadline of {args.deadline_hours} h reached"
                break
            if not record(_language_task(task)):
                ok = False
                break
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            min(args.workers, len(tasks)), initializer=_init_worker, initargs=init, maxtasksperchild=1
        ) as pool:
            pending = {pool.apply_async(_language_task, (t,)): t["language"] for t in tasks}
            while pending:
                for ar in [a for a in pending if a.ready()]:
                    del pending[ar]
                    if not record(ar.get()):
                        ok = False
                if not ok:
                    break
                if pending and time.monotonic() > deadline:
                    run["stopped"] = f"deadline of {args.deadline_hours} h reached"
                    break
                time.sleep(2)
            pool.terminate()
    # unfinished languages leave no files behind
    done = {p["language"] for p in run["packed"]}
    for lang in langs:
        if lang not in done:
            for f in [
                data_out / f"{lang}.jsonl.gz",
                data_out / f"{lang}.bin",
                data_out / f"{lang}.meta.json",
            ]:
                f.unlink(missing_ok=True)
            for f in data_out.glob(f"{lang}.*.tmp"):
                f.unlink()
            for f in data_out.glob(f"{lang}.bin.*.part"):
                f.unlink()
            for pf in files[lang]:
                for f in (data_out / "parts").glob(f"{pf.source_id}.*"):
                    f.unlink()
    shutil.rmtree(data_out / "parts", ignore_errors=True)
    (data_out / "ATTRIBUTION.txt").write_text(attribution(run), encoding="utf-8", newline="\n")
    run["complete"] = (
        ok and not run["stopped"] and args.max_docs is None and len(done) == len(PARTS[args.part])
    )
    run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run["seconds"] = round(time.monotonic() - t0, 1)
    write_outputs(out, data_out, run, samples)
    print(
        f"[slice2] done -> {out / 'summary.json'} (complete: {run['complete']}; "
        f"{run['totals']['tokens']:,} tokens; stopped: {run['stopped']})",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
