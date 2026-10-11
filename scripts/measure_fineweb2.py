#!/usr/bin/env python3
"""EXP-048: measure FineWeb-2 for our 12 Indian languages before deciding to use it (Kaggle CPU).

    python scripts/measure_fineweb2.py --input-dir /kaggle/input --heldout-jsonl <heldout-v1.jsonl> \\
        --belebele-dir /tmp/belebele --work /tmp/fw2 --out /kaggle/working/EXP-048

Approved 2026-10-09 ("approve A B C", item C), as planned in EXPERIMENTS.md EXP-048. The question is
not "is FineWeb-2 big" but "how many NEW, CLEAN documents would it add that we do not already have?"
Nothing here adds data to FrontierCorpus.

Per language (FineWeb-2 has no English):

1. **Pin:** the dataset revision (commit), its licence as stated on that revision, the subset names
   read from the listing (romanized and other-script subsets are listed, not measured), and every
   train file (path, size, SHA-256 from the Hugging Face API).
2. **Size:** train rows per subset two ways: the Hugging Face dataset-viewer ``/size`` service and
   the parquet footers of every train file (read with HTTP Range requests; no full download).
3. **Sample:** one train file per language, chosen by a fixed hash of the subset name, downloaded with
   the repository's SHA-256-checked downloader into ``--work`` and deleted after use. ``--sample``
   rows (default 2,000) are chosen by a fixed integer hash (seed 48), the same on every machine.
4. **Rules, overlap and protected suites in one pass:** the sample goes through the UNCHANGED v2 build
   (``slice_build.build_file`` with the D-045 ``BuildConfig``), with the protected suite + Belebele
   guard, and with ``exclude_final`` = the exact-text digests of the language's v2-slice1 AND
   v2-slice2 documents (``prior_docs``, from the mounted token files, each checked against its pinned
   manifest first). A sample document "already in our corpus" is therefore one that would survive
   every rule and is identical to a slice-1 or slice-2 document (reason ``slice1_duplicate``, which
   here means "in slice 1 or 2"). Exact text only: a LOWER bound on overlap.
5. **Tokens:** Frontier Tokenizer v2 tokens of every sampled and every kept document.
6. **Estimate:** new clean v2 tokens = train rows x (kept / sampled) x mean v2 tokens per kept
   document, with a 95% interval from the sampling error. NOT VERIFIED (an estimate from a sample).

Known limits (also printed in the report): the sample is 2,000 documents from ONE file, so the
in-file boilerplate and near-duplicate rules see far less repetition than in a full build (the keep
rate is likely an upper bound for a real build); exact-text overlap misses near-duplicates; URL
overlap is not measured (whether Sangraha Verified keeps source URLs is NOT VERIFIED).

Writes ``summary.json``, ``SUMMARY.txt`` and ``samples.jsonl`` (short excerpts with source URLs, for
reading before any decision; ODC-By attribution included). No FineWeb-2 corpus is written to ``--out``.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_slice2 as b2  # noqa: E402
from frontier_ai.corpus.decontaminate import ShortSuiteIndex, SuiteGuard  # noqa: E402
from frontier_ai.corpus.prior_docs import document_digests  # noqa: E402
from frontier_ai.corpus.sangraha import PinnedFile, download_verified, load_slice_pins  # noqa: E402
from frontier_ai.corpus.slice_build import BuildConfig, TokenCounter, build_file, mask_contacts  # noqa: E402
from frontier_ai.data import slices as sl  # noqa: E402
from frontier_ai.evaluation.suite import load_suite  # noqa: E402

SCHEMA = "frontier-exp048-fineweb2-measure-v1"
REPO = "HuggingFaceFW/fineweb-2"
HF = "https://huggingface.co"
VIEWER = "https://datasets-server.huggingface.co"
SEED = 48
SAMPLE = 2000
ATTRIBUTION = (
    "FineWeb-2 (Hugging Face): Penedo et al. 2025, FineWeb2: One Pipeline to Scale Them All "
    "(arXiv:2506.20920). ODC-By 1.0; also subject to the Common Crawl Terms of Use."
)
# our language -> (ISO 639-3 codes FineWeb-2 may use, the script we build in, ISO 15924)
LANGUAGES: dict[str, tuple[tuple[str, ...], str]] = {
    "ur": (("urd",), "Arab"),
    "as": (("asm",), "Beng"),
    "bn": (("ben",), "Beng"),
    "gu": (("guj",), "Gujr"),
    "hi": (("hin",), "Deva"),
    "kn": (("kan",), "Knda"),
    "ml": (("mal",), "Mlym"),
    "mr": (("mar",), "Deva"),
    "or": (("ory", "ori"), "Orya"),
    "pa": (("pan",), "Guru"),
    "ta": (("tam",), "Taml"),
    "te": (("tel",), "Telu"),
}
Opener = Callable[..., Any]


# ------------------------------------------------------------------------------------- http --
def http_get(
    url: str, opener: Opener, headers: dict[str, str] | None = None, tries: int = 4
) -> tuple[bytes, Any]:
    """GET with retries (5xx and network errors); returns (body, headers)."""
    last: Exception | None = None
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "frontier-ai-exp048", **(headers or {})})
        try:
            with opener(req, timeout=60) as resp:
                return resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(min(30, 3 * 2**attempt))
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


def http_json(url: str, opener: Opener) -> tuple[Any, Any]:
    body, headers = http_get(url, opener)
    return json.loads(body), headers


def list_tree(rev: str, path: str, opener: Opener) -> list[dict[str, Any]]:
    """Every entry of a folder in the dataset repository at ``rev`` (follows ``Link: rel=next``)."""
    url: str | None = f"{HF}/api/datasets/{REPO}/tree/{rev}/{path}?expand=true"
    out: list[dict[str, Any]] = []
    while url:
        page, headers = http_json(url, opener)
        out.extend(page)
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or "")
        url = m.group(1) if m else None
    return out


def pin_revision(opener: Opener) -> dict[str, Any]:
    info, _ = http_json(f"{HF}/api/datasets/{REPO}/revision/main", opener)
    rev = info["sha"]
    readme, _ = http_get(f"{HF}/datasets/{REPO}/resolve/{rev}/README.md", opener)
    text = readme.decode("utf-8", "replace")
    front = text.split("---", 2)[1] if text.startswith("---") else ""
    m = re.search(r"^license:\s*(.+)$", front, flags=re.M)
    return {
        "dataset": REPO,
        "revision": rev,
        "last_modified": info.get("lastModified"),
        "license_card": m.group(1).strip() if m else None,
        "license_api": (info.get("cardData") or {}).get("license"),
        "readme_sha256": hashlib.sha256(readme).hexdigest(),
    }


def subsets_for(names: list[str]) -> dict[str, dict[str, Any]]:
    """Per language: the subset to measure (code + our script) and the others that were listed."""
    out = {}
    for lang, (codes, script) in LANGUAGES.items():
        cands = sorted(n for n in names if n.split("_")[0] in codes and not n.endswith("_removed"))
        measure = [n for n in cands if n.split("_", 1)[1] == script]
        out[lang] = {
            "measure": measure,
            "listed_not_measured": [n for n in cands if n not in measure],
            "removed_subsets_present": sorted(
                n for n in names if n.split("_")[0] in codes and n.endswith("_removed")
            ),
        }
    return out


def viewer_rows(subset: str, opener: Opener) -> dict[str, Any]:
    try:
        d, _ = http_json(f"{VIEWER}/size?dataset={REPO}&config={subset}", opener)
        train = next(s for s in d["size"]["splits"] if s["split"] == "train")
        return {
            "train_rows": train["num_rows"],
            "train_parquet_bytes": train["num_bytes_parquet_files"],
            "partial": d.get("partial"),
        }
    except Exception as exc:  # noqa: BLE001 - recorded; the footers are the other source
        return {"error": f"{type(exc).__name__}: {exc}"}


def remote_parquet_rows(url: str, size: int, opener: Opener) -> int:
    """Row count from the parquet footer only (two HTTP Range requests)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tail, _ = http_get(url, opener, {"Range": f"bytes={size - 8}-{size - 1}"})
    if len(tail) != 8 or tail[4:] != b"PAR1":
        raise ValueError(f"{url}: not a parquet tail (Range not honoured?)")
    n = struct.unpack("<I", tail[:4])[0]
    footer, _ = http_get(url, opener, {"Range": f"bytes={size - 8 - n}-{size - 1}"})
    if len(footer) != n + 8:
        raise ValueError(f"{url}: footer of {len(footer)} bytes, expected {n + 8}")
    return pq.read_metadata(pa.BufferReader(b"PAR1" + footer)).num_rows


# --------------------------------------------------------------------------------- sampling --
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def sample_rows(n: int, k: int, seed: int) -> list[int]:
    """``min(k, n)`` distinct row numbers of ``range(n)``, fixed by ``(n, seed)``, in ascending order."""
    with np.errstate(over="ignore"):
        keys = _splitmix64(np.arange(n, dtype=np.uint64) ^ (np.uint64(seed) * _GOLDEN))
    return sorted(np.argsort(keys, kind="stable")[: min(k, n)].tolist())


def subset_seed(subset: str) -> int:
    return SEED ^ int.from_bytes(hashlib.sha256(subset.encode()).digest()[:4], "little")


def choose_file(subset: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(files, key=lambda f: f["path"])
    h = int.from_bytes(hashlib.sha256(f"exp048/{subset}".encode()).digest()[:8], "little")
    return ordered[h % len(ordered)]


def read_sample(path: Path, rows: list[int]) -> list[dict[str, Any]]:
    """The chosen rows (id, text, url when present), reading only the row groups that hold them."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    cols = [c for c in ("id", "text", "url") if c in pf.schema_arrow.names]
    if "text" not in cols:
        raise ValueError(f"{path}: no 'text' column (found {pf.schema_arrow.names})")
    out, start, want = [], 0, iter(rows)
    nxt = next(want, None)
    for g in range(pf.metadata.num_row_groups):
        stop = start + pf.metadata.row_group(g).num_rows
        if nxt is not None and nxt < stop:
            t = pf.read_row_group(g, columns=cols).to_pydict()
            while nxt is not None and nxt < stop:
                i = nxt - start
                out.append({"row": nxt, **{c: t[c][i] for c in cols}})
                nxt = next(want, None)
        start = stop
    if len(out) != len(rows):
        raise ValueError(f"{path}: read {len(out)} of {len(rows)} sampled rows")
    return out


# ---------------------------------------------------------------------------------- measure --
def estimate(rows_total: int | None, n: int, kept: int, kept_tokens: list[int]) -> dict[str, Any]:
    if not rows_total or n == 0:
        return {"note": "no row count or no sample; no estimate"}
    p = kept / n
    m = float(np.mean(kept_tokens)) if kept_tokens else 0.0
    sd = float(np.std(kept_tokens, ddof=1)) if len(kept_tokens) > 1 else 0.0
    est = rows_total * p * m
    var = rows_total**2 * (m**2 * p * (1 - p) / n + (p**2 * sd**2 / kept if kept else 0.0))
    se = math.sqrt(var)
    return {
        "keep_rate_new": p,
        "mean_v2_tokens_per_kept_doc": m,
        "new_clean_v2_tokens": est,
        "ci95": [max(0.0, est - 1.96 * se), est + 1.96 * se],
        "label": "NOT VERIFIED: estimate from a sample of one file",
    }


def measure_language(lang: str, ctx: dict[str, Any]) -> dict[str, Any]:
    opener, rev = ctx["opener"], ctx["rev"]
    say = ctx["say"]
    r: dict[str, Any] = {"language": lang, **ctx["subsets"][lang]}
    if len(r["measure"]) != 1:
        r["error"] = f"expected one subset to measure, found {r['measure']}"
        return r
    subset = r["measure"][0]
    entries = [e for e in list_tree(rev, f"data/{subset}/train", opener) if e.get("type") == "file"]
    files = [
        {"path": e["path"], "size": e["size"], "sha256": (e.get("lfs") or {}).get("oid")}
        for e in entries
        if e["path"].endswith(".parquet")
    ]
    r["subset"] = subset
    r["files"] = files
    r["viewer"] = viewer_rows(subset, opener)
    footer_rows: list[int | None] = []
    for f in files:
        try:
            footer_rows.append(
                remote_parquet_rows(f"{HF}/datasets/{REPO}/resolve/{rev}/{f['path']}", f["size"], opener)
            )
        except Exception as exc:  # noqa: BLE001 - recorded per file
            footer_rows.append(None)
            f["footer_error"] = f"{type(exc).__name__}: {exc}"
    for f, n in zip(files, footer_rows):
        f["rows_footer"] = n
    r["train_rows_footers"] = sum(footer_rows) if footer_rows and None not in footer_rows else None
    r["train_bytes"] = sum(f["size"] for f in files)
    v = r["viewer"].get("train_rows")
    r["rows_agree"] = None if v is None or r["train_rows_footers"] is None else v == r["train_rows_footers"]
    rows_total = r["train_rows_footers"] if r["train_rows_footers"] is not None else v
    r["train_rows_used"] = rows_total
    say(f"[exp048] {lang} {subset}: {len(files)} files, {r['train_bytes'] / 1e9:.2f} GB, rows {rows_total}")

    # the sample file, downloaded and checked
    f = choose_file(subset, files)
    if not f["sha256"]:
        r["error"] = f"{f['path']}: the listing gives no SHA-256; not downloaded"
        return r
    pf = PinnedFile(
        source_id=f"fineweb2-{subset}-{Path(f['path']).stem}",
        language=lang,
        sangraha_code=subset,
        script=ctx["pipeline_script"][lang],
        path=f["path"],
        url=f"{HF}/datasets/{REPO}/resolve/{rev}/{f['path']}",
        size=f["size"],
        sha256=f["sha256"],
    )
    t0 = time.monotonic()
    what = download_verified(pf, ctx["work"], log=say, opener=opener)
    local = pf.local_path(ctx["work"])
    r["sample_file"] = {
        "path": f["path"],
        "size": f["size"],
        "sha256": f["sha256"],
        "download": what,
        "seconds": round(time.monotonic() - t0, 1),
    }
    import pyarrow.parquet as pq

    n_file = pq.ParquetFile(str(local)).metadata.num_rows
    picks = sample_rows(n_file, ctx["sample"], subset_seed(subset))
    docs = read_sample(local, picks)
    local.unlink()
    r["sample_file"].update({"rows": n_file, "sampled": len(docs), "deleted_after_use": True})
    urls = {str(d.get("id") or d["row"]): d.get("url") for d in docs}

    # what we already have: exact texts of slice 1 + slice 2 for this language
    t1 = time.monotonic()
    prior: set[bytes] = set()
    for _s, d, entry in ctx["slice_files"][lang]:
        prior.update(document_digests(Path(d) / entry["path"], ctx["tok2"], ctx["eot"]))
    r["prior"] = {
        "distinct_texts": len(prior),
        "seconds": round(time.monotonic() - t1, 1),
        "slices": [s for s, _, _ in ctx["slice_files"][lang]],
    }

    # the unchanged v2 rules + protected suites + overlap, in one pass
    out_gz = Path(ctx["work"]) / f"{lang}.kept.jsonl.gz"
    rows = [(str(d.get("id") or d["row"]), "web", d["text"] or "") for d in docs]
    stats, smp = build_file(
        lambda: iter(rows),
        out_path=out_gz,
        exp_id="EXP-048",
        source_id=pf.source_id,
        language=lang,
        script=pf.script,
        revision=rev,
        guard=ctx["guard"],
        short_index=ctx["short"],
        token_counter=ctx["count"],
        config=ctx["config"],
        shared_exact=set(),
        exclude_final=prior,
    )
    kept_tokens, kept_bytes = [], 0
    with gzip.open(out_gz, "rt", encoding="utf-8") as fh:
        for line in fh:
            text = json.loads(line)["text"]
            kept_tokens.append(len(ctx["enc2"].encode(text)))
            kept_bytes += len(text.encode("utf-8"))
    out_gz.unlink()
    raw_tokens = [len(ctx["enc2"].encode(t)) for _, _, t in rows]
    raw_bytes = sum(len(t.encode("utf-8")) for _, _, t in rows)
    n = len(rows)
    kept = stats["kept"]["docs"]
    overlap = stats["removed_docs"]["slice1_duplicate"]["docs"]
    r["rules"] = {
        "sampled": n,
        "kept_new": kept,
        "already_in_slice1_or_2": overlap,
        "overlap_share_of_survivors": overlap / (kept + overlap) if kept + overlap else None,
        "removed": {k: v["docs"] for k, v in stats["removed_docs"].items() if v["docs"]},
        "suite_hits": stats.get("suite", {}),
        "config_fingerprint": ctx["config"].fingerprint(),
    }
    r["tokens_v2"] = {
        "sampled_docs_tokens": sum(raw_tokens),
        "sampled_docs_bytes": raw_bytes,
        "sampled_tokens_per_byte": sum(raw_tokens) / raw_bytes if raw_bytes else None,
        "kept_docs_tokens": sum(kept_tokens),
        "kept_docs_bytes": kept_bytes,
        "kept_tokens_per_byte": sum(kept_tokens) / kept_bytes if kept_bytes else None,
        "fineweb2_total_tokens_before_our_rules_est": (rows_total * float(np.mean(raw_tokens)))
        if rows_total
        else None,
    }
    r["estimate"] = estimate(rows_total, n, kept, kept_tokens)
    for s in smp:
        u = urls.get(str(s.get("doc_id")))
        s["url"] = mask_contacts(u)[0] if u else None  # a URL may carry an e-mail or phone number
        s["source"] = f"{REPO}@{rev} {f['path']}"
        if s.get("stratum") == "removed:slice1_duplicate":
            s["note"] = "identical to a document already in v2-slice1 or v2-slice2"
    ctx["samples"].extend(smp)
    say(
        f"[exp048] {lang}: kept {kept} of {n}, overlap {overlap}, "
        f"estimate {r['estimate'].get('new_clean_v2_tokens')}"
    )
    return r


# ----------------------------------------------------------------------------------- report --
def render(s: dict[str, Any]) -> str:
    p = s["pin"]
    out = [
        f"EXP-048 FineWeb-2 measurement ({s['created_utc']}); "
        f"code {s.get('environment', {}).get('code_commit')}",
        f"dataset {p.get('dataset')} @ {p.get('revision')}  "
        f"licence (card) {p.get('license_card')}  (api) {p.get('license_api')}",
        f"VERDICT: {s['verdict']}",
        "",
        "lang | subset | train rows (viewer / footers) | sampled | kept new | in slice 1/2 | v2 tok/byte | "
        "est. new clean v2 tokens (95% interval)  [NOT VERIFIED]",
    ]
    for r in s["languages"]:
        if r.get("error"):
            out.append(f"{r['language']:>4} | ERROR: {r['error'].splitlines()[-1][:150]}")
            continue
        e, ru, t = r["estimate"], r["rules"], r["tokens_v2"]
        ci = e.get("ci95") or [0, 0]
        out.append(
            f"{r['language']:>4} | {r['subset']} | "
            f"{r['viewer'].get('train_rows')} / {r['train_rows_footers']} | "
            f"{ru['sampled']} | {ru['kept_new']} | {ru['already_in_slice1_or_2']} | "
            f"{(t['kept_tokens_per_byte'] or 0):.3f} | {e.get('new_clean_v2_tokens', 0) / 1e9:.2f} B "
            f"({ci[0] / 1e9:.2f} - {ci[1] / 1e9:.2f})"
        )
    out += [""] + [
        f"Not measured but listed: {r['language']}: {r['listed_not_measured']}"
        for r in s["languages"]
        if r.get("listed_not_measured")
    ]
    out += ["", "Limits:"] + [f"- {x}" for x in s["limits"]]
    out += ["", f"Attribution: {ATTRIBUTION}", "Samples to read before deciding: samples.jsonl"]
    return "\n".join(out) + "\n"


LIMITS = [
    "2,000 documents from ONE file per language: in-file boilerplate and near-duplicate rules see far less "
    "repetition than in a full build, so the keep rate is likely an upper bound.",
    "Overlap = identical text only (a lower bound); near-duplicates of slice 1/2 are not found.",
    "URL overlap not measured (whether Sangraha Verified keeps source URLs is NOT VERIFIED).",
    "FineWeb-2 has no English; English stays from Sangraha.",
    "Estimates use train rows x keep rate x mean tokens; the interval covers sampling error only.",
]


def main(argv: list[str] | None = None, opener: Opener = urllib.request.urlopen) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="/kaggle/input")
    ap.add_argument("--heldout-jsonl", required=True)
    ap.add_argument("--suite", default=b2.DEFAULT_SUITE)
    ap.add_argument("--belebele-dir", required=True)
    ap.add_argument("--no-verify", action="store_true", help="tests only: Belebele not downloaded")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=SAMPLE)
    ap.add_argument("--languages", default=",".join(LANGUAGES))
    ap.add_argument("--slices-root", default=str(ROOT), help="tests only: where the slice manifests live")
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    say = lambda m: print(m, flush=True)  # noqa: E731
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    langs = [x for x in args.languages.split(",") if x]

    cfg = BuildConfig()
    suite = load_suite(args.suite if Path(args.suite).is_absolute() else ROOT / args.suite)
    held = b2.heldout_from_jsonl(Path(args.heldout_jsonl), suite, work)
    bele, bele_info = b2.belebele_pairs(Path(args.belebele_dir), verify=not args.no_verify)
    from frontier_ai.corpus.pack import OrdinaryEncoder
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer, load_frontier_tokenizer_v2

    tok2 = load_frontier_tokenizer_v2()
    _, pins = load_slice_pins(ROOT / b2.DEFAULT_PINS)
    pipeline_script = {pf.language: pf.script for pf in pins}

    # the slices we compare against, each checked against its pinned manifest
    root = Path(args.slices_root)
    slice_files: dict[str, list[tuple[str, str, dict[str, Any]]]] = {lang: [] for lang in langs}
    slice_record = []
    for s in sl.EXP047_SLICES:
        man = sl.load_manifest(s, root)
        d = sl.locate(s, man, Path(args.input_dir))
        sl.check_files(s, man, d, full_sha256=True)
        slice_record.append(
            {"name": s.name, "manifest_sha256": s.manifest_sha256, "dir": str(d), "full_sha256_checked": True}
        )
        for f in man["files"]:
            if f["language"] in slice_files:
                slice_files[f["language"]].append((s.name, str(d), f))

    pin = pin_revision(opener)
    say(f"[exp048] {REPO} @ {pin['revision']} licence {pin['license_card']}")
    names = [
        e["path"].split("/", 1)[1]
        for e in list_tree(pin["revision"], "data", opener)
        if e.get("type") == "directory"
    ]
    subsets = subsets_for(names)
    ctx = {
        "opener": opener,
        "rev": pin["revision"],
        "say": say,
        "subsets": subsets,
        "work": work,
        "sample": args.sample,
        "pipeline_script": pipeline_script,
        "slice_files": slice_files,
        "tok2": tok2,
        "eot": tok2.special_token_ids["<|endoftext|>"],
        "enc2": OrdinaryEncoder(tok2),
        "count": TokenCounter(load_frontier_tokenizer()),
        "guard": SuiteGuard.from_texts(held + bele),
        "short": ShortSuiteIndex.from_texts(held, min_words=cfg.short_suite_min_words),
        "config": cfg,
        "samples": [],
    }
    results = []
    for lang in langs:
        try:
            results.append(measure_language(lang, ctx))
        except Exception:  # noqa: BLE001 - one language's failure is recorded, the others continue
            results.append({"language": lang, **subsets.get(lang, {}), "error": traceback.format_exc()})
    ok = [r for r in results if not r.get("error")]
    s: dict[str, Any] = {
        "schema": SCHEMA,
        "exp_id": "EXP-048",
        "smoke": args.sample < SAMPLE,
        "complete": len(ok) == len(langs),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pin": pin,
        "subset_count_in_listing": len(names),
        "sample_per_language": args.sample,
        "seed": SEED,
        "config": asdict(cfg),
        "suite": {"suite_id": suite["suite_id"], "heldout_docs": len(held), "belebele": bele_info},
        "slices": slice_record,
        "languages": results,
        "limits": LIMITS,
        "attribution": ATTRIBUTION,
        "verdict": (
            f"MEASURED {len(ok)} of {len(langs)} languages "
            "(estimates NOT VERIFIED; read samples.jsonl before any decision)"
        ),
    }
    with contextlib.suppress(Exception):
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        s["environment"] = {"code_commit": head.stdout.strip() or None, "python": sys.version.split()[0]}
    s["seconds"] = round(time.monotonic() - t0, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with open(out / "samples.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for item in ctx["samples"]:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "SUMMARY.txt").write_text(render(s), encoding="utf-8")
    print(render(s), flush=True)
    return 0 if s["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
