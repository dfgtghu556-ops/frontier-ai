#!/usr/bin/env python3
"""A complete backup of v2-slice2's TEXT files (approved 2026-10-09, founder "DO ALL", backup item b).

    python scripts/slice2_text_backup.py rewrite --input-dir /kaggle/input \\
        --out /kaggle/working/EXP-046/text-rewrite --text-out /kaggle/working/slice2-text
    python scripts/slice2_text_backup.py verify --input-dir /kaggle/input \\
        --out /kaggle/working/EXP-046/text-verify

Why this exists
---------------
The backup check of 2026-10-09 (``evals/results/EXP-046/backup-check/``) found that the token files in
the founder's datasets ``frontier-v2-slice2-a`` / ``-b`` are complete, but the text files hold only the
data-1 half: each ``<lang>.jsonl.gz`` has two gzip members, and Kaggle's unzip kept only the first.
The full text exists only in the build kernels' outputs (``frontier-exp046-build-a`` / ``-b``).

``rewrite`` (one Kaggle CPU session, those two outputs mounted read only): for each of the 13 languages,
check the original's SHA-256 against ``evals/results/EXP-046/build-<part>/manifest.json``, unzip ALL of
its members into ONE new gzip member ``<text-out>/<lang>.jsonl.gz``, then read the new file back: it
must have exactly one member, the same unzipped SHA-256 as the original and as many lines as the
manifest's document count. The new files become the founder's dataset ``frontier-v2-slice2-text``.

``verify`` (one Kaggle CPU session, that dataset mounted): every language's file in the dataset
(``<lang>.jsonl`` if Kaggle unzipped it, else ``<lang>.jsonl.gz``) must have the unzipped SHA-256 and
line count recorded by ``rewrite`` (read from the committed ``evals/results/EXP-046/text-rewrite/``).

Both write ``summary.json`` and ``SUMMARY.txt``; the text never leaves Kaggle. Work runs in threads
(zlib and hashlib release the GIL on large blocks).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_slice2_backup as cb  # noqa: E402
from frontier_ai.corpus.gzjoin import gzip_members, unzipped_digest, write_single_member  # noqa: E402

SCHEMA = "frontier-exp046-text-backup-v1"
DATASET = "frontier-v2-slice2-text"
LANGUAGES = 13  # v2-slice2: 6 in part A + 7 in part B
REWRITE_SUMMARY = ROOT / "evals" / "results" / "EXP-046" / "text-rewrite" / "summary.json"


def find_original_dir(input_dir: Path, texts: list[str], part: str) -> Path:
    """The one mounted build output (path containing frontier-exp046-build-<part>) with all texts."""
    holders = sorted(
        {p.parent for p in input_dir.rglob(texts[0]) if all((p.parent / t).exists() for t in texts)}
    )
    found = [d for d in holders if cb.original_name(part) in str(d)]
    if len(found) != 1:
        raise SystemExit(
            f"part {part}: expected exactly one folder with all text files whose path contains "
            f"{cb.original_name(part)!r}; found {found} (all holders: {holders})"
        )
    return found[0]


def rewrite_language(entry: dict[str, Any], original: Path, text_out: Path) -> dict[str, Any]:
    t = entry["text"]
    lang = entry["language"]
    src = original / t["path"]
    r: dict[str, Any] = {"language": lang, "path": t["path"], "manifest_docs": t["docs"]}
    started = time.monotonic()
    osha = cb.sha256_file(src)
    r["original"] = {"sha256": osha, "sha256_ok": osha == t["sha256"], "gzip_members": gzip_members(src)}
    if not r["original"]["sha256_ok"]:
        r["ok"] = False
        r["note"] = "the original differs from the manifest; nothing written"
        return r
    w = write_single_member([src], text_out / t["path"])
    new = text_out / t["path"]
    back = unzipped_digest(new)
    members = gzip_members(new)
    r["new"] = {
        "path": t["path"],
        "gz_bytes": w["gz_bytes"],
        "gz_sha256": cb.sha256_file(new),
        "gzip_members": members,
        "unzipped_sha256": w["sha256"],
        "unzipped_bytes": w["bytes"],
        "lines": w["lines"],
    }
    r["checks"] = {
        "one_member": members == 1,
        "read_back_equal": back == {"sha256": w["sha256"], "bytes": w["bytes"], "lines": w["lines"]},
        "lines_equal_manifest_docs": w["lines"] == t["docs"],
    }
    r["ok"] = all(r["checks"].values())
    r["seconds"] = round(time.monotonic() - started, 1)
    return r


def run_rewrite(input_dir: Path, text_out: Path, workers: int) -> dict[str, Any]:
    text_out.mkdir(parents=True, exist_ok=True)
    parts = []
    for part in cb.PARTS:
        man = json.loads(cb.manifest_path(part).read_text(encoding="utf-8"))
        entries = man["files"]
        original = find_original_dir(input_dir, [e["text"]["path"] for e in entries], part)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda e, o=original: rewrite_language(e, o, text_out), entries))
        parts.append(
            {
                "part": part,
                "manifest": str(cb.manifest_path(part).relative_to(cb.ROOT)),
                "manifest_sha256": cb.sha256_file(cb.manifest_path(part)),
                "original_dir": str(original),
                "languages": rows,
                "ok": all(r["ok"] for r in rows),
                "seconds": round(time.monotonic() - started, 1),
            }
        )
    ok = all(p["ok"] for p in parts)
    return {
        "mode": "rewrite",
        "text_out": str(text_out),
        "parts": parts,
        "verdict": (
            "PASS: all 13 text files rewritten as one gzip member each (same content, all documents)"
            if ok
            else "FAIL: at least one text file could not be rewritten and checked (see the languages)"
        ),
    }


def find_dataset_file(input_dir: Path, path: str) -> Path:
    plain = path.removesuffix(".gz")
    hits = sorted(
        p
        for p in input_dir.rglob(f"{plain}*")
        if p.name in (plain, path) and DATASET in str(p) and p.is_file()
    )
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one {plain} or {path} under a path containing {DATASET!r}; found {hits}"
        )
    return hits[0]


def verify_language(row: dict[str, Any], input_dir: Path) -> dict[str, Any]:
    want = row["new"]
    f = find_dataset_file(input_dir, want["path"])
    if f.name.endswith(".gz"):
        got = unzipped_digest(f)
        form = "gz (not unzipped by Kaggle)"
    else:
        sha, lines = cb.text_digest_and_lines(f)
        got = {"sha256": sha, "bytes": f.stat().st_size, "lines": lines}
        form = "unzipped by Kaggle"
    ok = got["sha256"] == want["unzipped_sha256"] and got["lines"] == want["lines"]
    return {"language": row["language"], "file": str(f), "form": form, "found": got, "ok": ok}


def run_verify(input_dir: Path, workers: int) -> dict[str, Any]:
    rec = json.loads(REWRITE_SUMMARY.read_text(encoding="utf-8"))
    if not rec.get("verdict", "").startswith("PASS"):
        raise SystemExit(f"{REWRITE_SUMMARY} did not pass; nothing to verify")
    rows_in = [r for p in rec["parts"] for r in p["languages"]]
    mounted = sorted(str(d) for d in input_dir.rglob("*") if d.is_dir() and d.name == DATASET)
    if not mounted:
        # Kaggle starts the kernel even when the dataset is missing; say so in the summary
        return {
            "mode": "verify",
            "dataset": DATASET,
            "rewrite_summary_sha256": cb.sha256_file(REWRITE_SUMMARY),
            "languages": [],
            "verdict": f"FAIL: the dataset {DATASET} is not mounted (does it exist, and was it ready?)",
        }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda r: verify_language(r, input_dir), rows_in))
    ok = len(rows) == LANGUAGES and all(r["ok"] for r in rows)
    return {
        "mode": "verify",
        "dataset": DATASET,
        "rewrite_summary_sha256": cb.sha256_file(REWRITE_SUMMARY),
        "languages": rows,
        "verdict": (
            "PASS: the dataset holds all 13 text files in full (same content as the build outputs)"
            if ok
            else "FAIL: at least one text file in the dataset differs (see the languages)"
        ),
    }


def render(s: dict[str, Any]) -> str:
    out = [f"EXP-046 slice-2 text backup, {s['mode']} ({s['created_utc']})", f"VERDICT: {s['verdict']}", ""]
    if s["mode"] == "rewrite":
        for p in s["parts"]:
            out.append(f"Part {p['part']}: original {p['original_dir']} ({p['seconds']} s)")
            for r in p["languages"]:
                n = r.get("new", {})
                out.append(
                    f"  {r['language']:>3}  {'ok' if r['ok'] else 'BAD'}  original members "
                    f"{r['original']['gzip_members']}, new members {n.get('gzip_members', '-')}, "
                    f"{n.get('lines', 0):,} documents (manifest {r['manifest_docs']:,}), "
                    f"{n.get('gz_bytes', 0):,} bytes"
                )
        out.append("")
        out.append(f"New files: {s['text_out']} (make the dataset {DATASET} from this kernel's output).")
    else:
        for r in s["languages"]:
            out.append(
                f"  {r['language']:>3}  {'ok' if r['ok'] else 'BAD'}  {r['form']}, "
                f"{r['found']['lines']:,} documents"
            )
        out.append("")
        out.append("Nothing was changed; only reading and hashing.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["rewrite", "verify"])
    p.add_argument("--input-dir", default="/kaggle/input")
    p.add_argument("--out", required=True)
    p.add_argument("--text-out", default="/kaggle/working/slice2-text")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    started = time.monotonic()
    if args.mode == "rewrite":
        body = run_rewrite(Path(args.input_dir), Path(args.text_out), args.workers)
    else:
        body = run_verify(Path(args.input_dir), args.workers)
    s: dict[str, Any] = {
        "schema": SCHEMA,
        "exp_id": "EXP-046",
        "smoke": False,
        "complete": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **body,
    }
    with contextlib.suppress(Exception):
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        s["environment"] = {"code_commit": head.stdout.strip() or None, "python": sys.version.split()[0]}
    s["seconds"] = round(time.monotonic() - started, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    (out / "SUMMARY.txt").write_text(render(s), encoding="utf-8")
    print(render(s), flush=True)
    return 0 if s["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
