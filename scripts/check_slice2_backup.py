#!/usr/bin/env python3
"""Check the FrontierCorpus v2-slice2 backup (two private Kaggle datasets) against the build manifests.

    python scripts/check_slice2_backup.py --input-dir /kaggle/input --out /kaggle/working/EXP-046/backup-check

Why this exists
---------------
D-051 accepted v2-slice2. Its data existed only as two Kaggle kernel outputs
(``frontier-exp046-build-a`` / ``-b``). On 2026-10-09 the founder copied them into the private datasets
``frontier-v2-slice2-a`` / ``-b`` with Kaggle's web interface. Kaggle unzipped the ``<lang>.jsonl.gz``
text files on the way (confirmed: the datasets hold ``<lang>.jsonl``). This script checks, for each
part (A, B), using ``evals/results/EXP-046/build-<part>/manifest.json``:

1. **Token files (what training reads):** every ``<lang>.bin`` in the backup has the manifest's size and
   SHA-256, and every ``<lang>.meta.json`` has the manifest's ``n_train`` / ``n_val`` (and is
   byte-identical to the original's, when the original is mounted).
2. **Text files:** if the backup holds ``<lang>.jsonl.gz``, its SHA-256 must be the manifest's. If it
   holds the unzipped ``<lang>.jsonl``, its line count must be the manifest's document count and, when
   the original kernel output is mounted, its bytes must equal the original ``.jsonl.gz`` unzipped (the
   original's own SHA-256 is checked against the manifest first). Without the original, the text check
   is reported as "documents counted, content not compared", never as passed in full.

Which mounted folder is which is decided from the folder path: the backup's path contains the dataset
name (``frontier-v2-slice2-a``), the original's the kernel name (``frontier-exp046-build-a``). Exactly
one folder of each kind may hold all of a part's token files; anything else stops the check.

Writes ``summary.json`` and ``SUMMARY.txt``. Nothing is changed or copied. Reading is done in threads
(hashlib and zlib release the GIL on large blocks).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "frontier-exp046-backup-check-v1"
PARTS = ("A", "B")
BLOCK = 8 << 20


def manifest_path(part: str) -> Path:
    return ROOT / "evals" / "results" / "EXP-046" / f"build-{part}" / "manifest.json"


def backup_name(part: str) -> str:
    return f"frontier-v2-slice2-{part.lower()}"


def original_name(part: str) -> str:
    return f"frontier-exp046-build-{part.lower()}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def text_digest_and_lines(path: Path) -> tuple[str, int]:
    """SHA-256 and newline count of a plain file."""
    h = hashlib.sha256()
    lines = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
            lines += block.count(b"\n")
    return h.hexdigest(), lines


def gunzip_digest_and_lines(path: Path) -> tuple[str, str, int]:
    """(SHA-256 of the .gz file, SHA-256 of its unzipped bytes, unzipped newline count).

    Handles several concatenated gzip members (the build joins two parts that way)."""
    hz = hashlib.sha256()
    hu = hashlib.sha256()
    lines = 0
    d = zlib.decompressobj(wbits=31)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            hz.update(block)
            data = block
            while data:
                out = d.decompress(data)
                hu.update(out)
                lines += out.count(b"\n")
                if d.eof:
                    data = d.unused_data
                    d = zlib.decompressobj(wbits=31)
                else:
                    data = b""
        tail = d.flush()
        hu.update(tail)
        lines += tail.count(b"\n")
    return hz.hexdigest(), hu.hexdigest(), lines


def find_dirs(input_dir: Path, bins: list[str], part: str) -> tuple[Path, Path | None]:
    """(backup folder, original folder or None) holding all of a part's token files."""
    first = bins[0]
    holders = sorted({p.parent for p in input_dir.rglob(first) if all((p.parent / b).exists() for b in bins)})
    backups = [d for d in holders if backup_name(part) in str(d)]
    originals = [d for d in holders if original_name(part) in str(d)]
    others = [d for d in holders if d not in backups and d not in originals]
    if len(backups) != 1 or len(originals) > 1 or others:
        raise SystemExit(
            f"part {part}: expected one backup folder (path containing {backup_name(part)!r}) "
            f"and at most one original (path containing {original_name(part)!r}); "
            f"found backups {backups}, originals {originals}, others {others}"
        )
    return backups[0], (originals[0] if originals else None)


def check_language(entry: dict[str, Any], backup: Path, original: Path | None) -> dict[str, Any]:
    lang = entry["language"]
    r: dict[str, Any] = {"language": lang}
    # 1. token file
    b = backup / entry["path"]
    size = b.stat().st_size
    sha = sha256_file(b)
    r["bin"] = {
        "bytes": size,
        "sha256": sha,
        "ok": size == entry["bytes"] and sha == entry["sha256"],
    }
    # 1b. meta side-car
    m = backup / entry["meta"]
    meta_ok = False
    meta_note = "missing"
    if m.exists():
        meta = json.loads(m.read_text(encoding="utf-8"))
        meta_ok = meta.get("n_train") == entry["n_train"] and meta.get("n_val") == entry["n_val"]
        meta_note = "n_train/n_val match" if meta_ok else "n_train/n_val differ"
        if original is not None and (original / entry["meta"]).exists():
            same = m.read_bytes() == (original / entry["meta"]).read_bytes()
            meta_ok = meta_ok and same
            meta_note += "; identical to the original" if same else "; DIFFERENT from the original"
    r["meta"] = {"ok": meta_ok, "note": meta_note}
    # 2. text
    t = entry["text"]
    gz = backup / t["path"]
    plain = backup / t["path"].removesuffix(".gz")
    if gz.exists():
        gsha = sha256_file(gz)
        r["text"] = {"form": "gz (as built)", "sha256": gsha, "ok": gsha == t["sha256"], "full": True}
    elif plain.exists():
        psha, lines = text_digest_and_lines(plain)
        res: dict[str, Any] = {"form": "unzipped by Kaggle", "sha256_unzipped": psha, "lines": lines}
        docs_ok = lines == t["docs"]
        if original is not None and (original / t["path"]).exists():
            osha, ousha, olines = gunzip_digest_and_lines(original / t["path"])
            res["original_gz_sha256_ok"] = osha == t["sha256"]
            res["identical_to_original_unzipped"] = ousha == psha
            res["ok"] = docs_ok and res["original_gz_sha256_ok"] and res["identical_to_original_unzipped"]
            res["full"] = True
        else:
            res["ok"] = docs_ok
            res["full"] = False
            res["note"] = "documents counted, content not compared (original not mounted)"
        r["text"] = res
    else:
        r["text"] = {"form": "missing", "ok": False, "full": False}
    r["ok"] = r["bin"]["ok"] and r["meta"]["ok"] and r["text"]["ok"]
    return r


def check_part(part: str, input_dir: Path, workers: int) -> dict[str, Any]:
    man = json.loads(manifest_path(part).read_text(encoding="utf-8"))
    entries = man["files"]
    backup, original = find_dirs(input_dir, [e["path"] for e in entries], part)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda e: check_language(e, backup, original), entries))
    return {
        "part": part,
        "manifest": str(manifest_path(part).relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path(part)),
        "backup_dir": str(backup),
        "original_dir": str(original) if original else None,
        "languages": rows,
        "token_files_ok": all(r["bin"]["ok"] and r["meta"]["ok"] for r in rows),
        "text_ok": all(r["text"]["ok"] for r in rows),
        "text_fully_compared": all(r["text"].get("full") for r in rows),
        "seconds": round(time.monotonic() - started, 1),
    }


def verdict(parts: list[dict[str, Any]]) -> str:
    if not all(p["token_files_ok"] for p in parts):
        return "FAIL: at least one token file differs from the manifest"
    if not all(p["text_ok"] for p in parts):
        return "FAIL: token files match, but at least one text file differs"
    if not all(p["text_fully_compared"] for p in parts):
        return "PASS for token files; text documents counted but not compared byte for byte"
    return "PASS: every token file matches the manifest and every text file equals the original"


def render(s: dict[str, Any]) -> str:
    out = [
        f"EXP-046 slice-2 backup check ({s['created_utc']})",
        f"VERDICT: {s['verdict']}",
        "",
    ]
    for p in s["parts"]:
        out.append(
            f"Part {p['part']}: backup {p['backup_dir']}; original {p['original_dir'] or 'not mounted'}"
        )
        for r in p["languages"]:
            t = r["text"]
            out.append(
                f"  {r['language']:>3}  bin {'ok' if r['bin']['ok'] else 'DIFFERENT'}"
                f" ({r['bin']['bytes']:,} bytes)"
                f"  meta {'ok' if r['meta']['ok'] else 'BAD'} ({r['meta']['note']})"
                f"  text {t['form']}: {'ok' if t['ok'] else 'BAD'}"
                + (f" ({t['lines']:,} documents)" if "lines" in t else "")
            )
        out.append(f"  ({p['seconds']} s)")
    out.append("")
    out.append("Nothing was changed or copied; only reading and hashing.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="/kaggle/input")
    p.add_argument("--out", required=True)
    p.add_argument("--parts", default=",".join(PARTS))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    started = time.monotonic()
    parts = [check_part(x, Path(args.input_dir), args.workers) for x in args.parts.split(",") if x]
    s: dict[str, Any] = {
        "schema": SCHEMA,
        "exp_id": "EXP-046",
        "smoke": False,
        "complete": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parts": parts,
        "verdict": verdict(parts),
        "seconds": 0.0,
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
