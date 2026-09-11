#!/usr/bin/env python3
"""Fetch the small, clearly licensed real-text smoke corpora.

The manifest in ``corpora/smoke/sources.json`` says *what* we are allowed to use and
under which licence. This script retrieves it, cleans it, records its provenance, and
— once a human has seen a successful verified fetch — pins its SHA-256. Nothing is
ever hashed from memory: until ``--pin`` runs, ``sha256`` is ``null`` and the corpus is
explicitly unverified.

    python scripts/fetch_smoke_corpus.py --list
    python scripts/fetch_smoke_corpus.py --fetch                 # fetch everything
    python scripts/fetch_smoke_corpus.py --fetch en-alice-pd     # fetch one source
    python scripts/fetch_smoke_corpus.py --fetch --pin           # fetch and pin the hashes
    python scripts/fetch_smoke_corpus.py --check                 # verify what is on disk

Then feed it to the normal pipeline:

    python scripts/prepare_data.py --source data/raw/smoke/en-alice-pd.txt \\
        --provenance data/raw/smoke/en-alice-pd.provenance.json --out data/smoke-en

Exit codes: 0 ok · 1 ``--check`` found a missing or mismatching file · 2 a pinned
hash did not match during ``--fetch`` (the file was left untouched) · 3 a fetch failed
or the licence marker was missing while ``--pin`` was requested.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.data.corpora import (  # noqa: E402
    FetchError,
    SourceManifest,
    fetch_text,
    license_marker_found,
    prepare_source_text,
    sha256_file,
    sha256_text,
    validate_source,
    verify_file,
    write_provenance,
)

DEFAULT_SOURCES = "corpora/smoke/sources.json"
DEFAULT_OUT_DIR = "data/raw/smoke"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default=DEFAULT_SOURCES, help=f"corpus manifest (default: {DEFAULT_SOURCES})")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"where to write the text (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--list", action="store_true", help="list the manifest and exit")
    p.add_argument("--fetch", nargs="*", metavar="ID", default=None,
                   help="fetch the given source ids (default: all)")
    p.add_argument("--check", action="store_true", help="verify files on disk against pinned hashes")
    p.add_argument("--pin", action="store_true",
                   help="with --fetch: record the observed sha256 in the manifest")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds")
    p.add_argument("--retrieved-at", default=None,
                   help="override the retrieval timestamp (ISO UTC) for reproducible provenance")
    return p.parse_args()


def _print_list(manifest: SourceManifest) -> None:
    print(f"{'id':24s} {'lang':5s} {'licence':14s} {'state':10s} sha256")
    for source in manifest.sources:
        state = "pinned" if source.pinned else ("unverified" if source.sha256 else "not-fetched")
        digest = (source.sha256 or "")[:16] + ("…" if source.sha256 else "-")
        print(f"{source.id:24s} {source.language:5s} {source.license_id:14s} {state:10s} {digest}")
        print(f"{'':24s} {source.title}")
        print(f"{'':24s} {source.source_url}")


def _check(manifest: SourceManifest, out_dir: Path) -> int:
    failures = 0
    for source in manifest.sources:
        text_path = out_dir / f"{source.id}.txt"
        prov_path = out_dir / f"{source.id}.provenance.json"
        if not text_path.exists():
            if source.pinned:
                print(f"[corpus] MISSING {source.id}: pinned corpus not present at {text_path}")
                failures += 1
            else:
                print(f"[corpus] not fetched: {source.id} (no hash pinned yet)")
            continue
        if not source.sha256:
            print(f"[corpus] UNVERIFIED {source.id}: {text_path} exists but no hash is pinned")
            continue
        if verify_file(text_path, source.sha256):
            print(f"[corpus] ok       {source.id}: {text_path} matches {source.sha256[:16]}…")
        else:
            print(f"[corpus] MISMATCH {source.id}: {text_path} does not match {source.sha256[:16]}…")
            failures += 1
        if not prov_path.exists():
            print(f"[corpus] MISSING  {source.id}: provenance file {prov_path}")
            failures += 1
    return 1 if failures else 0


def _fetch_one(source, out_dir: Path, args: argparse.Namespace) -> tuple[int, dict]:
    """Fetch, clean and store one source.

    Returns ``(exit_code, info)``; ``info`` carries the digest of the bytes actually
    written and whether the *raw* download carried a licence marker, so the caller can
    decide about pinning without re-deriving either.
    """
    text_path = out_dir / f"{source.id}.txt"
    prov_path = out_dir / f"{source.id}.provenance.json"
    print(f"[corpus] fetching {source.id}: {source.source_url}")
    try:
        raw = fetch_text(source.source_url, timeout=args.timeout)
    except FetchError as exc:
        print(f"[corpus] FAILED   {source.id}: {exc}")
        return 3, {}

    text = prepare_source_text(raw, source)
    if len(text) < 1000:
        print(f"[corpus] FAILED   {source.id}: only {len(text)} chars after cleaning - refusing to write")
        return 3, {}

    # the licence marker lives in the raw download (Gutenberg's header, a Wikisource
    # licence template) which cleaning removes, so check it before cleaning
    marker = license_marker_found(raw, source)
    if not marker:
        print(f"[corpus] WARNING  {source.id}: no {source.license_id} marker found in the download "
              f"- read the licence at {source.license_url} before using this text")

    content = text + "\n"  # exactly the bytes we store; the provenance hashes these
    if source.sha256 and source.sha256 != sha256_text(content):
        print(f"[corpus] MISMATCH {source.id}: pinned {source.sha256[:16]}… but downloaded "
              f"{sha256_text(content)[:16]}… - leaving {text_path} untouched")
        return 2, {}

    out_dir.mkdir(parents=True, exist_ok=True)
    text_path.write_text(content, encoding="utf-8")
    write_provenance(prov_path, source, content, retrieved_at=args.retrieved_at)
    digest = sha256_file(text_path)
    print(f"[corpus] wrote    {text_path} ({len(text):,} chars, sha256 {digest})")
    print(f"[corpus] wrote    {prov_path} (licence {source.license_id})")
    if not source.sha256:
        print(f"[corpus] next     {source.id} has no pinned hash yet: re-run with --pin once you "
              f"have checked the licence, so --check can verify it later")
    return 0, {"digest": digest, "marker": marker}


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.sources)
    if not manifest_path.exists():
        raise SystemExit(f"[corpus] manifest not found: {manifest_path}")
    manifest = SourceManifest.load(manifest_path)

    problems = {source.id: validate_source(source) for source in manifest.sources}
    bad = {sid: p for sid, p in problems.items() if p}
    if bad:
        for sid, items in bad.items():
            print(f"[corpus] invalid manifest entry {sid}:")
            for item in items:
                print(f"    - {item}")
        return 1

    if args.list:
        _print_list(manifest)
        return 0

    out_dir = Path(args.out_dir)
    if args.check:
        return _check(manifest, out_dir)

    if args.fetch is None:
        # no action requested: show the manifest rather than doing something surprising
        _print_list(manifest)
        print("\n[corpus] nothing to do: pass --fetch (optionally with ids), --check or --list")
        return 0

    wanted = args.fetch or [source.id for source in manifest.sources]
    unknown = [sid for sid in wanted if sid not in {source.id for source in manifest.sources}]
    if unknown:
        raise SystemExit(f"[corpus] unknown source id(s): {', '.join(unknown)}")

    exit_code = 0
    pinned_now: list[str] = []
    for source in manifest.sources:
        if source.id not in wanted:
            continue
        code, info = _fetch_one(source, out_dir, args)
        exit_code = max(exit_code, code)
        if code != 0 or not args.pin or source.pinned or not info:
            continue
        if not info["marker"]:
            print(f"[corpus] NOT PINNED {source.id}: no {source.license_id} marker in the download "
                  f"- refusing to pin a hash for text of unverified licence")
            exit_code = max(exit_code, 3)
            continue
        for index, entry in enumerate(manifest.sources):
            if entry.id == source.id:
                manifest.sources[index] = replace(
                    entry, sha256=info["digest"], verified=True, retrieved_at=args.retrieved_at
                )
        pinned_now.append(f"{source.id} -> {info['digest']}")

    if pinned_now:
        manifest.save(manifest_path)
        print(f"[corpus] pinned {len(pinned_now)} hash(es) in {manifest_path}")
        for line in pinned_now:
            print(f"    {line}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
