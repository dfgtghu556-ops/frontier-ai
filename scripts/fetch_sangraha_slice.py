#!/usr/bin/env python3
"""Download the pinned Sangraha Verified slice (D-044, EXP-034) and verify every byte.

    python scripts/fetch_sangraha_slice.py                 # all 13 files (~5.1 GB)
    python scripts/fetch_sangraha_slice.py --only hi,ta    # a subset (our language codes)
    python scripts/fetch_sangraha_slice.py --check         # verify what is on disk, download nothing

* Reads the pins from ``corpora/frontier/v2/sangraha_slice1.json`` (revision, size, SHA-256).
* Checks free disk space first (the missing bytes + a safety margin); stops before downloading
  anything if the disk is too full.
* Resumes a dropped download where it stopped (HTTP Range) and retries with a growing wait.
* Accepts a file only if its size AND SHA-256 equal the pin; a file that fails is renamed
  ``*.rejected-*`` and never used. Files already downloaded and verified are skipped.
* Writes into the git-ignored ``data/`` directory; never touches tracked files.

Exit codes: 0 = every requested file present and verified; 1 = a download/verification failed;
2 = usage/pin/disk problem (nothing downloaded).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.corpus.sangraha import (
    SliceError,
    check_free_space,
    download_verified,
    load_slice_pins,
    sha256_file,
)

DEFAULT_PINS = "corpora/frontier/v2/sangraha_slice1.json"
DEFAULT_ROOT = "data/sangraha"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a legacy console code page
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pins", default=DEFAULT_PINS)
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="download root; files go to <root>/<revision[:12]>/<path> (default: %(default)s)")
    p.add_argument("--only", default="", help="comma-separated language codes (default: all)")
    p.add_argument("--margin-gb", type=float, default=3.0,
                   help="free space to leave on the disk after downloading (default: %(default)s GB)")
    p.add_argument("--check", action="store_true", help="verify files on disk only; download nothing")
    p.add_argument("--retries", type=int, default=5)
    args = p.parse_args(argv)

    try:
        header, files = load_slice_pins(args.pins)
    except (SliceError, OSError, KeyError, TypeError) as exc:
        print(f"[fetch] bad pin file: {exc}", file=sys.stderr)
        return 2
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        unknown = wanted - {f.language for f in files}
        if unknown:
            print(f"[fetch] unknown language code(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        files = tuple(f for f in files if f.language in wanted)
    root = Path(args.root) / header["revision"][:12]
    print(f"[fetch] {header['slice_id']} @ {header['revision'][:12]}: {len(files)} files, "
          f"{sum(f.size for f in files) / 1e9:.2f} GB -> {root}")

    if args.check:
        bad = 0
        for pf in files:
            path = pf.local_path(root)
            ok = path.is_file() and path.stat().st_size == pf.size and sha256_file(path) == pf.sha256
            bad += not ok
            print(f"[fetch] {'OK     ' if ok else 'MISSING/BAD'} {pf.source_id}")
        return 0 if bad == 0 else 1

    try:
        needed, free = check_free_space(files, root, int(args.margin_gb * 1e9))
    except SliceError as exc:
        print(f"[fetch] STOP: {exc}", file=sys.stderr)
        return 2
    print(f"[fetch] disk: {needed / 1e9:.2f} GB still to download, {free / 1e9:.2f} GB free (OK)")

    started = time.monotonic()
    failures = 0
    for i, pf in enumerate(files, 1):
        print(f"[fetch] ({i}/{len(files)}) {pf.source_id} ({pf.size / 1e6:,.0f} MB)", flush=True)
        try:
            download_verified(pf, root, log=lambda m: print(m, flush=True), retries=args.retries)
        except SliceError as exc:
            failures += 1
            print(f"[fetch] FAILED: {exc}", file=sys.stderr, flush=True)
    minutes = (time.monotonic() - started) / 60
    if failures:
        print(f"[fetch] {failures} of {len(files)} files FAILED ({minutes:.1f} min); "
              "re-running this command resumes where it stopped", file=sys.stderr)
        return 1
    print(f"[fetch] all {len(files)} files present and verified ({minutes:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
