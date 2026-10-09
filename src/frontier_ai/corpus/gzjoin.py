"""Single-member gzip files (approved 2026-10-09, founder "DO ALL": backup items b and b2).

Why: v2-slice2's ``<lang>.jsonl.gz`` text files were made by joining two gzip files byte for byte
(two "members"; valid gzip that Python, zcat and the packer read in full). When the founder made a
Kaggle dataset from those files, Kaggle unzipped them and silently kept only the FIRST member, i.e.
only the data-1 half (EXP-046 backup check, 2026-10-09). A gzip file with exactly one member has no
such trap. These helpers write and check such files; they stream, so memory stays small.

Output is deterministic: no file name and mtime 0 in the gzip header, fixed compression level.
"""

from __future__ import annotations

import gzip
import hashlib
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

BLOCK = 8 << 20


def iter_gunzip(path: Path, block: int = BLOCK) -> Iterator[bytes]:
    """Yield the unzipped bytes of ``path``, across ALL of its gzip members."""
    d = zlib.decompressobj(wbits=31)
    with Path(path).open("rb") as f:
        for raw in iter(lambda: f.read(block), b""):
            data = raw
            while data:
                out = d.decompress(data)
                if out:
                    yield out
                if d.eof:
                    data = d.unused_data
                    d = zlib.decompressobj(wbits=31)
                else:
                    data = b""
        tail = d.flush()
        if tail:
            yield tail


def gzip_members(path: Path, block: int = BLOCK) -> int:
    """Number of gzip members in ``path`` (an empty file has 0)."""
    members = 0
    d = zlib.decompressobj(wbits=31)
    started = False
    with Path(path).open("rb") as f:
        for raw in iter(lambda: f.read(block), b""):
            data = raw
            while data:
                started = True
                d.decompress(data)
                if d.eof:
                    members += 1
                    data = d.unused_data
                    d = zlib.decompressobj(wbits=31)
                    started = False
                else:
                    data = b""
    if started:
        raise ValueError(f"{path}: the last gzip member is truncated")
    return members


def unzipped_digest(path: Path) -> dict[str, Any]:
    """SHA-256, byte count and newline count of the unzipped content (all members)."""
    h = hashlib.sha256()
    size = lines = 0
    for chunk in iter_gunzip(path):
        h.update(chunk)
        size += len(chunk)
        lines += chunk.count(b"\n")
    return {"sha256": h.hexdigest(), "bytes": size, "lines": lines}


def write_single_member(sources: Iterable[Path], dst: Path, compresslevel: int = 6) -> dict[str, Any]:
    """Unzip every source (all members, in order) into ONE gzip member at ``dst``.

    Written to ``dst + '.tmp'`` first and renamed at the end, so a stopped run leaves no half file
    under the final name. Returns the unzipped SHA-256 / bytes / lines and the new file's size."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    h = hashlib.sha256()
    size = lines = 0
    with tmp.open("wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, compresslevel=compresslevel, mtime=0) as gz:
            for src in sources:
                for chunk in iter_gunzip(Path(src)):
                    gz.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
                    lines += chunk.count(b"\n")
    tmp.replace(dst)
    return {"sha256": h.hexdigest(), "bytes": size, "lines": lines, "gz_bytes": dst.stat().st_size}
