"""Sangraha Verified slice: pinned files, verified download, streaming reader (D-044, EXP-034).

Why this exists
---------------
D-044 (founder, 2026-09-29) approves the first level-B source for FrontierCorpus v2:
AI4Bharat's Sangraha *Verified* subset (CC-BY-4.0). The first slice is one parquet file per
language (``corpora/frontier/v2/sangraha_slice1.json``, ~5.1 GB total). This module is the
bridge from those pins to the existing pipeline:

* :func:`load_slice_pins` — reads and validates the pin file (revision, size, SHA-256 per file).
* :func:`download_verified` — stdlib-only HTTP download that resumes after a dropped connection
  (HTTP ``Range``) and accepts a file only when its size **and** SHA-256 match the pin. A file that
  fails the check is renamed aside, never used.
* :func:`iter_rows` / :func:`iter_documents` — stream a parquet file one row group batch at a
  time, so memory stays bounded on the 2-core laptop. Documents come out as the pipeline's
  :class:`~frontier_ai.corpus.pipeline.PipelineDocument` with a provenance-bearing ``doc_id``
  (``<source_id>-<sangraha doc_id>``), so every document can be traced and, if the law or our
  policy changes, removed by rebuilding without its source (D-044 rule).

``pyarrow`` is needed only for reading parquet; it is imported lazily (optional extra ``data``).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontier_ai.corpus.pipeline import PipelineDocument

PINS_SCHEMA = "frontier-v2-source-pins-v1"
ROW_COLUMNS = ("doc_id", "type", "text")
USER_AGENT = "frontier-ai-corpus/1 (+https://github.com/dfgtghu556-ops/frontier-ai)"
CHUNK = 1 << 20  # 1 MiB


class SliceError(RuntimeError):
    """A pin file, download or parquet file does not meet its contract."""


@dataclass(frozen=True)
class PinnedFile:
    source_id: str
    language: str  # our ISO 639-1 code (the pipeline's language key)
    sangraha_code: str  # Sangraha's 3-letter directory name
    script: str  # pipeline script block for the langid gate
    path: str  # path inside the dataset repository
    url: str
    size: int
    sha256: str

    def local_path(self, root: Path | str) -> Path:
        return Path(root) / self.path


def load_slice_pins(path: Path | str) -> tuple[dict[str, Any], tuple[PinnedFile, ...]]:
    """Read ``sangraha_slice*.json``; return (header, files). Fails loudly on any inconsistency."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != PINS_SCHEMA:
        raise SliceError(f"{path}: expected schema {PINS_SCHEMA!r}, got {data.get('schema')!r}")
    revision = data.get("revision", "")
    if len(revision) != 40:
        raise SliceError(f"{path}: revision must be a full 40-character commit id")
    files: list[PinnedFile] = []
    seen: set[str] = set()
    for i, f in enumerate(data.get("files", [])):
        pf = PinnedFile(**{k: f[k] for k in PinnedFile.__dataclass_fields__})
        if pf.source_id in seen:
            raise SliceError(f"{path}: duplicate source_id {pf.source_id!r}")
        seen.add(pf.source_id)
        if revision not in pf.url or not pf.url.endswith(pf.path):
            raise SliceError(f"{path}: file #{i} url is not pinned to revision {revision[:12]} / {pf.path}")
        if len(pf.sha256) != 64 or pf.size <= 0:
            raise SliceError(f"{path}: file #{i} needs a 64-hex sha256 and a positive size")
        if ".." in Path(pf.path).parts or Path(pf.path).is_absolute():
            raise SliceError(f"{path}: file #{i} path escapes the download root")
        files.append(pf)
    if not files:
        raise SliceError(f"{path}: no files")
    if data.get("total_bytes") != sum(f.size for f in files):
        raise SliceError(f"{path}: total_bytes does not equal the sum of the file sizes")
    return data, tuple(files)


# ------------------------------------------------------------------ download --
def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def bytes_still_needed(files: tuple[PinnedFile, ...], root: Path | str) -> int:
    """Bytes left to download (complete files count 0; partial ``.part`` files count the rest)."""
    need = 0
    for pf in files:
        final = pf.local_path(root)
        if final.is_file() and final.stat().st_size == pf.size:
            continue
        part = final.with_name(final.name + ".part")
        have = part.stat().st_size if part.is_file() else 0
        need += max(pf.size - min(have, pf.size), 0)
    return need


def check_free_space(files: tuple[PinnedFile, ...], root: Path | str, margin_bytes: int) -> tuple[int, int]:
    """Return (needed, free); raise :class:`SliceError` if free < needed + margin."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    needed = bytes_still_needed(files, root)
    free = shutil.disk_usage(root).free
    if free < needed + margin_bytes:
        raise SliceError(
            f"not enough disk space in {root}: need {needed / 1e9:.2f} GB + {margin_bytes / 1e9:.2f} GB "
            f"safety margin, only {free / 1e9:.2f} GB free"
        )
    return needed, free


def _set_aside(path: Path, why: str) -> Path:
    aside = path.with_name(f"{path.name}.rejected-{why}-{time.strftime('%Y%m%d-%H%M%S')}")
    path.rename(aside)
    return aside


def download_verified(
    pf: PinnedFile,
    root: Path | str,
    log: Callable[[str], None] = print,
    retries: int = 5,
    timeout: float = 60.0,
    backoff_seconds: float = 10.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> str:
    """Make ``root/pf.path`` exist with exactly the pinned bytes. Returns what happened.

    * already present and verified -> ``"verified"`` (nothing downloaded);
    * otherwise downloads into ``<file>.part``, resuming with ``Range`` when the server allows it
      (HTTP 206); a plain 200 restarts from byte 0 so a resume can never splice wrong bytes;
    * the finished file must match the pinned size and SHA-256, else it is renamed aside
      (``.rejected-<time>``) and a :class:`SliceError` is raised.
    """
    final = pf.local_path(root)
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + ".part")

    if final.is_file():
        if final.stat().st_size == pf.size and sha256_file(final) == pf.sha256:
            log(f"[fetch] {pf.source_id}: already downloaded and verified")
            return "verified"
        aside = _set_aside(final, "mismatch")
        log(f"[fetch] {pf.source_id}: existing file failed verification; moved to {aside.name}")

    attempt = 0
    while True:
        attempt += 1
        have = part.stat().st_size if part.is_file() else 0
        if have > pf.size:
            aside = _set_aside(part, "oversize")
            log(f"[fetch] {pf.source_id}: partial file larger than the pin; moved to {aside.name}")
            have = 0
        if have == pf.size:
            break
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(pf.url, headers=headers)
        try:
            with opener(request, timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                if have and status != 206:
                    log(f"[fetch] {pf.source_id}: server ignored the resume request (HTTP {status}); "
                        "restarting from byte 0")
                    have = 0
                mode = "ab" if have else "wb"
                started, done, last_log = time.monotonic(), have, 0.0
                with open(part, mode) as out:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        now = time.monotonic()
                        if now - last_log >= 60:
                            rate = (done - have) / max(now - started, 1e-9) / 1e6
                            log(f"[fetch] {pf.source_id}: {done / 1e6:,.0f} / {pf.size / 1e6:,.0f} MB "
                                f"({100 * done / pf.size:.1f}%, {rate:.2f} MB/s)")
                            last_log = now
            if part.stat().st_size >= pf.size:
                break
            raise OSError(f"connection ended early at {part.stat().st_size} of {pf.size} bytes")
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            if attempt > retries:
                raise SliceError(f"{pf.source_id}: download failed after {attempt} attempts: {exc}") from exc
            wait = backoff_seconds * attempt
            log(f"[fetch] {pf.source_id}: attempt {attempt} failed ({exc}); retrying in {wait:.0f}s "
                "(keeps the bytes already downloaded)")
            time.sleep(wait)

    size = part.stat().st_size
    digest = sha256_file(part)
    if size != pf.size or digest != pf.sha256:
        aside = _set_aside(part, "mismatch")
        raise SliceError(
            f"{pf.source_id}: downloaded file does not match its pin (size {size} vs {pf.size}, "
            f"sha256 {digest[:12]}... vs {pf.sha256[:12]}...); moved to {aside.name}, not used"
        )
    part.replace(final)
    log(f"[fetch] {pf.source_id}: downloaded and verified ({pf.size / 1e6:,.1f} MB, sha256 OK)")
    return "downloaded"


# -------------------------------------------------------------------- reader --
def _pyarrow_parquet():
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 - optional dependency, imported on use
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SliceError("reading parquet needs pyarrow: pip install -e \".[data]\"") from exc
    return pq


def parquet_row_count(path: Path | str) -> int:
    return _pyarrow_parquet().ParquetFile(str(path)).metadata.num_rows


def iter_rows(path: Path | str, batch_size: int = 1024) -> Iterator[tuple[str, str, str]]:
    """Yield ``(doc_id, type, text)`` rows in file order, one batch in memory at a time."""
    pq = _pyarrow_parquet()
    pfile = pq.ParquetFile(str(path))
    names = set(pfile.schema_arrow.names)
    missing = [c for c in ROW_COLUMNS if c not in names]
    if missing:
        raise SliceError(f"{path}: missing columns {missing}; found {sorted(names)}")
    for batch in pfile.iter_batches(batch_size=batch_size, columns=list(ROW_COLUMNS)):
        cols = [batch.column(i).to_pylist() for i in range(3)]
        for doc_id, kind, text in zip(*cols):
            yield (doc_id or "", kind or "", text or "")


def iter_documents(pf: PinnedFile, root: Path | str, batch_size: int = 1024) -> Iterator[PipelineDocument]:
    """Stream one pinned file as pipeline documents (raw text; the pipeline normalizes)."""
    for i, (doc_id, _kind, text) in enumerate(iter_rows(pf.local_path(root), batch_size=batch_size)):
        yield PipelineDocument(
            doc_id=f"{pf.source_id}-{doc_id or f'row{i:09d}'}",
            source_id=pf.source_id,
            language=pf.language,
            text=text,
        )
