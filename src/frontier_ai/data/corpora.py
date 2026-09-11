"""Real, clearly licensed smoke-test corpora: manifest, acquisition, provenance.

The synthetic generator (``synthetic.py``) stays the default CI fixture: zero network,
zero licensing questions. This module adds the *second* fixture the roadmap asks for —
a small amount of **real text with a known licence** — without ever guessing where it
came from:

* :data:`CorpusSource` / :func:`load_sources` — the manifest in
  ``corpora/smoke/sources.json``. Each entry names its source URL, licence (SPDX id),
  attribution string and, once somebody has actually fetched it, the SHA-256 of the
  cleaned text. A hash is only ever stored **after** a verified fetch, never asserted
  from memory.
* :func:`clean_gutenberg_text` / :func:`clean_wikitext` — deterministic text cleaning
  that strips Project Gutenberg's legal wrapper and wiki markup, so the corpus is text
  and nothing else.
* :func:`write_provenance` — the provenance record that travels with the corpus: what
  it is, where it came from, under which licence, when it was retrieved, and the hash
  of the bytes we actually kept.
* :func:`fetch_text` — the only network call in the package, isolated here so tests
  never need it.

The corpus is a **smoke-test / evaluation fixture**, deliberately small (tens of
kilobytes). It is not training-scale data, and nothing here selects a tokenizer or a
model.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

# Licences we are willing to ship as fixture text. Anything else is refused by
# validate_source() so a questionable dataset cannot slip into the manifest.
ALLOWED_LICENSES = {
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "PD-US",  # public domain in the United States (the Project Gutenberg case)
}

# A smoke fixture, not a training corpus: hard upper bound per source.
DEFAULT_MAX_CHARS = 200_000

GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"

_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_REF_BLOCK = re.compile(r"<ref[^>/]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
_RE_REF_SELF = re.compile(r"<ref[^>]*?/>", re.IGNORECASE)
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_HEADING = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
_RE_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_RE_TAG = re.compile(r"<[^>]+>")
_RE_BLANKS = re.compile(r"\n{3,}")
_RE_SPACES = re.compile(r"[ \t]{2,}")

# Markers we look for in the *raw* downloaded text before we are willing to pin a
# hash. They do not replace reading the licence; they stop a silent substitution.
_LICENSE_MARKERS = {
    "PD-US": ("project gutenberg",),
    "CC-BY-4.0": ("creative commons attribution", "cc-by-4.0", "cc by 4.0"),
    "CC-BY-SA-4.0": (
        "creative commons attribution-sharealike",
        "cc-by-sa",
        "cc by-sa",
        "cc-by-sa-4.0",
        "creative commons attribution/ share-alike",
    ),
    "CC0-1.0": ("cc0", "creative commons public domain dedication"),
}


@dataclass(frozen=True)
class CorpusSource:
    """One licensed text we may turn into a smoke corpus."""

    id: str
    title: str
    language: str
    script: str
    source_url: str
    license_id: str
    license_url: str
    attribution: str
    max_chars: int = DEFAULT_MAX_CHARS
    kind: str = "gutenberg"  # "gutenberg" | "wikitext" | "plain"
    sha256: str | None = None
    verified: bool = False
    retrieved_at: str | None = None
    notes: str = ""

    # ------------------------------------------------------------------ io --
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusSource:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown keys in corpus source {data.get('id')!r}: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def pinned(self) -> bool:
        return bool(self.sha256) and self.verified

    def license_markers(self) -> tuple[str, ...]:
        return _LICENSE_MARKERS.get(self.license_id, ())


@dataclass
class SourceManifest:
    """The on-disk manifest (``corpora/smoke/sources.json``)."""

    schema_version: str = SCHEMA_VERSION
    sources: list[CorpusSource] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> SourceManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported corpus manifest schema {version!r} (expected {SCHEMA_VERSION!r})")
        return cls(
            schema_version=version,
            sources=[CorpusSource.from_dict(item) for item in data.get("sources", [])],
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "notes": self.notes,
            "sources": [source.to_dict() for source in self.sources],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)

    def by_id(self, source_id: str) -> CorpusSource:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"no corpus source with id {source_id!r}")


def load_sources(path: str | Path) -> list[CorpusSource]:
    """Convenience wrapper: manifest file -> list of sources."""
    return SourceManifest.load(path).sources


def validate_source(source: CorpusSource) -> list[str]:
    """Return a list of problems with a manifest entry (empty means it is usable).

    Kept deliberately strict: a fixture with a vague licence is worse than no
    fixture, because it quietly makes every downstream number unreviewable.
    """
    problems: list[str] = []
    if not source.id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", source.id):
        problems.append(f"{source.id!r}: id must be lowercase [a-z0-9_-]")
    if source.license_id not in ALLOWED_LICENSES:
        problems.append(f"{source.id}: licence {source.license_id!r} is not in {sorted(ALLOWED_LICENSES)}")
    if not source.source_url.startswith("https://"):
        problems.append(f"{source.id}: source_url must be https")
    # an attribution you cannot paste into a report is not an attribution
    if len(source.attribution.strip()) < 16:
        problems.append(f"{source.id}: attribution text is too short to be usable")
    if not source.license_url.startswith("https://"):
        problems.append(f"{source.id}: license_url must be https")
    if source.max_chars <= 0 or source.max_chars > DEFAULT_MAX_CHARS:
        problems.append(f"{source.id}: max_chars must be in (0, {DEFAULT_MAX_CHARS}]")
    if source.kind not in {"gutenberg", "wikitext", "plain"}:
        problems.append(f"{source.id}: kind must be gutenberg, wikitext or plain")
    if source.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", source.sha256):
        problems.append(f"{source.id}: sha256 must be 64 lowercase hex characters")
    if source.verified and not source.sha256:
        problems.append(f"{source.id}: verified=true requires a pinned sha256")
    return problems


# ------------------------------------------------------------------ cleaning --


def clean_gutenberg_text(text: str) -> str:
    """Strip the Project Gutenberg header/footer, keeping only the work itself.

    Gutenberg wraps every file in a licence block delimited by ``*** START/END OF
    THE PROJECT GUTENBERG EBOOK ... ***``. The licence block is *their* text and is
    not the corpus, so it is removed; the licence is recorded in the provenance
    instead. If the markers are absent the text is returned unchanged (with CRLF
    normalised) — we never guess where a work begins.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    start = normalised.find(GUTENBERG_START)
    end = normalised.find(GUTENBERG_END)
    if start == -1 or end == -1 or end <= start:
        return normalised.strip()
    start = normalised.find("\n", start)
    if start == -1:
        return normalised.strip()
    return normalised[start + 1 : end].strip()


def clean_wikitext(text: str) -> str:
    """Reduce wikitext to plain text with deterministic, reviewable substitutions.

    Order matters (comments before tags, refs before tags) and every rule is a pure
    string operation, so the same input always yields the same output. This is a
    smoke-fixture cleaner, not a general wikitext parser: anything exotic is left for
    a human to notice in the review step.
    """
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _RE_COMMENT.sub("", out)
    out = _RE_REF_BLOCK.sub("", out)
    out = _RE_REF_SELF.sub("", out)
    for _ in range(3):  # nested templates, bounded so it always terminates
        out, before = _RE_TEMPLATE.sub("", out), out
        if out == before:
            break
    out = _RE_HEADING.sub(r"\1", out)
    out = _RE_LINK.sub(r"\1", out)
    out = out.replace("'''", "").replace("''", "")
    out = _RE_TAG.sub("", out)
    out = _RE_SPACES.sub(" ", out)
    out = _RE_BLANKS.sub("\n\n", out)
    lines = [line.strip() for line in out.split("\n")]
    return "\n".join(lines).strip()


def clean_text(text: str, kind: str) -> str:
    if kind == "gutenberg":
        return clean_gutenberg_text(text)
    if kind == "wikitext":
        return clean_wikitext(text)
    if kind == "plain":
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()
    raise ValueError(f"unknown corpus kind {kind!r}")


def trim_text(text: str, max_chars: int) -> str:
    """Trim to `max_chars`, ending on a line boundary so no word is cut in half."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars + 1)
    return text[: cut if cut > 0 else max_chars].strip()


def prepare_source_text(raw: str, source: CorpusSource) -> str:
    """Clean + trim according to the manifest entry."""
    return trim_text(clean_text(raw, source.kind), source.max_chars)


# ---------------------------------------------------------------- provenance --


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: str | Path, expected_sha256: str | None) -> bool:
    """True when the file's hash matches `expected_sha256`.

    An unpinned (``None``) hash is *not* a match — "we never checked" must never be
    reported as verified.
    """
    if not expected_sha256:
        return False
    return sha256_file(path) == expected_sha256


def license_marker_found(raw: str, source: CorpusSource) -> bool:
    """Whether the raw download carries a marker for the licence we claim.

    A weak check by design: it catches a substituted URL or a page that turned out to
    be under a different licence. It is not a substitute for reading the licence.
    """
    markers = source.license_markers()
    if not markers:
        return False
    haystack = raw[:50_000].lower()
    return any(marker in haystack for marker in markers)


def build_provenance(
    source: CorpusSource,
    text: str,
    *,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """The provenance record for a cleaned corpus file."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": source.id,
        "title": source.title,
        "language": source.language,
        "script": source.script,
        "source_url": source.source_url,
        "license_id": source.license_id,
        "license_url": source.license_url,
        "attribution": source.attribution,
        "kind": source.kind,
        "retrieved_at": retrieved_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256_text(text),
        "n_chars": len(text),
        "n_bytes": len(text.encode("utf-8")),
        "max_chars": source.max_chars,
        "verified": source.verified and source.sha256 == sha256_text(text),
        "notes": source.notes,
    }


def write_provenance(
    path: str | Path,
    source: CorpusSource,
    text: str,
    *,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Write the provenance JSON next to the corpus file and return it."""
    record = build_provenance(source, text, retrieved_at=retrieved_at)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def load_provenance(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- network --


class FetchError(RuntimeError):
    """Raised when a source cannot be retrieved (no network, 404, bad encoding)."""


def fetch_text(url: str, timeout: float = 30.0) -> str:
    """Download `url` and return it as text. The only network call in the package."""
    request = urllib.request.Request(url, headers={"User-Agent": "frontier-ai-smoke-corpus/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise FetchError(f"could not retrieve {url}: {exc}") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FetchError(f"{url} is not valid UTF-8: {exc}") from exc
