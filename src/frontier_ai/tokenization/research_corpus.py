"""Tokenizer-research corpus foundation (Project 004, Stage A).

Why this exists
---------------
Project 002's ``indic-eval-v1`` fixture is 109 hand-written probes across 14 languages
(6-8 per language), and its training text is *generated* from word banks that overlap the
probes. That is fine for regression, and useless for choosing a tokenizer. Stage A builds
the corpus foundation the later stages need: a versioned, licensed, deterministic corpus
with a real train/held-out split, leakage diagnostics and an honest coverage report.

What it deliberately does **not** do
------------------------------------
* It does not invent data. A source is ``verified`` only after it has actually been
  fetched, its licence marker has been found, and its SHA-256 has been computed (the
  D-031 rule: hashes are pinned after a verified fetch, never asserted from memory).
* It never normalizes Unicode. ``normalization_policy`` is ``none``: NFC/NFD/NFKC
  experiments are a later stage, and silently changing the source text would make every
  later comparison unreviewable.
* It does not make a language look covered. A slot without verified sources is reported
  as ``NOT_EVALUATED`` / ``UNVERIFIED`` / ``INSUFFICIENT`` with a reason.
* It does not select, train or compare tokenizers.

Reuse
-----
Everything text-, licence- and hash-related comes from :mod:`frontier_ai.data.corpora`
(fetch, cleaning, provenance, ``ALLOWED_LICENSES``, ``sha256_text``/``sha256_file``) and
the language slots come from :data:`frontier_ai.tokenization.corpus.LANGUAGES`. This
module adds only what a *tokenizer corpus* needs: the manifest, document splitting, the
deterministic split, leakage diagnostics, statistics and coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..data.corpora import (
    ALLOWED_LICENSES,
    CorpusSource,
    FetchError,
    fetch_text,
    license_marker_found,
    prepare_source_text,
    sha256_file,
    sha256_text,
    write_provenance,
)
from .corpus import LANGUAGES

MANIFEST_SCHEMA_VERSION = "1.0"
CORPUS_ID = "indic-tokenizer"
CORPUS_VERSION = "v2"

# The repo's corpus root, resolved relative to this file so the module works from any cwd.
CORPORA_ROOT = Path(__file__).resolve().parents[3] / "corpora"
DEFAULT_MANIFEST_PATH = CORPORA_ROOT / "tokenizer" / "indic-tokenizer-v2" / "sources.json"

# Corpus targets from the Project 004 design audit. They are goals, not quotas.
TARGET_SENTENCES = 500
TARGET_CHARS = 200_000

DEFAULT_SPLIT_SEED = 1337
DEFAULT_HELD_OUT_FRACTION = 0.1

# A tokenizer corpus is bigger than the smoke fixture, so its per-source cap is larger.
MAX_SOURCE_CHARS = 2_000_000
VALID_KINDS = {"gutenberg", "wikitext", "plain"}

# Splitting: documents are paragraphs; very long paragraphs are cut on sentence
# punctuation, and only then hard-wrapped, so no text is silently dropped.
MAX_DOC_CHARS = 1_200
_SENTENCE_BOUNDARY = re.compile(r"(?<=[\u0964\u0965।॥.!?])\s+")

# Leakage diagnostics: word n-grams (character n-grams for scripts with little
# whitespace), over a deterministically chosen sample so the check is bounded.
NGRAM_N = 8
MAX_NGRAM_DOCS = 2_000

STATUS_EVALUATED = "EVALUATED"
STATUS_INSUFFICIENT = "INSUFFICIENT"
STATUS_UNVERIFIED = "UNVERIFIED"
STATUS_NOT_EVALUATED = "NOT_EVALUATED"

CORPUS_FILENAME = "corpus.json"
TRAIN_FILENAME = "train.jsonl"
HELD_OUT_FILENAME = "heldout.jsonl"
STATS_FILENAME = "stats.json"
COVERAGE_FILENAME = "coverage.json"
LEAKAGE_FILENAME = "leakage.json"


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
@dataclass
class LanguageSlot:
    """One of the 14 language slots the research must cover."""

    code: str
    display: str = ""
    script: str = ""
    target_sentences: int = TARGET_SENTENCES
    target_chars: int = TARGET_CHARS
    sources: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageSlot:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown keys in language slot {data.get('code')!r}: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizerCorpusManifest:
    """``corpora/tokenizer/<corpus>/sources.json``."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    corpus: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    normalization_policy: str = "none"
    cleaning_policy: str = ""
    language_slots: list[LanguageSlot] = field(default_factory=list)
    sources: list[CorpusSource] = field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------------ io --
    @classmethod
    def load(cls, path: str | Path) -> TokenizerCorpusManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported tokenizer corpus manifest schema {version!r} "
                f"(expected {MANIFEST_SCHEMA_VERSION!r})"
            )
        return cls(
            schema_version=version,
            corpus=dict(data.get("corpus") or {}),
            targets=dict(data.get("targets") or {}),
            split=dict(data.get("split") or {}),
            normalization_policy=str(data.get("normalization_policy", "none")),
            cleaning_policy=str(data.get("cleaning_policy", "")),
            language_slots=[LanguageSlot.from_dict(item) for item in data.get("language_slots", [])],
            sources=[CorpusSource.from_dict(item) for item in data.get("sources", [])],
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus": self.corpus,
            "targets": self.targets,
            "split": self.split,
            "normalization_policy": self.normalization_policy,
            "cleaning_policy": self.cleaning_policy,
            "language_slots": [slot.to_dict() for slot in self.language_slots],
            "sources": [source.to_dict() for source in self.sources],
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)

    # -------------------------------------------------------------- helpers --
    @property
    def corpus_id(self) -> str:
        return str(self.corpus.get("id", CORPUS_ID))

    @property
    def corpus_version(self) -> str:
        return str(self.corpus.get("version", CORPUS_VERSION))

    @property
    def split_seed(self) -> int:
        return int(self.split.get("seed", DEFAULT_SPLIT_SEED))

    @property
    def held_out_fraction(self) -> float:
        return float(self.split.get("heldout_fraction", DEFAULT_HELD_OUT_FRACTION))

    def slot(self, code: str) -> LanguageSlot:
        for slot in self.language_slots:
            if slot.code == code:
                return slot
        raise KeyError(f"no language slot {code!r} in the manifest")

    def source(self, source_id: str) -> CorpusSource:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"no corpus source with id {source_id!r}")

    def sources_for(self, code: str) -> list[CorpusSource]:
        slot = self.slot(code)
        by_id = {source.id: source for source in self.sources}
        return [by_id[sid] for sid in slot.sources if sid in by_id]


def validate_manifest(manifest: TokenizerCorpusManifest) -> list[str]:
    """Problems with the manifest (empty means usable).

    Strict on purpose: a corpus entry with a vague licence, a non-https URL or a
    ``verified`` flag without a hash makes every downstream tokenizer number
    unreviewable, which is worse than having no corpus at all.
    """
    problems: list[str] = []
    if not manifest.corpus_id or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", manifest.corpus_id):
        problems.append(f"corpus id {manifest.corpus_id!r} must be lowercase [a-z0-9-]")
    if not re.fullmatch(r"v\d+", manifest.corpus_version):
        problems.append(f"corpus version {manifest.corpus_version!r} must look like 'v2'")

    codes = [slot.code for slot in manifest.language_slots]
    if len(set(codes)) != len(codes):
        problems.append(f"duplicate language slots: {sorted({c for c in codes if codes.count(c) > 1})}")
    for code in codes:
        if code not in LANGUAGES:
            problems.append(
                f"language slot {code!r} is not one of the researched languages "
                f"{sorted(LANGUAGES)}"
            )

    known_slots = set(codes)
    for source in manifest.sources:
        if source.language not in known_slots:
            problems.append(f"source {source.id!r}: language {source.language!r} has no language slot")
        problems.extend(_validate_source(source))
    return problems


def _validate_source(source: CorpusSource) -> list[str]:
    """Per-source checks: the smoke-corpus rules with the larger tokenizer cap."""
    problems: list[str] = []
    if source.license_id not in ALLOWED_LICENSES:
        problems.append(f"{source.id}: licence {source.license_id!r} is not in {sorted(ALLOWED_LICENSES)}")
    if not source.source_url.startswith("https://"):
        problems.append(f"{source.id}: source_url must be https")
    if not source.license_url.startswith("https://"):
        problems.append(f"{source.id}: license_url must be https")
    if len(source.attribution.strip()) < 16:
        problems.append(f"{source.id}: attribution text is too short to be usable")
    if source.max_chars <= 0 or source.max_chars > MAX_SOURCE_CHARS:
        problems.append(f"{source.id}: max_chars must be in (0, {MAX_SOURCE_CHARS}]")
    if source.kind not in VALID_KINDS:
        problems.append(f"{source.id}: kind must be one of {sorted(VALID_KINDS)}")
    if source.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", source.sha256):
        problems.append(f"{source.id}: sha256 must be 64 lowercase hex characters")
    if source.verified and not source.sha256:
        problems.append(f"{source.id}: verified=true requires a pinned sha256")
    return problems


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
@dataclass
class IngestedSource:
    """What happened when one declared source was ingested."""

    source_id: str
    language: str
    status: str = "not_attempted"      # verified | local_unverified | fetch_failed |
    #                                  # licence_marker_missing | not_attempted | empty
    path: str | None = None
    provenance_path: str | None = None
    sha256: str | None = None
    chars: int = 0
    bytes: int = 0
    documents: int = 0
    retrieved_at: str | None = None
    error: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    @property
    def has_text(self) -> bool:
        return self.path is not None and self.chars > 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verified"] = self.verified
        return payload


def ingest_source(
    source: CorpusSource,
    raw_dir: str | Path,
    *,
    fetch: bool = False,
    timeout: float = 30.0,
    local_text: str | None = None,
) -> IngestedSource:
    """Fetch (or accept) one source, clean it, and write text + provenance.

    A source becomes ``verified`` **only** when it was fetched over https, carries a
    marker for the licence it claims, and produced a non-empty cleaned text whose SHA-256
    is recorded. Locally supplied text (test fixtures, or text a human lawfully obtained)
    can be ingested but is never auto-verified: it is recorded as ``local_unverified``
    and cannot make a language slot count as evaluated.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if local_text is not None:
        text = prepare_source_text(local_text, source)
        status = "local_unverified"
    elif fetch:
        try:
            raw = fetch_text(source.source_url, timeout=timeout)
        except FetchError as exc:
            return IngestedSource(
                source_id=source.id,
                language=source.language,
                status="fetch_failed",
                retrieved_at=retrieved_at,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - a download can fail in many ways
            return IngestedSource(
                source_id=source.id,
                language=source.language,
                status="fetch_failed",
                retrieved_at=retrieved_at,
                error=f"{type(exc).__name__}: {exc}",
            )
        if not license_marker_found(raw, source):
            return IngestedSource(
                source_id=source.id,
                language=source.language,
                status="licence_marker_missing",
                retrieved_at=retrieved_at,
                error=(
                    f"no {source.license_id} marker found in the first 50,000 characters of "
                    f"{source.source_url}; refusing to pin a hash"
                ),
            )
        text = prepare_source_text(raw, source)
        status = "verified"
    else:
        return IngestedSource(source_id=source.id, language=source.language, status="not_attempted")

    if not text.strip():
        return IngestedSource(
            source_id=source.id,
            language=source.language,
            status="empty",
            retrieved_at=retrieved_at,
            error="cleaned text is empty",
        )

    text_path = raw_dir / f"{source.id}.txt"
    text_path.write_text(text, encoding="utf-8")
    provenance_path = raw_dir / f"{source.id}.provenance.json"
    write_provenance(provenance_path, source, text, retrieved_at=retrieved_at)

    return IngestedSource(
        source_id=source.id,
        language=source.language,
        status=status,
        path=str(text_path),
        provenance_path=str(provenance_path),
        sha256=sha256_text(text),
        chars=len(text),
        bytes=len(text.encode("utf-8")),
        retrieved_at=retrieved_at,
    )


# ---------------------------------------------------------------------------
# documents, split, leakage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusDocument:
    """One example: a paragraph, or a sentence-sized chunk of one."""

    doc_id: str
    source_id: str
    language: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "language": self.language,
            "chars": self.chars,
            "bytes": self.bytes,
            "text": self.text,
        }


def documents_from_text(source_id: str, language: str, text: str) -> list[CorpusDocument]:
    """Split cleaned text into documents deterministically (no text is dropped)."""
    documents: list[CorpusDocument] = []
    index = 0
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for chunk in _chunk(paragraph):
            documents.append(
                CorpusDocument(
                    doc_id=f"{source_id}-{index:06d}",
                    source_id=source_id,
                    language=language,
                    text=chunk,
                )
            )
            index += 1
    return documents


def _chunk(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_DOC_CHARS:
        return [paragraph]
    pieces = [piece.strip() for piece in _SENTENCE_BOUNDARY.split(paragraph) if piece.strip()]
    merged: list[str] = []
    for piece in pieces:
        if len(piece) > MAX_DOC_CHARS:  # a sentence longer than the cap: hard wrap it
            merged.extend(piece[i : i + MAX_DOC_CHARS] for i in range(0, len(piece), MAX_DOC_CHARS))
        else:
            merged.append(piece)
    return merged or [paragraph[:MAX_DOC_CHARS]]


def split_documents(
    documents: Sequence[CorpusDocument],
    seed: int = DEFAULT_SPLIT_SEED,
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
) -> tuple[list[CorpusDocument], list[CorpusDocument]]:
    """Deterministic document-level train/held-out split.

    The bucket of a document is ``sha256('<seed>:<doc_id>')`` mapped to ``[0, 1)``. There is
    no global RNG and no dependence on document order, so the same corpus and seed always
    produce the same split, and changing the seed re-splits reproducibly.
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError(f"held_out_fraction must be in (0, 1), got {held_out_fraction!r}")
    train: list[CorpusDocument] = []
    held_out: list[CorpusDocument] = []
    for document in documents:
        digest = hashlib.sha256(f"{seed}:{document.doc_id}".encode()).hexdigest()
        bucket = int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF
        (held_out if bucket < held_out_fraction else train).append(document)
    return train, held_out


def _ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) >= n:
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    compact = "".join(words)
    if len(compact) >= n:  # scripts with little whitespace: fall back to characters
        return {compact[i : i + n] for i in range(len(compact) - n + 1)}
    return set()


def leakage_report(
    train: Sequence[CorpusDocument],
    held_out: Sequence[CorpusDocument],
    n: int = NGRAM_N,
) -> dict[str, Any]:
    """Lightweight, reproducible leakage diagnostics.

    Two checks, both cheap and both deterministic:

    * **exact overlap** — the SHA-256 of every held-out document must not appear in train.
    * **n-gram overlap** — word ``n``-grams (character ``n``-grams when a document has too
      few words) of the held-out set that also occur in train. This is a *diagnostic*, not a
      proof: boilerplate, quotations and shared idiom produce genuine overlap. It is here to
      catch accidental duplication, not to certify independence.
    """
    train_digests = {sha256_text(document.text) for document in train}
    exact = [document.doc_id for document in held_out if sha256_text(document.text) in train_digests]

    sample = list(train[:MAX_NGRAM_DOCS])
    train_ngrams: set[str] = set()
    for document in sample:
        train_ngrams |= _ngrams(document.text, n)

    hits: list[str] = []
    for document in held_out:
        if train_ngrams & _ngrams(document.text, n):
            hits.append(document.doc_id)

    total_held_out = len(held_out)
    note = ""
    if not train or not held_out:
        note = (
            "no documents to compare: the corpus is empty because no source was verified. "
            "A clean result here means nothing was checked, not that leakage is impossible."
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "note": note,
        "exact": {
            "method": "sha256(text) set intersection",
            "overlap_count": len(exact),
            "overlap_doc_ids": exact[:20],
            "clean": not exact,
        },
        "ngram": {
            "n": n,
            "unit": "word (character fallback)",
            "train_documents_sampled": len(sample),
            "train_documents_total": len(train),
            "sampled": len(sample) < len(train),
            "heldout_documents_checked": total_held_out,
            "heldout_documents_with_overlap": len(hits),
            "overlap_ratio": round(len(hits) / total_held_out, 6) if total_held_out else 0.0,
            "overlap_doc_ids": hits[:20],
        },
    }


# ---------------------------------------------------------------------------
# statistics and coverage
# ---------------------------------------------------------------------------
def _totals(documents: Iterable[CorpusDocument]) -> dict[str, int]:
    documents = list(documents)
    return {
        "examples": len(documents),
        "chars": sum(document.chars for document in documents),
        "bytes": sum(document.bytes for document in documents),
    }


def language_statistics(
    manifest: TokenizerCorpusManifest,
    documents: Sequence[CorpusDocument],
    ingested: Sequence[IngestedSource],
) -> list[dict[str, Any]]:
    """Per-language statistics for **every** slot, including the empty ones."""
    by_language: dict[str, list[CorpusDocument]] = {slot.code: [] for slot in manifest.language_slots}
    for document in documents:
        by_language.setdefault(document.language, []).append(document)

    ingested_by_language: dict[str, list[IngestedSource]] = {}
    for item in ingested:
        ingested_by_language.setdefault(item.language, []).append(item)

    stats: list[dict[str, Any]] = []
    for slot in manifest.language_slots:
        docs = by_language.get(slot.code, [])
        train, held_out = split_documents(
            docs, seed=manifest.split_seed, held_out_fraction=manifest.held_out_fraction
        )
        items = ingested_by_language.get(slot.code, [])
        verified = [item for item in items if item.verified]
        licences = sorted({item.status for item in items})
        stats.append(
            {
                "language": slot.code,
                "display": slot.display or LANGUAGES.get(slot.code, slot.code),
                "script": slot.script,
                "source_count": len(manifest.sources_for(slot.code)),
                "source_ids": [source.id for source in manifest.sources_for(slot.code)],
                "ingested_count": len(items),
                "verified_count": len(verified),
                "license_ids": sorted(
                    {source.license_id for source in manifest.sources_for(slot.code)}
                ),
                "license_status": "allowed" if _licences_allowed(manifest, slot.code) else "unknown",
                "verification_status": "verified" if verified else "unverified",
                "ingest_statuses": licences,
                "examples": _totals(docs)["examples"],
                "chars": _totals(docs)["chars"],
                "bytes": _totals(docs)["bytes"],
                "train_examples": _totals(train)["examples"],
                "train_chars": _totals(train)["chars"],
                "train_bytes": _totals(train)["bytes"],
                "evaluation_examples": _totals(held_out)["examples"],
                "evaluation_chars": _totals(held_out)["chars"],
                "evaluation_bytes": _totals(held_out)["bytes"],
            }
        )
    return stats


def _licences_allowed(manifest: TokenizerCorpusManifest, code: str) -> bool:
    sources = manifest.sources_for(code)
    return bool(sources) and all(source.license_id in ALLOWED_LICENSES for source in sources)


def coverage_report(
    manifest: TokenizerCorpusManifest,
    statistics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Status of **all** language slots: EVALUATED / INSUFFICIENT / UNVERIFIED / NOT_EVALUATED.

    Nothing is hidden and nothing is imputed: a slot is ``EVALUATED`` only when it has
    verified sources meeting both targets. Missing slots keep the reason recorded in the
    manifest.
    """
    by_language = {row["language"]: row for row in statistics}
    rows: list[dict[str, Any]] = []
    for slot in manifest.language_slots:
        row = by_language.get(slot.code, {})
        declared = manifest.sources_for(slot.code)
        verified_count = int(row.get("verified_count", 0))
        chars = int(row.get("chars", 0))
        examples = int(row.get("examples", 0))
        target_chars = int(slot.target_chars or TARGET_CHARS)
        target_sentences = int(slot.target_sentences or TARGET_SENTENCES)

        if verified_count and chars >= target_chars and examples >= target_sentences:
            status = STATUS_EVALUATED
            reason = ""
        elif verified_count:
            status = STATUS_INSUFFICIENT
            reason = (
                f"verified data is below target: {examples} examples / {chars:,} characters "
                f"vs {target_sentences} examples / {target_chars:,} characters"
            )
        elif declared:
            status = STATUS_UNVERIFIED
            detail = ", ".join(sorted(set(row.get("ingest_statuses") or ["not_attempted"])))
            reason = f"sources declared but not verified (ingest status: {detail})"
        else:
            status = STATUS_NOT_EVALUATED
            reason = slot.reason or "no source declared"

        rows.append(
            {
                "language": slot.code,
                "display": slot.display or LANGUAGES.get(slot.code, slot.code),
                "script": slot.script,
                "status": status,
                "reason": reason,
                "target_sentences": target_sentences,
                "target_chars": target_chars,
                "actual_sentences": examples,
                "actual_chars": chars,
                "actual_bytes": int(row.get("bytes", 0)),
                "percent_of_target_chars": round(100.0 * chars / target_chars, 2) if target_chars else 0.0,
                "sufficient": status == STATUS_EVALUATED,
                "declared_sources": [source.id for source in declared],
                "candidate_providers": [
                    candidate.get("provider", "") for candidate in slot.candidates
                ],
            }
        )

    counts = {status: 0 for status in (STATUS_EVALUATED, STATUS_INSUFFICIENT, STATUS_UNVERIFIED,
                                       STATUS_NOT_EVALUATED)}
    for row in rows:
        counts[row["status"]] += 1
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus_id": manifest.corpus_id,
        "corpus_version": manifest.corpus_version,
        "targets": {
            "sentences_per_language": int(manifest.targets.get("sentences_per_language",
                                                                TARGET_SENTENCES)),
            "characters_per_language": int(manifest.targets.get("characters_per_language",
                                                                TARGET_CHARS)),
        },
        "languages": rows,
        "summary": {
            "slots": len(rows),
            **counts,
            "evaluated": counts[STATUS_EVALUATED],
        },
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
@dataclass
class BuildResult:
    """Everything a build produced, for the CLI and the experiment record."""

    out_dir: Path
    manifest: TokenizerCorpusManifest
    ingested: list[IngestedSource]
    train: list[CorpusDocument]
    held_out: list[CorpusDocument]
    statistics: list[dict[str, Any]]
    coverage: dict[str, Any]
    leakage: dict[str, Any]
    files: dict[str, dict[str, Any]]

    @property
    def verified_sources(self) -> int:
        return sum(1 for item in self.ingested if item.verified)

    def results_payload(self) -> dict[str, Any]:
        """Flat, numeric summary for an experiment record (`results`)."""
        summary = self.coverage["summary"]
        return {
            "corpus_id": self.manifest.corpus_id,
            "corpus_version": self.manifest.corpus_version,
            "sources_declared": len(self.manifest.sources),
            "sources_verified": self.verified_sources,
            "language_slots": summary["slots"],
            "slots_evaluated": summary[STATUS_EVALUATED],
            "slots_insufficient": summary[STATUS_INSUFFICIENT],
            "slots_unverified": summary[STATUS_UNVERIFIED],
            "slots_not_evaluated": summary[STATUS_NOT_EVALUATED],
            "documents": len(self.train) + len(self.held_out),
            "train_examples": len(self.train),
            "evaluation_examples": len(self.held_out),
            "train_chars": sum(d.chars for d in self.train),
            "evaluation_chars": sum(d.chars for d in self.held_out),
            "exact_overlap_count": self.leakage["exact"]["overlap_count"],
            "ngram_overlap_ratio": self.leakage["ngram"]["overlap_ratio"],
        }


def build_corpus(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    seed: int | None = None,
    held_out_fraction: float | None = None,
    fetch: bool = False,
    source_ids: Sequence[str] | None = None,
    include_unverified: bool = False,
    timeout: float = 30.0,
    pin: bool = False,
) -> BuildResult:
    """Ingest the declared sources, split them, and write the corpus artifacts.

    Files written under ``out_dir``:

    * ``corpus.json`` — the built corpus manifest (ids, versions, policies, split, sources,
      file hashes). Carries ``built_at``, so it is *not* byte-reproducible by design.
    * ``train.jsonl`` / ``heldout.jsonl`` — the documents.
    * ``stats.json``, ``coverage.json``, ``leakage.json`` — deterministic reports.
    * ``sources/<id>.txt`` and ``sources/<id>.provenance.json`` — the ingested text.

    ``include_unverified`` (default ``False``) is the safety valve: unverified text — a
    local fixture, or a file a human supplied — is never part of the research corpus unless
    a caller explicitly asks for it, and even then it cannot make a slot EVALUATED.
    """
    manifest = TokenizerCorpusManifest.load(manifest_path)
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("invalid tokenizer corpus manifest:\n  - " + "\n  - ".join(problems))

    out_dir = Path(out_dir)
    raw_dir = out_dir / "sources"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(manifest.sources)
    if source_ids:
        wanted = set(source_ids)
        unknown = sorted(wanted - {source.id for source in selected})
        if unknown:
            raise KeyError(f"no such source id(s) in the manifest: {unknown}")
        selected = [source for source in selected if source.id in wanted]

    ingested: list[IngestedSource] = []
    documents: list[CorpusDocument] = []
    for source in selected:
        item = ingest_source(source, raw_dir, fetch=fetch, timeout=timeout)
        if item.has_text:
            docs = documents_from_text(source.id, source.language,
                                       Path(item.path).read_text(encoding="utf-8"))
            item.documents = len(docs)
            if item.verified or include_unverified:
                documents.extend(docs)
        ingested.append(item)

    documents.sort(key=lambda document: (document.language, document.doc_id))
    train, held_out = split_documents(
        documents,
        seed=manifest.split_seed if seed is None else seed,
        held_out_fraction=manifest.held_out_fraction if held_out_fraction is None else held_out_fraction,
    )

    statistics = language_statistics(manifest, train + held_out, ingested)
    coverage = coverage_report(manifest, statistics)
    leakage = leakage_report(train, held_out)

    files: dict[str, dict[str, Any]] = {}
    files[TRAIN_FILENAME] = _write_jsonl(out_dir / TRAIN_FILENAME, train)
    files[HELD_OUT_FILENAME] = _write_jsonl(out_dir / HELD_OUT_FILENAME, held_out)
    files[STATS_FILENAME] = _write_json(
        out_dir / STATS_FILENAME,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_id": manifest.corpus_id,
            "corpus_version": manifest.corpus_version,
            "split": {
                "method": str(manifest.split.get("method", "deterministic-hash")),
                "seed": manifest.split_seed if seed is None else seed,
                "heldout_fraction": manifest.held_out_fraction if held_out_fraction is None
                else held_out_fraction,
                "level": "document",
            },
            "normalization_policy": manifest.normalization_policy,
            "languages": statistics,
            "totals": {
                "examples": len(train) + len(held_out),
                "train_examples": len(train),
                "evaluation_examples": len(held_out),
                "chars": sum(document.chars for document in train + held_out),
                "bytes": sum(document.bytes for document in train + held_out),
            },
        },
    )
    files[COVERAGE_FILENAME] = _write_json(out_dir / COVERAGE_FILENAME, coverage)
    files[LEAKAGE_FILENAME] = _write_json(out_dir / LEAKAGE_FILENAME, leakage)

    if pin:
        # CorpusSource is immutable, so pinning rebuilds the entry. Only genuinely
        # verified sources are touched: an unverified one keeps verified=false.
        pinned_ids = {
            item.source_id: item for item in ingested if item.verified and item.sha256
        }
        manifest.sources = [
            replace(
                source,
                sha256=pinned_ids[source.id].sha256,
                verified=True,
                retrieved_at=pinned_ids[source.id].retrieved_at,
            )
            if source.id in pinned_ids
            else source
            for source in manifest.sources
        ]
        manifest.save(manifest_path)

    corpus_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus_id": manifest.corpus_id,
        "corpus_version": manifest.corpus_version,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(Path(manifest_path)),
        "manifest_sha256": sha256_file(manifest_path),
        "language_slots": [slot.to_dict() for slot in manifest.language_slots],
        "sources": [
            {
                **source.to_dict(),
                "ingest": next(
                    (item.to_dict() for item in ingested if item.source_id == source.id),
                    {"status": "not_attempted"},
                ),
            }
            for source in selected
        ],
        "split": {
            "method": str(manifest.split.get("method", "deterministic-hash")),
            "seed": manifest.split_seed if seed is None else seed,
            "heldout_fraction": manifest.held_out_fraction if held_out_fraction is None
            else held_out_fraction,
            "level": "document",
            "train_documents": len(train),
            "heldout_documents": len(held_out),
        },
        "normalization_policy": manifest.normalization_policy,
        "cleaning_policy": manifest.cleaning_policy,
        "include_unverified": include_unverified,
        "files": files,
        "totals": {
            "documents": len(train) + len(held_out),
            "train_documents": len(train),
            "heldout_documents": len(held_out),
            "chars": sum(document.chars for document in train + held_out),
            "bytes": sum(document.bytes for document in train + held_out),
        },
    }
    files[CORPUS_FILENAME] = _write_json(out_dir / CORPUS_FILENAME, corpus_payload)

    return BuildResult(
        out_dir=out_dir,
        manifest=manifest,
        ingested=ingested,
        train=train,
        held_out=held_out,
        statistics=statistics,
        coverage=coverage,
        leakage=leakage,
        files=files,
    )


def load_corpus(path: str | Path) -> dict[str, Any]:
    """Load ``corpus.json`` and verify that the files it names still hash to what it says."""
    corpus_path = Path(path)
    if corpus_path.is_dir():
        corpus_path = corpus_path / CORPUS_FILENAME
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported corpus schema {version!r} (expected {MANIFEST_SCHEMA_VERSION!r})"
        )
    for name, meta in (payload.get("files") or {}).items():
        target = corpus_path.parent / name
        if not target.exists():
            raise FileNotFoundError(f"corpus file listed in {corpus_path.name} is missing: {name}")
        if meta.get("sha256") and sha256_file(target) != meta["sha256"]:
            raise ValueError(f"corpus file {name} does not match its recorded sha256")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_jsonl(path: Path, documents: Sequence[CorpusDocument]) -> dict[str, Any]:
    body = "".join(
        json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for document in documents
    )
    path.write_text(body, encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
