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
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..data.corpora import (
    ALLOWED_LICENSES,
    CorpusSource,
    FetchError,
    build_provenance,
    clean_text,
    fetch_text,
    license_marker_found,
    sha256_file,
    sha256_text,
    trim_text,
)
from .corpus import LANGUAGES

MANIFEST_SCHEMA_VERSION = "1.0"
CORPUS_ID = "indic-tokenizer"
CORPUS_VERSION = "v2"

# How the licence of a source may be proven *in addition to* a marker inside the text
# itself. Some hosts never put their licence in the payload we want: a Wikisource
# `?action=raw` fetch is bare wikitext, while the CC BY-SA notice lives in the rendered
# page. Rather than weaken the licence check, a source may declare an auditable evidence
# endpoint that is fetched, searched and *recorded*.
EVIDENCE_KINDS = {"mediawiki-api", "html", "plain"}
EVIDENCE_SCOPES = {"site", "page"}

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

# Preflight reads only a bounded prefix: enough to see what a page is, without
# downloading a whole book just to ask whether the host is reachable.
PREFLIGHT_SAMPLE_BYTES = 65_536
SAMPLING_RULE = (
    "stratified: an equal per-language budget of limit // n_languages documents "
    "in (language, doc_id) order, then any leftover budget filled in the same order"
)

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
# licence evidence
# ---------------------------------------------------------------------------
@dataclass
class LicenseEvidence:
    """An auditable, separately fetched proof of a source's licence.

    ``url`` is fetched over https, and ``marker`` must occur in what comes back
    (case-insensitive). ``marker`` defaults to the source's ``license_url``, so the
    usual check is "the host declares this exact licence". ``kind`` selects how the
    payload is read:

    * ``mediawiki-api`` — parse JSON, search the serialized document (this is how a
      wiki publishes its content licence: ``action=query&meta=siteinfo&siprop=rightsinfo``);
    * ``html`` — search the fetched page (for a rendered licence footer);
    * ``plain`` — search the fetched text.

    ``scope`` records what the evidence covers: ``page`` (this work only) or ``site``
    (the host's default content licence). Site-level evidence is weaker and is labelled
    as such everywhere it is recorded — it never silently upgrades to page-level proof.
    """

    url: str = ""
    kind: str = "mediawiki-api"
    marker: str = ""
    scope: str = "site"
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LicenseEvidence:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown keys in licence evidence: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_marker(self, source: CorpusSource) -> str:
        return self.marker or source.license_url


def check_license_evidence(
    source: CorpusSource,
    evidence: LicenseEvidence,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch and evaluate the declared licence evidence. Never raises on bad evidence.

    The result is recorded verbatim: what was fetched, what marker was looked for, and
    whether it was found. A failure of any kind returns ``accepted: False`` — an
    unreachable or malformed endpoint can never verify a source.
    """
    marker = evidence.effective_marker(source)
    result: dict[str, Any] = {
        "url": evidence.url,
        "kind": evidence.kind,
        "scope": evidence.scope,
        "marker": marker,
        "note": evidence.note,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "not_checked",
        "marker_found": False,
        "accepted": False,
        "error": "",
    }
    if not evidence.url.startswith("https://"):
        result["status"] = "invalid_url"
        result["error"] = "licence evidence url must be https"
        return result

    try:
        payload = fetch_text(evidence.url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - an evidence fetch can fail in many ways
        result["status"] = "fetch_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if evidence.kind == "mediawiki-api":
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            result["status"] = "malformed"
            result["error"] = f"evidence payload is not JSON: {exc}"
            return result
        haystack = json.dumps(document, ensure_ascii=False).lower()
    else:  # html | plain: search the payload as text
        haystack = payload.lower()

    result["marker_found"] = marker.lower() in haystack
    result["status"] = "ok" if result["marker_found"] else "marker_not_found"
    result["accepted"] = bool(result["marker_found"])
    return result


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

    def to_dict(self, *, omit_empty_optional: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if omit_empty_optional:
            # Writing the manifest back must not invent keys that were not there:
            # an empty candidates/reason pair is noise in a tracked file.
            if not payload.get("candidates"):
                payload.pop("candidates", None)
            if not payload.get("reason"):
                payload.pop("reason", None)
        return payload


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
    # source id -> licence evidence. Kept beside the sources because CorpusSource is the
    # shared smoke-corpus model and must not grow tokenizer-only fields.
    license_evidence: dict[str, LicenseEvidence] = field(default_factory=dict)
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
        evidence: dict[str, LicenseEvidence] = {}
        sources: list[CorpusSource] = []
        for item in data.get("sources", []):
            entry = dict(item)
            raw_evidence = entry.pop("license_evidence", None)
            source = CorpusSource.from_dict(entry)
            sources.append(source)
            if raw_evidence:
                evidence[source.id] = LicenseEvidence.from_dict(raw_evidence)
        return cls(
            schema_version=version,
            corpus=dict(data.get("corpus") or {}),
            targets=dict(data.get("targets") or {}),
            split=dict(data.get("split") or {}),
            normalization_policy=str(data.get("normalization_policy", "none")),
            cleaning_policy=str(data.get("cleaning_policy", "")),
            language_slots=[LanguageSlot.from_dict(item) for item in data.get("language_slots", [])],
            sources=sources,
            license_evidence=evidence,
            notes=str(data.get("notes", "")),
        )

    def to_dict(self, *, omit_empty_optional: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus": self.corpus,
            "targets": self.targets,
            "split": self.split,
            "normalization_policy": self.normalization_policy,
            "cleaning_policy": self.cleaning_policy,
            "language_slots": [
                slot.to_dict(omit_empty_optional=omit_empty_optional) for slot in self.language_slots
            ],
            "sources": [self._source_to_dict(source, omit_empty_optional) for source in self.sources],
            "notes": self.notes,
        }

    def _source_to_dict(self, source: CorpusSource, omit_empty_optional: bool) -> dict[str, Any]:
        payload = source.to_dict()
        evidence = self.license_evidence.get(source.id)
        if evidence is not None:
            payload["license_evidence"] = evidence.to_dict()
        return payload

    def save(self, path: str | Path, *, omit_empty_optional: bool = True) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(omit_empty_optional=omit_empty_optional), indent=2, ensure_ascii=False
        ) + "\n"
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)

    def evidence_for(self, source_id: str) -> LicenseEvidence | None:
        return self.license_evidence.get(source_id)

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

    seen_ids: set[str] = set()
    for source in manifest.sources:
        if source.id in seen_ids:
            problems.append(f"duplicate source id {source.id!r}")
        seen_ids.add(source.id)
        if source.language not in known_slots:
            problems.append(f"source {source.id!r}: language {source.language!r} has no language slot")
        problems.extend(_validate_source(source))

    # A slot that references a source which does not exist would silently become
    # NOT_EVALUATED, and an orphan source would never contribute to any slot. Both are
    # configuration mistakes that must be visible rather than quiet.
    declared_ids = seen_ids
    referenced: set[str] = set()
    for slot in manifest.language_slots:
        for source_id in slot.sources:
            if source_id not in declared_ids:
                problems.append(
                    f"language slot {slot.code!r} references unknown source id {source_id!r}"
                )
            else:
                referenced.add(source_id)
    for source_id in sorted(declared_ids - referenced):
        problems.append(f"source {source_id!r} is not referenced by any language slot")

    for source_id, evidence in manifest.license_evidence.items():
        if source_id not in declared_ids:
            problems.append(f"licence evidence declared for unknown source id {source_id!r}")
            continue
        if not evidence.url.startswith("https://"):
            problems.append(f"{source_id}: licence evidence url must be https")
        if evidence.kind not in EVIDENCE_KINDS:
            problems.append(f"{source_id}: licence evidence kind must be one of {sorted(EVIDENCE_KINDS)}")
        if evidence.scope not in EVIDENCE_SCOPES:
            problems.append(f"{source_id}: licence evidence scope must be one of {sorted(EVIDENCE_SCOPES)}")
        if not evidence.effective_marker(manifest.source(source_id)):
            problems.append(f"{source_id}: licence evidence needs a marker or a license_url to look for")
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
    # Was the cleaned source longer than max_chars? A truncated work is still usable,
    # but a reader must be able to see that it is partial.
    truncated: bool = False
    chars_before_trim: int = 0
    # Why the licence was accepted: "payload-marker", "licence-evidence" or "" when the
    # source was not verified.
    licence_proof: str = ""
    # The recorded result of the licence-evidence fetch, verbatim (None when not used).
    license_evidence: dict[str, Any] | None = None
    # Locally supplied text: a content hash is computed, but it proves nothing about
    # licensing, so the source stays local_unverified and can never make a slot
    # EVALUATED.
    local: bool = False
    local_origin: str | None = None

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


def _prepare_source_text(raw: str, source: CorpusSource) -> tuple[str, bool, int]:
    """``prepare_source_text`` plus the two facts a researcher needs about trimming.

    Returns ``(text, truncated, chars_before_trim)``. This is the same clean-then-trim
    composition as :func:`~frontier_ai.data.corpora.prepare_source_text`, only reporting
    whether the trim actually removed anything.
    """
    cleaned = clean_text(raw, source.kind)
    prepared = trim_text(cleaned, source.max_chars)
    return prepared, len(prepared) < len(cleaned), len(cleaned)


def ingest_source(
    source: CorpusSource,
    raw_dir: str | Path,
    *,
    fetch: bool = False,
    timeout: float = 30.0,
    local_text: str | None = None,
    local_origin: str | None = None,
    license_evidence: LicenseEvidence | None = None,
) -> IngestedSource:
    """Fetch (or accept) one source, clean it, and write text + provenance.

    A source becomes ``verified`` **only** when it produced a non-empty cleaned text and
    its licence was proven by one of two explicit, recorded mechanisms:

    * **payload marker** — the fetched payload itself carries a marker for the licence
      the manifest claims (``licence_proof: "payload-marker"``); or
    * **licence evidence** — a separately fetched, declared evidence endpoint proves the
      licence (``licence_proof: "licence-evidence"``), with the full result of that check
      recorded in ``license_evidence``.

    If neither succeeds the source is ``licence_marker_missing`` and no hash is pinned.
    Evidence cannot be bypassed, and a failed evidence fetch is a failure, never a
    fallback to trust.

    Locally supplied text (test fixtures, or text a human lawfully obtained) is ingested
    as ``local_unverified``: its hash is a content hash only and proves nothing about
    licensing, so it cannot make a language slot count as evaluated.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    evidence_result: dict[str, Any] | None = None
    licence_proof = ""  # set only when a licence marker or evidence endpoint proves it
    if local_text is not None:
        text, truncated, before_trim = _prepare_source_text(local_text, source)
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

        if license_marker_found(raw, source):
            licence_proof = "payload-marker"
        elif license_evidence is not None:
            evidence_result = check_license_evidence(source, license_evidence, timeout=timeout)
            licence_proof = "licence-evidence" if evidence_result["accepted"] else ""
        else:
            licence_proof = ""

        if not licence_proof:
            checked = (
                f"the licence evidence at {license_evidence.url} did not confirm it "
                f"({evidence_result['status']})" if evidence_result else
                "no licence evidence endpoint is declared for this source"
            )
            return IngestedSource(
                source_id=source.id,
                language=source.language,
                status="licence_marker_missing",
                retrieved_at=retrieved_at,
                license_evidence=evidence_result,
                error=(
                    f"no {source.license_id} marker found in the first 50,000 characters of "
                    f"{source.source_url}, and {checked}; refusing to pin a hash"
                ),
            )
        text, truncated, before_trim = _prepare_source_text(raw, source)
        status = "verified"
    else:
        return IngestedSource(source_id=source.id, language=source.language, status="not_attempted")

    if not text.strip():
        return IngestedSource(
            source_id=source.id,
            language=source.language,
            status="empty",
            retrieved_at=retrieved_at,
            license_evidence=evidence_result,
            error="cleaned text is empty",
        )

    text_path = raw_dir / f"{source.id}.txt"
    text_path.write_text(text, encoding="utf-8")
    provenance_path = raw_dir / f"{source.id}.provenance.json"
    _write_provenance(
        provenance_path,
        source,
        text,
        retrieved_at=retrieved_at,
        truncated=truncated,
        chars_before_trim=before_trim,
        licence_proof=licence_proof,
        license_evidence=evidence_result,
        local=local_text is not None,
        local_origin=local_origin,
    )

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
        truncated=truncated,
        chars_before_trim=before_trim,
        licence_proof=licence_proof,
        license_evidence=evidence_result,
        local=local_text is not None,
        local_origin=local_origin if local_text is not None else None,
    )


def _write_provenance(
    path: str | Path,
    source: CorpusSource,
    text: str,
    *,
    retrieved_at: str,
    truncated: bool,
    chars_before_trim: int,
    licence_proof: str,
    license_evidence: dict[str, Any] | None,
    local: bool,
    local_origin: str | None,
) -> dict[str, Any]:
    """The shared provenance record, extended (never weakened) with Stage A fields."""
    record = build_provenance(source, text, retrieved_at=retrieved_at)
    record.update(
        {
            "verification": "local_unverified" if local else ("verified" if licence_proof else "unverified"),
            "local_source": local,
            "local_origin_path": local_origin,
            "hash_is_licence_proof": False,  # a hash identifies bytes; it never proves a licence
            "licence_proof": licence_proof or "none",
            "license_evidence": license_evidence,
            "truncated": truncated,
            "chars_before_trim": chars_before_trim,
            "max_chars": source.max_chars,
        }
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


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

    @property
    def text_sha256(self) -> str:
        """Content identity: the key the split uses so duplicates cannot straddle it."""
        return sha256_text(self.text)

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

    The bucket of a document is ``sha256('<seed>:<content_sha256>')`` mapped to
    ``[0, 1]``, where ``content_sha256`` is the SHA-256 of the document text. Bucketing on
    **content** rather than on ``doc_id`` is what guarantees the property later stages
    rely on: byte-identical text cannot appear on both sides, even when a source repeats
    a paragraph (refrains, chapter headers, boilerplate). Two copies of the same text
    hash to the same bucket, so they go to the same side.

    There is no global RNG and no dependence on document order, so the same corpus and
    seed always produce the same split, and changing the seed re-splits reproducibly.
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError(f"held_out_fraction must be in (0, 1), got {held_out_fraction!r}")
    train: list[CorpusDocument] = []
    held_out: list[CorpusDocument] = []
    for document in documents:
        digest = hashlib.sha256(f"{seed}:{document.text_sha256}".encode()).hexdigest()
        bucket = int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF
        (held_out if bucket < held_out_fraction else train).append(document)
    return train, held_out


def _sample_documents(documents: Sequence[CorpusDocument], limit: int) -> list[CorpusDocument]:
    """Deterministic, language-stratified sample of at most ``limit`` documents.

    Rule (recorded in the report as ``sampling_rule``): order documents by
    ``(language, doc_id)``, give every language present an equal budget of
    ``limit // n_languages`` documents, then — if any budget is left because some languages
    had fewer documents — fill from the same deterministic order. No RNG, no dependence on
    the caller's ordering, and every language contributes whenever data exists.
    """
    if limit <= 0 or not documents:
        return []
    ordered = sorted(documents, key=lambda document: (document.language, document.doc_id))
    if len(ordered) <= limit:
        return ordered

    by_language: dict[str, list[CorpusDocument]] = {}
    for document in ordered:
        by_language.setdefault(document.language, []).append(document)

    per_language = max(1, limit // len(by_language))
    sample: list[CorpusDocument] = []
    seen: set[str] = set()
    for language in sorted(by_language):
        for document in by_language[language][:per_language]:
            sample.append(document)
            seen.add(document.doc_id)
    for document in ordered:  # spend any budget a short language left behind
        if len(sample) >= limit:
            break
        if document.doc_id not in seen:
            sample.append(document)
            seen.add(document.doc_id)
    return sample


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
      The split already makes this impossible; the check is here to catch a regression in
      that guarantee (and any other path that assembles the two files).
    * **n-gram overlap** — word ``n``-grams (character ``n``-grams when a document has too
      few words) of the held-out set that also occur in train. The train side is sampled
      **stratified by language** (see :func:`_sample_documents`) so that every language in
      the corpus contributes, instead of only whichever language sorts first. This is a
      *diagnostic*, not a proof: boilerplate, quotations and shared idiom produce genuine
      overlap. It is here to catch accidental duplication, not to certify independence.
    """
    train_digests = {document.text_sha256 for document in train}
    exact = [document.doc_id for document in held_out if document.text_sha256 in train_digests]

    sample = _sample_documents(train, MAX_NGRAM_DOCS)
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
            "sampling_rule": SAMPLING_RULE,
            "train_documents_sampled": len(sample),
            "train_documents_total": len(train),
            "sampled": len(sample) < len(train),
            "per_language_sampled": dict(sorted(Counter(d.language for d in sample).items())),
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
    train: Sequence[CorpusDocument],
    held_out: Sequence[CorpusDocument],
    ingested: Sequence[IngestedSource],
) -> list[dict[str, Any]]:
    """Per-language statistics for **every** slot, including the empty ones.

    The counts come from the *already split* document lists, never from re-splitting. An
    earlier version recomputed the split from the manifest seed, which silently disagreed
    with the written artifacts whenever a caller overrode the seed. Statistics that cannot
    disagree with the data are the only kind worth publishing.
    """
    by_language_train: dict[str, list[CorpusDocument]] = {
        slot.code: [] for slot in manifest.language_slots
    }
    by_language_held: dict[str, list[CorpusDocument]] = {
        slot.code: [] for slot in manifest.language_slots
    }
    for document in train:
        by_language_train.setdefault(document.language, []).append(document)
    for document in held_out:
        by_language_held.setdefault(document.language, []).append(document)

    ingested_by_language: dict[str, list[IngestedSource]] = {}
    for item in ingested:
        ingested_by_language.setdefault(item.language, []).append(item)

    stats: list[dict[str, Any]] = []
    for slot in manifest.language_slots:
        docs_train = by_language_train.get(slot.code, [])
        docs_held_out = by_language_held.get(slot.code, [])
        docs = list(docs_train) + list(docs_held_out)
        train, held_out = docs_train, docs_held_out
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
                "unique_texts": len({document.text_sha256 for document in docs}),
                "duplicate_documents": len(docs) - len({document.text_sha256 for document in docs}),
                "truncated_sources": sum(1 for item in items if item.truncated),
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
    local_texts: Mapping[str, str] | None = None,
    local_origins: Mapping[str, str] | None = None,
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

    ``local_texts`` maps a source id to text supplied from disk (``--local-file`` /
    ``--local-dir``). Local text is ingested as ``local_unverified``: its hash identifies
    the bytes and proves nothing about licensing.
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
        local_text = (local_texts or {}).get(source.id)
        item = ingest_source(
            source,
            raw_dir,
            fetch=fetch and local_text is None,
            timeout=timeout,
            local_text=local_text,
            local_origin=(local_origins or {}).get(source.id),
            license_evidence=manifest.evidence_for(source.id),
        )
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

    statistics = language_statistics(manifest, train, held_out, ingested)
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
        if pinned_ids:  # nothing pinned => the tracked manifest is not touched at all
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


# ---------------------------------------------------------------------------
# preflight (read-only: no acquisition, no verification, no pinning)
# ---------------------------------------------------------------------------
def probe_url(
    url: str,
    *,
    max_bytes: int = PREFLIGHT_SAMPLE_BYTES,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Bounded, read-only GET used **only** by preflight.

    Why a second network call exists next to ``fetch_text``: preflight must not download a
    whole book just to ask "is this reachable, and does it look like the work?". This asks
    for a byte range and reads at most ``max_bytes`` from the stream even if the server
    ignores the range.

    It writes nothing, hashes nothing, verifies nothing and never feeds the corpus. The
    sample is decoded with ``errors="replace"`` because a range cut can land mid-character;
    that only affects the heuristics below, never any stored text.
    """
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "frontier-ai-tokenizer-corpus/1.0",
            "Range": f"bytes=0-{max_bytes - 1}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None) or response.getcode()
            content_type = response.headers.get("Content-Type", "")
            body = response.read(max_bytes)
    except Exception as exc:  # noqa: BLE001 - connectivity fails in many ways
        return {
            "url": url,
            "ok": False,
            "http_status": None,
            "content_type": "",
            "bytes_read": 0,
            "elapsed_s": round(time.monotonic() - started, 3),
            "text": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "url": url,
        "ok": True,
        "http_status": status,
        "content_type": content_type,
        "bytes_read": len(body),
        "elapsed_s": round(time.monotonic() - started, 3),
        "text": body.decode("utf-8", errors="replace"),
        "error": "",
    }


def _link_line_ratio(lines: list[str]) -> tuple[float, int]:
    """Share of non-empty lines that are links/templates with almost no prose.

    A Wikisource work root is often just a contents page: lines such as
    ``[[गोदान/अध्याय १|अध्याय १]]``. Those lines shrink to almost nothing once the
    markup is removed, which is the signal preflight reports.
    """
    if not lines:
        return 0.0, 0
    link_lines = 0
    for line in lines:
        stripped = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", line)
        stripped = re.sub(r"\{\{[^{}]*\}\}", "", stripped).strip()
        if len(stripped) < 20:
            link_lines += 1
    return round(link_lines / len(lines), 3), link_lines


def _sample_shape(text: str, kind: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ratio, link_lines = _link_line_ratio(lines)
    return {
        "sample_chars": len(text),
        "sample_lines": len(lines),
        "wiki_links": len(re.findall(r"\[\[[^\]]+\]\]", text)),
        "wiki_templates": len(re.findall(r"\{\{[^{}]*\}\}", text)),
        "link_line_ratio": ratio,
        "link_lines": link_lines,
        # Conservative hint, not a verdict: a human still has to look.
        "looks_like_index_page": kind == "wikitext" and bool(lines) and ratio >= 0.5,
        "gutenberg_marker_seen": "*** start of the project gutenberg ebook" in text.lower(),
    }


def preflight_evidence(
    source: CorpusSource,
    evidence: LicenseEvidence,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Probe a declared licence-evidence endpoint. Indicative only — see the note."""
    probe = probe_url(evidence.url, timeout=timeout)
    marker = evidence.effective_marker(source)
    result: dict[str, Any] = {
        "url": evidence.url,
        "kind": evidence.kind,
        "scope": evidence.scope,
        "marker": marker,
        "ok": probe["ok"],
        "http_status": probe["http_status"],
        "bytes_read": probe["bytes_read"],
        "elapsed_s": probe["elapsed_s"],
        "marker_seen_in_sample": False,
        "error": probe["error"],
    }
    if probe["ok"]:
        haystack = probe["text"].lower()
        result["marker_seen_in_sample"] = marker.lower() in haystack
        if evidence.kind == "mediawiki-api":
            try:
                json.loads(probe["text"])
            except (json.JSONDecodeError, ValueError):
                result["error"] = "payload is not valid JSON (truncated sample?)"
    result["note"] = (
        "preflight is indicative only: it never verifies a source, never writes a file and "
        "never pins a hash. Verification happens only during a real build with --fetch."
    )
    return result


def preflight_source(
    source: CorpusSource,
    evidence: LicenseEvidence | None = None,
    *,
    timeout: float = 30.0,
    max_bytes: int = PREFLIGHT_SAMPLE_BYTES,
) -> dict[str, Any]:
    """Read-only reachability/shape probe for one declared source (nothing is stored)."""
    probe = probe_url(source.source_url, max_bytes=max_bytes, timeout=timeout)
    report: dict[str, Any] = {
        "source_id": source.id,
        "language": source.language,
        "kind": source.kind,
        "license_id": source.license_id,
        "url": source.source_url,
        "ok": probe["ok"],
        "http_status": probe["http_status"],
        "content_type": probe["content_type"],
        "bytes_read": probe["bytes_read"],
        "elapsed_s": probe["elapsed_s"],
        "error": probe["error"],
        "sample_bytes_cap": max_bytes,
        "truncated_sample": probe["bytes_read"] >= max_bytes,
        "payload_licence_marker_seen": False,
        "notes": [],
    }
    if probe["ok"]:
        report.update(_sample_shape(probe["text"], source.kind))
        report["payload_licence_marker_seen"] = license_marker_found(probe["text"], source)
        if source.kind == "gutenberg" and not report["gutenberg_marker_seen"]:
            report["notes"].append(
                "no Gutenberg START marker in the sampled bytes: this URL may not be a "
                "Project Gutenberg text file"
            )
        if report["looks_like_index_page"]:
            report["notes"].append(
                "the sampled lines are mostly wiki links: this looks like a contents/index "
                "page, not the work itself. Acquire the chapter subpages or use the API "
                "instead of trusting the work root"
            )
        if not report["payload_licence_marker_seen"] and evidence is None:
            report["notes"].append(
                "no licence marker in the sampled bytes and no evidence endpoint declared: "
                "a real build would refuse to verify this source"
            )
    else:
        report["notes"].append(f"unreachable: {probe['error']}")
    report["evidence"] = preflight_evidence(source, evidence, timeout=timeout) if evidence else None
    return report


def preflight_manifest(
    manifest: TokenizerCorpusManifest,
    *,
    timeout: float = 30.0,
    max_bytes: int = PREFLIGHT_SAMPLE_BYTES,
    source_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Probe every declared source (and its evidence endpoint). Writes nothing."""
    selected = [source for source in manifest.sources if not source_ids or source.id in source_ids]
    endpoints = 0
    reachable = 0
    rows = []
    for source in selected:
        row = preflight_source(
            source, manifest.evidence_for(source.id), timeout=timeout, max_bytes=max_bytes
        )
        endpoints += 1
        reachable += 1 if row["ok"] else 0
        if row["evidence"] is not None:
            endpoints += 1
            reachable += 1 if row["evidence"]["ok"] else 0
        rows.append(row)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus_id": manifest.corpus_id,
        "corpus_version": manifest.corpus_version,
        "manifest_path": None,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_bytes_cap": max_bytes,
        "sources": rows,
        "summary": {
            "sources": len(selected),
            "endpoints": endpoints,
            "reachable": reachable,
            "unreachable": endpoints - reachable,
        },
        "note": (
            "Read-only preflight. No text was stored, no source was verified, no hash was "
            "computed for the corpus and no manifest entry was changed."
        ),
    }


def render_preflight(report: dict[str, Any]) -> str:
    """Human-readable preflight table (one line per endpoint)."""
    lines = [
        f"[preflight] {report['corpus_id']}/{report['corpus_version']} — read-only probe "
        f"(first {report['sample_bytes_cap']:,} bytes per endpoint)",
        "[preflight] nothing is stored, verified or pinned by this command",
    ]
    for row in report["sources"]:
        status = f"OK  http={row['http_status']}" if row["ok"] else "FAIL"
        detail = (
            f"lines={row.get('sample_lines', 0)} links={row.get('wiki_links', 0)} "
            f"licence_marker={'yes' if row.get('payload_licence_marker_seen') else 'no'}"
            if row["ok"]
            else str(row["error"])[:70]
        )
        lines.append(f"[preflight]   {row['source_id']:<30} {status:<14} {detail}")
        for note in row.get("notes") or []:
            lines.append(f"[preflight]       note: {note}")
        evidence = row.get("evidence")
        if evidence:
            marker = "marker seen" if evidence["marker_seen_in_sample"] else "marker NOT seen"
            state = f"OK  {marker}" if evidence["ok"] else f"FAIL {str(evidence['error'])[:50]}"
            lines.append(f"[preflight]     evidence ({evidence['scope']}): {state}")
        elif evidence is None:
            lines.append("[preflight]     evidence: none declared (payload marker required)")
    summary = report["summary"]
    lines.append(
        f"[preflight] summary: {summary['endpoints']} endpoints, "
        f"{summary['reachable']} reachable, {summary['unreachable']} unreachable"
    )
    return "\n".join(lines)


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
