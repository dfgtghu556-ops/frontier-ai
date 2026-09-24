#!/usr/bin/env python3
"""Inspect the text a corpus build actually acquired — before anything is pinned.

Examples (run from the repository root, after a ``--fetch`` build)::

    python scripts/inspect_corpus_sources.py
    python scripts/inspect_corpus_sources.py --show hi-wikisource-godaan-ch23-ccbysa
    python scripts/inspect_corpus_sources.py --output inspection.txt   # UTF-8 file

Read-only: it reads ``acquisition.json``, ``stats.json`` and ``sources/<id>.txt`` /
``.provenance.json`` under ``--out`` and prints

* one line per source: status, characters, lines, the dominant writing system of its
  letters (and whether that is the expected one for the language), the scan pages a
  rendered source covered with their proofreading levels, and a hash prefix (so two runs
  can be compared for identical bytes);
* **flags** for anything that should never survive cleaning: HTML tags, wiki markup, CSS,
  ``&…;`` entities, invisible characters (U+200B, U+2060, U+FEFF, U+00AD, bidi controls),
  U+FFFD replacement characters, control characters, URLs, and long ASCII digit runs in a
  non-Latin text (hidden page ids such as ``38655`` look like that);
* **where to look**: for every flagged source, up to five places per kind of problem, each
  with the text around it (``«…»`` marks the spot), plus letters from another writing
  system than the language's own (information, not a flag: an English word can belong in
  a Hindi novel — a human decides);
* how the first and last source of every language (and any source named with ``--show``)
  starts and ends — the part a human has to read;
* identical-document counts per language from ``stats.json``.

A clean report is necessary, not sufficient: it cannot tell a work from a different work.
Reading the samples (runbook §8) is still the human step before ``--pin``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_OUT = "data/tokenizer/indic-tokenizer-v2"

# The writing system a language's letters should be in (first word of the Unicode name).
EXPECTED_SCRIPT = {
    "en": "LATIN", "hi": "DEVANAGARI", "mr": "DEVANAGARI", "bn": "BENGALI", "as": "BENGALI",
    "gu": "GUJARATI", "pa": "GURMUKHI", "or": "ORIYA", "ta": "TAMIL", "te": "TELUGU",
    "kn": "KANNADA", "ml": "MALAYALAM", "ur": "ARABIC", "hi-en": "LATIN",
}

INVISIBLE = {
    "\u200b": "ZWSP", "\u2060": "WJ", "\ufeff": "BOM", "\u00ad": "SHY", "\u200e": "LRM",
    "\u200f": "RLM", "\u202a": "LRE", "\u202b": "RLE", "\u202c": "PDF", "\u202d": "LRO",
    "\u202e": "RLO", "\u2066": "LRI", "\u2067": "RLI", "\u2068": "FSI", "\u2069": "PDI",
}
_PATTERNS = {
    "html": re.compile(r"</?[a-zA-Z][^<>]{0,200}>"),
    "wiki": re.compile(r"\{\{|\}\}|\[\[|\]\]|__[A-Z]+__"),
    "entity": re.compile(r"&(?:#\d{1,7}|#x[0-9a-fA-F]{1,6}|[a-zA-Z]{2,8});"),
    "css": re.compile(r"mw-parser-output|\.[a-z][\w-]*\s*\{[^{}]*:[^{}]*\}"),
    "url": re.compile(r"https?://"),
}
_ASCII_DIGIT_RUN = re.compile(r"(?<![0-9])[0-9]{4,}(?![0-9])")
EXAMPLES = 5  # places shown per kind of problem, per source
OTHER_SCRIPT_EXAMPLES = 3
CONTEXT = 30  # characters of text shown on each side of a place


def _script(char: str) -> str | None:
    """First word of the Unicode name of a letter or mark (its writing system), else None."""
    if unicodedata.category(char)[0] not in {"L", "M"}:
        return None
    name = unicodedata.name(char, "")
    return name.split(" ")[0] if name else "UNKNOWN"


def _context(text: str, start: int, end: int, mark: str | None = None) -> str:
    """One line of text around ``text[start:end]``; the spot itself between «…»."""
    left = text[max(0, start - CONTEXT):start]
    right = text[end:end + CONTEXT]
    spot = text[start:end] if mark is None else mark
    snippet = f"{left}«{spot}»{right}".replace("\n", " | ")
    return ("…" if start > CONTEXT else "") + snippet + ("…" if end + CONTEXT < len(text) else "")


def analyze_text(text: str, language: str) -> dict[str, Any]:
    """Facts and flags for one cleaned text (pure function; no I/O)."""
    lines = [line for line in text.split("\n") if line.strip()]
    scripts: Counter[str] = Counter()
    for char in text:
        if unicodedata.category(char)[0] in {"L", "M"}:
            name = unicodedata.name(char, "")
            scripts[name.split(" ")[0] if name else "UNKNOWN"] += 1
    letters = sum(scripts.values())
    dominant, dominant_count = scripts.most_common(1)[0] if scripts else ("NONE", 0)
    expected = EXPECTED_SCRIPT.get(language)

    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for name, pattern in _PATTERNS.items():
        found = list(pattern.finditer(text))
        counts[name] = len(found)
        if found:
            examples[name] = [_context(text, m.start(), m.end()) for m in found[:EXAMPLES]]
    invisible = Counter(INVISIBLE[char] for char in text if char in INVISIBLE)
    counts["replacement_char"] = text.count("\ufffd")
    counts["control"] = sum(
        1 for char in text if unicodedata.category(char) == "Cc" and char not in "\n\t"
    )
    digit_runs = list(_ASCII_DIGIT_RUN.finditer(text)) if expected not in {None, "LATIN"} else []
    counts["ascii_digit_runs"] = len(digit_runs)
    if digit_runs:
        examples["ascii_digit_runs"] = [_context(text, m.start(), m.end()) for m in digit_runs[:EXAMPLES]]
    odd: dict[str, list[str]] = {"replacement_char": [], "control": [], "invisible": []}
    for index, char in enumerate(text):
        if char == "\ufffd":
            kind, mark = "replacement_char", None
        elif char in INVISIBLE:
            kind, mark = "invisible", INVISIBLE[char]  # the character itself cannot be seen
        elif unicodedata.category(char) == "Cc" and char not in "\n\t":
            kind, mark = "control", f"U+{ord(char):04X}"
        else:
            continue
        if len(odd[kind]) < EXAMPLES:
            odd[kind].append(_context(text, index, index + 1, mark))
    examples.update({kind: places for kind, places in odd.items() if places})

    # letters from another writing system than the language's own: runs, with context
    other_letters: Counter[str] = Counter()
    other_places: list[str] = []
    if expected:
        index = 0
        while index < len(text):
            script = _script(text[index])
            if script is None or script == expected:
                index += 1
                continue
            end = index
            while end < len(text) and _script(text[end]) not in (None, expected):
                other_letters[_script(text[end]) or "UNKNOWN"] += 1
                end += 1
            if len(other_places) < OTHER_SCRIPT_EXAMPLES:
                other_places.append(_context(text, index, end))
            index = end
    counts["other_script_letters"] = sum(other_letters.values())
    if other_places:
        examples["other_script"] = other_places

    flags: list[str] = []
    for name in ("html", "wiki", "entity", "css", "url", "replacement_char", "control"):
        if counts[name]:
            flags.append(f"{name}={counts[name]}")
    if invisible:
        flags.append("invisible=" + ",".join(f"{k}:{v}" for k, v in sorted(invisible.items())))
    if counts["ascii_digit_runs"]:
        flags.append(f"ascii_digit_runs={counts['ascii_digit_runs']}")
    if expected and letters and dominant != expected:
        flags.append(f"script={dominant} (expected {expected})")
    return {
        "chars": len(text),
        "lines": len(lines),
        "dominant_script": dominant,
        "dominant_share": round(dominant_count / letters, 4) if letters else 0.0,
        "expected_script": expected,
        "scripts": dict(scripts.most_common(4)),
        "counts": counts,
        "invisible": dict(invisible),
        "flags": flags,
        "examples": examples,
        "other_script_letters": dict(other_letters.most_common()),
        # the opening and the ending, across lines (a first line may be just "२")
        "head": " | ".join(lines)[:400],
        "tail": " | ".join(lines)[-400:],
    }


def _short_page(name: str | None) -> str:
    return "-" if not name else name.rsplit("/", 1)[-1]


def inspect_build(out: Path) -> dict[str, Any]:
    """Everything the report prints, as data (``--print-json``)."""
    acquisition = json.loads((out / "acquisition.json").read_text(encoding="utf-8"))
    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    rows: list[dict[str, Any]] = []
    for source in acquisition.get("sources", []):
        source_id = source["source_id"]
        text_path = out / "sources" / f"{source_id}.txt"
        provenance_path = out / "sources" / f"{source_id}.provenance.json"
        row: dict[str, Any] = {
            "source_id": source_id,
            "language": source.get("language", ""),
            "status": source.get("verification_status", ""),
            "sha256": source.get("sha256"),
            "has_text": text_path.exists() and bool(source.get("sha256")),
            "analysis": None,
            "page_quality": None,
            "artifacts_removed": None,
            "error": source.get("error", ""),
        }
        # a file left over from an earlier run is not what this run acquired: only a file
        # whose hash is the one this run recorded is inspected
        if row["has_text"]:
            text = text_path.read_text(encoding="utf-8")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() == row["sha256"]:
                row["analysis"] = analyze_text(text, row["language"])
            else:
                row["has_text"] = False
                row["error"] = (row["error"] + " " if row["error"] else "") + (
                    "sources/ holds a file from a different run (hash differs): not inspected"
                )
        if row["has_text"] and provenance_path.exists():
            check = json.loads(provenance_path.read_text(encoding="utf-8")).get("content_check") or {}
            row["page_quality"] = check.get("page_quality")
            row["artifacts_removed"] = check.get("artifacts_removed")
        rows.append(row)
    duplicates = {
        entry["language"]: entry.get("duplicate_documents", 0)
        for entry in stats.get("languages", [])
        if entry.get("examples")
    }
    return {
        "out": str(out),
        "sources": rows,
        "flagged": [row["source_id"] for row in rows if row["analysis"] and row["analysis"]["flags"]],
        "duplicates": duplicates,
    }


def _where_to_look(rows: list[dict[str, Any]]) -> list[str]:
    """For each flagged source (or one with letters of another script): the places."""
    out: list[str] = []
    for row in rows:
        analysis = row["analysis"]
        if analysis is None or not analysis.get("examples"):
            continue
        if not out:
            out.append("[inspect] where to look (up to 5 places per kind; «…» marks the spot, "
                       "' | ' = line break):")
        out.append(f"[inspect]   {row['source_id']}")
        for kind, places in analysis["examples"].items():
            if kind == "other_script":
                letters = ", ".join(f"{k} {v}" for k, v in analysis["other_script_letters"].items())
                label = f"other-script letters ({letters}) — information, not a flag"
            else:
                label = kind
            out.append(f"[inspect]       {label}:")
            out.extend(f"[inspect]         {place}" for place in places)
    return out


def render(report: dict[str, Any], *, show: list[str], width: int) -> str:
    rows = report["sources"]
    verified = sum(1 for row in rows if row["status"] == "verified")
    lines = [
        f"[inspect] {report['out']} — {len(rows)} sources, {verified} verified "
        "(read-only: nothing is stored, verified or pinned)",
        f"[inspect]   {'source':<36} {'status':<19} {'chars':>9} {'lines':>6}  "
        f"{'script':<18} {'pages':<18} {'sha256':<12}  flags",
    ]
    for row in rows:
        analysis = row["analysis"]
        if analysis is None:
            lines.append(
                f"[inspect]   {row['source_id']:<36} {row['status']:<19} {'-':>9} {'-':>6}  "
                f"{'-':<18} {'-':<18} {'-':<12}  no text this run"
            )
            if row["error"]:
                lines.append(f"[inspect]       reason: {row['error'][:160]}")
            continue
        share = f"{analysis['dominant_script'][:10]} {analysis['dominant_share'] * 100:.1f}%"
        quality = row["page_quality"] or {}
        if quality.get("pages"):
            levels = ",".join(f"q{k}:{v}" for k, v in (quality.get("by_level") or {}).items())
            first, last = _short_page(quality.get("first_page")), _short_page(quality.get("last_page"))
            pages = f"{first}-{last} {levels}"
        else:
            pages = "-"
        flags = " ".join(analysis["flags"]) or "OK"
        lines.append(
            f"[inspect]   {row['source_id']:<36} {row['status']:<19} {analysis['chars']:>9,} "
            f"{analysis['lines']:>6}  {share:<18} {pages:<18} {(row['sha256'] or '')[:12]:<12}  {flags}"
        )
    if report["duplicates"]:
        lines.append(
            "[inspect] identical documents per language (always on the same side of the split): "
            + ", ".join(f"{lang}={count}" for lang, count in sorted(report["duplicates"].items()))
        )
    lines.append(
        "[inspect] flagged sources: "
        + (", ".join(report["flagged"]) if report["flagged"] else "none")
    )
    lines.extend(_where_to_look(rows))

    # samples: first and last source of each language, plus --show
    by_language: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["analysis"] is not None:
            by_language.setdefault(row["language"], []).append(row)
    chosen: list[str] = []
    for group in by_language.values():
        for row in (group[0], group[-1]):
            if row["source_id"] not in chosen:
                chosen.append(row["source_id"])
    chosen.extend(sid for sid in show if sid not in chosen)
    lines.append("[inspect] samples (read these — the report cannot tell a work from another work):")
    for source_id in chosen:
        row = next((r for r in rows if r["source_id"] == source_id), None)
        if row is None or row["analysis"] is None:
            lines.append(f"[inspect]   {source_id}: no text this run")
            continue
        head = row["analysis"]["head"][:width]
        tail = row["analysis"]["tail"][-width:]
        lines.append(f"[inspect]   {source_id}   (' | ' = line break)")
        lines.append(f"[inspect]       starts: {head}")
        lines.append(f"[inspect]       ends:   {tail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="build directory (default: %(default)s)")
    parser.add_argument("--show", action="append", default=[], metavar="SOURCE_ID",
                        help="also print how this source starts and ends (repeatable)")
    parser.add_argument("--chars", type=int, default=160, help="sample width (default: %(default)s)")
    parser.add_argument("--print-json", action="store_true", help="machine-readable report")
    parser.add_argument("--output", metavar="FILE", help="also write the report to FILE (UTF-8)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if not (out / "acquisition.json").exists():
        print(f"[inspect] no acquisition.json under {out}: run a --fetch build first", file=sys.stderr)
        return 2
    report = inspect_build(out)
    text = (
        json.dumps(report, indent=2, ensure_ascii=False)
        if args.print_json
        else render(report, show=args.show, width=args.chars)
    )
    try:
        print(text)
    except UnicodeEncodeError:  # a console that cannot show the script: keep going
        print(text.encode("ascii", "backslashreplace").decode("ascii"))
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"[inspect] wrote {args.output} (UTF-8)")
    return 1 if report["flagged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
