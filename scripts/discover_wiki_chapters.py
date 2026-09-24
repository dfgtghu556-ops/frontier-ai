#!/usr/bin/env python3
"""Find the chapter pages of a Wikisource work and propose ``mediawiki-parse`` sources.

Examples (run from the repository root; needs network access to the wiki)::

    # list the chapters of Godaan on Hindi Wikisource
    python scripts/discover_wiki_chapters.py --lang hi --work "गो-दान"

    # ... and write reviewable manifest entries to a UTF-8 file
    python scripts/discover_wiki_chapters.py --lang hi --work "गो-दान" \\
        --emit-sources hi-wikisource-godaan --output godaan_sources.json

What it does (read-only: it never edits the manifest, fetches no corpus text, pins nothing):

1. lists every main-namespace page under ``<work>/`` (the chapter subpages);
2. reads each subpage's wikitext and extracts its ProofreadPage ``<pages index=… from=…
   to=…>`` tag — the scan it renders and the page range;
3. sorts chapters by the number in their title (ASCII, Devanagari, Bengali, … digits);
4. flags what must **not** be declared blindly: subpages that render a *different scan*
   than the majority (legacy transcriptions — on Hindi Wikisource ``गो-दान/ भाग 16`` renders
   ``गोदान.pdf`` and would duplicate text), and subpages without a ``<pages>`` tag;
5. with ``--emit-sources``, proposes one source per clean chapter, using the canonical
   URL builder from :mod:`frontier_ai.data.mediawiki`, ``sha256: null`` and
   ``verified: false``. A human reviews them before they go into the manifest, and only
   a real ``--fetch`` build can verify them.

Windows PowerShell note: use ``--output FILE`` instead of ``> FILE`` — PowerShell 5.1
redirection writes UTF-16, which is not valid for the manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.data.corpora import USER_AGENT  # noqa: E402
from frontier_ai.data.mediawiki import parse_page_url  # noqa: E402

_PAGES_TAG = re.compile(r"<pages\b(?P<attrs>[^>]*?)/?>", re.IGNORECASE | re.DOTALL)
_ATTR = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'/>]+))')
_TRAILING_NUMBER = re.compile(r"(\d+)\s*$")


def _get(url: str, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _api(lang: str, **params: str) -> str:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    return f"https://{lang}.wikisource.org/w/api.php?{query}"


def list_subpages(lang: str, work: str) -> list[str]:
    """Every main-namespace title that starts with ``<work>/`` (follows continuation)."""
    titles: list[str] = []
    cont: dict[str, str] = {}
    while True:
        data = _get(_api(lang, action="query", list="allpages", apnamespace="0",
                         apprefix=f"{work}/", aplimit="500", **cont))
        titles.extend(page["title"] for page in data["query"]["allpages"])
        if "continue" not in data:
            return titles
        cont = {"apcontinue": data["continue"]["apcontinue"]}


def read_wikitext(lang: str, titles: list[str]) -> dict[str, str]:
    """Current wikitext of each title, 50 titles per request."""
    texts: dict[str, str] = {}
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        data = _get(_api(lang, action="query", prop="revisions", rvprop="content",
                         rvslots="main", titles="|".join(batch)))
        for page in data["query"]["pages"]:
            revisions = page.get("revisions") or []
            texts[page["title"]] = (
                revisions[0]["slots"]["main"].get("content", "") if revisions else ""
            )
    return texts


def pages_tag(wikitext: str) -> dict[str, str] | None:
    """The attributes of the first ProofreadPage ``<pages>`` tag, verbatim."""
    match = _PAGES_TAG.search(wikitext)
    if match is None:
        return None
    attrs: dict[str, str] = {}
    for key, double, single, bare in _ATTR.findall(match.group("attrs")):
        attrs[key.lower()] = double or single or bare
    return attrs


def as_number(text: str) -> int | None:
    """Read digits in any script (१२ → 12, ১২ → 12); ``None`` if there are none."""
    digits = []
    for char in text:
        try:
            digits.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            if digits:
                break
    return int("".join(digits)) if digits else None


def chapter_number(title: str) -> int | None:
    tail = title.rsplit("/", 1)[-1].strip()
    ascii_tail = "".join(
        str(unicodedata.digit(char)) if unicodedata.category(char) == "Nd" else char
        for char in tail
    )
    match = _TRAILING_NUMBER.search(ascii_tail)
    return int(match.group(1)) if match else None


def discover(lang: str, work: str) -> dict[str, Any]:
    """The chapter table for ``work`` on ``lang``.wikisource.org (see the module doc)."""
    titles = list_subpages(lang, work)
    texts = read_wikitext(lang, titles)
    rows: list[dict[str, Any]] = []
    for title in titles:
        tag = pages_tag(texts.get(title, ""))
        rows.append(
            {
                "title": title,
                "number": chapter_number(title),
                "index": (tag or {}).get("index"),
                "from": (tag or {}).get("from"),
                "to": (tag or {}).get("to"),
                "from_number": as_number((tag or {}).get("from", "")),
                "to_number": as_number((tag or {}).get("to", "")),
                "flags": [],
            }
        )
    indexes = Counter(row["index"] for row in rows if row["index"])
    majority = indexes.most_common(1)[0][0] if indexes else None
    for row in rows:
        if row["index"] is None:
            row["flags"].append("no <pages> tag: renders no scan (check by hand)")
        elif row["index"] != majority:
            row["flags"].append(
                f"different scan ({row['index']}, majority is {majority}): legacy or "
                "duplicate transcription, do not declare without review"
            )
        if row["number"] is None:
            row["flags"].append("no chapter number in the title")
    rows.sort(key=lambda row: (row["number"] is None, row["number"] or 0, row["title"]))
    numbers = [row["number"] for row in rows if not row["flags"]]
    missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers)) if numbers else []
    return {
        "lang": lang,
        "work": work,
        "scan": majority,
        "chapters": rows,
        "clean": [row for row in rows if not row["flags"]],
        "missing_numbers": missing,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def propose_sources(report: dict[str, Any], id_prefix: str, *, work_label: str) -> list[dict[str, Any]]:
    """Unverified ``mediawiki-parse`` entries for the clean chapters, for human review."""
    lang = report["lang"]
    host = f"{lang}.wikisource.org"
    evidence = {
        "url": f"https://{host}/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json",
        "kind": "mediawiki-api",
        "marker": "https://creativecommons.org/licenses/by-sa/4.0/",
        "scope": "site",
        "note": (
            f"Site-level evidence for {host} (the rendered payload carries no licence notice). "
            "A human still reviews the work page's own licence tag."
        ),
    }
    total = len(report["clean"])
    entries = []
    for row in report["clean"]:
        number = row["number"]
        entries.append(
            {
                "id": f"{id_prefix}-ch{number:02d}-ccbysa",
                "title": f"{work_label} — chapter {number} of {total} ({row['title']})",
                "language": lang,
                "script": "",
                "source_url": parse_page_url(host, row["title"]),
                "license_id": "CC-BY-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": (
                    f"{host} contributors, {work_label}, chapter {number} (page {row['title']}), "
                    f"transcribed from the scan {report['scan']}; licensed CC BY-SA 4.0."
                ),
                "max_chars": 400000,
                "kind": "mediawiki-parse",
                "sha256": None,
                "verified": False,
                "retrieved_at": None,
                "notes": (
                    f"Proposed by scripts/discover_wiki_chapters.py on {report['checked_at']}: "
                    f"renders scan pages {row['from']}–{row['to']} of {report['scan']} as declared "
                    "on the wiki. Fill in 'script', review the attribution and the work page's "
                    "licence tag before adding this to the manifest."
                ),
                "license_evidence": evidence,
            }
        )
    return entries


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"{report['lang']}.wikisource.org — {report['work']}: {len(report['chapters'])} subpages, "
        f"scan {report['scan']!r}, {len(report['clean'])} clean chapters",
    ]
    for row in report["chapters"]:
        pages = f"{row['from']}–{row['to']}" if row["index"] else "-"
        mark = "OK  " if not row["flags"] else "FLAG"
        lines.append(f"  {mark} {str(row['number'] or '?'):>4}  {row['title']:<32} pages {pages}")
        for flag in row["flags"]:
            lines.append(f"            ! {flag}")
    if report["missing_numbers"]:
        lines.append(f"  missing chapter numbers: {report['missing_numbers']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", required=True, help="Wikisource language code (hi, bn, mr, …)")
    parser.add_argument("--work", required=True, help="the work's page title, e.g. गो-दान")
    parser.add_argument("--emit-sources", metavar="ID_PREFIX",
                        help="also propose manifest entries, ids <ID_PREFIX>-chNN-ccbysa")
    parser.add_argument("--work-label", help="human title for attribution (default: --work)")
    parser.add_argument("--output", metavar="FILE",
                        help="write the proposed entries (or the report) as UTF-8 JSON to FILE")
    args = parser.parse_args(argv)

    report = discover(args.lang, args.work)
    print(render_report(report))
    payload: Any = report
    if args.emit_sources:
        payload = propose_sources(report, args.emit_sources, work_label=args.work_label or args.work)
        print(f"\nproposed {len(payload)} unverified sources (sha256=null, verified=false)")
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output} (UTF-8)")
    elif args.emit_sources:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
