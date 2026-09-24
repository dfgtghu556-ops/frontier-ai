"""Discover chapter subpages for declared Wikisource works.

Run from the repository root after 'git pull':

    python scripts/discover_wiki_chapters.py

The script uses the MediaWiki API (with a proper User-Agent so Wikimedia does
not 403 us) to list subpages for the redirect/SPARQL targets of the hi/bn
sources, and to probe a few alternative titles for Bengali Gitanjali (whose
root is an infocard with no subpages).

This is a READ-ONLY diagnostic. It writes nothing, pins nothing and does not
modify the manifest — it just prints what it finds so a human can decide which
subpages to declare.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

UA = "frontier-ai-tokenizer-corpus/1.0 (https://github.com/dfgtghu556-ops/frontier-ai)"


def _get(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def subpages(lang: str, prefix: str) -> list[str]:
    """List main-namespace pages whose title starts with '<prefix>/'.

    Uses the MediaWiki 'allpages' list.  aplimit=500 is generous (a novel with
    that many chapters is unusual) but bounded.
    """
    url = (
        f"https://{lang}.wikisource.org/w/api.php?"
        f"action=query&list=allpages&apprefix={urllib.parse.quote(prefix + '/')}"
        f"&apnamespace=0&aplimit=500&format=json"
    )
    data = _get(url)
    return [p["title"] for p in data["query"]["allpages"]]


def page_info(lang: str, title: str) -> dict[str, Any] | None:
    """Fetch basic info + sections list for a page. Returns None if missing."""
    url = (
        f"https://{lang}.wikisource.org/w/api.php?"
        f"action=parse&page={urllib.parse.quote(title)}"
        f"&prop=sections|categories|displaytitle&format=json"
    )
    try:
        return _get(url)["parse"]
    except urllib.error.HTTPError as exc:  # type: ignore[name-defined]
        if exc.code == 404:
            return None
        raise


def search(lang: str, term: str) -> list[str]:
    """Search titles for the given term (to find Gitanjali if it lives elsewhere)."""
    url = (
        f"https://{lang}.wikisource.org/w/api.php?"
        f"action=query&list=search&srsearch={urllib.parse.quote(term)}"
        f"&srnamespace=0&srlimit=30&format=json"
    )
    data = _get(url)
    return [h["title"] for h in data["query"]["search"]]


def probe(lang: str, title: str) -> dict[str, Any]:
    info = page_info(lang, title)
    if info is None:
        return {"title": title, "exists": False}
    sections = info.get("sections", [])
    cats = [c["*"] for c in info.get("categories", [])]
    # Count how many sections look like chapters (numbered headings)
    numbered = [s for s in sections if s.get("number")]
    return {
        "title": title,
        "exists": True,
        "displaytitle": info.get("displaytitle", title),
        "n_sections": len(sections),
        "n_numbered_sections": len(numbered),
        "first_section_numbers": [s.get("number") for s in numbered[:6]],
        "first_section_lines": [s.get("line") for s in sections[:8]],
        "n_categories": len(cats),
        "first_cats": cats[:4],
    }


def banner(msg: str) -> None:
    print()
    print("=" * 72)
    print(msg)
    print("=" * 72)


def main() -> int:
    banner("Hindi — गो-दान (Godaan, redirect target of the declared root)")
    hi_pages = subpages("hi", "गो-दान")
    print(f"Found {len(hi_pages)} subpages:")
    for p in hi_pages:
        print(f"  {p}")

    # Separately, probe the root for its sections list so we can tell whether the
    # numbered pages are chapters vs the 'भाग' pages are real parts.
    banner("Hindi — content shape of a few candidate pages")
    for title in ["गो-दान"] + hi_pages[:3] + hi_pages[-3:]:
        pr = probe("hi", title)
        if pr["exists"]:
            print(f"  {pr['title']:<30} sections={pr['n_sections']:>3} "
                  f"numbered={pr['n_numbered_sections']:>3}  "
                  f"first={pr['first_section_lines'][:2]}")
        else:
            print(f"  {title:<30} (not found)")

    banner("Bengali — গীতাঞ্জলি (Gitanjali): declared root and likely alternatives")
    bn_candidates: list[str] = [
        "গীতাঞ্জলি",  # declared root (is a SPARQL infocard)
        "গীতাঞ্জলী",  # alternate spelling with dirgho-I
        "গীতাঞ্জলি (গ্রন্থ)",  # common "(book)" disambiguation
        "গীতাঞ্জলি (রবীন্দ্রনাথ ঠাকুর)",
        "গীতাঞ্জলি/গীতাঞ্জলি",
        "Index:গীতাঞ্জলি",  # Index: namespace (multipart works)
    ]
    for cand in bn_candidates:
        pr = probe("bn", cand)
        if pr["exists"]:
            print(f"  {cand:<35} EXSTS  sections={pr['n_sections']:>3} "
                  f"numbered={pr['n_numbered_sections']:>3}  "
                  f"first={pr['first_section_lines'][:3]}")
            # Try listing subpages
            sp = subpages("bn", cand)
            if sp:
                print(f"    --> has {len(sp)} subpages, first few: {sp[:5]}")
        else:
            print(f"  {cand:<35} missing")

    banner("Bengali — search for Gitanjali-related pages on bn.wikisource")
    for term in ["গীতাঞ্জলি", "গীতাঞ্জলী"]:
        hits = search("bn", term)
        print(f"  search({term!r}) -> {len(hits)} hits:")
        for h in hits[:15]:
            print(f"    {h}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
