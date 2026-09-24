"""Discover chapter subpages for declared Wikisource works.

Run from the repository root after 'git pull':

    python scripts/discover_wiki_chapters.py

READ-ONLY diagnostic: prints what it finds; writes and pins nothing.
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
    url = (
        f"https://{lang}.wikisource.org/w/api.php?"
        f"action=query&list=allpages&apprefix={urllib.parse.quote(prefix + '/')}"
        f"&apnamespace=0&aplimit=500&format=json"
    )
    data = _get(url)
    return [p["title"] for p in data["query"]["allpages"]]


def raw_len(lang: str, title: str) -> tuple[int | None, str | None]:
    """Return (length_of_raw_wikitext, redirect_target_if_any)."""
    url = (
        f"https://{lang}.wikisource.org/w/api.php?"
        f"action=query&titles={urllib.parse.quote(title)}"
        f"&prop=revisions&rvprop=size|content&rvlimit=1&format=json&redirects"
    )
    try:
        data = _get(url)
    except Exception as exc:  # noqa: BLE001
        return None, f"fetch error: {exc}"
    pages = data["query"]["pages"]
    # redirects field is present if we were redirected
    redirects = data["query"].get("redirects", [])
    redirect_target = redirects[0]["to"] if redirects else None
    for pid, p in pages.items():
        if pid == "-1":
            return None, "missing"
        revs = p.get("revisions") or []
        if revs:
            content = revs[0].get("*", "")
            return len(content), redirect_target
        return 0, redirect_target
    return None, None


def search(lang: str, term: str) -> list[dict[str, Any]]:
    url = (
        f"https://{lang}.wikisource.org/w/api.php?"
        f"action=query&list=search&srsearch={urllib.parse.quote(term)}"
        f"&srnamespace=0|102|104&srlimit=30&format=json"
    )
    data = _get(url)
    return data["query"]["search"]


def resolve(lang: str, title: str) -> dict[str, Any]:
    """Follow redirects and return {'title', 'length', 'redirect_from'}"""
    length, redir = raw_len(lang, title)
    return {"candidate": title, "length": length, "redirect_to": redir}


def banner(msg: str) -> None:
    print()
    print("=" * 72)
    print(msg)
    print("=" * 72)


def main() -> int:
    banner("Hindi — गो-दान subpages (Godaan redirect target)")
    hi_pages = subpages("hi", "गो-दान")
    print(f"Found {len(hi_pages)} subpages of गो-दान/")
    # Spot-check a few: how large is the raw wikitext?
    for title in ["गो-दान"] + hi_pages[:4] + hi_pages[-2:]:
        r = resolve("hi", title)
        if r["length"] is not None:
            print(f"  {r['candidate']:<28}  wikitext_len={r['length']:>7}"
                  f"{'  (redirects to ' + r['redirect_to'] + ')' if r['redirect_to'] else ''}")
        else:
            print(f"  {r['candidate']:<28}  -> {r['redirect_to']}")

    banner("Bengali — Gitanjali: probing candidate titles (raw wikitext size)")
    bn_candidates = [
        "গীতাঞ্জলি",
        "গীতাঞ্জলী",
        "গীতাঞ্জলি (গ্রন্থ)",
        "গীতাঞ্জলি (রবীন্দ্রনাথ ঠাকুর)",
        "গীতাঞ্জলি/গীতাঞ্জলি",
        "Index:গীতাঞ্জলি",
        "Index:Gitanjali",
        "সঞ্চয়িতা/গীতাঞ্জলি",
        "গীতাঞ্জলি/১",
        "গীতাঞ্জলি/গীতাঞ্জলি/১",
        "গীতাঞ্জলি (কাব্যগ্রন্থ)",
        "Gitanjali",
    ]
    seen = set()
    for cand in bn_candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            r = resolve("bn", cand)
            if r["length"] is None:
                print(f"  {cand:<40}  -> {r['redirect_to']}")
            else:
                redir_note = f"  (-> {r['redirect_to']})" if r["redirect_to"] else ""
                print(f"  {cand:<40}  wikitext_len={r['length']:>7}{redir_note}")
                # If it's big (>5000 chars), also list subpages
                if r["length"] > 5000:
                    sp = subpages("bn", r["redirect_to"] or cand)
                    if sp:
                        print(f"      SUBPAGES ({len(sp)}): {sp[:10]}{'...' if len(sp)>10 else ''}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {cand:<40}  ERROR: {exc}")

    banner("Bengali — search for গীতাঞ্জলি (namespaces 0, 102 Index, 104 Page)")
    for term in ["গীতাঞ্জলি", "গীতাঞ্জলী", "Gitanjali"]:
        try:
            hits = search("bn", term)
            print(f"\n  search('{term}') -> {len(hits)} hits:")
            for h in hits[:20]:
                size = h.get("size", "?")
                title = h["title"]
                # Show the subpage list for any hit that looks like a container
                print(f"    size={size:>6}  ns={h.get('ns')}  {title}")
        except Exception as exc:  # noqa: BLE001
            print(f"  search('{term}') ERROR: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
