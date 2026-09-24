"""Tests for the ``mediawiki-parse`` source kind (rendered Wikisource scanned books).

Why these exist: hi/bn.wikisource chapter pages hold only a ProofreadPage ``<pages/>``
tag, so ``?action=raw`` never contains the work. The corpus therefore asks MediaWiki to
render the page and converts the HTML to text. These tests pin down what that conversion
keeps, what it drops, and every gate that refuses a rendered page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from frontier_ai.data.corpora import clean_text, sha256_text
from frontier_ai.data.mediawiki import (
    BROKEN_GAP,
    MISSING_TEMPLATE,
    STRAY_CLOSING_BRACES,
    TEMPLATE_NAMESPACE,
    clean_mediawiki_parse,
    read_parse_payload,
    render_html,
    validate_parse_url,
)
from frontier_ai.tokenization.research_corpus import (
    INDEX_PAGE_MIN_CLEANED_CHARS,
    LicenseEvidence,
    TokenizerCorpusManifest,
    build_corpus,
    content_shape,
    ingest_source,
    preflight_manifest,
    render_preflight,
    validate_manifest,
)
from tokenizer_corpus_fixtures import (
    BENGALI_VERSE,
    HINDI_PROSE,
    contents_page_html,
    parse_error_payload,
    parse_payload,
    redirect_html,
    rendered_chapter_html,
    rendered_poems_html,
)

REPO_MANIFEST = (
    Path(__file__).resolve().parents[1] / "corpora" / "tokenizer" / "indic-tokenizer-v2" / "sources.json"
)
KIND = "mediawiki-parse"
PAGE_URL = (
    "https://hi.wikisource.org/w/api.php?action=parse&format=json&formatversion=2"
    "&prop=text%7Crevid%7Ccategories&disablelimitreport=1&disableeditsection=1"
    "&page=%E0%A4%AA%E0%A4%B0%E0%A5%80%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A4%BE%2F%E0%A5%A8"
)
RANGE_URL = (
    "https://bn.wikisource.org/w/api.php?action=parse&format=json&formatversion=2&prop=text"
    "&disablelimitreport=1&disableeditsection=1&contentmodel=wikitext&title=G"
    "&text=%3Cpages%20index%3D%22G.djvu%22%20from%3D13%20to%3D190%20%2F%3E"
)
EVIDENCE_URL = "https://hi.wikisource.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json"
CC_BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"
SITEINFO = json.dumps({"query": {"general": {"rightsinfo": {"url": CC_BY_SA}}}})
EVIDENCE = {"url": EVIDENCE_URL, "kind": "mediawiki-api", "marker": CC_BY_SA, "scope": "site"}


def _source(source_id: str = "hi-chapter", *, url: str = PAGE_URL, language: str = "hi",
            evidence: dict | None = EVIDENCE) -> dict:
    payload = {
        "id": source_id,
        "title": f"Rendered test chapter {source_id}",
        "language": language,
        "script": "Devanagari" if language == "hi" else "Bengali",
        "source_url": url,
        "license_id": "CC-BY-SA-4.0",
        "license_url": CC_BY_SA,
        "attribution": "Test Wikisource contributors, rendered test chapter, CC BY-SA 4.0.",
        "kind": KIND,
        "max_chars": 400_000,
        "sha256": None,
        "verified": False,
    }
    if evidence is not None:
        payload["license_evidence"] = evidence
    return payload


def _manifest(tmp_path: Path, sources: list[dict]) -> Path:
    by_language: dict[str, list[str]] = {}
    for source in sources:
        by_language.setdefault(source["language"], []).append(source["id"])
    payload = {
        "schema_version": "1.0",
        "corpus": {"id": "indic-tokenizer", "version": "v2"},
        "targets": {"sentences_per_language": 500, "characters_per_language": 200_000},
        "split": {"method": "deterministic-hash", "seed": 1337, "heldout_fraction": 0.1},
        "normalization_policy": "none",
        "cleaning_policy": "kind-specific",
        "language_slots": [
            {"code": code, "display": code, "script": "x", "sources": ids}
            for code, ids in sorted(by_language.items())
        ],
        "sources": sources,
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _evidence() -> LicenseEvidence:
    return LicenseEvidence.from_dict(EVIDENCE)


def _fetcher(bodies: dict[str, str], calls: list[str] | None = None):
    """fetch_text stand-in: siteinfo for the evidence URL, otherwise the body whose key
    occurs in the URL (the source id is not in API URLs, so keys are URL fragments)."""

    def _fetch(url: str, *_args, **_kwargs) -> str:
        if calls is not None:
            calls.append(url)
        if "meta=siteinfo" in url:
            return SITEINFO
        for key, body in bodies.items():
            if key in url:
                return body
        raise AssertionError(f"unexpected fetch {url}")

    return _fetch


# ---------------------------------------------------------------------------
# rendering: what is kept, what is dropped
# ---------------------------------------------------------------------------
def test_render_drops_wikisource_chrome_and_keeps_the_text() -> None:
    page = render_html(rendered_chapter_html())
    text = page.text
    # the work: chapter label and every paragraph, one line each
    assert text.splitlines()[0] == "२"
    for paragraph in (HINDI_PROSE[0], HINDI_PROSE[2], HINDI_PROSE[3]):
        assert paragraph in text
    # the paragraph that straddles two scan pages stays one line (page-join <br>)
    half = len(HINDI_PROSE[1]) // 2
    assert f"{HINDI_PROSE[1][:half].rstrip()} {HINDI_PROSE[1][half:].lstrip()}" in text.splitlines()
    # the chrome: header template, hidden ws-data metadata, CSS, footnote marker and list
    for chrome in ("पीछे", "आगे", "लेखक-नाम", "(1936)", "38658", "wst-center", "[१]", "टिप्पणी"):
        assert chrome not in text, chrome
    assert page.has_prp_output is True
    assert page.navigation_lines == 0


def test_render_removes_page_join_artifacts_but_keeps_indic_joiners() -> None:
    html = rendered_chapter_html(
        ["क्\u200dष और र्\u200cय जैसे जोड़ बने रहने चाहिए, और आमि\u00a0तुमि के बीच सामान्य जगह।"]
        + HINDI_PROSE[1:],
    )
    page = render_html(html)
    assert "\u200d" in page.text and "\u200c" in page.text  # ZWJ / ZWNJ are orthographic
    for artifact in ("\u200b", "\u2060", "\ufeff", "\u00a0"):
        assert artifact not in page.text
    assert "आमि तुमि" in page.text
    assert page.artifacts_removed["U+FEFF ZERO WIDTH NO-BREAK SPACE"] == 1
    assert page.artifacts_removed["U+2060 WORD JOINER"] >= 1
    assert page.artifacts_removed["U+00A0 NO-BREAK SPACE -> space"] == 1
    # the zero width space inside the ws-noexport anchor never reaches the text at all
    assert "U+200B ZERO WIDTH SPACE" not in page.artifacts_removed


def test_render_keeps_verse_lines_and_stanza_breaks() -> None:
    page = render_html(rendered_poems_html())
    lines = page.text.split("\n")
    for verse in BENGALI_VERSE:
        assert verse in lines  # one verse line per text line, indentation dropped
    assert "1" in lines and "2" in lines  # poem numbers stay (they are in the book)
    assert "" in lines  # stanzas are separated by a blank line
    # <br><br> is an empty line, i.e. a stanza break, never two verse lines glued together
    two = render_html("<div class='poem'><p>প্রথম<br /><br />দ্বিতীয়</p></div>").text
    assert two == "প্রথম\n\nদ্বিতীয়"


def test_page_join_line_breaks_become_spaces_outside_poems() -> None:
    """A <br> right after a page anchor is the transcription's page-join convention: the
    printed sentence (here even a word) continues, so it must not split the text."""
    anchor = (
        '<span><span class="pagenum ws-pagenum" data-page-name="P/20" data-page-quality="3">'
        '<span class="pagenum-inner ws-noexport">&#8203;</span></span></span>'
    )
    prose = render_html(
        f"<p>दुखी होने का कोई अधि&#32;{anchor}\ufeff<br />कार ही नहीं है।</p>"
        "<p><br />\nनया अनुच्छेद यहाँ से।</p>"
    )
    assert prose.text == "दुखी होने का कोई अधि कार ही नहीं है।\n\nनया अनुच्छेद यहाँ से।"
    assert prose.artifacts_removed["page-join <br> -> space"] == 1
    # only the first <br> after the anchor is the convention; a second one is a real break
    twice = render_html(f"<p>एक{anchor}<br /><br />दो</p>")
    assert twice.text == "एक\nदो"
    # inside a poem every <br> is a verse line, anchor or not
    poem = render_html(f'<div class="poem"><p>পংক্তি এক{anchor}<br />\nপংক্তি দুই</p></div>')
    assert poem.text == "পংক্তি এক\nপংক্তি দুই"
    assert "page-join <br> -> space" not in poem.artifacts_removed
    # visible text between the anchor and the <br> means the break is the author's
    kept = render_html(f"<p>{anchor}शीर्षक<br />पहली पंक्ति</p>")
    assert kept.text == "शीर्षक\nपहली पंक्ति"


# every residue below was printed verbatim by hi.wikisource for a mistyped {{gap}} in गो-दान
# (checked 2026-09-24, e.g. पृष्ठ:गो-दान.djvu/११६ "{{Gap{}", /४१ "<gap>", /२२ "{Gap}}")
GODAAN_BROKEN_GAPS = [
    "{{Gap{}", "{{Gap{{", "{{Gap}]", "{{gap}]", "{{Gap]}", "{Gap}}", "{{Gap", "{{Gap))",
    "{{Gap@))", "&lt;gap&gt;",
]


def test_broken_gap_template_residue_is_removed_and_counted() -> None:
    prose = [f"{residue}{HINDI_PROSE[i % len(HINDI_PROSE)]}" for i, residue in enumerate(GODAAN_BROKEN_GAPS)]
    html = rendered_chapter_html(prose, qualities=(3,))
    page = render_html(html)
    for residue in ("{", "}", "<gap>", "Gap", "gap"):
        assert residue not in page.text, residue
    for paragraph in HINDI_PROSE:  # the transcription itself is untouched
        assert paragraph in page.text
    assert page.artifacts_removed[BROKEN_GAP] == len(GODAAN_BROKEN_GAPS)
    # the same text a correctly typed {{gap}} gives (it renders as a removed U+2060 spacer)
    correct = render_html(rendered_chapter_html(
        [HINDI_PROSE[i % len(HINDI_PROSE)] for i in range(len(GODAAN_BROKEN_GAPS))], qualities=(3,)))
    assert page.text == correct.text
    # the corpus cleaner applies it too
    assert clean_mediawiki_parse(parse_payload(html)) == page.text


def test_broken_gap_rule_leaves_look_alikes_alone() -> None:
    page = render_html(
        "<p>The gaps stay, and so does a gap. {gaps} and Gap year too.</p>"
        "<p>गाँव {{ और }} &lt;gaps&gt; बाकी रहे।</p>"
    )
    assert page.text == (
        "The gaps stay, and so does a gap. {gaps} and Gap year too.\n\n"
        "गाँव {{ और }} <gaps> बाकी रहे।"
    )
    assert BROKEN_GAP not in page.artifacts_removed  # anything else odd is left for inspection


# verbatim from the rendered पृष्ठ:गो-दान.djvu/२८ (hi.wikisource, 2026-09-24): the page calls
# {{GaP}}, which does not exist, so MediaWiki printed a red link to it
GAP_RED_LINK = (
    '<a href="/w/index.php?title=%E0%A4%B8%E0%A4%BE%E0%A4%81%E0%A4%9A%E0%A4%BE:GaP&amp;action=edit'
    '&amp;redlink=1" class="new" title="साँचा:GaP (पृष्ठ मौजूद नहीं है)">साँचा:GaP</a>'
)


def _red_link(title: str) -> str:
    """A red link as MediaWiki renders it for a page that does not exist."""
    target = quote(title.replace(" ", "_"), safe=":")
    return (f'<a href="/w/index.php?title={target}&amp;action=edit&amp;redlink=1" class="new"'
            f' title="{title} (page does not exist)">{title}</a>')


def test_red_links_to_missing_templates_are_removed_and_named() -> None:
    prose = [GAP_RED_LINK + HINDI_PROSE[0], _red_link("साँचा:GAP") + HINDI_PROSE[1],
             _red_link("साँचा:GAP") + HINDI_PROSE[2]]
    page = render_html(rendered_chapter_html(prose, qualities=(3,)))
    for residue in ("साँचा", "GaP", "GAP"):
        assert residue not in page.text, residue
    for paragraph in HINDI_PROSE[:3]:  # the transcription itself is untouched
        assert paragraph in page.text
    assert page.artifacts_removed[MISSING_TEMPLATE.format("साँचा:GaP")] == 1  # named, counted
    assert page.artifacts_removed[MISSING_TEMPLATE.format("साँचा:GAP")] == 2
    # exactly the text a correctly spelt {{gap}} gives (a U+2060 spacer, removed anyway)
    assert page.text == render_html(rendered_chapter_html(HINDI_PROSE[:3], qualities=(3,))).text
    assert clean_mediawiki_parse(parse_payload(rendered_chapter_html(prose, qualities=(3,)))) == page.text
    # Bengali Wikisource spells the namespace টেমপ্লেট; "Template" is valid on every wiki
    other = render_html(f"<p>{_red_link('টেমপ্লেট:Gpa')}আমার মাথা নত করে দাও</p>"
                        f"<p>{_red_link('Template:Gap indent')}The text.</p>")
    assert other.text == "আমার মাথা নত করে দাও\n\nThe text."
    assert other.artifacts_removed[MISSING_TEMPLATE.format("Template:Gap indent")] == 1


def test_other_red_links_and_existing_template_links_keep_their_text() -> None:
    page = render_html(
        f"<p>{_red_link('प्रेमचंद')} ने लिखा।</p>"  # a missing article
        f"<p>{_red_link('लेखक:प्रेमचंद')} की कहानी।</p>"  # a missing author page
        '<p><a href="/wiki/%E0%A4%B8%E0%A4%BE%E0%A4%81%E0%A4%9A%E0%A4%BE:Gap" title="साँचा:Gap">'
        "साँचा:Gap</a> नीला लिंक।</p>"  # a template that exists (a blue link)
    )
    assert page.text == "प्रेमचंद ने लिखा।\n\nलेखक:प्रेमचंद की कहानी।\n\nसाँचा:Gap नीला लिंक।"
    assert not any(key.startswith("link to missing template") for key in page.artifacts_removed)


def test_a_closing_brace_pair_that_closes_nothing_is_removed() -> None:
    # গীতাঞ্জলি, scan page ১৪৮ (2026-09-24): the page opens its block with {{Block center/s}}
    # and closes it with {{block center/e}}, yet ends the poem with "২৬ আষাঢ় ১৩১৭}}"
    page = render_html(
        '<div class="poem"><p>কে গো সেথায় স্নিগ্ধ দুনয়নে,<br>অনাদিকাল চাহে আমার তরে।</p></div>'
        "<p>২৬ আষাঢ় ১৩১৭}}\n</p><p>১১৯</p>"
    )
    assert "২৬ আষাঢ় ১৩১৭" in page.lines and "}" not in page.text
    assert page.artifacts_removed[STRAY_CLOSING_BRACES] == 1
    # a line with an opening brace or a template argument may be a broken template call:
    # its braces stay, for the inspection report to show
    kept = render_html("<p>{rh|১৩৬|গীতাঞ্জলি}}</p><p>যাত্রী | আমি}}</p>")
    assert kept.text == "{rh|১৩৬|গীতাঞ্জলি}}\n\nযাত্রী | আমি}}"
    assert STRAY_CLOSING_BRACES not in kept.artifacts_removed


def test_every_rendered_wiki_in_the_manifest_has_its_template_namespace_listed() -> None:
    # without it, a red link to a missing template on that wiki would stay in the text
    data = json.loads(REPO_MANIFEST.read_text(encoding="utf-8"))
    hosts = {urlsplit(source["source_url"]).hostname for source in data["sources"]
             if source.get("kind") == KIND}
    assert hosts, "the repository manifest declares rendered Wikisource sources"
    assert hosts <= set(TEMPLATE_NAMESPACE), f"add to TEMPLATE_NAMESPACE: {hosts - set(TEMPLATE_NAMESPACE)}"


def test_render_records_page_quality_and_flags_unproofread_pages() -> None:
    page = render_html(rendered_chapter_html(qualities=(4, 1, 2, None)))
    assert [level for _name, level in page.page_qualities] == [4, 1, 2, None]
    summary = page.quality_summary()
    assert summary["pages"] == 4
    assert summary["by_level"] == {"1": 1, "2": 1, "4": 1, "unknown": 1}
    assert [row["level"] for row in summary["unproofread"]] == [1, 2]
    assert render_html(rendered_chapter_html(qualities=(0, 3, 4))).unproofread_pages == []


def test_render_detects_redirects_and_contents_pages() -> None:
    assert render_html(redirect_html()).is_redirect is True
    toc = render_html(contents_page_html(12))
    assert toc.navigation_lines == 12  # every "chapter title + page number" row
    assert len(toc.lines) == 13  # plus the heading, which is not a link


# ---------------------------------------------------------------------------
# the API payload
# ---------------------------------------------------------------------------
def test_read_parse_payload_handles_versions_errors_and_truncation() -> None:
    html = rendered_chapter_html()
    v2 = read_parse_payload(parse_payload(html))
    v1 = read_parse_payload(parse_payload(html, formatversion=1))
    assert v2.ok and v1.ok and v2.html == v1.html == html
    assert v2.revid == 480196 and v2.categories == ["परीक्षा"] == v1.categories

    error = read_parse_payload(parse_error_payload())
    assert not error.ok and error.error_code == "missingtitle"

    cut = parse_payload(html)[:1500]
    assert read_parse_payload(cut).ok is False  # ingestion never accepts a cut payload
    sample = read_parse_payload(cut, allow_truncated=True)
    assert sample.ok and sample.truncated and sample.title == "परीक्षा-ग्रंथ/२"
    assert html.startswith(sample.html[:200])


def test_clean_mediawiki_parse_is_strict_and_deterministic() -> None:
    payload = parse_payload(rendered_chapter_html())
    assert clean_text(payload, KIND) == clean_mediawiki_parse(payload)
    assert sha256_text(clean_mediawiki_parse(payload)) == sha256_text(clean_mediawiki_parse(payload))
    assert clean_mediawiki_parse(parse_error_payload()) == ""
    assert clean_mediawiki_parse("<html>not json</html>") == ""
    assert clean_mediawiki_parse(payload[:1500]) == ""


# ---------------------------------------------------------------------------
# URL rules
# ---------------------------------------------------------------------------
def test_validate_parse_url_accepts_page_and_single_pages_tag_renders() -> None:
    assert validate_parse_url(PAGE_URL) == []
    assert validate_parse_url(RANGE_URL) == []


def test_validate_parse_url_rejects_everything_else() -> None:
    def problems(url: str) -> str:
        return " | ".join(validate_parse_url(url))

    assert "redirects" in problems(PAGE_URL + "&redirects=1")
    assert "format=json" in problems(PAGE_URL.replace("format=json", "format=xml"))
    assert "formatversion=2" in problems(PAGE_URL.replace("&formatversion=2", ""))
    assert "prop" in problems(PAGE_URL.replace("prop=text%7C", "prop="))
    assert "https" in problems(PAGE_URL.replace("https://", "http://"))
    assert "api.php" in problems(PAGE_URL.replace("/w/api.php", "/wiki/X"))
    assert "exactly one" in problems(PAGE_URL + "&pageid=5")
    assert "exactly one" in problems(PAGE_URL.split("&page=")[0])
    # text= must be exactly one <pages> tag: arbitrary text could never be "wiki content"
    assert "one ProofreadPage tag" in problems(RANGE_URL.split("&text=")[0] + "&text=Hello%20world")
    assert "one ProofreadPage tag" in problems(
        RANGE_URL + "%3Cpages%20index%3D%22H.djvu%22%20from%3D1%20to%3D2%20%2F%3E"
    )
    assert "from > to" in problems(RANGE_URL.replace("from%3D13", "from%3D200"))
    assert "title=" in problems(RANGE_URL.replace("&title=G", ""))
    assert "contentmodel=wikitext" in problems(RANGE_URL.replace("&contentmodel=wikitext", ""))
    assert "repeated" in problems(PAGE_URL + "&format=json")


def test_manifest_rejects_a_bad_mediawiki_parse_url(tmp_path: Path) -> None:
    good = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    assert validate_manifest(good) == []
    bad = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source(url=PAGE_URL + "&redirects=1")]))
    assert any("mediawiki-parse source_url must not use redirects" in p for p in validate_manifest(bad))


# ---------------------------------------------------------------------------
# ingestion gates
# ---------------------------------------------------------------------------
def test_ingest_verifies_a_rendered_chapter_with_site_evidence(tmp_path: Path, monkeypatch) -> None:
    payload = parse_payload(rendered_chapter_html())
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text", _fetcher({"action=parse": payload})
    )
    source = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()])).sources[0]
    item = ingest_source(source, tmp_path / "raw", fetch=True, license_evidence=_evidence())

    assert item.status == "verified" and item.licence_proof == "licence-evidence"
    text = Path(item.path).read_text(encoding="utf-8")
    assert item.sha256 == sha256_text(text) == sha256_text(clean_mediawiki_parse(payload))
    assert HINDI_PROSE[0] in text and "पीछे" not in text
    provenance = json.loads(Path(item.provenance_path).read_text(encoding="utf-8"))
    check = provenance["content_check"]
    assert check["page_quality"]["by_level"] == {"3": 1, "4": 1}
    assert check["mediawiki"]["revid"] == 480196 and check["mediawiki"]["has_prp_output"] is True
    assert check["artifacts_removed"]["U+FEFF ZERO WIDTH NO-BREAK SPACE"] == 1
    assert provenance["kind"] == KIND and provenance["hash_is_licence_proof"] is False


def test_ingest_treats_api_errors_and_cut_payloads_as_failed_fetches(tmp_path: Path, monkeypatch) -> None:
    source = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()])).sources[0]
    for body, expected in (
        (parse_error_payload(), "missingtitle"),
        (parse_payload(rendered_chapter_html())[:2000], "complete action=parse JSON"),
        ("<html>503 Service Unavailable</html>", "usable action=parse JSON"),
    ):
        monkeypatch.setattr(
            "frontier_ai.tokenization.research_corpus.fetch_text", _fetcher({"action=parse": body})
        )
        item = ingest_source(source, tmp_path / "raw", fetch=True, license_evidence=_evidence())
        assert item.status == "fetch_failed" and expected in item.error
        assert item.sha256 is None and item.path is None
    assert not (tmp_path / "raw" / "hi-chapter.txt").exists()


def test_ingest_refuses_a_rendered_redirect_contents_page_or_stub(tmp_path: Path, monkeypatch) -> None:
    source = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()])).sources[0]
    stub = rendered_chapter_html(["बस एक छोटी पंक्ति।"], qualities=(3,))
    cases = {
        "redirect": (redirect_html(), "#REDIRECT"),
        "contents": (contents_page_html(12), "contents/index page"),
        "stub": (stub, "rendered to only"),
    }
    for name, (html, expected) in cases.items():
        monkeypatch.setattr(
            "frontier_ai.tokenization.research_corpus.fetch_text",
            _fetcher({"action=parse": parse_payload(html)}),
        )
        item = ingest_source(source, tmp_path / name, fetch=True, license_evidence=_evidence())
        assert item.status == "index_page_refused", name
        assert expected in item.error, (name, item.error)
        assert not item.verified
    assert len(render_html(stub).text) < INDEX_PAGE_MIN_CLEANED_CHARS


def test_ingest_refuses_unproofread_scan_pages_and_keeps_the_text(tmp_path: Path, monkeypatch) -> None:
    payload = parse_payload(rendered_chapter_html(qualities=(3, 1, 4)))
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text", _fetcher({"action=parse": payload})
    )
    source = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()])).sources[0]
    item = ingest_source(source, tmp_path / "raw", fetch=True, license_evidence=_evidence())

    assert item.status == "unproofread_refused" and not item.verified
    assert "1 of the 3 scan pages" in item.error and "level 1: not proofread" in item.error
    assert Path(item.path).exists()  # kept so a human can read it
    provenance = json.loads(Path(item.provenance_path).read_text(encoding="utf-8"))
    # the licence was proven, but a refused source is never recorded as verified
    assert item.licence_proof == "licence-evidence"
    assert provenance["verification"] == "unverified"
    assert provenance["content_check"]["gate"] == "proofreading"


def test_build_excludes_unproofread_sources_and_counts_them(tmp_path: Path, monkeypatch) -> None:
    good = parse_payload(rendered_chapter_html(qualities=(3, 4)))
    ocr = parse_payload(rendered_chapter_html(qualities=(1,), chapter_label="३"))
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fetcher({"%E0%A5%A8": good, "%E0%A5%A9": ocr}),
    )
    path = _manifest(tmp_path, [
        _source("hi-ch2"),
        _source("hi-ch3", url=PAGE_URL.replace("%E0%A5%A8", "%E0%A5%A9")),
    ])
    result = build_corpus(path, tmp_path / "out", fetch=True, pin=True)
    statuses = {item.source_id: item.status for item in result.ingested}
    assert statuses == {"hi-ch2": "verified", "hi-ch3": "unproofread_refused"}
    assert {doc.source_id for doc in result.train + result.held_out} == {"hi-ch2"}
    assert result.acquisition["summary"]["unproofread_refused"] == 1
    pinned = {s.id: s.sha256 for s in TokenizerCorpusManifest.load(path).sources}
    assert pinned["hi-ch2"] and pinned["hi-ch3"] is None  # only what verified is pinned


def test_build_fetches_a_shared_licence_evidence_endpoint_once(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.fetch_text",
        _fetcher({"action=parse": parse_payload(rendered_chapter_html())}, calls),
    )
    sources = [
        _source(f"hi-ch{n}", url=PAGE_URL.replace("%E0%A5%A8", f"%E0%A5%A{n}")) for n in (2, 3, 4)
    ]
    result = build_corpus(_manifest(tmp_path, sources), tmp_path / "out", fetch=True)
    assert [item.status for item in result.ingested] == ["verified"] * 3
    assert sum("meta=siteinfo" in url for url in calls) == 1
    evidence = [item.license_evidence for item in result.ingested]
    assert "reused_within_run" not in evidence[0]
    assert evidence[1]["reused_within_run"] is True and evidence[2]["accepted"] is True


def test_build_retries_evidence_after_a_failed_check(tmp_path: Path, monkeypatch) -> None:
    """A transient evidence failure must not be cached for the rest of the build."""
    attempts = {"n": 0}
    payload = parse_payload(rendered_chapter_html())

    def _fetch(url: str, *_a, **_k) -> str:
        if "meta=siteinfo" in url:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("temporary failure")
            return SITEINFO
        return payload

    monkeypatch.setattr("frontier_ai.tokenization.research_corpus.fetch_text", _fetch)
    sources = [_source("hi-a"), _source("hi-b", url=PAGE_URL.replace("%E0%A5%A8", "%E0%A5%A9"))]
    result = build_corpus(_manifest(tmp_path, sources), tmp_path / "out", fetch=True)
    assert [item.status for item in result.ingested] == ["licence_marker_missing", "verified"]
    assert attempts["n"] == 2


def test_local_text_for_a_rendered_source_must_be_the_api_payload(tmp_path: Path) -> None:
    source = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()])).sources[0]
    plain = ingest_source(source, tmp_path / "raw", local_text="\n".join(HINDI_PROSE))
    assert plain.status == "empty" and "saved api.php?action=parse JSON" in plain.error
    saved = ingest_source(source, tmp_path / "raw", local_text=parse_payload(rendered_chapter_html()))
    assert saved.status == "local_unverified" and saved.chars > 0


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def _probe(body: str):
    def _fake(url: str, max_bytes: int = 65_536, timeout: float = 30.0) -> dict:
        payload = SITEINFO if "meta=siteinfo" in url else body
        raw = payload.encode("utf-8")[:max_bytes]
        return {
            "url": url, "ok": True, "http_status": 206, "content_type": "application/json",
            "bytes_read": len(raw), "elapsed_s": 0.01,
            "text": raw.decode("utf-8", errors="replace"), "error": "",
        }

    return _fake


def test_preflight_reads_a_truncated_rendered_chapter(tmp_path: Path, monkeypatch) -> None:
    long_chapter = rendered_chapter_html(HINDI_PROSE * 40, qualities=(3, 4, 4))
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.probe_url", _probe(parse_payload(long_chapter))
    )
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    report = preflight_manifest(manifest, max_bytes=8_000)
    row = report["sources"][0]
    assert row["sample_truncated"] is True and row["api_error"] is None
    assert row["looks_like_index_page"] is False and row["sample_lines"] > 3
    assert any("sampled the first 8,000 bytes" in note for note in row["notes"])
    rendered = render_preflight(report)
    assert "pages=" in rendered and "q3=1" in rendered


def test_preflight_warns_about_api_errors_and_unproofread_pages(tmp_path: Path, monkeypatch) -> None:
    manifest = TokenizerCorpusManifest.load(_manifest(tmp_path, [_source()]))
    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.probe_url", _probe(parse_error_payload())
    )
    report = preflight_manifest(manifest)
    assert "WARN APIERROR" in render_preflight(report)
    assert any("missingtitle" in note for note in report["sources"][0]["notes"])

    monkeypatch.setattr(
        "frontier_ai.tokenization.research_corpus.probe_url",
        _probe(parse_payload(rendered_chapter_html(qualities=(3, 1)))),
    )
    report = preflight_manifest(manifest)
    assert "WARN UNPROOFED" in render_preflight(report)
    assert any("unproofread_refused" in note for note in report["sources"][0]["notes"])


def test_content_shape_of_a_rendered_page_uses_the_shared_keys() -> None:
    shape = content_shape(parse_payload(rendered_chapter_html()), KIND)
    for key in ("sample_lines", "link_lines", "link_line_ratio", "is_redirect",
                "looks_like_index_page", "cleaned_chars_hint", "first_line_excerpt"):
        assert key in shape
    assert shape["first_line_excerpt"] == "२"
    assert shape["is_sparql_card"] is False and shape["wiki_links"] == 0


# ---------------------------------------------------------------------------
# the repository manifest: Godaan and Gitanjali are rendered, complete, not duplicated
# ---------------------------------------------------------------------------
DEVANAGARI_DIGITS = "०१२३४५६७८९"


def _devanagari(number: int) -> str:
    return "".join(DEVANAGARI_DIGITS[int(d)] for d in str(number))


def test_repo_manifest_declares_every_godaan_chapter_once_as_a_rendered_page() -> None:
    manifest = TokenizerCorpusManifest.load(REPO_MANIFEST)
    assert validate_manifest(manifest) == []
    hindi = manifest.sources_for("hi")
    assert len(hindi) == 36
    pages = []
    for source in hindi:
        assert source.kind == KIND
        params = parse_qs(urlsplit(source.source_url).query)
        assert "redirects" not in params
        pages.append(params["page"][0])
        evidence = manifest.evidence_for(source.id)
        assert evidence is not None and evidence.scope == "site"
    # the 36 chapters of the गो-दान.djvu transcription, in order, each exactly once
    assert pages == [f"गो-दान/{_devanagari(n)}" for n in range(1, 37)]
    # never the legacy pages built from a different scan (गोदान.pdf): duplicated text
    assert not any("भाग" in page for page in pages)
    assert len({source.source_url for source in manifest.sources}) == len(manifest.sources)


def _repo_source(manifest: TokenizerCorpusManifest, language: str, source_id: str):
    matches = [source for source in manifest.sources_for(language) if source.id == source_id]
    assert len(matches) == 1, f"{source_id} must be declared exactly once for {language}"
    return matches[0]


def test_repo_manifest_renders_all_gitanjali_poems_from_one_pages_tag() -> None:
    manifest = TokenizerCorpusManifest.load(REPO_MANIFEST)
    gitanjali = _repo_source(manifest, "bn", "bn-wikisource-gitanjali-1913-ccbysa")
    assert gitanjali.kind == KIND
    params = parse_qs(urlsplit(gitanjali.source_url).query)
    assert params["title"] == ["গীতাঞ্জলি (১৯১৩)"]
    tag = re.fullmatch(
        r'<pages index="গীতাঞ্জলি - রবীন্দ্রনাথ ঠাকুর\.djvu" from=(\d+) to=(\d+) />', params["text"][0]
    )
    assert tag is not None and (tag.group(1), tag.group(2)) == ("13", "190")


def _assert_pin_is_complete_or_absent(source) -> None:
    """Not pinned at all, or pinned the only way the code pins: build_corpus(pin=True) writes
    the hash, verified=true and the retrieval time together, for sources that passed every
    gate in that run. A hash without the rest (or the reverse) was typed in by hand."""
    if source.sha256 is None:
        assert source.verified is False and source.retrieved_at is None, source.id
    else:
        assert re.fullmatch(r"[0-9a-f]{64}", source.sha256), source.id
        assert source.verified is True, source.id
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", source.retrieved_at or ""), source.id


def test_repo_manifest_pins_are_complete_or_absent() -> None:
    for source in TokenizerCorpusManifest.load(REPO_MANIFEST).sources:
        _assert_pin_is_complete_or_absent(source)


def test_repo_manifest_renders_devdas_chapters_1_to_16_from_one_pages_tag() -> None:
    # scan pages 5–110 = chapter 1 (from=5) … chapter 16 (to=110), checked live on
    # 2026-09-24; pages 1–4 (cover, title, blank, the author's other books) stay out
    manifest = TokenizerCorpusManifest.load(REPO_MANIFEST)
    devdas = _repo_source(manifest, "bn", "bn-wikisource-devdas-ccbysa")
    assert devdas.kind == KIND
    _assert_pin_is_complete_or_absent(devdas)  # a hash only ever comes from a --pin run
    assert validate_parse_url(devdas.source_url) == []
    params = parse_qs(urlsplit(devdas.source_url).query)
    assert params["title"] == ["দেবদাস (শরৎচন্দ্র চট্টোপাধ্যায়)"]
    tag = re.fullmatch(
        r'<pages index="দেবদাস - শরৎচন্দ্র চট্টোপাধ্যায়\.pdf" from=(\d+) to=(\d+) />', params["text"][0]
    )
    assert tag is not None and (tag.group(1), tag.group(2)) == ("5", "110")
    evidence = manifest.evidence_for(devdas.id)
    assert evidence is not None and evidence.url.startswith("https://bn.wikisource.org/w/api.php?")
    assert "siprop=rightsinfo" in evidence.url


def test_repo_manifest_language_slots_list_the_second_english_and_bengali_works() -> None:
    manifest = TokenizerCorpusManifest.load(REPO_MANIFEST)
    assert [s.id for s in manifest.sources_for("bn")] == [
        "bn-wikisource-gitanjali-1913-ccbysa",
        "bn-wikisource-devdas-ccbysa",
    ]
    assert [s.id for s in manifest.sources_for("en")] == [
        "en-gutenberg-alice-pd",
        "en-gutenberg-sadhana-pd",
    ]
    sadhana = _repo_source(manifest, "en", "en-gutenberg-sadhana-pd")
    assert sadhana.kind == "gutenberg" and sadhana.license_id == "PD-US"
    assert sadhana.source_url == "https://www.gutenberg.org/cache/epub/6842/pg6842.txt"
    _assert_pin_is_complete_or_absent(sadhana)
    slots = json.loads(REPO_MANIFEST.read_text(encoding="utf-8"))["language_slots"]
    declared = {slot["code"]: slot["sources"] for slot in slots}
    assert declared["bn"] == [s.id for s in manifest.sources_for("bn")]
    assert declared["en"] == [s.id for s in manifest.sources_for("en")]


# ---------------------------------------------------------------------------
# scripts/discover_wiki_chapters.py (offline, with canned API answers)
# ---------------------------------------------------------------------------
def _load_discovery_script():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "discover_wiki_chapters.py"
    spec = importlib.util.spec_from_file_location("discover_wiki_chapters", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_discovery_orders_chapters_and_flags_legacy_scans(tmp_path: Path, monkeypatch) -> None:
    module = _load_discovery_script()
    # allpages returns titles in code-point order (१, १०, ११, २, …), like the real API
    titles = ["W/ भाग 2", "W/१", "W/१०", "W/२", "W/३", "W/४", "W/५", "W/६", "W/७", "W/८", "W/९"]
    wikitext = {f"W/{_devanagari(n)}": f'<pages\nheader=1\nindex="W.djvu"\nfrom={n}{"१" if n == 1 else ""}\nto={n + 1}\n/>'
                for n in range(1, 11)}
    wikitext["W/ भाग 2"] = '<pages header=1 index="W-old.pdf" from=150 to=165 />'

    def _fake_get(url: str, timeout: float = 30.0) -> dict:
        if "list=allpages" in url:
            return {"query": {"allpages": [{"title": title} for title in titles]}}
        return {"query": {"pages": [
            {"title": title, "revisions": [{"slots": {"main": {"content": wikitext[title]}}}]}
            for title in wikitext
        ]}}

    monkeypatch.setattr(module, "_get", _fake_get)
    report = module.discover("hi", "W")
    assert report["scan"] == "W.djvu"
    assert [row["number"] for row in report["clean"]] == list(range(1, 11))
    legacy = next(row for row in report["chapters"] if row["title"] == "W/ भाग 2")
    assert legacy["flags"] and "different scan" in legacy["flags"][0]
    assert report["chapters"][0]["from"] == "1१"  # recorded verbatim, never "fixed"

    entries = module.propose_sources(report, "hi-wikisource-w", work_label="W (test)")
    assert [entry["id"] for entry in entries][:2] == ["hi-wikisource-w-ch01-ccbysa", "hi-wikisource-w-ch02-ccbysa"]
    for entry in entries:
        assert validate_parse_url(entry["source_url"]) == []
        assert entry["sha256"] is None and entry["verified"] is False
    assert not any("भाग" in entry["source_url"] for entry in entries)

    out = tmp_path / "proposed.json"
    assert module.main(["--lang", "hi", "--work", "W", "--emit-sources", "hi-wikisource-w",
                        "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))[0]["kind"] == KIND


def test_discovery_reads_digits_in_any_script() -> None:
    module = _load_discovery_script()
    assert module.chapter_number("गो-दान/३६") == 36
    assert module.chapter_number("গীতাঞ্জলি (১৯১৩)/১৫৭") == 157
    assert module.chapter_number("गो-दान/ भाग 16") == 16
    assert module.as_number("३2") == 32 and module.as_number("1१") == 11
