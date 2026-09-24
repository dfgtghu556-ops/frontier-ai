"""Rendered MediaWiki pages as corpus text: the ``mediawiki-parse`` source kind.

Why this exists
---------------
Most Indic Wikisource works are transcriptions of *scanned books* (the ProofreadPage
extension). The main-namespace page a reader opens — ``गो-दान/२``, ``গীতাঞ্জলি (১৯১৩)/১`` —
contains no prose of its own, only a tag such as::

    <pages index="गो-दान.djvu" from=18 fromsection="1" to=२२ tosection="1" />

The text lives on one ``Page:`` page per scan and exists as a whole only after MediaWiki
has expanded that tag (joining pages, honouring ``fromsection``/``tosection``, reading
Devanagari or Bengali digits, applying templates). ``?action=raw`` therefore returns about
150 characters of markup and never the work. Re-implementing ProofreadPage would be a
second, divergent copy of that logic, so this kind asks MediaWiki itself to render the
page (``api.php?action=parse``) and converts the rendered HTML to text. What we store is
what a reader of the wiki sees — minus the parts Wikisource itself marks as not content.

The conversion rules (deterministic, reviewable, tested)
--------------------------------------------------------
* **Dropped, with everything inside them:** ``<style>`` / ``<script>`` (TemplateStyles
  CSS is emitted inline in the body), elements styled ``display:none``, and elements
  carrying a class in :data:`SKIP_CLASSES`. The most important of those is
  ``ws-noexport`` — Wikisource's own "do not export" marker, used by its official
  e-book exporter; the header template (← previous · title · author · next →), the
  hidden ``ws-data`` metadata and the inner page-number anchors all carry it. The rest
  are page-number anchors, edit links and footnote markers/lists (the wikitext cleaner
  drops ``<ref>`` for the same reason).
* **Kept:** every other piece of text. ``<br>`` becomes a line break; block elements
  (``p``, ``div``, ``li``, headings, table rows, …) become paragraph breaks, so poetry
  keeps one line per verse line and prose keeps one line per paragraph.
* **Page joins:** many transcriptions start every scanned page with ``<br>`` although
  the printed sentence (sometimes the printed *word*: ``अधि|कार``) simply continues from
  the previous page. A ``<br>`` that comes immediately after a page-number anchor —
  nothing visible in between — is therefore that page-join convention, not a printed
  line break, and becomes a space. Inside a ``<poem>`` block every ``<br>`` is a verse
  line and is always kept. Each conversion is counted.
* **Rendering artifacts removed:** U+200B ZERO WIDTH SPACE, U+2060 WORD JOINER and
  U+FEFF ZERO WIDTH NO-BREAK SPACE. On Wikisource they come from page joins, the
  ``{{gap}}`` indent template and byte-order marks left in OCR imports; none of them is
  part of the printed text. U+00A0 NO-BREAK SPACE (``&nbsp;``/``&#160;`` used for
  layout) becomes an ordinary space. Every removal is counted and recorded.
* **Mistyped templates removed:** MediaWiki prints what it cannot parse as text, and
  renders a call to a template that does not exist as a red link to it. Three such traces
  are removed, each counted: the verbatim residue of a mistyped ``{{gap}}`` (``{{Gap{}``,
  ``<gap>`` …); a red link into the Template namespace (``साँचा:GaP`` — the link names the
  missing template, never the work; see :data:`TEMPLATE_NAMESPACE`); and a ``}}`` that
  closes nothing, on a line without ``{`` or ``|``. Everything else odd stays in the text
  for the inspection report to show.
* **Never touched:** U+200C ZERO WIDTH NON-JOINER and U+200D ZERO WIDTH JOINER. In Indic
  scripts they decide how a conjunct is written; removing them would change the text.
  No Unicode normalization (NFC/NFD/NFKC) is applied — that remains a later-stage
  decision, exactly as for the other kinds.

Integrity rules for the URL
---------------------------
:func:`validate_parse_url` keeps a ``mediawiki-parse`` source honest: ``format=json``,
``formatversion=2`` and ``prop=text`` are required; ``redirects`` is forbidden (a redirect
is refused, never silently followed); and when the page is rendered from ``text=`` rather
than ``page=``, the text must be **exactly one** ProofreadPage ``<pages index="…" from=N
to=M />`` tag, so nothing except the wiki's own transcription can enter the corpus.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

MEDIAWIKI_PARSE_KIND = "mediawiki-parse"

# Elements whose whole subtree is never text.
SKIP_TAGS = frozenset({"style", "script", "noscript", "template", "head", "title"})

# Classes whose whole subtree is not part of the work. See the module docstring.
SKIP_CLASSES = frozenset(
    {
        "ws-noexport",        # Wikisource: header template, ws-data metadata, anchors
        "noprint",            # navigation boxes, never printed
        "pagenum",            # ProofreadPage page-number anchor (outer span)
        "ws-pagenum",
        "mw-editsection",     # "[edit]" links (also disabled via disableeditsection=1)
        "reference",          # footnote marker <sup class="reference">[१]</sup>
        "references",         # the footnote list
        "mw-references-wrap",
        "mw-cite-backlink",
    }
)

VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
     "param", "source", "track", "wbr"}
)
BLOCK_TAGS = frozenset(
    {"address", "article", "aside", "blockquote", "caption", "center", "dd", "div", "dl",
     "dt", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header",
     "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "tfoot", "thead",
     "tr", "ul"}
)
CELL_TAGS = frozenset({"td", "th"})

# Invisible characters that are rendering artifacts on Wikisource, never printed text.
ARTIFACT_CHARS: dict[str, str] = {
    "\u200b": "U+200B ZERO WIDTH SPACE",
    "\u2060": "U+2060 WORD JOINER",
    "\ufeff": "U+FEFF ZERO WIDTH NO-BREAK SPACE",
}
NBSP = "\u00a0"
# Deliberately NOT artifacts: they are orthographic in Indic scripts.
PRESERVED_JOINERS = ("\u200c", "\u200d")
PAGE_JOIN_BREAK = "page-join <br> -> space"
BROKEN_GAP = "broken {{gap}} template -> removed"
STRAY_CLOSING_BRACES = "stray }} -> removed"
LITERAL_POEM_TAG = "literal <poem> tag -> removed"
MISSING_TEMPLATE = "link to missing template {} -> removed"  # .format(template title)

# The Template namespace (number 10) as each wiki spells it in page titles and links —
# needed to recognise a call to a template that does not exist (see _missing_template).
# Checked 2026-09-24 in the API's own answers: hi lists the missing "साँचा:GAP" in namespace
# 10 (generator=templates), bn lists "টেমপ্লেট:Gap" in namespace 10 (action=parse
# &prop=templates). Add a wiki here before declaring sources from it: a test checks that
# every mediawiki-parse host in the repository manifest is listed.
TEMPLATE_NAMESPACE = {
    "hi.wikisource.org": "साँचा",
    "bn.wikisource.org": "টেমপ্লেট",
    # checked 2026-09-24 with meta=siteinfo&siprop=namespaces (namespace 10)
    "gu.wikisource.org": "ઢાંચો",
    "ml.wikisource.org": "ഫലകം",
    "or.wikisource.org": "ଛାଞ୍ଚ",
    "as.wikisource.org": "সাঁচ",
}
_TEMPLATE_PREFIXES = frozenset({"Template", *TEMPLATE_NAMESPACE.values()})

# ProofreadPage quality levels (the numbers are the extension's, the names ours).
QUALITY_NAMES = {
    0: "without text",
    1: "not proofread",
    2: "problematic",
    3: "proofread",
    4: "validated",
}
# Pages at these levels have text that no human has confirmed against the scan.
UNPROOFREAD_LEVELS = frozenset({1, 2})
# Each wiki renders ProofreadPage's page anchor with its own template, so the anchor comes
# in three shapes (all seen on 2026-09-24):
# * hi, bn, as: <span class="pagenum ws-pagenum" data-page-name="…" data-page-quality="4">
# * gu, or: the same span with only title="<percent-encoded page title>" — no name
#   attribute and no level;
# * ml: an older template, <span id="pr_page">[ <a class="prp-pagequality-4"
#   title="താൾ:…/4">4</a> ]</span>, whose "[ 4 ]" is a page number, not the work.
# A page whose level the render does not show is looked up through the wiki's API
# (page_quality_urls) and refused if it cannot be confirmed; it is never assumed proofread.
PR_PAGE_ID = "pr_page"
_PAGE_QUALITY_CLASS = re.compile(r"^prp-pagequality-(\d)$")

_DISPLAY_NONE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
# \s in a str pattern is Unicode-aware: it matches NBSP and the other space separators,
# but not U+200B/U+200C/U+200D/U+2060/U+FEFF (they are format characters, not spaces).
_WHITESPACE = re.compile(r"\s+")
_ARTIFACTS = re.compile("[" + "".join(ARTIFACT_CHARS) + "]")
# A mistyped {{gap}} (Wikisource's paragraph-indent template) that MediaWiki cannot parse
# is printed verbatim: "{{Gap{}", "{{Gap}]", "{{Gap]}", "{Gap}}", "{{Gap))", "{{Gap@))",
# "{{Gap" running straight into the text, or "<gap>" typed as a tag (all found in गो-दान
# on 2026-09-24, 20 times on 19 scan pages). A correctly typed {{gap}} renders as a U+2060
# spacer, which is removed anyway, so dropping the residue yields exactly the text a
# correctly typed page yields — and the text no longer changes when someone fixes the typo
# on the wiki. The pattern needs a brace or "<" right before the word and never takes a
# longer Latin word ("gaps"); anything else odd is left for the inspection report to show.
_BROKEN_GAP = re.compile(
    r"\{\{?[ \t]*[Gg]ap(?![A-Za-z])[ \t]*[\]\)\}\{@|]*"
    r"|</?[ \t]*[Gg]ap[ \t]*/?>"
)
# A "}}" that closes nothing is printed as text too: গীতাঞ্জলি's scan page ১৪৮ (checked
# 2026-09-24) ends a poem with "২৬ আষাঢ় ১৩১৭}}" although the page opens its block with
# {{Block center/s}} and closes it with {{block center/e}}. It is removed only from a line
# with no "{" and no "|" — there it carries nothing; a line with an opening brace or a
# template argument may be a broken template call, which stays for the inspection to show.
_STRAY_CLOSING_BRACES = re.compile(r"\}{2,}")
# A <poem> tag MediaWiki could not pair is printed as text: ଛମାଣ ଆଠଗୁଣ୍ଠ's scan page ୧୩୩
# (checked 2026-09-24) closes a verse block with "<poem/>" instead of "</poem>", so the
# opening "<poem>" appears in the middle of the text. A paired tag always renders as a
# <div class="poem">, never as text, so the literal tag is residue and nothing else; the
# lines it was meant to lay out stay as MediaWiki joined them.
_LITERAL_POEM_TAG = re.compile(r"<[ \t]*/?[ \t]*poem\b[^<>]*>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# the API payload
# ---------------------------------------------------------------------------
@dataclass
class ParsePayload:
    """What an ``action=parse`` JSON response contained (or why it could not be read)."""

    ok: bool
    html: str = ""
    title: str = ""
    pageid: int | None = None
    revid: int | None = None
    categories: list[str] = field(default_factory=list)
    error_code: str = ""
    error_info: str = ""
    # True when the payload was not complete JSON and the HTML was recovered from a
    # prefix (a preflight sample). Never acceptable for ingestion.
    truncated: bool = False


_TEXT_KEY = re.compile(r'"text"\s*:\s*"')
_TITLE_KEY = re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"')


def read_parse_payload(raw: str, *, allow_truncated: bool = False) -> ParsePayload:
    """Read an ``api.php?action=parse&format=json`` response.

    Accepts ``formatversion=2`` (``text`` is a string) and ``formatversion=1``
    (``text`` is ``{"*": …}``). An API error object (``{"error": {"code": …}}``) is
    reported, not raised. With ``allow_truncated`` a payload that is not complete JSON
    — the first N bytes read by preflight — is recovered as far as possible and marked
    ``truncated``; ingestion never allows that.
    """
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        if allow_truncated:
            return _read_prefix(raw)
        return ParsePayload(
            ok=False, error_code="malformed", error_info="the response is not complete JSON"
        )
    if not isinstance(document, dict):
        return ParsePayload(ok=False, error_code="malformed", error_info="the response is not a JSON object")
    if "error" in document:
        error = document.get("error") or {}
        return ParsePayload(
            ok=False,
            error_code=str(error.get("code", "unknown")),
            error_info=str(error.get("info", "")),
        )
    parse = document.get("parse")
    if not isinstance(parse, dict):
        return ParsePayload(ok=False, error_code="malformed", error_info="no 'parse' object in the response")
    text = parse.get("text")
    if isinstance(text, dict):  # formatversion=1
        text = text.get("*")
    if not isinstance(text, str):
        return ParsePayload(
            ok=False,
            error_code="malformed",
            error_info="no rendered 'text' in the response (is prop=text set?)",
        )
    categories: list[str] = []
    for item in parse.get("categories") or []:
        if isinstance(item, dict):
            name = item.get("category") or item.get("*")
            if name:
                categories.append(str(name))
    pageid = parse.get("pageid")
    revid = parse.get("revid")
    return ParsePayload(
        ok=True,
        html=text,
        title=str(parse.get("title", "")),
        pageid=pageid if isinstance(pageid, int) else None,
        revid=revid if isinstance(revid, int) else None,
        categories=categories,
    )


def _read_prefix(raw: str) -> ParsePayload:
    """Recover the rendered HTML from the first bytes of a response (preflight only)."""
    match = _TEXT_KEY.search(raw)
    if match is None:
        return ParsePayload(
            ok=False,
            error_code="malformed",
            error_info="the sample is not JSON and contains no rendered text",
            truncated=True,
        )
    title_match = _TITLE_KEY.search(raw, 0, match.start())
    title = _decode_json_string(title_match.group(1)) if title_match else ""
    return ParsePayload(
        ok=True, html=_decode_json_string(raw[match.end():]), title=title, truncated=True
    )


def _decode_json_string(fragment: str) -> str:
    """Decode a JSON string body that may be cut anywhere (even mid-escape)."""
    end = None
    index = 0
    while index < len(fragment):
        char = fragment[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            end = index
            break
        index += 1
    body = fragment if end is None else fragment[:end]
    decoder = json.JSONDecoder(strict=False)
    for cut in range(0, 7):  # a cut escape is at most 6 characters (\uXXXX)
        candidate = body[: len(body) - cut] if cut else body
        try:
            return decoder.decode('"' + candidate + '"')
        except (json.JSONDecodeError, ValueError):
            continue
    return body


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------
@dataclass
class RenderedPage:
    """The text of a rendered page plus the facts the content gates need."""

    text: str
    lines: list[str]
    # lines whose visible text is almost entirely link text (a table of contents)
    navigation_lines: int
    # (page name, quality level or None) for every transcluded scan page, in order
    page_qualities: list[tuple[str, int | None]]
    is_redirect: bool
    has_prp_output: bool  # ProofreadPage transcluded content is present
    artifacts_removed: dict[str, int]

    @property
    def unproofread_pages(self) -> list[tuple[str, int]]:
        return [
            (name, level)
            for name, level in self.page_qualities
            if level is not None and level in UNPROOFREAD_LEVELS
        ]

    def quality_summary(self) -> dict[str, Any]:
        counts = Counter(
            "unknown" if level is None else str(level) for _name, level in self.page_qualities
        )
        return {
            "pages": len(self.page_qualities),
            # the scan pages actually rendered, so a range can be checked, not assumed
            "first_page": self.page_qualities[0][0] if self.page_qualities else None,
            "last_page": self.page_qualities[-1][0] if self.page_qualities else None,
            "by_level": dict(sorted(counts.items())),
            "levels": {str(level): name for level, name in QUALITY_NAMES.items()},
            "unproofread": [
                {"page": name, "level": level, "meaning": QUALITY_NAMES[level]}
                for name, level in self.unproofread_pages
            ],
            # pages the render names but whose level it does not show (gu, or): the build
            # asks the wiki's API for them and refuses the source if it cannot confirm them
            "unknown_level": [name for name, level in self.page_qualities if level is None],
        }


class _Extractor(HTMLParser):
    """Stream the rendered HTML into text tokens, skipping non-content subtrees."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # ("text", data, in_link) | ("br", "", False) | ("block", "", False) | ("cell", "", False)
        self.tokens: list[tuple[str, str, bool]] = []
        # (tag, skipped, is_poem, is_pr_page)
        self._stack: list[tuple[str, bool, bool, bool]] = []
        self._link_depth = 0
        self._poem_depth = 0
        self._pr_page_depth = 0  # inside an old-style <span id="pr_page"> page anchor
        # True between a page-number anchor and the next visible text or block boundary
        self._after_page_anchor = False
        self.page_join_breaks = 0
        self.page_qualities: list[tuple[str, int | None]] = []
        self._seen_pages: set[str] = set()
        self.is_redirect = False
        self.has_prp_output = False
        self.missing_templates: Counter[str] = Counter()  # red links to missing templates

    @property
    def _skipping(self) -> bool:
        return bool(self._stack) and self._stack[-1][1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: (value or "") for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        self._observe(tag, attributes, classes)
        is_pr_page = attributes.get("id", "") == PR_PAGE_ID
        if (classes & {"pagenum", "ws-pagenum"} or is_pr_page) and not self._skipping:
            self._after_page_anchor = True
        if tag in VOID_TAGS:
            if not self._skipping:
                if tag == "br":
                    self._line_break()
                elif tag == "hr":
                    self._boundary("hr")
            return
        missing = (
            _missing_template(attributes.get("href", "")) if tag == "a" and "new" in classes else None
        )
        if missing and not self._skipping:
            self.missing_templates[missing] += 1
        skipped = (
            self._skipping
            or tag in SKIP_TAGS
            or bool(classes & SKIP_CLASSES)
            or bool(_DISPLAY_NONE.search(attributes.get("style", "")))
            or missing is not None
            or is_pr_page
        )
        is_poem = not skipped and "poem" in classes
        self._stack.append((tag, skipped, is_poem, is_pr_page))
        if is_pr_page:
            self._pr_page_depth += 1
        if not skipped:
            self._boundary(tag)
            if tag == "a":
                self._link_depth += 1
            if is_poem:
                self._poem_depth += 1

    def _line_break(self) -> None:
        if self._after_page_anchor and self._poem_depth == 0:
            # the page-join convention: the printed line continues across the page
            self.tokens.append(("text", " ", False))
            self.page_join_breaks += 1
            self._after_page_anchor = False
        else:
            self.tokens.append(("br", "", False))

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        for depth in range(len(self._stack) - 1, -1, -1):
            if self._stack[depth][0] == tag:
                break
        else:
            return  # a stray end tag: ignore it rather than unbalance the stack
        while len(self._stack) > depth:
            open_tag, skipped, is_poem, is_pr_page = self._stack.pop()
            if is_pr_page:
                self._pr_page_depth = max(0, self._pr_page_depth - 1)
            if not skipped:
                if open_tag == "a":
                    self._link_depth = max(0, self._link_depth - 1)
                if is_poem:
                    self._poem_depth = max(0, self._poem_depth - 1)
                self._boundary(open_tag)

    def handle_data(self, data: str) -> None:
        if data and not self._skipping:
            self.tokens.append(("text", data, self._link_depth > 0))
            if _visible(data):
                self._after_page_anchor = False

    def _boundary(self, tag: str) -> None:
        if tag in BLOCK_TAGS or tag == "hr":
            self.tokens.append(("block", "", False))
            self._after_page_anchor = False
        elif tag in CELL_TAGS:
            self.tokens.append(("cell", "", False))

    def _observe(self, tag: str, attributes: dict[str, str], classes: set[str]) -> None:
        """Facts recorded even from elements whose text is skipped."""
        if classes & {"pagenum", "ws-pagenum"}:
            # the name attribute when the wiki's template has it (hi, bn, as), else the
            # title attribute, which holds the percent-encoded page title (gu, or) — but
            # only on ProofreadPage's own anchor (it always carries data-page-number) and
            # only when the title looks like a scan page ("<namespace>:<file>/<n>")
            name = attributes.get("data-page-name", "").strip()
            if not name and "data-page-number" in attributes:
                candidate = _page_title_from_attribute(attributes.get("title", ""))
                if ":" in candidate and "/" in candidate.partition(":")[2]:
                    name = candidate
            if name:
                raw_level = attributes.get("data-page-quality", "").strip()
                self._record_page(name, int(raw_level) if raw_level.isdigit() else None)
        elif tag == "a" and self._pr_page_depth:
            # old-style anchor (ml): the level is the link's prp-pagequality-N class
            levels = [m.group(1) for m in map(_PAGE_QUALITY_CLASS.match, classes) if m]
            href = attributes.get("href", "")
            name = attributes.get("title", "").strip() or (
                _page_title_from_attribute(href.split("/wiki/", 1)[1]) if "/wiki/" in href else ""
            )
            if name:
                self._record_page(name, int(levels[0]) if len(levels) == 1 else None)
        if classes & {"redirectMsg", "redirectText"}:
            self.is_redirect = True
        if "prp-pages-output" in classes:
            self.has_prp_output = True

    def _record_page(self, name: str, level: int | None) -> None:
        if name not in self._seen_pages:
            self._seen_pages.add(name)
            self.page_qualities.append((name, level))


def _page_title_from_attribute(value: str) -> str:
    """A page title from a title/href attribute: percent-decoded, underscores as spaces.

    (HTMLParser has already turned ``&#95;`` into ``_``.)
    """
    return unquote(value).replace("_", " ").strip()


def _missing_template(href: str) -> str | None:
    """The title of the template a red link points to, when it is a missing template.

    When a page calls a template that does not exist (a misspelt name such as ``{{GaP}}``
    for ``{{Gap}}``), MediaWiki renders a red link to it instead — on hi.wikisource
    ``<a href="/w/index.php?title=साँचा:GaP&action=edit&redlink=1" class="new" …>साँचा:GaP</a>``
    (पृष्ठ:गो-दान.djvu/२८, 2026-09-24). That link is never text of the work; a correctly spelt
    ``{{gap}}`` renders as a U+2060 spacer, which is removed anyway. Red links into any other
    namespace (a missing article, an author page) keep their text.
    """
    query = parse_qs(urlsplit(href).query)
    if query.get("redlink") != ["1"]:
        return None
    title = (query.get("title") or [""])[0].replace("_", " ")
    namespace, colon, name = title.partition(":")
    return title if colon and name.strip() and namespace in _TEMPLATE_PREFIXES else None


def _visible(text: str) -> int:
    """Characters that are neither whitespace nor rendering artifacts."""
    return sum(1 for char in text if not char.isspace() and char not in ARTIFACT_CHARS)


def render_html(html: str, *, min_prose_chars: int = 20) -> RenderedPage:
    """Convert rendered MediaWiki HTML to corpus text (see the module docstring).

    ``min_prose_chars`` defines a navigation line: one that contains link text and has
    fewer than this many visible characters *outside* links — the HTML twin of the
    wikitext gate's "a link line with almost no prose left".
    """
    extractor = _Extractor()
    extractor.feed(html)
    extractor.close()

    artifacts: Counter[str] = Counter()
    if extractor.page_join_breaks:
        artifacts[PAGE_JOIN_BREAK] = extractor.page_join_breaks
    for title, count in extractor.missing_templates.items():
        artifacts[MISSING_TEMPLATE.format(title)] = count
    for kind, data, _in_link in extractor.tokens:
        if kind == "text":
            for char in data:
                if char in ARTIFACT_CHARS:
                    artifacts[ARTIFACT_CHARS[char]] += 1
                elif char == NBSP:
                    artifacts["U+00A0 NO-BREAK SPACE -> space"] += 1

    lines: list[str] = []  # "" marks a paragraph break
    navigation = 0
    current: list[tuple[str, bool]] = []

    def paragraph_break() -> None:
        if lines and lines[-1] != "":
            lines.append("")

    def flush_line() -> bool:
        nonlocal navigation
        joined = _ARTIFACTS.sub("", "".join(segment for segment, _link in current))
        joined, broken_gaps = _BROKEN_GAP.subn("", joined)
        if broken_gaps:
            artifacts[BROKEN_GAP] += broken_gaps
        joined, poem_tags = _LITERAL_POEM_TAG.subn("", joined)
        if poem_tags:
            artifacts[LITERAL_POEM_TAG] += poem_tags
        if "{" not in joined and "|" not in joined:
            joined, stray = _STRAY_CLOSING_BRACES.subn("", joined)
            if stray:
                artifacts[STRAY_CLOSING_BRACES] += stray
        cleaned = _WHITESPACE.sub(" ", joined).strip()
        linked = sum(_visible(segment) for segment, is_link in current if is_link)
        total = sum(_visible(segment) for segment, _link in current)
        current.clear()
        if not cleaned:
            return False
        lines.append(cleaned)
        if linked and total - linked < min_prose_chars:
            navigation += 1
        return True

    for kind, data, in_link in extractor.tokens:
        if kind == "text":
            current.append((data, in_link))
        elif kind == "cell":
            current.append((" ", False))
        elif kind == "br":
            if not flush_line():
                paragraph_break()  # an empty line (<br><br>) separates stanzas
        else:  # block
            flush_line()
            paragraph_break()
    flush_line()

    while lines and lines[-1] == "":
        lines.pop()
    text = "\n".join(lines)
    return RenderedPage(
        text=text,
        lines=[line for line in lines if line],
        navigation_lines=navigation,
        page_qualities=list(extractor.page_qualities),
        is_redirect=extractor.is_redirect,
        has_prp_output=extractor.has_prp_output,
        artifacts_removed=dict(sorted(artifacts.items())),
    )


def clean_mediawiki_parse(raw: str) -> str:
    """The ``mediawiki-parse`` cleaner: API JSON in, corpus text out.

    Strict: a response that is not complete, error-free ``action=parse`` JSON cleans to
    the empty string (the content gate reports *why* separately), so a broken download
    can never be mistaken for a short text.
    """
    payload = read_parse_payload(raw)
    if not payload.ok:
        return ""
    return render_html(payload.html).text


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------
PARSE_REQUIRED_PARAMS = {"action": "parse", "format": "json", "formatversion": "2"}
# ``redirects`` would silently follow a redirect to a different title; ``pst``/``onlypst``
# pre-save-transform the input; none of them belongs in a corpus source.
PARSE_FORBIDDEN_PARAMS = frozenset({"redirects", "pst", "onlypst", "summary", "preview"})
PARSE_TARGET_PARAMS = ("page", "pageid", "oldid", "text")
PAGES_TAG = re.compile(
    r'<pages\s+index="(?P<index>[^"<>]+)"\s+from=(?P<start>\d+)\s+to=(?P<end>\d+)'
    r"(?:\s+header=(?:0|1))?\s*/>"
)


def validate_parse_url(url: str) -> list[str]:
    """Problems with a ``mediawiki-parse`` source URL (empty means acceptable)."""
    problems: list[str] = []
    parts = urlsplit(url)
    if parts.scheme != "https":
        problems.append("must be https")
    if not parts.path.endswith("/api.php"):
        problems.append("must point at a MediaWiki api.php endpoint")
    query = parse_qs(parts.query, keep_blank_values=True)
    for key, values in sorted(query.items()):
        if len(values) > 1:
            problems.append(f"parameter {key!r} is repeated")
    params = {key: values[-1] for key, values in query.items()}
    for key, expected in PARSE_REQUIRED_PARAMS.items():
        if params.get(key) != expected:
            problems.append(f"needs {key}={expected}")
    if "text" not in params.get("prop", "").split("|"):
        problems.append("needs prop to include 'text'")
    for key in sorted(PARSE_FORBIDDEN_PARAMS & set(params)):
        problems.append(
            f"must not use {key}= (redirects are refused, not followed; the source must "
            "name the page that holds the work)"
            if key == "redirects"
            else f"must not use {key}="
        )
    targets = [key for key in PARSE_TARGET_PARAMS if key in params]
    if len(targets) != 1:
        problems.append("needs exactly one of page=, pageid=, oldid= or text=")
    if "text" in params:
        match = PAGES_TAG.fullmatch(params["text"].strip())
        if match is None:
            problems.append(
                'text= may only contain one ProofreadPage tag of the form '
                '<pages index="…" from=N to=M /> (so nothing but wiki content can enter)'
            )
        elif int(match.group("start")) > int(match.group("end")):
            problems.append("text= <pages> tag has from > to")
        if not params.get("title"):
            problems.append("text= needs title= (the work page the pages belong to)")
        if params.get("contentmodel") != "wikitext":
            problems.append("text= needs contentmodel=wikitext")
    return problems


# ---------------------------------------------------------------------------
# canonical URLs (what the manifest should contain)
# ---------------------------------------------------------------------------
_COMMON_PARAMS = "action=parse&format=json&formatversion=2"
_QUIET_PARAMS = "disablelimitreport=1&disableeditsection=1"


def parse_page_url(host: str, title: str) -> str:
    """The ``mediawiki-parse`` URL that renders one wiki page (e.g. a chapter).

    ``prop=text|revid|categories``: the rendered page, the revision that was rendered
    (recorded in provenance) and its categories (e.g. a "missing scan pages" tracking
    category). ``disablelimitreport`` drops the parser's timing comment, so the payload
    does not change between two fetches of the same revision.
    """
    return (
        f"https://{host}/w/api.php?{_COMMON_PARAMS}&prop={quote('text|revid|categories')}"
        f"&{_QUIET_PARAMS}&page={quote(title, safe='')}"
    )


def parse_pages_range_url(host: str, work_title: str, index: str, start: int, end: int) -> str:
    """The URL that renders scan pages ``start``–``end`` of ``index`` in one response.

    For works whose parts are too short to be sources of their own (a book of short
    poems, one poem per page): the same ``<pages>`` tag the wiki's own subpages use,
    covering the whole range. ``work_title`` is the page the scans belong to, so
    relative links and templates render as they do there.
    """
    if start > end:
        raise ValueError(f"start ({start}) must not be greater than end ({end})")
    tag = f'<pages index="{index}" from={start} to={end} />'
    return (
        f"https://{host}/w/api.php?{_COMMON_PARAMS}&prop=text&{_QUIET_PARAMS}"
        f"&contentmodel=wikitext&title={quote(work_title, safe='')}&text={quote(tag, safe='')}"
    )


# --------------------------------------------------------------- page levels --
# When a wiki's page anchor does not carry the proofreading level (gu, or), the build asks
# the wiki itself: api.php?action=query&prop=proofread answers with each page's level.
PAGE_QUALITY_BATCH = 50  # the API's limit on titles per request
# Indic titles are long once percent-encoded (9 characters per letter), so batches are
# also cut by URL length, far below what the servers accept.
PAGE_QUALITY_MAX_URL_CHARS = 4000


def page_quality_urls(source_url: str, titles: Sequence[str]) -> list[str]:
    """The ``prop=proofread`` query URLs that ask the source's wiki for ``titles``' levels.

    Same scheme, host and api.php path as ``source_url``; titles in the order given,
    batched by count and by URL length.
    """
    parts = urlsplit(source_url)
    base = (
        f"{parts.scheme}://{parts.netloc}{parts.path}"
        "?action=query&format=json&formatversion=2&prop=proofread&titles="
    )
    batches: list[list[str]] = []
    for title in titles:
        if batches:
            candidate = [*batches[-1], title]
            fits = len(base) + len(quote("|".join(candidate), safe="")) <= PAGE_QUALITY_MAX_URL_CHARS
            if len(candidate) <= PAGE_QUALITY_BATCH and fits:
                batches[-1] = candidate
                continue
        batches.append([title])
    return [base + quote("|".join(batch), safe="") for batch in batches]


def parse_page_quality_response(raw: str) -> dict[str, int | None]:
    """``{title: level or None}`` from one ``prop=proofread`` response (formatversion=2).

    Keys are the titles as the wiki spells them *and* as they were asked (the API reports
    its normalisation). A missing or invalid page, or one without a level, maps to None.
    Raises ``ValueError`` for an API error or a response that is not the expected JSON.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    if "error" in data:
        error = data["error"] or {}
        raise ValueError(f"API error {error.get('code')}: {error.get('info', '')}")
    query = data.get("query")
    if not isinstance(query, dict) or not isinstance(query.get("pages"), list):
        raise ValueError("no query.pages list in the response")
    asked_as = {row.get("to"): row.get("from") for row in query.get("normalized") or []}
    levels: dict[str, int | None] = {}
    for page in query["pages"]:
        title = page.get("title")
        if not isinstance(title, str):
            continue
        quality = (page.get("proofread") or {}).get("quality")
        level = quality if isinstance(quality, int) and quality in QUALITY_NAMES else None
        if page.get("missing") or page.get("invalid"):
            level = None
        levels[title] = level
        if asked_as.get(title):
            levels[asked_as[title]] = level
    return levels
