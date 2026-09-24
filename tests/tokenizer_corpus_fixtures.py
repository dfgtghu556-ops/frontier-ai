"""Small hand-written fixtures for the Stage A tokenizer-corpus tests.

These are not research data and are never part of a corpus build: they exist so the
ingestion, split, statistics, leakage and coverage code can be tested offline. Every text
here is written for this repository, is trivially below any size that would matter
statistically, and is deliberately short so that the corpus targets are *not* met — which
is exactly what the insufficient-coverage paths need to exercise.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

FIXTURES = Path(__file__).parent / "fixtures" / "tokenizer_corpus"

# ---------------------------------------------------------------------------
# Raw texts. Hand-written, this repository, no third-party copyright.
# ---------------------------------------------------------------------------
HINDI_TEXT = "\n".join(
    [
        "राम ने कहा कि काम पूरा हो गया है।",
        "सीता ने उत्तर दिया कि अभी समय है।",
        "बाज़ार में आज भीड़ थी और दुकानें खुली थीं।",
        "शिक्षा और स्वास्थ्य दोनों ज़रूरी हैं।",
        "गाँव की सड़क अब पक्की हो गई है।",
        "बच्चे स्कूल जा रहे हैं।",
        "पानी की समस्या हर गर्मी में आती है।",
        "सरकार ने नई योजना शुरू की।",
        "किसान खेत में काम कर रहा है।",
        "यह किताब पढ़ने लायक है।",
    ]
)

BENGALI_TEXT = "\n".join(
    [
        "আজ সকালে বাজারে ভিড় ছিল।",
        "ছেলেটি স্কুলে যাচ্ছে।",
        "গ্রামের রাস্তা এখন পাকা।",
        "শিক্ষা ও স্বাস্থ্য দুই-ই জরুরি।",
        "বইটি পড়ার মতো।",
        "চা ঠান্ডা হয়ে গেছে।",
        "মা রান্না করছেন।",
        "বৃষ্টি নামলে কাজ থেমে যাবে।",
        "নদীর জল বেড়েছে।",
        "সে কলকাতায় থাকে।",
    ]
)

MARATHI_TEXT = "\n".join(
    [
        "आज बाजारात गर्दी होती.",
        "मुलगा शाळेत जातो.",
        "गावाचा रस्ता आता पक्का झाला आहे.",
        "शिक्षण आणि आरोग्य दोन्ही महत्त्वाचे आहे.",
        "हे पुस्तक वाचण्याजोगे आहे.",
        "चहा थंड झाला आहे.",
        "आई स्वयंपाक करते आहे.",
        "पाऊस आला की काम थांबते.",
        "नदीचे पाणी वाढले आहे.",
        "ती पुण्यात राहते.",
    ]
)

TAMIL_TEXT = "\n".join(
    [
        "இன்று காலை சந்தையில் கூட்டம் அதிகம்.",
        "பையன் பள்ளிக்குச் செல்கிறான்.",
        "கிராமத்துச் சாலை இப்போது போடப்பட்டது.",
        "கல்வி மற்றும் சுகாதாரம் இரண்டும் முக்கியம்.",
        "இந்தப் புத்தகம் படிக்கத் தகுந்தது.",
        "டீ குளிர்ந்து விட்டது.",
        "அம்மா சமையல் செய்கிறார்.",
        "மழை வந்தால் வேலை நின்றுவிடும்.",
        "ஆற்று நீர் உயர்ந்துள்ளது.",
        "அவள் சென்னையில் வசிக்கிறாள்.",
    ]
)

URDU_TEXT = "\n".join(
    [
        "آج صبح بازار میں بھیڑ تھی۔",
        "لڑکا اسکول جا رہا ہے۔",
        "گاؤں کی سڑک اب پکی ہے۔",
        "تعلیم اور صحت دونوں ضروری ہیں۔",
        "یہ کتاب پڑھنے کے قابل ہے۔",
        "چائے ٹھنڈی ہو گئی ہے۔",
        "والدہ کھانا بنا رہی ہیں۔",
        "بارش آئے گی تو کام رک جائے گا۔",
        "دریا کا پانی بڑھ گیا ہے۔",
        "وہ لاہور میں رہتا ہے۔",
    ]
)

ENGLISH_TEXT = "\n".join(
    [
        "The market was crowded this morning.",
        "The boy is going to school.",
        "The village road is paved now.",
        "Education and health both matter.",
        "This book is worth reading.",
        "The tea has gone cold.",
        "Mother is cooking dinner.",
        "Work stops if the rain arrives.",
        "The river water has risen.",
        "She lives in the city.",
    ]
)

# A long single paragraph (no newline) to exercise sentence splitting / hard wrapping.
LONG_PARAGRAPH = " ".join(f"यह वाक्य संख्या {i} है, और यह परीक्षण के लिए है।" for i in range(1, 201))


def _gutenberg_text(body: str, title: str) -> str:
    """Gutenberg-shaped: the real header/footer markers the shared cleaner cuts on."""
    return (
        "The Project Gutenberg eBook of "
        + title
        + "\n\nThis ebook is for the use of anyone anywhere in the United States and\n"
        "most other parts of the world at no cost and with almost no restrictions\n"
        "whatsoever.\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK "
        + title.upper()
        + " ***\n"
        + body
        + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK "
        + title.upper()
        + " ***\n"
    )


def _wikisource_text(body: str) -> str:
    return (
        "This work is licensed under the Creative Commons "
        "Attribution-ShareAlike 4.0 License.\n\n" + body + "\n"
    )


def fake_gutenberg_text(title: str = "A Test Book") -> str:
    """A synthetic Gutenberg-shaped text with the PD-US marker in it."""
    return _gutenberg_text(HINDI_TEXT + "\n" + ENGLISH_TEXT, title)


def fake_wikisource_text(body: str = MARATHI_TEXT) -> str:
    """A synthetic Wikisource-shaped text with the CC BY-SA 4.0 marker in it."""
    return _wikisource_text(body)


# ---------------------------------------------------------------------------
# Rendered MediaWiki pages (the ``mediawiki-parse`` kind).
#
# The *markup* below copies the structure hi/bn.wikisource.org actually return from
# api.php?action=parse for a ProofreadPage chapter (header template in ws-noexport,
# hidden ws-data, page-number anchors with data-page-quality, TemplateStyles <style>,
# poem blocks, {{gap}} word joiners, a BOM left at a page join). The *text* is
# hand-written for this repository.
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402

HINDI_PROSE = [
    "गाँव में सुबह हुई और लोग खेतों की ओर चल पड़े। हवा ठंडी थी और आसमान साफ़ था।",
    "किसान ने बैलों को पानी पिलाया, फिर हल उठाकर मेड़ की ओर बढ़ा। रास्ते में उसे पड़ोसी मिला।",
    "दोनों ने देर तक फ़सल, बारिश और बाज़ार के भाव की बातें कीं। दोपहर होते-होते धूप तेज़ हो गई।",
    "शाम को घर लौटकर उसने बच्चों को कहानी सुनाई और सबने साथ बैठकर खाना खाया।",
    "रात को गाँव की चौपाल पर बुज़ुर्ग इकट्ठा हुए। किसी ने पुराने मेले की याद छेड़ी, किसी ने नहर की।",
    "अगले दिन बाज़ार लगा। व्यापारी दूर-दूर से आए और अनाज, कपड़े और बर्तनों की दुकानें सज गईं।",
    "लौटते समय उसने सोचा कि इस साल अगर बारिश ठीक रही तो वह बेटी की पढ़ाई के लिए कुछ पैसे बचा लेगा।",
]

BENGALI_VERSE = [
    "ভোরের আলো এসে পড়ে নদীর জলে",
    "মাঝি গান গায় নৌকা বেয়ে চলে",
    "দূরের গ্রামে বাজে মন্দিরের ঘণ্টা",
    "সারা দিন কাটে যেন একটি মুহূর্তের মতো",
]


def _pagenum(name: str, label: str, quality: int | None) -> str:
    quality_attr = "" if quality is None else f' data-page-quality="{quality}"'
    return (
        f'<span><span class="pagenum ws-pagenum" id="{label}" data-page-number="{label}" '
        f'data-page-name="{name}" data-page-index="{label}"{quality_attr} title="x">'
        f'<span id="pageindex&#95;{label}" class="pagenum-inner ws-noexport">&#8203;</span>'
        "</span></span>"
    )


def _pagenum_title_only(name: str, label: str) -> str:
    """The anchor gu/or render (2026-09-24): no data-page-name, no level, only the
    percent-encoded page title (underscores as &#95;) and the printed page number."""
    title = quote(name.replace(" ", "_"), safe=":/").replace("_", "&#95;")
    return (
        f'<span><span class="pagenum ws-pagenum" id="{label}" data-page-number="{label}" '
        f'title="{title}">&#8203;</span></span>'
    )


def _pr_page_anchor(name: str, label: str, quality: int | None) -> str:
    """ml's older anchor (2026-09-24): a bracketed link whose class carries the level."""
    level_class = "" if quality is None else f' class="prp-pagequality-{quality}"'
    href = quote(name.replace(" ", "_"), safe=":/")
    return (
        '<span><span style="position:absolute; left:1em; text-indent:0em; font-size:80%;">'
        '<span id="pr&#95;page"><span id="zzz" style="display:none;"></span>'
        f'<span id="{label}"></span>[&#8201;<a href="/wiki/{href}"{level_class} '
        f'title="{name}">{label}</a>&#8201;]</span></span> </span>'
    )


WIKISOURCE_HEADER = (
    '<div id="headerContainer" class="ws-noexport noprint dynlayout-exempt">\n'
    '<div class="header-mainblock headertemplate"><div class="gen&#95;header&#95;backlink searchaux">'
    '<div>←</div><div id="headerprevious"><a href="/wiki/W/1" title="W/1">पीछे</a></div></div>'
    '<div class="gen&#95;header&#95;central&#95;cell"><span id="header&#95;title&#95;text">परीक्षा-ग्रंथ</span>'
    ' <span id="header&#95;year&#95;text">&#160;(1936)&#160;</span> <br /><i>द्वारा</i> '
    '<span class="vcard"><a href="/wiki/A" title="A">लेखक-नाम</a></span></div>'
    '<div class="gen&#95;header&#95;forelink searchaux"><div id="headernext"><a href="/wiki/W/3">आगे</a></div>'
    "<div>→</div></div></div>\n"
    '<div class="ws-noexport" id="ws-data" style="speak:none;display:none"><span id="ws-article-id">'
    '38658</span><span id="ws-title">परीक्षा-ग्रंथ</span><span id="ws-year">1936</span></div>\n</div>\n'
)


def rendered_chapter_html(
    paragraphs: list[str] | None = None,
    *,
    qualities: tuple[int | None, ...] = (3, 4),
    header: bool = True,
    chapter_label: str = "२",
    anchor: str = "data",
) -> str:
    """A ProofreadPage chapter as MediaWiki renders it (prose, one anchor per scan page).

    ``anchor`` picks the wiki's page-anchor template: "data" (hi, bn, as: name and level
    attributes), "title" (gu, or: only the page title, no level) or "pr_page" (ml).
    """
    paragraphs = HINDI_PROSE if paragraphs is None else paragraphs
    names = [f"पृष्ठ:परीक्षा.djvu/{index + 18}" for index in range(len(qualities))]
    if anchor == "data":
        pages = [_pagenum(n, str(i + 16), q) for i, (n, q) in enumerate(zip(names, qualities))]
    elif anchor == "title":
        pages = [_pagenum_title_only(n, str(i + 16)) for i, n in enumerate(names)]
    elif anchor == "pr_page":
        pages = [_pr_page_anchor(n, str(i + 16), q) for i, (n, q) in enumerate(zip(names, qualities))]
    else:
        raise ValueError(f"unknown anchor style {anchor!r}")
    body = [f"<p>{pages[0] if pages else ''}\n</p>"]
    body.append(f'<div class="tiInherit" style="text-align:center;">\n<p><b>{chapter_label}</b>\n</p>\n</div>')
    for index, paragraph in enumerate(paragraphs):
        # the second scan page starts inside the second paragraph, as on the wiki: the
        # ProofreadPage join space, the anchor, a BOM left by the OCR import and the
        # transcribers' page-leading <br> (the printed sentence continues)
        if index == 1 and len(pages) > 1:
            half = len(paragraph) // 2
            paragraph = paragraph[:half] + "&#32;" + pages[1] + "\ufeff<br />" + paragraph[half:]
        gap = '<span class="&#95;&#95;gap" style="display:inline-block; width:2em;">&#8288;</span>'
        body.append(f"<p>{gap if index else ''}{paragraph}\n</p>")
    for page in pages[2:]:
        body.append(f"<p>{page}{HINDI_PROSE[0]}</p>")
    return (
        '<div class="mw-content-ltr mw-parser-output" lang="hi" dir="ltr">'
        '<div class="prp-pages-output" lang="hi">\n'
        + (WIKISOURCE_HEADER if header else "")
        + '<div style="max-width:36em; margin: 0px auto;">\n'
        + "".join(body)
        + '<style data-mw-deduplicate="TemplateStyles:r1950689">.mw-parser-output .wst-center'
        "{text-align:center;margin:0 auto}</style>"
        + '<sup id="cite_ref-1" class="reference"><a href="#cite_note-1">[१]</a></sup>'
        + '<div class="mw-references-wrap"><ol class="references"><li id="cite_note-1">'
        "एक टिप्पणी जो कॉर्पस में नहीं जानी चाहिए।</li></ol></div>"
        + "</div></div></div>"
    )


def rendered_poems_html(stanzas: list[list[str]] | None = None, *, quality: int = 4) -> str:
    """A range render of verse (<pages index=… from=N to=M />): poem blocks, indents, <br>."""
    stanzas = [BENGALI_VERSE[:2], BENGALI_VERSE[2:]] if stanzas is None else stanzas
    blocks = []
    for number, stanza in enumerate(stanzas, start=1):
        lines = "<br />\n".join(
            line if i % 2 == 0 else
            f'<span class="mw-poem-indented" style="display: inline-block; margin-inline-start: 4em;">{line}</span>'
            for i, line in enumerate(stanza)
        )
        blocks.append(
            _pagenum(f"পাতা:পরীক্ষা.djvu/{12 + number}", str(number), quality)
            + '<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1950689" />'
            + f'<div class="wst-center tiInherit wst-center-nomargin">\n<p><span style="font-size:120%;">{number}</span>\n</p>\n</div>\n'
            + f'<div class="wst-block-center" style="width:fit-content;">\n<div class="poem">\n<p>{lines}\n</p>\n</div>\n</div>'
        )
    return (
        '<div class="mw-content-ltr mw-parser-output" lang="bn" dir="ltr">'
        '<div class="prp-pages-output" lang="bn">\n' + "&#32;".join(blocks) + "</div></div>"
    )


def contents_page_html(entries: int = 12) -> str:
    """A rendered table of contents: every line is a link plus a page number."""
    rows = "".join(
        f'<tr><td><a href="/wiki/W/{i}" title="W/{i}">अध्याय {i} का शीर्षक</a></td><td>{i * 7}</td></tr>'
        for i in range(1, entries + 1)
    )
    return (
        '<div class="mw-content-ltr mw-parser-output" lang="hi" dir="ltr">'
        f"<p><b>सूचीपत्र</b></p><table>{rows}</table></div>"
    )


def redirect_html(target: str = "गो-दान") -> str:
    return (
        '<div class="mw-content-ltr mw-parser-output" lang="hi" dir="ltr">'
        '<div class="redirectMsg"><p>अनुप्रेषित करें:</p><ul class="redirectText"><li>'
        f'<a href="/wiki/{target}" title="{target}">{target}</a></li></ul></div></div>'
    )


def parse_payload(
    html: str,
    *,
    title: str = "परीक्षा-ग्रंथ/२",
    revid: int | None = 480196,
    categories: tuple[str, ...] = ("परीक्षा",),
    formatversion: int = 2,
) -> str:
    """A complete api.php?action=parse&format=json response around ``html``."""
    parse: dict = {"title": title, "pageid": 38658}
    if revid is not None:
        parse["revid"] = revid
    if formatversion == 2:
        parse["text"] = html
        parse["categories"] = [{"sortkey": "", "category": name} for name in categories]
    else:
        parse["text"] = {"*": html}
        parse["categories"] = [{"sortkey": "", "*": name} for name in categories]
    return _json.dumps({"parse": parse}, ensure_ascii=False)


def parse_error_payload(code: str = "missingtitle", info: str = "The page you specified doesn't exist.") -> str:
    return _json.dumps({"error": {"code": code, "info": info, "docref": "See api.php"}})
