"""Small hand-written fixtures for the Stage A tokenizer-corpus tests.

These are not research data and are never part of a corpus build: they exist so the
ingestion, split, statistics, leakage and coverage code can be tested offline. Every text
here is written for this repository, is trivially below any size that would matter
statistically, and is deliberately short so that the corpus targets are *not* met — which
is exactly what the insufficient-coverage paths need to exercise.
"""

from __future__ import annotations

from pathlib import Path

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
