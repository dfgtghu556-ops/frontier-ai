"""EXP-035 calibration (removes nothing): short-suite containment, near-duplicates, repeated
lines, language markers, masking, determinism and the CLI end to end."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from frontier_ai.corpus.decontaminate import ShortSuiteIndex
from frontier_ai.corpus.slice_calibrate import (
    LineStats,
    MinHasher,
    Reservoir,
    calibrate_rows,
    marker_counts,
    marker_label,
    mask_pii,
    near_duplicate_clusters,
)
from frontier_ai.evaluation.suite import build_suite, write_suite

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ short-suite --
def test_short_suite_index_finds_contained_passages_word_for_word():
    idx = ShortSuiteIndex.from_texts(
        [
            ("s-short", "रात को चाँद निकला और तारे चमके"),  # 7 words
            ("s-tiny", "अध्याय एक"),  # 2 words: not indexed
            ("s-long", " ".join(f"w{i}" for i in range(13))),
        ],  # 13 words: the 13-gram guard's job
        min_words=5,
    )
    assert idx.describe()["indexed_documents"] == 1 and idx.too_short == 1
    inside = "कल शाम बारिश हुई। रात को चाँद निकला और तारे चमके फिर सब सो गए।"
    hits = idx.find(inside)
    assert [(h.suite_doc_id, h.words) for h in hits] == [("s-short", 7)]
    assert idx.find("रात को चाँद निकला और तारे नहीं चमके") == []  # one word differs
    assert idx.find("अध्याय एक की शुरुआत") == []


def test_short_suite_index_validates_min_words():
    with pytest.raises(ValueError):
        ShortSuiteIndex.from_texts([("a", "x y z")], min_words=13)


# ------------------------------------------------------------- near-dups --
def _doc(seed: int, n: int = 120) -> str:
    rng = np.random.default_rng(seed)
    return " ".join(f"शब्द{int(x)}" for x in rng.integers(0, 5000, size=n))


def test_minhash_near_duplicates_cluster_and_distinct_docs_do_not():
    mh = MinHasher()
    base = _doc(1)
    words = base.split()
    near = " ".join(words[:-1] + ["बदला"])  # one word of 120 changed
    docs = [base, near, _doc(2), _doc(3), base]
    sigs = np.stack([mh.signature(d) for d in docs])
    assert (sigs[0] == sigs[4]).all()
    roots, pairs = near_duplicate_clusters(sigs, np.ones(len(docs), dtype=bool))
    assert roots[0] == roots[1] == roots[4]
    assert len({roots[0], roots[2], roots[3]}) == 3
    assert mh.signature("too short text") is None
    # deterministic across instances
    assert (MinHasher().signature(base) == sigs[0]).all()


def test_line_stats_counts_repeated_lines_exactly():
    ls = LineStats()
    ls.SAMPLE_EVERY = 1
    for i in range(30):
        ls.add(f"Welcome!\nForgot your password?\nunique line {i}")
    r = ls.finish()
    assert r["lines"] == 90 and r["distinct_lines"] == 32
    assert r["lines_repeated_ge_10"] == 60
    assert {x["text"]: x["count"] for x in r["top_repeated_lines"]} == {
        "Welcome!": 30,
        "Forgot your password?": 30,
    }


# --------------------------------------------------------------- markers --
def test_language_markers_separate_shared_scripts():
    uyghur = "بىرىنچىدىن ، ئۆتمۈشتىكى بارلىق كىشىلەرگە چىن كۆڭلىمىزدىن رەھمەت ئېيتىمىز ۋاقىت ئۇچىدۇ"
    urdu = "وزیر اعظم نے کہا کہ ٹیم کی کارکردگی بہت اچھی رہی اور بچوں نے بڑی محنت کی ہے"
    assert marker_label("ur", marker_counts(uyghur, "arabic_script"), 80) == "uyghur_like"
    assert marker_label("ur", marker_counts(urdu, "arabic_script"), 80) == "urdu_like"
    assamese = "অসমৰ ৰাজধানী দিছপুৰ। ৰাজ্যখনৰ সৰ্বাংগীন উন্নয়নৰ বাবে চৰকাৰে"
    bengali = "বাংলাদেশের রাজধানী ঢাকা। রাজ্যের সরকার উন্নয়নের জন্য কাজ করছে"
    assert marker_label("as", marker_counts(assamese, "bengali_script"), 60) == "assamese_like"
    assert marker_label("as", marker_counts(bengali, "bengali_script"), 60) == "bengali_like"


def test_mask_pii_keeps_short_numbers():
    text = "Call 98765 43210 or mail a.b@example.com, see https://x.in/p — score 473 in 2021."
    masked = mask_pii(text)
    assert "98765" not in masked and "[10 digits]" in masked
    assert "[email]" in masked and "[url]" in masked
    assert "473" in masked and "2021" in masked


def test_reservoir_is_deterministic_and_uniform_size():
    a, b = Reservoir(3, "seed"), Reservoir(3, "seed")
    for i in range(100):
        a.offer(lambda i=i: {"i": i})
        b.offer(lambda i=i: {"i": i})
    assert a.items == b.items and len(a.items) == 3 and a.seen == 100


# ------------------------------------------------------------ calibration --
HINDI = "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं। " * 3
SUITE_SHORT = "रात को चाँद निकला और तारे चमके"
ROWS = [
    ("d1", "web", HINDI + "\nWelcome!"),
    ("d2", "web", HINDI + "\nWelcome!"),  # exact = near duplicate
    ("d3", "pdf", "Lorem ipsum dolor sit amet, this is Latin script text entirely."),
    ("d4", "web", "पहला वाक्य। " + SUITE_SHORT + " और फिर फ़ोन 98765 43210"),
    ("d5", "pdf", ("लंबा पाठ " * 3000) + " end"),  # over 20k chars
    ("d6", "web", "\u0958" + " नुक्ता वाला शब्द और कुछ और शब्द यहाँ"),  # not NFC: precomposed QA
    ("d7", "web", "   "),
]


def _calibrate(rows=ROWS, **kw):
    idx = ShortSuiteIndex.from_texts([("s-1", SUITE_SHORT)], min_words=3)
    return calibrate_rows(
        rows,
        exp_id="EXP-035",
        source_id="src-hi",
        language="hi",
        script="devanagari",
        expected_rows=len(rows),
        short_index=idx,
        **kw,
    )


def test_calibrate_rows_measures_and_removes_nothing():
    stats, samples = _calibrate()
    assert stats["documents"] == len(ROWS) and stats["documents_empty_after_normalize"] == 1
    assert stats["near_duplicates"]["documents_removable_keep_one_per_cluster"] == 1
    assert stats["short_suite_containment"]["documents_hit"] == 1
    assert stats["short_suite_containment"]["hits_by_suite_doc_words"] == {"7-9": 1}
    assert stats["documents_withheld_from_samples"] == 1
    assert all(s.get("doc_id") != "d4" for s in samples)
    assert stats["long_documents"]["documents_by_type"] == {"pdf": 1}
    assert stats["normalization"]["not_nfc_documents"] == 1
    assert any("U+0958" in c["codepoint"] for c in stats["normalization"]["nfc_removed_codepoints_top"])
    assert stats["repeated_lines"]["lines_repeated_ge_2"] >= 2
    assert stats["long_document_windows"]["documents_longer_than_cap"] == 1
    kinds = {s["stratum"] for s in samples}
    assert {"random_pass", "long_over_20k", "pdf_pass"} <= kinds
    assert all("98765" not in json.dumps(s, ensure_ascii=False) for s in samples)
    long_sample = next(s for s in samples if s["stratum"] == "long_over_20k")
    assert "middle" in long_sample
    # reproducible (timing fields aside)
    s2, samples2 = _calibrate()
    s2.pop("seconds"), s2.pop("documents_per_second")
    stats.pop("seconds"), stats.pop("documents_per_second")
    assert s2 == stats and samples2 == samples


def test_calibrate_rows_refuses_more_rows_than_declared():
    idx = ShortSuiteIndex.from_texts([("s-1", SUITE_SHORT)], min_words=3)
    with pytest.raises(ValueError, match="more rows"):
        calibrate_rows(
            ROWS,
            exp_id="E",
            source_id="s",
            language="hi",
            script="devanagari",
            expected_rows=2,
            short_index=idx,
        )


def test_calibrate_cli_end_to_end(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rev = "0123456789abcdef0123456789abcdef01234567"
    root = tmp_path / "data"
    path = root / rev[:12] / "verified/hin/data-0.parquet"
    path.parent.mkdir(parents=True)
    long_suite = " ".join(f"शब्द{i}" for i in range(20))
    rows = [*ROWS, ("d8", "web", "शुरू में " + long_suite + " और अंत")]  # shares 13-grams with s-1
    pq.write_table(
        pa.table(
            {"doc_id": [r[0] for r in rows], "type": [r[1] for r in rows], "text": [r[2] for r in rows]}
        ),
        str(path),
        row_group_size=3,
    )
    data = path.read_bytes()
    pins = {
        "schema": "frontier-v2-source-pins-v1",
        "slice_id": "test-slice",
        "dataset": "t/sangraha",
        "revision": rev,
        "license_id": "CC-BY-4.0",
        "attribution": "test attribution",
        "total_bytes": len(data),
        "files": [
            {
                "source_id": "sangraha-verified-hin-data-0",
                "language": "hi",
                "sangraha_code": "hin",
                "script": "devanagari",
                "path": "verified/hin/data-0.parquet",
                "url": f"https://h/{rev}/verified/hin/data-0.parquet",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")

    class _D:
        def __init__(self, doc_id, text):
            self.doc_id, self.language, self.source_id, self.text = doc_id, "hi", "suite-src", text

    suite_texts = [("s-0", SUITE_SHORT), ("s-1", long_suite)]
    suite_path = tmp_path / "SUITE.json"
    write_suite(suite_path, build_suite([_D(*t) for t in suite_texts], "test-suite", {"corpus": "t"}))
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    (heldout / "heldout-000000.txt").write_text("\n".join(t for _, t in suite_texts), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "calib_cli", REPO / "scripts" / "calibrate_sangraha_slice.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    assert (
        mod.main(
            [
                "--pins",
                str(pins_path),
                "--root",
                str(root),
                "--suite",
                str(suite_path),
                "--heldout-dir",
                str(heldout),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True and summary["short_suite_status"].startswith("CHECKED")
    f = summary["files"][0]
    assert f["pinned_sha256_verified"] and f["short_suite_containment"]["documents_hit"] == 1
    lines = (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == summary["samples_written"] > 0
    assert summary["samples_status"].startswith("screened")
    assert f["samples_dropped_by_13gram_screen"] >= 1  # d8 was kept by a reservoir, then screened out
    assert f["documents_withheld_from_samples"] == 1
    assert all(json.loads(x).get("doc_id") not in ("d4", "d8") for x in lines)  # suite-touching docs
    assert "शब्द7 शब्द8 शब्द9" not in "".join(lines)
    assert SUITE_SHORT not in "".join(lines)
    assert "calibration of test-slice" in (out / "SUMMARY.txt").read_text(encoding="utf-8")

    # without the shard: NOT CHECKED and not complete
    assert (
        mod.main(
            [
                "--pins",
                str(pins_path),
                "--root",
                str(root),
                "--suite",
                str(suite_path),
                "--heldout-dir",
                str(tmp_path / "none"),
                "--out",
                str(tmp_path / "o2"),
            ]
        )
        == 0
    )
    s2 = json.loads((tmp_path / "o2" / "summary.json").read_text(encoding="utf-8"))
    assert s2["complete"] is False and s2["short_suite_status"].startswith("NOT CHECKED")
    assert s2["samples_status"].startswith("WITHHELD") and s2["samples_written"] == 0
    assert (tmp_path / "o2" / "samples.jsonl").read_text(encoding="utf-8") == ""


def test_fast_control_ratio_equals_the_per_character_definition():
    import random

    from frontier_ai.corpus.quality import _control_ratio, _control_ratio_reference

    rng = random.Random(7)
    alphabet = [chr(c) for c in range(0, 0x250)] + ["\u0915", "\u200d", "\u2028", "\ufeff"]
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        assert _control_ratio(text) == _control_ratio_reference(text)
