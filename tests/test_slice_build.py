"""EXP-036 build: every rule with its reason, provenance, determinism, exact token counts, a suite
check on the exact output, resume, and the CLI end to end (1 and 2 worker processes)."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.corpus.decontaminate import ShortSuiteIndex, SuiteGuard
from frontier_ai.corpus.slice_build import (
    REASONS,
    BuildConfig,
    TokenCounter,
    build_file,
    clean_lines,
    fused_latin_re,
    is_foreign_line,
    is_foreign_line_reference,
    iter_output,
    mask_contacts,
    render_text_report,
)
from frontier_ai.evaluation.suite import build_suite, write_suite
from frontier_ai.tokenization.frozen import load_frontier_tokenizer

REPO = Path(__file__).resolve().parents[1]
REV = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def frozen():
    return load_frontier_tokenizer()


# ------------------------------------------------------------------ foreign lines --
def test_foreign_line_fast_path_equals_the_definition():
    alphabet = list("abcXYZ éñ अकि़ं।०१ কৰ ਕਾ ગુ ଓ த తె ಕ മ اےٹ 中文 Жж 12 .,!?-– ½²") + ["\u200d", "\u093c"]
    rng = random.Random(36)
    lines = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12))) for _ in range(3000)]
    lines += ["।।", "१२३", "½", "123 - 456", "中文", "Hello world", "नमस्ते", "abc नमस्ते"]
    for script in ("latin", "devanagari", "bengali", "arabic", "tamil"):
        for line in lines:
            assert is_foreign_line(line, script) == is_foreign_line_reference(line, script), (script, line)
    assert is_foreign_line("Forgot your password?", "devanagari")
    assert not is_foreign_line("Post Office recruitment की लिस्ट", "devanagari")
    assert not is_foreign_line("12:30 - ।", "devanagari")  # no letters at all: kept
    assert is_foreign_line("中文网站", "tamil")  # letters of an unknown block count
    assert is_foreign_line("यह हिंदी है", "latin")


def test_clean_lines_removes_boilerplate_and_foreign_lines_and_keeps_paragraphs():
    from frontier_ai.corpus.slice_calibrate import _line_hash

    text = "पहला पैरा।\nFollow us on Google News\n\n\n\nदूसरा पैरा।\n  - First Published :  \n१२३"
    res = clean_lines(text, "devanagari", {_line_hash("- First Published :")})
    assert res.text == "पहला पैरा।\n\nदूसरा पैरा।\n१२३"
    assert (res.boilerplate_lines, res.foreign_lines) == (1, 1)
    assert res.foreign_chars == len("Follow us on Google News")
    assert res.boilerplate_chars == len("- First Published :")
    assert clean_lines("Only English here\nand here", "devanagari", set()).text == ""


# ------------------------------------------------------------------------ contacts --
@pytest.mark.parametrize(
    ("text", "expected", "phones", "emails"),
    [
        ("mail a.b+c@example.co.in now", "mail [email] now", 0, 1),
        ("call +91 98765 43210.", "call [phone].", 1, 0),
        ("call +91-9876543210", "call [phone]", 1, 0),
        ("फ़ोन 98765 43210 पर", "फ़ोन [phone] पर", 1, 0),
        ("फ़ोन 09876543210", "फ़ोन [phone]", 1, 0),
        ("ফোন ০১৭১২৩৪৫৬৭৮", "ফোন [phone]", 1, 0),
        ("years 1991 1992 1993 and 2019-2020", "years 1991 1992 1993 and 2019-2020", 0, 0),
        ("pin 110001, score 3-2, ₹ 1,20,00,000", "pin 110001, score 3-2, ₹ 1,20,00,000", 0, 0),
        ("pi 3.14159265358979", "pi 3.14159265358979", 0, 0),
    ],
)
def test_mask_contacts(text, expected, phones, emails):
    assert mask_contacts(text) == (expected, emails, phones)


# ------------------------------------------------------------------ token counting --
def test_token_counter_equals_encode_with_the_frozen_tokenizer(frozen):
    counter = TokenCounter(frozen, cache_limit=20)
    texts = [
        "नमस्ते दुनिया! यह एक परीक्षण है। नमस्ते फिर से।",
        "Hello, world! 123 — mixed हिंदी and English text.\n\nনতুন লাইন",
        "",
        "ٹ ڈ ڑ ں ے ھ اردو متن",
        "a" * 500 + " " + "ज़" * 50,
    ] * 3
    for t in texts:
        assert counter.count(t) == len(frozen.encode(t))
    assert len(counter.cache) == 20  # bounded (35 distinct pre-tokens)


def test_token_counter_handles_special_tokens(tmp_path):
    from frontier_ai.tokenization.bpe_python import PythonBPE

    corpus = tmp_path / "c.txt"
    corpus.write_text("नमस्ते दुनिया hello world " * 50, encoding="utf-8")
    tok = PythonBPE()
    tok.train(corpus, vocab_size=300, special_tokens=["<|endoftext|>"])
    text = "नमस्ते<|endoftext|>hello <|endoftext|> दुनिया"
    assert TokenCounter(tok).count(text) == len(tok.encode(text))


def test_fused_latin_marks_detects_mt_artefacts():
    rx = fused_latin_re("gurmukhi")
    assert rx.findall("architectਾਂਚਾਗਤ ਢਾਂਚਾ and ok") == ["tਾ"]
    assert fused_latin_re("latin") is None and fused_latin_re("arabic") is None


# ------------------------------------------------------------------------ build_file --
SUITE_SHORT = "रात को चाँद निकला और तारे चमके"
LONG_SUITE = " ".join(f"शब्द{i}" for i in range(20))
BOILER = "यह पंक्ति हर पेज पर है"


def _words(seed: int, n: int = 60) -> str:
    rng = random.Random(seed)
    vocab = [
        "नदी",
        "पहाड़",
        "गाँव",
        "शहर",
        "किताब",
        "पानी",
        "बादल",
        "सूरज",
        "हवा",
        "मिट्टी",
        "लोग",
        "खेत",
        "रास्ता",
        "घर",
        "दरवाज़ा",
        "खिड़की",
        "आसमान",
        "पेड़",
        "फूल",
        "पत्ता",
        "सड़क",
        "बाज़ार",
    ]
    return " ".join(rng.choice(vocab) + str(rng.randint(0, 99)) for _ in range(n))


DOC_B = _words(2)
ROWS_HI = [
    ("a", "web", _words(1) + "\n" + BOILER),  # 0 kept (boilerplate line removed)
    ("b", "web", "   "),  # 1 empty
    ("c", "web", "शुरू में " + LONG_SUITE + " अंत"),  # 2 suite_ngram
    ("d", "web", "पहले " + SUITE_SHORT + " बाद में कुछ और शब्द"),  # 3 suite_short
    ("a", "web", _words(1) + "\n" + BOILER),  # 4 exact_duplicate (same doc_id as row 0!)
    ("e", "web", "नमस्ते\x01\x02\x03\x04"),  # 5 control_chars
    ("f", "web", "Welcome!\nForgot your password?"),  # 6 empty_after_cleaning
    ("g", "web", "१२३ ४५६ -- !!"),  # 7 no_letters
    ("h", "web", "यह test page is mostly English words here है"),  # 8 script_share
    ("i", "web", "{{साँचा}} हिंदी पाठ यहाँ है {{अंत}}"),  # 9 wiki_markup
    ("j", "web", "नमस्ते दुनिया " * 20),  # 10 repetition
    ("k", "pdf", DOC_B + "\n" + BOILER),  # 11 kept, cluster root
    ("l", "pdf", DOC_B + " अंतिम\n" + BOILER),  # 12 near_duplicate of 11
    (
        "m",
        "web",
        _words(3, 30)
        + "\nFollow us on Google News\n"
        + _words(4, 30)
        + " ईमेल x.y@mail.com फ़ोन 98765 43210\n"
        + BOILER,
    ),  # 13 kept: lines removed, contacts masked
    (
        "n",
        "web",
        " ".join(LONG_SUITE.split()[:7]) + "\nRead more\n" + " ".join(LONG_SUITE.split()[7:13]),
    ),  # 14 suite_after_cleaning: removing "Read more" joins a 13-gram
    ("o", "web", "ळ ळ ळ " + _words(5, 20)),  # 15 kept, measured (lla >= 3)
]
REASON_OF_ROW = {
    1: "empty",
    2: "suite_ngram",
    3: "suite_short",
    4: "exact_duplicate",
    5: "control_chars",
    6: "empty_after_cleaning",
    7: "no_letters",
    8: "script_share",
    9: "wiki_markup",
    10: "repetition",
    12: "near_duplicate",
    14: "suite_after_cleaning",
}
PAIRS = [("s-0", SUITE_SHORT), ("s-1", LONG_SUITE)]
CFG = BuildConfig(boilerplate_min_repeats=3)


def _build(tmp_path: Path, rows, frozen, language="hi", script="devanagari", name="x", cfg=CFG):
    out = tmp_path / name / f"{language}.jsonl.gz"
    stats, samples = build_file(
        lambda: iter(rows),
        out_path=out,
        exp_id="EXP-036",
        source_id=f"sangraha-verified-{language}-data-0",
        language=language,
        script=script,
        revision=REV,
        guard=SuiteGuard.from_texts(PAIRS),
        short_index=ShortSuiteIndex.from_texts(PAIRS, min_words=3),
        token_counter=TokenCounter(frozen),
        config=cfg,
        expected_rows=len(rows),
    )
    return out, stats, samples


def test_build_file_applies_every_rule_with_its_reason(tmp_path, frozen):
    out, stats, samples = _build(tmp_path, ROWS_HI, frozen)
    removed = {r: stats["removed_docs"][r]["docs"] for r in REASONS}
    expected = {r: 0 for r in REASONS}
    for reason in REASON_OF_ROW.values():
        expected[reason] += 1
    assert removed == expected
    docs = list(iter_output(out))
    assert [d["row"] for d in docs] == [0, 11, 13, 15]
    assert stats["kept"]["docs"] == 4 and stats["input"]["docs"] == len(ROWS_HI)
    # provenance: unique id even though Sangraha doc_id "a" repeats
    assert docs[0]["id"] == "sangraha-verified-hi-data-0#0" and docs[0]["doc_id"] == "a"
    assert {d["revision"] for d in docs} == {REV} and docs[1]["type"] == "pdf"
    assert all(
        set(d) == {"id", "text", "lang", "source", "row", "doc_id", "type", "revision", "tokens"}
        for d in docs
    )
    # rules 3, 4, 10 on the kept text
    assert all(BOILER not in d["text"] for d in docs)
    m = docs[2]["text"]
    assert "Follow us" not in m and "[email]" in m and "[phone]" in m and "98765" not in m
    assert stats["removed_lines"]["boilerplate"]["lines"] == 4  # rows 0, 11, 13 (+12, removed later)
    assert stats["removed_lines"]["foreign"]["lines"] >= 3
    assert stats["masked"] == {"emails": 1, "phones": 1, "docs_masked": 1}
    # exact tokens, and the totals add up
    assert all(d["tokens"] == len(frozen.encode(d["text"])) for d in docs)
    assert stats["kept"]["tokens"] == sum(d["tokens"] for d in docs)
    assert stats["kept"]["chars"] == sum(len(d["text"]) for d in docs)
    assert stats["measured_only"]["docs"]["kept_lla_3plus"] == 1
    # independent suite re-check of the exact output
    guard, short = SuiteGuard.from_texts(PAIRS), ShortSuiteIndex.from_texts(PAIRS, min_words=3)
    assert all(guard.check(d["text"]) is None and not short.find(d["text"]) for d in docs)
    assert stats["suite"]["touching_docs"] == 3
    # samples: masked, never from suite-touching documents
    blob = json.dumps(samples, ensure_ascii=False)
    assert SUITE_SHORT not in blob and "शब्द7 शब्द8" not in blob and "98765" not in blob
    assert {s["row"] for s in samples if s["stratum"] != "removed_line:foreign"}.isdisjoint({1, 2, 3, 14})
    # a removed line is a substring of a text that passed the suite check, so it is safe to show
    assert [s["excerpt"] for s in samples if s["row"] == 14] == ["Read more"]
    strata = {s["stratum"] for s in samples}
    assert {"removed:near_duplicate", "removed:wiki_markup", "kept", "removed_line:foreign"} <= strata
    near = [s for s in samples if s["stratum"] == "removed:near_duplicate"]
    assert near[0]["row"] == 12 and near[0]["kept_row"] == 11
    assert stats["top_boilerplate_lines"][0] == {"count": 5, "text": BOILER}  # rows 0, 4, 11, 12, 13
    # every removal is below its review bound except the ones this fixture forces
    assert {r["rule"] for r in stats["review"]} >= {"suite_after_cleaning"}
    run = {
        "exp_id": "EXP-036",
        "slice_id": "t",
        "dataset": "d",
        "revision": REV,
        "complete": False,
        "suite_status": "CHECKED",
        "tokenizer": {"name": "t"},
        "config_fingerprint": "f",
        "files": [stats],
    }
    report = render_text_report(run)
    assert "near_duplicate [2 duplicates]: hi 1/" in report and "Output suite hits: 0" in report


def test_build_file_is_deterministic_byte_for_byte(tmp_path, frozen):
    a, sa, _ = _build(tmp_path, ROWS_HI, frozen, name="one")
    b, sb, _ = _build(tmp_path, ROWS_HI, frozen, name="two")
    assert a.read_bytes() == b.read_bytes()
    assert sa["output"] == sb["output"]
    assert sa["output"]["sha256_gz"] == hashlib.sha256(a.read_bytes()).hexdigest()
    raw = gzip.decompress(a.read_bytes())
    assert sa["output"]["sha256_jsonl"] == hashlib.sha256(raw).hexdigest()
    assert not list(a.parent.glob("*.tmp"))  # temporary files are gone


def test_urdu_check_drops_persian_and_uyghur_documents(tmp_path, frozen):
    urdu = "یہ ایک اردو جملہ ہے جس میں ٹ ڈ ڑ ں ے ھ جیسے حروف آتے ہیں اور یہ بہت اچھا ہے۔ " * 2
    persian = "این یک متن فارسی است که در آن هیچ حرف مخصوص زبان دیگر نیست و خبرگزاری تسنیم گزارش داد " * 2
    uyghur = "بۇ ئۇيغۇرچە تېكىست ۋە ئۇنىڭدا ئۆزگىچە ھەرپلەر بار ئەمەسمۇ " * 3
    rows = [("u1", "web", urdu), ("p1", "web", persian), ("y1", "web", uyghur), ("s1", "web", "مختصر")]
    out, stats, _ = _build(tmp_path, rows, frozen, language="ur", script="arabic")
    assert stats["removed_docs"]["urdu_persian_like"]["docs"] == 1
    assert stats["removed_docs"]["urdu_uyghur_like"]["docs"] == 1
    assert [d["row"] for d in iter_output(out)] == [0, 3]  # short text: too few letters to judge


def test_build_file_refuses_a_row_count_mismatch_and_a_wrong_short_index(tmp_path, frozen):
    with pytest.raises(ValueError, match="parquet metadata"):
        build_file(
            lambda: iter(ROWS_HI[:3]),
            out_path=tmp_path / "hi.jsonl.gz",
            exp_id="E",
            source_id="s",
            language="hi",
            script="devanagari",
            revision=REV,
            guard=SuiteGuard.from_texts(PAIRS),
            short_index=ShortSuiteIndex.from_texts(PAIRS, min_words=3),
            token_counter=TokenCounter(frozen),
            expected_rows=4,
        )
    with pytest.raises(ValueError, match="min_words"):
        build_file(
            lambda: iter(ROWS_HI[:3]),
            out_path=tmp_path / "hi.jsonl.gz",
            exp_id="E",
            source_id="s",
            language="hi",
            script="devanagari",
            revision=REV,
            guard=SuiteGuard.from_texts(PAIRS),
            short_index=ShortSuiteIndex.from_texts(PAIRS, min_words=5),
            token_counter=TokenCounter(frozen),
        )


# ------------------------------------------------------------------------ CLI --
def _slice(tmp_path: Path, files: dict[str, list[tuple[str, str, str]]]):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "data"
    entries = []
    codes = {"hi": ("hin", "devanagari"), "ur": ("urd", "arabic")}
    for lang, rows in files.items():
        code, script = codes[lang]
        path = root / REV[:12] / f"verified/{code}/data-0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {"doc_id": [r[0] for r in rows], "type": [r[1] for r in rows], "text": [r[2] for r in rows]}
            ),
            str(path),
            row_group_size=4,
        )
        data = path.read_bytes()
        entries.append(
            {
                "source_id": f"sangraha-verified-{code}-data-0",
                "language": lang,
                "sangraha_code": code,
                "script": script,
                "path": f"verified/{code}/data-0.parquet",
                "url": f"https://h/{REV}/verified/{code}/data-0.parquet",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    pins = {
        "schema": "frontier-v2-source-pins-v1",
        "slice_id": "test-slice",
        "dataset": "t/sangraha",
        "subset": "verified",
        "revision": REV,
        "license_id": "CC-BY-4.0",
        "attribution": "test attribution",
        "total_bytes": sum(e["size"] for e in entries),
        "files": entries,
    }
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")

    class _D:
        def __init__(self, doc_id, text):
            self.doc_id, self.language, self.source_id, self.text = doc_id, "hi", "suite-src", text

    suite_path = tmp_path / "SUITE.json"
    write_suite(suite_path, build_suite([_D(*t) for t in PAIRS], "test-suite", {"corpus": "t"}))
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    (heldout / "heldout-000000.txt").write_text("\n".join(t for _, t in PAIRS), encoding="utf-8")
    return root, pins_path, suite_path, heldout


def _args(root, pins, suite, heldout, out, data_out, *extra):
    return [
        "--pins",
        str(pins),
        "--root",
        str(root),
        "--suite",
        str(suite),
        "--heldout-dir",
        str(heldout),
        "--out",
        str(out),
        "--data-out",
        str(data_out),
        *extra,
    ]


def _load_cli():
    spec = importlib.util.spec_from_file_location("build_cli", REPO / "scripts" / "build_sangraha_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


URDU_ROWS = [
    ("u1", "web", "یہ ایک اردو جملہ ہے جس میں ٹ ڈ ڑ ں ے ھ جیسے حروف آتے ہیں۔ " * 3),
    ("u2", "web", "دوسرا اردو مضمون یہاں ہے اور اس میں بھی ے اور ں موجود ہیں۔ " * 3),
]


def test_build_cli_end_to_end_resume_and_stop_without_suite(tmp_path, monkeypatch, capsys):
    root, pins, suite, heldout = _slice(tmp_path, {"hi": ROWS_HI, "ur": URDU_ROWS})
    mod = _load_cli()
    monkeypatch.setattr(mod, "FREE_SPACE_MARGIN", 0)
    out, data_out = tmp_path / "out", tmp_path / "corpus"
    assert mod.main(_args(root, pins, suite, heldout, out, data_out, "--workers", "1")) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True and summary["suite_status"].startswith("CHECKED")
    assert [f["language"] for f in summary["files"]] == ["hi", "ur"]
    assert all(f["pinned_sha256_verified"] for f in summary["files"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == json.loads((data_out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True and manifest["source"]["license_id"] == "CC-BY-4.0"
    for f in manifest["files"]:
        path = data_out / f["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256_gz"]
        assert sum(d["tokens"] for d in iter_output(path)) == f["tokens"]
    assert manifest["totals"]["docs"] == 4 + 2
    assert "test attribution" in (data_out / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    lines = (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == summary["samples_written"] > 0 and SUITE_SHORT not in "".join(lines)
    assert "Output suite hits: 0" in (out / "SUMMARY.txt").read_text(encoding="utf-8")
    first = (data_out / "hi.jsonl.gz").read_bytes()

    # run again: both files are reused, nothing is rebuilt, the reports are identical in content
    capsys.readouterr()
    assert mod.main(_args(root, pins, suite, heldout, out, data_out, "--workers", "1")) == 0
    assert capsys.readouterr().out.count("reused") == 2
    again = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert again["reused_files"] == ["sangraha-verified-hin-data-0", "sangraha-verified-urd-data-0"]
    assert again["files"] == summary["files"] and (data_out / "hi.jsonl.gz").read_bytes() == first

    # a changed output file is not reused
    (data_out / "ur.jsonl.gz").write_bytes(b"broken")
    assert mod.main(_args(root, pins, suite, heldout, out, data_out, "--workers", "1")) == 0
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["reused_files"] == [
        "sangraha-verified-hin-data-0"
    ]

    # no held-out shard: STOP before reading any data
    assert mod.main(_args(root, pins, suite, tmp_path / "none", tmp_path / "o2", tmp_path / "c2")) == 2
    assert not (tmp_path / "o2").exists()

    # not enough disk space: STOP
    monkeypatch.setattr(mod, "FREE_SPACE_MARGIN", 10**18)
    assert mod.main(_args(root, pins, suite, heldout, tmp_path / "o3", tmp_path / "c3")) == 2


def test_build_cli_two_workers_matches_one_worker(tmp_path):
    root, pins, suite, heldout = _slice(tmp_path, {"hi": ROWS_HI, "ur": URDU_ROWS})
    outs = {}
    for workers in ("1", "2"):
        data_out = tmp_path / f"corpus{workers}"
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "build_sangraha_v2.py"),
            *_args(root, pins, suite, heldout, tmp_path / f"out{workers}", data_out, "--workers", workers),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        outs[workers] = {p.name: p.read_bytes() for p in sorted(data_out.glob("*.jsonl.gz"))}
    assert outs["1"] == outs["2"] and len(outs["1"]) == 2
