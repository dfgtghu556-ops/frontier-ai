"""EXP-046 (data slice 2): the pinned file list and the 15-minute Kaggle probe kernel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from frontier_ai.corpus.sangraha import load_slice_pins

ROOT = Path(__file__).resolve().parents[1]
SLICE1 = ROOT / "corpora" / "frontier" / "v2" / "sangraha_slice1.json"
SLICE2 = ROOT / "corpora" / "frontier" / "v2" / "sangraha_slice2.json"
KERNEL = ROOT / "scripts" / "kaggle" / "exp046_probe_kernel.py"


def test_slice2_pins_follow_the_selection_rule():
    h1, f1 = load_slice_pins(SLICE1)
    h2, f2 = load_slice_pins(SLICE2)
    assert h2["slice_id"] == "sangraha-verified-slice2"
    for key in ("schema", "dataset", "subset", "revision", "license_id", "row_schema"):
        assert h2[key] == h1[key], key
    assert len(f2) == 26
    langs1 = {(f.language, f.sangraha_code, f.script) for f in f1}
    expected = {f"verified/{code}/data-{n}.parquet" for (_lang, code, _script) in langs1 for n in (1, 2)}
    assert {f.path for f in f2} == expected
    assert {(f.language, f.sangraha_code, f.script) for f in f2} == langs1
    assert not {f.sha256 for f in f1} & {f.sha256 for f in f2}
    assert not {f.source_id for f in f1} & {f.source_id for f in f2}
    assert len({f.sha256 for f in f2}) == 26
    assert h2["total_bytes"] == 9_787_246_071
    assert all(f.sha256 == f.sha256.lower() and int(f.sha256, 16) >= 0 for f in f2)


def _load_kernel(monkeypatch, tmp_path, pins: Path):
    spec = importlib.util.spec_from_file_location("exp046_probe_kernel", KERNEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SRC", ROOT)
    monkeypatch.setattr(mod, "OUT", tmp_path / "working" / "EXP-046" / "probe")
    monkeypatch.setattr(mod, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(mod, "INPUT", tmp_path / "input")
    monkeypatch.setattr(mod, "PINS", str(pins))
    return mod


def _fake_pins(tmp_path: Path) -> tuple[Path, int]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    h1 = json.loads(SLICE1.read_text(encoding="utf-8"))
    rev = h1["revision"]
    rel = "verified/asm/data-1.parquet"
    served = tmp_path / "hf" / rev / rel
    served.parent.mkdir(parents=True)
    rows = 37
    table = pa.table({"doc_id": [str(i) for i in range(rows)], "type": ["web"] * rows, "text": ["x"] * rows})
    pq.write_table(table, served)
    data = served.read_bytes()
    f = dict(h1["files"][0])
    f.update(
        source_id="sangraha-verified-asm-data-1",
        path=rel,
        url=served.as_uri(),
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    pins = dict(h1, files=[f], total_bytes=len(data))
    p = tmp_path / "pins.json"
    p.write_text(json.dumps(pins), encoding="utf-8")
    return p, rows


def test_probe_kernel_measures_one_download_and_deletes_it(monkeypatch, tmp_path):
    pins, rows = _fake_pins(tmp_path)
    mod = _load_kernel(monkeypatch, tmp_path, pins)
    dl = mod.download_one()
    assert dl["result"] == "downloaded"
    assert dl["sha256_again"] is True
    assert dl["parquet_rows_from_footer"] == rows
    assert dl["deleted_after_measuring"] is True
    assert not any(p.is_file() for p in (tmp_path / "scratch").rglob("*"))

    deps = mod.dependencies()
    assert deps["tokenizer_v2"].startswith("loaded and hash-verified")
    assert not deps["numpy"].startswith("IMPORT FAILED")

    report = {"schema": mod.PROBE_SCHEMA, "machine": mod.machine(), "disks": mod.disks()}
    report.update(dependencies=deps, download=dl, token_dataset=mod.token_dataset(), errors=[], ok=True)
    mod.write(report)
    saved = json.loads(
        (tmp_path / "working" / "EXP-046" / "probe" / "summary.json").read_text(encoding="utf-8")
    )
    assert saved["download"]["parquet_rows_from_footer"] == rows
    text = (tmp_path / "working" / "EXP-046" / "probe" / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "no data processed" in text and "rows in footer: 37" in text
    # the output folder never holds Sangraha files
    assert not list((tmp_path / "working").rglob("*.parquet"))


def test_probe_kernel_finds_the_token_files(monkeypatch, tmp_path):
    pins, _ = _fake_pins(tmp_path)
    mod = _load_kernel(monkeypatch, tmp_path, pins)
    names = [f["path"] for f in json.loads((ROOT / mod.MANIFEST).read_text(encoding="utf-8"))["files"]]
    d = tmp_path / "input" / "frontier-v2-tokens"
    d.mkdir(parents=True)
    for n in names:
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        (d / n).write_bytes(b"\0\0")
    td = mod.token_dataset()
    assert td["folders_with_all_files"] == [str(d)]
    assert td["files"] == len(names) and td["total_bytes"] == 2 * len(names)


def test_probe_kernel_records_a_failed_download_without_crashing(monkeypatch, tmp_path):
    pins, _ = _fake_pins(tmp_path)
    bad = json.loads(pins.read_text(encoding="utf-8"))
    bad["files"][0]["sha256"] = "0" * 64
    pins.write_text(json.dumps(bad), encoding="utf-8")
    mod = _load_kernel(monkeypatch, tmp_path, pins)
    monkeypatch.setattr(mod, "COMMIT", "a" * 40)
    monkeypatch.setattr(mod, "sh", lambda cmd: None)  # no clone / pip install in the test
    assert mod.main() == 0
    saved = json.loads(
        (tmp_path / "working" / "EXP-046" / "probe" / "summary.json").read_text(encoding="utf-8")
    )
    assert saved["ok"] is False and saved["complete"] is True and saved["smoke"] is False
    assert saved["environment"]["code_commit"] == "a" * 40
    assert any("does not match its pin" in e for e in saved["errors"])
    assert not list((tmp_path / "working").rglob("*.parquet"))


def test_probe_kernel_refuses_an_unfilled_commit(monkeypatch, tmp_path):
    pins, _ = _fake_pins(tmp_path)
    mod = _load_kernel(monkeypatch, tmp_path, pins)
    with pytest.raises(SystemExit):
        mod.main()


# ----------------------------------------------------------------- build kernels A/B (step 2) --
import gzip  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from frontier_ai.corpus.prior_docs import document_digests  # noqa: E402
from frontier_ai.corpus.slice_build import iter_output, text_digest  # noqa: E402
from frontier_ai.evaluation import belebele as bb  # noqa: E402
from frontier_ai.evaluation.suite import build_suite, write_suite  # noqa: E402
from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2  # noqa: E402

BUILD = ROOT / "scripts" / "build_slice2.py"
REV = "0123456789abcdef0123456789abcdef01234567"
_HI = "नदी पहाड़ गाँव शहर किताब पानी बादल सूरज हवा मिट्टी लोग खेत रास्ता घर पेड़ फूल".split()
LONG_SUITE = " ".join(f"शब्द{i}" for i in range(20))
PAIRS = [("s-0", "रात को चाँद निकला और तारे चमके"), ("s-1", LONG_SUITE)]
URDU = [
    "یہ ایک اردو جملہ ہے جس میں ٹ ڈ ڑ ں ے ھ جیسے حروف آتے ہیں۔ " * 3,
    "دوسرا اردو مضمون یہاں ہے اور اس میں بھی ے اور ں موجود ہیں۔ " * 3,
    "تیسرا اردو مضمون صرف پہلے حصے میں تھا اور یہاں ٹ ڈ ڑ ں ے ھ ہیں۔ " * 3,
]


def _hi(seed: int, n: int = 40) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_HI) + str(rng.randint(0, 99)) for _ in range(n))


def _bele_passage(lang: str) -> str:
    return f"Passage zero in {lang}: " + " ".join(f"w{i}{lang}" for i in range(16)) + "."


def _fake_belebele(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for lang in bb.LANGUAGES:
        rows = []
        for q in (1, 2):
            r = {
                "link": "https://example.org/0",
                "question_number": q,
                "flores_passage": _bele_passage(lang),
                "question": f"Question {q}?",
                "correct_answer_num": "1",
                "dialect": bb.FILES[lang][0],
                "ds": "x",
            }
            r.update({f"mc_answer{j}": f"option {j}" for j in range(1, 5)})
            rows.append(r)
        (d / f"{bb.FILES[lang][0]}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(scope="module")
def tok2():
    return load_frontier_tokenizer_v2()


def _pack(texts: list[str], path: Path, tok) -> None:
    # imported here, not at the top: importing the submodule at collection time would replace the
    # function frontier_ai.corpus.pack for test files collected after this one
    from frontier_ai.corpus.pack import OrdinaryEncoder, pack_file

    enc = OrdinaryEncoder(tok)
    src = path.with_suffix(".jsonl.gz")
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        for i, t in enumerate(texts):
            rec = {"id": f"slice1-{path.stem}#{i}", "text": t, "tokens": len(enc.encode(t))}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pack_file(src, path, tokenizer=tok, encoder=enc, expected_docs=len(texts), expected_tokens=None)


def test_document_digests_recover_every_packed_text(tmp_path, tok2):
    texts = [_hi(i, 1 + 37 * i) for i in range(300)] + ["x", "<|endoftext|> as text", "😀 ਪਾਣੀ اردو"]
    _pack(texts, tmp_path / "t.bin", tok2)
    eot = tok2.special_token_ids["<|endoftext|>"]
    for batch in (50, 10**7):  # documents split over many batches, and one batch
        got = document_digests(tmp_path / "t.bin", tok2, eot, batch_tokens=batch)
        assert len(got) == len(texts) and set(got) == {text_digest(t) for t in texts}
    raw = (tmp_path / "t.bin").read_bytes()
    (tmp_path / "cut.bin").write_bytes(raw[:-2])
    with pytest.raises(ValueError, match="endoftext"):
        document_digests(tmp_path / "cut.bin", tok2, eot)


def _slice2_fixture(tmp_path: Path, tok2) -> dict[str, Path]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    work = tmp_path / "work"
    in_slice1 = _hi(50)
    rows = {
        ("hi", 1): [
            ("a", "web", _hi(1)),  # kept
            ("b", "web", in_slice1),  # slice1_duplicate
            ("c", "web", "शुरू " + _bele_passage("bn") + " अंत"),  # suite_ngram (Belebele)
            ("d", "web", _hi(2)),  # kept
        ],
        ("hi", 2): [
            ("e", "web", _hi(1)),  # cross_file_duplicate (same text as data-1 row 0)
            ("f", "web", _hi(3)),  # kept
            ("g", "web", "पहले " + LONG_SUITE + " बाद"),  # suite_ngram (held-out suite)
        ],
        ("ur", 1): [("u1", "web", URDU[0]), ("u3", "web", URDU[2])],
        ("ur", 2): [("u2", "web", URDU[1])],
    }
    codes = {"hi": ("hin", "devanagari"), "ur": ("urd", "arabic")}
    entries = []
    for (lang, n), rs in rows.items():
        code, script = codes[lang]
        rel = f"verified/{code}/data-{n}.parquet"
        path = work / "parquet" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table({"doc_id": [r[0] for r in rs], "type": [r[1] for r in rs], "text": [r[2] for r in rs]})
        pq.write_table(table, str(path), row_group_size=2)
        data = path.read_bytes()
        entries.append(
            {
                "source_id": f"sangraha-verified-{code}-data-{n}",
                "language": lang,
                "sangraha_code": code,
                "script": script,
                "path": rel,
                "url": f"https://h/{REV}/{rel}",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "schema": "frontier-v2-source-pins-v1",
                "slice_id": "test-slice2",
                "dataset": "t/sangraha",
                "subset": "verified",
                "revision": REV,
                "license_id": "CC-BY-4.0",
                "attribution": "test attribution",
                "total_bytes": sum(e["size"] for e in entries),
                "files": entries,
            }
        )
    )

    class _D:
        def __init__(self, doc_id, text):
            self.doc_id, self.language, self.source_id, self.text = doc_id, "hi", "suite-src", text

    suite = tmp_path / "SUITE.json"
    write_suite(suite, build_suite([_D(*t) for t in PAIRS], "test-suite", {"corpus": "t"}))
    held = tmp_path / "heldout-v1.jsonl"
    held.write_text(
        "".join(json.dumps({"doc_id": i, "language": "hi", "text": t}, ensure_ascii=False) + "\n" for i, t in PAIRS),
        encoding="utf-8",
    )
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    s1 = {"hi": [in_slice1, _hi(60)], "ur": ["کچھ اور اردو متن جو صرف پہلے حصے میں ہے"]}
    files = []
    for lang, texts in s1.items():
        _pack(texts, tokens / f"{lang}.bin", tok2)
        data = (tokens / f"{lang}.bin").read_bytes()
        files.append({"language": lang, "path": f"{lang}.bin", "sha256": hashlib.sha256(data).hexdigest(), "docs": len(texts)})
    s1m = tmp_path / "slice1_manifest.json"
    s1m.write_text(json.dumps({"packed_id": "test-slice1", "files": files}))
    bdir = tmp_path / "belebele"
    _fake_belebele(bdir)
    return {"work": work, "pins": pins, "suite": suite, "held": held, "tokens": tokens, "s1m": s1m, "bdir": bdir}


def _load_build(monkeypatch):
    spec = importlib.util.spec_from_file_location("build_slice2", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "FREE_SPACE_MARGIN", 0)
    return mod


def _build_args(fx: dict[str, Path], out: Path, data: Path, *extra: str) -> list[str]:
    return [
        "--part", "A", "--languages", "hi,ur", "--pins", str(fx["pins"]), "--suite", str(fx["suite"]),
        "--heldout-jsonl", str(fx["held"]), "--tokens-dir", str(fx["tokens"]), "--slice1-manifest",
        str(fx["s1m"]), "--belebele-dir", str(fx["bdir"]), "--no-verify", "--work", str(fx["work"]),
        "--data-out", str(data), "--out", str(out), "--keep-parquet", *extra,
    ]  # fmt: skip


def test_build_slice2_applies_the_rules_and_the_four_safeguards(tmp_path, monkeypatch, tok2):
    fx = _slice2_fixture(tmp_path, tok2)
    mod = _load_build(monkeypatch)
    out, data = tmp_path / "out", tmp_path / "data"
    assert mod.main(_build_args(fx, out, data, "--workers", "1")) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["languages"] == ["ur", "hi"] and s["done"] == ["ur", "hi"] and s["not_done"] == []
    assert s["complete"] is False  # a subset of part A is never complete
    by = {f["source_id"]: f for f in s["files"]}
    d1, d2 = by["sangraha-verified-hin-data-1"], by["sangraha-verified-hin-data-2"]
    assert d1["removed_docs"]["slice1_duplicate"]["docs"] == 1
    assert d1["removed_docs"]["suite_ngram"]["docs"] == 1 and d1["suite"]["belebele_touching_docs"] == 1
    assert d2["removed_docs"]["cross_file_duplicate"]["docs"] == 1
    assert d2["removed_docs"]["suite_ngram"]["docs"] == 1 and d2["suite"]["belebele_touching_docs"] == 0
    assert d1["kept"]["docs"] == 2 and d2["kept"]["docs"] == 1
    assert s["slice1"]["per_language"]["hi"] == {"docs": 2, "distinct_texts": 2, "sha256_verified": True}
    assert s["suite"]["belebele"]["texts"] == 13 * (1 + 2 + 8)
    # the corpus: one text file and one token file per language, both sources joined
    recs = list(iter_output(data / "hi.jsonl.gz"))
    assert [r["id"] for r in recs] == [
        "sangraha-verified-hin-data-1#0", "sangraha-verified-hin-data-1#3", "sangraha-verified-hin-data-2#1",
    ]  # fmt: skip
    text = (data / "hi.jsonl.gz").read_bytes()
    assert _hi(50) not in "".join(r["text"] for r in recs) and b"Passage zero" not in gzip.decompress(text)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m == json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert m["schema"] == "frontier-packed-manifest-v1" and [f["language"] for f in m["files"]] == ["ur", "hi"]
    for f in m["files"]:
        assert hashlib.sha256((data / f["path"]).read_bytes()).hexdigest() == f["sha256"]
        assert hashlib.sha256((data / f["text"]["path"]).read_bytes()).hexdigest() == f["text"]["sha256"]
        assert f["n_train"] + f["n_val"] == f["text"]["tokens"] + f["docs"]  # one <|endoftext|> per document
    hi = next(f for f in m["files"] if f["language"] == "hi")
    eot = tok2.special_token_ids["<|endoftext|>"]
    assert set(document_digests(data / "hi.bin", tok2, eot)) == {text_digest(r["text"]) for r in recs}
    assert hi["sources"] == ["sangraha-verified-hin-data-1", "sangraha-verified-hin-data-2"]
    assert not (data / "parts").exists() and (data / "ATTRIBUTION.txt").is_file()
    strata = {json.loads(x)["stratum"] for x in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()}
    assert {"removed:slice1_duplicate", "removed:cross_file_duplicate", "kept"} <= strata
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "Safeguards (EXP-046)" in summary and "Output suite hits: 0" in summary
    published = (out / "samples.jsonl").read_text(encoding="utf-8")
    assert LONG_SUITE not in published and "Passage zero" not in published  # no protected text is published

    # two worker processes (as on Kaggle) give the same corpus, byte for byte
    out2, data2 = tmp_path / "out2", tmp_path / "data2"
    cmd = [sys.executable, str(BUILD), *_build_args(fx, out2, data2, "--workers", "2", "--max-docs", "100")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in ("hi.jsonl.gz", "hi.bin", "ur.jsonl.gz", "ur.bin"):
        assert (data2 / name).read_bytes() == (data / name).read_bytes(), name


def test_build_slice2_stops_cleanly(tmp_path, monkeypatch, tok2):
    fx = _slice2_fixture(tmp_path, tok2)
    mod = _load_build(monkeypatch)
    # an edited held-out text: STOP before reading any data
    good = fx["held"].read_text(encoding="utf-8")
    fx["held"].write_text(good.replace("चाँद", "सूरज"), encoding="utf-8")
    assert mod.main(_build_args(fx, tmp_path / "o1", tmp_path / "d1", "--workers", "1")) == 2
    assert not (tmp_path / "o1").exists()
    fx["held"].write_text(good, encoding="utf-8")
    # a slice-1 token file that is not EXP-037's: the run stops and that language leaves no files
    raw = bytearray((fx["tokens"] / "hi.bin").read_bytes())
    raw[0] ^= 1  # one token changed
    (fx["tokens"] / "hi.bin").write_bytes(bytes(raw))
    out, data = tmp_path / "o2", tmp_path / "d2"
    assert mod.main(_build_args(fx, out, data, "--workers", "1")) == 1
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    # hi is built first (larger files); a failure stops the run, so ur is not started either
    assert s["done"] == [] and s["not_done"] == ["ur", "hi"] and "hi failed" in s["stopped"]
    assert "does not match the EXP-037 manifest" in s["stopped"] and s["complete"] is False
    assert not list(data.glob("hi.*")) and not list(data.glob("ur.*"))
    # the deadline: languages not started are not built and are listed
    out, data = tmp_path / "o3", tmp_path / "d3"
    assert mod.main(_build_args(fx, out, data, "--workers", "1", "--deadline-hours", "0")) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["done"] == [] and s["stopped"].startswith("deadline") and s["complete"] is False
