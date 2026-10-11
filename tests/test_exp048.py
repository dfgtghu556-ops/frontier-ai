"""EXP-048 (approved 2026-10-09, "approve A B C", item C): the FineWeb-2 measurement.

Everything runs offline: a fake Hugging Face server (revision, README, folder listings with
pagination, the dataset-viewer size service, parquet files with HTTP Range) feeds the real script.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import random
import re
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "measure_fineweb2.py"
KERNEL = ROOT / "scripts" / "kaggle" / "exp048_kernel.py"
RUNNER = ROOT / "scripts" / "run_kaggle_exp038.ps1"
WRAPPER = ROOT / "scripts" / "run_kaggle_exp048.ps1"
REV = "af9c13333eb981300149d5ca60a8e9d659b276b9"
_HI = "नदी पहाड़ गाँव शहर किताब पानी बादल सूरज हवा मिट्टी लोग खेत रास्ता घर पेड़ फूल".split()
LONG_SUITE = " ".join(f"शब्द{i}" for i in range(20))
PAIRS = [("s-0", "रात को चाँद निकला और तारे चमके"), ("s-1", LONG_SUITE)]
URDU = [
    "یہ ایک اردو جملہ ہے جس میں ٹ ڈ ڑ ں ے ھ جیسے حروف آتے ہیں۔ " * 3,
    "دوسرا اردو مضمون یہاں ہے اور اس میں بھی ے اور ں موجود ہیں۔ " * 3,
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load("measure_fineweb2", SCRIPT)


def _hi(seed: int, n: int = 40) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_HI) + str(rng.randint(0, 99)) for _ in range(n))


# ------------------------------------------------------------------------- fake HF server --
class _Resp:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self._buf, self.status, self.headers = io.BytesIO(body), status, headers or {}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeHF:
    """Routes exact URLs to bodies; honours ``Range: bytes=a-b``; can fail a URL N times first."""

    def __init__(self):
        self.routes: dict[str, tuple[bytes, dict]] = {}
        self.fail: dict[str, list[int]] = {}
        self.seen: list[tuple[str, str | None]] = []
        self.ignore_range = False

    def add(self, url: str, body, headers: dict | None = None) -> None:
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.routes[url] = (body, headers or {})

    def __call__(self, req, timeout=None):
        url = req.full_url
        rng = req.get_header("Range")
        self.seen.append((url, rng))
        if self.fail.get(url):
            code = self.fail[url].pop(0)
            raise urllib.error.HTTPError(url, code, "fake", {}, None)
        if url not in self.routes:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        body, headers = self.routes[url]
        if rng and not self.ignore_range:
            a, b = re.fullmatch(r"bytes=(\d+)-(\d*)", rng).groups()
            return _Resp(body[int(a) : int(b) + 1 if b else None], 206, headers)
        return _Resp(body, 200, headers)


def _parquet(rows: list[tuple[str, str, str]], group: int = 2) -> bytes:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    buf = io.BytesIO()
    t = pa.table({"text": [r[1] for r in rows], "id": [r[0] for r in rows], "url": [r[2] for r in rows]})
    pq.write_table(t, buf, row_group_size=group)
    return buf.getvalue()


# ------------------------------------------------------------------------------- pieces --
def test_sample_rows_are_fixed_distinct_and_sorted(m):
    a = m.sample_rows(10_000, 2_000, 48)
    assert a == m.sample_rows(10_000, 2_000, 48) and a == sorted(set(a)) and len(a) == 2_000
    assert a != m.sample_rows(10_000, 2_000, 49)
    assert m.sample_rows(5, 2_000, 48) == [0, 1, 2, 3, 4]  # small files: every row
    assert m.sample_rows(20, 3, 48) == m.sample_rows(20, 3, 48)
    # the splitmix64 copy matches a plain-integer implementation (the same on every machine)
    M = (1 << 64) - 1

    def ref(x: int) -> int:
        z = (x + 0x9E3779B97F4A7C15) & M
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        return z ^ (z >> 31)

    seed = (48 * 0x9E3779B97F4A7C15) & M
    keys = [ref(i ^ seed) for i in range(50)]
    assert m.sample_rows(50, 7, 48) == sorted(sorted(range(50), key=lambda i: keys[i])[:7])


def test_subsets_are_matched_by_code_and_script(m):
    names = [
        "hin_Deva",
        "hin_Latn",
        "hin_Deva_removed",
        "ory_Orya",
        "urd_Arab",
        "urd_Latn",
        "ben_Beng",
        "tel_Telu",
    ]
    s = m.subsets_for(names)
    assert s["hi"] == {
        "measure": ["hin_Deva"],
        "listed_not_measured": ["hin_Latn"],
        "removed_subsets_present": ["hin_Deva_removed"],
    }
    assert s["or"]["measure"] == ["ory_Orya"] and s["ur"]["listed_not_measured"] == ["urd_Latn"]
    assert s["as"]["measure"] == []  # missing in the listing -> an error later, never a guess
    assert set(s) == {"ur", "as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}  # no English


def test_choose_file_is_fixed_by_the_subset_name(m):
    files = [{"path": f"data/x/train/{i:03d}_00000.parquet"} for i in range(8)]
    a = m.choose_file("hin_Deva", files)
    assert a == m.choose_file("hin_Deva", list(reversed(files)))  # order of the listing does not matter
    assert len({m.choose_file(f"s{i}", files)["path"] for i in range(40)}) > 1


def test_listing_follows_pages_and_retries_server_errors(m, monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    hf = FakeHF()
    first = f"{m.HF}/api/datasets/{m.REPO}/tree/{REV}/data?expand=true"
    nxt = f"{m.HF}/api/datasets/{m.REPO}/tree/{REV}/data?expand=true&cursor=abc"
    hf.add(first, [{"type": "directory", "path": "data/a"}], {"Link": f'<{nxt}>; rel="next"'})
    hf.add(nxt, [{"type": "directory", "path": "data/b"}])
    hf.fail[first] = [502, 503]
    assert [e["path"] for e in m.list_tree(REV, "data", hf)] == ["data/a", "data/b"]
    hf.fail[first] = [404]
    with pytest.raises(urllib.error.HTTPError):  # a 4xx is not retried
        m.list_tree(REV, "data", hf)


def test_parquet_rows_from_the_footer_only(m):
    body = _parquet([(str(i), "t" * 30, "u") for i in range(1234)], group=100)
    hf = FakeHF()
    hf.add("https://x/f.parquet", body)
    assert m.remote_parquet_rows("https://x/f.parquet", len(body), hf) == 1234
    assert all(r for _, r in hf.seen)  # Range requests only, never the whole file
    hf.ignore_range = True
    with pytest.raises(ValueError, match="Range"):
        m.remote_parquet_rows("https://x/f.parquet", len(body), hf)


def test_read_sample_reads_rows_across_row_groups(m, tmp_path):
    rows = [(f"id{i}", f"text {i}", f"https://e/{i}") for i in range(11)]
    p = tmp_path / "f.parquet"
    p.write_bytes(_parquet(rows, group=3))
    got = m.read_sample(p, [0, 2, 3, 10])
    assert [(g["row"], g["id"], g["text"], g["url"]) for g in got] == [(i, *rows[i]) for i in (0, 2, 3, 10)]
    with pytest.raises(ValueError, match="sampled rows"):
        m.read_sample(p, [0, 11])


def test_estimate_and_its_interval(m):
    e = m.estimate(1_000_000, 100, 40, [100] * 20 + [300] * 20)
    assert e["keep_rate_new"] == 0.4 and e["mean_v2_tokens_per_kept_doc"] == 200
    assert e["new_clean_v2_tokens"] == pytest.approx(80_000_000)
    lo, hi = e["ci95"]
    assert lo < 80_000_000 < hi and "NOT VERIFIED" in e["label"]
    assert "note" in m.estimate(None, 100, 1, [1])


# ---------------------------------------------------------------------------- full run --
def _pack(texts: list[str], path: Path, tok) -> None:
    from frontier_ai.corpus.pack import OrdinaryEncoder, pack_file

    enc = OrdinaryEncoder(tok)
    src = path.with_suffix(".jsonl.gz")
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        for i, t in enumerate(texts):
            fh.write(
                json.dumps(
                    {"id": f"{path.stem}#{i}", "text": t, "tokens": len(enc.encode(t))}, ensure_ascii=False
                )
                + "\n"
            )
    pack_file(src, path, tokenizer=tok, encoder=enc, expected_docs=len(texts), expected_tokens=None)


def _fake_belebele(d: Path) -> None:
    from frontier_ai.evaluation import belebele as bb

    d.mkdir(parents=True, exist_ok=True)
    for lang in bb.LANGUAGES:
        rows = []
        for q in (1, 2):
            r = {
                "link": "https://example.org/0",
                "question_number": q,
                "flores_passage": f"Passage zero in {lang}: "
                + " ".join(f"w{i}{lang}" for i in range(16))
                + ".",
                "question": f"Question {q}?",
                "correct_answer_num": "1",
                "dialect": bb.FILES[lang][0],
                "ds": "x",
            }
            r.update({f"mc_answer{j}": f"option {j}" for j in range(1, 5)})
            rows.append(r)
        (d / f"{bb.FILES[lang][0]}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _fixture(tmp_path: Path, m, monkeypatch):
    from frontier_ai.data import slices as sl
    from frontier_ai.evaluation.suite import build_suite, write_suite
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    tok2 = load_frontier_tokenizer_v2()
    in_s1, in_s2 = _hi(50), _hi(70)
    # what we already have: two fake slices, mounted like Kaggle datasets
    root = tmp_path / "repo"
    fake_slices = []
    for name, hint, texts in (("s1", "tok2-13lang", {"hi": [in_s1, _hi(51)], "ur": [URDU[1].strip()]}),  # slices hold cleaned text
                              ("s2a", "slice2-a", {"hi": [in_s2], "ur": ["کچھ اور اردو متن"]})):  # fmt: skip
        d = tmp_path / "input" / "datasets" / "u" / hint
        d.mkdir(parents=True)
        files = []
        for lang, ts in texts.items():
            _pack(ts, d / f"{lang}.bin", tok2)
            raw = (d / f"{lang}.bin").read_bytes()
            (d / f"{lang}.meta.json").write_text(json.dumps({"n_train": 1, "n_val": 0}))
            files.append({"language": lang, "path": f"{lang}.bin", "meta": f"{lang}.meta.json", "bytes": len(raw),
                          "n_train": 1, "n_val": 0, "sha256": hashlib.sha256(raw).hexdigest()})  # fmt: skip
        man = root / f"{name}.json"
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps({"files": files}))
        fake_slices.append(
            sl.Slice(name, f"{name}.json", hashlib.sha256(man.read_bytes()).hexdigest(), hint, name)
        )
    monkeypatch.setattr(sl, "EXP047_SLICES", tuple(fake_slices))

    class _D:
        def __init__(self, doc_id, text):
            self.doc_id, self.language, self.source_id, self.text = doc_id, "hi", "suite-src", text

    suite = tmp_path / "SUITE.json"
    write_suite(suite, build_suite([_D(*t) for t in PAIRS], "test-suite", {"corpus": "t"}))
    held = tmp_path / "heldout-v1.jsonl"
    held.write_text(
        "".join(
            json.dumps({"doc_id": i, "language": "hi", "text": t}, ensure_ascii=False) + "\n"
            for i, t in PAIRS
        )
    )
    bdir = tmp_path / "belebele"
    _fake_belebele(bdir)

    # the fake Hugging Face
    hf = FakeHF()
    api, res = f"{m.HF}/api/datasets/{m.REPO}", f"{m.HF}/datasets/{m.REPO}/resolve/{REV}"
    hf.add(
        f"{api}/revision/main", {"sha": REV, "lastModified": "2025-10-27", "cardData": {"license": "odc-by"}}
    )
    hf.add(f"{res}/README.md", b"---\nlicense: odc-by\npretty_name: FineWeb 2\n---\n# FineWeb 2\n")
    page2 = f"{api}/tree/{REV}/data?expand=true&cursor=p2"
    hf.add(f"{api}/tree/{REV}/data?expand=true",
           [{"type": "directory", "path": f"data/{n}"} for n in ("hin_Deva", "hin_Deva_removed", "hin_Latn")],
           {"Link": f'<{page2}>; rel="next"'})  # fmt: skip
    hf.add(page2, [{"type": "directory", "path": "data/urd_Arab"}, {"type": "file", "path": "data/x.md"}])
    hi_rows = lambda k: [  # noqa: E731 - each file: 2 new documents, 2 we have, 1 protected-suite hit
        (f"{k}a", _hi(100 + k), "https://a.example/x?mail=someone@example.com"),
        (f"{k}b", in_s1, "https://b.example/"),
        (f"{k}c", in_s2, "https://c.example/"),
        (f"{k}d", "पहले " + LONG_SUITE + " बाद", "https://d.example/"),
        (f"{k}e", _hi(200 + k), "https://e.example/"),
    ]
    data = {
        "hin_Deva": [_parquet(hi_rows(0)), _parquet(hi_rows(1))],
        "urd_Arab": [
            _parquet([("u0", URDU[0], "https://u.example/0"), ("u1", URDU[1], "https://u.example/1")])
        ],
    }
    for subset, bodies in data.items():
        listing = []
        for i, body in enumerate(bodies):
            path = f"data/{subset}/train/{i:03d}_00000.parquet"
            hf.add(f"{res}/{path}", body)
            listing.append(
                {
                    "type": "file",
                    "path": path,
                    "size": len(body),
                    "lfs": {"oid": hashlib.sha256(body).hexdigest()},
                }
            )
        hf.add(f"{api}/tree/{REV}/data/{subset}/train?expand=true", listing)
    hf.add(f"{m.VIEWER}/size?dataset={m.REPO}&config=hin_Deva",
           {"size": {"splits": [{"split": "train", "num_rows": 10, "num_bytes_parquet_files": 1}]}, "partial": False})  # fmt: skip
    # urd_Arab: the size service fails -> recorded; the footers still count the rows
    args = ["--input-dir", str(tmp_path / "input"), "--heldout-jsonl", str(held), "--suite", str(suite),
            "--belebele-dir", str(bdir), "--no-verify", "--work", str(tmp_path / "work"), "--out", str(tmp_path / "out"),
            "--languages", "hi,ur", "--slices-root", str(root)]  # fmt: skip
    return hf, args


def test_measure_end_to_end(tmp_path, monkeypatch, m):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    hf, args = _fixture(tmp_path, m, monkeypatch)
    assert m.main(args, opener=hf) == 0
    out = tmp_path / "out"
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["schema"] == m.SCHEMA and s["complete"] and not s["smoke"]
    assert (
        s["pin"]["revision"] == REV
        and s["pin"]["license_card"] == "odc-by"
        and s["pin"]["license_api"] == "odc-by"
    )
    assert s["subset_count_in_listing"] == 4 and s["suite"]["heldout_docs"] == 2
    hi, ur = s["languages"]
    assert hi["subset"] == "hin_Deva" and hi["listed_not_measured"] == ["hin_Latn"]
    assert hi["removed_subsets_present"] == ["hin_Deva_removed"]
    assert hi["train_rows_footers"] == 10 and hi["viewer"]["train_rows"] == 10 and hi["rows_agree"] is True
    assert all(f["sha256"] and f["rows_footer"] == 5 for f in hi["files"])
    assert hi["sample_file"]["download"] == "downloaded" and hi["sample_file"]["deleted_after_use"]
    assert hi["rules"]["sampled"] == 5 and hi["rules"]["kept_new"] == 2
    assert hi["rules"]["already_in_slice1_or_2"] == 2 and hi["rules"]["overlap_share_of_survivors"] == 0.5
    assert hi["rules"]["removed"]["suite_ngram"] == 1
    assert hi["prior"]["slices"] == ["s1", "s2a"] and hi["prior"]["distinct_texts"] == 3
    e = hi["estimate"]
    assert e["keep_rate_new"] == 0.4 and e["new_clean_v2_tokens"] == pytest.approx(
        10 * 0.4 * e["mean_v2_tokens_per_kept_doc"]
    )
    assert hi["tokens_v2"]["kept_docs_tokens"] > 0 and 0 < hi["tokens_v2"]["kept_tokens_per_byte"] < 1
    # urdu: one new, one already in slice 1; the size service failed and is recorded as such
    assert ur["rules"]["kept_new"] == 1 and ur["rules"]["already_in_slice1_or_2"] == 1
    assert "error" in ur["viewer"] and ur["rows_agree"] is None and ur["train_rows_used"] == 2
    # no FineWeb-2 data left behind; only the three small files are written
    assert sorted(p.name for p in out.iterdir()) == ["SUMMARY.txt", "samples.jsonl", "summary.json"]
    assert not list((tmp_path / "work").rglob("*.parquet")) and not list((tmp_path / "work").rglob("*.gz"))
    samples = [json.loads(x) for x in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert samples and all(x["source"].startswith(f"{m.REPO}@{REV}") for x in samples)
    urls = {x["url"] for x in samples}
    assert "https://a.example/x?mail=[email]" in urls and not any(
        "someone@" in (u or "") for u in urls
    )  # masked
    assert "https://e.example/" in urls  # the link itself is kept (attribution)
    assert any(x.get("note", "").startswith("identical to a document") for x in samples)
    txt = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "NOT VERIFIED" in txt and "ODC-By" in txt and "hin_Latn" in txt and "MEASURED 2 of 2" in txt
    # every download used the pinned revision, never "main"
    assert not any("/resolve/main/" in u or "/tree/main/" in u for u, _ in hf.seen)


def test_measure_records_a_failed_language_and_continues(tmp_path, monkeypatch, m):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    hf, args = _fixture(tmp_path, m, monkeypatch)
    # a corrupted file for Hindi: the SHA-256 check refuses it
    url = next(u for u in hf.routes if u.endswith("hin_Deva/train/000_00000.parquet"))
    url2 = url.replace("000_", "001_")
    for u in (url, url2):
        body, h = hf.routes[u]
        hf.routes[u] = (body[:-1] + b"X", h)
    assert m.main(args, opener=hf) == 1
    s = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    hi, ur = s["languages"]
    assert not s["complete"] and "does not match its pin" in hi["error"]
    assert ur["rules"]["kept_new"] == 1  # the other language was still measured
    assert "MEASURED 1 of 2" in (tmp_path / "out" / "SUMMARY.txt").read_text(encoding="utf-8")


# ------------------------------------------------------------------------ kernel / runner --
def test_kernel_is_ascii_and_refuses_an_unfilled_commit(monkeypatch, tmp_path):
    text = KERNEL.read_text(encoding="utf-8")
    assert text.isascii() and '"scripts" / "measure_fineweb2.py"' in text and '"EXP-048"' in text
    mod = _load("exp048_kernel", KERNEL)
    with pytest.raises(SystemExit, match="placeholder"):
        mod.main()
    monkeypatch.setattr(mod, "COMMIT", "a" * 40)
    monkeypatch.setattr(mod, "INPUT", tmp_path)
    monkeypatch.setattr(mod, "sh", lambda cmd: pytest.fail("must not clone without the held-out texts"))
    with pytest.raises(SystemExit, match="heldout-v1.jsonl"):
        mod.main()


def test_runner_wires_exp048_as_a_cpu_run_with_the_existing_datasets():
    text = RUNNER.read_text(encoding="ascii")
    i = text.index('if ($Exp -eq "EXP-048") {\n    # EXP-048')
    block = text[i : text.index("\n}\n", i)]
    assert '$template = "scripts\\kaggle\\exp048_kernel.py"' in block and "$dataFiles = @()" in block
    assert '$kernelSlug = "frontier-exp048"' in block
    assert '-or $Exp -eq "EXP-048")\n' in text[text.index("$cpuOnly = ") :]  # CPU, no GPU quota
    meta = text[text.index('        machine_shape = "NvidiaTeslaT4"\n    }\n') :]
    assert (
        '$meta.dataset_sources = @($datasetId, $heldId, "$user/frontier-v2-slice2-a", "$user/frontier-v2-slice2-b")'
        in meta
    )
    assert '--force --file-pattern EXP-048/.*"' in text  # only the small result folder
    assert 'if ($Exp -eq "EXP-048") { $msgText = "${Exp}: FineWeb-2 measurement' in text
    w = WRAPPER.read_text(encoding="utf-8")
    assert w.isascii() and '-Exp "EXP-048" -Relaunch:$Relaunch' in w
