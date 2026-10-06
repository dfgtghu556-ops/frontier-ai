"""EXP-045: Belebele scoring, same-text bits, the 13-gram contamination scan, the evaluation script,
the fp16 weights copy and ``generate.py --weights/--interactive`` (CPU, tiny models, fake data)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frontier_ai.config import ModelConfig
from frontier_ai.evaluation import belebele as bb
from frontier_ai.evaluation import token_ngrams as tn
from frontier_ai.model.gpt import GPT
from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import belebele_contamination as bc  # noqa: E402
import eval_exp045 as ev  # noqa: E402

BLOCK = 64
_WORDS = "river mill village market morning tea rain field school road temple train city song".split()


@pytest.fixture(scope="module")
def tok():
    return load_frontier_tokenizer_v2()


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=2, n_embd=32, block_size=BLOCK, pos="rope")
    return GPT(cfg).eval()


def _row(lang: str, i: int, q: int, passage: str, question: str, opts: list[str], ans: int) -> dict:
    r = {
        "link": f"https://example.org/{i}",
        "question_number": q,
        "flores_passage": passage,
        "question": question,
        "correct_answer_num": str(ans),
        "dialect": bb.FILES[lang][0],
        "ds": "x",
    }
    r.update({f"mc_answer{j}": o for j, o in enumerate(opts, 1)})
    return r


def _fake_belebele(d: Path, langs=("en", "hi"), n_passages: int = 3) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        rows = []
        for i in range(n_passages):
            words = np.random.default_rng([i, ord(lang[0]), ord(lang[1])]).permutation(_WORDS).tolist()
            passage = f"Passage {i} in {lang}: " + " ".join(words) + "."
            for q in (1, 2):
                rows.append(
                    _row(
                        lang,
                        i,
                        q,
                        passage,
                        f"Question {q} about passage {i}?",
                        [f"answer one {i}", f"answer two {q}", "a third answer", "four"],
                        1 + (i + q) % 4,
                    )
                )
        (d / f"{bb.FILES[lang][0]}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )


def _weights_file(model: GPT, path: Path, compiled_prefix: bool = True) -> Path:
    pre = "_orig_mod." if compiled_prefix else ""
    sd = {pre + k: v.detach().float().clone() for k, v in model.state_dict().items()}
    torch.save({"model_config": asdict(model.cfg), "state_dict": sd, "exp_id": "EXP-043", "step": 7}, path)
    return path


# ------------------------------------------------------------------------------ data ------------
def test_git_blob_sha1_matches_git(tmp_path):
    data = "नमस्ते\nhello\n".encode()
    got = subprocess.run(["git", "hash-object", "--stdin"], input=data, capture_output=True, check=True)
    assert bb.git_blob_sha1(data) == got.stdout.decode().strip()


def test_pinned_files_cover_our_13_languages():
    manifest = json.loads((ROOT / "evals/results/EXP-037/manifest.json").read_text(encoding="utf-8"))
    assert sorted(bb.FILES) == sorted(f["language"] for f in manifest["files"])
    assert all(len(oid) == 40 and size > 100_000 for _d, oid, size in bb.FILES.values())
    assert bb.file_url("hi").endswith(f"/resolve/{bb.REVISION}/data/hin_Deva.jsonl")


def test_parse_items_checks_fields(tmp_path):
    _fake_belebele(tmp_path)
    data = (tmp_path / "eng_Latn.jsonl").read_bytes()
    items = bb.parse_items(data, "en", expected=6)
    assert items[0].answer == 1 and items[0].key == "https://example.org/0#1"  # 1-based "2" -> 0-based 1
    with pytest.raises(bb.BelebeleError, match="expected 900"):
        bb.parse_items(data, "en")
    with pytest.raises(bb.BelebeleError, match="dialect"):
        bb.parse_items(data, "hi", expected=None)
    with pytest.raises(bb.BelebeleError, match="duplicate"):
        bb.parse_items(data + data, "en", expected=None)
    with pytest.raises(bb.BelebeleError, match="pinned"):
        bb.verify_bytes("en", data)
    assert len(bb.passages(items)) == 3
    assert all(k.count("|") == 1 for k in bb.passages(items))  # two questions per passage


def test_download_verifies(tmp_path, monkeypatch):
    _fake_belebele(tmp_path / "src")
    data = (tmp_path / "src" / "eng_Latn.jsonl").read_bytes()
    monkeypatch.setitem(bb.FILES, "en", ("eng_Latn", bb.git_blob_sha1(data), len(data)))
    calls = []
    path = bb.download("en", tmp_path / "dl", opener=lambda url: calls.append(url) or data)
    assert path.read_bytes() == data and len(calls) == 1
    bb.download("en", tmp_path / "dl", opener=lambda url: calls.append(url) or data)
    assert len(calls) == 1  # a verified copy is reused
    with pytest.raises(bb.BelebeleError):
        bb.download("en", tmp_path / "dl2", opener=lambda url: data + b"x")


def test_wilson_interval():
    lo, hi = bb.wilson(450, 900)
    assert lo == pytest.approx(0.4675, abs=1e-3) and hi == pytest.approx(0.5325, abs=1e-3)
    assert bb.wilson(0, 900)[0] == pytest.approx(0.0, abs=1e-12)
    assert not bb.accuracy_report([True] * 250 + [False] * 650)["above_chance"]  # 27.8%: interval reaches 25%
    assert bb.accuracy_report([True] * 300 + [False] * 600)["above_chance"]  # 33.3%


# ---------------------------------------------------------------------------- scoring -----------
def test_segments_score_each_token_once():
    seq = list(range(300))
    for first in (1, 5):
        segs = bb.segments(seq, first, BLOCK)
        covered = []
        for ids, n in segs:
            assert len(ids) <= BLOCK + 1 and 1 <= n <= len(ids) - 1
            covered += ids[len(ids) - n :]
        assert covered == seq[first:]
        assert all(len(ids) - n >= BLOCK // 2 for ids, n in segs[1:])  # at least half a context before
    assert bb.segments(list(range(10)), 1, BLOCK) == [(list(range(10)), 9)]


def test_score_pieces_equals_one_by_one(model):
    rng = np.random.default_rng(0)
    pieces = [(rng.integers(0, 1000, size=n).tolist(), k) for n, k in ((5, 2), (40, 39), (17, 1), (65, 10))]
    got = bb.score_pieces(model, pieces, "cpu", batch_size=3)
    for (ids, n), g in zip(pieces, got):
        x = torch.tensor([ids[:-1]])
        logp = F.log_softmax(model(x).logits.float(), dim=-1)[0]
        want = sum(logp[t, ids[t + 1]].item() for t in range(len(ids) - 1 - n, len(ids) - 1))
        assert g == pytest.approx(want, abs=1e-4)


def test_option_pieces_cut_long_passages(tok):
    item = bb.Item("l", 1, "word " * 200, "Why?", ("yes", "no", "maybe so", "never"), 0)
    pieces, cut = bb.option_pieces(tok, item, BLOCK, eot=32768)
    assert cut
    for (ids, n), opt in zip(pieces, item.options):
        assert len(ids) <= BLOCK + 1 and ids[-n:] == tok.encode_ordinary(opt) and ids[0] != 32768
    short = bb.Item("l", 1, "A short passage.", "Why?", ("yes", "no", "maybe", "never"), 0)
    pieces, cut = bb.option_pieces(tok, short, BLOCK, eot=32768)
    assert not cut and all(ids[0] == 32768 for ids, _ in pieces)


def test_belebele_answer_is_best_score_per_byte(tok, model, monkeypatch):
    items = [bb.Item("l", 1, "p", "q", ("aa", "a", "aaaa", "aaa"), 2)]
    # equal log-probabilities: per byte, the longest option is the least negative
    monkeypatch.setattr(bb, "score_pieces", lambda *a, **k: np.array([-4.0, -4.0, -4.0, -4.0]))
    res = bb.belebele_language(model, tok, items, BLOCK, "cpu")
    assert res["rows"][0]["pred"] == 2 and res["report"]["correct"] == 1
    monkeypatch.setattr(bb, "score_pieces", lambda *a, **k: np.array([-2.0, -1.0, -4.0, -3.0]))  # all -1/byte
    assert bb.belebele_language(model, tok, items, BLOCK, "cpu")["rows"][0]["pred"] == 0  # tie: lowest number


def test_document_bits_matches_manual(tok, model):
    text = "The river flows past the old mill."
    bits, windowed = bb.document_bits(model, tok, [text, "word " * 100], BLOCK, "cpu")
    ids = [32768, *tok.encode_ordinary(text)]
    logp = F.log_softmax(model(torch.tensor([ids[:-1]])).logits.float(), dim=-1)[0]
    want = -sum(logp[t, ids[t + 1]].item() for t in range(len(ids) - 1)) / math.log(2)
    assert bits[0] == pytest.approx(want, abs=1e-3) and windowed == 1


# ------------------------------------------------------------------------- 13-grams -------------
def test_window_hashes_and_scan_across_chunks():
    rng = np.random.default_rng(1)
    gram = rng.integers(0, 30000, size=13)
    stream = rng.integers(0, 30000, size=5000).astype(np.uint16)
    stream[997:1010] = gram  # crosses the chunk boundary at 1000
    near = gram.copy()
    near[-1] = (near[-1] + 1) % 30000
    stream[3000:3013] = near  # 12 of 13 tokens: must not match
    idx = tn.NgramIndex()
    idx.add("planted", [7, 7, *gram.tolist(), 9])
    idx.add("short", [1, 2, 3])
    idx.freeze()
    matched = np.zeros(len(idx.keys), dtype=bool)
    hits = tn.scan(idx, stream, chunk=1000, matched=matched)
    assert hits == {"planted": {"windows": 1, "first_offset": 997}}
    # coverage: "planted" (16 tokens) has 4 distinct 13-grams; only the planted one is in the stream
    assert idx.ref_keys == {"planted": 4} and matched.sum() == 1
    assert tn.coverage(idx, matched) == {"planted": (1, 4)}
    assert idx.too_short == ["short"] and len(idx) == 4
    h = tn.window_hashes(np.arange(20), 13)
    assert len(h) == 8 and len(set(h.tolist())) == 8


def test_contamination_script_end_to_end(tmp_path, tok):
    bdir = tmp_path / "belebele"
    _fake_belebele(bdir)
    items = bb.load_language(bdir, "hi", verify=False, expected=None)
    planted = tok.encode_ordinary(items[0].passage)
    q_planted = tok.encode_ordinary(items[2].question + " and some more words to reach thirteen tokens")
    rng = np.random.default_rng(2)
    files = []
    for lang in ("en", "hi"):
        train = rng.integers(0, 30000, size=4000).astype(np.uint16)
        val = rng.integers(0, 30000, size=500).astype(np.uint16)
        if lang == "hi":
            train[100 : 100 + len(planted)] = planted  # passage 0 inside the TRAIN part
            val[10 : 10 + 13] = planted[:13]  # inside VAL only: not training data, not scanned
        np.concatenate([train, val]).tofile(tmp_path / f"{lang}.bin")
        files.append({"language": lang, "path": f"{lang}.bin", "sha256": "x", "n_train": 4000, "n_val": 500})
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    out = tmp_path / "out"
    rc = bc.main(
        [
            "--data-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--belebele-dir",
            str(bdir),
            "--no-verify",
            "--smoke",
            "--languages",
            "en,hi",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--chunk",
            "1000",
        ]
    )
    assert rc == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    c = json.loads((out / "contamination.json").read_text(encoding="utf-8"))
    assert s["smoke"] and s["train_tokens"] == 8000
    assert s["per_language"]["hi"]["flagged"] == 2  # both questions of passage 0
    assert s["per_language"]["hi"]["flagged_by"]["passage"] == 2
    assert sorted(c["flagged"]["hi"]) == ["https://example.org/0#1", "https://example.org/0#2"]
    assert len(q_planted) >= 13  # (sanity of the fixture text)
    assert "TOTAL: 2 of 12 questions flagged" in (out / "SUMMARY.txt").read_text(encoding="utf-8")
    # coverage follow-up: same scan, written elsewhere, compared with the run above
    cov_out = tmp_path / "coverage"
    rc = bc.main(
        ["--data-dir", str(tmp_path), "--out", str(cov_out), "--belebele-dir", str(bdir), "--no-verify",
         "--smoke", "--languages", "en,hi", "--manifest", str(tmp_path / "manifest.json"), "--chunk", "1000",
         "--coverage", "--test4", str(out / "contamination.json")]
    )  # fmt: skip
    assert rc == 0 and not (cov_out / "contamination.json").exists()
    s2 = json.loads((cov_out / "summary.json").read_text(encoding="utf-8"))
    cv = s2["coverage"]
    assert cv["threshold"] == bc.COVERAGE_THRESHOLD == 0.70 and cv["reproduces_test4"] is True
    assert s2["per_language"] == s["per_language"]
    hi = cv["per_language"]["hi"]
    assert hi["substantially_present"] == 2 and hi["substantially_present_by"]["passage"] == 2
    assert hi["passage_coverage_hist"][-1] == 1 and hi["passages_no_overlap"] == 2
    assert cv["per_language"]["en"]["substantially_present"] == 0
    cf = json.loads((cov_out / "coverage.json").read_text(encoding="utf-8"))
    assert sorted(cf["substantially_present"]["hi"]) == sorted(c["flagged"]["hi"])
    ref = next(r for r in cf["texts"] if r.startswith("hi|passage|"))
    assert cf["texts"][ref][0] == cf["texts"][ref][1] == len(planted) - 12
    assert "TOTAL: 2 of 12 questions substantially present" in (cov_out / "SUMMARY.txt").read_text(
        encoding="utf-8"
    )
    # a single shared 13-token run flags a passage but is far below 70% coverage
    en = bb.load_language(bdir, "en", verify=False, expected=None)
    piece = tok.encode_ordinary(en[0].passage)[:13]
    assert len(tok.encode_ordinary(en[0].passage)) > 30
    raw = np.fromfile(tmp_path / "en.bin", dtype=np.uint16)
    raw[50:63] = piece
    raw.tofile(tmp_path / "en.bin")
    cov3 = tmp_path / "coverage3"
    assert bc.main(
        ["--data-dir", str(tmp_path), "--out", str(cov3), "--belebele-dir", str(bdir), "--no-verify",
         "--smoke", "--languages", "en,hi", "--manifest", str(tmp_path / "manifest.json"), "--chunk", "1000",
         "--coverage", "--test4", str(out / "contamination.json")]
    ) == 0  # fmt: skip
    s3 = json.loads((cov3 / "summary.json").read_text(encoding="utf-8"))
    assert s3["per_language"]["en"]["flagged"] == 2 and s3["coverage"]["reproduces_test4"] is False
    assert s3["coverage"]["per_language"]["en"]["substantially_present"] == 0
    assert s3["coverage"]["per_language"]["en"]["passage_coverage_hist"][0] == 1


# ----------------------------------------------------------------- evaluation script ------------
def test_eval_script_end_to_end(tmp_path, model):
    from frontier_ai.engine.weights import load_weights
    from frontier_ai.evaluation.suite import build_suite, write_suite

    bdir = tmp_path / "belebele"
    _fake_belebele(bdir)
    w = _weights_file(model, tmp_path / "model_final.pt")
    docs = [
        SimpleNamespace(
            doc_id=f"hi-x-{i:06d}", language=lang, source_id="hi-x", text=f"दस्तावेज़ {i} " * (i + 1)
        )
        for i, lang in enumerate(["hi", "hi", "en"])
    ]
    suite = build_suite(docs, "test-suite", {"corpus": "test", "content_sha256": "0"})
    write_suite(tmp_path / "SUITE.json", suite)
    held = tmp_path / "held.jsonl"
    held.write_text("".join(json.dumps(vars(d), ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    contam = tmp_path / "contamination.json"
    contam.write_text(
        json.dumps({"belebele_revision": bb.REVISION, "flagged": {"hi": ["https://example.org/0#1"]}})
    )
    covf = tmp_path / "coverage.json"
    covf.write_text(
        json.dumps(
            {
                "belebele_revision": bb.REVISION,
                "substantially_present": {"hi": ["https://example.org/0#1", "https://example.org/0#2"]},
            }
        )
    )
    out = tmp_path / "out"
    rc = ev.main(
        [
            "--coverage",
            str(covf),
            "--weights",
            str(w),
            "--out",
            str(out),
            "--belebele-dir",
            str(bdir),
            "--languages",
            "en,hi",
            "--no-verify",
            "--smoke",
            "--heldout-jsonl",
            str(held),
            "--suite",
            str(tmp_path / "SUITE.json"),
            "--contamination",
            str(contam),
            "--export-fp16",
            str(out / "model_fp16.pt"),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    t = s["tests"]
    assert s["model"]["step"] == 7 and s["model"]["sha256"] and s["amp"].startswith("none")
    assert t["1_heldout"]["status"] == "RUN" and t["1_heldout"]["documents"] == 3
    assert set(t["1_heldout"]["per_language"]) == {"en", "hi"}
    hi = t["2_belebele"]["per_language"]["hi"]
    assert hi["questions"] == 6 and hi["without_flagged"]["flagged_removed"] == 1
    assert hi["without_substantially_present"]["removed"] == 2
    assert hi["without_substantially_present"]["questions"] == 4
    assert t["2_belebele"]["coverage"] == "applied"
    assert t["3_parallel"]["passages"] == 3 and t["3_parallel"]["per_language"]["en"]["bits_vs_en"] == 1.0
    rows = [json.loads(x) for x in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 * 5 * 2 and rows[0]["prompt"] == "The weather today is"
    items = (out / "belebele_items.jsonl").read_text(encoding="utf-8")
    assert "river" not in items and len(items.splitlines()) == 12  # keys and choices only, no Belebele text
    assert "SMOKE RUN" in (out / "SUMMARY.txt").read_text(encoding="utf-8")
    # the fp16 copy: about half the size, the same model up to fp16 rounding
    f16 = out / "model_fp16.pt"
    assert s["fp16_copy"]["bytes"] == f16.stat().st_size < 0.6 * w.stat().st_size
    m16, info = load_weights(f16)
    x = torch.randint(0, 1000, (1, 20))
    assert info["format"] == "frontier-weights-fp16-v1"
    assert torch.allclose(m16(x).logits, model(x).logits, atol=2e-2)
    # test 1 refuses texts that are not the suite's
    bad = tmp_path / "bad.jsonl"
    bad.write_text(held.read_text(encoding="utf-8").replace("दस्तावेज़ 0", "बदला 0"), encoding="utf-8")
    args = SimpleNamespace(
        heldout_jsonl=str(bad), suite=str(tmp_path / "SUITE.json"), no_verify=False, batch_size=4
    )
    from frontier_ai.evaluation.suite import SuiteError

    with pytest.raises(SuiteError):
        ev.test_heldout(model, load_frontier_tokenizer_v2(), args, BLOCK, torch.device("cpu"), None)
    assert (
        ev.test_heldout(model, None, SimpleNamespace(heldout_jsonl=None), BLOCK, "cpu", None)["status"]
        == "NOT RUN"
    )


def test_everyday_prompts_are_fixed_and_complete(tok):
    spec = json.loads((ROOT / "evals/prompts/everyday-v1.json").read_text(encoding="utf-8"))
    assert spec["prompt_set"] == "everyday-v1" and sorted(spec["prompts"]) == sorted(bb.LANGUAGES)
    for lang, prompts in spec["prompts"].items():
        assert len(prompts) == 5 and all(p.strip() and tok.encode_ordinary(p) for p in prompts), lang
    assert "NOT VERIFIED" in spec["provenance"]


def test_generate_weights_and_interactive(tmp_path, model):
    w = _weights_file(model, tmp_path / "model_final.pt", compiled_prefix=False)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/generate.py"),
        "--weights",
        str(w),
        "--device",
        "cpu",
        "--max-new-tokens",
        "5",
    ]
    one = subprocess.run([*cmd, "--prompt", "भारत"], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert one.returncode == 0, one.stderr
    assert one.stdout.startswith("भारत") and "BASE model" in one.stderr
    loop = subprocess.run(
        [*cmd, "--interactive"],
        input="hello\nनमस्ते\n\n",
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert loop.returncode == 0, loop.stderr
    assert loop.stdout.count("hello") >= 1 and "नमस्ते" in loop.stdout
    assert loop.stderr.count("tokens/s") == 2
