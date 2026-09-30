"""EXP-037: Frontier Tokenizer v2 (v1 + special tokens) and packing v2-slice1 into token files."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from frontier_ai.corpus.pack import VAL_PPM, OrdinaryEncoder, PackError, is_validation, pack_file
from frontier_ai.data.dataset import TokenDataset
from frontier_ai.tokenization.frozen import (
    ENDOFTEXT,
    V2_SPECIAL_TOKENS,
    artifact_dir_sha256,
    derive_with_special_tokens,
    load_frontier_tokenizer,
    load_frontier_tokenizer_v2,
    read_freeze,
)

REPO = Path(__file__).resolve().parents[1]
V1_DIR_SHA = "b39afa08fa86b0d1ccf7b7e1b59df305b5678bdd59061c6fae01245061321144"

TEXTS = [
    "The quick brown fox jumps over the lazy dog, 42 times!",
    "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं।\nदूसरी पंक्ति।",
    "আমি বাংলায় কথা বলতে ভালোবাসি। [phone] [email]",
    "اردو ایک خوبصورت زبان ہے۔ ٹھیک ہے",
    f"A page that literally contains {ENDOFTEXT} and <|pad|> in its text.",
    "தமிழ் ஒரு பழமையான மொழி ஆகும். Kal meeting 10:30 AM ko hai — मैं ready हूँ!",
    "ਪੰਜਾਬੀ ਇੱਕ ਮਿੱਠੀ ਬੋਲੀ ਹੈ। architectਾਂ",
    "क़िला ज़रूर; ক্ষ; க்‍ஷ; a\u200cb\tline\nnext  double  space",
]


@pytest.fixture(scope="module")
def v1():
    return load_frontier_tokenizer()


@pytest.fixture(scope="module")
def v2():
    return load_frontier_tokenizer_v2()


# ------------------------------------------------------------------ tokenizer v2 --
def test_v2_is_v1_plus_128_specials(v1, v2):
    assert v2.vocab_size == 32896 == 257 * 128
    assert v2.merges == v1.merges and v2._pretoken == v1._pretoken  # noqa: SLF001
    assert list(v2.special_token_ids) == list(V2_SPECIAL_TOKENS)
    assert v2.special_token_ids[ENDOFTEXT] == 32768 and v2.special_token_ids["<|pad|>"] == 32769
    assert max(v2.special_token_ids.values()) == 32895
    for s in V2_SPECIAL_TOKENS:
        assert v2.encode(s) == [v2.special_token_ids[s]] and v2.decode([v2.special_token_ids[s]]) == s


def test_ordinary_text_encodes_exactly_as_v1(v1, v2):
    for text in TEXTS:
        ids = v2.encode_ordinary(text)
        assert ids == v1.encode(text) == v1.encode_ordinary(text)
        assert max(ids) < 32768 and v2.decode(ids) == text
    # the plain encode() WOULD turn special strings in text into boundaries: that is why training
    # text goes through encode_ordinary
    assert 32768 in v2.encode(TEXTS[4]) and 32768 not in v2.encode_ordinary(TEXTS[4])


def test_v2_freeze_record_and_v1_untouched():
    freeze = read_freeze(REPO / "tokenizers" / "frontier-tokenizer-v2")
    assert freeze["tokenizer_version"] == "v2" and freeze["status"] == "frozen"
    assert freeze["lineage"]["base_dir_sha256"] == V1_DIR_SHA
    assert all(g["pass"] for g in freeze["gates"].values())
    assert freeze["gates"]["C_ordinary_equals_v1"]["texts"] >= 1000
    assert freeze["artifact"]["dir_sha256"] == artifact_dir_sha256(
        REPO / "tokenizers/frontier-tokenizer-v2/tokenizer"
    )
    assert artifact_dir_sha256(REPO / "tokenizers/frontier-tokenizer-v1/tokenizer") == V1_DIR_SHA


def test_derive_refuses_bad_specials(v1, v2):
    from frontier_ai.tokenization.frozen import FrozenTokenizerError

    with pytest.raises(FrozenTokenizerError):
        derive_with_special_tokens(v2, ["<|x|>"])  # base already has specials
    with pytest.raises(FrozenTokenizerError):
        derive_with_special_tokens(v1, ["<|a|>", "<|a|>"])


def test_freeze_script_is_deterministic_and_never_overwrites(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("freeze_v2", REPO / "scripts" / "freeze_tokenizer_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--root", str(REPO)]) == 2  # the committed freeze exists: refused
    root = tmp_path / "repo"
    shutil.copytree(
        REPO / "tokenizers" / "frontier-tokenizer-v1", root / "tokenizers" / "frontier-tokenizer-v1"
    )
    for exp in ("EXP-035", "EXP-036"):
        (root / "evals/results" / exp).mkdir(parents=True)
        shutil.copy(
            REPO / "evals/results" / exp / "samples.jsonl", root / "evals/results" / exp / "samples.jsonl"
        )
    assert mod.main(["--root", str(root)]) == 0
    made = root / "tokenizers/frontier-tokenizer-v2/tokenizer/bpe_python.json"
    assert (
        made.read_bytes()
        == (REPO / "tokenizers/frontier-tokenizer-v2/tokenizer/bpe_python.json").read_bytes()
    )


# ------------------------------------------------------------------ packing --
def test_ordinary_encoder_matches_and_is_bounded(v2):
    enc = OrdinaryEncoder(v2, cache_limit=50)
    for text in TEXTS * 3:
        assert enc.encode(text) == v2.encode_ordinary(text)
    assert len(enc.cache) <= 50


def test_validation_split_is_deterministic_and_about_half_a_percent():
    ids = [f"ai4bharat-sangraha-verified-hin-0#{i}" for i in range(200_000)]
    picked = [i for i in ids if is_validation(i)]
    assert 0.004 < len(picked) / len(ids) < 0.006
    assert picked == [i for i in ids if is_validation(i, VAL_PPM)]
    assert not is_validation("x", 0) and is_validation("x", 1_000_000)


def _write_corpus(path: Path, v1, n: int = 300, bad_row: int | None = None) -> tuple[int, int]:
    tokens = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        for i in range(n):
            text = f"{TEXTS[i % len(TEXTS)]} #{i}"
            count = len(v1.encode(text)) + (1 if i == bad_row else 0)
            tokens += count
            fh.write(
                json.dumps({"id": f"src-hin-0#{i}", "text": text, "tokens": count}, ensure_ascii=False) + "\n"
            )
    return n, tokens


def test_pack_file_layout_counts_and_split(tmp_path, v1, v2):
    src = tmp_path / "hi.jsonl.gz"
    docs, tokens = _write_corpus(src, v1)
    out = tmp_path / "out" / "hi.bin"
    stats = pack_file(
        src,
        out,
        tokenizer=v2,
        encoder=OrdinaryEncoder(v2),
        expected_docs=docs,
        expected_tokens=tokens,
        val_ppm=200_000,
        provenance={"exp_id": "EXP-037"},
    )
    ds = TokenDataset(out)
    assert ds.meta.vocab_size == 32896 and ds.meta.dtype == "uint16" and ds.meta.n_tokens == stats["tokens"]
    assert ds.meta.n_train + ds.meta.n_val - docs == tokens
    assert ds.meta.source_provenance == {"exp_id": "EXP-037"}
    # decode each split back into documents: order and membership follow the rule exactly
    rows = [json.loads(line) for line in gzip.open(src, "rt", encoding="utf-8")]
    for split, want in (("train", False), ("val", True)):
        arr = np.asarray(ds.split(split)).tolist()
        assert arr[-1] == 32768
        pieces, cur = [], []
        for t in arr:
            if t == 32768:
                pieces.append(v2.decode(cur))
                cur = []
            else:
                cur.append(t)
        assert pieces == [r["text"] for r in rows if is_validation(r["id"], 200_000) is want]
    assert 0 < stats["splits"]["val"]["docs"] < docs
    assert stats["splits"]["train"]["bytes"] + stats["splits"]["val"]["bytes"] == sum(
        len(r["text"].encode("utf-8")) for r in rows
    )
    assert stats["checks"]["file_scan"]["endoftext"] == docs
    assert stats["checks"]["file_scan"]["ids_above_endoftext"] == 0
    assert not list(out.parent.glob("*.part")) and not list(out.parent.glob("*.tmp"))


def test_pack_file_refuses_a_count_mismatch(tmp_path, v1, v2):
    src = tmp_path / "hi.jsonl.gz"
    _write_corpus(src, v1, n=50, bad_row=17)
    with pytest.raises(PackError, match="src-hin-0#17"):
        pack_file(
            src,
            tmp_path / "hi.bin",
            tokenizer=v2,
            encoder=OrdinaryEncoder(v2),
            expected_docs=None,
            expected_tokens=None,
        )
    assert not (tmp_path / "hi.bin").exists()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cli_fixture(tmp_path: Path, v1) -> tuple[Path, Path, str]:
    data_in = tmp_path / "v2"
    data_in.mkdir()
    files = []
    for lang, n in (("hi", 240), ("ta", 130)):
        path = data_in / f"{lang}.jsonl.gz"
        docs, tokens = _write_corpus(path, v1, n=n)
        files.append(
            {
                "language": lang,
                "source_id": f"src-{lang}",
                "path": path.name,
                "docs": docs,
                "tokens": tokens,
                "chars": 0,
                "sha256_gz": _sha(path),
            }
        )
    manifest = {
        "schema": "frontier-corpus-manifest-v1",
        "corpus_id": "test-corpus",
        "complete": True,
        "source": {"revision": "r", "license_id": "CC-BY-4.0", "attribution": "test"},
        "files": files,
        "totals": {"docs": sum(f["docs"] for f in files), "tokens": sum(f["tokens"] for f in files)},
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return data_in, mpath, _sha(mpath)


def _run_cli(args: list[str]) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location("pack_cli", REPO / "scripts" / "pack_sangraha_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(args)


def test_cli_end_to_end_reuse_and_refusals(tmp_path, v1):
    data_in, mpath, msha = _cli_fixture(tmp_path, v1)
    common = [
        "--manifest",
        str(mpath),
        "--data-in",
        str(data_in),
        "--pins",
        str(tmp_path / "none.json"),
        "--workers",
        "1",
    ]
    # only the accepted manifest is packed
    assert _run_cli(common + ["--data-out", str(tmp_path / "a"), "--out", str(tmp_path / "ra")]) == 2
    base = common + ["--expect-manifest-sha256", msha]
    assert _run_cli(base + ["--data-out", str(tmp_path / "a"), "--out", str(tmp_path / "ra")]) == 0
    summary = json.loads((tmp_path / "ra" / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is False  # not the D-045 manifest
    assert summary["input"]["manifest_is_d045"] is False
    assert summary["totals_equal_manifest"] is True
    assert summary["totals"]["docs"] == 370
    assert summary["totals"]["tokens"] - 370 == summary["totals"]["source_tokens_v1"]
    man = json.loads((tmp_path / "ra" / "manifest.json").read_text(encoding="utf-8"))
    assert [f["language"] for f in man["files"]] == ["hi", "ta"]
    for f in man["files"]:
        assert _sha(tmp_path / "a" / f["path"]) == f["sha256"]
    assert "check" in (tmp_path / "ra" / "SUMMARY.txt").read_text(encoding="utf-8")
    # second run reuses both files
    assert _run_cli(base + ["--data-out", str(tmp_path / "a"), "--out", str(tmp_path / "ra")]) == 0
    again = json.loads((tmp_path / "ra" / "summary.json").read_text(encoding="utf-8"))
    assert again["reused_files"] == ["hi", "ta"]
    # a fresh run gives byte-identical token files
    assert _run_cli(base + ["--data-out", str(tmp_path / "b"), "--out", str(tmp_path / "rb")]) == 0
    for name in ("hi.bin", "ta.bin", "hi.meta.json", "ta.meta.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    # a damaged input file stops the run
    (data_in / "ta.jsonl.gz").write_bytes(b"not the corpus")
    assert _run_cli(base + ["--data-out", str(tmp_path / "c"), "--out", str(tmp_path / "rc")]) == 1


def test_cli_two_workers_equal_one(tmp_path, v1):
    data_in, mpath, msha = _cli_fixture(tmp_path, v1)
    outs = {}
    for workers in (1, 2):
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "pack_sangraha_v2.py"),
            "--manifest",
            str(mpath),
            "--expect-manifest-sha256",
            msha,
            "--data-in",
            str(data_in),
            "--data-out",
            str(tmp_path / f"d{workers}"),
            "--out",
            str(tmp_path / f"r{workers}"),
            "--pins",
            str(tmp_path / "none.json"),
            "--workers",
            str(workers),
        ]
        subprocess.run(cmd, check=True, cwd=REPO, capture_output=True)
        outs[workers] = {n: (tmp_path / f"d{workers}" / n).read_bytes() for n in ("hi.bin", "ta.bin")}
    assert outs[1] == outs[2]
