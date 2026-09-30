"""EXP-038: the GPU bring-up script runs end to end (CPU smoke mode) and stops on bad data."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gpu_bringup.py"

HINDI = (
    "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं। किसान खेतों में काम करते हैं। "
    "बच्चे सुबह स्कूल जाते हैं और शाम को खेलते हैं। आज मौसम साफ है और हवा ठंडी है। "
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """A tiny hi.bin packed like EXP-037 (tokenizer v2, <|endoftext|> between documents)."""
    sys.path.insert(0, str(ROOT / "src"))
    from frontier_ai.data.dataset import write_tokens
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    tok = load_frontier_tokenizer_v2()
    eot = tok.special_token_ids["<|endoftext|>"]
    ids: list[int] = []
    nbytes: list[int] = []
    nchars: list[int] = []
    for i in range(60):
        doc = HINDI * (1 + i % 3)
        doc_ids = tok.encode_ordinary(doc)
        # document byte/char totals on its first token: split sums stay right for bits per byte
        nbytes += [len(doc.encode("utf-8"))] + [0] * len(doc_ids)
        nchars += [len(doc)] + [0] * len(doc_ids)
        ids += doc_ids + [eot]
    d = tmp_path_factory.mktemp("packed")
    write_tokens(d / "hi.bin", ids, 32896, "bpe", val_frac=0.1, token_bytes=nbytes, token_chars=nchars)
    n_val = int(len(ids) * 0.1)
    manifest = {
        "packed_id": "test-packed",
        "files": [
            {
                "language": "hi",
                "path": "hi.bin",
                "sha256": _sha(d / "hi.bin"),
                "n_train": len(ids) - n_val,
                "n_val": n_val,
            }
        ],
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True, timeout=900
    )


def test_smoke_run_writes_all_parts(packed, tmp_path):
    out = tmp_path / "EXP-038"
    res = _run(
        [
            "--smoke",
            "--skip-tests",
            "--data",
            str(packed / "hi.bin"),
            "--manifest",
            str(packed / "manifest.json"),
            "--out",
            str(out),
            "--scratch",
            str(tmp_path / "scratch"),
        ]
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["schema"] == "frontier-gpu-bringup-v1" and s["smoke"] is True and s["complete"] is True
    p = s["parts"]
    assert p["part0"]["data_sha256_ok"] is True
    # CPU-only: the device comparison is CPU vs CPU and must be exact; fp16 and compile are n/a
    assert p["part1"]["cpu_equals_gpu"]["pass"] is True
    assert p["part1"]["resume"]["pass"] is True and p["part1"]["resume"]["max_abs_diff"] <= 1e-5
    assert p["part1"]["fp16_vs_fp32"]["pass"] is None
    assert p["part1"]["compile"]["pass"] is None
    assert [(r["model"], r["precision"]) for r in p["part2"]] == [
        (m, prec) for m in "SML" for prec in ("fp32", "fp16")
    ]
    assert all(r["status"] == "ok" and r["tokens_per_s"] > 0 for r in p["part2"])
    r3 = p["part3"]
    assert r3["status"] == "done" and r3["finite"] is True
    assert r3["final_val_loss"] < r3["initial_val_loss"]
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(samples) == 5 and all(isinstance(x["completion"], str) for x in samples)
    text = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "SMOKE TEST, not a result" in text and "Part 3" in text
    # only the three small result files are written to --out (checkpoints stay in scratch)
    assert sorted(x.name for x in out.iterdir()) == ["SUMMARY.txt", "samples.jsonl", "summary.json"]


def test_stops_when_the_data_file_does_not_match_the_manifest(packed, tmp_path):
    bad = json.loads((packed / "manifest.json").read_text(encoding="utf-8"))
    bad["files"][0]["sha256"] = "0" * 64
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    out = tmp_path / "out"
    res = _run(
        [
            "--smoke",
            "--skip-tests",
            "--data",
            str(packed / "hi.bin"),
            "--manifest",
            str(tmp_path / "bad.json"),
            "--out",
            str(out),
            "--scratch",
            str(tmp_path / "scratch"),
        ]
    )
    assert res.returncode == 1
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["complete"] is False and "does not match" in s["stopped"]
    assert "part1" not in s["parts"]


def test_the_real_manifest_lists_the_hindi_file():
    """The pre-registered data file is hi.bin of the canonical EXP-037 packing (D-046)."""
    m = json.loads((ROOT / "evals/results/EXP-037/manifest.json").read_text(encoding="utf-8"))
    hi = next(f for f in m["files"] if f["path"] == "hi.bin")
    assert hi["language"] == "hi" and len(hi["sha256"]) == 64
