"""EXP-030: freezing Frontier Tokenizer v1 + the hash-verified loader.

End-to-end on the tiny fake frozen corpus (same pattern as the EXP-B tests):
train a small mark-aware tokenizer, prepare EXP-B-style data (which records the
artifact fingerprint and token counts), then run the real freeze script against
that record. The artifact is rewritten with CRLF line endings first, as the PC
(Windows) writes it, so the byte-exact hashing path is the one under test.

The last test verifies the REAL frozen tokenizer once it is committed
(``tokenizers/frontier-tokenizer-v1``); until then it skips.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from frontier_ai.tokenization.frozen import (
    FREEZE_NAME,
    FrozenTokenizerError,
    artifact_dir_sha256,
    frontier_tokenizer_v1_dir,
    load_frontier_tokenizer,
    load_frozen_tokenizer,
)
from test_exp_b import _build_frontier, _run_script, _train_two_tokenizers
from test_tokenizer_sweep import _make_fake_frozen_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]


def _to_crlf(artifact_dir: Path) -> None:
    for path in artifact_dir.iterdir():
        if path.is_file():
            data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            path.write_bytes(data)


def _setup(tmp_path: Path) -> dict:
    fake = _make_fake_frozen_corpus(tmp_path / "fake", n_hi=30, n_en=12)
    _build_frontier(fake, tmp_path / "frontier")
    name, artifact = _train_two_tokenizers(tmp_path / "frontier", tmp_path)[0]
    _to_crlf(artifact)  # as written on Windows
    data_dir = tmp_path / "expb_data"
    proc = _run_script(
        "prepare_exp_b_data.py",
        ["--frontier-dir", str(tmp_path / "frontier"), "--manifest", str(fake["manifest"]),
         "--freeze", str(fake["freeze"]), "--corpus-dir", str(fake["corpus_dir"]),
         "--tokens", f"{name}={artifact}", "--out", str(data_dir), "--no-record"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return {"fake": fake, "name": name, "artifact": artifact,
            "data": data_dir / "manifest.json", "dest": tmp_path / "tokenizers" / "tok-v1"}


def _freeze(tmp_path: Path, s: dict, vocab: int = 300):
    return _run_script(
        "freeze_tokenizer.py",
        ["--source", str(s["artifact"]), "--exp029-data", str(s["data"]), "--name", s["name"],
         "--dest", str(s["dest"]), "--expect-vocab", str(vocab),
         "--frontier-dir", str(tmp_path / "frontier"), "--manifest", str(s["fake"]["manifest"]),
         "--freeze", str(s["fake"]["freeze"]), "--corpus-dir", str(s["fake"]["corpus_dir"]),
         "--no-record"],
    )


def test_artifact_dir_sha256_is_the_exp_b_algorithm(tmp_path):
    """The frozen fingerprint must be comparable with what EXP-029 recorded."""
    spec = importlib.util.spec_from_file_location("prep", REPO_ROOT / "scripts" / "prepare_exp_b_data.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    d = tmp_path / "a"
    d.mkdir()
    (d / "bpe_python.json").write_bytes(b'{\r\n  "x": 1\r\n}')
    (d / "extra.txt").write_bytes("ग\n".encode())
    assert artifact_dir_sha256(d) == prep._artifact_sha256(d)


def test_freeze_e2e_crlf_artifact(tmp_path):
    s = _setup(tmp_path)
    proc = _freeze(tmp_path, s)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for gate in ("A_identity", "B_structure", "C_token_counts", "D_lossless"):
        assert f"gate {gate}: PASS" in proc.stdout

    tok, freeze = load_frozen_tokenizer(s["dest"])
    assert freeze["status"] == "frozen" and freeze["decision"] == "D-040"
    assert tok.vocab_size == 300 == freeze["artifact"]["vocab_size"]
    # frozen bytes are exactly the source bytes (CRLF preserved) and match EXP-029's record
    src = (s["artifact"] / "bpe_python.json").read_bytes()
    assert b"\r\n" in src
    assert (s["dest"] / "tokenizer" / "bpe_python.json").read_bytes() == src
    exp029 = json.loads(s["data"].read_text(encoding="utf-8"))["tokenizers"][0]
    assert freeze["artifact"]["dir_sha256"] == exp029["artifact_sha256"]
    assert freeze["gates"]["C_token_counts"]["train"]["observed"] == exp029["n_tokens_train"]
    assert freeze["gates"]["C_token_counts"]["held_out"]["observed"] == exp029["n_tokens_val"]
    # golden samples: stored ids reproduce and round-trip
    assert len(freeze["golden_samples"]) == 15
    for sample in freeze["golden_samples"]:
        assert tok.encode(sample["text"]) == sample["ids"]
        assert tok.decode(sample["ids"]) == sample["text"]
    assert not (s["dest"].parent / "tok-v1.tmp").exists()


def test_freeze_refuses_tampered_artifact_and_writes_nothing(tmp_path):
    s = _setup(tmp_path)
    art = s["artifact"] / "bpe_python.json"
    art.write_bytes(art.read_bytes() + b" ")  # still valid JSON, different bytes
    proc = _freeze(tmp_path, s)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "gate A_identity: FAIL" in proc.stdout
    assert not s["dest"].exists()


def test_freeze_refuses_wrong_expected_vocab(tmp_path):
    s = _setup(tmp_path)
    proc = _freeze(tmp_path, s, vocab=32768)
    assert proc.returncode == 1
    assert "gate B_structure: FAIL" in proc.stdout
    assert not s["dest"].exists()


def test_freeze_never_overwrites_an_existing_freeze(tmp_path):
    s = _setup(tmp_path)
    assert _freeze(tmp_path, s).returncode == 0
    before = (s["dest"] / FREEZE_NAME).read_bytes()
    proc = _freeze(tmp_path, s)
    assert proc.returncode == 2
    assert "never edited in place" in proc.stderr
    assert (s["dest"] / FREEZE_NAME).read_bytes() == before


def test_loader_detects_tampering(tmp_path):
    s = _setup(tmp_path)
    assert _freeze(tmp_path, s).returncode == 0
    frozen_file = s["dest"] / "tokenizer" / "bpe_python.json"
    original = frozen_file.read_bytes()

    # 1) git-style line-ending conversion (CRLF -> LF) must be caught
    frozen_file.write_bytes(original.replace(b"\r\n", b"\n"))
    with pytest.raises(FrozenTokenizerError, match="sha256"):
        load_frozen_tokenizer(s["dest"])
    frozen_file.write_bytes(original)

    # 2) an extra file in the artifact dir
    (s["dest"] / "tokenizer" / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FrozenTokenizerError, match="files"):
        load_frozen_tokenizer(s["dest"])
    (s["dest"] / "tokenizer" / "stray.txt").unlink()

    # 3) a golden sample whose recorded ids no longer match
    freeze_path = s["dest"] / FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["golden_samples"][0]["ids"] = freeze["golden_samples"][0]["ids"][:-1]
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(FrozenTokenizerError, match="golden sample"):
        load_frozen_tokenizer(s["dest"])


V1_DIR = frontier_tokenizer_v1_dir()


@pytest.mark.skipif(not (V1_DIR / FREEZE_NAME).is_file(),
                    reason="Frontier Tokenizer v1 not committed yet (EXP-030 PC step)")
def test_frontier_tokenizer_v1_is_frozen_and_verifies():
    """The real v1: byte hashes, structure and PC-recorded golden ids reproduce here."""
    tok = load_frontier_tokenizer()
    freeze = json.loads((V1_DIR / FREEZE_NAME).read_text(encoding="utf-8"))
    assert freeze["decision"] == "D-040"
    assert tok.vocab_size == 32768 and len(tok.merges) == 32512
    assert freeze["artifact"]["pretoken"] == "mark_aware"
    assert freeze["artifact"]["special_tokens"] == []
    assert freeze["lineage"]["sweep_cell"] == "py-mark_aware-32768"
    for key in ("A_identity", "B_structure", "C_token_counts", "D_lossless"):
        assert freeze["gates"][key]["pass"] is True, key
    for sample in freeze["golden_samples"]:
        assert tok.decode(sample["ids"]) == sample["text"]


def test_freeze_recorded_mode_writes_experiment_record(tmp_path):
    """The default (recorded) path — the one the PC runs — must also succeed."""
    import subprocess
    import sys

    s = _setup(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "freeze_tokenizer.py"),
         "--exp-id", "EXP-098",
         "--source", str(s["artifact"]), "--exp029-data", str(s["data"]), "--name", s["name"],
         "--dest", str(s["dest"]), "--expect-vocab", "300",
         "--frontier-dir", str(tmp_path / "frontier"), "--manifest", str(s["fake"]["manifest"]),
         "--freeze", str(s["fake"]["freeze"]), "--corpus-dir", str(s["fake"]["corpus_dir"])],
        capture_output=True, text=True, timeout=900, cwd=work,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((work / "out" / "tokenizer_freeze" / "EXP-098" / "experiment.json")
                        .read_text(encoding="utf-8"))
    assert record["execution"]["status"] == "success"
    assert record["results"]["frozen"] is True
    assert record["results"]["dir_sha256"] == load_frozen_tokenizer(s["dest"])[1]["artifact"]["dir_sha256"]
