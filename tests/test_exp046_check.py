"""EXP-046 slice-2 backup check: scripts/check_slice2_backup.py, its Kaggle kernel and runner wiring."""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_slice2_backup as cb  # noqa: E402

KERNEL = ROOT / "scripts" / "kaggle" / "exp046_check_kernel.py"
WRAPPER = ROOT / "scripts" / "run_kaggle_exp046_check.ps1"
RUNNER = ROOT / "scripts" / "run_kaggle_exp038.ps1"


def _gz_two_members(text: bytes) -> bytes:
    half = len(text) // 2
    out = io.BytesIO()
    for chunk in (text[:half], text[half:]):
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as g:
            g.write(chunk)
    return out.getvalue()


def _fake(tmp_path: Path, monkeypatch, *, original: bool = True, keep_gz: bool = False) -> Path:
    """A fake part A (two languages) laid out like the Kaggle mounts; returns the input folder."""
    inp = tmp_path / "input"
    backup = inp / "frontier-v2-slice2-a" / "slice2-A"
    orig = inp / "frontier-exp046-build-a" / "slice2-A"
    backup.mkdir(parents=True)
    if original:
        orig.mkdir(parents=True)
    files = []
    for lang, n_docs in (("en", 5), ("hi", 3)):
        tokens = bytes(range(256)) * (10 + n_docs)
        text = "".join(json.dumps({"id": f"{lang}-{i}", "text": f"doc {i} \u0939"}) + "\n" for i in range(n_docs))
        gz = _gz_two_members(text.encode("utf-8"))
        meta = json.dumps({"n_train": 100 + n_docs, "n_val": 7}).encode()
        for d in [backup] + ([orig] if original else []):
            (d / f"{lang}.bin").write_bytes(tokens)
            (d / f"{lang}.meta.json").write_bytes(meta)
        if keep_gz:
            (backup / f"{lang}.jsonl.gz").write_bytes(gz)
        else:
            (backup / f"{lang}.jsonl").write_bytes(text.encode("utf-8"))
        if original:
            (orig / f"{lang}.jsonl.gz").write_bytes(gz)
        files.append(
            {
                "language": lang,
                "path": f"{lang}.bin",
                "meta": f"{lang}.meta.json",
                "sha256": hashlib.sha256(tokens).hexdigest(),
                "bytes": len(tokens),
                "n_train": 100 + n_docs,
                "n_val": 7,
                "text": {"path": f"{lang}.jsonl.gz", "sha256": hashlib.sha256(gz).hexdigest(), "docs": n_docs},
            }
        )
    man = tmp_path / "manifest-A.json"
    man.write_text(json.dumps({"files": files}), encoding="utf-8")
    monkeypatch.setattr(cb, "manifest_path", lambda part: man)
    monkeypatch.setattr(cb, "ROOT", tmp_path)
    return inp


def test_gunzip_handles_concatenated_members(tmp_path):
    text = b"line one\nline two\n\xe0\xa4\xb9 three\n" * 1000
    p = tmp_path / "x.jsonl.gz"
    p.write_bytes(_gz_two_members(text))
    zsha, usha, lines = cb.gunzip_digest_and_lines(p)
    assert zsha == hashlib.sha256(p.read_bytes()).hexdigest()
    assert usha == hashlib.sha256(text).hexdigest() and lines == 3000
    assert gzip.decompress(p.read_bytes()) == text  # the standard library agrees


def test_full_pass(tmp_path, monkeypatch):
    inp = _fake(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert cb.main(["--input-dir", str(inp), "--out", str(out), "--parts", "A", "--workers", "2"]) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["verdict"].startswith("PASS: every token file")
    p = s["parts"][0]
    assert p["token_files_ok"] and p["text_ok"] and p["text_fully_compared"]
    en = p["languages"][0]
    assert en["text"]["form"] == "unzipped by Kaggle" and en["text"]["identical_to_original_unzipped"]
    assert "identical to the original" in en["meta"]["note"]
    assert "VERDICT: PASS" in (out / "SUMMARY.txt").read_text(encoding="utf-8")


def test_corrupted_token_file_fails(tmp_path, monkeypatch):
    inp = _fake(tmp_path, monkeypatch)
    b = inp / "frontier-v2-slice2-a" / "slice2-A" / "hi.bin"
    raw = bytearray(b.read_bytes())
    raw[0] ^= 1  # same size, different content
    b.write_bytes(bytes(raw))
    out = tmp_path / "out"
    assert cb.main(["--input-dir", str(inp), "--out", str(out), "--parts", "A"]) == 1
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["verdict"].startswith("FAIL: at least one token file")
    assert [r["bin"]["ok"] for r in s["parts"][0]["languages"]] == [True, False]


def test_changed_text_fails_even_with_the_same_document_count(tmp_path, monkeypatch):
    inp = _fake(tmp_path, monkeypatch)
    t = inp / "frontier-v2-slice2-a" / "slice2-A" / "en.jsonl"
    t.write_text(t.read_text(encoding="utf-8").replace("doc 0", "doc X"), encoding="utf-8")
    out = tmp_path / "out"
    assert cb.main(["--input-dir", str(inp), "--out", str(out), "--parts", "A"]) == 1
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    en = s["parts"][0]["languages"][0]["text"]
    assert en["lines"] == 5 and en["identical_to_original_unzipped"] is False
    assert s["verdict"].startswith("FAIL: token files match")


def test_without_the_original_text_is_never_a_full_pass(tmp_path, monkeypatch):
    inp = _fake(tmp_path, monkeypatch, original=False)
    out = tmp_path / "out"
    assert cb.main(["--input-dir", str(inp), "--out", str(out), "--parts", "A"]) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["verdict"] == "PASS for token files; text documents counted but not compared byte for byte"
    assert s["parts"][0]["original_dir"] is None


def test_backup_that_kept_the_gz_form(tmp_path, monkeypatch):
    inp = _fake(tmp_path, monkeypatch, keep_gz=True)
    out = tmp_path / "out"
    assert cb.main(["--input-dir", str(inp), "--out", str(out), "--parts", "A"]) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["parts"][0]["languages"][0]["text"]["form"] == "gz (as built)"


def test_ambiguous_folders_stop_the_check(tmp_path, monkeypatch):
    inp = _fake(tmp_path, monkeypatch)
    extra = inp / "frontier-v2-slice2-a" / "copy"
    extra.mkdir()
    for f in (inp / "frontier-v2-slice2-a" / "slice2-A").iterdir():
        (extra / f.name).write_bytes(f.read_bytes())
    with pytest.raises(SystemExit, match="expected one backup folder"):
        cb.main(["--input-dir", str(inp), "--out", str(tmp_path / "out"), "--parts", "A"])


def test_real_manifests_have_what_the_check_needs():
    for part in cb.PARTS:
        m = json.loads(cb.manifest_path(part).read_text(encoding="utf-8"))
        assert len(m["files"]) in (6, 7)
        for e in m["files"]:
            assert {"language", "path", "meta", "sha256", "bytes", "n_train", "n_val", "text"} <= set(e)
            assert e["text"]["path"].endswith(".jsonl.gz") and len(e["text"]["sha256"]) == 64


# ------------------------------------------------------------------------------ kernel + runner --
def test_kernel_is_ascii_and_calls_the_check():
    text = KERNEL.read_text(encoding="ascii")
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in ast.parse(text).body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__" and consts["OUT"] == "/kaggle/working/EXP-046/backup-check"
    assert "check_slice2_backup.py" in text and "nvidia-smi" not in text
    flags = set(re.findall(r'"(--[a-z-]+)"', text)) - {"--quiet"}
    assert flags == {"--input-dir", "--out", "--workers"}
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_slice2_backup.py"), "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    ).stdout  # fmt: skip
    for f in flags:
        assert f in helptext


def test_runner_check_block():
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-046" -Check -Relaunch:$Relaunch' in WRAPPER.read_text(
        encoding="ascii"
    )
    text = RUNNER.read_text(encoding="ascii")
    i = text.index("if ($Check) {\n    # the slice-2 backup check")
    block = text[i : text.index("\n}\n", i)]
    assert '$template = "scripts\\kaggle\\exp046_check_kernel.py"' in block
    assert '$kernelSlug = "frontier-exp046-check"' in block and '$outDir = "out\\kaggle\\EXP-046-check"' in block
    assert "$dataFiles = @()" in block and "$maxWaitHours = 3" in block
    # after the EXP-046 probe defaults it replaces, before they are used
    assert text.index('if ($Exp -eq "EXP-046") {\n    # EXP-046 step 1') < i < text.index('$kernelId = "$user/$kernelSlug"')
    assert 'if ($Check -and ($Exp -ne "EXP-046" -or $Build))' in text
    # inputs: the backup datasets and the original build outputs; a CPU session (EXP-046 is in the switch)
    j = text.index('        machine_shape = "NvidiaTeslaT4"\n    }\n')
    meta = text[j : text.index('Write-Ascii "$stage\\kernel-metadata.json"', j)]
    assert '$meta.dataset_sources = @("$user/frontier-v2-slice2-a", "$user/frontier-v2-slice2-b")' in meta
    assert '$meta.kernel_sources = @("$user/frontier-exp046-build-a", "$user/frontier-exp046-build-b")' in meta
    assert '$Exp -eq "EXP-046" -or $Exp -eq "EXP-048") {' in meta and "$meta.enable_gpu = $false" in meta
    # only the small report folder is downloaded; results under EXP-046\backup-check
    d = text.index("} elseif ($Check) {")
    assert "--file-pattern EXP-046/backup-check/.*" in text[d : text.index("} elseif ($Build) {", d)]
    k = text.index('if ($Check) { $resultDir = "$outputDir\\$Exp\\backup-check" }')
    assert text.index('if ($Build) { $resultDir') < k < text.index('$summaryPath = "$resultDir\\summary.json"')
    assert 'if ($Check) { $msgText = "${Exp}: slice-2 backup check' in text
