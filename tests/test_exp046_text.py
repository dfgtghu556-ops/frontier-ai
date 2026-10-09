"""Backup items b and b2 (approved 2026-10-09, "DO ALL"): single-member gzip, the text rewrite/verify
script, its two Kaggle kernels, the runner's -Text wiring, and the builder's one-member join."""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import re
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_slice2_backup as cb  # noqa: E402
import slice2_text_backup as tb  # noqa: E402
from frontier_ai.corpus.gzjoin import (  # noqa: E402
    gzip_members,
    iter_gunzip,
    unzipped_digest,
    write_single_member,
)

RUNNER = ROOT / "scripts" / "run_kaggle_exp038.ps1"
WRAPPER = ROOT / "scripts" / "run_kaggle_exp046_text.ps1"
KERNELS = {m: ROOT / "scripts" / "kaggle" / f"exp046_text_{m}_kernel.py" for m in ("rewrite", "verify")}


def _two_members(text: bytes) -> bytes:
    half = len(text) // 2
    out = io.BytesIO()
    for chunk in (text[:half], text[half:]):
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as g:
            g.write(chunk)
    return out.getvalue()


def _first_member_only(gz: bytes) -> bytes:
    """What Kaggle's dataset unzip kept (EXP-046 backup check, 2026-10-09)."""
    d = zlib.decompressobj(wbits=31)
    return d.decompress(gz)


def test_gzjoin_basics(tmp_path):
    text = ("\u0939\u093f\u0928\u094d\u0926\u0940 line\n" * 5000).encode("utf-8")
    a = tmp_path / "a.gz"
    a.write_bytes(_two_members(text))
    assert gzip_members(a) == 2
    assert b"".join(iter_gunzip(a, block=1000)) == text
    assert unzipped_digest(a) == {
        "sha256": hashlib.sha256(text).hexdigest(),
        "bytes": len(text),
        "lines": 5000,
    }
    out = tmp_path / "one.gz"
    w = write_single_member([a, a], out)
    assert gzip_members(out) == 1 and gzip.decompress(out.read_bytes()) == text + text
    assert w["lines"] == 10000 and w["sha256"] == hashlib.sha256(text + text).hexdigest()
    assert w["gz_bytes"] == out.stat().st_size and not (tmp_path / "one.gz.tmp").exists()
    # the first-member-only reader (Kaggle's unzip) now gets everything
    assert _first_member_only(out.read_bytes()) == text + text
    again = tmp_path / "again.gz"
    write_single_member([a, a], again)
    assert again.read_bytes() == out.read_bytes()  # deterministic: no name, mtime 0
    trunc = tmp_path / "t.gz"
    trunc.write_bytes(out.read_bytes()[:-20])
    with pytest.raises(ValueError):
        gzip_members(trunc)
    empty = tmp_path / "e.gz"
    empty.write_bytes(b"")
    assert gzip_members(empty) == 0


def _fake_originals(tmp_path: Path, monkeypatch) -> Path:
    inp = tmp_path / "input"
    orig = inp / "notebooks" / "someone" / "frontier-exp046-build-a" / "slice2-A"
    orig.mkdir(parents=True)
    files = []
    for lang, n in (("en", 7), ("hi", 4)):
        text = "".join(json.dumps({"id": f"{lang}-{i}", "text": f"doc {i} \u0939"}) + "\n" for i in range(n))
        gz = _two_members(text.encode("utf-8"))
        (orig / f"{lang}.jsonl.gz").write_bytes(gz)
        files.append(
            {
                "language": lang,
                "path": f"{lang}.bin",
                "text": {"path": f"{lang}.jsonl.gz", "sha256": hashlib.sha256(gz).hexdigest(), "docs": n},
            }
        )
    man = tmp_path / "manifest-A.json"
    man.write_text(json.dumps({"files": files}), encoding="utf-8")
    monkeypatch.setattr(cb, "manifest_path", lambda part: man)
    monkeypatch.setattr(cb, "ROOT", tmp_path)
    monkeypatch.setattr(cb, "PARTS", ("A",))
    monkeypatch.setattr(tb, "LANGUAGES", 2)
    return inp


def _rewrite(tmp_path, monkeypatch):
    inp = _fake_originals(tmp_path, monkeypatch)
    text_out = tmp_path / "working" / "slice2-text"
    out = tmp_path / "working" / "EXP-046" / "text-rewrite"
    code = tb.main(["rewrite", "--input-dir", str(inp), "--out", str(out), "--text-out", str(text_out)])
    return inp, text_out, out, code


def test_rewrite_passes_and_files_are_single_member(tmp_path, monkeypatch):
    inp, text_out, out, code = _rewrite(tmp_path, monkeypatch)
    assert code == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["schema"] == "frontier-exp046-text-backup-v1" and s["mode"] == "rewrite" and s["complete"]
    assert s["verdict"].startswith("PASS")
    rows = s["parts"][0]["languages"]
    assert [r["original"]["gzip_members"] for r in rows] == [2, 2]
    for r in rows:
        assert r["new"]["gzip_members"] == 1 and all(r["checks"].values())
        new = text_out / r["path"]
        orig = next(inp.rglob(r["path"]))
        assert gzip.decompress(new.read_bytes()) == gzip.decompress(orig.read_bytes())
        assert _first_member_only(new.read_bytes()) == gzip.decompress(orig.read_bytes())
    assert "VERDICT: PASS" in (out / "SUMMARY.txt").read_text(encoding="utf-8")


def test_rewrite_refuses_an_original_that_differs_from_the_manifest(tmp_path, monkeypatch):
    inp = _fake_originals(tmp_path, monkeypatch)
    f = next(inp.rglob("hi.jsonl.gz"))
    f.write_bytes(_two_members(b'{"id": "x"}\n'))
    out = tmp_path / "out"
    code = tb.main(["rewrite", "--input-dir", str(inp), "--out", str(out), "--text-out", str(tmp_path / "t")])
    assert code == 1
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    hi = s["parts"][0]["languages"][1]
    assert not hi["ok"] and not hi["original"]["sha256_ok"] and "new" not in hi
    assert not (tmp_path / "t" / "hi.jsonl.gz").exists()


def _dataset_from(text_out: Path, inp_root: Path, *, unzip: str) -> Path:
    """Lay the rewrite's files out as the founder's dataset; unzip like Kaggle (full or first member)."""
    ds = inp_root / "datasets" / "someone" / "frontier-v2-slice2-text" / "slice2-text"
    ds.mkdir(parents=True)
    for gz in sorted(text_out.glob("*.jsonl.gz")):
        raw = gz.read_bytes()
        if unzip == "none":
            (ds / gz.name).write_bytes(raw)
        elif unzip == "full":
            (ds / gz.name.removesuffix(".gz")).write_bytes(gzip.decompress(raw))
        else:
            (ds / gz.name.removesuffix(".gz")).write_bytes(_first_member_only(raw))
    return inp_root


@pytest.mark.parametrize("unzip", ["full", "none"])
def test_verify_passes_on_a_complete_dataset(tmp_path, monkeypatch, unzip):
    _, text_out, out, _ = _rewrite(tmp_path, monkeypatch)
    monkeypatch.setattr(tb, "REWRITE_SUMMARY", out / "summary.json")
    inp2 = _dataset_from(text_out, tmp_path / "input2", unzip=unzip)
    vout = tmp_path / "verify"
    assert tb.main(["verify", "--input-dir", str(inp2), "--out", str(vout)]) == 0
    s = json.loads((vout / "summary.json").read_text(encoding="utf-8"))
    assert s["verdict"].startswith("PASS") and len(s["languages"]) == 2


def test_verify_catches_the_old_failure_and_edits(tmp_path, monkeypatch):
    # the original two-member files, unzipped Kaggle-style, must FAIL verification
    inp, text_out, out, _ = _rewrite(tmp_path, monkeypatch)
    monkeypatch.setattr(tb, "REWRITE_SUMMARY", out / "summary.json")
    old = tmp_path / "old"
    old.mkdir()
    for f in inp.rglob("*.jsonl.gz"):
        (old / f.name).write_bytes(f.read_bytes())
    inp2 = _dataset_from(old, tmp_path / "input2", unzip="first")
    vout = tmp_path / "verify"
    assert tb.main(["verify", "--input-dir", str(inp2), "--out", str(vout)]) == 1
    s = json.loads((vout / "summary.json").read_text(encoding="utf-8"))
    assert s["verdict"].startswith("FAIL") and not any(r["ok"] for r in s["languages"])


def test_verify_refuses_a_failed_rewrite(tmp_path, monkeypatch):
    bad = tmp_path / "summary.json"
    bad.write_text(json.dumps({"verdict": "FAIL: x", "parts": []}), encoding="utf-8")
    monkeypatch.setattr(tb, "REWRITE_SUMMARY", bad)
    with pytest.raises(SystemExit):
        tb.main(["verify", "--input-dir", str(tmp_path), "--out", str(tmp_path / "o")])


def test_builder_joins_into_one_member():
    src = (ROOT / "scripts" / "build_slice2.py").read_text(encoding="utf-8")
    assert "from frontier_ai.corpus.gzjoin import write_single_member" in src
    assert "write_single_member(parts, text_path)" in src
    assert "shutil.copyfileobj(fh, out" not in src  # the byte-for-byte (two-member) join is gone


@pytest.mark.parametrize("mode", ["rewrite", "verify"])
def test_kernels(mode):
    text = KERNELS[mode].read_text(encoding="ascii")
    ast.parse(text)
    assert 'COMMIT = "__PINNED_COMMIT__"' in text
    assert f'OUT = Path("/kaggle/working/EXP-046/text-{mode}")' in text
    assert f'"{mode}",' in text and "slice2_text_backup.py" in text
    assert ("/kaggle/working/slice2-text" in text) == (mode == "rewrite")
    for bad in ("KAGGLE_KEY", "kaggle.json", "HF_TOKEN", "secrets"):
        assert bad not in text


def test_runner_text_wiring():
    text = RUNNER.read_text(encoding="ascii")
    assert '[switch]$Check, [ValidateSet("rewrite", "verify")][string]$TextStep)' in text
    assert 'if ($TextStep -and ($Exp -ne "EXP-046" -or $Build -or $Check)) {' in text
    i = text.index("if ($TextStep) {\n    # the complete slice-2 text backup")
    block = text[i : text.index("\n}\n", i)]
    for needle in (
        '$template = "scripts\\kaggle\\exp046_text_${TextStep}_kernel.py"',
        '$kernelSlug = "frontier-exp046-text-$TextStep"',
        '$outDir = "out\\kaggle\\EXP-046-text-$TextStep"',
        "$dataFiles = @()",
    ):
        assert needle in block, needle
    j = text.index('if ($TextStep -eq "rewrite") {')
    rw = text[j : text.index("\n    }\n", j)]
    assert "$meta.dataset_sources = @()" in rw
    assert '$meta.kernel_sources = @("$user/frontier-exp046-build-a", "$user/frontier-exp046-build-b")' in rw
    assert 'if ($TextStep -eq "verify") { $meta.dataset_sources = @("$user/frontier-v2-slice2-text") }' in text
    # never the 5.6 GB of text: only the small report folder is downloaded
    dl = re.findall(r'\} elseif \(\$TextStep\) \{\n(?:\s*#[^\n]*\n)?\s*\$r = Invoke-Logged "([^"]+)"', text)
    assert dl == [
        "$kaggle kernels output $kernelId -p $outputDir --force --file-pattern EXP-046/text-$TextStep/.*"
    ]
    assert 'if ($TextStep) { $resultDir = "$outputDir\\$Exp\\text-$TextStep" }' in text
    assert text.index('if ($Exp -eq "EXP-046") { $resultDir =') < text.index("if ($TextStep) { $resultDir =")
    assert 'if ($TextStep) { $msgText = "${Exp}: slice-2 text backup, $TextStep step' in text


def test_wrapper():
    text = WRAPPER.read_text(encoding="ascii")
    assert "param([switch]$Verify, [switch]$Relaunch)" in text
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-046" -TextStep $step -Relaunch:$Relaunch' in text
    assert 'if ($Verify) { $step = "verify" }' in text and "exit $LASTEXITCODE" in text


def test_runner_tells_the_founder_how_to_make_the_dataset():
    text = RUNNER.read_text(encoding="ascii")
    i = text.index('if ($TextStep -eq "rewrite") {\n    Add-Report ""')
    block = text[i : text.index("\n}\n", i)]
    assert 'StartsWith("PASS")' in block and "Output tab" in block and "New Dataset" in block
    assert "frontier-v2-slice2-text" in block and "run_kaggle_exp046_text.ps1 -Verify" in block
    assert i < text.index('Add-Report "RESULT: COMPLETE - finished')
