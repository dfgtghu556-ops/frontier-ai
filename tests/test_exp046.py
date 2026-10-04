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
