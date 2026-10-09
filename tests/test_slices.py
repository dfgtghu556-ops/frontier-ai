"""EXP-047 step 1 (approved 2026-10-09, "approve D"): slice-aware loading with pinned manifests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from frontier_ai.data import slices as sl
from frontier_ai.data.dataset import write_tokens
from frontier_ai.data.multi import MultiTokenDataset, OnePassDataset

ROOT = Path(__file__).resolve().parents[1]


def _slice(tmp_path: Path, name: str, langs: dict[str, int], seed: int) -> tuple[sl.Slice, Path]:
    """Token files + a manifest like EXP-037's; returns a Slice pinned to that manifest."""
    d = tmp_path / "input" / f"mount-{name}" / "data"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    files = []
    for lang, n in langs.items():
        ids = rng.integers(0, 600, size=n)
        write_tokens(
            d / f"{lang}.bin", ids, 600, "bpe", val_frac=0.2, token_bytes=[2] * n, token_chars=[1] * n
        )
        meta = json.loads((d / f"{lang}.meta.json").read_text(encoding="utf-8"))
        raw = (d / f"{lang}.bin").read_bytes()
        files.append(
            {
                "language": lang,
                "path": f"{lang}.bin",
                "meta": f"{lang}.meta.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "n_train": meta["n_train"],
                "n_val": meta["n_val"],
            }
        )
    man = tmp_path / f"manifest-{name}.json"
    man.write_text(json.dumps({"files": files}), encoding="utf-8")
    pinned = hashlib.sha256(man.read_bytes()).hexdigest()
    return sl.Slice(name, man.name, pinned, f"mount-{name}", f"test slice {name}"), d


def test_pinned_hashes_match_the_committed_manifests():
    for s in sl.EXP047_SLICES:
        man = sl.load_manifest(s)  # raises if the committed manifest was changed
        assert len(man["files"]) == (13 if s.name == "s1" else {"s2a": 6, "s2b": 7}[s.name])
    s1 = sl.load_manifest(sl.V2_SLICE1)
    assert sum(f["n_train"] for f in s1["files"]) == 2_783_830_088  # EXP-037
    s2 = [sl.load_manifest(s) for s in (sl.V2_SLICE2_A, sl.V2_SLICE2_B)]
    s2_train = sum(f["n_train"] for m in s2 for f in m["files"])
    assert s2_train == 5_375_435_604  # D-051
    assert sum(f["n_train"] for f in s1["files"]) + s2_train == 8_159_265_692  # EXP-047 section 1
    langs1 = {f["language"] for f in s1["files"]}
    assert langs1 == {f["language"] for m in s2 for f in m["files"]}  # the same 13 languages


def test_a_changed_manifest_is_refused(tmp_path):
    s, _ = _slice(tmp_path, "s1", {"hi": 3000}, 0)
    (tmp_path / s.manifest).write_text((tmp_path / s.manifest).read_text() + " ", encoding="utf-8")
    with pytest.raises(sl.SliceError, match="SHA-256"):
        sl.load_manifest(s, tmp_path)


def test_open_two_slices_with_the_same_file_names(tmp_path):
    s1, d1 = _slice(tmp_path, "s1", {"hi": 4000, "en": 6000}, 1)
    s2, d2 = _slice(tmp_path, "s2a", {"en": 5000, "hi": 3000}, 2)
    with pytest.raises(ValueError, match="duplicate file name"):
        MultiTokenDataset([d1 / "hi.bin", d2 / "hi.bin"])  # why step 1 is needed
    ds, rec = sl.open_slices([(s1, d1), (s2, d2)], full_sha256=True, root=tmp_path)
    assert ds.names == ["s1/hi", "s1/en", "s2a/en", "s2a/hi"]  # slice order, then manifest order
    assert ds.n_train == sum(r["train_tokens"] for r in rec["slices"]) == rec["train_tokens"]
    assert [r["files"] for r in rec["slices"]] == [2, 2] and rec["slices"][0]["full_sha256_checked"]
    w = ds.weights("train")
    assert np.isclose(w.sum(), 1) and np.allclose(w, [ds.parts[n].n_train / ds.n_train for n in ds.names])
    # the one-pass order works over the combined files, every window once
    op = OnePassDataset(ds, block_size=16, seed=3)
    assert op.n_windows == sum((ds.parts[n].n_train - 1) // 16 for n in ds.names)
    assert sorted(op.order.tolist()) == list(range(op.n_windows))
    assert {op.window(k)[0] for k in range(op.n_windows)} == set(ds.names)


def test_default_names_are_unchanged_for_exp043(tmp_path):
    _, d = _slice(tmp_path, "s1", {"hi": 3000, "en": 3000}, 4)
    ds = MultiTokenDataset([d / "hi.bin", d / "en.bin"])
    assert ds.names == ["hi", "en"]  # EXP-040..043 keep their names and one-pass fingerprint
    with pytest.raises(ValueError, match="names for"):
        MultiTokenDataset([d / "hi.bin"], names=["a", "b"])


def test_file_checks_catch_wrong_files(tmp_path):
    s, d = _slice(tmp_path, "s1", {"hi": 3000, "en": 3000}, 5)
    man = sl.load_manifest(s, tmp_path)
    assert all(r["bytes_ok"] and r["meta_ok"] for r in sl.check_files(s, man, d))
    raw = bytearray((d / "en.bin").read_bytes())
    raw[0] ^= 1
    (d / "en.bin").write_bytes(bytes(raw))  # same size, different content
    sl.check_files(s, man, d)  # size and meta cannot see it ...
    with pytest.raises(sl.SliceError, match="s1/en: sha256 differs"):
        sl.check_files(s, man, d, full_sha256=True)  # ... the full hash can
    (d / "hi.bin").write_bytes((d / "hi.bin").read_bytes()[:-2])  # truncated
    with pytest.raises(sl.SliceError, match="s1/hi: bytes differs"):
        sl.check_files(s, man, d)
    (d / "hi.meta.json").unlink()
    with pytest.raises(sl.SliceError, match="missing"):
        sl.check_files(s, man, d)


def test_locate_uses_the_mount_hint_only_when_needed(tmp_path):
    s, d = _slice(tmp_path, "s2a", {"hi": 3000}, 6)
    man = sl.load_manifest(s, tmp_path)
    assert sl.locate(s, man, tmp_path / "input") == d
    other = tmp_path / "input" / "some-other-copy"
    other.mkdir()
    for f in ("hi.bin", "hi.meta.json"):
        (other / f).write_bytes((d / f).read_bytes())
    assert sl.locate(s, man, tmp_path / "input") == d  # two holders: the hint decides
    with pytest.raises(sl.SliceError):
        sl.locate(replace(s, mount_hint="nowhere"), man, tmp_path / "input")


def test_validation_per_language_pools_both_slices(tmp_path):
    s1, d1 = _slice(tmp_path, "s1", {"hi": 4000, "en": 6000}, 7)
    s2, d2 = _slice(tmp_path, "s2a", {"hi": 3000}, 8)
    ds, _ = sl.open_slices([(s1, d1), (s2, d2)], root=tmp_path)
    assert sl.by_language(ds.names) == {"hi": ["s1/hi", "s2a/hi"], "en": ["s1/en"]}
    nats = {"s1/hi": 2.0, "s1/en": 1.0, "s2a/hi": 4.0}
    out = sl.combine_bpb(ds, nats)
    # every token is 2 bytes here, so bpb = nats / ln 2 / 2
    assert out["per_part"]["s1/hi"] == pytest.approx(2.0 / math.log(2) / 2)
    h1, h2 = ds.parts["s1/hi"].n_val, ds.parts["s2a/hi"].n_val
    pooled = (2.0 * h1 + 4.0 * h2) / (h1 + h2) / math.log(2) / 2
    assert out["per_language"]["hi"] == pytest.approx(pooled)
    assert out["per_language"]["en"] == pytest.approx(out["per_part"]["s1/en"])
    with pytest.raises(sl.SliceError):
        sl.by_language(["hi"])


def test_a_slice_given_twice_is_refused(tmp_path):
    s, d = _slice(tmp_path, "s1", {"hi": 3000}, 9)
    with pytest.raises(sl.SliceError, match="once"):
        sl.open_slices([(s, d), (s, d)], root=tmp_path)
