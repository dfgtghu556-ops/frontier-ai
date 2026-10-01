"""EXP-040: multi-file dataset, the learning-rate/architecture runner (CPU smoke) and its rules."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gpu_lr_arch.py"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from frontier_ai.data.dataset import TokenDataset, write_tokens  # noqa: E402
from frontier_ai.data.multi import MultiTokenDataset, full_split_loss  # noqa: E402

TEXT = {
    "hi": "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं। बच्चे सुबह स्कूल जाते हैं। ",
    "en": "India is a large country. Many languages are spoken here. Children go to school. ",
    "ta": "இந்தியா ஒரு பெரிய நாடு. இங்கு பல மொழிகள் பேசப்படுகின்றன. குழந்தைகள் பள்ளிக்குச் செல்கின்றனர். ",
}
DOCS = {"hi": 60, "en": 120, "ta": 40}  # unequal sizes, so natural sampling is visible


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def langs(tmp_path_factory):
    """Three tiny language files packed like EXP-037, plus a manifest listing them."""
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    tok = load_frontier_tokenizer_v2()
    eot = tok.special_token_ids["<|endoftext|>"]
    d = tmp_path_factory.mktemp("langs")
    files = []
    for lang, text in TEXT.items():
        ids: list[int] = []
        nbytes: list[int] = []
        nchars: list[int] = []
        for i in range(DOCS[lang]):
            doc = text * (1 + i % 3)
            doc_ids = tok.encode_ordinary(doc)
            nbytes += [len(doc.encode("utf-8"))] + [0] * len(doc_ids)
            nchars += [len(doc)] + [0] * len(doc_ids)
            ids += doc_ids + [eot]
        write_tokens(
            d / f"{lang}.bin", ids, 32896, "bpe", val_frac=0.1, token_bytes=nbytes, token_chars=nchars
        )
        ds = TokenDataset(d / f"{lang}.bin")
        files.append(
            {
                "language": lang,
                "path": f"{lang}.bin",
                "meta": f"{lang}.meta.json",
                "sha256": _sha(d / f"{lang}.bin"),
                "n_train": ds.n_train,
                "n_val": ds.n_val,
            }
        )
    (d / "manifest.json").write_text(json.dumps({"packed_id": "test", "files": files}), encoding="utf-8")
    return d


def _multi(d: Path) -> MultiTokenDataset:
    return MultiTokenDataset([d / f"{x}.bin" for x in TEXT])


def test_sums_and_weights_follow_the_files(langs):
    m = _multi(langs)
    parts = [TokenDataset(langs / f"{x}.bin") for x in TEXT]
    assert m.names == list(TEXT) and m.vocab_size == 32896
    assert m.n_train == sum(p.n_train for p in parts) and m.n_val == sum(p.n_val for p in parts)
    for split, attr in (("train", "n_train"), ("val", "n_val")):
        w = m.weights(split)
        assert math.isclose(w.sum(), 1.0)
        assert np.allclose(w, [getattr(p, attr) / getattr(m, attr) for p in parts])
    nb = sum(p.split_lengths("val")[0] for p in parts)
    assert m.split_lengths("val")[0] == nb
    assert math.isclose(m.tokens_per_byte("val"), m.n_val / nb)
    assert m.batches_per_epoch(4, 16) == (m.n_train - 17) // 64


def test_batches_are_deterministic_and_sampled_in_proportion(langs):
    m = _multi(langs)
    cpu = torch.device("cpu")
    a = m.get_batch("train", 8, 16, cpu, torch.Generator().manual_seed(3))
    b = m.get_batch("train", 8, 16, cpu, torch.Generator().manual_seed(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert torch.equal(a[0][:, 1:], a[1][:, :-1])  # targets are the inputs shifted by one
    # where each row came from: sample many rows and compare with the token shares
    g = torch.Generator().manual_seed(0)
    counts = dict.fromkeys(TEXT, 0)
    first = {x: set(m.parts[x].split("train")[:].tolist()) for x in TEXT}
    for _ in range(150):
        x, _y = m.get_batch("train", 16, 8, cpu, g)
        for row in x.tolist():
            owners = [k for k in TEXT if set(row) <= first[k]]
            if len(owners) == 1:  # script-specific rows identify the language
                counts[owners[0]] += 1
    total = sum(counts.values())
    assert total > 1500
    for k, w in zip(TEXT, m.weights("train")):
        assert abs(counts[k] / total - w) < 0.06, (k, counts[k] / total, w)


def test_mismatched_vocab_is_refused(langs, tmp_path):
    write_tokens(tmp_path / "x.bin", list(range(200)), 300, "bpe", val_frac=0.1)
    with pytest.raises(ValueError, match="one vocabulary"):
        MultiTokenDataset([langs / "hi.bin", tmp_path / "x.bin"])


def test_full_split_loss_scores_every_whole_window():
    class Uniform(torch.nn.Module):
        def forward(self, x, targets=None):
            logits = torch.zeros(*x.shape, 10)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 10), targets.view(-1))
            return type("O", (), {"loss": loss})()

    class FakeDs:
        def split(self, name):
            return np.arange(101, dtype=np.uint16) % 10

    loss, n = full_split_loss(Uniform(), FakeDs(), 8, 5, torch.device("cpu"))
    assert math.isclose(loss, math.log(10), rel_tol=1e-6)
    assert n == (100 // 8) * 8


def test_rules_follow_the_preregistration():
    import gpu_lr_arch as g

    def run(arch, lr, seed, bpb, grid="A", **kw):
        return {
            "name": g.run_name(arch, lr, seed),
            "arch": arch,
            "lr": lr,
            "seed": seed,
            "grid": grid,
            "status": "done",
            "failed": False,
            "bpb_mean": bpb,
            **kw,
        }

    runs = [run("baseline", lr, 1, 1.0 + abs(lr - 1e-3)) for lr in g.LRS]
    runs += [run("rope_gqa2", lr, 1, 0.95 + abs(lr - 1e-3)) for lr in g.LRS]
    runs += [
        run(a, lr, 2, r["bpb_mean"] + 0.001, "B")
        for lr in (1e-3, 2e-3)
        for a in g.ARCHS
        for r in runs
        if r["name"] == g.run_name(a, lr, 1)
    ]
    rules = g.evaluate_rules(runs, candidate_ok=True)
    assert rules["rule1_best_lr"]["baseline"] == "0.001"
    assert math.isclose(rules["rule2_noise"], 0.001)
    assert rules["rule3_adopt"] == "YES" and all(
        v.startswith("BETTER") for v in rules["rule3_per_lr"].values()
    )
    # best at the edge of the range is flagged
    edge = [
        dict(r, bpb_mean=2.0 - r["lr"]) if r["arch"] == "baseline" and r["seed"] == 1 else r for r in runs
    ]
    assert "EDGE" in g.evaluate_rules(edge, True)["rule1_best_lr"]["baseline"]
    # a difference inside the seed noise is not a win
    close = [dict(r, bpb_mean=r["bpb_mean"] + 0.0495) if r["arch"] == "rope_gqa2" else r for r in runs]
    assert g.evaluate_rules(close, True)["rule3_adopt"].startswith("NO")
    # missing grid-B runs or a failed float64 check: no decision, baseline stays
    assert g.evaluate_rules(runs[:8], True)["rule3_adopt"].startswith("NOT DECIDED")
    assert g.evaluate_rules(runs, False)["rule3_adopt"].startswith("NOT TESTED")
    failed = [dict(r, failed=True, status="failed: x") if r["name"] == runs[0]["name"] else r for r in runs]
    assert g.evaluate_rules(failed, True)["rule4_failed"] == [runs[0]["name"]]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True, timeout=1200
    )


def test_smoke_run_writes_every_run_and_the_rules(langs, tmp_path):
    out = tmp_path / "EXP-040"
    res = _run(
        [
            "--smoke",
            "--skip-tests",
            "--data-dir",
            str(langs),
            "--manifest",
            str(langs / "manifest.json"),
            "--out",
            str(out),
            "--scratch",
            str(tmp_path / "scratch"),
        ]
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["schema"] == "frontier-lr-arch-v1" and s["smoke"] is True and s["complete"] is True
    assert s["part0"]["files_ok"] == 3 and s["part0"]["fp64_check"]["pass"] is None  # CPU vs CPU: n/a
    assert s["part0"]["fp64_check"]["max_abs_diff"] == 0.0
    runs = s["runs"]
    assert len(runs) == 12 and all(r["status"] == "done" for r in runs)
    assert [r["grid"] for r in runs].count("B") == 4 and len(s["grid_b_lrs"]) == 2
    for r in runs:
        assert set(r["bpb"]) == set(TEXT) and all(math.isfinite(v) and v > 0 for v in r["bpb"].values())
        assert math.isclose(r["bpb_mean"], sum(r["bpb"].values()) / 3)
        assert min(r["bpb"].values()) <= r["bpb_token_weighted"] <= max(r["bpb"].values())
        assert r["tokens"] == 12 * 4 * 32 and r["skipped_steps"] == 0 and len(r["val_curve"]) == 2
    params = {r["arch"]: r["n_params"] for r in runs}
    assert params["rope_gqa2"] < params["baseline"]  # no position table, fewer key/value heads
    assert set(s["rules"]) >= {"rule1_best_lr", "rule2_noise", "rule3_adopt", "rule4_failed"}
    text = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "SMOKE TEST, not a result" in text and "Pre-registered rules" in text
    assert sorted(x.name for x in out.iterdir()) == ["SUMMARY.txt", "summary.json"]
    assert not (tmp_path / "scratch").exists() or not any((tmp_path / "scratch").iterdir())


def test_stops_when_a_language_file_is_missing(langs, tmp_path):
    m = json.loads((langs / "manifest.json").read_text(encoding="utf-8"))
    m["files"].append(dict(m["files"][0], language="xx", path="xx.bin", meta="xx.meta.json"))
    (tmp_path / "m.json").write_text(json.dumps(m), encoding="utf-8")
    out = tmp_path / "out"
    res = _run(
        [
            "--smoke",
            "--skip-tests",
            "--data-dir",
            str(langs),
            "--manifest",
            str(tmp_path / "m.json"),
            "--out",
            str(out),
            "--scratch",
            str(tmp_path / "scratch"),
        ]
    )
    assert res.returncode == 1
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["complete"] is False and "xx.bin: missing" in s["stopped"] and s["runs"] == []


def test_full_plan_is_the_preregistered_one():
    import gpu_lr_arch as g

    p, shape = g.PLAN_FULL, g.SHAPE_FULL
    assert p["steps"] * p["batch"] * shape["block_size"] == 100_007_936
    assert (p["warmup"], p["min_lr_ratio"], p["check_steps"]) == (200, 0.1, 50)
    assert g.LRS == [5e-4, 1e-3, 2e-3, 4e-3] and g.TOL_FP64 == 1e-8
    from frontier_ai.model.gpt import GPT

    counts = {}
    for arch, kw in g.ARCHS.items():
        cfg = g.gb.make_cfg({**shape, **kw}, Path("x.bin"), Path("o"))
        counts[arch] = GPT(cfg.model).n_params()
    assert counts == {"baseline": 31_709_568, "rope_gqa2": 29_940_096}
