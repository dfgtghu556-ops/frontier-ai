"""EXP-047 code (approved 2026-10-11, "approve EXP-047 code"): the next-model runner on the slices.

CPU smoke of the whole flow on three tiny slices named like the real ones (s1, s2a, s2b): the
learning-rate check, a one-process session, then a two-process (D-050, gloo) session that finishes the
pass and scores slice 1, slice 2 and both pooled. Also: the plan values (family, windows, steps), no
default size, the gate rule against EXP-043's final, the data refusals, and that the EXP-043 runner is
left exactly as it was.
"""

# ruff: noqa: F811  (pytest fixtures named like the imported `langs` fixture)
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import gpu_pretrain as gp  # noqa: E402
import pretrain_exp047 as px  # noqa: E402
from frontier_ai.data import slices as sl  # noqa: E402
from frontier_ai.data.dataset import TokenDataset  # noqa: E402
from test_gpu_lr_arch import TEXT, langs  # noqa: E402, F401  (the 3-language fixture)

SLICE_LANGS = {"s1": ["hi", "en", "ta"], "s2a": ["hi", "en"], "s2b": ["ta"]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_slices(langs: Path, base: Path) -> list[dict]:
    """Copies of the fixture's packed files as three slices, with manifests like EXP-037's."""
    spec = []
    for name, codes in SLICE_LANGS.items():
        d = base / "input" / f"mount-{name}" / "data"
        d.mkdir(parents=True)
        files = []
        for lang in codes:
            for suffix in (".bin", ".meta.json"):
                shutil.copy(langs / f"{lang}{suffix}", d / f"{lang}{suffix}")
            ds = TokenDataset(d / f"{lang}.bin")
            files.append(
                {
                    "language": lang,
                    "path": f"{lang}.bin",
                    "meta": f"{lang}.meta.json",
                    "sha256": _sha(d / f"{lang}.bin"),
                    "bytes": (d / f"{lang}.bin").stat().st_size,
                    "n_train": ds.n_train,
                    "n_val": ds.n_val,
                }
            )
        man = base / f"manifest-{name}.json"
        man.write_text(json.dumps({"files": files}), encoding="utf-8")
        spec.append(
            {
                "name": name,
                "manifest": str(man),
                "manifest_sha256": _sha(man),
                "mount_hint": f"mount-{name}",
                "description": f"test slice {name}",
            }
        )
    return spec


def _run(base: Path, res: Path, chain_in: Path, chain_out: Path, *extra: str) -> int:
    return px.main(
        [
            "--smoke",
            "--skip-tests",
            "--size",
            "18x1152",
            "--cap-hours",
            "200",
            "--data-dir",
            str(base / "input"),
            "--out",
            str(res),
            "--prev-dir",
            str(res),
            "--chain-in",
            str(chain_in),
            "--chain-out",
            str(chain_out / "exp047_chain"),
            "--final-dir",
            str(base / "final"),
            "--scratch",
            str(base / "scratch"),
            *extra,
        ]
    )


def _summary(res: Path, n: int) -> dict:
    return json.loads((res / f"session-{n}" / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run(langs, tmp_path_factory):
    """Session 1 = learning-rate check; 2 = 100 steps on one process; 3 = two processes to the end."""
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("exp047")
    mp.setenv(px.TEST_SLICES_ENV, json.dumps(_make_slices(langs, base)))
    mp.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.setenv("OMP_NUM_THREADS", "1")
    mp.delenv("FRONTIER_TEST_DDP_FAIL", raising=False)
    res = base / "res"
    try:
        assert _run(base, res, base / "nothing", base / "c1", "--part", "lrcheck") == 0
        assert _run(base, res, base / "c1", base / "c2", "--session-steps", "100") == 0
        assert _run(base, res, base / "c2", base / "c3", "--gpus", "2") == 0
    finally:
        mp.undo()
    return base, res


# ------------------------------------------------------------- the plan --
def test_family_matches_the_plan_table():
    import torch

    from frontier_ai.config import ModelConfig
    from frontier_ai.model.gpt import GPT

    plan = {"14x896": 190.1, "16x1024": 272.8, "18x1152": 377.7, "20x1280": 507.5, "22x1408": 664.9}
    assert list(px.FAMILY) == list(plan)
    for size, millions in plan.items():
        m = px.model_for(size)
        assert m["n_embd"] // m["n_layer"] == 64 and m["n_embd"] // m["n_head"] == 64
        mc = ModelConfig(
            vocab_size=32896, dropout=0.0, norm="rmsnorm", ffn="swiglu", tie_embeddings=True, **m
        )
        with torch.device("meta"):
            n = GPT(mc).n_params()
        assert round(n / 1e6, 1) == millions, size
    assert px.model_for("14x896") == gp.MODEL_FULL  # the EXP-043 model is the family's first member


def test_windows_and_steps_follow_from_the_pinned_manifests():
    windows = sum((f["n_train"] - 1) // 512 for s in sl.EXP047_SLICES for f in sl.load_manifest(s)["files"])
    assert windows == px.EXPECTED_WINDOWS == 15_936_053
    assert windows // 32 == px.EXPECTED_STEPS == 498_001


def test_setup_changes_only_the_exp047_parts():
    before = json.dumps(gp.SETUP_FULL, sort_keys=True)
    s = px.make_setup("20x1280", 271, False, sl.EXP047_SLICES)
    assert json.dumps(gp.SETUP_FULL, sort_keys=True) == before  # EXP-043's plan is not touched
    changed = {k for k in set(s) | set(gp.SETUP_FULL) if s.get(k) != gp.SETUP_FULL.get(k)}
    assert changed == {
        "model", "expected_windows", "expected_steps", "micro_options", "total_cap_hours", "reserve_minutes",
        "predicted_bpb", "reference_run", "reference_bpb", "size", "slices", "yardstick", "reference_dir",
    }  # fmt: skip
    assert s["model"]["n_layer"] == 20 and s["predicted_bpb"] == 0.6138 and s["total_cap_hours"] == 271.0
    assert all(accum % 2 == 0 for _, accum in s["micro_options"])  # splits over two GPUs
    assert all(micro * accum == s["windows_per_step"] == 32 for micro, accum in s["micro_options"])
    assert [x["manifest_sha256"] for x in s["slices"]] == [x.manifest_sha256 for x in sl.EXP047_SLICES]
    for key in ("lr_grid", "lr_extension", "lr_check_steps", "warmup", "noise", "mem_limit_gb", "seed"):
        assert s[key] == gp.SETUP_FULL[key], key  # learning rate check and rules as EXP-043
    with pytest.raises(ValueError):
        px.make_setup("24x1536", 100, False, sl.EXP047_SLICES)
    with pytest.raises(ValueError):
        px.make_setup("18x1152", 0, False, sl.EXP047_SLICES)


@pytest.mark.parametrize("missing", ["--size", "--cap-hours"])
def test_no_default_size_or_cap(missing, capsys):
    argv = ["--size", "16x1024", "--cap-hours", "150", "--data-dir", "/nowhere"]
    i = argv.index(missing)
    assert px.main(argv[:i] + argv[i + 2 :]) == 2
    assert "no default" in capsys.readouterr().out


def test_test_slices_need_smoke(monkeypatch):
    monkeypatch.setenv(px.TEST_SLICES_ENV, "[]")
    with pytest.raises(sl.SliceError):
        px.slice_defs(smoke=False)
    monkeypatch.delenv(px.TEST_SLICES_ENV)
    assert px.slice_defs(smoke=False) == sl.EXP047_SLICES


def test_the_exp043_runner_is_restored_after_each_call():
    original = {
        n: getattr(gp, n) for n in (*px.HOOKS, "EXP_ID", "WORKER_SCRIPT", "SETUP_FULL", "SETUP_SMOKE")
    }
    with px.installed(setup_full={"x": 1}):
        assert gp.EXP_ID == "EXP-047" and gp.SETUP_FULL == {"x": 1}
        assert gp.WORKER_SCRIPT.name == "pretrain_exp047.py" and gp.score_all is px.score_all
    assert {n: getattr(gp, n) for n in original} == original
    assert gp.EXP_ID == "EXP-043" and gp.WORKER_SCRIPT.name == "gpu_pretrain.py"


# ------------------------------------------------------------- the gate --
def _ref(tmp_path: Path, bpb: dict) -> dict:
    d = tmp_path / "EXP-043" / "session-9"
    d.mkdir(parents=True)
    scores = {"bpb": bpb, "bpb_mean": sum(bpb.values()) / len(bpb)}
    (d / "summary.json").write_text(
        json.dumps({"session": 9, "complete": True, "smoke": False, "final": {"scores": scores}}),
        encoding="utf-8",
    )
    s = px.make_setup("18x1152", 200, False, sl.EXP047_SLICES)
    s["reference_dir"] = str(tmp_path / "EXP-043")
    return s


def _scores(bpb: dict) -> dict:
    return {"bpb": bpb, "bpb_mean": sum(bpb.values()) / len(bpb), "slice2": None}


def test_gate_pass_and_the_two_ways_to_fail(tmp_path):
    setup = _ref(tmp_path, {"hi": 0.70, "en": 0.80, "ta": 0.60})
    ok = px.final_checks(_scores({"hi": 0.67, "en": 0.77, "ta": 0.59}), setup)  # mean 0.677 < 0.7 - 0.0174
    assert ok["check1_pass"] is True and ok["check1_gate"].startswith("PASS")
    small = px.final_checks(_scores({"hi": 0.69, "en": 0.79, "ta": 0.59}), setup)  # better, not by the noise
    assert small["check1_pass"] is False
    worse = px.final_checks(_scores({"hi": 0.60, "en": 0.70, "ta": 0.62}), setup)  # ta worse by 0.02 > noise
    assert worse["check1_pass"] is False and worse["check1_worse_languages"] == {"ta": 0.02}
    assert worse["per_language_reference"] == {"hi": 0.70, "en": 0.80, "ta": 0.60}


def test_gate_is_not_decided_without_exp043_final(tmp_path):
    s = px.make_setup("18x1152", 200, False, sl.EXP047_SLICES)
    s["reference_dir"] = str(tmp_path / "missing")
    c = px.final_checks(_scores({"hi": 0.6}), s)
    assert c["check1_pass"] is None and c["check1_gate"].startswith("NOT DECIDED")
    other = _ref(tmp_path, {"hi": 0.7, "en": 0.8})
    assert px.final_checks(_scores({"hi": 0.6}), other)["check1_pass"] is None


# ------------------------------------------------------------ the smoke --
def test_three_sessions_run_the_pass_over_all_slices(run):
    base, res = run
    s1, s2, s3 = (_summary(res, n) for n in (1, 2, 3))
    for s in (s1, s2, s3):
        assert s["exp_id"] == "EXP-047" and s["setup"]["size"] == "18x1152" and not s.get("stopped")
    p0 = s1["part0"]
    assert p0["files_ok"] == p0["files_expected"] == 6 and p0["sha256_checked"] is True
    assert [x["name"] for x in p0["slices"]] == ["s1", "s2a", "s2b"]
    assert s1["languages"] == ["s1/hi", "s1/en", "s1/ta", "s2a/hi", "s2a/en", "s2b/ta"]
    assert s1["lr_choice"]["decided"] and s2["main"]["end_step"] == 100
    # session 2 ends with slice-1 validation only, per language
    se = s2["main"]["session_eval"]
    assert sorted(se["bpb"]) == ["en", "hi", "ta"] and sorted(se["nats"]) == ["s1/en", "s1/hi", "s1/ta"]
    # session 3: two processes, to the end of the one pass over all six files
    m = s3["main"]
    assert m["gpus"] == 2 and m["end_step"] == s3["order"]["steps_per_pass"] and s3["complete"]
    f = s3["final"]["scores"]
    assert sorted(f["nats"]) == sorted(s1["languages"])
    assert f["bpb"] == m["session_eval"]["bpb"]  # the yardstick is slice 1
    assert sorted(f["slice2"]["bpb"]) == ["en", "hi", "ta"] and sorted(f["pooled_by_language"]) == [
        "en",
        "hi",
        "ta",
    ]
    for lang in ("hi", "en", "ta"):  # pooled lies between the two slices' values
        lo, hi = sorted((f["bpb"][lang], f["slice2"]["bpb"][lang]))
        assert lo - 1e-9 <= f["pooled_by_language"][lang] <= hi + 1e-9
    checks = s3["final"]["checks"]
    assert checks["check1_pass"] is None and checks["check1_gate"].startswith("NOT DECIDED")
    assert checks["check3_slice2"].startswith("reported, not gated")
    rows = [
        json.loads(x) for x in (res / "session-3" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {r["language"] for r in rows} == {"hi", "en", "ta"}  # slice-1 prompts, by language
    info = json.loads((base / "final" / "model_final.json").read_text(encoding="utf-8"))
    assert info["step"] == m["end_step"]
    text = (res / "session-3" / "SUMMARY.txt").read_text(encoding="utf-8")
    assert text.startswith("EXP-047 - train the next base model (18x1152, v2-slice1 + v2-slice2)")
    assert "slice-2 validation (reported, not gated)" in text and "C3-s3" not in text


def test_another_size_cannot_continue_the_run(run, tmp_path, monkeypatch, langs):
    base, res = run
    res2 = tmp_path / "res"
    shutil.copytree(res, res2)
    monkeypatch.setenv(px.TEST_SLICES_ENV, json.dumps(_make_slices(langs, tmp_path)))
    argv = ["--smoke", "--skip-tests", "--size", "16x1024", "--cap-hours", "200", "--data-dir", str(tmp_path)]
    assert px.main([*argv, "--out", str(res2), "--prev-dir", str(res2)]) == 1
    assert not (res2 / "session-4").exists()


def test_changed_data_is_refused_before_training(langs, tmp_path, monkeypatch):
    spec = _make_slices(langs, tmp_path)
    monkeypatch.setenv(px.TEST_SLICES_ENV, json.dumps(spec))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    target = tmp_path / "input" / "mount-s2b" / "data" / "ta.bin"
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 1  # same size, other content: only the full SHA-256 sees it
    target.write_bytes(bytes(raw))
    res = tmp_path / "res"
    assert _run(tmp_path, res, tmp_path / "none", tmp_path / "c1", "--part", "lrcheck") == 1
    s = _summary(res, 1)
    assert "pinned EXP-047 slice manifests" in s["stopped"] and "s2b/ta: sha256 differs" in s["stopped"]
    assert not s.get("lr_runs")
    # an edited manifest is refused too (its pinned SHA-256 no longer matches)
    spec[0]["manifest_sha256"] = "0" * 64
    monkeypatch.setenv(px.TEST_SLICES_ENV, json.dumps(spec))
    res2 = tmp_path / "res2"
    assert _run(tmp_path, res2, tmp_path / "none", tmp_path / "c1", "--part", "lrcheck") == 1
    assert "recorded value" in _summary(res2, 1)["stopped"]


def test_first_kaggle_session_is_probe_only_then_the_check_continues(langs, tmp_path):
    """The EXP-047 kernel's session 1 (``--max-runs 0``, EXP-047 note 2026-10-13): Part 0 and the memory
    probe, no learning-rate run, nothing stopped; session 2 (``--part auto``, as on Kaggle) reuses the
    micro-batch and starts the learning-rate check."""
    mp = pytest.MonkeyPatch()
    mp.setenv(px.TEST_SLICES_ENV, json.dumps(_make_slices(langs, tmp_path)))
    mp.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.setenv("OMP_NUM_THREADS", "1")
    res = tmp_path / "res"
    try:
        assert (
            _run(tmp_path, res, tmp_path / "nothing", tmp_path / "c1", "--part", "auto", "--max-runs", "0")
            == 0
        )
        s1 = _summary(res, 1)
        assert s1["micro"] and s1["micro_note"] and not s1.get("stopped") and not s1.get("stop_rule")
        assert s1["lr_runs"] == [] and not s1["lr_choice"]["decided"] and not s1.get("complete")
        assert not s1.get("main")
        assert _run(tmp_path, res, tmp_path / "c1", tmp_path / "c2", "--part", "auto", "--max-runs", "1") == 0
        s2 = _summary(res, 2)
        assert s2["micro"] == s1["micro"] and s2["micro_note"] == "chosen in an earlier session"
        assert not s2.get("stopped") and len(s2["lr_runs"]) == 1 and s2["lr_runs"][0]["session"] == 2
    finally:
        mp.undo()
