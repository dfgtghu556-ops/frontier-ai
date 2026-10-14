"""The private Hugging Face demo bundle (scripts/hf_demo.py, apps/hf-demo/app.py; D-053)."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("hf_demo_script", ROOT / "scripts" / "hf_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hd():
    return _load_script()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, hd):
    """A tiny float32 weights file in the EXP-043 format, with the real tokenizer's vocabulary."""
    from frontier_ai.config import ModelConfig
    from frontier_ai.model.gpt import GPT

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32896, n_layer=1, n_head=2, n_embd=16, block_size=64, dropout=0.0, bias=False,
        norm="rmsnorm", ffn="swiglu", ffn_mult=4.0, pos="rope", tie_embeddings=True, n_kv_head=1,
    )  # fmt: skip
    path = tmp_path_factory.mktemp("w") / "model_final.pt"
    torch.save({"model_config": asdict(cfg), "state_dict": GPT(cfg).state_dict()}, path)
    recorded = {"sha256": hd.sha256_file(path), "step": 7, "model_config": {"n_layer": 1, "n_embd": 16}}
    return path, recorded


def test_bundle_is_complete_and_works_like_the_space(hd, tiny, tmp_path):
    path, recorded = tiny
    out = tmp_path / "bundle"
    info = hd.build_bundle(path, out, recorded, smoke_tokens=4)
    for name in ("app.py", "requirements.txt", "README.md", "model_fp16.pt", "MODEL_INFO.json"):
        assert (out / name).is_file(), name
    assert (out / "tokenizers" / "frontier-tokenizer-v2" / "FREEZE.json").is_file()
    assert (out / "frontier_ai" / "engine" / "weights.py").is_file()
    assert not list(out.rglob("__pycache__")) and not [
        p for p in (out / "frontier_ai").rglob("*") if p.suffix == ".pyc"
    ]
    assert info["source"] == {
        "file": "model_final.pt",
        "sha256": recorded["sha256"],
        "step": 7,
        "experiment": "EXP-043",
    }
    assert info["fp16"]["sha256"] == hd.sha256_file(out / "model_fp16.pt")
    assert isinstance(info["smoke_test"]["text"], str)
    assert json.loads((out / "MODEL_INFO.json").read_text(encoding="utf-8"))["n_params"] == info["n_params"]
    # the Space's own code path: load the fp16 copy, continue a text, same seed = same text
    app = hd.load_bundle_app(out)
    model, tok, minfo = app.load(out / "model_fp16.pt", out / "tokenizers" / "frontier-tokenizer-v2")
    assert minfo["source"]["sha256"] == recorded["sha256"]
    a = app.continue_text(model, tok, "The monsoon season", max_new_tokens=5, seed=3)
    assert a == app.continue_text(model, tok, "The monsoon season", max_new_tokens=5, seed=3)
    assert app.continue_text(model, tok, "", max_new_tokens=5) == ""
    long = "भारत " * 200  # longer than the context: the start is dropped, room is kept for new tokens
    assert isinstance(app.continue_text(model, tok, long, max_new_tokens=10), str)


def test_wrong_weights_are_refused(hd, tiny, tmp_path):
    path, recorded = tiny
    with pytest.raises(SystemExit, match="does not match"):
        hd.build_bundle(path, tmp_path / "b", {**recorded, "sha256": "0" * 64})
    assert not (tmp_path / "b").exists()
    with pytest.raises(SystemExit, match="differs from the recorded"):
        hd.build_bundle(path, tmp_path / "b", {**recorded, "model_config": {"n_layer": 14}})


class _FakeApi:
    def __init__(self, existing):
        self.existing, self.calls = existing, []

    def whoami(self):
        return {"name": "someone"}

    def repo_info(self, repo_id, repo_type):
        if self.existing is None:
            raise RuntimeError("404")
        return self.existing

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload", kw["repo_id"], kw["repo_type"]))


def test_upload_goes_only_to_a_private_space(hd, tmp_path):
    api = _FakeApi(existing=None)
    url = hd.upload(tmp_path, "frontier-exp043-preview", api=api)
    assert url == "https://huggingface.co/spaces/someone/frontier-exp043-preview"
    assert api.calls[0] == (
        "create",
        "someone/frontier-exp043-preview",
        {"repo_type": "space", "space_sdk": "gradio", "private": True},
    )
    assert api.calls[1] == ("upload", "someone/frontier-exp043-preview", "space")
    api = _FakeApi(existing=SimpleNamespace(private=True))  # an earlier private upload: updated
    hd.upload(tmp_path, "frontier-exp043-preview", api=api)
    assert [c[0] for c in api.calls] == ["upload"]
    api = _FakeApi(existing=SimpleNamespace(private=False))  # public: refused, nothing uploaded
    with pytest.raises(SystemExit, match="PUBLIC"):
        hd.upload(tmp_path, "frontier-exp043-preview", api=api)
    assert api.calls == []


def test_report_on_failure(hd, tmp_path):
    report = tmp_path / "REPORT.txt"
    code = hd.main(["--weights", str(tmp_path / "missing.pt"), "--build-only", "--report", str(report)])
    assert code == 1 and "RESULT: FAILED" in report.read_text(encoding="utf-8")


def test_recorded_model_is_exp043_final(hd):
    rec = hd.recorded_model_final()
    assert rec["file"] == "model_final.pt" and len(rec["sha256"]) == 64 and rec["n_params"] == 190_064_000


def test_pc_runner_script_is_safe_and_ascii() -> None:
    ps1 = (Path(__file__).resolve().parents[1] / "scripts" / "run_hf_demo.ps1").read_bytes()
    text = ps1.decode("ascii")  # Windows PowerShell 5.1 misreads non-ASCII BOM-less scripts
    assert "login(add_to_git_credential=False)" in text  # the token is typed on the PC, never stored in git
    assert "hf_" not in text.replace("hf_demo", "")  # no token in the file
    assert '--file-pattern "exp043_final/.*"' in text  # never download the whole kernel output
    assert "frontier-exp043-b" in text and "session-6\\summary.json" in text
    assert "scripts\\hf_demo.py --weights" in text and "--build-only" not in text
