"""EXP-045 final evaluation (tests 1-3 and 5): the GPU-session kernel and its runner wiring.

The kernel reads the EXP-043 final weights (chosen ONLY by the sha256 committed in the complete
EXP-043 session summary) and the protected held-out texts, then runs ``scripts/eval_exp045.py``
with test 4's contamination flags and the coverage re-scan. These tests run on CPU with a tiny model;
the real session needs EXP-043 to be complete.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "scripts" / "kaggle" / "exp045_final_kernel.py"
WRAPPER = ROOT / "scripts" / "run_kaggle_exp045_final.ps1"
RUNNER = ROOT / "scripts" / "run_kaggle_exp038.ps1"
sys.path.insert(0, str(ROOT / "scripts"))


def _kernel():
    spec = importlib.util.spec_from_file_location("exp045_final_kernel", KERNEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _session(d: Path, n: int, complete: bool, sha: str | None = None) -> None:
    s = {"session": n, "complete": complete}
    if sha is not None:
        s["final"] = {"model_final": {"file": "model_final.pt", "sha256": sha, "step": 169911}}
    (d / f"session-{n}").mkdir(parents=True)
    (d / f"session-{n}" / "summary.json").write_text(json.dumps(s), encoding="utf-8")


# ----------------------------------------------------------------------------- kernel ----------
def test_kernel_is_ascii_with_safe_paths():
    text = KERNEL.read_text(encoding="ascii")  # Kaggle templates must be ASCII
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in ast.parse(text).body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__"
    # its own results folder: test 4's and the coverage re-scan's committed files are never touched
    assert consts["OUT"] == "/kaggle/working/EXP-045/final"
    assert str(consts["BELEBELE"]).startswith("/tmp/")  # ShareAlike data never enters the output
    assert str(consts["FP16"]).startswith("/kaggle/working/exp045_fp16/")  # outside EXP-045/: not published
    assert consts["CONTAMINATION"] == "evals/results/EXP-045/contamination.json"
    assert consts["COVERAGE"] == "evals/results/EXP-045/coverage/coverage.json"
    for rel in (consts["CONTAMINATION"], consts["COVERAGE"]):
        assert (ROOT / rel).is_file()
    assert "nvidia-smi" in text  # a GPU session


def test_final_record_refuses_until_exp043_is_complete(tmp_path):
    k = _kernel()
    # the real repository today: EXP-043 is still running, so the kernel must refuse
    if not any(
        json.loads(f.read_text(encoding="utf-8")).get("complete")
        for f in (ROOT / "evals" / "results" / "EXP-043").glob("session-*/summary.json")
    ):
        with pytest.raises(SystemExit, match="only after EXP-043 has finished"):
            k.final_record(ROOT / "evals" / "results" / "EXP-043")
    d = tmp_path / "EXP-043"
    d.mkdir()
    with pytest.raises(SystemExit):
        k.final_record(d)  # no sessions
    _session(d, 1, False)
    _session(d, 2, False)
    with pytest.raises(SystemExit):
        k.final_record(d)  # none complete
    _session(d, 3, True, sha="a" * 64)
    assert k.final_record(d) == {"session": "session-3", "sha256": "a" * 64, "step": 169911}
    _session(d, 4, True, sha="b" * 64)
    with pytest.raises(SystemExit):
        k.final_record(d)  # two complete sessions: ambiguous
    e = tmp_path / "e"
    e.mkdir()
    _session(e, 1, True)  # complete but no model_final sha256
    with pytest.raises(SystemExit, match="no model_final.pt sha256"):
        k.final_record(e)


def test_find_weights_only_by_committed_sha256(tmp_path):
    k = _kernel()
    a = tmp_path / "input" / "frontier-exp043-a" / "exp043_final"
    b = tmp_path / "input" / "other" / "exp043_final"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "model_final.pt").write_bytes(b"right weights")
    (b / "model_final.pt").write_bytes(b"wrong weights")
    good = hashlib.sha256(b"right weights").hexdigest()
    assert k.find_weights(tmp_path / "input", good) == a / "model_final.pt"
    with pytest.raises(SystemExit, match="committed sha256"):
        k.find_weights(tmp_path / "input", "0" * 64)


def test_find_heldout(tmp_path):
    k = _kernel()
    assert k.find_heldout(tmp_path) is None  # then test 1 is NOT RUN, never estimated
    (tmp_path / "h1").mkdir()
    (tmp_path / "h1" / "heldout-v1.jsonl").write_text("{}\n", encoding="utf-8")
    assert k.find_heldout(tmp_path) == tmp_path / "h1" / "heldout-v1.jsonl"
    (tmp_path / "h2").mkdir()
    (tmp_path / "h2" / "heldout-v1.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        k.find_heldout(tmp_path)


def test_eval_command_flags_exist_in_the_eval_script(tmp_path):
    k = _kernel()
    cmd = k.eval_command(ROOT, tmp_path / "model_final.pt", tmp_path / "heldout-v1.jsonl")
    assert cmd[1].endswith("eval_exp045.py") and "--heldout-jsonl" in cmd
    assert "--heldout-jsonl" not in k.eval_command(ROOT, tmp_path / "w.pt", None)
    flags = {c for c in cmd if c.startswith("--")}
    assert flags == {
        "--weights", "--out", "--belebele-dir", "--contamination", "--coverage", "--export-fp16",
        "--heldout-jsonl",
    }  # fmt: skip
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_exp045.py"), "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    ).stdout  # fmt: skip
    for f in flags:
        assert f in helptext


def test_kernel_command_runs_end_to_end_with_the_committed_flags(tmp_path, monkeypatch):
    """The exact command, with the REAL contamination.json and coverage.json, on a tiny CPU model."""
    import eval_exp045 as ev
    from frontier_ai.config import ModelConfig
    from frontier_ai.evaluation import belebele as bb
    from frontier_ai.model.gpt import GPT
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    k = _kernel()
    monkeypatch.setattr(k, "OUT", tmp_path / "EXP-045" / "final")
    monkeypatch.setattr(k, "FP16", tmp_path / "exp045_fp16" / "model_fp16.pt")
    torch.manual_seed(0)
    tok = load_frontier_tokenizer_v2()
    model = GPT(ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=2, n_embd=32, block_size=64, pos="rope"))
    w = tmp_path / "model_final.pt"
    sd = {"_orig_mod." + n: v.detach().float().clone() for n, v in model.state_dict().items()}
    torch.save({"model_config": asdict(model.cfg), "state_dict": sd, "exp_id": "EXP-043", "step": 7}, w)
    bdir = tmp_path / "belebele"
    bdir.mkdir()
    for lang in ("en", "hi"):
        rows = []
        for q in (1, 2):
            r = {
                "link": "https://example.org/0",
                "question_number": q,
                "flores_passage": f"A short passage in {lang} about a river and a mill.",
                "question": f"Question {q}?",
                "correct_answer_num": str(q),
                "dialect": bb.FILES[lang][0],
                "ds": "x",
            }
            r.update({f"mc_answer{j}": f"option {j}" for j in range(1, 5)})
            rows.append(r)
        (bdir / f"{bb.FILES[lang][0]}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    cmd = k.eval_command(ROOT, w, None)
    rc = ev.main(
        cmd[2:] + ["--belebele-dir", str(bdir), "--languages", "en,hi", "--no-verify", "--smoke", "--device", "cpu"]
    )
    assert rc == 0
    s = json.loads((tmp_path / "EXP-045" / "final" / "summary.json").read_text(encoding="utf-8"))
    assert s["tests"]["1_heldout"]["status"] == "NOT RUN"  # no held-out texts mounted: never estimated
    assert s["tests"]["2_belebele"]["coverage"] == "applied"  # the committed coverage.json was accepted
    assert s["model"]["sha256"] == hashlib.sha256(w.read_bytes()).hexdigest()
    assert s["fp16_copy"]["file"] == "model_fp16.pt" and (tmp_path / "exp045_fp16" / "model_fp16.pt").is_file()
    published = {p.name for p in (tmp_path / "EXP-045" / "final").iterdir()}
    assert published == {"summary.json", "SUMMARY.txt", "samples.jsonl", "belebele_items.jsonl"}


# ----------------------------------------------------------------------------- runner ----------
def test_wrapper_calls_the_shared_runner_with_final():
    text = WRAPPER.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-045" -Final -Relaunch:$Relaunch' in text


def test_runner_final_block_and_guard():
    text = RUNNER.read_text(encoding="ascii")
    assert "frontier-exp045" not in text  # the slug is built from $expSlug
    i = text.index("if ($Final) {\n    # EXP-045 tests 1-3 and 5")
    block = text[i : text.index("\n}\n", i)]
    assert '$template = "scripts\\kaggle\\exp045_final_kernel.py"' in block
    assert '$kernelSlug = "frontier-$expSlug-final"' in block
    assert '$outDir = "out\\kaggle\\EXP-045-final"' in block and '$statePath = "$outDir\\state.json"' in block
    assert "$dataFiles = @()" in block and "$maxWaitHours = 4" in block
    # after the defaults it replaces and before they are used
    assert text.index('$kernelSlug = "frontier-$expSlug"') < i < text.index('$kernelId = "$user/$kernelSlug"')
    assert i < text.index("foreach ($f in @($python, $template")
    assert 'if ($Final -and $Exp -ne "EXP-045")' in text
    # the final-model guard: refuses before EXP-043 is complete; picks -a (odd) or -b (even)
    g = text.index("if ($Final -and $launching) {")
    guard = text[g : text.index("\n}\n", g)]
    assert ".complete" in guard and "Stop-Run" in guard and "$done.Count -ne 1" in guard
    assert "% 2 -eq 0" in guard and '$kernelSources = @("$user/frontier-exp043-$finalKernel")' in guard
    assert text.index("$kernelSources = @()") < g < text.index("$kaggle kernels push -p $stage")
    # the between-sessions guard (EXP-045 is in it) still comes first
    assert text.index('if ($launching -and ($Exp -eq "EXP-044" -or $Exp -eq "EXP-045"') < g


def test_runner_final_inputs_gpu_and_downloads():
    text = RUNNER.read_text(encoding="ascii")
    # GPU stays on: the CPU switch excludes -Final
    j = text.index('        machine_shape = "NvidiaTeslaT4"\n    }\n')
    meta = text[j : text.index('Write-Ascii "$stage\\kernel-metadata.json"', j)]
    assert 'if (($Exp -eq "EXP-045" -and -not $Final) -or $Exp -eq "EXP-046") {' in meta
    assert "if ($Final) { $meta.dataset_sources = @($heldId) }" in meta
    assert "if ($Build -or $Final) {\n    # EXP-046 build: the protected held-out texts" in text
    # only the small result files first; never the whole output
    d = text.index("} elseif ($Final) {")
    assert "--file-pattern EXP-045/final/.*" in text[d : text.index("} elseif ($Build) {", d)]
    k = text.index('if ($Final) { $resultDir = "$outputDir\\$Exp\\final" }')
    assert text.index('if ($Exp -eq "EXP-045") { $resultDir') < k < text.index('$summaryPath = "$resultDir\\summary.json"')
    assert 'if ($Final) { $msgText = "${Exp}: final evaluation of the EXP-043 model' in text
    # the fp16 copy: after the push, by file pattern, sha256 checked against the summary
    f = text.index("if ($Final -and $summary.fp16_copy) {")
    fp = text[f : text.index("\n}\n", f)]
    assert text.index('Add-Report "PASS 6d: pushed to origin/$branch"') < f
    assert "--file-pattern exp045_fp16/.*" in fp and "$summary.fp16_copy.sha256" in fp
    assert re.search(r"Get-FileHash -Algorithm SHA256 \$fp", fp)
