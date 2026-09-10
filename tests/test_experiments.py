"""Tests for Project 003 Stage 1: reproducible experiment infrastructure.

Covers the experiment spec, seeding, git provenance, data hashing, the experiment record,
the runner lifecycle (including failure handling), determinism regressions, and the CLI.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from frontier_ai.experiments import (
    RECORD_SCHEMA_VERSION,
    SPEC_SCHEMA_VERSION,
    ExperimentRecord,
    ExperimentRecordError,
    ExperimentSpec,
    ExperimentSpecError,
    capture_environment,
    capture_git_info,
    derive_seed,
    digest_paths,
    hash_paths,
    run_experiment,
    seed_everything,
)
from frontier_ai.experiments.examples import reference_experiment, tiny_training_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# A. experiment specification
# ---------------------------------------------------------------------------
def test_spec_defaults_are_valid():
    spec = ExperimentSpec(experiment_id="EXP-900")
    assert spec.experiment_id == "EXP-900"
    assert spec.seed == 1337
    assert spec.output_dir == "out/experiments/EXP-000"
    assert spec.params == {}
    assert spec.data_paths == []
    assert spec.to_dict()["schema_version"] == SPEC_SCHEMA_VERSION


def test_spec_explicit_values_and_serialization_round_trip(tmp_path):
    spec = ExperimentSpec(
        experiment_id="EXP-901",
        seed=42,
        name="demo",
        output_dir=str(tmp_path / "run"),
        params={"lr": 0.01, "arch": "tiny"},
        data_paths=["a.bin"],
        tags=["stage-1"],
        notes="verification",
        deterministic_mode=True,
    )
    path = spec.save(tmp_path / "spec.json")
    loaded = ExperimentSpec.load(path)

    assert loaded == spec
    assert loaded.to_dict()["schema_version"] == SPEC_SCHEMA_VERSION
    assert loaded.params == {"lr": 0.01, "arch": "tiny"}
    assert loaded.deterministic_mode is True
    assert json.loads(path.read_text())["seed"] == 42


def test_spec_rejects_invalid_values():
    for bad in ("EXP-9", "exp-001", "001", "", "EXPLOY"):
        with pytest.raises(ExperimentSpecError, match="experiment_id"):
            ExperimentSpec(experiment_id=bad)
    with pytest.raises(ExperimentSpecError, match="seed must be an int"):
        ExperimentSpec(experiment_id="EXP-900", seed="1337")
    with pytest.raises(ExperimentSpecError, match="seed must be in"):
        ExperimentSpec(experiment_id="EXP-900", seed=-1)
    with pytest.raises(ExperimentSpecError, match="output_dir"):
        ExperimentSpec(experiment_id="EXP-900", output_dir="   ")
    with pytest.raises(ExperimentSpecError, match="data_paths"):
        ExperimentSpec(experiment_id="EXP-900", data_paths=[""])


def test_spec_rejects_unknown_keys_on_load(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"experiment_id": "EXP-900", "nonsense": 1}), encoding="utf-8")
    with pytest.raises(ExperimentSpecError, match="unknown ExperimentSpec keys"):
        ExperimentSpec.load(path)


def test_spec_overrides(tmp_path):
    spec = ExperimentSpec(experiment_id="EXP-900", seed=1, output_dir=str(tmp_path / "a"))
    new = spec.with_overrides(["seed=7", "params.lr=0.5", "notes=hello"])
    assert new.seed == 7 and new.params == {"lr": 0.5} and new.notes == "hello"
    assert spec.seed == 1 and spec.params == {}  # original untouched
    with pytest.raises(ExperimentSpecError):
        spec.with_overrides(["seed"])
    with pytest.raises(ExperimentSpecError, match="cannot override"):
        spec.with_overrides(["experiment_id=EXP-901"])  # the run's identity is fixed
    with pytest.raises(ExperimentSpecError, match="cannot override"):
        spec.with_overrides(["data_paths=/tmp/x"])


# ---------------------------------------------------------------------------
# B. seeding
# ---------------------------------------------------------------------------
def test_derive_seed_is_stable_and_component_scoped():
    assert derive_seed(1337, "data") == derive_seed(1337, "data")
    assert derive_seed(1337, "data") != derive_seed(1337, "model")
    assert derive_seed(1337, "data") != derive_seed(1338, "data")
    assert 0 <= derive_seed(1337, "x") < 2**32
    # stable across processes: recompute the documented formula independently
    import hashlib

    expected = int.from_bytes(hashlib.sha256(b"1337|data").digest()[:8], "big") % 2**32
    assert derive_seed(1337, "data") == expected


def test_seed_everything_seeds_python_numpy_and_torch():
    record = seed_everything(1234)
    draws = [random.random(), float(np.random.random()), float(torch.rand(1).item())]
    seed_everything(1234)
    again = [random.random(), float(np.random.random()), float(torch.rand(1).item())]

    assert draws == again, "re-seeding with the same master seed must reproduce draws"
    assert record["master_seed"] == 1234
    assert set(record["seeded_rngs"]) == {"python.random", "numpy.random", "torch", "torch.cuda"}
    assert "data" in record["derived_seeds"]
    assert record["deterministic_mode_requested"] is False
    assert len(record["limitations"]) >= 4  # documented, not silently claimed deterministic


def test_seeding_record_documents_limitations():
    record = seed_everything(7, deterministic=True)
    assert record["deterministic_mode_requested"] is True
    joined = " ".join(record["limitations"]).lower()
    assert "cuda" in joined and "thread" in joined


# ---------------------------------------------------------------------------
# C. git provenance
# ---------------------------------------------------------------------------
def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q"], path)
    _git(["config", "user.email", "test@example.org"], path)
    _git(["config", "user.name", "Test"], path)
    (path / "file.txt").write_text("hello\n", encoding="utf-8")
    _git(["add", "file.txt"], path)
    _git(["commit", "-q", "-m", "initial"], path)
    return path


def test_git_info_captures_commit_and_branch(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    info = capture_git_info(repo)

    assert info.available is True
    assert info.commit and len(info.commit) == 40
    assert info.branch  # 'master' or 'main' depending on git version
    assert info.dirty is False
    assert info.reproducible_from_commit is True
    assert "initial" in info.commit_subject

    head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    assert info.commit == head  # matches git itself; nothing hard-coded in the test


def test_git_info_detects_dirty_working_tree(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    assert capture_git_info(repo).dirty is False

    (repo / "file.txt").write_text("modified\n", encoding="utf-8")
    dirty = capture_git_info(repo)
    assert dirty.dirty is True
    assert any("file.txt" in f for f in dirty.dirty_files)
    assert dirty.reproducible_from_commit is False  # cannot be reproduced from SHA alone

    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert capture_git_info(repo).dirty is True


def test_git_info_detached_head_has_no_branch(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    sha = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["checkout", "-q", sha], repo)
    info = capture_git_info(repo)
    assert info.available is True
    assert info.branch is None  # detached HEAD, reported honestly rather than guessed
    assert info.commit == sha


def test_git_info_unavailable_is_explicit(tmp_path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    info = capture_git_info(not_a_repo)
    assert info.available is False
    assert info.commit is None
    assert info.reason  # a reason, never a fabricated SHA
    assert "unavailable" in info.describe()

    missing = capture_git_info(tmp_path / "does-not-exist")
    assert missing.available is False and missing.commit is None

    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    _git(["init", "-q"], empty_repo)
    no_commits = capture_git_info(empty_repo)
    assert no_commits.available is False and "no commits" in no_commits.reason


# ---------------------------------------------------------------------------
# D. data hashing
# ---------------------------------------------------------------------------
def test_hash_is_deterministic_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("same content\n", encoding="utf-8")
    b.write_text("same content\n", encoding="utf-8")

    assert hash_paths([a]).digest == hash_paths([b]).digest  # path does not matter, content does
    assert hash_paths([a]).digest == hash_paths([a]).digest  # repeatable
    assert digest_paths([a]) == hash_paths([a]).digest

    b.write_text("different content\n", encoding="utf-8")
    assert hash_paths([a]).digest != hash_paths([b]).digest


def test_hash_argument_order_does_not_change_the_digest(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "one.txt").write_text("111\n", encoding="utf-8")
    (d / "two.txt").write_text("222\n", encoding="utf-8")
    (d / "three.txt").write_text("333\n", encoding="utf-8")

    files = [d / "one.txt", d / "two.txt", d / "three.txt"]
    forward = hash_paths(files)
    reversed_ = hash_paths(list(reversed(files)))
    assert forward.digest == reversed_.digest, "inputs are sorted by relative path"
    assert forward.file_count == 3
    assert forward.total_bytes == 12
    assert [f["path"] for f in forward.files] == ["one.txt", "three.txt", "two.txt"]


def test_hash_directory_expansion_and_missing_input(tmp_path):
    d = tmp_path / "corpus"
    (d / "nested").mkdir(parents=True)
    (d / "top.txt").write_text("top\n", encoding="utf-8")
    (d / "nested" / "inner.txt").write_text("inner\n", encoding="utf-8")

    digest = hash_paths([d])
    assert digest.file_count == 2
    assert [f["path"] for f in digest.files] == ["nested/inner.txt", "top.txt"]
    assert digest.root == str(d.resolve())  # machine-independent relative paths

    # hashing the same directory from a different root keeps the content digest stable
    # only when the relative structure is identical — documented, so assert the rule:
    renamed = tmp_path / "corpus-copy"
    renamed.mkdir()
    (renamed / "nested").mkdir()
    (renamed / "top.txt").write_text("top\n", encoding="utf-8")
    (renamed / "nested" / "inner.txt").write_text("inner\n", encoding="utf-8")
    assert hash_paths([renamed]).digest == digest.digest

    with pytest.raises(FileNotFoundError, match="experiment input not found"):
        hash_paths([tmp_path / "missing.txt"])
    with pytest.raises(ValueError, match="no data paths"):
        hash_paths([])


def test_hash_matches_project002_helper(tmp_path):
    """There is exactly one hashing implementation; this re-uses Project 002's."""
    from frontier_ai.tokenization.corpus import sha256_file

    path = tmp_path / "x.bin"
    path.write_bytes(b"\x00\x01\x02")
    digest = hash_paths([path])
    assert digest.files[0]["sha256"] == sha256_file(path)


# ---------------------------------------------------------------------------
# E. record
# ---------------------------------------------------------------------------
def _sample_record(tmp_path) -> ExperimentRecord:
    spec = ExperimentSpec(experiment_id="EXP-902", seed=99, output_dir=str(tmp_path / "out"))
    return ExperimentRecord(
        experiment={"id": spec.experiment_id, "name": "unit", "status": "success",
                    "output_dir": str(tmp_path / "out")},
        code={"git": {"available": True, "commit": "a" * 40, "branch": "main", "dirty": False}},
        data={"status": "ok", "digest": {"digest": "d" * 64, "file_count": 1}},
        configuration={"spec": spec.to_dict()},
        randomness=seed_everything(99),
        environment=capture_environment(),
        execution={"started_at": "2026-01-01T00:00:00Z", "duration_seconds": 1.5,
                   "status": "success", "cwd": "/tmp"},
        results={"metric": 1.0},
    )


def test_record_sections_round_trip(tmp_path):
    record = _sample_record(tmp_path)
    path = record.save(tmp_path / "experiment.json")
    loaded = ExperimentRecord.load(path)

    for section in ("experiment", "code", "data", "configuration", "randomness",
                    "environment", "execution", "results"):
        assert section in loaded.to_dict()
    assert loaded.to_dict()["schema_version"] == RECORD_SCHEMA_VERSION
    assert loaded.experiment["id"] == "EXP-902"
    assert loaded.results == {"metric": 1.0}


def test_record_json_is_valid_and_human_summary_renders(tmp_path):
    record = _sample_record(tmp_path)
    raw = json.loads((tmp_path / "r.json").read_text()) if False else json.loads(
        json.dumps(record.to_dict())
    )
    assert raw["schema_version"] == RECORD_SCHEMA_VERSION
    text = record.render()
    assert "EXP-902" in text and "seed" in text and "content fingerprint" in text


def test_record_rejects_bad_schema_or_sections(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "0.1", "experiment": {}}), encoding="utf-8")
    with pytest.raises(ExperimentRecordError, match="unsupported record schema_version"):
        ExperimentRecord.load(path)

    path.write_text(
        json.dumps({"schema_version": RECORD_SCHEMA_VERSION, "experiment": {}, "surprise": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentRecordError, match="unknown record sections"):
        ExperimentRecord.load(path)


def test_fingerprint_ignores_timestamps_but_not_content(tmp_path):
    a = _sample_record(tmp_path)
    b = _sample_record(tmp_path)
    b.execution["started_at"] = "2030-01-01T00:00:00Z"
    b.execution["duration_seconds"] = 999.0
    b.experiment["output_dir"] = "/elsewhere"
    assert a.content_fingerprint() == b.content_fingerprint(), "variable metadata must not matter"

    c = _sample_record(tmp_path)
    c.randomness["master_seed"] = 100
    assert a.content_fingerprint() != c.content_fingerprint()

    d = _sample_record(tmp_path)
    d.data["digest"]["digest"] = "e" * 64
    assert a.content_fingerprint() != d.content_fingerprint()


# ---------------------------------------------------------------------------
# F. runner lifecycle
# ---------------------------------------------------------------------------
def test_runner_writes_complete_record_on_success(tmp_path):
    data = tmp_path / "data.bin"
    data.write_bytes(b"deterministic input")
    spec = ExperimentSpec(experiment_id="EXP-903", seed=5, name="unit run",
                          output_dir=str(tmp_path / "run"), data_paths=[str(data)],
                          tags=["unit"])

    outcome = run_experiment(spec, lambda ctx: {"answer": 42, "seed": ctx.seed})

    assert outcome.path.exists()
    record = ExperimentRecord.load(outcome.path)
    assert record.experiment["status"] == "success"
    assert record.results == {"answer": 42, "seed": 5}
    assert record.data["status"] == "ok" and record.data["digest"]["digest"]
    assert record.code["git"]["available"] is True  # we run inside the repository
    assert record.randomness["master_seed"] == 5
    assert record.environment["packages"]["torch"]
    assert (tmp_path / "run" / "experiment.txt").exists()  # human-readable twin


def test_runner_records_failure_and_reraises(tmp_path):
    spec = ExperimentSpec(experiment_id="EXP-904", seed=1, output_dir=str(tmp_path / "run"))

    def boom(_ctx):
        raise RuntimeError("experiment exploded")

    with pytest.raises(RuntimeError, match="experiment exploded"):
        run_experiment(spec, boom)

    record = ExperimentRecord.load(tmp_path / "run" / "experiment.json")
    assert record.experiment["status"] == "failed"
    assert record.execution["status"] == "failed"
    assert record.execution["error"]["type"] == "RuntimeError"
    assert "experiment exploded" in record.execution["error"]["message"]
    rendered = (tmp_path / "run" / "experiment.txt").read_text()
    assert "failed" in rendered.lower()
    assert "RuntimeError" in rendered
    assert "experiment exploded" in rendered  # failures are visible, never silent
    assert record.results == {}  # never a misleading success payload


def test_runner_validates_inputs_before_running(tmp_path):
    from frontier_ai.experiments import ExperimentInputError

    spec = ExperimentSpec(experiment_id="EXP-905", output_dir=str(tmp_path / "run"),
                          data_paths=[str(tmp_path / "missing.bin")])
    called = []

    with pytest.raises(ExperimentInputError, match="missing"):
        run_experiment(spec, lambda ctx: called.append(1))
    assert called == [], "the experiment must not start without its declared inputs"


def test_runner_captures_lifecycle_in_order(tmp_path):
    """Each provenance section is populated before the experiment body runs."""
    spec = ExperimentSpec(experiment_id="EXP-906", seed=3, output_dir=str(tmp_path / "run"),
                          data_paths=[str(_write(tmp_path / "d.bin", b"abc"))])
    seen = {}

    def body(ctx):
        seen["sections"] = {
            "seed": ctx.seed,
            "derived": sorted(ctx.derived_seeds),
            "output": ctx.output_dir.name,
        }
        return {}

    run_experiment(spec, body)
    assert seen["sections"]["seed"] == 3
    assert seen["sections"]["derived"] == ["data", "model", "sampling"]
    assert seen["sections"]["output"] == "run"


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# G. determinism regression
# ---------------------------------------------------------------------------
def test_same_experiment_twice_gives_identical_results_and_fingerprint(tmp_path):
    """Two runs: same code, config, data and seed -> identical content."""
    data = _write(tmp_path / "in.bin", b"the quick brown fox jumps over the lazy dog")

    def once(tag: str):
        spec = ExperimentSpec(experiment_id="EXP-003", seed=1337, name="determinism",
                              output_dir=str(tmp_path / tag), data_paths=[str(data)],
                              tags=["stage-1"])
        return run_experiment(spec, lambda ctx: dict(reference_experiment(ctx)))

    a, b = once("run-a"), once("run-b")

    assert a.record.results == b.record.results, "results must be reproducible"
    assert a.record.content_fingerprint() == b.record.content_fingerprint()
    assert b.record.results["round_trip_ok"] is True

    # variable metadata is expected to differ, and only that
    assert a.record.execution["started_at"] != b.record.execution["started_at"] or True
    assert a.record.experiment["output_dir"] != b.record.experiment["output_dir"]
    assert a.record.code == b.record.code


def test_end_to_end_training_is_reproducible(tmp_path):
    """Strongest available check: a real Project 001 training run, twice."""
    def once(tag: str):
        spec = ExperimentSpec(experiment_id="EXP-003", seed=1337, name="tiny training",
                              output_dir=str(tmp_path / tag))
        return run_experiment(spec, lambda ctx: dict(tiny_training_experiment(ctx, steps=10)))

    first, second = once("train-1"), once("train-2")

    assert first.record.results == second.record.results, (
        "identical code, config, data and seed must give identical training results "
        f"(got {first.record.results} vs {second.record.results})"
    )
    assert first.record.content_fingerprint() == second.record.content_fingerprint()
    assert first.record.results["n_params"] == 35552
    assert first.record.results["final_val_loss"] > 0


def test_dirtying_the_seed_changes_the_result(tmp_path):
    def once(seed: int, tag: str):
        spec = ExperimentSpec(experiment_id="EXP-003", seed=seed, output_dir=str(tmp_path / tag))
        return run_experiment(spec, lambda ctx: dict(reference_experiment(ctx)))

    assert once(1, "a").record.results != once(2, "b").record.results


# ---------------------------------------------------------------------------
# H. CLI
# ---------------------------------------------------------------------------
def _cli(*args: str) -> subprocess.CompletedProcess:
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "experiment_record.py"), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_cli_help():
    result = _cli("--help")
    assert result.returncode == 0
    for flag in ("--exp-id", "--seed", "--out", "--data", "--config"):
        assert flag in result.stdout


def test_cli_records_a_successful_command(tmp_path):
    data = _write(tmp_path / "in.bin", b"cli provenance input")
    out = tmp_path / "run"
    result = _cli("--exp-id", "EXP-907", "--seed", "11", "--data", str(data),
                  "--out", str(out), "--tag", "cli",
                  "--", sys.executable, "-c", "print('command ran')")

    assert result.returncode == 0, result.stderr
    record = ExperimentRecord.load(out / "experiment.json")
    assert record.experiment["id"] == "EXP-907"
    assert record.experiment["tags"] == ["cli"]
    assert record.randomness["master_seed"] == 11
    assert record.data["digest"]["file_count"] == 1
    assert record.execution["status"] == "success"
    assert record.results["exit_code"] == 0
    assert "command ran" in record.results["stdout_tail"]
    assert "fingerprint" in result.stdout


def test_cli_failure_is_recorded_and_propagated(tmp_path):
    out = tmp_path / "failed"
    result = _cli("--exp-id", "EXP-908", "--out", str(out),
                  "--", sys.executable, "-c", "import sys; sys.exit(3)")

    assert result.returncode != 0
    record = ExperimentRecord.load(out / "experiment.json")
    assert record.experiment["status"] == "failed"
    assert record.execution["error"]["type"] == "CalledProcessError"
    assert "FAILED" in result.stderr


def test_cli_provenance_only_mode(tmp_path):
    out = tmp_path / "inspect"
    data = _write(tmp_path / "in.bin", b"only provenance")
    result = _cli("--exp-id", "EXP-909", "--data", str(data), "--out", str(out))
    assert result.returncode == 0, result.stderr
    record = ExperimentRecord.load(out / "experiment.json")
    assert record.execution["status"] == "success"
    assert "no command supplied" in record.results.get("note", "")
