"""Tests for automatic experiment-record wiring in the Project 001/002 CLIs (D-032).

The rule under test: the **outer** run owns the record.

    python scripts/train.py ...                                -> train.py writes a record
    python scripts/experiment_record.py --exp-id EXP-001 -- python scripts/train.py ...
                                                               -> the wrapper writes the only record

Scripts keep working exactly as before (same stdout, same exit codes); what is new is a
standard ``experiment.json`` next to the artifacts an artifact-producing run writes, and
an explicit failure mode when the run does not succeed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from frontier_ai.experiments import ExperimentRecord, ExperimentSpec, hash_paths
from frontier_ai.experiments.autowire import (
    RECORD_FILENAME_NAME,
    id_is_recordable,
    run_self_recorded,
    stable_results,
)
from frontier_ai.experiments.runner import EXPERIMENT_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _run(script: str, *args: str, env_extra: dict[str, str] | None = None):
    env = {**dict(os.environ), "PYTHONPATH": str(REPO_ROOT / "src"), **(env_extra or {})}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _train(tmp_path: Path, *extra: str, out: str | None = None):
    out_dir = out or str(tmp_path / "run")
    return _run(
        "train.py",
        "--config",
        "configs/cpu_smoke.json",
        "--set",
        f"train.out_dir={out_dir}",
        "--max-steps",
        "3",
        *extra,
    )


def _record(path: Path) -> ExperimentRecord:
    assert path.exists(), f"no record at {path}"
    return ExperimentRecord.load(path)


# ---------------------------------------------------------------------------
# A. train.py records itself
# ---------------------------------------------------------------------------
def test_train_writes_a_standard_record(tmp_path):
    out = tmp_path / "run"
    result = _train(tmp_path, "--exp-id", "EXP-880")

    assert result.returncode == 0, result.stderr
    record = _record(out / RECORD_FILENAME_NAME)
    assert record.experiment["status"] == "success"
    assert record.experiment["id"] == "EXP-880"
    assert record.experiment["tags"] == ["project-001", "train"]
    assert record.execution["status"] == "success"
    assert record.execution["error"] is None

    # provenance: seed, git, environment, command, config file
    assert record.randomness["master_seed"] == json.loads(
        (REPO_ROOT / "configs" / "cpu_smoke.json").read_text()
    )["train"]["seed"]
    assert record.code["git"]["available"] is True
    assert record.code["git"]["commit"]
    assert record.environment["packages"]["torch"]
    assert record.configuration["spec"]["command"][0].endswith("train.py")
    assert record.configuration["spec"]["params"]["max_steps"] == 3
    assert record.configuration["config_file"]["path"] == "configs/cpu_smoke.json"

    # inputs and metrics
    assert record.data["status"] == "ok"
    assert record.data["digest"]["file_count"] == 1
    for key in ("best_val", "best_bpb", "steps", "n_params", "tokens_seen"):
        assert key in record.results, key
    assert record.results["steps"] == 3

    # the human-readable companion file is written too
    assert (out / "experiment.txt").exists()
    assert "[record]" in result.stderr


def test_train_repeat_run_shares_one_fingerprint(tmp_path):
    out = tmp_path / "run"
    first = _train(tmp_path, "--exp-id", "EXP-881")
    assert first.returncode == 0, first.stderr
    second = _train(tmp_path, "--exp-id", "EXP-881")
    assert second.returncode == 0, second.stderr

    a = _record(out / RECORD_FILENAME_NAME).content_fingerprint()
    b = _record(out / RECORD_FILENAME_NAME).content_fingerprint()
    assert a == b


def test_train_failure_is_recorded_as_failed(tmp_path):
    out = tmp_path / "failed"
    result = _train(tmp_path, "--set", "model.n_head=7", "--exp-id", "EXP-882", out=str(out))

    assert result.returncode == 1
    assert "[record] FAILED" in result.stderr
    record = _record(out / RECORD_FILENAME_NAME)
    assert record.experiment["status"] == "failed"
    assert record.execution["status"] == "failed"
    assert record.execution["error"]["type"] == "ValueError"
    assert "n_head" in record.execution["error"]["message"]
    assert record.results == {}


def test_train_missing_declared_input_fails_loudly_without_a_success_record(tmp_path):
    out = tmp_path / "missing"
    result = _run(
        "train.py",
        "--config",
        "configs/cpu_smoke.json",
        "--set",
        f"train.out_dir={out}",
        "--set",
        "data.path=out/does-not-exist.bin",
        "--exp-id",
        "EXP-883",
    )

    assert result.returncode == 1
    assert "declared data inputs are missing" in result.stderr
    assert "[record] FAILED" in result.stderr
    assert not (out / RECORD_FILENAME_NAME).exists()


def test_train_no_record_writes_nothing(tmp_path):
    out = tmp_path / "quiet"
    result = _train(tmp_path, "--no-record", "--exp-id", "EXP-884", out=str(out))

    assert result.returncode == 0, result.stderr
    assert (out / "config.json").exists()
    assert not (out / RECORD_FILENAME_NAME).exists()
    assert "[record]" not in result.stderr


def test_train_inspection_runs_do_not_record(tmp_path):
    printed = _train(tmp_path, "--print-model", "--exp-id", "EXP-885", out=str(tmp_path / "printed"))
    assert printed.returncode == 0, printed.stderr
    assert "GPT(" in printed.stdout
    assert not (tmp_path / "printed" / RECORD_FILENAME_NAME).exists()

    evaluated = _train(tmp_path, "--eval-only", "--exp-id", "EXP-885", out=str(tmp_path / "evaled"))
    assert evaluated.returncode == 0, evaluated.stderr
    assert "val_loss" in evaluated.stdout
    assert not (tmp_path / "evaled" / RECORD_FILENAME_NAME).exists()


# ---------------------------------------------------------------------------
# B. tokenizer CLIs record themselves
# ---------------------------------------------------------------------------
def test_tokenizer_clis_each_write_a_record(tmp_path):
    corpus = tmp_path / "corpus"
    prepared = _run(
        "tokenizer_prepare_corpus.py",
        "--out",
        str(corpus),
        "--sentences-per-language",
        "2",
        "--english-chars",
        "800",
        "--seed",
        "7",
        "--exp-id",
        "EXP-886",
    )
    assert prepared.returncode == 0, prepared.stderr
    corpus_record = _record(corpus / RECORD_FILENAME_NAME)
    assert corpus_record.experiment["tags"] == ["project-002", "corpus"]
    assert corpus_record.randomness["master_seed"] == 7
    assert corpus_record.execution["status"] == "success"

    bpe = tmp_path / "tok-bpe"
    trained = _run(
        "tokenizer_train.py",
        "--corpus",
        str(corpus),
        "--impl",
        "bpe_python",
        "--vocab-size",
        "400",
        "--out",
        str(bpe),
        "--exp-id",
        "EXP-886",
    )
    assert trained.returncode == 0, trained.stderr
    bpe_record = _record(bpe / RECORD_FILENAME_NAME)
    assert bpe_record.experiment["tags"] == ["project-002", "tokenizer"]
    assert bpe_record.data["digest"]["file_count"] == 1          # the corpus train file
    assert bpe_record.results["vocab_size"] == 400
    assert bpe_record.results["sample_round_trip_ok"] is True
    # nothing that varies per run may leak into a fingerprint
    assert "created_at" not in bpe_record.results
    assert "artifact_dir" not in bpe_record.results

    char = tmp_path / "tok-char"
    built = _run(
        "tokenizer_train.py",
        "--corpus",
        str(corpus),
        "--impl",
        "char",
        "--out",
        str(char),
        "--exp-id",
        "EXP-886",
    )
    assert built.returncode == 0, built.stderr
    char_record = _record(char / RECORD_FILENAME_NAME)
    assert char_record.execution["status"] == "success"
    assert char_record.results["impl"] == "char"

    compared = _run(
        "tokenizer_compare.py",
        "--corpus",
        str(corpus),
        "--tokenizer",
        str(bpe),
        str(char),
        "--out",
        str(tmp_path / "compare.json"),
        "--exp-id",
        "EXP-886",
    )
    assert compared.returncode == 0, compared.stderr
    compare_record = _record(tmp_path / RECORD_FILENAME_NAME)
    assert compare_record.experiment["tags"] == ["project-002", "comparison"]
    assert compare_record.data["digest"]["file_count"] >= 3      # corpus + two artifacts
    hashed = [entry["path"] for entry in compare_record.data["digest"]["files"]]
    assert hashed, "the comparison must fingerprint the inputs it read"
    assert not any(name.endswith(("experiment.json", "experiment.txt")) for name in hashed)
    assert compare_record.results["labels"]

    # every stage recorded itself, and every record is a standard one
    for path in (
        corpus / RECORD_FILENAME_NAME,
        bpe / RECORD_FILENAME_NAME,
        char / RECORD_FILENAME_NAME,
        tmp_path / RECORD_FILENAME_NAME,
    ):
        assert _record(path).record_type == ExperimentRecord.load(path).record_type
        assert _record(path).schema_version == ExperimentRecord.load(path).schema_version


def test_tokenizer_results_do_not_depend_on_the_output_directory(tmp_path):
    corpus = tmp_path / "corpus"
    assert _run(
        "tokenizer_prepare_corpus.py", "--out", str(corpus), "--sentences-per-language", "2",
        "--english-chars", "800", "--seed", "7",
    ).returncode == 0

    def train_once(out_dir: Path) -> ExperimentRecord:
        result = _run(
            "tokenizer_train.py",
            "--corpus",
            str(corpus),
            "--impl",
            "bpe_python",
            "--vocab-size",
            "400",
            "--out",
            str(out_dir),
        )
        assert result.returncode == 0, result.stderr
        return _record(out_dir / RECORD_FILENAME_NAME)

    a = train_once(tmp_path / "a")
    b = train_once(tmp_path / "b")
    assert a.results == b.results
    assert a.data["digest"] == b.data["digest"]
    # the record knows where it was written, but that is not part of its content identity
    assert a.experiment["output_dir"] != b.experiment["output_dir"]


# ---------------------------------------------------------------------------
# C. no duplicate records: the outer run owns the record
# ---------------------------------------------------------------------------
def test_nested_run_produces_exactly_one_record(tmp_path):
    wrapped = tmp_path / "wrapped"
    inner = tmp_path / "inner-run"
    result = _run(
        "experiment_record.py",
        "--exp-id",
        "EXP-887",
        "--seed",
        "1337",
        "--data",
        "data/synthetic.bin",
        "--out",
        str(wrapped),
        "--",
        sys.executable,
        "scripts/train.py",
        "--config",
        "configs/cpu_smoke.json",
        "--set",
        f"train.out_dir={inner}",
        "--max-steps",
        "3",
    )

    assert result.returncode == 0, result.stderr
    # one record, the wrapper's
    records = sorted(p.name for p in tmp_path.rglob(RECORD_FILENAME_NAME))
    assert records == [RECORD_FILENAME_NAME]
    outer = _record(wrapped / RECORD_FILENAME_NAME)
    assert outer.experiment["id"] == "EXP-887"
    assert outer.execution["status"] == "success"
    # the inner run did its work and its metrics are visible in the wrapper's stdout
    assert "[train] done." in "\n".join(outer.results["stdout_tail"])
    # and it said out loud that it is not the owner
    assert "nested inside" in result.stderr + "\n".join(outer.results.get("stderr_tail", []))


def test_env_var_marks_a_run_as_nested(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    outer.mkdir()
    monkeypatch.setenv(EXPERIMENT_ENV_VAR, str(outer))

    out = tmp_path / "inner"
    spec = ExperimentSpec(experiment_id="EXP-888", seed=1, output_dir=str(out))
    outcome = run_self_recorded(spec, lambda: {"metric": 1.0})

    assert outcome.nested is True
    assert outcome.recorded is False
    assert outcome.results == {"metric": 1.0}
    assert not (out / RECORD_FILENAME_NAME).exists()


# ---------------------------------------------------------------------------
# D. the adapter itself
# ---------------------------------------------------------------------------
def test_run_self_recorded_failure_writes_a_failed_record(tmp_path):
    out = tmp_path / "boom"
    spec = ExperimentSpec(experiment_id="EXP-889", seed=1, output_dir=str(out))

    def body():
        raise RuntimeError("kaboom")

    outcome = run_self_recorded(spec, body, quiet=True)

    assert outcome.failed is True
    assert outcome.exit_code() == 1
    assert outcome.record_path is not None and Path(outcome.record_path).exists()
    record = ExperimentRecord.load(outcome.record_path)
    assert record.experiment["status"] == "failed"
    assert record.execution["error"]["type"] == "RuntimeError"


def test_sys_exit_inside_the_body_is_not_recorded_as_success(tmp_path):
    out = tmp_path / "exiting"
    spec = ExperimentSpec(experiment_id="EXP-890", seed=1, output_dir=str(out))

    def body():
        raise SystemExit("user asked to stop")

    outcome = run_self_recorded(spec, body, quiet=True)

    assert outcome.failed is True
    assert outcome.exit_code() == 1
    record = ExperimentRecord.load(out / RECORD_FILENAME_NAME)
    assert record.experiment["status"] == "failed"


def test_unrecordable_experiment_id_skips_recording_without_breaking_the_run(tmp_path):
    out = tmp_path / "legacy"
    # EXP-TEST is not an EXP-<3+ digits> id; the tokenizer CLIs accepted it before this
    # wiring existed, so the work must still happen - just without a record.
    outcome = run_self_recorded(
        lambda: ExperimentSpec(experiment_id="EXP-TEST", seed=1, output_dir=str(out)),
        lambda: {"ok": True},
    )

    assert outcome.skipped is True
    assert outcome.recorded is False
    assert outcome.exit_code() == 0
    assert outcome.results == {"ok": True}
    assert not (out / RECORD_FILENAME_NAME).exists()


def test_a_record_inside_an_input_directory_does_not_change_the_digest(tmp_path):
    """D-033: records are generated output, not source data."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "data.bin").write_text("payload")
    before = hash_paths([str(inputs)])

    (inputs / "experiment.json").write_text('{"started_at": "one"}')
    (inputs / "experiment.txt").write_text("report one")
    after = hash_paths([str(inputs)])

    assert before.digest == after.digest
    assert [entry["path"] for entry in after.files] == ["data.bin"]
    # declared explicitly, a record is data like anything else
    assert hash_paths([str(inputs / "experiment.json")]).file_count == 1


def test_id_is_recordable():
    assert id_is_recordable("EXP-000") is True
    assert id_is_recordable("EXP-042") is True
    assert id_is_recordable("EXP-TEST") is False
    assert id_is_recordable("") is False


def test_stable_results_drops_run_varying_keys():
    payload = {
        "created_at": "2026-01-01T00:00:00Z",
        "artifact_dir": "/tmp/one",
        "rows": [{"created_at": "x", "score": 1}],
        "kept": {"nested": [{"artifact_dir": "/tmp/two", "value": 2}]},
    }
    assert stable_results(payload) == {
        "rows": [{"score": 1}],
        "kept": {"nested": [{"value": 2}]},
    }
