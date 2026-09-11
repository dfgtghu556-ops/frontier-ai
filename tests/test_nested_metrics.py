"""Regression tests for D-034: a nested run keeps its metrics without a second record.

D-032 still holds — **the outer run owns the record** — but owning the record used to mean
losing the inner run's numbers: a swept ``scripts/train.py`` wrote no record of its own and
published nothing, so ``--metric best_val`` found only ``exit_code`` and log tails in the
outer record and the sweep reported ``metric_missing``.

The fix keeps one record and hands the metrics over in the project's existing
machine-readable form: a nested script prints one JSON line
(:data:`~frontier_ai.experiments.runner.NESTED_RESULTS_MARKER`) and
:func:`~frontier_ai.experiments.runner.run_command` merges it into the outer record's
``results``. Sweeps then read the metric from ``results`` exactly as they always did — no
log scraping, no argv inspection, no schema change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from frontier_ai.experiments import ExperimentRecord
from frontier_ai.experiments.runner import (
    NESTED_RESULTS_MARKER,
    NESTED_RESULTS_SCHEMA,
    nested_results_line,
    parse_nested_results,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
RECORD = "experiment.json"


def _run(script: str, *args: str, env_extra: dict[str, str] | None = None):
    env = {**dict(os.environ), "PYTHONPATH": str(REPO_ROOT / "src"), **(env_extra or {})}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _records(root: Path) -> list[Path]:
    return sorted(root.rglob(RECORD))


# ---------------------------------------------------------------------------
# the protocol itself
# ---------------------------------------------------------------------------
def test_nested_results_line_carries_the_results_under_the_marker():
    line = nested_results_line({"best_val": 1.5, "steps": 3}, script="train.py")
    payload = json.loads(line)
    assert payload[NESTED_RESULTS_MARKER]["schema"] == NESTED_RESULTS_SCHEMA
    assert payload[NESTED_RESULTS_MARKER]["script"] == "train.py"
    assert payload[NESTED_RESULTS_MARKER]["results"] == {"best_val": 1.5, "steps": 3}


def test_parse_nested_results_ignores_logs_and_plain_json():
    stdout = "\n".join(
        [
            "[    0.0s] event=run.end  steps=3  best_val=3.9",  # human-readable log
            '{"vocab_size": 300, "impl": "char"}',  # a command's own JSON report
            "not json at all",
            nested_results_line({"best_val": 3.9}, script="train.py"),
        ]
    )
    assert parse_nested_results(stdout) == {"best_val": 3.9}


def test_parse_nested_results_is_order_stable_and_first_wins():
    stdout = "\n".join(
        [
            nested_results_line({"best_val": 1.0, "steps": 3}),
            nested_results_line({"best_val": 2.0, "n_params": 10}),
        ]
    )
    assert parse_nested_results(stdout) == {"best_val": 1.0, "steps": 3, "n_params": 10}


def test_parse_nested_results_handles_empty_and_malformed_input():
    assert parse_nested_results("") == {}
    assert parse_nested_results("{not json}") == {}
    assert parse_nested_results('{"frontier_ai_nested_results": "not a dict"}') == {}


# ---------------------------------------------------------------------------
# bare run: its own record, no published line
# ---------------------------------------------------------------------------
def test_bare_train_run_records_itself_and_publishes_nothing(tmp_path):
    out = tmp_path / "bare"
    result = _run(
        "train.py",
        "--config",
        "configs/cpu_smoke.json",
        "--set",
        f"train.out_dir={out}",
        "--max-steps",
        "3",
        "--exp-id",
        "EXP-940",
    )
    assert result.returncode == 0, result.stderr
    assert _records(out) == [out / RECORD]
    assert NESTED_RESULTS_MARKER not in result.stdout  # nothing to publish: it owns the record
    record = ExperimentRecord.load(out / RECORD)
    assert record.results["best_val"] is not None


# ---------------------------------------------------------------------------
# wrapped run: one record, outer record carries the training metrics
# ---------------------------------------------------------------------------
def test_wrapped_train_run_merges_training_metrics_into_the_single_record(tmp_path):
    wrapped = tmp_path / "wrapped"
    inner = tmp_path / "inner"
    result = _run(
        "experiment_record.py",
        "--exp-id",
        "EXP-941",
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

    # one record only, and it is the wrapper's
    assert _records(wrapped) == [wrapped / RECORD]
    assert _records(inner) == []

    record = ExperimentRecord.load(wrapped / RECORD)
    assert record.experiment["status"] == "success"
    assert record.results["exit_code"] == 0
    # the structured training metrics survived the trip
    for key in ("best_val", "best_bpb", "steps", "n_params", "tokens_seen"):
        assert key in record.results, f"{key} missing from the outer record"
    assert record.results["steps"] == 3
    # the wrapper's own keys were not overwritten by the merge
    assert record.results["stdout_tail"]
    assert "nested inside" in "\n".join(record.results.get("stderr_tail", []))


def test_wrapped_failure_keeps_one_failed_record_and_no_inner_record(tmp_path):
    wrapped = tmp_path / "wrapped-failed"
    inner = tmp_path / "inner-failed"
    result = _run(
        "experiment_record.py",
        "--exp-id",
        "EXP-942",
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
        "--set",
        "model.n_head=7",
        "--max-steps",
        "3",
    )
    assert result.returncode != 0
    assert _records(wrapped) == [wrapped / RECORD]
    assert _records(inner) == []

    record = ExperimentRecord.load(wrapped / RECORD)
    assert record.experiment["status"] == "failed"
    assert record.execution["error"]["type"] == "CalledProcessError"


# ---------------------------------------------------------------------------
# swept run: the whole point - best_val must aggregate
# ---------------------------------------------------------------------------
def test_train_sweep_aggregates_best_val_from_nested_runs(tmp_path):
    out = tmp_path / "sweep"
    runs = tmp_path / "run"          # one training directory per seed: run-1, run-2
    result = _run(
        "experiment_sweep.py",
        "--exp-id",
        "EXP-943",
        "--seeds",
        "1,2",
        "--metric",
        "best_val",
        "--data",
        "data/synthetic.bin",
        "--out",
        str(out),
        "--",
        sys.executable,
        "scripts/train.py",
        "--config",
        "configs/cpu_smoke.json",
        "--set",
        "train.seed={seed}",
        "--set",
        f"train.out_dir={runs}-{{seed}}",
        "--max-steps",
        "3",
    )
    assert result.returncode == 0, result.stderr

    sweep = json.loads((out / "sweep.json").read_text())
    assert sweep["sweep"]["status"] == "success", sweep["sweep"]
    statistics = sweep["statistics"]
    assert statistics["n"] == 2
    assert isinstance(statistics["mean"], float)
    assert statistics["spread"] is not None
    # the metric came from the run results, not from scraping the logs
    assert {run["metric_source"] for run in sweep["runs"]} == {"results"}
    assert all(isinstance(run["metric_value"], float) for run in sweep["runs"])
    # one record per seed, inside the sweep directory - and none in the training
    # directories themselves, because the sweep owns the records (D-032)
    assert _records(out) == [out / "seed-0000000001" / RECORD, out / "seed-0000000002" / RECORD]
    assert _records(runs.parent) == _records(out)
    for seed in (1, 2):
        assert _records(tmp_path / f"run-{seed}") == []


# ---------------------------------------------------------------------------
# tokenizer scripts behave the same way
# ---------------------------------------------------------------------------
def test_wrapped_tokenizer_run_merges_metrics_into_one_record(tmp_path):
    corpus = tmp_path / "corpus"
    prepared = _run(
        "tokenizer_prepare_corpus.py",
        "--out",
        str(corpus),
        "--sentences-per-language",
        "2",
        "--english-chars",
        "800",
    )
    assert prepared.returncode == 0, prepared.stderr

    wrapped = tmp_path / "wrapped-tokenizer"
    artifact = tmp_path / "artifact"
    result = _run(
        "experiment_record.py",
        "--exp-id",
        "EXP-944",
        "--out",
        str(wrapped),
        "--",
        sys.executable,
        "scripts/tokenizer_train.py",
        "--corpus",
        str(corpus),
        "--impl",
        "char",
        "--out",
        str(artifact),
    )
    assert result.returncode == 0, result.stderr

    assert _records(wrapped) == [wrapped / RECORD]
    assert _records(artifact) == []
    record = ExperimentRecord.load(wrapped / RECORD)
    assert record.results["impl"] == "char"
    assert record.results["vocab_size"] > 0
    # the round-trip flag is a real measurement, not a constant: on this 800-character
    # probe corpus the char tokenizer has not seen every character of the Hindi sample
    assert isinstance(record.results["sample_round_trip_ok"], bool)
    assert record.results["exit_code"] == 0
    # the tokenizer artifact itself is untouched by any of this
    assert (artifact / "manifest.json").exists()
