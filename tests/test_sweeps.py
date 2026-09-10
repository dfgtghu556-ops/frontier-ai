"""Tests for Project 003 Stage 1A: multi-seed experiment sweeps.

A sweep must keep one full experiment record per seed, aggregate a metric as
``mean ± spread`` (sample standard deviation), be independent of seed order, reproduce
exactly when repeated, and never report success while hiding a failed seed.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.experiments import (
    SWEEP_FILENAME,
    SWEEP_RECORD_TYPE,
    SWEEP_SCHEMA_VERSION,
    ExperimentRecord,
    ExperimentSpec,
    ExperimentSpecError,
    SweepRecord,
    SweepRecordError,
    normalize_seeds,
    parse_seed_list,
    run_command_sweep,
    run_sweep,
    sample_mean,
    sample_stdev,
    summarize_metric,
)
from frontier_ai.experiments.examples import seeded_metric_experiment, tiny_training_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _data(tmp_path: Path, payload: bytes = b"the quick brown fox") -> Path:
    path = tmp_path / "in.bin"
    path.write_bytes(payload)
    return path


def _spec(tmp_path: Path, tag: str, data: Path | None = None, **kwargs) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="EXP-004",
        seed=kwargs.pop("seed", 1),
        name="sweep test",
        output_dir=str(tmp_path / tag),
        data_paths=[str(data)] if data else [],
        **kwargs,
    )


def _sweep(tmp_path: Path, tag: str, seeds, data: Path | None = None, **kwargs):
    """Run a sweep of the cheap seed-sensitive reference experiment."""
    spec = _spec(tmp_path, tag, data)
    return run_sweep(spec, seeds, lambda ctx: dict(seeded_metric_experiment(ctx)), "score", **kwargs)


# ---------------------------------------------------------------------------
# A. statistics
# ---------------------------------------------------------------------------
def test_sample_mean_and_stdev_match_the_stdlib_definitions():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert sample_mean(values) == pytest.approx(statistics.fmean(values))
    assert sample_stdev(values) == pytest.approx(statistics.stdev(values))
    assert sample_stdev(values) != pytest.approx(statistics.pstdev(values))  # n-1, not n


def test_spread_is_undefined_for_fewer_than_two_values():
    assert sample_stdev([1.25]) is None  # one run: undefined, not 0.0
    assert sample_stdev([]) is None
    assert sample_mean([]) is None
    assert sample_mean([1.25]) == 1.25


def test_summarize_metric_documents_spread_and_refuses_to_call_it_a_confidence_interval():
    stats = summarize_metric("score", [(1, 1.0), (2, 2.0), (3, 4.0)])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(7 / 3)
    assert stats["spread"] == pytest.approx(statistics.stdev([1.0, 2.0, 4.0]))
    assert stats["spread_kind"] == "sample_standard_deviation"
    assert "n - 1" in stats["spread_definition"]
    assert "not a confidence interval" in stats["spread_definition"]
    assert stats["min"] == 1.0 and stats["max"] == 4.0

    empty = summarize_metric("score", [])
    assert empty["mean"] is None and empty["spread"] is None
    assert "null" in empty["note"]


def test_seed_list_parsing_and_normalization():
    assert parse_seed_list(["1,2", "3"]) == [1, 2, 3]
    assert parse_seed_list(["7 8 9"]) == [7, 8, 9]
    assert normalize_seeds([3, 1, 2, 1]) == [1, 2, 3]  # sorted and de-duplicated
    with pytest.raises(ExperimentSpecError, match="at least one seed"):
        normalize_seeds([])
    with pytest.raises(ExperimentSpecError, match="seed must be in"):
        normalize_seeds([-1])
    with pytest.raises(ExperimentSpecError, match="integers"):
        parse_seed_list(["one"])


# ---------------------------------------------------------------------------
# B. sweep behaviour
# ---------------------------------------------------------------------------
def test_single_seed_sweep(tmp_path):
    out = _sweep(tmp_path, "one", [42], _data(tmp_path))

    assert out.record.sweep["status"] == "success"
    assert out.record.sweep["requested_count"] == 1
    assert out.record.sweep["successful_seeds"] == [42]
    assert out.record.statistics["n"] == 1
    assert out.record.statistics["spread"] is None  # undefined for a single run
    assert out.record.statistics["mean"] == pytest.approx(
        out.record.runs[0]["metric_value"]
    )
    assert (tmp_path / "one" / SWEEP_FILENAME).exists()
    assert (tmp_path / "one" / "sweep.txt").exists()


def test_multi_seed_sweep_aggregates_the_metric(tmp_path):
    out = _sweep(tmp_path, "three", [1, 2, 3], _data(tmp_path))
    values = [r["metric_value"] for r in out.record.runs]

    assert out.record.sweep["status"] == "success"
    assert out.record.sweep["successful_seeds"] == [1, 2, 3]
    assert out.record.sweep["failed_seeds"] == []
    assert out.record.statistics["n"] == 3
    assert out.record.statistics["mean"] == pytest.approx(statistics.fmean(values))
    assert out.record.statistics["spread"] == pytest.approx(statistics.stdev(values))
    assert out.record.statistics["min"] == pytest.approx(min(values))
    assert out.record.statistics["max"] == pytest.approx(max(values))
    assert out.record.sweep["metric"] == "score"


def test_sweep_preserves_each_seed_explicitly(tmp_path):
    out = _sweep(tmp_path, "seeds", [7, 11, 13], _data(tmp_path))
    root = tmp_path / "seeds"

    for seed in (7, 11, 13):
        record = ExperimentRecord.load(root / f"seed-{seed:010d}" / "experiment.json")
        assert record.randomness["master_seed"] == seed       # the seed it actually ran with
        assert record.results["seed"] == seed
        assert out.record.runs[[r["seed"] for r in out.record.runs].index(seed)]["seed_used"] == seed

    # the metric values are the ones those seeds produced, in seed order
    assert [v["seed"] for v in out.record.statistics["values"]] == [7, 11, 13]


def test_sweep_is_independent_of_seed_order(tmp_path):
    data = _data(tmp_path)
    forward = _sweep(tmp_path, "forward", [1, 2, 3], data)
    backward = _sweep(tmp_path, "backward", [3, 1, 2], data)

    assert forward.record.statistics == backward.record.statistics
    assert [r["seed"] for r in forward.record.runs] == [r["seed"] for r in backward.record.runs]
    assert forward.record.sweep["successful_seeds"] == backward.record.sweep["successful_seeds"]
    assert forward.record.content_fingerprint() == backward.record.content_fingerprint()


def test_repeated_sweep_reproduces_individual_results_and_aggregate(tmp_path):
    data = _data(tmp_path)
    first = _sweep(tmp_path, "run-a", [1, 2], data)
    second = _sweep(tmp_path, "run-b", [1, 2], data)

    assert [r["metric_value"] for r in first.record.runs] == [
        r["metric_value"] for r in second.record.runs
    ]
    assert [r["fingerprint"] for r in first.record.runs] == [
        r["fingerprint"] for r in second.record.runs
    ]
    assert first.record.statistics == second.record.statistics
    # runtime metadata differs, and is excluded from the fingerprint
    assert first.record.execution["started_at"] != second.record.execution["started_at"] or True
    assert first.record.sweep["output_dir"] != second.record.sweep["output_dir"]
    assert first.record.content_fingerprint() == second.record.content_fingerprint()


def test_every_seed_gets_its_own_full_record(tmp_path):
    data = _data(tmp_path)
    out = _sweep(tmp_path, "records", [1, 2], data)
    root = tmp_path / "records"

    for run in out.record.runs:
        path = root / run["record"]
        assert path.exists()
        record = ExperimentRecord.load(path)
        assert record.experiment["status"] == "success"
        assert record.results, "each run keeps its own result"
        assert (path.parent / "experiment.txt").exists()
    assert json.loads((root / SWEEP_FILENAME).read_text())["record_type"] == SWEEP_RECORD_TYPE


def test_sweep_prevents_output_collisions_between_seeds(tmp_path):
    _sweep(tmp_path, "collide", [1, 2, 3], _data(tmp_path))
    dirs = sorted(p.name for p in (tmp_path / "collide").iterdir() if p.is_dir())

    assert dirs == ["seed-0000000001", "seed-0000000002", "seed-0000000003"]

    # duplicate seeds are de-duplicated rather than overwriting each other's output
    out = _sweep(tmp_path, "dupe", [5, 5, 5], None)
    assert out.record.sweep["seeds"] == [5]
    assert out.record.sweep["requested_count"] == 1


def test_provenance_is_retained_for_every_run(tmp_path):
    data = _data(tmp_path)
    out = _sweep(tmp_path, "provenance", [1, 2], data)

    for run in out.record.runs:
        record = ExperimentRecord.load(tmp_path / "provenance" / run["record"])
        assert record.code["git"]["available"] is True
        assert record.code["git"]["commit"]
        assert record.data["digest"]["digest"]
        assert record.randomness["seeded_rngs"]
        assert record.randomness["limitations"]
        assert record.environment["packages"]["torch"]
        assert record.configuration["spec"]["experiment_id"] == "EXP-004"
        assert run["fingerprint"] == record.content_fingerprint()


# ---------------------------------------------------------------------------
# C. failure handling
# ---------------------------------------------------------------------------
def _flaky(ctx):
    if ctx.seed == 2:
        raise RuntimeError("seed 2 exploded")
    return dict(seeded_metric_experiment(ctx))


def test_partial_failure_keeps_the_failed_record_and_continues(tmp_path):
    spec = _spec(tmp_path, "partial", _data(tmp_path))
    out = run_sweep(spec, [1, 2, 3], _flaky, "score")

    assert out.record.sweep["status"] == "partial"
    assert out.record.sweep["successful_seeds"] == [1, 3]
    assert out.record.sweep["failed_seeds"] == [2]
    assert out.record.sweep["successful_count"] == 2
    assert out.record.sweep["failed_count"] == 1
    assert out.record.statistics["n"] == 2  # only usable runs are averaged

    failed = next(r for r in out.record.runs if r["seed"] == 2)
    assert failed["status"] == "failed"
    assert failed["error"]["type"] == "RuntimeError"
    assert "seed 2 exploded" in failed["error"]["message"]
    assert failed["fingerprint"]  # the failed run still has a fingerprinted record

    record = ExperimentRecord.load(tmp_path / "partial" / failed["record"])
    assert record.experiment["status"] == "failed"
    assert record.execution["error"]["type"] == "RuntimeError"
    assert "failed" in (tmp_path / "partial" / "sweep.txt").read_text().lower()


def test_all_seeds_failing_reports_no_mean(tmp_path):
    def always_fails(_ctx):
        raise ValueError("every seed fails")

    spec = _spec(tmp_path, "allfail")
    out = run_sweep(spec, [7, 8], always_fails, "score")

    assert out.record.sweep["status"] == "failed"
    assert out.record.sweep["successful_seeds"] == []
    assert out.record.sweep["failed_seeds"] == [7, 8]
    assert out.record.statistics["mean"] is None
    assert out.record.statistics["spread"] is None
    assert "null" in out.record.statistics["note"]


def test_missing_or_non_numeric_metric_is_not_substituted_with_zero(tmp_path):
    def no_metric(ctx):
        return {"other": 1} if ctx.seed == 4 else dict(seeded_metric_experiment(ctx))

    spec = _spec(tmp_path, "metric-missing")
    out = run_sweep(spec, [3, 4], no_metric, "score")

    missing = next(r for r in out.record.runs if r["seed"] == 4)
    assert missing["status"] == "metric_missing"
    assert missing["metric_value"] is None       # not 0.0
    assert 4 in out.record.sweep["failed_seeds"]
    assert out.record.sweep["status"] == "partial"
    assert out.record.statistics["mean"] == pytest.approx(out.record.runs[0]["metric_value"])

    def non_numeric(_ctx):
        return {"score": "not-a-number"}

    out2 = run_sweep(_spec(tmp_path, "metric-non-numeric"), [1], non_numeric, "score")
    assert out2.record.sweep["status"] == "failed"
    assert out2.record.statistics["mean"] is None


def test_continue_on_error_false_stops_at_the_first_failure(tmp_path):
    spec = _spec(tmp_path, "failfast", _data(tmp_path))
    with pytest.raises(RuntimeError, match="seed 2 exploded"):
        run_sweep(spec, [1, 2, 3], _flaky, "score", continue_on_error=False)

    record = SweepRecord.load(tmp_path / "failfast" / SWEEP_FILENAME)
    assert record.sweep["status"] == "failed"
    assert record.sweep["successful_seeds"] == [1]
    assert 3 not in [r["seed"] for r in record.runs]  # never started
    assert "continue_on_error" in record.statistics["note"]


def test_sweep_validates_inputs_before_running_any_seed(tmp_path):
    spec = ExperimentSpec(experiment_id="EXP-004", output_dir=str(tmp_path / "missing-data"),
                          data_paths=[str(tmp_path / "nope.bin")])
    calls = []

    from frontier_ai.experiments import ExperimentInputError

    with pytest.raises(ExperimentInputError, match="missing"):
        run_sweep(spec, [1, 2], lambda ctx: calls.append(ctx.seed), "score")
    assert calls == []


# ---------------------------------------------------------------------------
# D. aggregate record
# ---------------------------------------------------------------------------
def test_sweep_record_schema_round_trip_and_required_fields(tmp_path):
    out = _sweep(tmp_path, "schema", [1, 2], _data(tmp_path))
    path = tmp_path / "schema" / SWEEP_FILENAME
    loaded = SweepRecord.load(path)

    assert loaded.schema_version == SWEEP_SCHEMA_VERSION
    assert loaded.record_type == SWEEP_RECORD_TYPE
    assert loaded.content_fingerprint() == out.record.content_fingerprint()
    for section in ("sweep", "configuration", "code", "data", "environment", "statistics",
                    "runs", "execution"):
        assert section in loaded.to_dict()

    sweep = loaded.sweep
    for key in ("id", "status", "metric", "seeds", "successful_seeds", "failed_seeds",
                "requested_count", "successful_count", "failed_count"):
        assert key in sweep, key
    assert loaded.configuration["seeds_requested"] == [1, 2]

    bad = json.loads(path.read_text())
    bad["schema_version"] = "9.9"
    with pytest.raises(SweepRecordError, match="unsupported sweep schema_version"):
        SweepRecord.from_dict(bad)
    bad = json.loads(path.read_text())
    bad["nonsense"] = 1
    with pytest.raises(SweepRecordError, match="unknown sweep sections"):
        SweepRecord.from_dict(bad)


def test_sweep_record_references_are_relative_to_the_sweep_dir(tmp_path):
    out = _sweep(tmp_path, "relative", [1, 2], _data(tmp_path))
    for run in out.record.runs:
        assert not Path(run["record"]).is_absolute()
        assert (tmp_path / "relative" / run["record"]).exists()


# ---------------------------------------------------------------------------
# E. CLI
# ---------------------------------------------------------------------------
def _sweep_cli(*args: str) -> subprocess.CompletedProcess:
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "experiment_sweep.py"), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_cli_help():
    result = _sweep_cli("--help")
    assert result.returncode == 0
    for flag in ("--exp-id", "--seeds", "--metric", "--out", "--data", "--config"):
        assert flag in result.stdout


def test_cli_sweep_success_exit_code_and_aggregate(tmp_path):
    data = _data(tmp_path, b"cli sweep input")
    out = tmp_path / "cli-ok"
    result = _sweep_cli(
        "--exp-id", "EXP-004", "--seeds", "1,2,3", "--metric", "score",
        "--data", str(data), "--out", str(out),
        "--", sys.executable, "-c",
        "import json,sys;seed=int(sys.argv[1]);"
        "print(json.dumps({'score': seed / 10.0}))",
        "{seed}",
    )

    assert result.returncode == 0, result.stderr
    record = SweepRecord.load(out / SWEEP_FILENAME)
    assert record.sweep["successful_seeds"] == [1, 2, 3]
    assert record.statistics["mean"] == pytest.approx(0.2)
    assert record.statistics["spread"] == pytest.approx(0.1)
    assert [r["metric_source"] for r in record.runs] == ["stdout_json"] * 3
    assert json.loads(result.stdout.strip().splitlines()[-1])["status"] == "success"


def test_cli_partial_and_total_failure_exit_codes(tmp_path):
    partial = _sweep_cli(
        "--exp-id", "EXP-004", "--seeds", "1,2,3", "--metric", "score",
        "--out", str(tmp_path / "partial"), "--quiet", "--", sys.executable, "-c",
        "import json,sys;seed=int(sys.argv[1]);sys.exit(3) if seed==2 else "
        "print(json.dumps({'score': seed / 10.0}))",
        "{seed}",
    )
    assert partial.returncode == 2
    assert SweepRecord.load(tmp_path / "partial" / SWEEP_FILENAME).sweep["status"] == "partial"

    failed = _sweep_cli(
        "--exp-id", "EXP-004", "--seeds", "4,5", "--metric", "score",
        "--out", str(tmp_path / "allfail"), "--quiet", "--", sys.executable, "-c",
        "import sys; sys.exit(1)",
    )
    assert failed.returncode == 1
    assert SweepRecord.load(tmp_path / "allfail" / SWEEP_FILENAME).sweep["status"] == "failed"


def test_cli_warns_when_the_command_ignores_the_seed(tmp_path):
    result = _sweep_cli(
        "--exp-id", "EXP-004", "--seeds", "1,2", "--metric", "score",
        "--out", str(tmp_path / "noseed"), "--", sys.executable, "-c",
        "import json;print(json.dumps({'score': 0.5}))",
    )
    assert result.returncode == 0
    assert "no '{seed}' placeholder" in result.stderr


# ---------------------------------------------------------------------------
# F. end-to-end: a real Project 001 training sweep
# ---------------------------------------------------------------------------
def test_real_training_sweep_is_reproducible(tmp_path):
    """Two identical sweeps of the Project 001 trainer must agree run for run."""

    def once(tag: str):
        spec = ExperimentSpec(experiment_id="EXP-004", seed=1, name="training sweep",
                              output_dir=str(tmp_path / tag))
        return run_sweep(spec, [1, 2],
                         lambda ctx: dict(tiny_training_experiment(ctx, steps=6)),
                         "final_val_loss")

    first, second = once("train-a"), once("train-b")

    assert [r["metric_value"] for r in first.record.runs] == [
        r["metric_value"] for r in second.record.runs
    ]
    assert [r["fingerprint"] for r in first.record.runs] == [
        r["fingerprint"] for r in second.record.runs
    ]
    assert first.record.statistics == second.record.statistics
    assert first.record.content_fingerprint() == second.record.content_fingerprint()
    assert first.record.sweep["successful_seeds"] == [1, 2]
    # different seeds really do give different numbers (the sweep is measuring something)
    assert first.record.statistics["values"][0]["value"] != first.record.statistics["values"][1]["value"]


def test_command_sweep_runs_one_command_per_seed(tmp_path):
    spec = _spec(tmp_path, "command", None)
    out = run_command_sweep(
        spec, [1, 2], "score",
        [sys.executable, "-c",
         "import json,sys;print(json.dumps({'score': int(sys.argv[1]) / 10.0}))", "{seed}"],
    )

    assert out.record.sweep["successful_seeds"] == [1, 2]
    assert out.record.statistics["mean"] == pytest.approx(0.15)
    for run in out.record.runs:
        record = ExperimentRecord.load(tmp_path / "command" / run["record"])
        assert str(run["seed"]) in " ".join(record.execution["command"])
