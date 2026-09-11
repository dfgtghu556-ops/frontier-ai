"""Tests for Project 003 Stage 1B: multi-configuration sweeps.

A sweep may now run several *named configurations* across several seeds. These tests cover
the configuration × seed matrix, order independence, reproducibility, failure semantics,
schema/compatibility with the Stage 1A seed-only sweep, and the CLI.

`tests/test_sweeps.py` remains the Stage 1A suite and is the behavioural backward
compatibility check: it must keep passing unchanged.
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
    SWEEP_SCHEMA_VERSION,
    ExperimentRecord,
    ExperimentSpec,
    ExperimentSpecError,
    SweepRecord,
    normalize_configurations,
    parse_configuration_list,
    run_command_sweep,
    run_sweep,
)
from frontier_ai.experiments.examples import seeded_metric_experiment, tiny_training_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _data(tmp_path: Path, payload: bytes = b"the quick brown fox") -> Path:
    path = tmp_path / "in.bin"
    path.write_bytes(payload)
    return path


def _spec(tmp_path: Path, tag: str, data: Path | None = None) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="EXP-005",
        seed=1,
        name="config sweep",
        output_dir=str(tmp_path / tag),
        data_paths=[str(data)] if data else [],
    )


def _body(ctx):
    """Reference experiment whose score scales with the configuration's `scale` param."""
    scale = float(ctx.spec.params.get("scale", 1.0))
    out = dict(seeded_metric_experiment(ctx))
    out["score"] = round(out["score"] * scale, 6)
    out["configuration"] = ctx.configuration
    return out


def _sweep(tmp_path: Path, tag: str, seeds, configs=None, data: Path | None = None, **kwargs):
    spec = _spec(tmp_path, tag, data)
    return run_sweep(spec, seeds, _body, "score", configurations=configs, **kwargs)


TWO_CONFIGS = {"alpha": {"params": {"scale": 1.0}}, "beta": {"params": {"scale": 2.0}}}


def _without_paths(payload):
    """Drop output locations: they legitimately differ between two sweeps."""
    if isinstance(payload, dict):
        return {k: _without_paths(v) for k, v in payload.items() if k != "output_dir"}
    if isinstance(payload, list):
        return [_without_paths(v) for v in payload]
    return payload


# ---------------------------------------------------------------------------
# A. configuration × seed matrix
# ---------------------------------------------------------------------------
def test_one_configuration_across_multiple_seeds(tmp_path):
    out = _sweep(tmp_path, "one", [1, 2, 3], {"only": {"params": {"scale": 3.0}}},
                 _data(tmp_path))

    assert out.record.sweep["status"] == "success"
    assert out.record.sweep["configuration_names"] == ["only"]
    assert out.record.sweep["configuration_count"] == 1
    assert out.record.sweep["runs_requested"] == 3
    assert out.record.sweep["runs_successful"] == 3

    cfg = out.record.configurations[0]
    assert cfg["name"] == "only"
    assert cfg["status"] == "success"
    assert cfg["successful_seeds"] == [1, 2, 3]
    assert cfg["statistics"]["n"] == 3
    values = [v["value"] for v in cfg["statistics"]["values"]]
    assert cfg["statistics"]["mean"] == pytest.approx(statistics.fmean(values))
    assert cfg["statistics"]["spread"] == pytest.approx(statistics.stdev(values))


def test_multiple_configurations_across_multiple_seeds(tmp_path):
    out = _sweep(tmp_path, "two", [1, 2, 3], TWO_CONFIGS, _data(tmp_path))

    assert out.record.sweep["configuration_names"] == ["alpha", "beta"]
    assert out.record.sweep["runs_requested"] == 6
    assert out.record.sweep["runs_successful"] == 6
    assert len(out.record.runs) == 6

    by_name = {cfg["name"]: cfg for cfg in out.record.configurations}
    alpha = [v["value"] for v in by_name["alpha"]["statistics"]["values"]]
    beta = [v["value"] for v in by_name["beta"]["statistics"]["values"]]

    assert by_name["alpha"]["statistics"]["mean"] == pytest.approx(statistics.fmean(alpha))
    assert by_name["beta"]["statistics"]["mean"] == pytest.approx(statistics.fmean(beta))
    # the configuration actually changed the experiment (beta scales the score by 2)
    assert by_name["beta"]["statistics"]["mean"] == pytest.approx(
        2 * by_name["alpha"]["statistics"]["mean"], rel=1e-6
    )
    assert by_name["beta"]["statistics"]["spread"] > by_name["alpha"]["statistics"]["spread"]


def test_configuration_reaches_the_experiment_body(tmp_path):
    out = _sweep(tmp_path, "body", [7], TWO_CONFIGS)

    for run in out.record.runs:
        record = ExperimentRecord.load(tmp_path / "body" / run["record"])
        assert record.results["configuration"] == run["configuration"]
        assert f"config:{run['configuration']}" in record.experiment["tags"]
        assert record.configuration["spec"]["params"]["scale"] == (
            1.0 if run["configuration"] == "alpha" else 2.0
        )
        # the named configuration's spec is recorded with the run, not just the template
        assert record.experiment["status"] == "success"


def test_configuration_may_override_name_notes_and_config_path(tmp_path):
    configs = {"variant": {"params": {"scale": 1.5}, "name": "variant name",
                           "notes": "variant notes", "config_path": "configs/cpu_smoke.json"}}
    out = _sweep(tmp_path, "override", [1], configs)

    cfg = out.record.configurations[0]
    assert cfg["spec"]["name"] == "variant name"
    assert cfg["spec"]["notes"] == "variant notes"
    assert cfg["spec"]["config_path"] == "configs/cpu_smoke.json"
    assert cfg["spec"]["params"]["scale"] == 1.5
    record = ExperimentRecord.load(tmp_path / "override" / out.record.runs[0]["record"])
    assert record.experiment["name"] == "variant name"


# ---------------------------------------------------------------------------
# B. order independence and reproducibility
# ---------------------------------------------------------------------------
def test_configuration_order_does_not_matter(tmp_path):
    data = _data(tmp_path)
    forward = _sweep(tmp_path, "fwd", [1, 2, 3], TWO_CONFIGS, data)
    backward = _sweep(tmp_path, "rev", [1, 2, 3], dict(reversed(list(TWO_CONFIGS.items()))), data)

    assert [c["name"] for c in forward.record.configurations] == [
        c["name"] for c in backward.record.configurations
    ]
    assert _without_paths(forward.record.configurations) == _without_paths(
        backward.record.configurations
    )
    assert _without_paths(forward.record.sweep) == _without_paths(backward.record.sweep)
    assert forward.record.statistics == backward.record.statistics
    assert forward.record.content_fingerprint() == backward.record.content_fingerprint()


def test_seed_order_does_not_matter_with_configurations(tmp_path):
    data = _data(tmp_path)
    forward = _sweep(tmp_path, "seeds-fwd", [1, 2, 3], TWO_CONFIGS, data)
    backward = _sweep(tmp_path, "seeds-rev", [3, 1, 2], TWO_CONFIGS, data)

    assert forward.record.statistics == backward.record.statistics
    assert _without_paths(forward.record.configurations) == _without_paths(
        backward.record.configurations
    )
    assert forward.record.content_fingerprint() == backward.record.content_fingerprint()


def test_configuration_and_seed_order_together_do_not_matter(tmp_path):
    """The CLI hands over the raw (un-normalised) seed list; the record must not care."""
    data = _data(tmp_path)
    # spec.seed is the first listed seed on purpose: the template must canonicalise it
    forward = run_sweep(
        ExperimentSpec(experiment_id="EXP-005", seed=1, output_dir=str(tmp_path / "f"),
                       data_paths=[str(data)]),
        [1, 2, 3], _body, "score", configurations=TWO_CONFIGS,
    )
    reversed_ = run_sweep(
        ExperimentSpec(experiment_id="EXP-005", seed=3, output_dir=str(tmp_path / "r"),
                       data_paths=[str(data)]),
        [3, 2, 1], _body, "score",
        configurations=dict(reversed(list(TWO_CONFIGS.items()))),
    )

    assert _without_paths(forward.record.configurations) == _without_paths(
        reversed_.record.configurations
    )
    assert forward.record.content_fingerprint() == reversed_.record.content_fingerprint()


def test_repeated_multi_configuration_sweep_reproduces(tmp_path):
    data = _data(tmp_path)
    first = _sweep(tmp_path, "rep-a", [1, 2], TWO_CONFIGS, data)
    second = _sweep(tmp_path, "rep-b", [1, 2], TWO_CONFIGS, data)

    assert [r["fingerprint"] for r in first.record.runs] == [
        r["fingerprint"] for r in second.record.runs
    ]
    assert [r["metric_value"] for r in first.record.runs] == [
        r["metric_value"] for r in second.record.runs
    ]
    assert _without_paths(first.record.configurations) == _without_paths(
        second.record.configurations
    )
    # runtime metadata differs and is excluded from the fingerprint
    assert first.record.sweep["output_dir"] != second.record.sweep["output_dir"]
    assert first.record.content_fingerprint() == second.record.content_fingerprint()


# ---------------------------------------------------------------------------
# C. layout and provenance
# ---------------------------------------------------------------------------
def test_every_configuration_and_seed_gets_a_unique_directory(tmp_path):
    out = _sweep(tmp_path, "dirs", [1, 2, 3], TWO_CONFIGS, _data(tmp_path))
    root = tmp_path / "dirs"

    dirs = sorted(p.relative_to(root).as_posix() for p in root.rglob("experiment.json"))
    assert dirs == [f"alpha/seed-000000000{i}/experiment.json" for i in (1, 2, 3)] + \
                   [f"beta/seed-000000000{i}/experiment.json" for i in (1, 2, 3)]
    assert all((root / run["record"]).exists() for run in out.record.runs)
    assert not any(Path(run["record"]).is_absolute() for run in out.record.runs)


def test_provenance_is_retained_for_every_configuration_and_seed(tmp_path):
    out = _sweep(tmp_path, "prov", [1, 2], TWO_CONFIGS, _data(tmp_path))

    for run in out.record.runs:
        record = ExperimentRecord.load(tmp_path / "prov" / run["record"])
        assert record.code["git"]["available"] is True
        assert record.code["git"]["commit"]
        assert record.data["digest"]["digest"]
        assert record.randomness["master_seed"] == run["seed_used"]
        assert record.randomness["limitations"]
        assert record.environment["packages"]["torch"]
        assert record.configuration["spec"]["experiment_id"] == "EXP-005"
        assert (tmp_path / "prov" / Path(run["record"]).parent / "experiment.txt").exists()


def test_results_from_different_configurations_are_never_mixed(tmp_path):
    out = _sweep(tmp_path, "mix", [1, 2, 3], TWO_CONFIGS, _data(tmp_path))

    # the top-level statistics section explicitly refuses to aggregate across configurations
    assert out.record.statistics["aggregated"] is False
    assert "mean" not in out.record.statistics, "no cross-configuration mean may be published"
    assert "spread" not in out.record.statistics
    assert "never mixed" in out.record.statistics["note"]

    alpha_mean = out.record.configurations[0]["statistics"]["mean"]
    beta_mean = out.record.configurations[1]["statistics"]["mean"]
    assert abs(beta_mean - 2 * alpha_mean) < 1e-6  # each configuration keeps its own mean
    assert "seed_status_note" in out.record.sweep


# ---------------------------------------------------------------------------
# D. failure semantics
# ---------------------------------------------------------------------------
def _flakey_body(ctx):
    if ctx.configuration == "broken":
        raise RuntimeError("configuration broken")
    if ctx.configuration == "nomeas":
        return {"other": 1}
    if ctx.configuration == "text":
        return {"score": "not-a-number"}
    return _body(ctx)


def test_partial_failure_keeps_failed_records_and_continues(tmp_path):
    out = run_sweep(_spec(tmp_path, "partial", _data(tmp_path)), [1, 2], _flakey_body, "score",
                    configurations={"ok": {"params": {"scale": 1.0}}, "broken": {}})

    assert out.record.sweep["status"] == "partial"
    assert out.record.sweep["runs_successful"] == 2
    assert out.record.sweep["runs_requested"] == 4

    by_name = {cfg["name"]: cfg for cfg in out.record.configurations}
    assert by_name["ok"]["status"] == "success"
    assert by_name["broken"]["status"] == "failed"
    assert by_name["broken"]["statistics"]["mean"] is None
    assert by_name["broken"]["failed_seeds"] == [1, 2]

    failed = [r for r in out.record.runs if r["configuration"] == "broken"]
    assert all(r["status"] == "failed" for r in failed)
    assert all(r["error"]["type"] == "RuntimeError" for r in failed)
    assert all(r["metric_value"] is None for r in failed)  # never zero
    for run in failed:
        record = ExperimentRecord.load(tmp_path / "partial" / run["record"])
        assert record.experiment["status"] == "failed"
    assert "partial" in (tmp_path / "partial" / "sweep.txt").read_text().lower()


def test_complete_failure_reports_no_mean_anywhere(tmp_path):
    def always_fails(_ctx):
        raise ValueError("every configuration fails")

    out = run_sweep(_spec(tmp_path, "allfail"), [1, 2], always_fails, "score",
                    configurations={"a": {}, "b": {}})

    assert out.record.sweep["status"] == "failed"
    assert out.record.sweep["runs_successful"] == 0
    assert out.record.statistics["aggregated"] is False
    for cfg in out.record.configurations:
        assert cfg["status"] == "failed"
        assert cfg["statistics"]["mean"] is None
        assert cfg["statistics"]["spread"] is None
        assert "null" in cfg["statistics"]["note"]


def test_missing_and_non_numeric_metrics_are_not_substituted(tmp_path):
    out = run_sweep(_spec(tmp_path, "metric"), [1, 2], _flakey_body, "score",
                    configurations={"nomeas": {}, "text": {}, "ok": {"params": {"scale": 1.0}}})

    by_name = {cfg["name"]: cfg for cfg in out.record.configurations}
    for name in ("nomeas", "text"):
        assert by_name[name]["status"] == "failed"
        assert by_name[name]["statistics"]["n"] == 0
        runs = [r for r in out.record.runs if r["configuration"] == name]
        assert all(r["status"] == "metric_missing" for r in runs)
        assert all(r["metric_value"] is None for r in runs)
    assert by_name["ok"]["status"] == "success"


def test_continue_on_error_false_stops_the_whole_sweep(tmp_path):
    with pytest.raises(RuntimeError, match="configuration broken"):
        run_sweep(_spec(tmp_path, "failfast"), [1, 2], _flakey_body, "score",
                  configurations={"broken": {}, "ok": {}}, continue_on_error=False)

    record = SweepRecord.load(tmp_path / "failfast" / SWEEP_FILENAME)
    assert record.sweep["status"] == "failed"
    assert not any(r["configuration"] == "ok" for r in record.runs)  # never started
    assert "continue_on_error" in record.statistics["note"]
    assert all("continue_on_error" in cfg["statistics"]["note"] for cfg in record.configurations)


# ---------------------------------------------------------------------------
# E. schema and backward compatibility
# ---------------------------------------------------------------------------
def test_sweep_schema_version_and_sections(tmp_path):
    _sweep(tmp_path, "schema", [1, 2], TWO_CONFIGS, _data(tmp_path))
    payload = json.loads((tmp_path / "schema" / SWEEP_FILENAME).read_text())

    assert payload["schema_version"] == SWEEP_SCHEMA_VERSION
    assert payload["record_type"] == "frontier-ai.experiment-sweep"
    for section in ("sweep", "configuration", "code", "data", "environment", "statistics",
                    "runs", "execution", "configurations"):
        assert section in payload
    for cfg in payload["configurations"]:
        for key in ("name", "status", "metric", "seeds", "successful_seeds", "failed_seeds",
                    "requested_count", "successful_count", "failed_count", "spec",
                    "overrides", "statistics"):
            assert key in cfg
    with pytest.raises(Exception, match="unsupported sweep schema_version"):
        SweepRecord.from_dict({**payload, "schema_version": "9.9"})


def test_seed_only_sweep_keeps_the_stage_1a_record_shape(tmp_path):
    """Backward compatibility: no named configurations ⇒ no new fields, Stage 1A layout."""
    out = run_sweep(_spec(tmp_path, "solo", _data(tmp_path)), [1, 2], _body, "score")
    payload = json.loads((tmp_path / "solo" / SWEEP_FILENAME).read_text())

    assert "configurations" not in payload                       # no empty extra section
    assert "configuration_names" not in payload["sweep"]
    assert "runs_requested" not in payload["sweep"]
    assert all("configuration" not in run for run in payload["runs"])
    assert payload["statistics"]["mean"] is not None             # aggregated as before
    assert "aggregated" not in payload["statistics"]
    assert sorted(p.relative_to(tmp_path / "solo").as_posix()
                  for p in (tmp_path / "solo").rglob("experiment.json")) == [
        "seed-0000000001/experiment.json", "seed-0000000002/experiment.json",
    ]
    assert out.record.sweep["status"] == "success"


def test_seed_only_and_single_named_configuration_are_distinguished(tmp_path):
    solo = run_sweep(_spec(tmp_path, "solo2", _data(tmp_path)), [1], _body, "score")
    named = run_sweep(_spec(tmp_path, "named", _data(tmp_path)), [1], _body, "score",
                      configurations={"only": {"params": {"scale": 1.0}}})

    assert "configurations" not in solo.record.to_dict()
    assert "configurations" in named.record.to_dict()
    assert named.record.configurations[0]["name"] == "only"


def test_configuration_validation(tmp_path):
    spec = _spec(tmp_path, "validation")
    with pytest.raises(ExperimentSpecError, match="invalid configuration name"):
        run_sweep(spec, [1], _body, "score", configurations={"bad name!": {}})
    with pytest.raises(ExperimentSpecError, match="unsupported keys"):
        run_sweep(spec, [1], _body, "score", configurations={"a": {"seed": 5}})
    with pytest.raises(ExperimentSpecError, match="must be a mapping"):
        run_sweep(spec, [1], _body, "score", configurations={"a": ["not", "a", "mapping"]})
    with pytest.raises(ExperimentSpecError, match="name of the result field"):
        run_sweep(spec, [1], _body, "   ", configurations={"a": {}})


def test_configuration_parsing_helpers():
    parsed = parse_configuration_list(["lr_high:params.lr=0.02",
                                       "low:params.lr=0.005,name=low,notes=slow"])
    assert parsed["lr_high"]["params"] == {"lr": 0.02}
    assert parsed["low"]["params"] == {"lr": 0.005}
    assert parsed["low"]["name"] == "low"
    assert parsed["low"]["notes"] == "slow"

    for bad in ("no-colon", "bad name:x=1", "a:seed=3", "a:params.lr"):
        with pytest.raises(ExperimentSpecError):
            parse_configuration_list([bad])

    configs = normalize_configurations(
        ExperimentSpec(experiment_id="EXP-005", seed=1, params={"lr": 0.01, "keep": "me"}),
        {"b": {"params": {"lr": 0.2}}, "a": {"params": {"lr": 0.1}}},
    )
    assert [c.name for c in configs] == ["a", "b"]  # sorted: order-independent
    assert configs[0].spec.params == {"lr": 0.1, "keep": "me"}  # merged, not replaced


# ---------------------------------------------------------------------------
# F. CLI
# ---------------------------------------------------------------------------
def _sweep_cli(*args: str) -> subprocess.CompletedProcess:
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "experiment_sweep.py"), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


PRINT_SCORE = (
    "import json,sys;cfg=sys.argv[1];seed=int(sys.argv[2]);"
    "scale=2.0 if cfg=='beta' else 1.0;"
    "print(json.dumps({'score': round(seed/10.0*scale, 6), 'configuration': cfg}))"
)


def test_cli_help_mentions_configurations():
    result = _sweep_cli("--help")
    assert result.returncode == 0
    assert "--configs" in result.stdout
    assert "{config}" in result.stdout


def test_cli_multi_configuration_sweep(tmp_path):
    out = tmp_path / "cli"
    result = _sweep_cli(
        "--exp-id", "EXP-005", "--seeds", "1,2,3",
        "--configs", "alpha:params.scale=1.0", "beta:params.scale=2.0",
        "--metric", "score", "--data", str(_data(tmp_path, b"cli config input")),
        "--out", str(out), "--", sys.executable, "-c", PRINT_SCORE, "{config}", "{seed}",
    )

    assert result.returncode == 0, result.stderr
    record = SweepRecord.load(out / SWEEP_FILENAME)
    assert record.sweep["configuration_names"] == ["alpha", "beta"]
    assert record.sweep["runs_successful"] == 6

    alpha = next(c for c in record.configurations if c["name"] == "alpha")
    beta = next(c for c in record.configurations if c["name"] == "beta")
    assert alpha["statistics"]["mean"] == pytest.approx(0.2)
    assert beta["statistics"]["mean"] == pytest.approx(0.4)

    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["status"] == "success"
    assert [c["name"] for c in summary["configurations"]] == ["alpha", "beta"]
    assert summary["configurations"][0]["mean"] == pytest.approx(0.2)

    # the {config} placeholder really reached the command
    for run in record.runs:
        record_path = out / run["record"]
        experiment = ExperimentRecord.load(record_path)
        assert run["configuration"] in " ".join(experiment.execution["command"])


def test_cli_partial_failure_exit_code_with_configurations(tmp_path):
    out = tmp_path / "cli-partial"
    result = _sweep_cli(
        "--exp-id", "EXP-005", "--seeds", "1,2",
        "--configs", "ok:params.scale=1.0", "broken:params.scale=1.0",
        "--metric", "score", "--out", str(out), "--quiet", "--", sys.executable, "-c",
        "import sys,json;cfg=sys.argv[1];seed=int(sys.argv[2]);"
        "sys.exit(3) if cfg=='broken' else print(json.dumps({'score': seed/10.0}))",
        "{config}", "{seed}",
    )
    assert result.returncode == 2
    record = SweepRecord.load(out / SWEEP_FILENAME)
    assert record.sweep["status"] == "partial"
    by_name = {c["name"]: c for c in record.configurations}
    assert by_name["ok"]["status"] == "success"
    assert by_name["broken"]["status"] == "failed"


def test_cli_warns_when_the_command_ignores_the_configuration(tmp_path):
    result = _sweep_cli(
        "--exp-id", "EXP-005", "--seeds", "1", "--configs", "a:params.scale=1.0",
        "b:params.scale=2.0", "--metric", "score", "--out", str(tmp_path / "noconfig"),
        "--", sys.executable, "-c", "import json;print(json.dumps({'score': 0.5}))",
    )
    assert result.returncode == 0
    assert "no '{config}' placeholder" in result.stderr


def test_command_sweep_substitutes_both_placeholders(tmp_path):
    out = run_command_sweep(
        _spec(tmp_path, "cmd", _data(tmp_path)), [1, 2], "score",
        [sys.executable, "-c", PRINT_SCORE, "{config}", "{seed}"],
        configurations=TWO_CONFIGS,
    )
    assert out.record.sweep["runs_successful"] == 4
    for run in out.record.runs:
        experiment = ExperimentRecord.load(tmp_path / "cmd" / run["record"])
        argv = experiment.execution["command"]
        assert run["configuration"] in argv and str(run["seed"]) in argv


# ---------------------------------------------------------------------------
# G. end-to-end: a real Project 001 training sweep over two configurations
# ---------------------------------------------------------------------------
def test_real_training_sweep_over_two_configurations(tmp_path):
    """Two learning rates × two seeds of the real trainer: reproducible, and different."""
    configs = {"lr_low": {"params": {"lr": 0.005}}, "lr_high": {"params": {"lr": 0.05}}}

    def once(tag: str):
        spec = ExperimentSpec(experiment_id="EXP-005", seed=1, name="training config sweep",
                              output_dir=str(tmp_path / tag))
        return run_sweep(spec, [1, 2],
                         lambda ctx: dict(tiny_training_experiment(ctx, steps=8)),
                         "final_val_loss", configurations=configs)

    first, second = once("train-a"), once("train-b")

    assert [r["fingerprint"] for r in first.record.runs] == [
        r["fingerprint"] for r in second.record.runs
    ]
    assert _without_paths(first.record.configurations) == _without_paths(
        second.record.configurations
    )
    assert first.record.content_fingerprint() == second.record.content_fingerprint()

    by_name = {c["name"]: c for c in first.record.configurations}
    assert by_name["lr_low"]["status"] == "success"
    assert by_name["lr_high"]["status"] == "success"
    assert by_name["lr_low"]["statistics"]["mean"] != by_name["lr_high"]["statistics"]["mean"]
    assert by_name["lr_low"]["statistics"]["spread"] is not None
    assert first.record.sweep["status"] == "success"
