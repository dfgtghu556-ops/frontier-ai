"""Multi-seed experiment sweeps (Project 003, ROADMAP Stage 1).

A sweep answers one question: **how much does the measured result vary across independent
seeds?** One experiment specification is executed once per seed, and the results are
summarised as ``mean ± spread``.

Design rules (D-028)
--------------------
Metric resolution: ``metric`` is read from the run record's ``results`` first, and — for
wrapped commands, which only record ``exit_code`` and log tails — from the last JSON object
the command printed (the convention ``scripts/train.py --eval-only`` and
``scripts/evaluate.py`` already use). The source actually used is recorded per run as
``metric_source``.

Design rules (D-028)
--------------------
1. **Individual runs are never collapsed.** Every seed produces its own normal
   :class:`~frontier_ai.experiments.record.ExperimentRecord` — same sections, same git/data/
   environment provenance, same content fingerprint — written to its own directory
   (``seed-<zero-padded seed>/``) so seeds cannot collide.
2. **Seed order does not matter.** The seed list is normalised (ints, de-duplicated, sorted
   ascending) before anything runs, and statistics are computed over values sorted by seed,
   so ``[1, 2, 3]`` and ``[3, 1, 2]`` produce the same aggregate. Each run still records the
   seed it actually used.
3. **Failures are recorded, not discarded.** A seed that raises keeps its failed record, is
   listed in ``failed_seeds`` with its error, and does not stop the other seeds. The sweep
   status is ``success`` (all seeds), ``partial`` (some seeds) or ``failed`` (no seed
   produced a usable metric). It never reports success when a seed is missing.
4. **No silent substitution.** A run that succeeded but did not produce a *numeric* value
   for the requested metric is counted as failed for aggregation; missing values are never
   replaced by zero.
5. **Spread is the sample standard deviation** (``n - 1`` denominator), undefined — and
   explicitly ``null`` — for fewer than two successful runs. It is not a confidence
   interval and is never called one.

Reuses, does not reimplement: the runner lifecycle (:mod:`.runner`), the record format and
fingerprint (:mod:`.record`), data hashing (:mod:`.hashing`) and seeding (:mod:`.seeding`).
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .environment import capture_environment
from .gitinfo import capture_git_info, git_section
from .record import RECORD_FILENAME, ExperimentRecord, content_fingerprint_of
from .runner import (
    ExperimentContext,
    RunOutcome,
    configuration_section,
    data_section,
    run_command,
    run_experiment,
    validate_inputs,
)
from .seeding import DEFAULT_COMPONENTS
from .spec import SEED_MAX, SEED_MIN, ExperimentSpec, ExperimentSpecError

SWEEP_SCHEMA_VERSION = "1.0"
SWEEP_RECORD_TYPE = "frontier-ai.experiment-sweep"
SWEEP_FILENAME = "sweep.json"

SAMPLE_STDEV_DEFINITION = (
    "sample standard deviation: sqrt(sum((x_i - mean)^2) / (n - 1)) over the successful "
    "runs; undefined (null) for n < 2. It is a dispersion estimate, not a confidence "
    "interval: no interval is computed or implied."
)

SEED_DIR_TEMPLATE = "seed-{seed:010d}"
SEED_PLACEHOLDER = "{seed}"


class SweepRecordError(ValueError):
    """Raised when a sweep record cannot be read or is malformed."""


class SweepMetricError(ValueError):
    """Raised when the aggregated metric cannot be read from a run's results."""


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------
def parse_seed_list(values: Sequence[str | int]) -> list[int]:
    """Parse CLI seed arguments: ``--seeds 1,2,3``, ``--seeds 1 2 3`` or a mix."""
    seeds: list[int] = []
    for raw in values:
        for part in re.split(r"[,\s]+", str(raw).strip()):
            if not part:
                continue
            try:
                seeds.append(int(part))
            except ValueError as exc:
                raise ExperimentSpecError(f"invalid seed {part!r}; seeds must be integers") from exc
    return seeds


def normalize_seeds(seeds: Sequence[int]) -> list[int]:
    """Validate seeds and return them sorted and de-duplicated (order-independent)."""
    cleaned: list[int] = []
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ExperimentSpecError(f"seeds must be ints, got {type(seed).__name__}")
        if not SEED_MIN <= seed <= SEED_MAX:
            raise ExperimentSpecError(f"seed must be in [{SEED_MIN}, {SEED_MAX}], got {seed}")
        cleaned.append(int(seed))
    if not cleaned:
        raise ExperimentSpecError("a sweep needs at least one seed")
    # sorted + de-duplicated: the aggregate must not depend on input order (D-028)
    return sorted(set(cleaned))


# ---------------------------------------------------------------------------
# statistics (stdlib only: numpy is not needed for a mean and a stdev)
# ---------------------------------------------------------------------------
def sample_mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` when there is nothing to average."""
    if not values:
        return None
    return float(statistics.fmean(values))


def sample_stdev(values: Sequence[float]) -> float | None:
    """Sample standard deviation (``n - 1``), or ``None`` when undefined (``n < 2``)."""
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def summarize_metric(metric: str, pairs: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Statistics section for ``[(seed, value), ...]`` (sorted by seed by the caller)."""
    values = [value for _seed, value in pairs]
    section: dict[str, Any] = {
        "metric": metric,
        "n": len(values),
        "values": [{"seed": seed, "value": value} for seed, value in pairs],
        "mean": sample_mean(values),
        "spread": sample_stdev(values),
        "spread_kind": "sample_standard_deviation",
        "spread_definition": SAMPLE_STDEV_DEFINITION,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
    if not values:
        section["note"] = (
            "no successful run produced a numeric value for this metric; mean and spread "
            "are undefined (they are null, not zero)"
        )
    elif len(values) == 1:
        section["note"] = (
            "single successful run: mean is that value, spread is undefined "
            "(sample standard deviation needs n >= 2)"
        )
    return section


def _json_from_stdout(tail: Sequence[str]) -> dict | None:
    """Last JSON object printed by a wrapped command (Project 001/002 print JSON summaries).

    Tried only as a fallback: command runs record ``exit_code`` and log tails, not domain
    metrics, so ``--metric score`` has to be read from the command's own JSON output.
    """
    lines = [line for line in tail if str(line).strip()]
    for start in range(len(lines)):
        try:
            payload = json.loads("\n".join(lines[start:]))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def extract_metric(results: Mapping[str, Any], metric: str) -> tuple[float, str]:
    """Return ``(value, source)`` for ``metric``; source is ``results`` or ``stdout_json``."""
    try:
        return _metric_value(results, metric), "results"
    except SweepMetricError as first:
        payload = _json_from_stdout(results.get("stdout_tail") or [])
        if payload is not None:
            try:
                return _metric_value(payload, metric), "stdout_json"
            except SweepMetricError:
                pass  # fall through: report the original, more precise error
        raise first


def _metric_value(results: Mapping[str, Any], metric: str) -> float:
    """Read a (possibly dotted) numeric metric out of a run's results."""
    node: Any = results
    for part in metric.split("."):
        if not isinstance(node, Mapping) or part not in node:
            available = sorted(node) if isinstance(node, Mapping) else None
            raise SweepMetricError(
                f"metric '{metric}' not found in the run results (available keys: {available})"
            )
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise SweepMetricError(f"metric '{metric}' is not numeric: got {type(node).__name__} ({node!r})")
    return float(node)


# ---------------------------------------------------------------------------
# the sweep record
# ---------------------------------------------------------------------------
@dataclass
class SweepRecord:
    """Machine-readable aggregate of one multi-seed sweep."""

    sweep: dict[str, Any]
    configuration: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SWEEP_SCHEMA_VERSION
    record_type: str = SWEEP_RECORD_TYPE

    _SECTIONS = ("sweep", "configuration", "code", "data", "environment", "statistics",
                 "runs", "execution")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            **{name: getattr(self, name) for name in self._SECTIONS},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SweepRecord:
        if not isinstance(data, Mapping):
            raise SweepRecordError("sweep record must be a JSON object")
        version = data.get("schema_version")
        if version != SWEEP_SCHEMA_VERSION:
            raise SweepRecordError(
                f"unsupported sweep schema_version {version!r} (expected {SWEEP_SCHEMA_VERSION!r})"
            )
        known = set(cls._SECTIONS) | {"schema_version", "record_type"}
        extra = set(data) - known
        if extra:
            raise SweepRecordError(f"unknown sweep sections: {sorted(extra)}")
        return cls(
            **{name: data.get(name, [] if name == "runs" else {}) for name in cls._SECTIONS},
            schema_version=SWEEP_SCHEMA_VERSION,
            record_type=str(data.get("record_type", SWEEP_RECORD_TYPE)),
        )

    @classmethod
    def load(cls, path: str | Path) -> SweepRecord:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def content_fingerprint(self) -> str:
        """Stable SHA-256 over the sweep content (variable runtime fields excluded)."""
        return content_fingerprint_of(self.to_dict())

    # -------------------------------------------------------------- view --
    def render(self) -> str:
        sweep = self.sweep
        stats = self.statistics
        git = self.code.get("git", {})
        mean = stats.get("mean")
        spread = stats.get("spread")

        lines = [
            f"Sweep {sweep.get('id')} — {sweep.get('name') or '(unnamed)'}",
            "=" * 60,
            f"status        : {sweep.get('status')} "
            f"({sweep.get('successful_count')}/{sweep.get('requested_count')} seeds usable)",
            f"metric        : {sweep.get('metric')}",
            f"code          : {_fmt_git(git)}",
            f"data          : {_fmt_data(self.data)}",
            f"seeds         : {len(sweep.get('seeds') or [])} requested · "
            f"{sweep.get('successful_count')} succeeded · {sweep.get('failed_count')} failed",
        ]
        if mean is None:
            lines.append("mean          : undefined (no usable runs)")
            lines.append("spread        : undefined (no usable runs)")
        else:
            lines.append(f"mean          : {mean:.6f}")
            lines.append(
                f"spread        : {spread:.6f} (sample stdev, n-1)"
                if spread is not None
                else "spread        : undefined (needs at least 2 usable runs)"
            )
            lines.append(f"range         : {stats.get('min'):.6f} … {stats.get('max'):.6f}")

        lines.append("")
        lines.append("per-seed:")
        for run in self.runs:
            value = run.get("metric_value")
            shown = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(
                f"  seed {run.get('seed'):<12} {run.get('status'):<15} "
                f"{sweep.get('metric')}={shown}  {str(run.get('fingerprint') or '')[:8]}"
            )
        failures = [r for r in self.runs if r.get("status") != "success"]
        if failures:
            lines.append("")
            lines.append("failed/incomplete seeds:")
            for run in failures:
                error = run.get("error") or {}
                lines.append(
                    f"  seed {run.get('seed')}: {error.get('type', 'unknown')}: {error.get('message', '')}"
                )

        lines.append("")
        lines.append(f"content fingerprint: {self.content_fingerprint()[:16]}… (timestamps excluded)")
        if git.get("dirty"):
            lines.append("WARNING: ran from a dirty working tree — not reproducible from the commit alone")
        if not git.get("available", False):
            lines.append("WARNING: git provenance unavailable — this sweep cannot be traced to a commit")
        return "\n".join(lines)


@dataclass
class SweepOutcome:
    record: SweepRecord
    path: Path
    rendered: str
    runs: list[RunOutcome]


# ---------------------------------------------------------------------------
# running a sweep
# ---------------------------------------------------------------------------
def _seed_dir(root: Path, seed: int) -> Path:
    return root / SEED_DIR_TEMPLATE.format(seed=seed)


def _fill_seed(argv: Sequence[str], seed: int) -> list[str]:
    """Substitute the ``{seed}`` placeholder in a command template."""
    return [arg.replace(SEED_PLACEHOLDER, str(seed)) for arg in argv]


def _run_sweep(
    spec: ExperimentSpec,
    seeds: Sequence[int],
    metric: str,
    execute_one: Callable[[ExperimentSpec, Path], RunOutcome],
    repo_path: str | Path = ".",
    output_dir: str | Path | None = None,
    overrides: Sequence[str] | None = None,
    extra_packages: tuple[str, ...] = (),
    continue_on_error: bool = True,
) -> SweepOutcome:
    """Shared sweep implementation: provenance once, then one run per seed."""
    if not str(metric).strip():
        raise ExperimentSpecError("a sweep needs the name of the result field to aggregate")
    canonical_seeds = normalize_seeds(seeds)
    validate_inputs(spec)  # fail before spending compute on any seed

    out_dir = Path(output_dir or spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The template spec is recorded with the *first canonical* seed so that the sweep's own
    # configuration cannot depend on the order the seeds were supplied in.
    template = ExperimentSpec.from_dict({**spec.to_dict(), "seed": canonical_seeds[0]})
    configuration = configuration_section(template, overrides)  # 2. configuration
    configuration["seeds_requested"] = list(canonical_seeds)
    configuration["spec_is_template"] = True
    configuration["spec_note"] = (
        "the embedded spec is the sweep template: 'seed' and 'output_dir' are replaced per "
        "run; 'seeds_requested' lists the seeds that were actually executed"
    )
    code = {"git": git_section(capture_git_info(repo_path)), "repository": "frontier-ai"}  # 3. git
    data = data_section(spec)                                   # 4. data provenance
    environment = capture_environment(extra_packages)           # 6. environment

    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outcomes: list[RunOutcome] = []
    entries: list[dict[str, Any]] = []
    values: list[tuple[int, float]] = []

    for seed in canonical_seeds:                                # 5+7. seed and execute per seed
        seed_dir = _seed_dir(out_dir, seed)
        # one spec per seed: same configuration, its own seed and its own output directory
        seed_spec = ExperimentSpec.from_dict(
            {**template.to_dict(), "seed": int(seed), "output_dir": str(seed_dir)}
        )
        entry: dict[str, Any] = {"seed": int(seed)}
        try:
            outcome = execute_one(seed_spec, seed_dir)
            outcomes.append(outcome)
            record = outcome.record
            try:
                value, source = extract_metric(record.results, metric)
            except SweepMetricError as exc:
                entry.update(status="metric_missing", metric_value=None, metric_source=None,
                             error={"type": type(exc).__name__, "message": str(exc)})
            else:
                entry.update(status="success", metric_value=value, metric_source=source,
                             error=None)
                values.append((int(seed), value))
        except Exception as exc:  # noqa: BLE001 - recorded, then the sweep continues
            record = _load_partial(seed_dir)
            entry.update(
                status="failed",
                metric_value=None,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            if not continue_on_error:
                entries.append(_finish_entry(entry, record, seed_dir, out_dir))
                _write_sweep(out_dir, spec, metric, canonical_seeds, configuration, code, data,
                             environment, entries, started_at, started, status="failed",
                             extra_note=f"stopped after seed {seed} (continue_on_error=False)")
                raise

        entries.append(_finish_entry(entry, record, seed_dir, out_dir))

    successful = [e["seed"] for e in entries if e["status"] == "success"]
    failed = [e["seed"] for e in entries if e["status"] != "success"]
    if not failed:
        status = "success"
    elif successful:
        status = "partial"
    else:
        status = "failed"

    record, path, rendered = _write_sweep(
        out_dir, template, metric, canonical_seeds, configuration, code, data, environment,
        entries, started_at, started, status=status,
    )
    return SweepOutcome(record=record, path=path, rendered=rendered, runs=outcomes)


def _load_partial(seed_dir: Path) -> ExperimentRecord | None:
    """Load the run record the runner wrote before it re-raised (if it got that far)."""
    path = seed_dir / RECORD_FILENAME
    if not path.exists():
        return None
    try:
        return ExperimentRecord.load(path)
    except Exception:  # noqa: BLE001 - a corrupt partial record must not hide the sweep failure
        return None


def _finish_entry(entry: dict[str, Any], record: ExperimentRecord | None, seed_dir: Path,
                  root: Path) -> dict[str, Any]:
    """Attach the record reference (relative to the sweep dir) and its fingerprint."""
    entry["record"] = str((seed_dir / RECORD_FILENAME).relative_to(root)) if record is not None else None
    entry["record_dir"] = str(seed_dir.relative_to(root))
    entry["fingerprint"] = record.content_fingerprint() if record is not None else None
    entry["run_status"] = record.experiment.get("status") if record is not None else "no-record"
    entry["seed_used"] = record.randomness.get("master_seed") if record is not None else entry["seed"]
    return entry


def _write_sweep(
    out_dir: Path,
    spec: ExperimentSpec,
    metric: str,
    seeds: Sequence[int],
    configuration: dict[str, Any],
    code: dict[str, Any],
    data: dict[str, Any],
    environment: dict[str, Any],
    entries: Sequence[dict[str, Any]],
    started_at: str,
    started: float,
    status: str,
    extra_note: str = "",
) -> tuple[SweepRecord, Path, str]:
    successful = [e["seed"] for e in entries if e["status"] == "success"]
    failed = [e["seed"] for e in entries if e["status"] != "success"]
    statistics_section = summarize_metric(
        metric,
        sorted((e["seed"], e["metric_value"]) for e in entries if e["metric_value"] is not None),
    )
    if extra_note:
        statistics_section["note"] = " ".join(
            filter(None, [statistics_section.get("note"), extra_note])
        )

    record = SweepRecord(
        sweep={
            "id": spec.experiment_id,
            "name": spec.name,
            "status": status,
            "metric": metric,
            "tags": list(spec.tags),
            "notes": spec.notes,
            "output_dir": str(out_dir),
            "seeds": list(seeds),
            "successful_seeds": successful,
            "failed_seeds": failed,
            "requested_count": len(seeds),
            "successful_count": len(successful),
            "failed_count": len(failed),
        },
        configuration=configuration,
        code=code,
        data=data,
        environment=environment,
        statistics=statistics_section,
        runs=list(entries),
        execution={
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": round(time.time() - started, 3),
            "cwd": str(Path.cwd()),
            "pid": __import__("os").getpid(),
        },
    )
    path = record.save(out_dir / SWEEP_FILENAME)
    rendered = record.render()
    (out_dir / "sweep.txt").write_text(rendered + "\n", encoding="utf-8")
    return record, path, rendered


def run_sweep(
    spec: ExperimentSpec,
    seeds: Sequence[int],
    experiment_fn: Callable[[ExperimentContext], Mapping[str, Any]],
    metric: str,
    repo_path: str | Path = ".",
    output_dir: str | Path | None = None,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    overrides: Sequence[str] | None = None,
    extra_packages: tuple[str, ...] = (),
    continue_on_error: bool = True,
) -> SweepOutcome:
    """Run ``experiment_fn`` once per seed and aggregate ``metric`` as mean ± spread.

    Each seed gets a full :class:`ExperimentRecord` under ``<output_dir>/seed-<seed>/``.
    Per-seed failures are recorded and (by default) do not stop the remaining seeds.
    """
    def execute_one(seed_spec: ExperimentSpec, seed_dir: Path) -> RunOutcome:
        return run_experiment(
            seed_spec,
            experiment_fn,
            repo_path=repo_path,
            output_dir=seed_dir,
            components=components,
            overrides=overrides,
            extra_packages=extra_packages,
        )

    return _run_sweep(
        spec, seeds, metric, execute_one, repo_path=repo_path, output_dir=output_dir,
        overrides=overrides, extra_packages=extra_packages, continue_on_error=continue_on_error,
    )


def run_command_sweep(
    spec: ExperimentSpec,
    seeds: Sequence[int],
    metric: str,
    command: Sequence[str] | None = None,
    repo_path: str | Path = ".",
    output_dir: str | Path | None = None,
    cwd: str | Path | None = None,
    extra_packages: tuple[str, ...] = (),
    timeout: float | None = None,
    continue_on_error: bool = True,
) -> SweepOutcome:
    """Run a command once per seed, substituting ``{seed}`` in the argv.

    A command with no ``{seed}`` placeholder runs identically for every seed, which is
    almost never what a seed sweep means — the caller is warned (see the CLI).
    """
    argv = list(command or spec.command)

    def execute_one(seed_spec: ExperimentSpec, seed_dir: Path) -> RunOutcome:
        argv_filled = _fill_seed(argv, int(seed_spec.seed))
        # record the exact argv that ran (seed substituted), not the {seed} template
        seed_spec = ExperimentSpec.from_dict({**seed_spec.to_dict(), "command": argv_filled})
        return run_command(
            seed_spec,
            argv_filled,
            repo_path=repo_path,
            output_dir=seed_dir,
            cwd=cwd,
            extra_packages=extra_packages,
            timeout=timeout,
        )

    return _run_sweep(
        spec, seeds, metric, execute_one, repo_path=repo_path, output_dir=output_dir,
        extra_packages=extra_packages, continue_on_error=continue_on_error,
    )


def command_has_seed_placeholder(command: Sequence[str]) -> bool:
    """True if a command template will actually differ between seeds."""
    return any(SEED_PLACEHOLDER in str(arg) for arg in command)


# ---------------------------------------------------------------------------
def _fmt_git(git: dict[str, Any]) -> str:
    if not git.get("available", False):
        return f"unavailable ({git.get('reason', 'unknown reason')})"
    short = (git.get("commit") or "")[:8]
    branch = git.get("branch") or "detached HEAD"
    return f"{short} on {branch} ({'dirty' if git.get('dirty') else 'clean'})"


def _fmt_data(data: dict[str, Any]) -> str:
    digest = data.get("digest")
    if data.get("status") == "none":
        return "no data inputs declared"
    if isinstance(digest, dict):
        return f"{digest.get('file_count')} file(s), {digest.get('digest', '')[:12]}…"
    return str(digest)

