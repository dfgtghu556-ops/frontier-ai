"""Experiment sweeps: one specification across seeds and configurations (ROADMAP Stage 1).

A sweep answers: **how much does the measured result vary across independent seeds, and
between configurations?** It is the mechanism behind the roadmap rule "report mean ± spread
for any headline number" and "a 2-config sweep runs unattended".

Two shapes, one implementation
------------------------------
* **Seed sweep** (Stage 1A, D-028): one specification, one seed list, one record per seed.
* **Configuration × seed sweep** (Stage 1B, D-029): several *named configurations*, each run
  across the same seed list, one record per configuration × seed.

Backward compatibility (D-029): a sweep with no named configurations is byte-identical to a
Stage 1A sweep — same directories (``seed-<seed>/``), same record sections, no
``configurations`` section, no ``configuration`` key on run entries, same fingerprint. The
extra fields appear **only** when named configurations are supplied.

Design rules (D-028, D-029)
---------------------------
Metric resolution: ``metric`` is read from the run record's ``results`` first, and — for
wrapped commands, which only record ``exit_code`` and log tails — from the last JSON object
the command printed (the convention ``scripts/train.py --eval-only`` and
``scripts/evaluate.py`` already use). The source actually used is recorded per run as
``metric_source``.

1. **Individual runs are never collapsed.** Every configuration × seed produces its own
   normal :class:`~frontier_ai.experiments.record.ExperimentRecord` — same sections, same
   git/data/environment provenance, same content fingerprint — written to its own directory
   (``<config>/seed-<zero-padded seed>/``, or ``seed-<seed>/`` for a seed-only sweep) so
   runs cannot collide.
2. **Order does not matter.** Configurations are sorted by name and seeds are normalised
   (ints, de-duplicated, sorted ascending) before anything runs, and statistics are computed
   over values sorted by seed. ``{"a": …, "b": …}`` and ``{"b": …, "a": …}`` (and
   ``[1, 2, 3]`` vs ``[3, 1, 2]``) give the same statistics and the same fingerprint. Each
   run still records the seed and configuration it actually used.
3. **Failures are recorded, not discarded.** A run that raises keeps its failed record, is
   listed in ``failed_seeds`` with its error, and does not stop the other runs. Status is
   ``success`` (all usable), ``partial`` (some usable) or ``failed`` (none usable) — for each
   configuration and for the sweep as a whole. A sweep never reports success with runs
   missing.
4. **No silent substitution.** A run that succeeded but did not produce a *numeric* value
   for the metric is counted as failed for aggregation; missing values are never replaced by
   zero.
5. **Spread is the sample standard deviation** (``n - 1`` denominator), undefined — and
   explicitly ``null`` — for fewer than two usable runs. It is not a confidence interval and
   is never called one.
6. **Configurations are never mixed.** With two or more configurations the top-level
   ``statistics`` section states that no cross-configuration aggregate was computed, and
   every mean/spread lives in the ``configurations`` section next to the configuration that
   produced it.

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
from .spec import SEED_MAX, SEED_MIN, ExperimentSpec, ExperimentSpecError, coerce_value

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
CONFIG_PLACEHOLDER = "{config}"

# Configuration names become directory names, so they are restricted rather than escaped.
CONFIGURATION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# What a configuration variant is allowed to change. Seed, output_dir, experiment_id,
# data_paths and command are sweep-level and are deliberately not overridable here.
CONFIGURATION_KEYS = ("params", "config_path", "name", "notes")
CONFIG_TAG_PREFIX = "config:"

NO_CROSS_CONFIG_NOTE = (
    "two or more configurations: no cross-configuration mean or spread is computed "
    "(results from different configurations are never mixed, D-029); see the "
    "'configurations' section for each configuration's own mean ± spread"
)
SEED_STATUS_NOTE = (
    "sweep-level seed lists are seed-wise across configurations: a seed counts as "
    "successful only if every configuration produced a usable metric for it. "
    "Per-configuration seed lists are in the 'configurations' section."
)


class SweepRecordError(ValueError):
    """Raised when a sweep record cannot be read or is malformed."""


class SweepMetricError(ValueError):
    """Raised when the aggregated metric cannot be read from a run's results."""


# ---------------------------------------------------------------------------
# seeds and configurations
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


@dataclass
class SweepConfiguration:
    """One named configuration inside a sweep, resolved to a full spec."""

    name: str
    spec: ExperimentSpec
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "spec": self.spec.to_dict(), "overrides": self.overrides}


def parse_configuration_list(values: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Parse CLI configuration arguments: ``NAME:params.lr=0.01,config_path=a.json``.

    Keys may be ``params.<name>``, ``config_path``, ``name`` or ``notes``; values are
    JSON-coerced (so ``0.01`` is a float and ``"tiny"`` stays a string).
    """
    configs: dict[str, dict[str, Any]] = {}
    for raw in values:
        text = str(raw).strip()
        if ":" not in text:
            raise ExperimentSpecError(
                f"configuration must look like NAME:key=value, got {raw!r}"
            )
        name, rest = text.split(":", 1)
        name = name.strip()
        if not CONFIGURATION_NAME_RE.match(name):
            raise ExperimentSpecError(
                f"invalid configuration name {name!r}: use letters, digits, '.', '_' or '-' "
                "(it becomes a directory name)"
            )
        if name in configs:
            raise ExperimentSpecError(f"duplicate configuration name: {name!r}")
        parsed: dict[str, Any] = {"params": {}}
        for item in rest.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ExperimentSpecError(
                    f"configuration {name!r}: override must look like key=value, got {item!r}"
                )
            key, value = (part.strip() for part in item.split("=", 1))
            if key.startswith("params."):
                parsed["params"][key[len("params."):]] = coerce_value(value)
            elif key in CONFIGURATION_KEYS:
                parsed[key] = coerce_value(value)
            else:
                raise ExperimentSpecError(
                    f"configuration {name!r} cannot override {key!r}; allowed: "
                    "params.<name>, " + ", ".join(CONFIGURATION_KEYS[1:])
                )
        configs[name] = parsed
    if not configs:
        raise ExperimentSpecError("a multi-configuration sweep needs at least one configuration")
    return configs


def normalize_configurations(
    spec: ExperimentSpec, configurations: Mapping[str, Mapping[str, Any]] | None
) -> list[SweepConfiguration]:
    """Resolve configurations into specs, sorted by name (order-independent).

    ``None`` (or an empty mapping) means the single, unnamed configuration: the sweep then
    behaves exactly like a Stage 1A seed sweep.
    """
    if not configurations:
        return [SweepConfiguration(name="", spec=spec, overrides={})]

    resolved: list[SweepConfiguration] = []
    for name in sorted(configurations):  # sorted: the aggregate must not depend on order
        if not CONFIGURATION_NAME_RE.match(str(name)):
            raise ExperimentSpecError(
                f"invalid configuration name {name!r}: use letters, digits, '.', '_' or '-' "
                "(it becomes a directory name)"
            )
        variant = configurations[name]
        if not isinstance(variant, Mapping):
            raise ExperimentSpecError(f"configuration {name!r} must be a mapping of overrides")
        unknown = set(variant) - set(CONFIGURATION_KEYS)
        if unknown:
            raise ExperimentSpecError(
                f"configuration {name!r} has unsupported keys {sorted(unknown)}; "
                f"allowed: {sorted(CONFIGURATION_KEYS)}"
            )
        params = dict(spec.params)
        params.update(variant.get("params") or {})
        merged = ExperimentSpec.from_dict({
            **spec.to_dict(),
            "params": params,
            "config_path": str(variant.get("config_path", spec.config_path)),
            "name": str(variant.get("name", spec.name)),
            "notes": str(variant.get("notes", spec.notes)),
        })
        resolved.append(SweepConfiguration(name=str(name), spec=merged, overrides=dict(variant)))
    if not resolved:
        raise ExperimentSpecError("a multi-configuration sweep needs at least one configuration")
    return resolved


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
    """Machine-readable aggregate of one sweep (seeds, or configurations × seeds)."""

    sweep: dict[str, Any]
    configuration: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)
    # Present only for multi-configuration sweeps (D-029): omitted entirely otherwise, so a
    # seed-only sweep keeps the Stage 1A record shape and fingerprint.
    configurations: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = SWEEP_SCHEMA_VERSION
    record_type: str = SWEEP_RECORD_TYPE

    _SECTIONS = ("sweep", "configuration", "code", "data", "environment", "statistics",
                 "runs", "execution", "configurations")

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
        }
        for name in self._SECTIONS:
            if name == "configurations" and not self.configurations:
                continue  # backward compatibility: no empty section in seed-only sweeps
            payload[name] = getattr(self, name)
        return payload

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
            **{name: data.get(name, [] if name in {"runs", "configurations"} else {})
               for name in cls._SECTIONS},
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
        metric = sweep.get("metric")
        multi = bool(self.configurations)
        mean = stats.get("mean")

        lines = [
            f"Sweep {sweep.get('id')} — {sweep.get('name') or '(unnamed)'}",
            "=" * 60,
            f"status        : {sweep.get('status')} "
            f"({sweep.get('successful_count')}/{sweep.get('requested_count')} seeds usable)",
            f"metric        : {metric}",
            f"code          : {_fmt_git(git)}",
            f"data          : {_fmt_data(self.data)}",
            f"seeds         : {len(sweep.get('seeds') or [])} requested · "
            f"{sweep.get('successful_count')} succeeded · {sweep.get('failed_count')} failed",
        ]
        if multi:
            lines.append(
                f"configurations: {len(self.configurations)} · "
                f"{sweep.get('runs_successful')}/{sweep.get('runs_requested')} runs usable"
            )
            for cfg in self.configurations:
                cfg_stats = cfg.get("statistics") or {}
                cfg_mean = cfg_stats.get("mean")
                cfg_spread = cfg_stats.get("spread")
                summary = (
                    f"mean {cfg_mean:.6f} ± {cfg_spread:.6f} (sample stdev, n={cfg_stats.get('n')})"
                    if cfg_mean is not None and cfg_spread is not None
                    else (f"mean {cfg_mean:.6f}, spread undefined (n={cfg_stats.get('n')})"
                          if cfg_mean is not None else "no usable runs")
                )
                lines.append(
                    f"  {cfg.get('name'):<16} {cfg.get('status'):<8} "
                    f"{cfg.get('successful_count')}/{cfg.get('requested_count')} seeds   {summary}"
                )
        else:
            if mean is None:
                lines.append("mean          : undefined (no usable runs)")
            else:
                lines.append(f"mean          : {mean:.6f}")
                lines.append(
                    f"spread        : {stats.get('spread'):.6f} (sample stdev, n-1)"
                    if stats.get("spread") is not None
                    else "spread        : undefined (needs at least 2 usable runs)"
                )
                lines.append(f"range         : {stats.get('min'):.6f} … {stats.get('max'):.6f}")

        lines.append("")
        lines.append("per-run:")
        for run in self.runs:
            value = run.get("metric_value")
            shown = f"{value:.6f}" if isinstance(value, float) else str(value)
            label = f"{run.get('configuration')}/{run.get('seed')}" if multi else f"seed {run.get('seed')}"
            lines.append(
                f"  {label:<28} {run.get('status'):<15} {metric}={shown}  "
                f"{str(run.get('fingerprint') or '')[:8]}"
            )
        failures = [r for r in self.runs if r.get("status") != "success"]
        if failures:
            lines.append("")
            lines.append("failed/incomplete runs:")
            for run in failures:
                error = run.get("error") or {}
                where = (f"{run.get('configuration')}/seed {run.get('seed')}" if multi
                         else f"seed {run.get('seed')}")
                lines.append(
                    f"  {where}: {error.get('type', 'unknown')}: {error.get('message', '')}"
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


def _fill_template(argv: Sequence[str], seed: int, configuration: str) -> list[str]:
    """Substitute the ``{seed}`` and ``{config}`` placeholders in a command template."""
    return [
        str(arg).replace(SEED_PLACEHOLDER, str(seed)).replace(CONFIG_PLACEHOLDER, str(configuration))
        for arg in argv
    ]


def _status_of(successful: int, failed: int) -> str:
    if not failed:
        return "success"
    return "partial" if successful else "failed"


def _run_sweep(
    spec: ExperimentSpec,
    seeds: Sequence[int],
    metric: str,
    execute_one: Callable[[ExperimentSpec, Path, str], RunOutcome],
    repo_path: str | Path = ".",
    output_dir: str | Path | None = None,
    overrides: Sequence[str] | None = None,
    extra_packages: tuple[str, ...] = (),
    continue_on_error: bool = True,
    configurations: Mapping[str, Mapping[str, Any]] | None = None,
) -> SweepOutcome:
    """Shared sweep implementation: provenance once, then one run per configuration × seed."""
    if not str(metric).strip():
        raise ExperimentSpecError("a sweep needs the name of the result field to aggregate")
    canonical_seeds = normalize_seeds(seeds)
    validate_inputs(spec)  # fail before spending compute on any run

    out_dir = Path(output_dir or spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The template spec is recorded with the *first canonical* seed so that the sweep's own
    # configuration cannot depend on the order the seeds were supplied in.
    template = ExperimentSpec.from_dict({**spec.to_dict(), "seed": canonical_seeds[0]})
    # Configurations are resolved from the *template*, never from the raw spec: the raw
    # spec carries whichever seed the caller happened to list first, which would make the
    # record depend on seed order (the same class of bug as D-028's template fix).
    configs = normalize_configurations(template, configurations)
    # A sweep with named configurations is "multi": it gets the extra sections and the
    # <config>/seed-<seed>/ layout. A seed-only sweep keeps the Stage 1A shape exactly.
    multi = any(cfg.name for cfg in configs)
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
    entries_by_config: dict[str, list[dict[str, Any]]] = {cfg.name: [] for cfg in configs}

    for cfg in configs:                                          # 5+7. seed and execute
        cfg_dir = out_dir / cfg.name if multi else out_dir
        for seed in canonical_seeds:
            seed_dir = _seed_dir(cfg_dir, seed)
            # one spec per run: this configuration's params, its own seed and output dir
            run_spec = ExperimentSpec.from_dict(
                {**cfg.spec.to_dict(), "seed": int(seed), "output_dir": str(seed_dir)}
            )
            if multi and f"{CONFIG_TAG_PREFIX}{cfg.name}" not in run_spec.tags:
                run_spec = ExperimentSpec.from_dict(
                    {**run_spec.to_dict(),
                     "tags": [*run_spec.tags, f"{CONFIG_TAG_PREFIX}{cfg.name}"]}
                )
            entry: dict[str, Any] = {"seed": int(seed)}
            if multi:
                entry["configuration"] = cfg.name
            record: ExperimentRecord | None = None
            try:
                outcome = execute_one(run_spec, seed_dir, cfg.name)
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
            except Exception as exc:  # noqa: BLE001 - recorded, then the sweep continues
                record = _load_partial(seed_dir)
                entry.update(
                    status="failed",
                    metric_value=None,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                if not continue_on_error:
                    entries.append(_finish_entry(entry, record, seed_dir, out_dir))
                    entries_by_config[cfg.name].append(entries[-1])
                    _write_sweep(out_dir, template, metric, canonical_seeds, configuration, code,
                                 data, environment, entries, started_at, started, status="failed",
                                 multi=multi, configs=configs,
                                 entries_by_config=entries_by_config,
                                 extra_note=f"stopped after {cfg.name or 'seed'}/{seed} "
                                            "(continue_on_error=False)")
                    raise

            entries.append(_finish_entry(entry, record, seed_dir, out_dir))
            entries_by_config[cfg.name].append(entries[-1])

    if multi:
        usable_per_seed = {seed: 0 for seed in canonical_seeds}
        for entry in entries:
            if entry["status"] == "success":
                usable_per_seed[entry["seed"]] += 1
        successful = [s for s in canonical_seeds if usable_per_seed[s] == len(configs)]
        failed = [s for s in canonical_seeds if usable_per_seed[s] != len(configs)]
        status = _status_of(
            sum(1 for e in entries if e["status"] == "success"),
            sum(1 for e in entries if e["status"] != "success"),
        )
    else:
        successful = [e["seed"] for e in entries if e["status"] == "success"]
        failed = [e["seed"] for e in entries if e["status"] != "success"]
        status = _status_of(len(successful), len(failed))

    record, path, rendered = _write_sweep(
        out_dir, template, metric, canonical_seeds, configuration, code, data, environment,
        entries, started_at, started, status=status, multi=multi, configs=configs,
        entries_by_config=entries_by_config,
    )
    return SweepOutcome(record=record, path=path, rendered=rendered, runs=outcomes)


def _configuration_sections(
    configs: Sequence[SweepConfiguration],
    metric: str,
    seeds: Sequence[int],
    out_dir: Path,
    multi: bool,
    entries_by_config: Mapping[str, Sequence[dict[str, Any]]],
    extra_note: str = "",
) -> list[dict[str, Any]]:
    """Per-configuration status, counts and mean ± spread (D-029)."""
    sections: list[dict[str, Any]] = []
    for cfg in configs:
        cfg_entries = list(entries_by_config.get(cfg.name, ()))
        successful = [e["seed"] for e in cfg_entries if e["status"] == "success"]
        failed = [e["seed"] for e in cfg_entries if e["status"] != "success"]
        section: dict[str, Any] = {
            "name": cfg.name,
            "status": _status_of(len(successful), len(failed)),
            "metric": metric,
            "seeds": list(seeds),
            "successful_seeds": successful,
            "failed_seeds": failed,
            "requested_count": len(seeds),
            "successful_count": len(successful),
            "failed_count": len(failed),
            # relative, like runs[].record_dir: a sweep record must not embed machine paths
            "output_dir": cfg.name if multi else ".",
            "spec": cfg.spec.to_dict(),
            "overrides": cfg.overrides,
            "statistics": summarize_metric(
                metric,
                sorted((e["seed"], e["metric_value"])
                       for e in cfg_entries if e["metric_value"] is not None),
            ),
        }
        if extra_note:
            note = section["statistics"].get("note")
            section["statistics"]["note"] = " ".join(filter(None, [note, extra_note]))
        sections.append(section)
    return sections


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
    multi: bool = False,
    configs: Sequence[SweepConfiguration] = (),
    entries_by_config: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    extra_note: str = "",
) -> tuple[SweepRecord, Path, str]:
    config_sections = _configuration_sections(configs, metric, seeds, out_dir, multi,
                                              entries_by_config or {}, extra_note=extra_note)

    if multi:
        # never mix configurations: state explicitly that no cross-config aggregate exists
        statistics_section: dict[str, Any] = {
            "metric": metric,
            "aggregated": False,
            "note": NO_CROSS_CONFIG_NOTE if not extra_note else f"{NO_CROSS_CONFIG_NOTE} {extra_note}",
        }
    else:
        statistics_section = summarize_metric(
            metric,
            sorted((e["seed"], e["metric_value"]) for e in entries if e["metric_value"] is not None),
        )
        if extra_note:
            statistics_section["note"] = " ".join(
                filter(None, [statistics_section.get("note"), extra_note])
            )

    if multi:
        usable_per_seed = {seed: 0 for seed in seeds}
        for entry in entries:
            if entry["status"] == "success":
                usable_per_seed[entry["seed"]] += 1
        successful = [s for s in seeds if usable_per_seed[s] == len(config_sections)]
        failed = [s for s in seeds if usable_per_seed[s] != len(config_sections)]
    else:
        successful = [e["seed"] for e in entries if e["status"] == "success"]
        failed = [e["seed"] for e in entries if e["status"] != "success"]

    sweep_section: dict[str, Any] = {
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
    }
    if multi:
        sweep_section.update(
            configuration_names=[cfg["name"] for cfg in config_sections],
            configuration_count=len(config_sections),
            runs_requested=len(seeds) * len(config_sections),
            runs_successful=sum(1 for e in entries if e["status"] == "success"),
            runs_failed=sum(1 for e in entries if e["status"] != "success"),
            seed_status_note=SEED_STATUS_NOTE,
        )

    record = SweepRecord(
        sweep=sweep_section,
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
        configurations=config_sections if multi else [],
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
    *,
    configurations: Mapping[str, Mapping[str, Any]] | None = None,
) -> SweepOutcome:
    """Run ``experiment_fn`` once per configuration × seed and aggregate ``metric``.

    With ``configurations=None`` (the default) this is exactly the Stage 1A seed sweep:
    every seed gets a full :class:`ExperimentRecord` under ``<output_dir>/seed-<seed>/``.
    With ``configurations={"a": {...}, "b": {...}}`` each named configuration is run across
    all seeds under ``<output_dir>/<config>/seed-<seed>/``, and each configuration is
    aggregated separately — results from different configurations are never mixed.

    A configuration may override ``params.<name>``, ``config_path``, ``name`` and ``notes``;
    the experiment body reads its configuration from ``ctx.spec.params`` (and
    ``ctx.configuration``). Per-run failures are recorded and, by default, do not stop the
    remaining runs.
    """
    def execute_one(run_spec: ExperimentSpec, run_dir: Path, configuration: str) -> RunOutcome:
        return run_experiment(
            run_spec,
            experiment_fn,
            repo_path=repo_path,
            output_dir=run_dir,
            components=components,
            overrides=overrides,
            extra_packages=extra_packages,
            configuration=configuration,
        )

    return _run_sweep(
        spec, seeds, metric, execute_one, repo_path=repo_path, output_dir=output_dir,
        overrides=overrides, extra_packages=extra_packages, continue_on_error=continue_on_error,
        configurations=configurations,
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
    *,
    configurations: Mapping[str, Mapping[str, Any]] | None = None,
) -> SweepOutcome:
    """Run a command once per configuration × seed.

    ``{seed}`` and ``{config}`` in the argv are substituted per run. A command with no
    ``{seed}`` placeholder runs identically for every seed — the CLI warns about that, and
    about a missing ``{config}`` placeholder when several configurations are requested.
    """
    argv = list(command or spec.command)

    def execute_one(run_spec: ExperimentSpec, run_dir: Path, configuration: str) -> RunOutcome:
        argv_filled = _fill_template(argv, int(run_spec.seed), configuration)
        # record the exact argv that ran (placeholders substituted), not the template
        run_spec = ExperimentSpec.from_dict({**run_spec.to_dict(), "command": argv_filled})
        return run_command(
            run_spec,
            argv_filled,
            repo_path=repo_path,
            output_dir=run_dir,
            cwd=cwd,
            extra_packages=extra_packages,
            timeout=timeout,
        )

    return _run_sweep(
        spec, seeds, metric, execute_one, repo_path=repo_path, output_dir=output_dir,
        extra_packages=extra_packages, continue_on_error=continue_on_error,
        configurations=configurations,
    )


def command_has_seed_placeholder(command: Sequence[str]) -> bool:
    """True if a command template will actually differ between seeds."""
    return any(SEED_PLACEHOLDER in str(arg) for arg in command)


def command_has_config_placeholder(command: Sequence[str]) -> bool:
    """True if a command template will actually differ between configurations."""
    return any(CONFIG_PLACEHOLDER in str(arg) for arg in command)


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
