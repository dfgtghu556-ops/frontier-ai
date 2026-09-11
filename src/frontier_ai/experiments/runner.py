"""The experiment runner: a standard lifecycle with provenance captured around it.

Lifecycle (ROADMAP Stage 1)::

    validate inputs → capture configuration → capture git provenance → capture data
    provenance → initialise deterministic randomness → execute experiment →
    capture environment → capture results → write experiment record

The environment is captured **after** the body, not before it: torch's CPU thread count is
owned by the experiment body (a ``Trainer`` sets it), so only the value the run actually
used is true provenance. The value the caller had is restored afterwards, so one run in a
process cannot change the environment — or the fingerprint — of the next (D-034, Q-13).

Failure policy
--------------
The runner **never swallows exceptions**. If the experiment raises, a record is written
with ``status="failed"`` and the error details, and the original exception is re-raised so
the caller (CLI, test, notebook) fails loudly. A failed run never produces a record that
looks like a success.

The runner is usable from Python (:func:`run_experiment`) and from the CLI
(:func:`run_command`, exposed by ``scripts/experiment_record.py``). Neither requires any
change to Project 001 or Project 002 — they are just commands or callables wrapped here.

Nested runs
-----------
Project 001/002 CLIs can also record themselves (``frontier_ai.experiments.autowire``).
To keep one run = one record, the runner exports the active run directory through
:data:`EXPERIMENT_ENV_VAR` while the body executes — subprocesses inherit it — and a
self-recording script that sees it writes no record of its own. The **outer** run owns
the record (D-032).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .environment import capture_environment
from .gitinfo import capture_git_info, git_section
from .hashing import hash_paths
from .record import RECORD_FILENAME, ExperimentRecord
from .seeding import DEFAULT_COMPONENTS, seed_everything
from .spec import ExperimentSpec

OUTPUT_TAIL_LINES = 20

# Nested runs: a script that can record itself needs to know when it is already
# inside a run (scripts/experiment_record.py, a sweep, a Python caller), so one run
# never produces two records. The runner exports the active run's directory through
# this variable for the duration of the experiment body; child processes inherit it.
# Ownership rule (D-032): the **outer** run owns the record.
EXPERIMENT_ENV_VAR = "FRONTIER_AI_EXPERIMENT_DIR"


def active_experiment_dir() -> Path | None:
    """Directory of the experiment currently in progress, if any."""
    value = os.environ.get(EXPERIMENT_ENV_VAR, "")
    return Path(value) if value else None


def experiment_is_active() -> bool:
    """True when this process is already running inside an experiment lifecycle."""
    return active_experiment_dir() is not None


# A nested self-recording script writes no record of its own (D-032), so its metrics would
# be lost to the run that owns the record. Instead of making the outer run scrape
# human-readable logs, the script prints one machine-readable line on stdout and
# :func:`run_command` merges it into the outer record's ``results`` (D-034). One line, one
# JSON object, last-wins-free: existing record and sweep schemas are unchanged.
NESTED_RESULTS_MARKER = "frontier_ai_nested_results"
NESTED_RESULTS_SCHEMA = "1.0"


def nested_results_line(results: Mapping[str, Any], script: str = "") -> str:
    """Render ``results`` as the one stdout line a nested run publishes to its outer run."""
    payload = {
        NESTED_RESULTS_MARKER: {
            "schema": NESTED_RESULTS_SCHEMA,
            "script": script,
            "results": dict(results or {}),
        }
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_nested_results(stdout: str) -> dict[str, Any]:
    """Collect the results published by nested self-recording scripts on ``stdout``.

    Recognises only lines that are JSON objects carrying :data:`NESTED_RESULTS_MARKER`;
    everything else (log lines, a command's own JSON report) is ignored. When two nested
    runs publish the same key the **first** one wins, so the merge is order-stable for a
    given command.
    """
    merged: dict[str, Any] = {}
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        block = payload.get(NESTED_RESULTS_MARKER)
        if not isinstance(block, dict):
            continue
        published = block.get("results")
        if not isinstance(published, dict):
            continue
        for key, value in published.items():
            merged.setdefault(key, value)
    return merged


@contextlib.contextmanager
def _activated(output_dir: str | Path) -> Iterator[None]:
    """Export the active run's directory for the duration of the body (and children)."""
    previous = os.environ.get(EXPERIMENT_ENV_VAR)
    os.environ[EXPERIMENT_ENV_VAR] = str(Path(output_dir).resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(EXPERIMENT_ENV_VAR, None)
        else:
            os.environ[EXPERIMENT_ENV_VAR] = previous


@dataclass
class ExperimentContext:
    """Everything an experiment function is allowed to depend on."""

    spec: ExperimentSpec
    output_dir: Path
    seed: int
    derived_seeds: dict[str, int] = field(default_factory=dict)
    # Name of the sweep configuration this run belongs to ("" outside a sweep, or in a
    # single-configuration sweep). Informational: the configuration is also visible in the
    # run's params/config_path and in its `config:<name>` tag.
    configuration: str = ""

    def derived(self, name: str, fallback: int | None = None) -> int:
        """Component-specific seed derived from the master seed."""
        if name in self.derived_seeds:
            return self.derived_seeds[name]
        if fallback is not None:
            return fallback
        raise KeyError(
            f"no derived seed for component '{name}'. Request it via "
            f"run_experiment(..., components=('{name}',)) or use ctx.seed."
        )

    def write_artifact(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.output_dir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                        encoding="utf-8")
        return path


def torch_thread_count() -> int | None:
    """torch's current intra-op thread count, or None when torch is unavailable."""
    try:
        import torch
    except Exception:                                    # torch is optional at import time
        return None
    try:
        return int(torch.get_num_threads())
    except Exception:                                    # pragma: no cover - defensive
        return None


def restore_torch_threads(count: int | None) -> None:
    """Put torch's thread count back to ``count`` so a run leaves no global state behind.

    The **body** owns the thread count (a ``Trainer`` sets it from ``train.num_threads`` or
    a device policy). The runner neither imposes nor overrides a value - it records what
    the run used and restores what the caller had (D-034).
    """
    if count is None:
        return
    try:
        import torch

        if int(torch.get_num_threads()) != int(count):
            torch.set_num_threads(int(count))
    except Exception:                                    # pragma: no cover - defensive
        pass


@dataclass
class RunOutcome:
    record: ExperimentRecord
    path: Path
    rendered: str


class ExperimentInputError(FileNotFoundError):
    """Raised when declared inputs are missing before the experiment starts."""


def validate_inputs(spec: ExperimentSpec) -> None:
    """Fail early on missing declared inputs (before spending compute)."""
    missing = [p for p in spec.data_paths if not Path(p).exists()]
    if missing:
        raise ExperimentInputError(f"declared data inputs are missing: {missing}")
    if spec.config_path and not Path(spec.config_path).exists():
        raise ExperimentInputError(f"declared config file is missing: {spec.config_path}")


def configuration_section(spec: ExperimentSpec, overrides: Sequence[str] | None = None) -> dict:
    section: dict[str, Any] = {"spec": spec.to_dict(), "overrides": list(overrides or [])}
    if spec.config_path:
        path = Path(spec.config_path)
        try:
            section["config_file"] = {
                "path": str(path),
                "content": json.loads(path.read_text(encoding="utf-8")),
            }
        except json.JSONDecodeError:
            # non-JSON config files (e.g. raw text) are recorded by hash + size only
            section["config_file"] = {
                "path": str(path),
                "content": None,
                "note": "config file is not JSON; recorded by path only",
            }
    return section


def data_section(spec: ExperimentSpec) -> dict:
    if not spec.data_paths:
        return {
            "status": "none",
            "inputs": [],
            "note": "experiment declared no data inputs",
        }
    digest = hash_paths(spec.data_paths)
    return {"status": "ok", "inputs": list(spec.data_paths), "digest": digest.to_dict()}


def run_experiment(
    spec: ExperimentSpec,
    experiment_fn: Callable[[ExperimentContext], Mapping[str, Any]],
    repo_path: str | Path = ".",
    output_dir: str | Path | None = None,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    overrides: Sequence[str] | None = None,
    extra_packages: tuple[str, ...] = (),
    configuration: str = "",
) -> RunOutcome:
    """Run ``experiment_fn`` inside the standard provenance lifecycle.

    ``configuration`` names the sweep configuration this run belongs to, if any; it is
    passed through to :class:`ExperimentContext` and is not part of the record.
    """
    # captured before the local `configuration` variable is reused for the record's
    # configuration *section* below (shadowing it would silently drop the name)
    configuration_name = str(configuration or "")
    validate_inputs(spec)
    out_dir = Path(output_dir or spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configuration = configuration_section(spec, overrides)          # 2. configuration
    code = {"git": git_section(capture_git_info(repo_path)), "repository": "frontier-ai"}  # 3. git
    data = data_section(spec)                                       # 4. data
    randomness = seed_everything(                                    # 5. randomness
        spec.seed, deterministic=spec.deterministic_mode, components=components
    )
    # 6. environment is captured in the `finally` block, *after* the body: torch's thread
    # count is chosen by the body, so the value it ran with - not the value the caller
    # happened to have - is the provenance that belongs in the record (D-034, Q-13).
    threads_before = torch_thread_count()

    ctx = ExperimentContext(
        spec=spec,
        output_dir=out_dir,
        seed=spec.seed,
        derived_seeds=dict(randomness.get("derived_seeds", {})),
        configuration=configuration_name,
    )

    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results: dict[str, Any] = {}
    status = "success"
    error: dict[str, Any] | None = None
    try:                                                             # 7. execute
        with _activated(out_dir):
            results = dict(experiment_fn(ctx) or {})
    # BaseException, not Exception: a CLI body that calls sys.exit() (or a Ctrl-C) must
    # still produce a *failed* record. Catching only Exception here would let SystemExit
    # fall through to the finally block, which would then write status="success" for a
    # run that never completed - exactly the silent success this lifecycle exists to
    # prevent. The exception is re-raised either way.
    except BaseException as exc:                                      # 8/9. capture + re-raise
        status = "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().strip().splitlines()[-5:],
        }
        raise
    finally:
        finished = time.time()
        environment = capture_environment(extra_packages)            # 6. environment
        restore_torch_threads(threads_before)      # no global thread leakage into the next run
        execution = {
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": round(finished - started, 3),
            "cwd": str(Path.cwd()),
            "pid": os.getpid(),
            "status": status,
            "command": list(spec.command),
            "error": error,
        }
        record = ExperimentRecord(
            experiment={
                "id": spec.experiment_id,
                "name": spec.name,
                "tags": list(spec.tags),
                "notes": spec.notes,
                "status": status,
                "output_dir": str(out_dir),
            },
            code=code,
            data=data,
            configuration=configuration,
            randomness=randomness,
            environment=environment,
            execution=execution,
            results=results,
        )
        path = record.save(out_dir / RECORD_FILENAME)
        (out_dir / "experiment.txt").write_text(record.render() + "\n", encoding="utf-8")
        outcome = RunOutcome(record=record, path=path, rendered=record.render())

    return outcome


def run_command(
    spec: ExperimentSpec,
    command: Sequence[str] | None = None,
    repo_path: str | Path = ".",
    output_dir: str | Path | None = None,
    cwd: str | Path | None = None,
    extra_packages: tuple[str, ...] = (),
    timeout: float | None = None,
) -> RunOutcome:
    """Run a subprocess inside the same lifecycle (used by scripts/experiment_record.py)."""
    argv = list(command or spec.command)

    def _execute(ctx: ExperimentContext) -> dict:
        if not argv:
            return {"note": "no command supplied; recorded provenance only"}
        completed = subprocess.run(
            argv,
            cwd=str(cwd or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        ctx.write_artifact(
            "command_output.json",
            {
                "exit_code": completed.returncode,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            },
        )
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=__import__("sys").stderr)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode, argv, output=completed.stdout, stderr=completed.stderr
            )
        results = {
            "exit_code": completed.returncode,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
        # a nested self-recording script published its metrics on stdout (D-034): merge
        # them so sweeps can aggregate `best_val` without parsing human-readable logs.
        # The wrapper's own keys win, so the merge can never rewrite exit_code or the tails.
        for key, value in parse_nested_results(completed.stdout).items():
            results.setdefault(key, value)
        return results

    return run_experiment(
        spec,
        _execute,
        repo_path=repo_path,
        output_dir=output_dir,
        extra_packages=extra_packages,
    )


def _tail(text: str, lines: int = OUTPUT_TAIL_LINES) -> list[str]:
    return [line for line in (text or "").splitlines()[-lines:]]
