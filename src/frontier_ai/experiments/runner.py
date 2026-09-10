"""The experiment runner: a standard lifecycle with provenance captured around it.

Lifecycle (ROADMAP Stage 1)::

    validate inputs → capture configuration → capture git provenance → capture data
    provenance → initialise deterministic randomness → capture environment →
    execute experiment → capture results → write experiment record

Failure policy
--------------
The runner **never swallows exceptions**. If the experiment raises, a record is written
with ``status="failed"`` and the error details, and the original exception is re-raised so
the caller (CLI, test, notebook) fails loudly. A failed run never produces a record that
looks like a success.

The runner is usable from Python (:func:`run_experiment`) and from the CLI
(:func:`run_command`, exposed by ``scripts/experiment_record.py``). Neither requires any
change to Project 001 or Project 002 — they are just commands or callables wrapped here.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from collections.abc import Mapping, Sequence
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


@dataclass
class ExperimentContext:
    """Everything an experiment function is allowed to depend on."""

    spec: ExperimentSpec
    output_dir: Path
    seed: int
    derived_seeds: dict[str, int] = field(default_factory=dict)

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


def _configuration_section(spec: ExperimentSpec, overrides: Sequence[str] | None = None) -> dict:
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


def _data_section(spec: ExperimentSpec) -> dict:
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
) -> RunOutcome:
    """Run ``experiment_fn`` inside the standard provenance lifecycle."""
    validate_inputs(spec)
    out_dir = Path(output_dir or spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configuration = _configuration_section(spec, overrides)          # 2. configuration
    code = {"git": git_section(capture_git_info(repo_path)), "repository": "frontier-ai"}  # 3. git
    data = _data_section(spec)                                       # 4. data
    randomness = seed_everything(                                    # 5. randomness
        spec.seed, deterministic=spec.deterministic_mode, components=components
    )
    environment = capture_environment(extra_packages)                # 6. environment

    ctx = ExperimentContext(
        spec=spec,
        output_dir=out_dir,
        seed=spec.seed,
        derived_seeds=dict(randomness.get("derived_seeds", {})),
    )

    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results: dict[str, Any] = {}
    status = "success"
    error: dict[str, Any] | None = None
    try:                                                             # 7. execute
        results = dict(experiment_fn(ctx) or {})
    except Exception as exc:                                         # 8/9. capture + re-raise
        status = "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().strip().splitlines()[-5:],
        }
        raise
    finally:
        finished = time.time()
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
        return {
            "exit_code": completed.returncode,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }

    return run_experiment(
        spec,
        _execute,
        repo_path=repo_path,
        output_dir=output_dir,
        extra_packages=extra_packages,
    )


def _tail(text: str, lines: int = OUTPUT_TAIL_LINES) -> list[str]:
    return [line for line in (text or "").splitlines()[-lines:]]
