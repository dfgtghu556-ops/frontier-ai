"""Let a CLI script record itself through the standard experiment lifecycle.

Why this exists: ``scripts/experiment_record.py`` wraps *any* command, which is right for
one-off provenance but leaves the Project 001/002 CLIs silent when they are run directly.
This module gives those scripts the same record, using the **same** lifecycle — there is
no second framework here, just a thin adapter over :func:`~frontier_ai.experiments.runner
.run_experiment`.

Ownership rule (D-032): **the outer run owns the record.** A script records itself only
when it is not already inside a run:

    python scripts/train.py --config configs/cpu_smoke.json     # writes its own record
    python scripts/experiment_record.py --exp-id EXP-003 -- \\
        python scripts/train.py --config configs/cpu_smoke.json # the wrapper's record only

The nesting signal is :data:`~frontier_ai.experiments.runner.EXPERIMENT_ENV_VAR`, exported
by the runner while the body executes and inherited by subprocesses.

A nested run still has to hand its metrics to the run that owns the record. It does that
in the one way the project already uses for machine-readable output: a single JSON line on
stdout carrying :data:`~frontier_ai.experiments.runner.NESTED_RESULTS_MARKER`, which
:func:`~frontier_ai.experiments.runner.run_command` merges into the outer record's
``results``. No log scraping, no argv inspection, no second record (D-034).

Failure semantics are the runner's: a body that raises produces a record with
``status: "failed"``, the error type/message and a traceback tail, and the exception is
re-raised. This helper turns it into a non-zero exit code and a message on stderr; a
failed run never looks like a success.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .record import RECORD_FILENAME
from .runner import (
    active_experiment_dir,
    experiment_is_active,
    nested_results_line,
    run_experiment,
)
from .spec import EXPERIMENT_ID_RE, ExperimentSpec, ExperimentSpecError

RECORD_FILENAME_NAME = RECORD_FILENAME


@dataclass
class SelfRecorded:
    """Outcome of a self-recording script run."""

    results: dict[str, Any] = field(default_factory=dict)
    # None when the run was nested: the outer experiment owns the record.
    record_path: Path | None = None
    fingerprint: str | None = None
    nested: bool = False
    failed: bool = False
    skipped: bool = False
    error: str = ""

    @property
    def recorded(self) -> bool:
        return self.record_path is not None

    def exit_code(self) -> int:
        return 1 if self.failed else 0


def run_self_recorded(
    spec: ExperimentSpec | Callable[[], ExperimentSpec],
    body: Callable[[], Mapping[str, Any]],
    *,
    repo_path: str | Path = ".",
    quiet: bool = False,
) -> SelfRecorded:
    """Run ``body`` and record it, unless an experiment is already active.

    ``body`` takes no arguments: a CLI script already has its parsed arguments in scope,
    and everything the record needs (seed, config, data paths, command) comes from
    ``spec``. Returns a :class:`SelfRecorded` — the script decides its exit code from
    :meth:`SelfRecorded.exit_code`.

    ``spec`` may also be a zero-argument callable returning one, which is built only once
    we know we are going to record. That matters because the CLIs accept experiment ids
    that predate the ``EXP-<3+ digits>`` rule (``--exp-id EXP-TEST``): a spec that cannot
    be recorded must not break a command that worked before, so building it lazily lets
    the failure downgrade to a ``skipped`` note on stderr instead of a crash.
    """
    if experiment_is_active():
        outer = active_experiment_dir()
        if not quiet:
            print(
                f"[record] nested inside {outer}; the outer run owns the experiment record",
                file=sys.stderr,
            )
        results = dict(body() or {})
        if results and not quiet:
            # No record of our own - so publish the metrics instead of making the outer
            # run reconstruct them from human-readable logs (D-034).
            print(nested_results_line(results, script=Path(sys.argv[0]).name))
        return SelfRecorded(results=results, nested=True)

    try:
        experiment = spec() if callable(spec) else spec
    except ExperimentSpecError as exc:
        if not quiet:
            print(f"[record] SKIPPED — {exc}", file=sys.stderr)
        return SelfRecorded(results=dict(body() or {}), skipped=True)

    try:
        outcome = run_experiment(experiment, lambda _ctx: dict(body() or {}), repo_path=repo_path)
    except BaseException as exc:  # the runner has already written a failed record
        message = f"{type(exc).__name__}: {exc}"
        # an input-validation failure happens before the lifecycle starts: no record.
        # otherwise the runner has already written one - point at it if it is there.
        record = getattr(exc, "record", None)
        if record is None:
            candidate = Path(experiment.output_dir) / RECORD_FILENAME_NAME
            record = candidate if candidate.exists() else None
        if not quiet:
            print(f"[record] FAILED — {message}", file=sys.stderr)
        return SelfRecorded(failed=True, error=message, record_path=record)

    if not quiet:
        print(
            f"[record] {outcome.path} | fingerprint {outcome.record.content_fingerprint()[:16]}…",
            file=sys.stderr,
        )
    return SelfRecorded(
        results=dict(outcome.record.results),
        record_path=outcome.path,
        fingerprint=outcome.record.content_fingerprint(),
    )


def should_record(record_enabled: bool) -> bool:
    """Whether a script should run the self-recording path at all."""
    return bool(record_enabled)


def id_is_recordable(experiment_id: str) -> bool:
    """Whether ``experiment_id`` can be the identity of a record (``EXP-<3+ digits>``)."""
    return bool(EXPERIMENT_ID_RE.match(experiment_id or ""))


# Two failure shapes, both non-zero and both explicit (never a success record):
#   1. the body raises -> run_experiment has already written a status="failed" record
#      (error type, message and traceback tail) and re-raised; we report and return 1.
#   2. validate_inputs rejects a declared input (missing corpus, missing tokenizer
#      artifact) -> that happens *before* the lifecycle starts, so no record exists and
#      stderr gets "[record] FAILED - ExperimentInputError: ...". The run cannot be
#      recorded because the record would have to hash inputs that are not there.


# Keys that vary between two otherwise identical runs. Project 002's manifests carry a
# `created_at` timestamp and an `artifact_dir` path; both would make the experiment
# fingerprint unstable, so they are dropped before anything enters `results`. The shared
# strip list in record.py is deliberately left untouched - this is about what a script
# chooses to put in its results, not about the record format.
VARIABLE_RESULT_KEYS = frozenset({"created_at", "generated_at", "artifact_dir", "output_dir"})


def stable_results(payload: Any) -> Any:
    """Recursively drop run-varying keys from a results payload."""
    if isinstance(payload, Mapping):
        return {k: stable_results(v) for k, v in payload.items() if k not in VARIABLE_RESULT_KEYS}
    if isinstance(payload, list):
        return [stable_results(v) for v in payload]
    return payload
