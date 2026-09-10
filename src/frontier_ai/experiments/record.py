"""The experiment record: one JSON document per run.

Layout (mirrors the standard entry format in EXPERIMENTS.md)::

    {
      "schema_version": "1.0",
      "record_type": "frontier-ai.experiment",
      "experiment": {...},     # id, name, tags, notes, status
      "code": {...},           # git provenance (commit, branch, dirty, reason if unavailable)
      "data": {...},           # deterministic input digests (or an explicit none/unavailable)
      "configuration": {...},  # spec + embedded config file + CLI overrides
      "randomness": {...},     # master seed, derived seeds, determinism request, limitations
      "environment": {...},    # python / platform / selected package versions
      "execution": {...},      # command, cwd, exit code, timings, stdout/stderr tails
      "results": {...}         # whatever the experiment itself returned
    }

Two kinds of fields live in here and must not be confused:

* **content fields** — everything that can change the result (code, data, config, seed,
  environment, command). These define the experiment's identity.
* **variable fields** — wall-clock timestamps, duration, pid, and output locations. These
  necessarily differ between two runs of the *same* experiment and are therefore excluded
  from :func:`content_fingerprint`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECORD_SCHEMA_VERSION = "1.0"
RECORD_TYPE = "frontier-ai.experiment"
RECORD_FILENAME = "experiment.json"

# Keys stripped from the fingerprint because they vary between identical runs.
VARIABLE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "started_at",
        "finished_at",
        "duration_seconds",
        "pid",
        "cwd",
        "output_dir",
        "artifact_dir",
        "record_path",
        "stdout_tail",
        "stderr_tail",
    }
)


class ExperimentRecordError(ValueError):
    """Raised when a record cannot be read or is malformed."""


@dataclass
class ExperimentRecord:
    """A complete, machine-readable description of one experiment run."""

    experiment: dict[str, Any]
    code: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    randomness: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RECORD_SCHEMA_VERSION
    record_type: str = RECORD_TYPE

    # ---------------------------------------------------------------- io --
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "experiment": self.experiment,
            "code": self.code,
            "data": self.data,
            "configuration": self.configuration,
            "randomness": self.randomness,
            "environment": self.environment,
            "execution": self.execution,
            "results": self.results,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentRecord:
        if not isinstance(data, Mapping):
            raise ExperimentRecordError("experiment record must be a JSON object")
        version = data.get("schema_version")
        if version != RECORD_SCHEMA_VERSION:
            raise ExperimentRecordError(
                f"unsupported record schema_version {version!r} (expected {RECORD_SCHEMA_VERSION!r}); "
                "migrate the record or read it with the matching version of this package"
            )
        known = {
            "experiment", "code", "data", "configuration", "randomness",
            "environment", "execution", "results", "schema_version", "record_type",
        }
        extra = set(data) - known
        if extra:
            raise ExperimentRecordError(f"unknown record sections: {sorted(extra)}")
        return cls(**{k: dict(data.get(k, {})) for k in known - {"schema_version", "record_type"}})

    @classmethod
    def load(cls, path: str | Path) -> ExperimentRecord:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------- fingerprint --
    def content_dict(self) -> dict:
        """Record with variable runtime metadata removed."""
        return strip_variable(self.to_dict())

    def content_fingerprint(self) -> str:
        """Stable SHA-256 over the *content* of the experiment.

        Two runs of the same experiment (same code, data, config, seed, environment and
        command) produce the same fingerprint even though their timestamps differ.
        """
        return content_fingerprint_of(self.to_dict())

    # -------------------------------------------------------------- view --
    def render(self) -> str:
        """Human-readable summary (the JSON file stays the source of truth)."""
        exp = self.experiment
        git = self.code.get("git", {})
        data = self.data
        rand = self.randomness
        env = self.environment
        exec_ = self.execution

        lines = [
            f"Experiment {exp.get('id')} — {exp.get('name') or '(unnamed)'}",
            "=" * 60,
            f"status        : {exp.get('status')}",
            f"code          : {_fmt_git(git)}",
            f"data          : {_fmt_data(data)}",
            f"seed          : {rand.get('master_seed')} "
            f"(deterministic mode: {rand.get('deterministic_mode_requested')})",
            f"environment   : python {env.get('python', {}).get('version')} / "
            f"torch {env.get('torch', {}).get('version')} / {env.get('platform', {}).get('system')}",
            f"execution     : {exec_.get('status')}"
            + (f" (exit {exec_.get('exit_code')})" if exec_.get("exit_code") is not None else ""),
            f"duration      : {exec_.get('duration_seconds')}s",
        ]
        if exec_.get("command"):
            lines.append(f"command       : {' '.join(exec_['command'])}")
        if exp.get("tags"):
            lines.append(f"tags          : {', '.join(exp['tags'])}")
        if exec_.get("error"):
            lines.append(f"error         : {exec_['error'].get('type')}: {exec_['error'].get('message')}")
        if self.results:
            lines.append("")
            lines.append("results:")
            for key, value in self.results.items():
                lines.append(f"  {key:<24} {value}")
        content_fp = self.content_fingerprint()
        lines.append("")
        lines.append(f"content fingerprint: {content_fp[:16]}… (timestamps excluded)")
        if git.get("dirty"):
            lines.append("WARNING: ran from a dirty working tree — not reproducible from the commit alone")
        if not git.get("available", False):
            lines.append("WARNING: git provenance unavailable — this run cannot be traced to a commit")
        return "\n".join(lines)


def strip_variable(payload: Any) -> Any:
    """Recursively drop keys listed in :data:`VARIABLE_FIELD_NAMES`.

    Public because other record types (see :mod:`frontier_ai.experiments.sweep`) must strip
    exactly the same fields — two record types that disagree about what is variable would
    break the whole point of a fingerprint.
    """
    if isinstance(payload, dict):
        return {k: strip_variable(v) for k, v in payload.items() if k not in VARIABLE_FIELD_NAMES}
    if isinstance(payload, list):
        return [strip_variable(v) for v in payload]
    return payload


def content_fingerprint_of(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON of ``payload`` with variable fields removed.

    Shared by every record type so "the same experiment" means the same thing everywhere.
    """
    canonical = json.dumps(strip_variable(payload), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    if data.get("status") == "unavailable":
        return f"unavailable ({data.get('reason', 'unknown reason')})"
    if isinstance(digest, dict):
        return f"{digest.get('file_count')} file(s), {digest.get('digest', '')[:12]}…"
    return str(digest)
