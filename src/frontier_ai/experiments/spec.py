"""Experiment specification: the *inputs* that define a reproducible experiment.

This is the central configuration mechanism requested by ROADMAP Stage 1. It is
deliberately not a general hyperparameter system — Project 001 already owns
:class:`~frontier_ai.config.ExperimentConfig` for model training, and Project 002 owns its
own artifact configuration. Those stay as they are; an :class:`ExperimentSpec` *wraps* them
by pointing at their config file and recording it verbatim.

A spec answers: which experiment, under what identity, with what seed, over what data,
where do outputs go, and what free-form parameters describe it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPEC_SCHEMA_VERSION = "1.0"

# EXP- followed by 3+ digits; extend later if the lab outgrows it.
EXPERIMENT_ID_RE = re.compile(r"^EXP-\d{3,}$")

SEED_MIN = 0
SEED_MAX = 2**32 - 1


class ExperimentSpecError(ValueError):
    """Raised when an experiment specification is malformed."""


@dataclass
class ExperimentSpec:
    """Inputs that identify one experiment run."""

    experiment_id: str
    seed: int = 1337
    name: str = ""
    output_dir: str = "out/experiments/EXP-000"
    # Free-form parameters: anything the experiment itself needs (hyperparameters,
    # dataset version strings, ablation axis, ...). Kept as a flat JSON-friendly dict so
    # it serializes deterministically (keys are sorted on write).
    params: dict[str, Any] = field(default_factory=dict)
    # Data / corpus inputs that will be hashed for provenance.
    data_paths: list[str] = field(default_factory=list)
    # Optional path to an existing config file (Project 001 ExperimentConfig, a tokenizer
    # spec, ...). Its contents are embedded verbatim in the record.
    config_path: str = ""
    # Optional identity of the artifact under study (model checkpoint, tokenizer artifact).
    component: dict[str, str] = field(default_factory=dict)
    # Command that will be executed (used by scripts/experiment_record.py).
    command: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    deterministic_mode: bool = False  # request torch deterministic algorithms

    def __post_init__(self) -> None:
        if not EXPERIMENT_ID_RE.match(self.experiment_id):
            raise ExperimentSpecError(
                f"experiment_id must match EXP-<3+ digits>, got {self.experiment_id!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ExperimentSpecError(f"seed must be an int, got {type(self.seed).__name__}")
        if not SEED_MIN <= self.seed <= SEED_MAX:
            raise ExperimentSpecError(f"seed must be in [{SEED_MIN}, {SEED_MAX}], got {self.seed}")
        if not str(self.output_dir).strip():
            raise ExperimentSpecError("output_dir must not be empty")
        if self.params and not isinstance(self.params, dict):
            raise ExperimentSpecError("params must be a mapping")
        for key in self.params:
            if not isinstance(key, str):
                raise ExperimentSpecError(f"params keys must be strings, got {key!r}")
        if self.data_paths is None:
            self.data_paths = []
        if any(not isinstance(p, str) or not p.strip() for p in self.data_paths):
            raise ExperimentSpecError("data_paths entries must be non-empty strings")
        if self.command is None:
            self.command = []
        if any(not isinstance(c, str) for c in self.command):
            raise ExperimentSpecError("command entries must be strings")

    # ----------------------------------------------------------------- io --
    def to_dict(self) -> dict:
        data = asdict(self)
        data["schema_version"] = SPEC_SCHEMA_VERSION
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentSpec:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known - {"schema_version"}
        if unknown:
            raise ExperimentSpecError(f"unknown ExperimentSpec keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ExperimentSpecError(f"{path}: expected a JSON object at the top level")
        return cls.from_dict(data)

    def with_overrides(self, overrides: list[str] | None) -> ExperimentSpec:
        """Apply ``key=value`` overrides (flat keys only: seed, output_dir, params.x)."""
        import copy

        spec = copy.deepcopy(self)
        for item in overrides or []:
            if "=" not in item:
                raise ExperimentSpecError(f"override must look like key=value, got '{item}'")
            key, raw = item.split("=", 1)
            if key.startswith("params."):
                spec.params[key[len("params.") :]] = coerce_value(raw)
                continue
            if key not in {"seed", "name", "output_dir", "notes",
                           "config_path", "deterministic_mode"}:
                raise ExperimentSpecError(
                    f"cannot override '{key}' (use params.<name>=value for free-form parameters)"
                )
            current = getattr(spec, key)
            if isinstance(current, bool):
                setattr(spec, key, raw.lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                setattr(spec, key, int(raw))
            else:
                setattr(spec, key, raw)
        spec.__post_init__()
        return spec


def coerce_value(raw: str) -> Any:
    """Parse a CLI string into a JSON value (falling back to the raw string).

    Public because the sweep CLI parses ``--config NAME:params.x=...`` with the same rules
    as ``--set params.x=...``; two parsers that disagree would be a bug.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
