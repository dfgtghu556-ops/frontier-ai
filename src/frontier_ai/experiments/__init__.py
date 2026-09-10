"""Reproducible experiment infrastructure (Project 003, ROADMAP Stage 1).

Answers one question for any run: *exactly what code, configuration, data, seed,
environment and command produced this result?*

    from frontier_ai.experiments import ExperimentSpec, run_experiment

    spec = ExperimentSpec(experiment_id="EXP-003", seed=1337, name="demo",
                          output_dir="out/experiments/demo", data_paths=["data/synthetic.bin"])
    outcome = run_experiment(spec, my_experiment_fn)
    print(outcome.record.content_fingerprint())   # stable across identical runs

Components:
    spec.py         experiment specification + validation + JSON round-trip
    gitinfo.py      commit / branch / dirty-state capture (never invents a SHA)
    hashing.py      deterministic input digests (re-uses Project 002's SHA-256 helpers)
    seeding.py      one master seed + derived component seeds + documented limits
    environment.py  selected, stable environment metadata (no env dumps)
    record.py       the experiment record: sections, fingerprint, save/load, render
    runner.py       lifecycle runner for Python callables and subprocess commands
    examples.py     small reference experiments used by docs and determinism tests
"""

from .environment import capture_environment
from .gitinfo import GitInfo, capture_git_info
from .hashing import DataDigest, digest_paths, hash_paths, sha256_file, sha256_text
from .record import RECORD_FILENAME, RECORD_SCHEMA_VERSION, ExperimentRecord, ExperimentRecordError
from .runner import ExperimentContext, ExperimentInputError, RunOutcome, run_command, run_experiment
from .seeding import SEED_LIMITATIONS, derive_seed, derive_seeds, seed_everything
from .spec import SPEC_SCHEMA_VERSION, ExperimentSpec, ExperimentSpecError

__all__ = [
    "ExperimentSpec",
    "ExperimentSpecError",
    "SPEC_SCHEMA_VERSION",
    "ExperimentRecord",
    "ExperimentRecordError",
    "RECORD_SCHEMA_VERSION",
    "RECORD_FILENAME",
    "ExperimentContext",
    "ExperimentInputError",
    "RunOutcome",
    "run_experiment",
    "run_command",
    "GitInfo",
    "capture_git_info",
    "DataDigest",
    "hash_paths",
    "digest_paths",
    "sha256_file",
    "sha256_text",
    "seed_everything",
    "derive_seed",
    "derive_seeds",
    "SEED_LIMITATIONS",
    "capture_environment",
]
