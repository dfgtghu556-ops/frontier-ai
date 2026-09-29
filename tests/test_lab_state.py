"""Tests for the Lab OS snapshot exporter (frontier_ai.lab_state, scripts/export_lab_state.py).

The dashboard may only show what the repository records. These tests pin the parsers to the
append-only record formats, check that the hand-maintained lab/registry.json is cross-checked
against EXPERIMENTS.md / DECISIONS.md / MASTER_CONTEXT §37, and fail while the committed
snapshot is stale.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_ai.lab_state import (
    DEFAULT_OUTPUT,
    REGISTRY_PATH,
    build_state,
    parse_decisions,
    parse_experiments,
    parse_roadmap_titles,
    render,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = """\
# Experiments

### EXP-001 — First thing (§37 step 3)

**Date:** 2026-09-10 · **Status:** in progress

- **Objective:** measure the first thing
  across two lines.
- **Result:** fine.

**Status:** complete → D-001.

### EXP-002 — Second thing

**Date:** 2026-09-11 · **Status:** pre-registered

**Purpose:** plan the second thing.
"""

DECISIONS = """\
# Decisions

## D-001 — Adopt the first thing

**Date:** 2026-09-10 · **Status:** accepted

**Decision:** we adopt it, per EXP-001.

## D-002 — Tentative choice

**Date:** 2026-09-11 · **Status:** provisional

**Decision:** maybe.

## Open items

| ID | Question | Deferred to |
|---|---|---|
| Q-1 | Which size? | Stage 2 |
"""


def test_experiment_status_is_the_last_status_line():
    exps = parse_experiments(EXPERIMENTS)
    assert [e["id"] for e in exps] == ["EXP-001", "EXP-002"]
    first, second = exps
    assert first["category"] == "complete"  # the follow-up line wins over the stale header
    assert first["step"] == 3
    assert first["date"] == "2026-09-10"
    assert first["decisions"] == ["D-001"]
    assert first["summary"] == "measure the first thing across two lines."
    assert second["category"] == "planned"
    assert second["summary"] == "plan the second thing."
    assert "?plain=1#L" in first["url"]


def test_decisions_and_open_questions():
    decs, questions = parse_decisions(DECISIONS)
    assert [(d["id"], d["category"]) for d in decs] == [("D-001", "accepted"), ("D-002", "provisional")]
    assert decs[0]["experiments"] == ["EXP-001"]
    assert questions == [{"id": "Q-1", "question": "Which size?", "deferred_to": "Stage 2"}]


def test_roadmap_has_24_steps_from_master_context():
    titles = parse_roadmap_titles((ROOT / "MASTER_CONTEXT.md").read_text(encoding="utf-8"))
    assert len(titles) == 24
    assert all(t for t in titles)


@pytest.fixture(scope="module")
def state():
    return build_state(ROOT)


def test_real_repository_builds(state):
    assert state["schema"] == "frontier-lab-state-v1"
    assert len(state["roadmap"]) == 24
    ids = {e["id"] for e in state["experiments"]}
    assert {"EXP-001", "EXP-031", "EXP-033"} <= ids
    assert state["experiment_numbers_not_in_log"] == list(range(8, 18))  # kept in docs/tokenizer_corpus_*
    assert state["tokenizer"]["vocab_size"] == 32768
    assert state["suite"]["id"] == "frontier-heldout-v1"
    assert state["evaluation_reports"], "published EXP-031 reports must be read"
    assert state["baseline_model"]["record"]["seeds"] == 3
    # hashes of every input, and nothing time-dependent
    assert set(state["sources"]) >= {"EXPERIMENTS.md", "DECISIONS.md", REGISTRY_PATH}
    assert "generated_at" not in state


def test_no_local_paths_leak_into_the_snapshot(state):
    text = render(state)
    assert "E:/" not in text and "E:\\\\" not in text and "C:\\\\Users" not in text


def test_export_is_deterministic():
    assert render(build_state(ROOT)) == render(build_state(ROOT))


def _inputs():
    exps = parse_experiments((ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8"))
    decs, _ = parse_decisions((ROOT / "DECISIONS.md").read_text(encoding="utf-8"))
    titles = parse_roadmap_titles((ROOT / "MASTER_CONTEXT.md").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / REGISTRY_PATH).read_text(encoding="utf-8"))
    return registry, exps, decs, titles


def test_committed_registry_is_valid():
    registry, exps, decs, titles = _inputs()
    assert validate_registry(registry, ROOT, exps, decs, titles) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r: r["roadmap_steps"][0].update(title="Something else"), "title"),
        (lambda r: r["roadmap_steps"][0]["evidence"].append("EXP-999"), "EXP-999"),
        (lambda r: r["roadmap_steps"][0]["evidence"].append("D-999"), "D-999"),
        (lambda r: r["roadmap_steps"][0]["evidence_files"].append("no/such/file.md"), "no/such/file.md"),
        (lambda r: r["roadmap_steps"][20].update(status="complete"), "complete after an incomplete"),
        (lambda r: r["roadmap_steps"][0].update(status="done"), "status must be"),
        (lambda r: r["roadmap_steps"].pop(), "steps 1..24"),
    ],
)
def test_registry_mistakes_are_caught(mutate, expected):
    registry, exps, decs, titles = _inputs()
    bad = copy.deepcopy(registry)
    mutate(bad)
    problems = validate_registry(bad, ROOT, exps, decs, titles)
    assert any(expected in p for p in problems), problems


def test_complete_step_cannot_cite_unfinished_work():
    registry, exps, decs, titles = _inputs()
    planned = next(e["id"] for e in exps if e["category"] == "planned")
    provisional = next(d["id"] for d in decs if d["category"] == "provisional")
    bad = copy.deepcopy(registry)
    bad["roadmap_steps"][0]["evidence"] += [planned, provisional]
    problems = validate_registry(bad, ROOT, exps, decs, titles)
    assert any(planned in p and "planned" in p for p in problems), problems
    assert any(provisional in p and "provisional" in p for p in problems), problems


def test_committed_snapshot_is_current():
    """Fails when records changed but `python scripts/export_lab_state.py` was not re-run."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_lab_state.py"), "--check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (ROOT / DEFAULT_OUTPUT).is_file()
