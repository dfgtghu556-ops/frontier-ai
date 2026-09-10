import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def pinned_threads():
    """Start every test from a single-threaded torch.

    Multi-threaded CPU reductions can change floating-point summation order, and
    ``Trainer`` sets a thread count of its own (``threads_for``), so a test that
    trains leaves a different global thread count behind than the one it started
    with. Experiment records capture ``environment.torch.num_threads`` *before*
    the run body executes, so without this pin two identical sweeps in one
    process can record different environments — and therefore different content
    fingerprints — depending on which test ran first. See Q-13 in DECISIONS.md.
    """
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def tiny_text() -> str:
    """A few hundred characters of structured text, deterministic across runs."""
    from frontier_ai.data.synthetic import generate_corpus

    return generate_corpus(target_chars=4000, seed=7)
