import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def pinned_threads():
    """Start every test from a single-threaded torch.

    This is about **numerical** determinism, not provenance: multi-threaded CPU
    reductions can change floating-point summation order, so two otherwise identical
    training runs can differ in the last bits. Provenance is handled by the runner
    itself, which records the thread count the run actually used and restores the
    caller's value afterwards (D-034, formerly Q-13) — see
    ``tests/test_experiments.py`` section F2 and ``tests/test_nested_metrics.py``.
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
