import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def tiny_text() -> str:
    """A few hundred characters of structured text, deterministic across runs."""
    from frontier_ai.data.synthetic import generate_corpus

    return generate_corpus(target_chars=4000, seed=7)
