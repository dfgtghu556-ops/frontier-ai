"""Tokenizer research subsystem (Project 002).

Pluggable tokenizers, a deterministic Indic probe corpus, an evaluator and a comparator.
Importing this package registers every implementation that is currently importable;
optional dependencies that are missing simply do not register.

    from frontier_ai.tokenization import available, create
    print(available())            # {'bpe_python': '...PythonBPE', 'bpe_hf': '...', ...}
    tok = create("bpe_python")
    tok.train("data/tokenizer/indic-v1/train.txt", vocab_size=512, special_tokens=["<pad>"])
"""

from . import artifact, bpe_python, compare, corpus, evaluate, registry  # noqa: F401
from .adapters import CharTokenizerAdapter, WordTokenizerAdapter  # noqa: F401
from .base import SubwordTokenizer, TokenizerError, TokenizerInfo  # noqa: F401
from .registry import available, create, is_available, names  # noqa: F401

try:  # optional dependency: registers 'bpe_hf' when installed
    from . import bpe_hf  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    bpe_hf = None  # type: ignore[assignment]

__all__ = [
    "artifact",
    "bpe_python",
    "compare",
    "corpus",
    "evaluate",
    "registry",
    "SubwordTokenizer",
    "TokenizerError",
    "TokenizerInfo",
    "CharTokenizerAdapter",
    "WordTokenizerAdapter",
    "available",
    "create",
    "is_available",
    "names",
]
