"""Minimal, dependency-free tokenizers.

Two levels are supported, both fully invertible and JSON-serializable:

* ``char``  - one token per character. Tiny vocab, learns fast, great for smoke tests.
* ``word``  - regex word/punctuation tokens with an optional frequency cutoff.

For anything serious you would swap in a real BPE (tiktoken / HF tokenizers); the
`Tokenizer` interface here is deliberately the same shape so that is a small change.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

# Matches a word, a run of digits, or a single non-space character (punctuation).
WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|\s+|.", re.UNICODE)


class Tokenizer:
    """Base class: int <-> str with save/load."""

    kind = "base"

    def __init__(self, itos: list[str]) -> None:
        self.itos = list(itos)
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("tokenizer vocab contains duplicates")

    # ---------------------------------------------------------------- api --
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # ----------------------------------------------------------------- io --
    def to_dict(self) -> dict:
        return {"kind": self.kind, "itos": self.itos}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = data.get("kind", "char")
        klass = {"char": CharTokenizer, "word": WordTokenizer}.get(kind)
        if klass is None:
            raise ValueError(f"unknown tokenizer kind '{kind}'")
        return klass(itos=data["itos"])

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"


class CharTokenizer(Tokenizer):
    """One token per character (or multi-byte unicode code point)."""

    kind = "char"

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] if ch in self.stoi else self.stoi.get("\ufffd", -1) for ch in text]

    @classmethod
    def fit(cls, text: str) -> CharTokenizer:
        vocab = sorted(set(text))
        if "\ufffd" not in vocab:  # replacement token for unknown chars at inference
            vocab.append("\ufffd")
        return cls(vocab)


class WordTokenizer(Tokenizer):
    """Regex word tokenizer with `<unk>` for rare/unseen tokens."""

    kind = "word"

    def __init__(self, itos: list[str], unk: str = "<unk>") -> None:
        super().__init__(itos)
        self.unk = unk
        self.unk_id = self.stoi.get(unk, 0)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in WORD_RE.findall(text)]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @classmethod
    def fit(cls, text: str, min_count: int = 1, vocab_cap: int = 32000) -> WordTokenizer:
        from collections import Counter

        counts = Counter(WORD_RE.findall(text))
        kept = [tok for tok, c in counts.most_common() if c >= min_count]
        # reserve slot 0 for <unk>, then fill with the most frequent tokens
        itos = ["<unk>"] + kept[: max(0, vocab_cap - 1)]
        return cls(itos)


def fit_tokenizer(text: str, level: str = "char", min_count: int = 1) -> Tokenizer:
    if level == "char":
        return CharTokenizer.fit(text)
    if level == "word":
        return WordTokenizer.fit(text, min_count=min_count)
    raise ValueError(f"unknown tokenizer level '{level}'")
