"""The pluggable tokenizer interface used by the tokenizer research subsystem.

Design rules
------------
* Anything that can turn text into ids and back can be plugged in: our own BPE, a
  library tokenizer, or an adapter over the Project 001 char/word tokenizers.
* The training engine is **not** touched by this subsystem: a future tokenizer just
  implements this interface and gets evaluated by the same harness.
* Implementations must be savable to a directory and loadable back, so that an
  artifact is a reproducible object rather than in-memory state.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPECIAL_SEGMENT = "special"


@dataclass(frozen=True)
class TokenizerInfo:
    """Everything needed to identify a tokenizer artifact later."""

    impl: str
    impl_version: str
    vocab_size: int
    special_tokens: list[str] = field(default_factory=list)
    unk_token: str | None = None
    train_params: dict[str, Any] = field(default_factory=dict)
    artifact_dir: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class TokenizerError(RuntimeError):
    """Raised for tokenizer misconfiguration or unsupported operations."""


def compile_special_pattern(special_tokens: Sequence[str]) -> re.Pattern | None:
    """Regex that matches any special token, longest first (so '<pad>x' can't shadow '<pad>')."""
    cleaned = [t for t in special_tokens if t]
    if not cleaned:
        return None
    ordered = sorted(set(cleaned), key=len, reverse=True)
    return re.compile("(" + "|".join(re.escape(t) for t in ordered) + ")")


def iter_segments(text: str, special_tokens: Sequence[str]) -> list[tuple[str, bool]]:
    """Split ``text`` into ``(segment, is_special_token)`` pairs.

    Used by implementations that do not handle special tokens natively (our own BPE and
    the Project 001 adapters), so special tokens always receive their own id.
    """
    pattern = compile_special_pattern(special_tokens)
    if pattern is None:
        return [(text, False)] if text else []
    segments: list[tuple[str, bool]] = []
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            segments.append((text[last : match.start()], False))
        segments.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        segments.append((text[last:], False))
    return segments


class SubwordTokenizer(ABC):
    """Common interface for every tokenizer evaluated by this subsystem."""

    name: str = "base"

    # ---------------------------------------------------------------- train --
    @abstractmethod
    def train(
        self,
        corpus_path: str | Path,
        vocab_size: int,
        special_tokens: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        """Fit the tokenizer on a **local** text file (no network access)."""

    # ------------------------------------------------------------- encoding --
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Text -> ids."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Ids -> text. Byte-level implementations are expected to be lossless."""

    # ----------------------------------------------------------------- meta --
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of ids the tokenizer can emit, including special tokens."""

    @property
    def special_tokens(self) -> list[str]:
        return []

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {}

    @property
    def unk_token(self) -> str | None:
        """``None`` means the tokenizer cannot produce unknowns (byte-level fallback)."""
        return None

    @property
    def unk_token_id(self) -> int | None:
        return None

    # ------------------------------------------------------------------ io --
    @abstractmethod
    def save(self, out_dir: str | Path) -> Path:
        """Write the tokenizer files into ``out_dir`` and return it."""

    @classmethod
    @abstractmethod
    def load(cls, artifact_dir: str | Path) -> SubwordTokenizer:
        """Load from a directory written by :meth:`save`."""

    # ------------------------------------------------------------- helpers --
    def round_trip(self, text: str) -> bool:
        return self.decode(self.encode(text)) == text

    def info(self) -> TokenizerInfo:
        return TokenizerInfo(
            impl=self.name,
            impl_version=self.impl_version(),
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens),
            unk_token=self.unk_token,
        )

    def impl_version(self) -> str:
        return "unknown"

    def save_json(self, path: str | Path, payload: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def read_json(path: str | Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"
