"""Adapters that let the Project 001 tokenizers be evaluated in the same harness.

The Project 001 char/word tokenizers stay exactly as they are — they are still used by
the training pipeline. These adapters simply wrap them behind the research
:class:`SubwordTokenizer` interface so that "character-level" and "word-level" can be
measured against the same fixture as any subword tokenizer.

Special-token support is implemented here (split on specials before encoding) so that no
change to the Project 001 code is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..data.tokenizer import CharTokenizer, Tokenizer, WordTokenizer
from .base import SubwordTokenizer, TokenizerError, iter_segments
from .registry import register

UNKNOWN_CHAR = "\ufffd"


class _Project001Adapter(SubwordTokenizer):
    """Shared machinery for the char/word adapters."""

    wrapped_class: type[Tokenizer]

    def __init__(self) -> None:
        self._tokenizer: Tokenizer | None = None
        self._special_tokens: list[str] = []

    # ---------------------------------------------------------------- train --
    def train(
        self,
        corpus_path: str | Path,
        vocab_size: int = 0,  # ignored: the corpus determines the vocabulary
        special_tokens: Sequence[str] = (),
        **kwargs: object,
    ) -> None:
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"training corpus not found: {path}")
        text = path.read_text(encoding="utf-8")
        if not text:
            raise TokenizerError(f"training corpus is empty: {path}")

        base = self._fit(text, **kwargs)
        specials = [t for t in dict.fromkeys(special_tokens) if t not in set(base.itos)]
        itos = list(base.itos) + specials
        self._tokenizer = self.wrapped_class(itos)
        self._special_tokens = specials

    def _fit(self, text: str, **kwargs: object) -> Tokenizer:
        raise NotImplementedError

    # ------------------------------------------------------------- encoding --
    def _require(self) -> Tokenizer:
        if self._tokenizer is None:
            raise TokenizerError("tokenizer is not trained or loaded; call train() or load() first")
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for segment, is_special in iter_segments(text, self._special_tokens):
            if is_special:
                ids.append(self._require().stoi[segment])
            else:
                ids.extend(self._require().encode(segment))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._require().decode(ids)

    # ----------------------------------------------------------------- meta --
    @property
    def vocab_size(self) -> int:
        return self._require().vocab_size

    @property
    def special_tokens(self) -> list[str]:
        return list(self._special_tokens)

    @property
    def special_token_ids(self) -> dict[str, int]:
        stoi = self._require().stoi
        return {t: stoi[t] for t in self._special_tokens if t in stoi}

    def id_to_token(self, index: int) -> str:
        itos = self._require().itos
        if 0 <= index < len(itos):
            return itos[index]
        raise TokenizerError(f"token id {index} out of range (vocab_size={self.vocab_size})")

    @property
    def wrapped(self) -> Tokenizer:
        """The underlying Project 001 tokenizer (for reuse in the training pipeline)."""
        return self._require()

    # ------------------------------------------------------------------ io --
    def save(self, out_dir: str | Path) -> Path:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._require().save(directory / "tokenizer.json")
        self.save_json(
            directory / "adapter_config.json",
            {
                "format": f"project001-{self.name}",
                "special_tokens": self._special_tokens,
                "base_impl": self.wrapped_class.__name__,
            },
        )
        return directory

    @classmethod
    def load(cls, artifact_dir: str | Path) -> _Project001Adapter:
        path = Path(artifact_dir) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"no Project 001 tokenizer artifact at {path}")
        obj = cls()
        obj._tokenizer = Tokenizer.load(path)
        config_path = Path(artifact_dir) / "adapter_config.json"
        if config_path.exists():
            obj._special_tokens = list(cls.read_json(config_path).get("special_tokens", []))
        return obj


class CharTokenizerAdapter(_Project001Adapter):
    """Character-level baseline (Project 001 ``CharTokenizer``)."""

    name = "char"
    wrapped_class = CharTokenizer

    def _fit(self, text: str, **kwargs: object) -> Tokenizer:
        return CharTokenizer.fit(text)

    def impl_version(self) -> str:
        return "project001-char-1"

    @property
    def unk_token(self) -> str:
        return UNKNOWN_CHAR

    @property
    def unk_token_id(self) -> int | None:
        return self._require().stoi.get(UNKNOWN_CHAR)


class WordTokenizerAdapter(_Project001Adapter):
    """Word-level baseline (Project 001 ``WordTokenizer``)."""

    name = "word"
    wrapped_class = WordTokenizer

    def _fit(self, text: str, min_count: int = 1, **kwargs: object) -> Tokenizer:
        return WordTokenizer.fit(text, min_count=int(min_count))

    def impl_version(self) -> str:
        return "project001-word-1"

    @property
    def unk_token(self) -> str:
        return self.wrapped.unk if isinstance(self.wrapped, WordTokenizer) else "<unk>"

    @property
    def unk_token_id(self) -> int | None:
        stoi = self._require().stoi
        return stoi.get("<unk>")


register(CharTokenizerAdapter.name, CharTokenizerAdapter)
register(WordTokenizerAdapter.name, WordTokenizerAdapter)
