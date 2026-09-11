"""A dependency-free byte-level BPE — our own reference implementation.

Why this exists when we also wrap a library tokenizer:

* **No dependency floor.** The research harness runs anywhere Python runs, even without
  the optional ``tokenizers`` package.
* **Ablations we control.** Pre-tokenisation, merge accounting and special-token handling
  are visible and editable here.
* **A correctness oracle.** A simple implementation is a useful cross-check against a
  fast library implementation.

It is byte-level (the base vocabulary is the 256 possible byte values), so any UTF-8
string — Devanagari, emoji, ZWJ sequences, URLs — can be encoded, and decoding is
lossless. There is no UNK token unless you ask for one.

This is a *research reference*, not an optimised production tokenizer: it is O(merges ×
corpus) and is meant for corpora of a few megabytes at most.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from .base import SubwordTokenizer, TokenizerError, iter_segments
from .registry import register

FORMAT_NAME = "frontier-bpe-python"
FORMAT_VERSION = 1

PRETOKEN_PATTERN = "mark-aware-classifier (word | whitespace | punctuation runs)"

# Character classes used by the pre-tokenizer: word, whitespace, punctuation.
_CHAR_WORD = "W"
_CHAR_SPACE = "S"
_CHAR_PUNCT = "P"


@lru_cache(maxsize=8192)
def _char_class(ch: str) -> str:
    r"""Classify one character.

    IMPORTANT: combining marks (Unicode categories Mn/Mc/Me) count as *word* characters.
    Python's ``\w`` (and the GPT-2 regex used by many tokenizers, which matches ``\p{L}``
    and ``\p{N}``) exclude them, which silently shatters Indic syllables such as
    "मैं" into "म" + "ैं" — and because BPE merges never cross a pre-token boundary, the
    fragment can never be recovered. Keeping marks attached is the script-aware choice.
    """
    if ch.isspace():
        return _CHAR_SPACE
    if ch.isalnum() or unicodedata.category(ch).startswith("M"):
        return _CHAR_WORD
    return _CHAR_PUNCT


def pretokenize(text: str) -> list[str]:
    """Split text into the units BPE merges are learned over."""
    chunks: list[str] = []
    buffer: list[str] = []
    current: str | None = None
    for ch in text:
        kind = _char_class(ch)
        if kind != current and buffer:
            chunks.append("".join(buffer))
            buffer = []
        current = kind
        buffer.append(ch)
    if buffer:
        chunks.append("".join(buffer))
    return chunks


class PythonBPE(SubwordTokenizer):
    """Byte-level BPE trained with the classic count-and-merge loop."""

    name = "bpe_python"

    def __init__(self) -> None:
        self._merges: list[tuple[int, int]] = []
        self._ranks: dict[tuple[int, int], int] = {}
        self._special_tokens: list[str] = []
        self._special_ids: dict[str, int] = {}
        self._id_to_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        self._target_vocab_size = 256
        self._unk_token: str | None = None
        self._trained_chars = 0

    # ---------------------------------------------------------------- train --
    def train(
        self,
        corpus_path: str | Path,
        vocab_size: int,
        special_tokens: Sequence[str] = (),
        max_train_chars: int | None = None,
        **kwargs: object,
    ) -> None:
        """Fit merges on a local text file.

        Deterministic: merges are chosen by (frequency desc, pair asc), so the same
        corpus and vocab size always produce the same tokenizer.
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"training corpus not found: {path}")
        text = path.read_text(encoding="utf-8")
        if max_train_chars is not None:
            text = text[:max_train_chars]

        specials = list(dict.fromkeys(special_tokens))  # de-duplicate, keep order
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise TokenizerError(
                f"vocab_size={vocab_size} is too small: 256 byte tokens + {len(specials)} "
                "special tokens already exceed it"
            )
        if not text:
            raise TokenizerError(f"training corpus is empty: {path}")

        word_counts: Counter[tuple[int, ...]] = Counter()
        for chunk in pretokenize(text):
            word_counts[tuple(chunk.encode("utf-8"))] += 1

        merges: list[tuple[int, int]] = []
        for _ in range(n_merges):
            stats: Counter[tuple[int, int]] = Counter()
            for symbols, count in word_counts.items():
                for i in range(len(symbols) - 1):
                    stats[(symbols[i], symbols[i + 1])] += count
            if not stats:
                break  # no pairs left to merge
            best = sorted(stats.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            merges.append(best)
            word_counts = _apply_merge(word_counts, best, 256 + len(merges) - 1)

        self._merges = merges
        self._ranks = {pair: i for i, pair in enumerate(merges)}
        self._special_tokens = specials
        self._special_ids = {t: 256 + len(merges) + i for i, t in enumerate(specials)}
        self._id_to_bytes = [bytes([i]) for i in range(256)]
        for a, b in merges:
            self._id_to_bytes.append(self._id_to_bytes[a] + self._id_to_bytes[b])
        for token in specials:
            self._id_to_bytes.append(token.encode("utf-8"))
        self._target_vocab_size = vocab_size
        self._unk_token = None  # byte-level: nothing is out of vocabulary
        self._trained_chars = len(text)

    # ------------------------------------------------------------- encoding --
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for segment, is_special in iter_segments(text, self._special_tokens):
            if is_special:
                ids.append(self._special_ids[segment])
            else:
                for chunk in pretokenize(segment):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        symbols = list(chunk.encode("utf-8"))
        if len(symbols) < 2:
            return symbols
        while True:
            best_rank: int | None = None
            best_i = -1
            for i in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_rank is None:
                return symbols
            symbols[best_i : best_i + 2] = [256 + best_rank]

    def decode(self, ids: Sequence[int]) -> str:
        buf = bytearray()
        for i in ids:
            if i < 0 or i >= len(self._id_to_bytes):
                raise TokenizerError(f"token id {i} out of range (vocab_size={self.vocab_size})")
            buf.extend(self._id_to_bytes[i])
        return buf.decode("utf-8", errors="replace")

    # ----------------------------------------------------------------- meta --
    @property
    def vocab_size(self) -> int:
        return 256 + len(self._merges) + len(self._special_tokens)

    @property
    def special_tokens(self) -> list[str]:
        return list(self._special_tokens)

    @property
    def special_token_ids(self) -> dict[str, int]:
        return dict(self._special_ids)

    @property
    def unk_token(self) -> str | None:
        return self._unk_token

    @property
    def merges(self) -> list[tuple[int, int]]:
        return list(self._merges)

    def id_to_token(self, index: int) -> str:
        if 0 <= index < len(self._id_to_bytes):
            return self._id_to_bytes[index].decode("utf-8", errors="replace")
        raise TokenizerError(f"token id {index} out of range (vocab_size={self.vocab_size})")

    def impl_version(self) -> str:
        return f"{FORMAT_NAME}-{FORMAT_VERSION}"

    # ------------------------------------------------------------------ io --
    def save(self, out_dir: str | Path) -> Path:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_json(
            directory / "bpe_python.json",
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "vocab_size": self.vocab_size,
                "special_tokens": self._special_tokens,
                "pretoken_pattern": PRETOKEN_PATTERN,
                "trained_chars": self._trained_chars,
                "merges": [[a, b] for a, b in self._merges],
            },
        )
        return directory

    @classmethod
    def load(cls, artifact_dir: str | Path) -> PythonBPE:
        path = Path(artifact_dir) / "bpe_python.json"
        if not path.exists():
            raise FileNotFoundError(f"no bpe_python artifact at {path}")
        data = cls.read_json(path)
        if data.get("format") != FORMAT_NAME:
            raise TokenizerError(f"{path}: expected format '{FORMAT_NAME}', got '{data.get('format')}'")

        tok = cls()
        tok._merges = [(int(a), int(b)) for a, b in data["merges"]]
        tok._ranks = {pair: i for i, pair in enumerate(tok._merges)}
        tok._special_tokens = list(data.get("special_tokens", []))
        tok._id_to_bytes = [bytes([i]) for i in range(256)]
        for a, b in tok._merges:
            tok._id_to_bytes.append(tok._id_to_bytes[a] + tok._id_to_bytes[b])
        for token in tok._special_tokens:
            tok._id_to_bytes.append(token.encode("utf-8"))
        tok._special_ids = {t: 256 + len(tok._merges) + i for i, t in enumerate(tok._special_tokens)}
        tok._target_vocab_size = int(data.get("vocab_size", tok.vocab_size))
        tok._trained_chars = int(data.get("trained_chars", 0))
        return tok


def _apply_merge(
    word_counts: Counter, pair: tuple[int, int], new_id: int
) -> Counter:
    """Replace every occurrence of ``pair`` in all words with ``new_id``."""
    updated: Counter = Counter()
    first, second = pair
    for symbols, count in word_counts.items():
        out: list[int] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                out.append(new_id)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        updated[tuple(out)] += count
    return updated


register(PythonBPE.name, PythonBPE)
