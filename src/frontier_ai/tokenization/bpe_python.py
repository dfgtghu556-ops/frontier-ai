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

This is a *research reference*, but fast enough for corpus-scale training: the
merge loop maintains the pair counts incrementally (only the words containing the
merged pair are touched each iteration) instead of re-scanning the whole corpus per
merge. The incremental update maintains *exactly* the same pair counts as the classic
full re-scan, and the tie-break (frequency desc, pair asc) is unchanged, so the merge
sequence is identical — a regression test asserts that against a full-rescan
implementation.
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
    """Split text into the units BPE merges are learned over (mark-aware)."""
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


@lru_cache(maxsize=8192)
def _char_class_gpt2(ch: str) -> str:
    """GPT-2-style character classes: letter run, number run, whitespace, other.

    Mirrors the classes of the GPT-2 pre-tokenization regex
    (`` ?\\p{L}+ | ?\\p{N}+ | ?[^\\s\\p{L}\\p{N}]+ | \\s+(?!\\S) | \\s``) with stdlib
    only: ``\\p{L}`` -> Unicode category ``L*``, ``\\p{N}`` -> ``N*``. Combining marks
    (``M*``) are **not** letters in that regex, so they fall into ``other`` — which is
    precisely the documented difference from the mark-aware pre-tokenizer (Indic
    syllables shatter at their marks; BPE then never re-joins them across the
    pre-token boundary).
    """
    if ch.isspace():
        return "S"
    cat = unicodedata.category(ch)
    if cat[0] == "L":
        return "L"
    if cat[0] == "N":
        return "N"
    return "P"


def pretokenize_gpt2_style(text: str) -> list[str]:
    """GPT-2-style split, stdlib-only, faithful to the regex's chunk boundaries:

    * letter runs and number runs are separate chunks; a single space immediately
      before such a run belongs to that run (the regex's `` ?`` prefix);
    * every other character (punctuation, marks, symbols) forms runs of its own —
      combining marks land here, so "मैं" splits into "म" + "ैं", exactly like
      GPT-2's ``\\p{L}``-based regex;
    * whitespace not attached to a following letter/number run is one chunk per
      character, as the regex's single-character ``\\s`` alternative matches.
    """
    n = len(text)
    i = 0
    chunks: list[str] = []
    while i < n:
        kind = _char_class_gpt2(text[i])
        if kind == "S":
            j = i
            while j < n and text[j].isspace():
                j += 1
            attaches = j < n and _char_class_gpt2(text[j]) in ("L", "N")
            # all spaces except (optionally) the last, which joins the next run
            end = j - 1 if attaches else j
            chunks.extend(text[k] for k in range(i, end))
            i = j
            continue
        j = i
        while j < n and _char_class_gpt2(text[j]) == kind:
            j += 1
        start = i
        if kind in ("L", "N") and i > 0 and text[i - 1].isspace():
            start = i - 1  # the regex glues one leading space onto the run
        chunks.append(text[start:j])
        i = j
    return chunks


_PRETOKENIZERS = {
    "mark_aware": pretokenize,
    "gpt2_style": pretokenize_gpt2_style,
}


def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Merge every non-overlapping occurrence of ``pair`` in one word, left to right."""
    first, second = pair
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _pair_multiset(word: tuple[int, ...]) -> Counter:
    """Counts of adjacent pairs in one word (empty for single-symbol words)."""
    counts: Counter = Counter()
    for i in range(len(word) - 1):
        counts[(word[i], word[i + 1])] += 1
    return counts


def train_merges(word_counts: Counter, n_merges: int) -> list[tuple[int, int]]:
    """The BPE merge loop with incremental pair counts.

    Maintains exactly the pair counts the classic full re-scan would compute at
    every step (only the words containing the merged pair are touched), and
    applies the same tie-break — (frequency desc, pair asc) — so the merge
    sequence is identical to the classic algorithm. See the regression test in
    ``tests/test_tokenization.py`` that asserts this against a full re-scan.
    """
    pair_counts: Counter = Counter()
    pair_words: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] += count
            pair_words.setdefault(p, set()).add(word)

    merges: list[tuple[int, int]] = []
    for _ in range(n_merges):
        if not pair_counts:
            break  # no pairs left to merge
        pair, _total = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        new_id = 256 + len(merges)
        merges.append(pair)
        affected = pair_words.pop(pair)
        del pair_counts[pair]
        for word in affected:
            count = word_counts[word]
            merged = _merge_word(word, pair, new_id)
            if merged == word:
                continue
            old_pairs = _pair_multiset(word)
            new_pairs = _pair_multiset(merged)
            for p in set(old_pairs) | set(new_pairs):
                if p == pair:
                    continue  # its words are exactly ``affected``; count is now 0
                pair_counts[p] += (new_pairs.get(p, 0) - old_pairs.get(p, 0)) * count
                if old_pairs.get(p, 0) > 0 and pair_words.get(p) is not None:
                    ws = pair_words[p]
                    ws.discard(word)
                    if not ws:
                        pair_words.pop(p, None)
                if new_pairs.get(p, 0) > 0:
                    pair_words.setdefault(p, set()).add(merged)
                if not pair_counts.get(p, 0):
                    pair_counts.pop(p, None)
            word_counts[merged] += count
            del word_counts[word]
    return merges


class PythonBPE(SubwordTokenizer):
    """Byte-level BPE trained with the count-and-merge loop (incremental pair counts)."""

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
        self._pretoken: str = "mark_aware"

    # ---------------------------------------------------------------- train --
    def train(
        self,
        corpus_path: str | Path,
        vocab_size: int,
        special_tokens: Sequence[str] = (),
        max_train_chars: int | None = None,
        pretoken: str = "mark_aware",
        **kwargs: object,
    ) -> None:
        """Fit merges on a local text file.

        Deterministic: merges are chosen by (frequency desc, pair asc), so the same
        corpus, vocab size and pre-tokenization always produce the same tokenizer.

        ``pretoken`` selects the pre-tokenization boundaries: ``"mark_aware"``
        (default; combining marks stay attached to words — the script-aware choice)
        or ``"gpt2_style"`` (GPT-2's ``\\p{L}``/``\\p{N}``-based classes, where marks
        shatter Indic syllables — the comparison variant of the EXP-A sweep).
        """
        if pretoken not in _PRETOKENIZERS:
            raise TokenizerError(
                f"unknown pre-tokenization {pretoken!r}; expected one of {sorted(_PRETOKENIZERS)}"
            )
        self._pretoken = pretoken
        pre_fn = _PRETOKENIZERS[pretoken]

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
        for chunk in pre_fn(text):
            word_counts[tuple(chunk.encode("utf-8"))] += 1

        self._merges = train_merges(word_counts, n_merges)
        self._ranks = {pair: i for i, pair in enumerate(self._merges)}
        self._special_tokens = specials
        self._special_ids = {t: 256 + len(self._merges) + i for i, t in enumerate(specials)}
        self._id_to_bytes = [bytes([i]) for i in range(256)]
        for a, b in self._merges:
            self._id_to_bytes.append(self._id_to_bytes[a] + self._id_to_bytes[b])
        for token in specials:
            self._id_to_bytes.append(token.encode("utf-8"))
        self._target_vocab_size = vocab_size
        self._unk_token = None  # byte-level: nothing is out of vocabulary
        self._trained_chars = len(text)

    # ------------------------------------------------------------- encoding --
    def encode(self, text: str) -> list[int]:
        pre_fn = _PRETOKENIZERS[self._pretoken]
        ids: list[int] = []
        for segment, is_special in iter_segments(text, self._special_tokens):
            if is_special:
                ids.append(self._special_ids[segment])
            else:
                for chunk in pre_fn(segment):
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
                "pretoken": self._pretoken,
                "pretoken_pattern": PRETOKEN_PATTERN if self._pretoken == "mark_aware"
                else "gpt2-style (letter | number | whitespace | other; one leading space glues to a run)",
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
        # older artifacts predate the pretoken field and are always mark-aware
        tok._pretoken = data.get("pretoken", "mark_aware")
        if tok._pretoken not in _PRETOKENIZERS:
            raise TokenizerError(f"{path}: unknown pretoken {tok._pretoken!r}")
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


register(PythonBPE.name, PythonBPE)
