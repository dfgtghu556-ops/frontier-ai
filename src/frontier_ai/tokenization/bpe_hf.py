"""Baseline subword tokenizer: byte-level BPE via the HuggingFace ``tokenizers`` library.

Why this implementation is our baseline
---------------------------------------
* **Established and permissive.** Rust implementation, Apache-2.0 licensed, used
  widely in production and research.
* **Offline.** We only ever call ``train`` on a local file and ``save``/``from_file``
  locally. No hub access, no network.
* **Byte level.** Base vocabulary is the 256 byte values, so every UTF-8 string
  (Indic scripts, emoji, ZWJ sequences) is encodable and decoding is lossless.
* **Complete.** Special tokens, save/load, vocab introspection and fast encoding.

Two pre-tokenization strategies (the ``pretoken`` argument of :meth:`train`):
``"byte_level"`` (GPT-2 boundaries, the EXP-A baseline) and ``"mark_aware"``
(same boundaries as ``bpe_python``'s mark-aware pre-tokenizer — a combining
mark never crosses a word boundary — composed with ByteLevel's byte remapping;
see :class:`MarkAwarePreTokenizer`).

Alternatives considered are documented in ``docs/tokenization.md`` and DECISIONS.md
(D-018). This module is optional: if the package is not installed, the implementation
simply is not registered, and the rest of the subsystem keeps working.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .base import SubwordTokenizer, TokenizerError
from .registry import register

try:  # optional dependency
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
    from tokenizers import __version__ as _TOKENIZERS_VERSION

    HF_AVAILABLE = True
    HF_IMPORT_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - depends on environment
    HF_AVAILABLE = False
    HF_IMPORT_ERROR = str(_exc)
    _TOKENIZERS_VERSION = "unavailable"

# Pre-tokenization strategies (mirror bpe_python's ``pretoken`` values).
PRETOKEN_BYTE_LEVEL = "byte_level"
PRETOKEN_MARK_AWARE = "mark_aware"


class MarkAwarePreTokenizer:
    """Mark-aware word splitting for the HF pipeline (a *custom* pre-tokenizer).

    Splits text into the same chunks as ``bpe_python.pretokenize`` — runs of
    whitespace | alnum+combining-marks | other characters — so that BPE merges
    never cross a script boundary (a matra can never be glued to the next word).
    The classification is imported from ``bpe_python._char_class``: one single
    source of truth for the mark-aware boundaries across both implementations.

    In :meth:`HuggingFaceBPE.train` it is composed with the built-in
    ``ByteLevel(use_regex=False)`` pre-tokenizer, which byte-escapes each chunk
    so the model stays in the lossless 256-byte space (exactly the ``hf-
    byte_level`` baseline's encoding, with mark-aware boundaries instead of
    GPT-2's).

    Custom pre-tokenizers **cannot be serialized** (``tokenizers`` raises
    "Custom PreTokenizer cannot be serialized"), so :meth:`HuggingFaceBPE.save`
    writes a ``ByteLevel`` placeholder in their place in ``tokenizer.json`` and
    records the true pre-tokenizer in ``hf_config.json``;
    :meth:`HuggingFaceBPE.load` re-attaches this component afterwards.
    """

    def pre_tokenize(self, pretokenized) -> None:  # noqa: ANN001 - tokenizers protocol
        pretokenized.split(self._split_chunks)

    def _split_chunks(self, index: int, normalized):  # noqa: ANN001 - tokenizers protocol
        # imported lazily: bpe_python has no hard dependency on 'tokenizers'
        from .bpe_python import _char_class

        text = normalized.normalized
        pieces: list = []
        start = 0
        current: str | None = None
        for pos, ch in enumerate(text):
            kind = _char_class(ch)
            if kind != current and pos > start:
                pieces.append(normalized.slice((start, pos)))
                start = pos
            current = kind
        if len(text) > start:
            pieces.append(normalized.slice((start, len(text))))
        return pieces


class HuggingFaceBPE(SubwordTokenizer):
    """Byte-level BPE trained with HuggingFace ``tokenizers``."""

    name = "bpe_hf"

    def __init__(self, unk_token: str | None = None) -> None:
        if not HF_AVAILABLE:
            raise TokenizerError(
                "the 'tokenizers' package is not installed. Install it with: "
                "pip install '.[tokenizer]'  (or use the dependency-free 'bpe_python' implementation)"
            )
        self._tokenizer: Tokenizer | None = None
        self._special_tokens: list[str] = []
        self._unk_token = unk_token
        self._pretoken: str = PRETOKEN_BYTE_LEVEL
        self._train_params: dict[str, object] = {}

    # ---------------------------------------------------------------- train --
    def train(
        self,
        corpus_path: str | Path,
        vocab_size: int,
        special_tokens: Sequence[str] = (),
        min_frequency: int = 2,
        pretoken: str = PRETOKEN_BYTE_LEVEL,
        **kwargs: object,
    ) -> None:
        """Train from a **local** text file. No network access is performed.

        ``pretoken`` selects the pre-tokenization boundaries:

        * ``"byte_level"`` (default) — the built-in ``ByteLevel`` pre-tokenizer,
          i.e. GPT-2-style word boundaries (the EXP-A baseline cell);
        * ``"mark_aware"`` — the custom :class:`MarkAwarePreTokenizer` composed
          with ``ByteLevel(use_regex=False)``: mark-aware boundaries (a matra
          stays with its base character; BPE never joins across the boundary)
          while keeping ByteLevel's lossless byte remapping. Requires
          ``tokenizers`` with the custom pre-tokenizer API (``>= 0.22``).
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"training corpus not found: {path}")
        if vocab_size < 256 + len(special_tokens):
            raise TokenizerError(
                f"vocab_size={vocab_size} is too small for 256 byte tokens plus "
                f"{len(special_tokens)} special tokens"
            )
        if pretoken not in (PRETOKEN_BYTE_LEVEL, PRETOKEN_MARK_AWARE):
            raise TokenizerError(
                f"unknown pre-tokenization {pretoken!r} "
                f"(expected one of {PRETOKEN_BYTE_LEVEL!r}, {PRETOKEN_MARK_AWARE!r})"
            )

        specials = list(dict.fromkeys(special_tokens))
        added = [AddedToken(t, special=True) for t in specials]
        if self._unk_token:
            added.append(AddedToken(self._unk_token, special=True))

        tokenizer = Tokenizer(models.BPE())
        # add_prefix_space=False keeps leading whitespace intact so decode(encode(x)) == x
        if pretoken == PRETOKEN_MARK_AWARE:
            if not hasattr(pre_tokenizers.PreTokenizer, "custom"):
                raise TokenizerError(
                    "the 'tokenizers' version does not provide the custom pre-tokenizer "
                    "API (PreTokenizer.custom); the mark_aware pre-tokenization needs "
                    "tokenizers >= 0.22 (pip install -U tokenizers)"
                )
            # custom mark-aware splitting + built-in byte remapping (no GPT-2 re-split)
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.PreTokenizer.custom(MarkAwarePreTokenizer()),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ])
            pre_tokenizer_desc = (
                "Sequence([MarkAwarePreTokenizer(custom), "
                "ByteLevel(add_prefix_space=False, use_regex=False)])"
            )
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            pre_tokenizer_desc = "ByteLevel(add_prefix_space=False)"
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=added,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train([str(path)], trainer)

        self._tokenizer = tokenizer
        self._special_tokens = specials
        self._pretoken = pretoken
        self._train_params = {
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "pre_tokenizer": pre_tokenizer_desc,
            "model": "BPE",
        }

    # ------------------------------------------------------------- encoding --
    def _require(self) -> Tokenizer:
        if self._tokenizer is None:
            raise TokenizerError("tokenizer is not trained or loaded; call train() or load() first")
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._require().encode(text).ids)

    def decode(self, ids: Sequence[int]) -> str:
        # skip_special_tokens must be False: decode() has to be the exact inverse of
        # encode(), otherwise any text containing a special token fails a round trip.
        return self._require().decode(list(ids), skip_special_tokens=False)

    # ----------------------------------------------------------------- meta --
    @property
    def vocab_size(self) -> int:
        return self._require().get_vocab_size()

    @property
    def special_tokens(self) -> list[str]:
        return list(self._special_tokens)

    @property
    def special_token_ids(self) -> dict[str, int]:
        vocab = self._require().get_vocab()
        return {t: vocab[t] for t in self._special_tokens if t in vocab}

    @property
    def unk_token(self) -> str | None:
        return self._unk_token

    @property
    def unk_token_id(self) -> int | None:
        if not self._unk_token:
            return None
        return self._require().token_to_id(self._unk_token)

    def id_to_token(self, index: int) -> str | None:
        return self._require().id_to_token(index)

    def impl_version(self) -> str:
        return f"tokenizers-{_TOKENIZERS_VERSION}"

    # ------------------------------------------------------------------ io --
    def _mark_aware_pre_tokenizer(self):
        return pre_tokenizers.Sequence([
            pre_tokenizers.PreTokenizer.custom(MarkAwarePreTokenizer()),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])

    def save(self, out_dir: str | Path) -> Path:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer = self._require()
        if self._pretoken == PRETOKEN_MARK_AWARE:
            # custom pre-tokenizers cannot be serialized: save a ByteLevel
            # placeholder in their place (a readable, still-usable artifact);
            # load() re-attaches the mark-aware component from hf_config.json.
            real = tokenizer.pre_tokenizer
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False
            )
            try:
                tokenizer.save(str(directory / "tokenizer.json"))
            finally:
                tokenizer.pre_tokenizer = real
        else:
            tokenizer.save(str(directory / "tokenizer.json"))
        self.save_json(
            directory / "hf_config.json",
            {
                "format": "hf-tokenizers-bpe",
                "pre_tokenizer": self._pretoken,
                "special_tokens": self._special_tokens,
                "unk_token": self._unk_token,
                "train_params": self._train_params,
                "library_version": _TOKENIZERS_VERSION,
            },
        )
        return directory

    @classmethod
    def load(cls, artifact_dir: str | Path) -> HuggingFaceBPE:
        path = Path(artifact_dir) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"no HuggingFace tokenizer artifact at {path}")
        if not HF_AVAILABLE:
            raise TokenizerError(f"cannot load {path}: the 'tokenizers' package is not installed")

        obj = cls()
        obj._tokenizer = Tokenizer.from_file(str(path))
        config_path = Path(artifact_dir) / "hf_config.json"
        if config_path.exists():
            config = cls.read_json(config_path)
            obj._special_tokens = list(config.get("special_tokens", []))
            obj._unk_token = config.get("unk_token")
            obj._train_params = dict(config.get("train_params", {}))
            # artifacts saved before the pretoken field existed are byte_level
            obj._pretoken = config.get("pre_tokenizer", PRETOKEN_BYTE_LEVEL)
            if obj._pretoken == PRETOKEN_MARK_AWARE:
                # re-attach the custom component the placeholder in tokenizer.json
                # cannot represent (see save())
                obj._tokenizer.pre_tokenizer = obj._mark_aware_pre_tokenizer()
        else:
            added = obj._tokenizer.get_added_tokens_decoder()
            obj._special_tokens = [t.content for t in added.values()]
        return obj


if HF_AVAILABLE:
    register(HuggingFaceBPE.name, HuggingFaceBPE)
