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
        self._train_params: dict[str, object] = {}

    # ---------------------------------------------------------------- train --
    def train(
        self,
        corpus_path: str | Path,
        vocab_size: int,
        special_tokens: Sequence[str] = (),
        min_frequency: int = 2,
        **kwargs: object,
    ) -> None:
        """Train from a **local** text file. No network access is performed."""
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"training corpus not found: {path}")
        if vocab_size < 256 + len(special_tokens):
            raise TokenizerError(
                f"vocab_size={vocab_size} is too small for 256 byte tokens plus "
                f"{len(special_tokens)} special tokens"
            )

        specials = list(dict.fromkeys(special_tokens))
        added = [AddedToken(t, special=True) for t in specials]
        if self._unk_token:
            added.append(AddedToken(self._unk_token, special=True))

        tokenizer = Tokenizer(models.BPE())
        # add_prefix_space=False keeps leading whitespace intact so decode(encode(x)) == x
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
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
        self._train_params = {
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
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
    def save(self, out_dir: str | Path) -> Path:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._require().save(str(directory / "tokenizer.json"))
        self.save_json(
            directory / "hf_config.json",
            {
                "format": "hf-tokenizers-bpe",
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
        else:
            added = obj._tokenizer.get_added_tokens_decoder()
            obj._special_tokens = [t.content for t in added.values()]
        return obj


if HF_AVAILABLE:
    register(HuggingFaceBPE.name, HuggingFaceBPE)
