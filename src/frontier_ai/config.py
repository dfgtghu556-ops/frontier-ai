"""Configuration objects.

Configs are plain dataclasses so they are easy to construct in Python, serialize to
JSON, and override from the CLI (`--set optim.lr=1e-3`). No YAML dependency needed.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the decoder-only transformer."""

    vocab_size: int = 256  # set from the tokenizer at train time
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128  # context length
    dropout: float = 0.0
    bias: bool = False  # bias in Linear layers (False is the modern default)
    norm: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    ffn: str = "swiglu"  # "swiglu" | "gelu"
    ffn_mult: float = 4.0  # hidden = int(ffn_mult * n_embd) (SwiGLU rounds up to 64)
    pos: str = "learned"  # "learned" | "rope"
    tie_embeddings: bool = True
    n_kv_head: int = 0  # 0 -> multi-head attention (n_kv_head == n_head)

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.norm not in ("rmsnorm", "layernorm"):
            raise ValueError(f"unknown norm '{self.norm}'")
        if self.ffn not in ("swiglu", "gelu"):
            raise ValueError(f"unknown ffn '{self.ffn}'")
        if self.pos not in ("learned", "rope"):
            raise ValueError(f"unknown pos '{self.pos}'")
        if self.n_kv_head <= 0:
            self.n_kv_head = self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def ffn_hidden(self) -> int:
        hidden = int(self.ffn_mult * self.n_embd)
        if self.ffn == "swiglu":
            # keep SwiGLU hidden size a multiple of 64 for tensor-core friendliness
            hidden = ((hidden + 63) // 64) * 64
        return max(hidden, self.n_embd)


@dataclass
class DataConfig:
    """Everything about turning raw text into training batches."""

    path: str = "data/tokens.bin"  # prepared token file
    tokenizer: str = "data/tokenizer.json"  # tokenizer vocab file
    level: str = "char"  # "char" | "word" (used by prepare_data.py)
    source: str = "synthetic"  # "synthetic" | path to a raw .txt file
    val_frac: float = 0.1
    batch_size: int = 16
    num_workers: int = 0  # 0 = simplest + fastest for tiny in-memory data
    seed: int = 1337
    shuffle_buffer: int = 0  # 0 = uniform sampling of random offsets (default)

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_frac < 1.0:
            raise ValueError(f"val_frac must be in [0, 1), got {self.val_frac}")


@dataclass
class OptimConfig:
    """Optimizer + LR schedule."""

    lr: float = 3e-3
    min_lr_ratio: float = 0.1  # cosine floor as a fraction of lr
    weight_decay: float = 0.1
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 50
    schedule: str = "cosine"  # "cosine" | "constant"
    decay_params: bool = True  # separate decay groups for 2-D (matrix) params


@dataclass
class TrainConfig:
    """Training-loop knobs. Defaults are sized for a 2-CPU / 3 GB laptop."""

    out_dir: str = "out/cpu-smoke"
    max_steps: int = 300
    accum_steps: int = 4  # gradient accumulation micro-steps per optimizer step
    eval_interval: int = 50
    eval_iters: int = 20
    log_interval: int = 10
    save_interval: int = 0  # 0 = only save best + last
    save_best: bool = True
    resume: str = ""  # path to a checkpoint directory
    init_from: str = "scratch"  # "scratch" | "resume"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:N" | "mps"
    precision: str = "auto"  # "auto" | "fp32" | "bf16" | "fp16"
    seed: int = 1337
    deterministic: bool = False  # slower, but bit-reproducible on the same device
    compile: bool = False  # torch.compile (CUDA-only benefit, off by default)
    num_threads: int = 0  # 0 = auto
    grad_checkpointing: bool = False  # trade compute for memory on big models
    early_stop_on_nan: bool = True


@dataclass
class GenConfig:
    """Sampling defaults for scripts/generate.py."""

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 1.0
    prompt: str = ""
    seed: int = 1234


@dataclass
class ExperimentConfig:
    """Top-level config bundling the sections above."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gen: GenConfig = field(default_factory=GenConfig)

    # ------------------------------------------------------------------ io --
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        return cls(
            model=_build(ModelConfig, data.get("model")),
            data=_build(DataConfig, data.get("data")),
            optim=_build(OptimConfig, data.get("optim")),
            train=_build(TrainConfig, data.get("train")),
            gen=_build(GenConfig, data.get("gen")),
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: expected a JSON object at the top level")
        return cls.from_dict(data)

    def with_overrides(self, overrides: list[str] | None) -> ExperimentConfig:
        """Apply `--set section.key=value` style overrides and return a new config."""
        cfg = copy.deepcopy(self)
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override must look like section.key=value, got '{item}'")
            key, raw = item.split("=", 1)
            if key.count(".") != 1:
                raise ValueError(f"override key must be 'section.key', got '{key}'")
            section, name = key.split(".")
            if not hasattr(cfg, section):
                raise ValueError(f"unknown config section '{section}'")
            sub = getattr(cfg, section)
            if name not in {f.name for f in fields(sub)}:
                raise ValueError(f"unknown key '{name}' in section '{section}'")
            sub.__dict__[name] = _coerce(type(getattr(sub, name)), raw)
            sub.__post_init__() if hasattr(sub, "__post_init__") else None
        return cfg


def _build(cls: type[T], data: Mapping[str, Any] | None) -> T:
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**data)  # type: ignore[call-arg]


def _coerce(target: Any, raw: str) -> Any:
    """Parse a CLI string into the type of the field it is overriding."""
    raw = raw.strip()
    if target is bool:
        return raw.lower() in ("1", "true", "yes", "y", "on")
    if target is int:
        return int(float(raw)) if ("." in raw or "e" in raw.lower()) else int(raw)
    if target is float:
        return float(raw)
    if target is list:
        return json.loads(raw)
    return raw
