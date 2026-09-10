"""A tiny registry so tokenizers are pluggable rather than hard-coded.

Adding a new tokenizer = write a class implementing
:class:`~frontier_ai.tokenization.base.SubwordTokenizer` and register it here (or import
it in ``tokenization/__init__.py``). Nothing else in the subsystem needs to change:
the training CLI, evaluator and comparison tool all work through this registry.
"""

from __future__ import annotations

from .base import SubwordTokenizer

_REGISTRY: dict[str, type[SubwordTokenizer]] = {}


def register(name: str, cls: type[SubwordTokenizer]) -> None:
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"tokenizer implementation '{name}' is already registered")
    _REGISTRY[name] = cls


def create(name: str, **kwargs: object) -> SubwordTokenizer:
    """Instantiate a registered tokenizer (untrained; call ``.train(...)`` next)."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown tokenizer implementation '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)  # type: ignore[call-arg]


def available() -> dict[str, str]:
    """``{implementation name: 'module.Class'}`` for everything currently importable."""
    return {name: f"{cls.__module__}.{cls.__name__}" for name, cls in sorted(_REGISTRY.items())}


def names() -> list[str]:
    return sorted(_REGISTRY)


def is_available(name: str) -> bool:
    return name in _REGISTRY
