"""frontier-ai: a small, readable, GPU-ready PyTorch language-model training stack.

The package is intentionally dependency-light (torch + numpy) and device-agnostic:
the same code trains on a laptop CPU (tiny defaults) and on a CUDA box (bf16,
gradient accumulation, DDP-ready) without edits.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - depends on install state
    __version__ = version("frontier-ai")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
