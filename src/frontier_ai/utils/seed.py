"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch RNGs.

    deterministic=True additionally forces deterministic cuDNN kernels. It costs
    performance and is only worth it for debugging / ablations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def make_generator(seed: int | None) -> torch.Generator | None:
    """A CPU torch.Generator; None means 'let torch pick'."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
