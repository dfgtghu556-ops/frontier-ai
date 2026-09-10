"""Deterministic randomness for experiments.

Design
------
One **master seed** per experiment. Global RNGs (Python ``random``, NumPy, Torch, and CUDA
when present) are seeded with that master seed through the existing helper
:func:`frontier_ai.utils.seed.set_seed`, so Project 001 and Project 002 keep their current
behaviour and stay consistent with any experiment wrapped by this subsystem.

Components that need their own stream (data sampling, dropout, generation, ...) should call
:func:`derive_seed` instead of inventing a new literal seed. Derived seeds are a documented
function of the master seed and a component name, so:

* every stream in the experiment is traceable to one recorded number, and
* two components never accidentally share the same stream.

Deliberately **not** used for derivation: Python's built-in ``hash()`` (randomized per
process for strings) and unseeded generators.

Honest limitations
------------------
Seeding every RNG is necessary but **not sufficient** for bitwise reproducibility:

1. **CUDA / cuDNN.** Some kernels (attention, convolutions, scatter/reduction ops) are
   nondeterministic by design. ``deterministic=True`` requests deterministic algorithms,
   but a number of ops have no deterministic implementation; PyTorch then raises or warns
   (we use ``warn_only=True``), so a "deterministic" run can still differ on GPU.
2. **Thread count.** Multi-threaded CPU reductions can change floating-point summation
   order. Reproducing a CPU result requires the same thread configuration.
3. **Library versions.** Results can change across torch/NumPy versions even with an
   identical seed; that is why the record stores the environment.
4. **Data ordering and the filesystem.** Data identity is captured separately by hashing
   the inputs; this module only covers RNG state.
5. **Anything outside these libraries** (subprocesses, third-party samplers, OS-level
   nondeterminism) is not covered.

We therefore record determinism as a *request* plus these caveats, never as a guarantee.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from ..utils.seed import set_seed

SEED_LIMITATIONS: tuple[str, ...] = (
    "CUDA/cuDNN kernels may be nondeterministic even when deterministic mode is requested; "
    "torch.use_deterministic_algorithms(True, warn_only=True) can still warn or fall back.",
    "CPU results depend on the thread configuration: multi-threaded reductions can change "
    "floating-point summation order.",
    "Library versions (torch, numpy, tokenizers) can change results for the same seed; the "
    "record stores the environment for this reason.",
    "Only Python random, NumPy and Torch RNGs are seeded. Subprocesses, third-party samplers "
    "and OS-level nondeterminism are not covered.",
)

SEEDED_RNGS: tuple[str, ...] = ("python.random", "numpy.random", "torch", "torch.cuda")

SEED_SPACE = 2**32

# Components that get their own derived stream by default. Keeping this here (rather than
# in runner.py) means the runner and any direct caller share the same convention.
DEFAULT_COMPONENTS: tuple[str, ...] = ("data", "model", "sampling")


def derive_seed(master: int, *components: str) -> int:
    """Derive a component seed from the master seed.

    Stable across processes and Python versions: SHA-256 over
    ``"<master>|<component names joined by '/'>"``. No reliance on ``hash()``.
    """
    if not isinstance(master, int) or isinstance(master, bool):
        raise TypeError(f"master seed must be an int, got {type(master).__name__}")
    payload = "|".join([str(master), *[str(c) for c in components]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_SPACE


def derive_seeds(master: int, components: Sequence[str] | None = None) -> dict[str, int]:
    """Derive a named seed per component (e.g. ``{"data": ..., "model": ...}``)."""
    names: Iterable[str] = components or ()
    return {name: derive_seed(master, name) for name in names}


def seed_everything(
    seed: int,
    deterministic: bool = False,
    components: Sequence[str] | None = DEFAULT_COMPONENTS,
) -> dict:
    """Seed every RNG this repository uses and return a record of what was done.

    The global RNGs receive the **master** seed; component-specific streams receive
    :func:`derive_seed` values that are recorded alongside it.
    """
    set_seed(seed, deterministic=deterministic)
    derived = derive_seeds(seed, components)
    return {
        "master_seed": int(seed),
        "deterministic_mode_requested": bool(deterministic),
        "seeded_rngs": list(SEEDED_RNGS),
        "derived_seeds": derived,
        "derivation": "sha256('<master>|<component>') mod 2**32",
        "limitations": list(SEED_LIMITATIONS),
    }
