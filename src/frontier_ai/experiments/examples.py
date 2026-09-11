"""Small reference experiments used by the documentation and the determinism tests.

They exist so the reproducibility guarantee is demonstrated on real code paths from
Project 001 and Project 002, not on a toy that only exercises the runner itself.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..data.synthetic import generate_corpus
from ..data.tokenizer import CharTokenizer
from .hashing import sha256_text


def reference_experiment(ctx) -> Mapping[str, Any]:
    """Deterministic text/tokenizer probe.

    Uses a *derived* seed for the data stream, which is the pattern the runner wants
    experiments to follow: one master seed recorded in the experiment, per-component
    streams derived from it.
    """
    data_seed = ctx.derived("data")
    rng = random.Random(data_seed)
    text = generate_corpus(target_chars=4000, seed=ctx.seed)
    tokenizer = CharTokenizer.fit(text)
    ids = tokenizer.encode(text)

    # exercise both Python and NumPy RNGs so the record's seeded_rngs claim is real
    python_draws = [round(rng.random(), 12) for _ in range(4)]
    numpy_draws = [
        round(float(v), 12)
        for v in np.random.default_rng(ctx.derived("model")).random(4)
    ]

    return {
        "text_sha256": sha256_text(text),
        "characters": len(text),
        "vocab_size": tokenizer.vocab_size,
        "tokens": len(ids),
        "tokens_per_char": round(len(ids) / len(text), 6),
        "round_trip_ok": tokenizer.decode(ids) == text,
        "python_draws": python_draws,
        "numpy_draws": numpy_draws,
        "data_seed": data_seed,
    }


def seeded_metric_experiment(ctx, draws: int = 8) -> Mapping[str, Any]:
    """Cheap, seed-sensitive reference experiment used by the sweep tests and docs.

    Returns a synthetic ``score`` whose value is a deterministic function of the master
    seed (a mean of seeded NumPy draws plus a small seeded Python jitter), so a sweep over
    seeds has something to average — and something whose spread is meaningful. It measures
    nothing about language modelling; it exists to exercise the sweep machinery cheaply.
    """
    rng = random.Random(ctx.seed)
    values = np.random.default_rng(ctx.derived("model")).random(draws)
    jitter = rng.uniform(-0.05, 0.05)
    return {
        "score": round(float(values.mean()) + jitter, 6),
        "draws": int(draws),
        "seed": int(ctx.seed),
    }


def tiny_training_experiment(ctx, steps: int = 10) -> Mapping[str, Any]:
    """Train a tiny GPT for a few steps with the Project 001 trainer.

    This is the strongest end-to-end reproducibility check available on CPU: same code,
    data, config and seed must give the same loss trajectory. See docs/experiments.md for
    the measured result and its caveats (thread count, torch version).
    """
    import torch

    from ..config import DataConfig, ExperimentConfig, ModelConfig, OptimConfig, TrainConfig
    from ..data.dataset import TokenDataset, write_tokens
    from ..data.tokenizer import CharTokenizer
    from ..engine.trainer import Trainer

    # A configuration sweep (Stage 1B) sets these through spec.params; the defaults keep
    # the Stage 1A behaviour and its recorded fingerprints exactly as they were.
    steps = int(ctx.spec.params.get("steps", steps))
    lr = float(ctx.spec.params.get("lr", 0.01))

    text = generate_corpus(target_chars=6000, seed=ctx.derived("data"))
    tokenizer = CharTokenizer.fit(text)
    tmp_dir = ctx.output_dir / "data"
    tokens_path = tmp_dir / "tokens.bin"
    write_tokens(tokens_path, tokenizer.encode(text), tokenizer.vocab_size, "char", val_frac=0.1)
    dataset = TokenDataset(tokens_path)

    cfg = ExperimentConfig(
        model=ModelConfig(vocab_size=tokenizer.vocab_size, n_layer=2, n_head=2,
                          n_embd=32, block_size=32),
        data=DataConfig(path=str(tokens_path), batch_size=2, seed=ctx.derived("data")),
        optim=OptimConfig(lr=lr, warmup_steps=1),
        train=TrainConfig(
            out_dir=str(ctx.output_dir / "tiny-run"),
            max_steps=steps,
            num_threads=1,
            accum_steps=1,
            eval_interval=0,
            log_interval=max(1, steps),
            device="cpu",
            precision="fp32",
            seed=ctx.seed,
        ),
    )
    trainer = Trainer(cfg, dataset)
    trainer.fit()

    return {
        "steps": steps,
        "n_params": int(trainer.model.n_params()),
        "tokens_seen": int(trainer.state.tokens_seen),
        "final_val_loss": round(float(trainer.evaluate(max_iters=2)), 6),
        "dtype": str(torch.get_default_dtype()),
    }
