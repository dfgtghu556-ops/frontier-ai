"""A tiny synthetic corpus generator.

Why this exists: you want a smoke test whose loss *visibly* goes down in a few
hundred steps on a CPU, without downloading anything. The text is generated from a
small grammar (templates over a ~100-word vocabulary plus a couple of deterministic
patterns), so there is real structure for even a 0.5M-parameter model to absorb.

This is a test fixture, not training data for anything you would ship.
"""

from __future__ import annotations

import random

SUBJECTS = [
    "cat", "dog", "robot", "scholar", "engineer", "river", "city", "signal",
    "tensor", "kernel", "garden", "lantern", "comet", "archive", "foundry",
]
VERBS = [
    "watches", "builds", "counts", "maps", "trains", "reads", "shapes",
    "balances", "carries", "holds", "measures", "guides",
]
OBJECTS = [
    "numbers", "gradients", "tokens", "stars", "bridges", "memories", "fields",
    "weights", "songs", "gardens", "signals", "paths",
]
ADJECTIVES = [
    "quiet", "bright", "steady", "small", "vast", "gentle", "sharp", "patient",
    "hidden", "parallel", "local", "global",
]
ADVERBS = [
    "slowly", "carefully", "twice", "always", "rarely", "smoothly", "exactly",
]
PATTERNS = [
    "a {adj} {sub} {verb} the {obj}",
    "the {sub} {adv} {verb} {obj}",
    "{sub} and {sub2} {verb} the {adj} {obj}",
    "if the {sub} is {adj}, then the {obj} stay {adj2}",
]


def _sentence(rng: random.Random) -> str:
    template = rng.choice(PATTERNS)
    text = template.format(
        adj=rng.choice(ADJECTIVES),
        adj2=rng.choice(ADJECTIVES),
        adv=rng.choice(ADVERBS),
        sub=rng.choice(SUBJECTS),
        sub2=rng.choice(SUBJECTS),
        verb=rng.choice(VERBS),
        obj=rng.choice(OBJECTS),
    )
    return text[0].upper() + text[1:] + "."


def _counting_line(rng: random.Random) -> str:
    start = rng.randint(1, 20)
    step = rng.choice([1, 2, 3])
    n = rng.randint(4, 9)
    return " ".join(str(start + step * i) for i in range(n)) + "."


def _identity_line(rng: random.Random) -> str:
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    return f"{a} plus {b} is {a + b}, and {a} times {b} is {a * b}."


def generate_corpus(target_chars: int = 200_000, seed: int = 1337, newline_every: int = 3) -> str:
    """Generate ~`target_chars` of structured pseudo-English.

    Deterministic for a given seed, so smoke tests and CI are reproducible.
    """
    rng = random.Random(seed)
    lines: list[str] = []
    total = 0
    makers = (_sentence, _sentence, _counting_line, _identity_line)
    while total < target_chars:
        chunk: list[str] = []
        for _ in range(newline_every):
            chunk.append(rng.choice(makers)(rng))
        line = " ".join(chunk)
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) + "\n"
