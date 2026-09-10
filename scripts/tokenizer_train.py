#!/usr/bin/env python3
"""Train a tokenizer from a **local** corpus and save it as a versioned artifact.

Examples
--------
    python scripts/tokenizer_prepare_corpus.py --out data/tokenizer/indic-v1

    python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 \
        --impl bpe_hf --vocab-size 1024 --out artifacts/tokenizers/bpe_hf_1k --exp-id EXP-003

    python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 \
        --impl bpe_python --vocab-size 1024 --special-tokens "<pad>" "<eos>" \
        --out artifacts/tokenizers/bpe_py_1k --exp-id EXP-004

    python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 --impl char \
        --out artifacts/tokenizers/char --exp-id EXP-005
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.tokenization import available, create  # noqa: E402
from frontier_ai.tokenization.artifact import save_artifact  # noqa: E402
from frontier_ai.tokenization.cli import (  # noqa: E402
    corpus_meta,
    resolve_corpus_dir,
    train_path_for,
)

DEFAULT_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="corpus directory (or its train.txt)")
    p.add_argument(
        "--impl",
        required=True,
        help="tokenizer implementation; one of: " + ", ".join(sorted(available())),
    )
    p.add_argument("--vocab-size", type=int, default=1024, help="target vocabulary size")
    p.add_argument("--out", required=True, help="artifact output directory")
    p.add_argument("--exp-id", default="EXP-000", help="experiment id recorded in the manifest")
    p.add_argument(
        "--special-tokens",
        nargs="*",
        default=list(DEFAULT_SPECIALS),
        help="special tokens added to the vocabulary (pass an empty list to disable)",
    )
    p.add_argument("--min-frequency", type=int, default=2, help="[bpe_hf] minimum pair frequency")
    p.add_argument("--min-count", type=int, default=1, help="[word] minimum token count")
    p.add_argument(
        "--max-train-chars",
        type=int,
        default=None,
        help="[bpe_python] truncate the training text to N characters",
    )
    p.add_argument("--seed", type=int, default=None, help="recorded in the manifest (informational)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.impl not in available():
        raise SystemExit(
            f"[train] unknown implementation '{args.impl}'. Available: {sorted(available())}"
        )

    corpus_dir = resolve_corpus_dir(args.corpus)
    train_path = train_path_for(corpus_dir)

    tokenizer = create(args.impl)
    train_kwargs: dict = {}
    if args.impl == "bpe_hf":
        train_kwargs["min_frequency"] = args.min_frequency
    elif args.impl == "word":
        train_kwargs["min_count"] = args.min_count
    elif args.impl == "bpe_python":
        train_kwargs["max_train_chars"] = args.max_train_chars

    print(f"[train] impl={args.impl} corpus={train_path} vocab_size={args.vocab_size}")
    tokenizer.train(
        train_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        **train_kwargs,
    )

    manifest = save_artifact(
        tokenizer,
        args.out,
        experiment_id=args.exp_id,
        corpus=corpus_meta(corpus_dir),
        train_params={
            "vocab_size": args.vocab_size,
            "special_tokens": list(args.special_tokens),
            **train_kwargs,
        },
        seed=args.seed,
    )

    sample = "मैं स्कूल जा रहा हूँ।"
    ids = tokenizer.encode(sample)
    print(
        f"[train] saved {manifest.artifact_dir} | vocab={manifest.vocab_size} "
        f"| impl_version={manifest.impl_version}"
    )
    print(f"[train] sample: {sample!r} -> {len(ids)} ids -> round_trip_ok={tokenizer.round_trip(sample)}")
    print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
