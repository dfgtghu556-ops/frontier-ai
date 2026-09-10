#!/usr/bin/env python3
"""Turn raw text (or the built-in synthetic corpus) into a token file.

Examples
--------
    # zero-download smoke data
    python scripts/prepare_data.py --source synthetic --level char --out data/synthetic

    # real text file
    python scripts/prepare_data.py --source data/raw/shakespeare.txt --level word \
        --out data/shakes --min-count 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.data.dataset import text_lengths, write_tokens  # noqa: E402
from frontier_ai.data.synthetic import generate_corpus  # noqa: E402
from frontier_ai.data.tokenizer import fit_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="synthetic", help="'synthetic' or a path to a .txt file")
    p.add_argument("--out", default="data/synthetic", help="output prefix (writes <out>.bin)")
    p.add_argument("--level", default="char", choices=["char", "word"], help="tokenizer granularity")
    p.add_argument("--min-count", type=int, default=1, help="word-level vocab cutoff")
    p.add_argument("--target-chars", type=int, default=200_000, help="synthetic corpus size")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction held out for validation")
    p.add_argument(
        "--provenance",
        default=None,
        help="provenance JSON for a real licensed corpus (written by fetch_smoke_corpus.py); "
             "it is copied into the metadata so the prepared corpus states its source and licence",
    )
    p.add_argument("--seed", type=int, default=1337, help="synthetic corpus seed")
    return p.parse_args()


def load_text(source: str, target_chars: int, seed: int) -> str:
    if source == "synthetic":
        print(f"[data] generating synthetic corpus (~{target_chars:,} chars, seed={seed})")
        return generate_corpus(target_chars=target_chars, seed=seed)
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"[data] source not found: {path}")
    text = path.read_text(encoding="utf-8")
    print(f"[data] read {len(text):,} chars from {path}")
    return text


def main() -> int:
    args = parse_args()
    text = load_text(args.source, args.target_chars, args.seed)

    tokenizer = fit_tokenizer(text, level=args.level, min_count=args.min_count)
    ids = tokenizer.encode(text)
    if len(ids) < 1000:
        raise SystemExit(f"[data] only {len(ids)} tokens - need a larger corpus")

    # Byte/character counts per token, measured on the tokenizer's own pieces so
    # they add up to the source text. That is what makes bits-per-byte (rather
    # than per-token loss) comparable between a char-level and a BPE corpus.
    token_bytes, token_chars = text_lengths(tokenizer.tokenize(text))
    assert len(token_bytes) == len(ids) and len(token_chars) == len(ids)

    out_prefix = Path(args.out)
    tok_path = out_prefix.with_suffix(".tokenizer.json")
    bin_path = out_prefix.with_suffix(".bin")
    tokenizer.save(tok_path)
    source_provenance = None
    if args.provenance:
        prov_path = Path(args.provenance)
        if not prov_path.exists():
            raise SystemExit(f"[data] provenance file not found: {prov_path}")
        source_provenance = json.loads(prov_path.read_text(encoding="utf-8"))
        for key in ("id", "license_id", "source_url", "sha256"):
            if key not in source_provenance:
                raise SystemExit(f"[data] {prov_path} is missing required key {key!r}")
        print(
            f"[data] provenance: {source_provenance['id']} "
            f"({source_provenance['license_id']}) sha256={str(source_provenance['sha256'])[:16]}…"
        )
        digest = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
        if source_provenance.get("sha256") and digest != source_provenance["sha256"]:
            print(
                f"[data] WARNING: {args.source} no longer matches the pinned sha256 "
                f"({digest[:16]}… vs {str(source_provenance['sha256'])[:16]}…); "
                f"the prepared corpus is NOT the corpus the provenance describes"
            )

    meta = write_tokens(
        bin_path,
        ids,
        tokenizer.vocab_size,
        args.level,
        val_frac=args.val_frac,
        token_bytes=token_bytes,
        token_chars=token_chars,
        source_provenance=source_provenance,
    )

    print(
        f"[data] vocab={tokenizer.vocab_size} ({args.level}) tokens={meta.n_tokens:,} "
        f"train={meta.n_train:,} val={meta.n_val:,}"
    )
    print(
        f"[data] bytes={meta.n_bytes:,} chars={meta.n_chars:,} "
        f"bytes/token={meta.n_bytes / meta.n_tokens:.4f} "
        f"chars/token={meta.n_chars / meta.n_tokens:.4f}"
    )
    print(f"[data] wrote {bin_path}  {tok_path}  {bin_path.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
