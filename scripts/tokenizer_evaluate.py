#!/usr/bin/env python3
"""Evaluate one or more tokenizer artifacts against the research corpus.

Examples
--------
    python scripts/tokenizer_evaluate.py --corpus data/tokenizer/indic-v1 \
        --tokenizer artifacts/tokenizers/bpe_hf_1k --out out/tokenizer/bpe_hf_1k.json

    python scripts/tokenizer_evaluate.py --corpus data/tokenizer/indic-v1 \
        --tokenizer artifacts/tokenizers/char artifacts/tokenizers/bpe_hf_1k \
        --out out/tokenizer/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.tokenization.cli import (  # noqa: E402
    corpus_meta,
    load_tokenizer_artifacts,
    resolve_corpus_dir,
)
from frontier_ai.tokenization.compare import render_statblock  # noqa: E402
from frontier_ai.tokenization.evaluate import evaluate_tokenizer, report_to_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="corpus directory written by the prepare CLI")
    p.add_argument("--tokenizer", nargs="+", required=True, help="one or more artifact directories")
    p.add_argument("--label", nargs="*", default=None, help="optional display names per tokenizer")
    p.add_argument("--out", default="out/tokenizer/eval.json", help="output file (1 tokenizer) or directory")
    p.add_argument("--exp-id", default="EXP-000", help="experiment id recorded in the report")
    p.add_argument(
        "--jsonl",
        default=None,
        help="also write per-example rows (id, lang, category, tokens, chars, round_trip) here",
    )
    p.add_argument("--quiet", action="store_true", help="only print the output paths")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    corpus_dir = resolve_corpus_dir(args.corpus)

    from frontier_ai.tokenization.corpus import load_corpus

    examples, manifest = load_corpus(corpus_dir)
    loaded = load_tokenizer_artifacts(args.tokenizer, args.label)

    multiple = len(loaded) > 1
    out_path = Path(args.out)
    if multiple:
        out_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for label, tokenizer, artifact_manifest in loaded:
        report = evaluate_tokenizer(
            tokenizer,
            examples,
            manifest,
            experiment_id=args.exp_id,
            tokenizer_meta={
                "artifact_dir": str(artifact_manifest.artifact_dir or ""),
                "train_params": artifact_manifest.train_params,
                "impl_version_recorded": artifact_manifest.impl_version,
                "corpus_used": corpus_meta(corpus_dir),
            },
        )
        target = out_path / f"{label}.json" if multiple else out_path
        if not multiple:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report_to_json(report) + "\n", encoding="utf-8")
        written.append(str(target))

        if args.jsonl:
            rows = []
            for ex in examples:
                ids = tokenizer.encode(ex.text)
                rows.append(
                    {
                        "tokenizer": label,
                        "id": ex.id,
                        "lang": ex.lang,
                        "category": ex.category,
                        "chars": len(ex.text),
                        "tokens": len(ids),
                        "round_trip_ok": tokenizer.decode(ids) == ex.text,
                    }
                )
            jsonl_path = Path(args.jsonl)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
            )

        if not args.quiet:
            print()
            print(f"=== {label} ({report.tokenizer.get('impl')} {report.tokenizer.get('impl_version')}) ===")
            print(render_statblock(report.overall, "OVERALL"))
            specials = report.checks.get("special_tokens", {})
            print(f"\n  utf-8 round trip ok     : {report.checks.get('utf8_round_trip_ok')}")
            print(f"  special tokens all ok   : {specials.get('all_ok')} ({specials.get('declared')})")
            print("  chars/token by language :")
            for lang, stats in sorted(report.per_language.items()):
                print(f"      {lang:<8} {stats.chars_per_token:>7.3f}  (tokens={stats.tokens:>5})")
            if report.failing_examples:
                print(f"  FAILING EXAMPLES ({report.overall.round_trip_failures}):")
                for item in report.failing_examples[:5]:
                    print(f"      {item['id']}: {item['text'][:60]!r}")

    print()
    print(f"[eval] wrote: {', '.join(written)}")
    print(
        "[eval] NOTE: metrics come from a tiny hand-written probe fixture; they are comparable "
        "between tokenizers on this same fixture only, and are not real-world benchmarks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
