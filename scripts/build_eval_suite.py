#!/usr/bin/env python3
"""Build the protected evaluation suite ``frontier-heldout-v1`` (fingerprints only).

Re-derives the FrontierCorpus v1 held-out side through the shared identity gates (the
same call as EXP-A / EXP-B, exit 2 on any mismatch) and writes
``evals/suites/frontier-heldout-v1/SUITE.json``: one record per document (doc_id,
language, source_id, text SHA-256, bytes, chars) in evaluation order, plus per-language
totals and an order-sensitive fingerprint. No corpus text is written.

Idempotent: if the suite file exists and is identical, exit 0 without writing; if it
exists and differs, exit 2 (a protected suite is never silently replaced).

    python scripts/build_eval_suite.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.corpus import verify_and_derive_frontier  # noqa: E402
from frontier_ai.evaluation import make_console_safe  # noqa: E402
from frontier_ai.evaluation.suite import (  # noqa: E402
    FRONTIER_HELDOUT_V1,
    SUITES_ROOT,
    build_suite,
    load_suite,
    write_suite,
)


def main() -> int:
    make_console_safe()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(SUITES_ROOT / FRONTIER_HELDOUT_V1 / "SUITE.json"))
    p.add_argument("--frontier-dir", default="corpora/frontier/v1")
    p.add_argument("--manifest", default="corpora/tokenizer/indic-tokenizer-v2/sources.json")
    p.add_argument("--freeze", default="corpora/tokenizer/indic-tokenizer-v2/FREEZE.json")
    p.add_argument("--corpus-dir", default="data/tokenizer/indic-tokenizer-v2")
    args = p.parse_args()

    _train, held, manifest = verify_and_derive_frontier(
        args.frontier_dir, Path(args.manifest), Path(args.freeze), Path(args.corpus_dir)
    )
    suite = build_suite(
        held,
        FRONTIER_HELDOUT_V1,
        source={"corpus": "frontier-corpus-v1", "side": "held_out",
                "content_sha256": manifest["content_sha256"],
                "note": "pilot-scale corpus (~4.2M characters); see EXPERIMENTS.md EXP-027"},
    )
    out = Path(args.out)
    if out.exists():
        existing = load_suite(out)
        if existing["fingerprint"] == suite["fingerprint"] and existing["source"] == suite["source"]:
            print(f"[suite] {out} already up to date (fingerprint {suite['fingerprint'][:16]}…)")
            return 0
        print(f"[suite] REFUSED: {out} exists with a different definition — a protected suite is "
              "never replaced in place (new suite id + founder approval)", file=sys.stderr)
        return 2
    write_suite(out, suite)
    t = suite["totals"]
    print(f"[suite] wrote {out}: {t['documents']:,} documents, {t['bytes']:,} bytes, "
          f"{len(suite['per_language'])} languages, fingerprint {suite['fingerprint'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
