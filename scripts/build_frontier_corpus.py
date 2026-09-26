#!/usr/bin/env python3
"""Build (or check) FrontierCorpus v1 (pilot) from the frozen ``indic-tokenizer/v2`` corpus.

FrontierCorpus v1 (MASTER_CONTEXT §37 step 3, D-037) turns the frozen,
hash-locked tokenizer corpus into a versioned, training-ready, **pre-tokenization**
text dataset:

    normalize (NFC, D-036) -> langid (script gate) -> quality -> exact dedup
        -> split (content-hash, reused from P004B) -> seeded shuffle -> pack -> shards
        -> dataset manifest

The pilot corpus is exactly the frozen ``indic-tokenizer/v2`` (D-035): the script
refuses to run unless the manifest bytes hash to the FREEZE.json identity and
every source's on-disk text hashes to its pinned sha256. No fetching, no new
sources, no licensing surface, no tokenization.

Examples
--------
    # build the pilot dataset (needs the v2 corpus text, i.e. a completed
    # build_tokenizer_corpus.py run under data/tokenizer/indic-tokenizer-v2)
    python scripts/build_frontier_corpus.py --exp-id EXP-027

    # verify an existing build against its manifest (read-only; no record)
    python scripts/build_frontier_corpus.py --check

Exit codes: 0 = built (or verified) OK; 1 = build or verification failed;
2 = bad or missing input (freeze mismatch, corpus text not built or drifted).

Experiment records (D-032): the outer run owns the record; ``run_self_recorded``
records this script itself when run directly and publishes metrics when nested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.corpus import (  # noqa: E402
    FRONTIER_CORPUS_ID,
    FRONTIER_CORPUS_VERSION,
    POLICY_VERSION,
    FrontierSource,
    PipelineDocument,
    RegistryError,
    check_shards,
    exact_dedup,
    langid_documents,
    language_scripts,
    load_frontier_registry,
    normalize_documents,
    pack,
    quality_filter,
    seeded_shuffle,
    train_holdout,
    write_manifest,
    write_shards,
)
from frontier_ai.corpus.manifest import build_manifest  # noqa: E402
from frontier_ai.experiments import ExperimentSpec  # noqa: E402
from frontier_ai.experiments.autowire import run_self_recorded, stable_results  # noqa: E402
from frontier_ai.tokenization.research_corpus import (  # noqa: E402
    DEFAULT_HELD_OUT_FRACTION,
    DEFAULT_SPLIT_SEED,
    documents_from_text,
)

DEFAULT_MANIFEST = "corpora/tokenizer/indic-tokenizer-v2/sources.json"
DEFAULT_FREEZE = "corpora/tokenizer/indic-tokenizer-v2/FREEZE.json"
DEFAULT_CORPUS_DIR = "data/tokenizer/indic-tokenizer-v2"
DEFAULT_OUT = "corpora/frontier/v1"
DEFAULT_MAX_SHARD_CHARS = 1_000_000


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="frozen v2 manifest (default: %(default)s)")
    p.add_argument("--freeze", default=DEFAULT_FREEZE,
                   help="FREEZE.json identity record (default: %(default)s)")
    p.add_argument(
        "--corpus-dir", default=DEFAULT_CORPUS_DIR,
        help="v2 build directory holding sources/<id>.txt (default: %(default)s)",
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="Frontier v1 output directory (default: %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED,
                   help="split/shuffle seed (default: %(default)s)")
    p.add_argument(
        "--held-out", type=float, default=DEFAULT_HELD_OUT_FRACTION,
        help="held-out fraction (default: %(default)s)",
    )
    p.add_argument(
        "--max-shard-chars", type=int, default=DEFAULT_MAX_SHARD_CHARS,
        help="character budget per shard file (default: %(default)s)",
    )
    p.add_argument(
        "--check", action="store_true",
        help="read-only: verify the on-disk shards against the written manifest (no record)",
    )
    p.add_argument("--exp-id", default="EXP-000",
                   help="experiment id for the record and report (default: %(default)s)")
    p.add_argument("--notes", default="", help="notes for the experiment record")
    p.add_argument("--no-record", action="store_true", help="do not write an experiment record")
    return p


def _took(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def _verify_inputs(
    manifest_path: Path, freeze_path: Path, corpus_dir: Path
) -> tuple[list[FrontierSource], str]:
    """Load the freeze-verified registry and verify every on-disk text file.

    Returns (registry, freeze_sha256). Raises RegistryError on any input problem
    (the caller maps it to exit 2).
    """
    registry = list(load_frontier_registry(manifest_path, freeze_path))
    freeze_sha256 = json.loads(freeze_path.read_text(encoding="utf-8"))["manifest"]["sha256"]
    problems: list[str] = []
    for source in registry:
        text_path = corpus_dir / "sources" / f"{source.source_id}.txt"
        if not text_path.is_file():
            problems.append(f"missing text file {text_path}")
            continue
        text = text_path.read_text(encoding="utf-8")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != source.sha256:
            problems.append(
                f"{source.source_id}: text hash {actual} != pinned {source.sha256} "
                "(stale or drifted file — rebuild the v2 corpus first)"
            )
    if problems:
        raise RegistryError(
            "the v2 corpus text under "
            f"{corpus_dir} is not the frozen corpus ({len(problems)} problem(s)); the pilot "
            f"refuses to build on it:\n  " + "\n  ".join(problems)
        )
    return registry, freeze_sha256


def _derive_documents(registry: list[FrontierSource], corpus_dir: Path) -> list[PipelineDocument]:
    """The frozen corpus documents: the same document derivation the v2 build used."""
    documents: list[PipelineDocument] = []
    for source in sorted(registry, key=lambda s: s.source_id):
        text = (corpus_dir / "sources" / f"{source.source_id}.txt").read_text(encoding="utf-8")
        for d in documents_from_text(source.source_id, source.language, text):
            documents.append(PipelineDocument(d.doc_id, d.source_id, d.language, d.text))
    return documents


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown (not a git checkout)"


def _render_report(
    *,
    exp_id: str,
    git_sha: str,
    n_sources: int,
    freeze_sha256: str,
    stage_lines: list[str],
    split_stats: dict,
    shard_lines: list[str],
    manifest_sha: str,
    manifest_path: Path,
    took: float,
) -> str:
    lines = [
        f"FrontierCorpus v1 pilot build — {exp_id}",
        f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"git sha: {git_sha}",
        f"source registry: indic-tokenizer/v2 (frozen) manifest sha256 {freeze_sha256} — {n_sources} sources",
        f"normalization: nfc (policy version {POLICY_VERSION})",
        f"split: {split_stats['method']} seed={split_stats['seed']} "
        f"held_out={split_stats['held_out_fraction']}",
        "",
        "stage summary (documents in -> kept; removed with reasons; flagged kept)",
        *stage_lines,
        "",
        f"train: {split_stats['train']['documents']} documents / "
        f"{split_stats['train']['chars']} chars",
        f"held_out: {split_stats['held_out']['documents']} documents / "
        f"{split_stats['held_out']['chars']} chars",
        "",
        "shards",
        *shard_lines,
        "",
        f"manifest: {manifest_path} (sha256 {manifest_sha})",
        f"build took {_took(took)}",
        "",
        "Review: compare the exact_dedup removals against the frozen corpus's own identical-document",
        "counts (reports/EXP-023-inspection.txt: 396 total) — any delta must be explained by the",
        "stages that ran before dedup (normalization can merge composed/decomposed twins).",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    freeze_path = Path(args.freeze)
    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out)

    # ---- --check: read-only verification of an existing build -------------------
    if args.check:
        manifest_file = out_dir / "manifest.json"
        if not manifest_file.is_file():
            print(f"[frontier] --check: no manifest at {manifest_file} (nothing built yet)", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        problems: list[str] = []
        for side, sub in (("train", "train"), ("held_out", "heldout")):
            expected = {s["name"]: s["sha256"] for s in manifest["sides"][side]["shards"]}
            problems.extend(f"[{side}] {p}" for p in check_shards(out_dir / "shards" / sub, expected))
        if problems:
            print(f"[frontier] --check: FAIL ({len(problems)} problem(s))", file=sys.stderr)
            for problem in problems:
                print(f"[frontier]   {problem}", file=sys.stderr)
            return 1
        n_train = manifest["sides"]["train"]["documents"]
        n_held = manifest["sides"]["held_out"]["documents"]
        print(f"[frontier] --check: OK — train {n_train} docs, "
              f"held_out {n_held} docs, all shard hashes match")
        return 0

    # ---- input gate (exit 2 on bad/missing input, before any record) -------------
    if not manifest_path.is_file() or not freeze_path.is_file():
        print(f"[frontier] missing input: {manifest_path} or {freeze_path}", file=sys.stderr)
        return 2
    try:
        registry, freeze_sha256 = _verify_inputs(manifest_path, freeze_path, corpus_dir)
    except RegistryError as exc:
        print(f"[frontier] {exc}", file=sys.stderr)
        return 2

    lang_scripts = language_scripts(registry)

    def body() -> dict:
        started = time.monotonic()
        documents = _derive_documents(registry, corpus_dir)

        n = normalize_documents(documents, "nfc")
        lang = langid_documents(list(n.kept), lang_scripts)
        q = quality_filter(list(lang.kept))
        d = exact_dedup(list(q.kept))
        stages = (n, lang, q, d)

        split_out, held = train_holdout(list(d.kept), seed=args.seed, held_out_fraction=args.held_out)
        train_docs = seeded_shuffle(list(split_out.kept), args.seed)
        held_docs = seeded_shuffle(list(held), args.seed + 1)  # derived seed, recorded in the manifest
        train_shards = pack(train_docs, args.max_shard_chars, "train")
        held_shards = pack(held_docs, args.max_shard_chars, "heldout")
        write_shards(train_shards, out_dir / "shards" / "train")
        write_shards(held_shards, out_dir / "shards" / "heldout")

        git_sha = _git_sha()
        manifest = build_manifest(
            git_sha=git_sha,
            registry=registry,
            source_registry_path=str(manifest_path),
            freeze_sha256=freeze_sha256,
            seed=args.seed,
            held_out_fraction=args.held_out,
            normalization_policy="nfc",
            policy_version=POLICY_VERSION,
            split_stats=split_out.stats,
            stages=stages,
            train_shards=train_shards,
            heldout_shards=held_shards,
            train_per_language=split_out.stats["train"]["per_language"],
            heldout_per_language=split_out.stats["held_out"]["per_language"],
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        manifest_sha = write_manifest(manifest, out_dir / "manifest.json")

        stage_lines = [
            f"  {stage.stage:<13} in={stage.stats['documents_in']:>6}  "
            f"kept={stage.stats['documents_out']:>6}  removed={stage.stats.get('removed', 0):>4}  "
            f"flagged_docs={stage.stats.get('flagged_documents', 0):>4}"
            for stage in stages
        ]
        shard_lines = [
            f"  {side:<8} {sh.name}  {len(sh.documents):>6} docs  {sh.chars:>9} chars  {sh.sha256[:16]}"
            for side, shards in (("train", train_shards), ("heldout", held_shards))
            for sh in shards
        ]

        report_path = out_dir / "reports" / f"{args.exp_id}-build.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            _render_report(
                exp_id=args.exp_id,
                git_sha=git_sha,
                n_sources=len(registry),
                freeze_sha256=freeze_sha256,
                stage_lines=stage_lines,
                split_stats=split_out.stats,
                shard_lines=shard_lines,
                manifest_sha=manifest_sha,
                manifest_path=out_dir / "manifest.json",
                took=time.monotonic() - started,
            ),
            encoding="utf-8",
            newline="\n",
        )

        print(f"[frontier] {FRONTIER_CORPUS_ID} v{FRONTIER_CORPUS_VERSION} built under {out_dir}")
        print(
            f"[frontier] documents: {len(documents)} in -> {len(d.kept)} after stages "
            f"-> train {len(train_docs)} / held_out {len(held_docs)}"
        )
        print(
            f"[frontier] shards: train {len(train_shards)}, held_out {len(held_shards)}; "
            f"manifest sha256 {manifest_sha[:16]}...; report: {report_path}"
        )
        print(f"[frontier] this run took {_took(time.monotonic() - started)}", file=sys.stderr, flush=True)
        return stable_results(
            {
                "stage_stats": [s.to_dict() for s in stages],
                "split": split_out.stats,
                "sides": {
                    "train": {
                        "documents": len(train_docs),
                        "chars": sum(sh.chars for sh in train_shards),
                        "shards": [sh.to_dict() for sh in train_shards],
                    },
                    "held_out": {
                        "documents": len(held_docs),
                        "chars": sum(sh.chars for sh in held_shards),
                        "shards": [sh.to_dict() for sh in held_shards],
                    },
                },
                "manifest_sha256": manifest_sha,
                "report": str(report_path),
            }
        )

    if args.no_record:
        body()
        return 0

    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            seed=args.seed,
            name="FrontierCorpus v1 pilot build",
            output_dir=str(Path("out/experiments") / args.exp_id),
            params={
                "source_registry": str(manifest_path),
                "freeze_sha256": freeze_sha256,
                "corpus_dir": str(corpus_dir),
                "out": str(out_dir),
                "seed": args.seed,
                "held_out_fraction": args.held_out,
                "max_shard_chars": args.max_shard_chars,
                "normalization": "nfc",
            },
            data_paths=[str(manifest_path), str(freeze_path)],
            command=list(sys.argv),
            tags=["frontier-corpus", "v1-pilot"],
            notes=args.notes,
        )

    return run_self_recorded(build_spec, body).exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
