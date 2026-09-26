#!/usr/bin/env python3
"""EXP-A: the production tokenizer sweep over FrontierCorpus v1 (MASTER_CONTEXT §37 step 5).

Trains the approved 15-configuration grid — ``bpe_python`` x {mark_aware,
gpt2_style} x 5 vocab sizes, plus ``bpe_hf`` (built-in ByteLevel) x 5 vocab
sizes — on the FrontierCorpus v1 train side, gates every configuration on
losslessness (exact round-trip of EVERY document, both sides), and scores the
held-out side with model-free token-density metrics. No tokenizer is selected
here: the top-2 hand-off to EXP-B happens after the sweep is reviewed.

Input identity (hard gates; exit 2 on any failure)
--------------------------------------------------
The dataset is the FROZEN ``corpora/frontier/v1``: on-disk shards must hash to
their manifest, the manifest's split parameters drive a deterministic
re-derivation of the per-language documents from the frozen v2 sources, and
the derivation must reproduce the frozen shards EXACTLY (per-language counts
and the full shard line sequence, side by side). The frozen dataset is never
modified.

Exit codes: 0 = sweep complete and every configuration passed; 1 = sweep ran
but at least one configuration failed (training error or losslessness gate);
2 = bad or missing input (gates above).

Experiment records (D-032): one full record per configuration (the sweep
framework), plus this script's own record when run directly.

Examples
--------
    # the full approved sweep (the PC job; ~a few hours, CPU only)
    python scripts/run_tokenizer_sweep.py --exp-id EXP-028

    # smoke: tiny vocabs, truncated train side (sandbox / preflight)
    python scripts/run_tokenizer_sweep.py --exp-id EXP-028 \
        --vocab-sizes 512,1024 --max-train-chars 200000 --no-record
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.corpus import (  # noqa: E402
    check_shards,
    derive_frontier_documents,
    seeded_shuffle,
)
from frontier_ai.corpus.pipeline import sha256_text  # noqa: E402
from frontier_ai.corpus.registry import RegistryError  # noqa: E402
from frontier_ai.experiments import ExperimentSpec  # noqa: E402
from frontier_ai.experiments.autowire import run_self_recorded  # noqa: E402
from frontier_ai.experiments.sweep import run_sweep  # noqa: E402
from frontier_ai.tokenization import is_available  # noqa: E402
from frontier_ai.tokenization.sweep import (  # noqa: E402
    HEADLINE_METRIC,
    IMPL_BPE_HF,
    SweepConfig,
    run_one_config,
)

DEFAULT_FRONTIER_DIR = "corpora/frontier/v1"
DEFAULT_MANIFEST = "corpora/tokenizer/indic-tokenizer-v2/sources.json"
DEFAULT_FREEZE = "corpora/tokenizer/indic-tokenizer-v2/FREEZE.json"
DEFAULT_CORPUS_DIR = "data/tokenizer/indic-tokenizer-v2"
DEFAULT_VOCAB_SIZES = "2048,4096,8192,16384,32768"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="EXP-028", help="experiment id (default: %(default)s)")
    p.add_argument("--frontier-dir", default=DEFAULT_FRONTIER_DIR,
                   help="FrontierCorpus v1 dataset dir (default: %(default)s)")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="frozen v2 manifest (default: %(default)s)")
    p.add_argument("--freeze", default=DEFAULT_FREEZE,
                   help="FREEZE.json identity record (default: %(default)s)")
    p.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR,
                   help="v2 build directory holding sources/<id>.txt (default: %(default)s)")
    p.add_argument("--vocab-sizes", default=DEFAULT_VOCAB_SIZES,
                   help="comma-separated vocab sizes (default: %(default)s)")
    p.add_argument("--out", default=None,
                   help="sweep output dir (default: out/experiments/<exp-id>)")
    p.add_argument("--seed", type=int, default=1337,
                   help="sweep record seed (the split seed comes from the dataset manifest)")
    p.add_argument("--max-train-chars", type=int, default=None,
                   help="SMOKE MODE: truncate the train text file to this many characters")
    p.add_argument("--notes", default="", help="notes for the experiment record")
    p.add_argument("--no-record", action="store_true", help="do not write the outer experiment record")
    return p


def _fail_input(message: str) -> int:
    print(f"[sweep] INPUT GATE FAILED: {message}", file=sys.stderr)
    return 2


def _parse_vocab_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            size = int(part)
        except ValueError:
            raise SystemExit(
                f"[sweep] bad --vocab-sizes entry {part!r} (expected integers)"
            ) from None
        if size < 256:
            raise SystemExit(
                f"[sweep] vocab size {size} is below the 256-byte base vocabulary"
            ) from None
        sizes.append(size)
    if not sizes:
        raise SystemExit("[sweep] --vocab-sizes must name at least one size") from None
    return sorted(set(sizes))


# ---------------------------------------------------------------------------
# input gates (exit 2)
# ---------------------------------------------------------------------------
def _load_frontier(frontier_dir: Path) -> dict:
    manifest_file = frontier_dir / "manifest.json"
    if not manifest_file.is_file():
        print(
            f"[sweep] INPUT GATE FAILED: no dataset manifest at {manifest_file} — build the "
            "pilot first (scripts/build_frontier_corpus.py) or point --frontier-dir at the "
            "existing build (on the PC: corpora/frontier/v1)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return json.loads(manifest_file.read_text(encoding="utf-8"))


def _gate_shards_match_manifest(frontier_dir: Path, manifest: dict) -> None:
    problems: list[str] = []
    for side, sub in (("train", "train"), ("held_out", "heldout")):
        expected = {s["name"]: s["sha256"] for s in manifest["sides"][side]["shards"]}
        problems.extend(f"[{side}] {p}" for p in check_shards(frontier_dir / "shards" / sub, expected))
    if problems:
        print(
            "[sweep] INPUT GATE FAILED: on-disk shards do not match the frozen dataset "
            f"manifest ({len(problems)} problem(s)):\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        raise SystemExit(2)


def _gate_derivation_matches_shards(
    frontier_dir: Path, manifest: dict, derivation
) -> None:
    """The re-derived documents must reproduce the frozen shards exactly."""
    split = derivation.split.stats
    problems: list[str] = []

    # (a) counts: totals and per-language, side by side, against the manifest
    for side in ("train", "held_out"):
        manifest_side = manifest["sides"][side]
        derived_side = split[side]
        if derived_side["documents"] != manifest_side["documents"]:
            problems.append(
                f"[{side}] derived {derived_side['documents']} documents, "
                f"manifest has {manifest_side['documents']}"
            )
        for lang in sorted(set(manifest_side["per_language"]) | set(derived_side["per_language"])):
            m = manifest_side["per_language"].get(lang, {"documents": 0, "chars": 0})
            d = derived_side["per_language"].get(lang, {"documents": 0, "chars": 0})
            if (m["documents"], m["chars"]) != (d["documents"], d["chars"]):
                problems.append(
                    f"[{side}/{lang}] derived {d['documents']} docs / {d['chars']} chars, "
                    f"manifest has {m['documents']} docs / {m['chars']} chars"
                )

    # (b) exact sequence: the shard lines, in shard order, side by side
    for side, sub, docs in (
        ("train", "train", list(derivation.train)),
        ("held_out", "heldout", list(derivation.held_out)),
    ):
        shard_text: list[str] = []
        for shard in manifest["sides"][side]["shards"]:
            shard_text.append((frontier_dir / "shards" / sub / f"{shard['name']}.txt")
                              .read_text(encoding="utf-8"))
        shard_lines = [line for text in shard_text for line in text.split("\n")]
        seed = split["seed"]
        shuffled = seeded_shuffle(docs, seed) if side == "train" else seeded_shuffle(docs, seed + 1)
        derived_lines = [d.text for d in shuffled]
        if shard_lines != derived_lines:
            n_shard, n_derived = len(shard_lines), len(derived_lines)
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(shard_lines, derived_lines)) if a != b),
                min(n_shard, n_derived),
            )
            problems.append(
                f"[{side}] shard line sequence differs from the deterministic re-derivation "
                f"({n_shard} vs {n_derived} lines; first difference at line {first_diff}) — "
                "the frozen build and the pipeline on this machine disagree"
            )
    if problems:
        print(
            "[sweep] INPUT GATE FAILED: the re-derived documents do not reproduce the frozen "
            f"shards ({len(problems)} problem(s)):\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# shared training input (written once, consumed by every configuration)
# ---------------------------------------------------------------------------
def _write_shared_inputs(
    out_dir: Path, train_docs, held_docs, manifest: dict, max_train_chars: int | None
) -> Path:
    shared = out_dir / "_shared"
    shared.mkdir(parents=True, exist_ok=True)

    def docs_jsonl(docs, path: Path) -> None:
        lines = []
        for d in docs:
            lines.append(
                json.dumps(
                    {"doc_id": d.doc_id, "source_id": d.source_id, "language": d.language,
                     "chars": len(d.text), "text_sha256": sha256_text(d.text), "text": d.text},
                    ensure_ascii=False, sort_keys=True,
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    train_sorted = sorted(train_docs, key=lambda d: d.doc_id)
    held_sorted = sorted(held_docs, key=lambda d: d.doc_id)
    train_text = "\n".join(d.text for d in train_sorted)
    if max_train_chars is not None:
        train_text = train_text[:max_train_chars]
    (shared / "train.txt").write_text(train_text, encoding="utf-8", newline="\n")
    (shared / "heldout.txt").write_text("\n".join(d.text for d in held_sorted),
                                        encoding="utf-8", newline="\n")
    docs_jsonl(train_sorted, shared / "train_docs.jsonl")
    docs_jsonl(held_sorted, shared / "heldout_docs.jsonl")
    (shared / "input_note.json").write_text(
        json.dumps(
            {
                "train_documents": len(train_sorted),
                "train_file_chars": len(train_text),
                "max_train_chars": max_train_chars,
                "train_truncated": max_train_chars is not None
                and len(train_text) == max_train_chars,
                "heldout_documents": len(held_sorted),
                "order": "doc_id ascending (canonical; BPE pair counts are order-independent)",
                "special_tokens": [],
                "frontier_manifest_content_sha256": manifest["content_sha256"],
            },
            indent=2, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return shared / "train.txt"


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace) -> int:
    frontier_dir = Path(args.frontier_dir)
    manifest = _load_frontier(frontier_dir)
    _gate_shards_match_manifest(frontier_dir, manifest)

    split_identity = manifest["identity"]["split"]
    split_seed = int(split_identity["seed"])
    held_out_fraction = float(split_identity["held_out_fraction"])

    if not Path(args.manifest).is_file() or not Path(args.freeze).is_file():
        return _fail_input(f"missing frozen input: {args.manifest} or {args.freeze}")
    try:
        derivation = derive_frontier_documents(
            manifest_path=Path(args.manifest),
            freeze_path=Path(args.freeze),
            corpus_dir=Path(args.corpus_dir),
            seed=split_seed,
            held_out_fraction=held_out_fraction,
        )
    except RegistryError as exc:
        return _fail_input(str(exc))
    _gate_derivation_matches_shards(frontier_dir, manifest, derivation)

    train_docs = list(derivation.train)
    held_docs = list(derivation.held_out)
    print(
        f"[sweep] input verified: {manifest['content_sha256'][:16]}… (frozen shards match; "
        f"re-derivation reproduces them exactly)"
    )
    print(
        f"[sweep] documents: train {len(train_docs)} / {sum(d.chars for d in train_docs):,} chars; "
        f"held_out {len(held_docs)} / {sum(d.chars for d in held_docs):,} chars "
        f"(split seed {split_seed}, held_out {held_out_fraction})"
    )
    if args.max_train_chars is not None:
        print(f"[sweep] SMOKE MODE: train file truncated to {args.max_train_chars} chars")

    out_dir = Path(args.out or f"out/experiments/{args.exp_id}")
    train_file = _write_shared_inputs(out_dir, train_docs, held_docs, manifest, args.max_train_chars)

    vocab_sizes = _parse_vocab_sizes(args.vocab_sizes)
    configs = list(SweepConfig.grid(vocab_sizes))
    skipped_hf = False
    if not is_available(IMPL_BPE_HF):
        print("[sweep] WARNING: the 'tokenizers' package is not installed — "
              "the 5 bpe_hf configurations are skipped (pip install '.[tokenizer]')")
        configs = [c for c in configs if c.impl != IMPL_BPE_HF]
        skipped_hf = True

    per_language: dict[str, list[str]] = {}
    for doc in held_docs:
        per_language.setdefault(doc.language, []).append(doc.doc_id)
    print("[sweep] held_out languages: "
          + ", ".join(f"{lang}={len(ids)}" for lang, ids in sorted(per_language.items())))

    def body() -> dict:
        configurations = {
            cfg.name: {
                "params": {"impl": cfg.impl, "pretoken": cfg.pretoken, "vocab_size": cfg.vocab_size},
                "name": f"EXP-A cell: {cfg.impl} / {cfg.pretoken} / vocab {cfg.vocab_size}",
            }
            for cfg in configs
        }
        spec = ExperimentSpec(
            experiment_id=args.exp_id,
            seed=args.seed,
            name="EXP-A production tokenizer sweep",
            output_dir=str(out_dir),
            params={
                "frontier_dir": str(frontier_dir),
                "frontier_manifest_content_sha256": manifest["content_sha256"],
                "source_registry": str(args.manifest),
                "corpus_dir": str(args.corpus_dir),
                "vocab_sizes": vocab_sizes,
                "max_train_chars": args.max_train_chars,
                "split_seed": split_seed,
                "held_out_fraction": held_out_fraction,
            },
            data_paths=[
                str(frontier_dir / "manifest.json"),
                str(args.manifest),
                str(args.freeze),
            ]
            + [
                str(frontier_dir / "shards" / sub / f"{shard['name']}.txt")
                for side, sub in (("train", "train"), ("held_out", "heldout"))
                for shard in manifest["sides"][side]["shards"]
            ],
            command=list(sys.argv),
            tags=["tokenizer", "exp-a", "frontier-corpus-v1"],
            notes=args.notes,
        )

        def experiment_fn(ctx) -> dict:
            cfg = SweepConfig(
                impl=str(ctx.spec.params["impl"]),
                pretoken=str(ctx.spec.params["pretoken"]),
                vocab_size=int(ctx.spec.params["vocab_size"]),
            )
            print(f"[sweep]   {cfg.name}: training…", flush=True)
            started = time.monotonic()
            results = run_one_config(
                cfg,
                train_file=train_file,
                train_documents=train_docs,
                heldout_documents=held_docs,
                out_dir=ctx.output_dir,
                special_tokens=(),
            )
            gate = results["gate"]["lossless"]
            density = results["heldout"]["overall"].get("chars_per_token")
            print(
                f"[sweep]   {cfg.name}: gate={'PASS' if gate else 'FAIL'} "
                f"chars/token={density} train={results['train']['seconds']}s "
                f"({time.monotonic() - started:.1f}s total)",
                flush=True,
            )
            return results

        return run_sweep(
            spec,
            seeds=[args.seed],
            experiment_fn=experiment_fn,
            metric=HEADLINE_METRIC,
            output_dir=out_dir,
            extra_packages=("tokenizers",),
            configurations=configurations,
        ).record.to_dict()

    if args.no_record:
        sweep = body()
    else:
        def build_spec() -> ExperimentSpec:
            return ExperimentSpec(
                experiment_id=args.exp_id,
                seed=args.seed,
                name="EXP-A production tokenizer sweep",
                output_dir=str(out_dir),
                params={"see": "sweep.json"},
                data_paths=[str(frontier_dir / "manifest.json")],
                command=list(sys.argv),
                tags=["tokenizer", "exp-a", "frontier-corpus-v1"],
                notes=args.notes,
            )

        sweep = run_self_recorded(build_spec, body).results or {}

    return _report_and_exit(sweep, out_dir, skipped_hf)


def _report_and_exit(sweep: dict, out_dir: Path, skipped_hf: bool) -> int:
    section = sweep.get("sweep", {})
    runs = sweep.get("runs", [])
    lines = [
        "EXP-A tokenizer sweep — results",
        f"generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"sweep status: {section.get('status')}; runs {section.get('runs_successful')}/"
        f"{section.get('runs_requested')} usable",
        "",
        f"{'configuration':<24} {'gate':<6} {'train s':>9} {'chars/tok':>10} "
        f"{'tok/char':>9} {'vocab':>6}",
    ]
    all_ok = bool(runs)
    for run in runs:
        if run.get("status") != "success":
            all_ok = False
            continue
        record = _load_run_results(out_dir, run.get("record"))
        if record is None:
            all_ok = False
            continue
        gate_ok = bool(record.get("gate", {}).get("lossless"))
        all_ok = all_ok and gate_ok
        overall = record.get("heldout", {}).get("overall", {})
        lines.append(
            f"{run.get('configuration', ''):<24} "
            f"{'PASS' if gate_ok else 'FAIL':<6} "
            f"{record.get('train', {}).get('seconds', 0):>9.1f} "
            f"{overall.get('chars_per_token', 0):>10.4f} "
            f"{overall.get('tokens_per_char', 0):>9.4f} "
            f"{record.get('vocab_size', 0):>6}"
        )
    if skipped_hf:
        lines.append("")
        lines.append("note: the 5 bpe_hf configurations were skipped ('tokenizers' not installed)")

    # per-language held-out density (the number EXP-B's small model must beat)
    lines += ["", "held-out per-language chars_per_token (higher = denser):"]
    langs: list[str] = []
    per_config_lang: dict[str, dict[str, float]] = {}
    for run in runs:
        if run.get("status") != "success":
            continue
        record = _load_run_results(out_dir, run.get("record"))
        if record is None:
            continue
        pl = record.get("heldout", {}).get("per_language", {})
        langs.extend(pl)
        per_config_lang[run.get("configuration", "")] = {
            lang: block.get("chars_per_token", 0) for lang, block in pl.items()
        }
    langs = sorted(set(langs))
    if langs:
        header = f"{'configuration':<24}" + "".join(f"{lang:>10}" for lang in langs)
        lines.append(header)
        for cfg in sorted(per_config_lang):
            row = f"{cfg:<24}"
            for lang in langs:
                value = per_config_lang[cfg].get(lang)
                row += f"{value:>10.4f}" if value is not None else f"{'n/a':>10}"
            lines.append(row)

    report = "\n".join(lines) + "\n"
    (out_dir / "report.txt").write_text(report, encoding="utf-8", newline="\n")
    print()
    print(report)
    print(f"[sweep] report: {out_dir / 'report.txt'}; sweep record: {out_dir / 'sweep.json'}")
    if not all_ok:
        print("[sweep] exit 1: at least one configuration failed (see failed runs in sweep.json)")
        return 1
    return 0


def _load_run_results(out_dir: Path, record_rel: str | None) -> dict | None:
    if not record_rel:
        return None
    path = (out_dir / record_rel).resolve()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("results") or {}
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    args = build_parser().parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
