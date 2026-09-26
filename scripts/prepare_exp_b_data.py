#!/usr/bin/env python3
"""EXP-B data preparation: tokenise FrontierCorpus v1 with the sweep's artifacts.

For each candidate tokenizer (an EXP-028 artifact directory), this script
encodes the frozen corpus's train side and held-out side — documents in
canonical doc_id order, same order the sweep trained on — into the model's
training format (``<name>.bin`` + ``<name>.meta.json``, document-aligned
splits with exact per-split UTF-8 byte/character counts for bits-per-byte).

Input identity is the same hard gate as the sweep (exit 2 on any failure):
shards hash to their manifest, per-language counts match, and the
deterministic re-derivation reproduces the frozen shards exactly. The frozen
dataset is never modified.

Tokenizer loading is auto-detected per artifact: ``bpe_python.json`` ->
``PythonBPE``, ``tokenizer.json`` -> ``HuggingFaceBPE`` (which re-attaches the
mark-aware pre-tokenizer for the mark_aware cells).

Example
-------
    python scripts/prepare_exp_b_data.py --exp-id EXP-029 --tokens \\
        mark_aware-32768=out/experiments/EXP-028/py-mark_aware-32768/seed-1337/tokenizer, \\
        mark_aware-16384=out/experiments/EXP-028/py-mark_aware-16384/seed-1337/tokenizer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.corpus import verify_and_derive_frontier  # noqa: E402
from frontier_ai.data.dataset import write_split_tokens  # noqa: E402
from frontier_ai.experiments import ExperimentSpec  # noqa: E402
from frontier_ai.experiments.autowire import run_self_recorded  # noqa: E402
from frontier_ai.tokenization.base import SubwordTokenizer  # noqa: E402

DEFAULT_FRONTIER_DIR = "corpora/frontier/v1"
DEFAULT_MANIFEST = "corpora/tokenizer/indic-tokenizer-v2/sources.json"
DEFAULT_FREEZE = "corpora/tokenizer/indic-tokenizer-v2/FREEZE.json"
DEFAULT_CORPUS_DIR = "data/tokenizer/indic-tokenizer-v2"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="EXP-029", help="experiment id (default: %(default)s)")
    p.add_argument("--frontier-dir", default=DEFAULT_FRONTIER_DIR,
                   help="FrontierCorpus v1 dataset dir (default: %(default)s)")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="frozen v2 manifest (default: %(default)s)")
    p.add_argument("--freeze", default=DEFAULT_FREEZE,
                   help="FREEZE.json identity record (default: %(default)s)")
    p.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR,
                   help="v2 build directory holding sources/<id>.txt (default: %(default)s)")
    p.add_argument("--tokens", required=True,
                   help="comma-separated NAME=ARTIFACT_DIR pairs (the sweep's per-config "
                        "tokenizer dirs), e.g. mark_aware-32768=out/experiments/EXP-028/"
                        "py-mark_aware-32768/seed-1337/tokenizer")
    p.add_argument("--out", default=None,
                   help="output dir (default: out/exp_b/<exp-id>)")
    p.add_argument("--notes", default="", help="notes for the experiment record")
    p.add_argument("--no-record", action="store_true", help="do not write the experiment record")
    return p


def _parse_tokens(raw: str) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"[exp-b] bad --tokens entry {part!r} (expected NAME=ARTIFACT_DIR)")
        name, path = part.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"[exp-b] bad --tokens entry {part!r}")
        specs.append((name, Path(path)))
    if not specs:
        raise SystemExit("[exp-b] --tokens must name at least one tokenizer")
    return specs


def _load_tokenizer(artifact_dir: Path) -> SubwordTokenizer:
    if (artifact_dir / "bpe_python.json").is_file():
        from frontier_ai.tokenization.bpe_python import PythonBPE

        return PythonBPE.load(artifact_dir)
    if (artifact_dir / "tokenizer.json").is_file():
        from frontier_ai.tokenization.bpe_hf import HuggingFaceBPE

        return HuggingFaceBPE.load(artifact_dir)
    raise SystemExit(
        f"[exp-b] no recognised tokenizer artifact in {artifact_dir} "
        "(expected bpe_python.json or tokenizer.json)"
    )


def _artifact_sha256(artifact_dir: Path) -> str:
    """Fingerprint of the artifact files (provenance for the data manifest)."""
    h = hashlib.sha256()
    for name in sorted(p.name for p in artifact_dir.iterdir() if p.is_file()):
        h.update(name.encode("utf-8"))
        h.update(artifact_dir.joinpath(name).read_bytes())
    return h.hexdigest()


def _encode_docs(tokenizer: SubwordTokenizer, docs) -> tuple[list[int], int, int]:
    """Token ids (doc_id order) + UTF-8 byte and char totals of the given side."""
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer.encode(doc.text))
    return ids, sum(len(d.text.encode("utf-8")) for d in docs), sum(d.chars for d in docs)


def prepare(
    out_dir: Path,
    specs: list[tuple[str, Path]],
    train_docs,
    held_docs,
    manifest: dict,
) -> dict:
    started = time.monotonic()
    tokenizer_records: list[dict] = []
    for name, artifact_dir in specs:
        if not artifact_dir.is_dir():
            raise SystemExit(f"[exp-b] tokenizer artifact dir missing: {artifact_dir}")
        t0 = time.monotonic()
        tok = _load_tokenizer(artifact_dir)
        vocab_size = tok.vocab_size
        train_ids, train_bytes, train_chars = _encode_docs(tok, train_docs)
        val_ids, val_bytes, val_chars = _encode_docs(tok, held_docs)
        write_split_tokens(
            out_dir / f"{name}.bin",
            train_ids,
            val_ids,
            vocab_size,
            level="bpe",
            train_bytes=train_bytes,
            train_chars=train_chars,
            val_bytes=val_bytes,
            val_chars=val_chars,
            source_provenance={
                "id": "frontier-corpus-v1",
                "content_sha256": manifest["content_sha256"],
                "license_id": "public-domain-sources (see frozen v2 registry)",
            },
        )
        tokenizer_records.append(
            {
                "name": name,
                "artifact": str(artifact_dir),
                "artifact_sha256": _artifact_sha256(artifact_dir),
                "vocab_size": vocab_size,
                "n_tokens_train": len(train_ids),
                "n_tokens_val": len(val_ids),
                "n_bytes_train": train_bytes,
                "n_bytes_val": val_bytes,
                "n_chars_train": train_chars,
                "n_chars_val": val_chars,
                "file": f"{name}.bin",
            }
        )
        print(
            f"[exp-b] {name}: vocab {vocab_size} | train {len(train_ids):,} tokens "
            f"({train_bytes:,} bytes) | val {len(val_ids):,} tokens ({val_bytes:,} bytes) "
            f"({time.monotonic() - t0:.1f}s)",
            flush=True,
        )

    out_data = {
        "schema": "frontier-exp-b-data-v1",
        "exp_id": "EXP-029",
        "frontier_manifest_content_sha256": manifest["content_sha256"],
        "document_order": "doc_id ascending (canonical; same order the sweep trained on)",
        "tokenizers": tokenizer_records,
        "prepare_seconds": round(time.monotonic() - started, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
    return out_data


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out or f"out/exp_b/{args.exp_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _parse_tokens(args.tokens)
    train_docs, held_docs, manifest = verify_and_derive_frontier(
        args.frontier_dir, Path(args.manifest), Path(args.freeze), Path(args.corpus_dir)
    )

    def body() -> dict:
        out_data = prepare(out_dir, specs, train_docs, held_docs, manifest)
        # no paths in results: two runs into different directories must fingerprint the same
        return {
            "tokenizers": [
                {k: v for k, v in t.items() if k not in ("artifact", "file")}
                for t in out_data["tokenizers"]
            ],
            "prepare_seconds": out_data["prepare_seconds"],
        }

    if args.no_record:
        body()
        print(f"[exp-b] prepared data in {out_dir}")
        return 0

    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            seed=1337,
            name="EXP-B data preparation (FrontierCorpus v1 -> sweep tokenizers)",
            output_dir=str(out_dir),
            params={
                "frontier_dir": str(args.frontier_dir),
                "frontier_manifest_content_sha256": manifest["content_sha256"],
                "tokenizers": [f"{n}={a}" for n, a in specs],
            },
            data_paths=[
                str(Path(args.frontier_dir) / "manifest.json"),
                str(args.manifest),
                str(args.freeze),
            ],
            command=list(sys.argv),
            tags=["tokenizer", "exp-b", "frontier-corpus-v1"],
            notes=args.notes,
        )

    recorded = run_self_recorded(build_spec, body)
    if recorded.failed:
        print(f"[exp-b] FAILED — {recorded.error}", file=sys.stderr)
        return 1
    print(f"[exp-b] prepared data in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
