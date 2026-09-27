#!/usr/bin/env python3
"""EXP-030: freeze Frontier Tokenizer v1 (D-040) into a tracked, hash-verified directory.

Copies the selected EXP-028 artifact into ``tokenizers/frontier-tokenizer-v1/`` and
writes ``FREEZE.json`` — but only after four gates pass on the source artifact:

  A  identity   the artifact directory fingerprint equals the ``artifact_sha256`` that
                EXP-029 recorded for this tokenizer (proves it is the exact tokenizer
                that won EXP-B, not a look-alike);
  B  structure  it loads, with the expected vocabulary size, merge count,
                pre-tokenizer and no special tokens;
  C  counts     re-encoding the frozen FrontierCorpus v1 (identity re-gated as in
                EXP-A/EXP-B) reproduces EXP-029's train and held-out token counts
                exactly, on the same corpus fingerprint;
  D  lossless   every document decodes back to exactly its original text.

Any gate failure: nothing is written to the destination, exit 1. Input problems
(missing artifact / EXP-029 data manifest / corpus): exit 2. After writing, the frozen
copy is re-verified through ``load_frozen_tokenizer`` (the loader downstream code
uses), including a set of golden samples whose token ids are stored in FREEZE.json so
another machine can prove it encodes identically.

Example (PC, from the repo root; the defaults are the EXP-030 values):
    .venv\\Scripts\\python.exe scripts\\freeze_tokenizer.py
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.corpus import verify_and_derive_frontier  # noqa: E402
from frontier_ai.experiments import ExperimentSpec  # noqa: E402
from frontier_ai.experiments.autowire import run_self_recorded  # noqa: E402
from frontier_ai.tokenization.bpe_python import PythonBPE  # noqa: E402
from frontier_ai.tokenization.frozen import (  # noqa: E402
    ARTIFACT_FILE,
    ARTIFACT_SUBDIR,
    FREEZE_NAME,
    FREEZE_SCHEMA,
    FrozenTokenizerError,
    artifact_dir_sha256,
    file_sha256,
    load_frozen_tokenizer,
    verify_structure,
)

DEFAULT_SOURCE = "out/experiments/EXP-028/py-mark_aware-32768/seed-0000001337/tokenizer"
DEFAULT_EXP029_DATA = "out/exp_b/EXP-029/manifest.json"
DEFAULT_NAME = "mark_aware-32768"
DEFAULT_DEST = "tokenizers/frontier-tokenizer-v1"
DEFAULT_FRONTIER_DIR = "corpora/frontier/v1"
DEFAULT_MANIFEST = "corpora/tokenizer/indic-tokenizer-v2/sources.json"
DEFAULT_FREEZE = "corpora/tokenizer/indic-tokenizer-v2/FREEZE.json"
DEFAULT_CORPUS_DIR = "data/tokenizer/indic-tokenizer-v2"

# Fixed texts whose token ids are recorded at freeze time. Any machine that loads the
# frozen tokenizer must reproduce these ids exactly (cross-platform determinism).
# One line per corpus language + code-mixing + combining-mark / whitespace edge cases.
GOLDEN_TEXTS: list[tuple[str, str]] = [
    ("en", "The quick brown fox jumps over the lazy dog, 42 times!"),
    ("hi", "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं।"),
    ("bn", "আমি বাংলায় কথা বলতে ভালোবাসি।"),
    ("mr", "महाराष्ट्रातील लोक मराठी भाषा बोलतात."),
    ("gu", "ગુજરાતી ભાષા ખૂબ સુંદર છે."),
    ("ta", "தமிழ் ஒரு பழமையான மொழி ஆகும்."),
    ("te", "తెలుగు భాష చాలా మధురమైనది."),
    ("kn", "ಕನ್ನಡ ಒಂದು ಸುಂದರ ಭಾಷೆ."),
    ("ml", "മലയാളം കേരളത്തിലെ ഭാഷയാണ്."),
    ("pa", "ਪੰਜਾਬੀ ਇੱਕ ਮਿੱਠੀ ਬੋਲੀ ਹੈ।"),
    ("or", "ଓଡ଼ିଆ ଏକ ପୁରୁଣା ଭାଷା ଅଟେ।"),
    ("as", "অসমীয়া ভাষা অসমৰ মানুহে কয়।"),
    ("ur", "اردو ایک خوبصورت زبان ہے۔"),
    ("mixed", "Kal meeting 10:30 AM ko hai — मैं ready हूँ! ✓ ٹھیک ہے 2026"),
    ("edge", "क़िला ज़रूर; ক্ষ; க்‍ஷ; a\u200cb\tline\nnext  double  space"),
]

CHANGE_POLICY = (
    "Frontier Tokenizer v1 is never edited in place. Any change (retraining, a different "
    "vocabulary, added special tokens, a different pre-tokenizer or implementation "
    "artifact) requires the founder's approval, is recorded as a new experiment, and "
    "produces frontier-tokenizer-v2 with its own freeze record (D-041)."
)
DOWNSTREAM_USE = (
    "Load only via frontier_ai.tokenization.frozen.load_frontier_tokenizer(), which refuses "
    "a tokenizer whose bytes, structure or golden-sample encodings differ from this record. "
    "Cite 'frontier-tokenizer-v1' plus artifact.dir_sha256."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="EXP-030", help="experiment id (default: %(default)s)")
    p.add_argument("--source", default=DEFAULT_SOURCE, help="artifact dir to freeze (default: %(default)s)")
    p.add_argument("--exp029-data", default=DEFAULT_EXP029_DATA,
                   help="EXP-029 prepared-data manifest holding the artifact fingerprint and token "
                        "counts (default: %(default)s)")
    p.add_argument("--name", default=DEFAULT_NAME,
                   help="tokenizer name inside the EXP-029 data manifest (default: %(default)s)")
    p.add_argument("--dest", default=DEFAULT_DEST, help="frozen destination dir (default: %(default)s)")
    p.add_argument("--decision", default="D-040", help="decision that selected it (default: %(default)s)")
    p.add_argument("--expect-vocab", type=int, default=32768,
                   help="expected vocab size (default: %(default)s)")
    p.add_argument("--expect-pretoken", default="mark_aware",
                   help="expected pre-tokenizer (default: %(default)s)")
    p.add_argument("--lineage-cell", default="py-mark_aware-32768",
                   help="EXP-028 sweep cell name (default: %(default)s)")
    p.add_argument("--lineage-seed", type=int, default=1337, help="EXP-028 cell seed (default: %(default)s)")
    p.add_argument("--frontier-dir", default=DEFAULT_FRONTIER_DIR)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--freeze", default=DEFAULT_FREEZE)
    p.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    p.add_argument("--notes", default="", help="notes for the experiment record")
    p.add_argument("--no-record", action="store_true", help="do not write the experiment record")
    return p


def _die(message: str) -> SystemExit:
    print(f"[freeze] INPUT ERROR: {message}", file=sys.stderr)
    return SystemExit(2)


def _encode_side(tok: PythonBPE, docs) -> tuple[int, int, list[str]]:
    """(token count, documents, first failing doc_ids) — one encode per document."""
    n_tokens = 0
    failures: list[str] = []
    for doc in docs:
        ids = tok.encode(doc.text)
        n_tokens += len(ids)
        if tok.decode(ids) != doc.text and len(failures) < 5:
            failures.append(doc.doc_id)
    return n_tokens, len(docs), failures


def run_gates(args, source: Path, entry: dict, exp029: dict, train_docs, held_docs, corpus_manifest) -> dict:
    """All four gates on the SOURCE artifact. Returns the gates dict (each has 'pass')."""
    gates: dict = {}

    observed_dir = artifact_dir_sha256(source)
    gates["A_identity"] = {
        "what": "artifact dir fingerprint == EXP-029 recorded artifact_sha256",
        "expected": entry["artifact_sha256"],
        "observed": observed_dir,
        "pass": observed_dir == entry["artifact_sha256"],
    }

    tok = PythonBPE.load(source)
    expected_struct = {
        "vocab_size": args.expect_vocab,
        "merges": args.expect_vocab - 256,
        "pretoken": args.expect_pretoken,
        "special_tokens": [],
    }
    problems = verify_structure(tok, expected_struct)
    if tok.vocab_size != entry["vocab_size"]:
        problems.append(f"vocab_size {tok.vocab_size} != EXP-029 record {entry['vocab_size']}")
    gates["B_structure"] = {"expected": expected_struct, "problems": problems, "pass": not problems}

    started = time.monotonic()
    n_train, d_train, f_train = _encode_side(tok, train_docs)
    n_held, d_held, f_held = _encode_side(tok, held_docs)
    same_corpus = corpus_manifest["content_sha256"] == exp029["frontier_manifest_content_sha256"]
    gates["C_token_counts"] = {
        "corpus_content_sha256": corpus_manifest["content_sha256"],
        "same_corpus_as_exp029": same_corpus,
        "train": {"expected": entry["n_tokens_train"], "observed": n_train},
        "held_out": {"expected": entry["n_tokens_val"], "observed": n_held},
        "pass": same_corpus and n_train == entry["n_tokens_train"] and n_held == entry["n_tokens_val"],
    }
    gates["D_lossless"] = {
        "documents": d_train + d_held,
        "failing_doc_ids_sample": f_train + f_held,
        "pass": not (f_train or f_held),
    }
    gates["encode_seconds"] = round(time.monotonic() - started, 1)
    return gates


def write_frozen(args, source: Path, dest: Path, gates: dict, exp029: dict) -> dict:
    """Copy the artifact byte-exactly + write FREEZE.json; returns the freeze record."""
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / ARTIFACT_SUBDIR).mkdir(parents=True)
    files = sorted(p.name for p in source.iterdir() if p.is_file())
    for name in files:
        shutil.copyfile(source / name, tmp / ARTIFACT_SUBDIR / name)

    tok = PythonBPE.load(tmp / ARTIFACT_SUBDIR)
    art_json = json.loads((tmp / ARTIFACT_SUBDIR / ARTIFACT_FILE).read_text(encoding="utf-8"))
    freeze = {
        "schema": FREEZE_SCHEMA,
        "tokenizer_id": "frontier-tokenizer",
        "tokenizer_version": "v1",
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_by": f"{args.exp_id} (scripts/freeze_tokenizer.py)",
        "decision": args.decision,
        "artifact": {
            "dir": f"{Path(args.dest).as_posix()}/{ARTIFACT_SUBDIR}",
            "impl": "bpe_python",
            "impl_version": tok.impl_version(),
            "files": {name: file_sha256(tmp / ARTIFACT_SUBDIR / name) for name in files},
            "dir_sha256": artifact_dir_sha256(tmp / ARTIFACT_SUBDIR),
            "size_bytes": sum((tmp / ARTIFACT_SUBDIR / n).stat().st_size for n in files),
            "vocab_size": tok.vocab_size,
            "merges": len(tok.merges),
            "pretoken": art_json.get("pretoken"),
            "special_tokens": list(tok.special_token_ids),
            "note": "hashes are over raw bytes (the file may carry CRLF line endings from Windows); "
                    "git stores this directory byte-exactly via .gitattributes",
        },
        "lineage": {
            "source_artifact": source.as_posix(),
            "sweep_experiment": "EXP-028",
            "sweep_cell": args.lineage_cell,
            "seed": args.lineage_seed,
            "trained_chars": art_json.get("trained_chars"),
            "training_corpus": {
                "id": "frontier-corpus-v1",
                "side": "train",
                "content_sha256": exp029["frontier_manifest_content_sha256"],
            },
            "selected_by": "EXP-029 (EXP-B small-model comparison)",
            "exp029_name": args.name,
        },
        "gates": gates,
        "golden_samples": [
            {"id": sid, "text": text, "ids": tok.encode(text)} for sid, text in GOLDEN_TEXTS
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "change_policy": CHANGE_POLICY,
        "downstream_use": DOWNSTREAM_USE,
    }
    with open(tmp / FREEZE_NAME, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")

    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return freeze


def freeze_tokenizer(args) -> tuple[bool, dict]:
    source, dest = Path(args.source), Path(args.dest)
    if (dest / FREEZE_NAME).exists():
        raise _die(f"{dest / FREEZE_NAME} already exists — a frozen tokenizer is never edited in place "
                   "(change policy: new version + founder approval)")
    if not (source / ARTIFACT_FILE).is_file():
        raise _die(f"no {ARTIFACT_FILE} in {source}")
    exp029_path = Path(args.exp029_data)
    if not exp029_path.is_file():
        raise _die(f"EXP-029 data manifest not found: {exp029_path}")
    exp029 = json.loads(exp029_path.read_text(encoding="utf-8"))
    entries = [t for t in exp029.get("tokenizers", []) if t.get("name") == args.name]
    if len(entries) != 1:
        raise _die(f"{exp029_path}: expected exactly one tokenizer named {args.name!r}")
    entry = entries[0]

    train_docs, held_docs, corpus_manifest = verify_and_derive_frontier(
        args.frontier_dir, Path(args.manifest), Path(args.freeze), Path(args.corpus_dir)
    )
    gates = run_gates(args, source, entry, exp029, train_docs, held_docs, corpus_manifest)
    for key in ("A_identity", "B_structure", "C_token_counts", "D_lossless"):
        print(f"[freeze] gate {key}: {'PASS' if gates[key]['pass'] else 'FAIL'}", flush=True)
    c = gates["C_token_counts"]
    print(f"[freeze]   train tokens {c['train']['observed']:,} (EXP-029 {c['train']['expected']:,}) | "
          f"held-out {c['held_out']['observed']:,} (EXP-029 {c['held_out']['expected']:,}) | "
          f"{gates['D_lossless']['documents']:,} docs round-tripped ({gates['encode_seconds']} s)")
    ok = all(gates[k]["pass"] for k in ("A_identity", "B_structure", "C_token_counts", "D_lossless"))
    if not ok:
        print("[freeze] FAILED — nothing was written to the destination.", file=sys.stderr)
        return False, {"gates": gates}

    freeze = write_frozen(args, source, dest, gates, exp029)
    try:
        load_frozen_tokenizer(dest)
    except FrozenTokenizerError as exc:
        shutil.rmtree(dest)
        print(f"[freeze] FAILED — frozen copy did not verify ({exc}); destination removed.", file=sys.stderr)
        return False, {"gates": gates}
    print(f"[freeze] frozen: {dest} | dir_sha256 {freeze['artifact']['dir_sha256'][:16]}… | "
          f"{freeze['artifact']['size_bytes']:,} bytes | {len(freeze['golden_samples'])} golden samples "
          "verified", flush=True)
    return True, {
        "gates": {k: v["pass"] for k, v in gates.items() if isinstance(v, dict)},
        "dir_sha256": freeze["artifact"]["dir_sha256"],
        "files": freeze["artifact"]["files"],
        "vocab_size": freeze["artifact"]["vocab_size"],
        "train_tokens": c["train"]["observed"],
        "held_out_tokens": c["held_out"]["observed"],
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.no_record:
        ok, _ = freeze_tokenizer(args)
        return 0 if ok else 1

    outcome: dict = {}

    def body() -> dict:
        ok, results = freeze_tokenizer(args)
        outcome["ok"] = ok
        return {"frozen": ok, **results}

    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            seed=args.lineage_seed,
            name="Freeze Frontier Tokenizer v1 (D-040)",
            output_dir=f"out/tokenizer_freeze/{args.exp_id}",
            params={"source": args.source, "dest": args.dest, "name": args.name,
                    "expect_vocab": args.expect_vocab, "expect_pretoken": args.expect_pretoken},
            data_paths=[str(Path(args.frontier_dir) / "manifest.json"), str(args.manifest),
                        str(args.freeze), str(args.exp029_data)],
            command=list(sys.argv),
            tags=["tokenizer", "freeze", "frontier-tokenizer-v1"],
            notes=args.notes,
        )

    recorded = run_self_recorded(build_spec, body)
    if recorded.failed:
        print(f"[freeze] record FAILED — {recorded.error}", file=sys.stderr)
        return 1
    return 0 if outcome.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
