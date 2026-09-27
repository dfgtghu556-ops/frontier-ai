#!/usr/bin/env python3
"""Evaluation harness v1: one command, one checkpoint, one full report card.

    python scripts/eval_report.py --ckpt out/exp_b/EXP-029/runs/mark_aware-32768/seed-1337/best

Scores the protected evaluation suite (default: ``frontier-heldout-v1`` = the FrontierCorpus
v1 held-out side) with every token scored exactly once (protocol in
``frontier_ai.evaluation.scoring``) and writes, into ``--out``:

  report.json          machine-readable report (schema frontier-eval-report-v1)
  report.txt           human-readable summary
  per_document.jsonl   per-document bits (input for scripts/eval_compare.py)

Checks performed before any number is reported (exit 2 on failure):
  * the checkpoint's vocabulary equals the tokenizer's;
  * the re-derived evaluation documents are exactly the suite's (hash by hash, in order).
Reported checks:
  * data identity — the re-encoded stream equals the validation split of the dataset
    the checkpoint was trained with (exit 1 if it differs, NOT CHECKED if absent);
  * contamination — exact and 13-gram overlap with the FrontierCorpus v1 train side.

Tokenizer: ``--tokenizer frozen`` (default) loads Frontier Tokenizer v1 through the
hash-verifying loader (D-041); ``--tokenizer DIR`` loads an artifact directory
(bpe_python.json or tokenizer.json) and records its fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from frontier_ai.config import ExperimentConfig  # noqa: E402
from frontier_ai.corpus import verify_and_derive_frontier  # noqa: E402
from frontier_ai.data.dataset import TokenDataset  # noqa: E402
from frontier_ai.engine import checkpoint as ckpt  # noqa: E402
from frontier_ai.evaluation import HARNESS_VERSION  # noqa: E402
from frontier_ai.evaluation.contamination import contamination_report  # noqa: E402
from frontier_ai.evaluation.scoring import PROTOCOL, doc_bits, encode_stream, token_nats  # noqa: E402
from frontier_ai.evaluation.stats import group_summary  # noqa: E402
from frontier_ai.evaluation.suite import (  # noqa: E402
    FRONTIER_HELDOUT_V1,
    SUITES_ROOT,
    SuiteError,
    load_suite,
    script_of,
    verify_docs_against_suite,
)
from frontier_ai.experiments import ExperimentSpec  # noqa: E402
from frontier_ai.experiments.autowire import run_self_recorded  # noqa: E402
from frontier_ai.tokenization.frozen import (  # noqa: E402
    FREEZE_NAME,
    artifact_dir_sha256,
    frontier_tokenizer_v1_dir,
    load_frontier_tokenizer,
)

REPORT_SCHEMA = "frontier-eval-report-v1"
DEFAULT_SUITE = SUITES_ROOT / FRONTIER_HELDOUT_V1 / "SUITE.json"
DOMAIN_NOTE = ("not available: FrontierCorpus v1 carries no domain labels — all sources are "
               "Wikisource/Gutenberg literature; per-source results are reported instead")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint dir (model.pt + config.json + meta.json)")
    p.add_argument("--tokenizer", default="frozen",
                   help="'frozen' (Frontier Tokenizer v1, hash-verified) or an artifact dir")
    p.add_argument("--suite", default=str(DEFAULT_SUITE), help="suite file (default: %(default)s)")
    p.add_argument("--data", default=None,
                   help="dataset .bin for the data-identity check (default: the checkpoint "
                        "config's data.path; 'none' to skip)")
    p.add_argument("--frontier-dir", default="corpora/frontier/v1")
    p.add_argument("--manifest", default="corpora/tokenizer/indic-tokenizer-v2/sources.json")
    p.add_argument("--freeze", default="corpora/tokenizer/indic-tokenizer-v2/FREEZE.json")
    p.add_argument("--corpus-dir", default="data/tokenizer/indic-tokenizer-v2")
    p.add_argument("--no-contamination", action="store_true", help="skip the contamination check")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu", help="default cpu (reproducible fp32)")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (default 1000)")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--label", default=None, help="report label (default: derived from --ckpt)")
    p.add_argument("--exp-id", default="EXP-031")
    p.add_argument("--out", default=None, help="output dir (default out/eval/<exp-id>/<label>)")
    p.add_argument("--notes", default="")
    p.add_argument("--no-record", action="store_true", help="do not write the experiment record")
    return p


def _die(msg: str) -> SystemExit:
    print(f"[eval] INPUT ERROR: {msg}", file=sys.stderr)
    return SystemExit(2)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_label(ckpt_dir: Path) -> str:
    if ckpt_dir.name in ("best", "last") and ckpt_dir.parent.parent != ckpt_dir.parent:
        return f"{ckpt_dir.parent.parent.name}-{ckpt_dir.parent.name}"
    return ckpt_dir.name


def _load_tokenizer(spec: str):
    """(tokenizer, info dict)."""
    frozen_dir = frontier_tokenizer_v1_dir()
    frozen_sha = None
    if (frozen_dir / FREEZE_NAME).is_file():
        freeze = json.loads((frozen_dir / FREEZE_NAME).read_text(encoding="utf-8"))
        frozen_sha = freeze["artifact"]["dir_sha256"]
    if spec == "frozen":
        tok = load_frontier_tokenizer()
        return tok, {"spec": "frozen", "id": "frontier-tokenizer-v1", "dir_sha256": frozen_sha,
                     "vocab_size": tok.vocab_size, "matches_frozen_v1": True}
    art = Path(spec)
    if (art / "bpe_python.json").is_file():
        from frontier_ai.tokenization.bpe_python import PythonBPE

        tok = PythonBPE.load(art)
    elif (art / "tokenizer.json").is_file():
        from frontier_ai.tokenization.bpe_hf import HuggingFaceBPE

        tok = HuggingFaceBPE.load(art)
    else:
        raise _die(f"no tokenizer artifact (bpe_python.json / tokenizer.json) in {art}")
    sha = artifact_dir_sha256(art)
    return tok, {"spec": art.as_posix(), "id": None, "dir_sha256": sha, "vocab_size": tok.vocab_size,
                 "matches_frozen_v1": (sha == frozen_sha) if frozen_sha else None}


def _groups(keys: list[str], bits, bytes_, chars, tokens, n_boot, seed) -> dict:
    out = {}
    for key in sorted(set(keys)):
        m = np.array([k == key for k in keys])
        out[key] = group_summary(bits[m], bytes_[m], chars[m], tokens[m], n_boot, seed)
    return out


def evaluate(args) -> tuple[int, dict]:
    ckpt_dir = Path(args.ckpt)
    for name in ("model.pt", "config.json", "meta.json"):
        if not (ckpt_dir / name).is_file():
            raise _die(f"{ckpt_dir / name} not found")
    cfg = ExperimentConfig.load(ckpt_dir / "config.json")
    meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))

    tok, tok_info = _load_tokenizer(args.tokenizer)
    if tok.vocab_size != cfg.model.vocab_size:
        raise _die(f"tokenizer vocab {tok.vocab_size} != checkpoint vocab {cfg.model.vocab_size} "
                   "(wrong tokenizer for this model)")

    try:
        suite = load_suite(args.suite)
    except SuiteError as exc:
        raise _die(str(exc)) from exc
    train_docs, held_docs, corpus_manifest = verify_and_derive_frontier(
        args.frontier_dir, Path(args.manifest), Path(args.freeze), Path(args.corpus_dir)
    )
    if suite["source"].get("content_sha256") != corpus_manifest["content_sha256"]:
        raise _die("suite was built from a different corpus version than the one on disk")
    try:
        verify_docs_against_suite(held_docs, suite)
    except SuiteError as exc:
        raise _die(f"evaluation documents do not match the protected suite: {exc}") from exc

    device = torch.device(args.device)
    model = ckpt.build_model_from_config(cfg, device)
    ckpt.load_checkpoint(ckpt_dir, model, map_location=str(device))

    started = time.monotonic()
    stream = encode_stream(tok, held_docs)

    data_path = None if args.data == "none" else Path(args.data or cfg.data.path)
    if data_path is None:
        identity = {"status": "NOT CHECKED", "detail": "skipped (--data none)"}
    elif not data_path.is_file():
        identity = {"status": "NOT CHECKED", "path": data_path.as_posix(), "detail": "file not found"}
    else:
        val = np.asarray(TokenDataset(data_path).val, dtype=np.int64)
        same = val.shape == stream.ids.shape and bool(np.array_equal(val, stream.ids))
        identity = {"status": "PASS" if same else "FAIL", "path": data_path.as_posix(),
                    "detail": f"re-encoded stream {len(stream.ids):,} tokens vs dataset val split "
                              f"{len(val):,} tokens — {'identical' if same else 'DIFFERENT'}"}

    nats = token_nats(model, stream.ids, cfg.model.block_size, args.batch_size, device)
    bits = doc_bits(stream, nats)
    score_seconds = round(time.monotonic() - started, 1)

    recs = suite["documents"]
    keep = np.arange(len(recs)) >= 1  # document 0 is context-only (see scoring.PROTOCOL)
    b_bits = bits[keep]
    b_bytes = np.array([r["bytes"] for r in recs], dtype=np.float64)[keep]
    b_chars = np.array([r["chars"] for r in recs], dtype=np.float64)[keep]
    b_tokens = stream.tokens_per_doc.astype(np.float64)[keep]
    langs = [r["language"] for r in recs[1:]]
    scripts = [script_of(x) for x in langs]
    sources = [r["source_id"] for r in recs[1:]]
    nb, bs = args.bootstrap, args.bootstrap_seed

    per_doc_lines = [
        json.dumps({"doc_id": r["doc_id"], "language": r["language"], "source_id": r["source_id"],
                    "bytes": r["bytes"], "chars": r["chars"], "tokens": int(t), "bits": repr(float(b)),
                    "context_only": i == 0}, ensure_ascii=False)
        for i, (r, t, b) in enumerate(zip(recs, stream.tokens_per_doc, bits))
    ]
    scores_sha = hashlib.sha256("\n".join(per_doc_lines).encode("utf-8")).hexdigest()

    overall = group_summary(b_bits, b_bytes, b_chars, b_tokens, nb, bs)
    training = None
    rec_path = ckpt_dir.parent / "experiment.json"
    if rec_path.is_file():
        res = json.loads(rec_path.read_text(encoding="utf-8")).get("results") or {}
        if res.get("best_bpb") is not None:
            training = {"record": rec_path.as_posix(), "best_bpb": res.get("best_bpb"),
                        "best_val": res.get("best_val"),
                        "note": "training-time estimate from eval_iters sampled batches"}

    if args.no_contamination:
        contamination = {"status": "NOT CHECKED", "detail": "skipped (--no-contamination)"}
    else:
        contamination = {"status": "CHECKED",
                         "reference": "FrontierCorpus v1 train side (the training data of models "
                                      "trained through prepare_exp_b_data.py)",
                         **contamination_report(held_docs[1:], train_docs)}

    report = {
        "schema": REPORT_SCHEMA,
        "harness_version": HARNESS_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": args.label,
        "checkpoint": {
            "dir": ckpt_dir.as_posix(),
            "model_sha256": _sha256_file(ckpt_dir / "model.pt"),
            "config_sha256": _sha256_file(ckpt_dir / "config.json"),
            "step": meta.get("step"),
            "n_params": sum(p.numel() for p in model.parameters()),
            "vocab_size": cfg.model.vocab_size,
            "block_size": cfg.model.block_size,
            "training_estimate": training,
        },
        "tokenizer": tok_info,
        "suite": {"id": suite["suite_id"], "path": Path(args.suite).as_posix(),
                  "fingerprint": suite["fingerprint"], "documents": len(recs),
                  "corpus_content_sha256": corpus_manifest["content_sha256"]},
        "data_identity": identity,
        "protocol": {"description": PROTOCOL, "block_size": cfg.model.block_size,
                     "batch_size": args.batch_size, "device": str(device), "precision": "fp32",
                     "torch_threads": torch.get_num_threads(),
                     "bootstrap": {"resamples": nb, "seed": bs, "interval": "95% percentile"}},
        "coverage": {"stream_tokens": int(len(stream.ids)),
                     "scored_tokens": int(np.count_nonzero(~np.isnan(nats))),
                     "context_only_documents": 1,
                     "context_only_doc_id": recs[0]["doc_id"]},
        "results": {
            "overall": overall,
            "per_language": _groups(langs, b_bits, b_bytes, b_chars, b_tokens, nb, bs),
            "per_script": _groups(scripts, b_bits, b_bytes, b_chars, b_tokens, nb, bs),
            "per_source": _groups(sources, b_bits, b_bytes, b_chars, b_tokens, 0, bs),
            "domain": DOMAIN_NOTE,
        },
        "contamination": contamination,
        "reproducibility": {"scores_sha256": scores_sha, "score_seconds": score_seconds},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    (out_dir / "per_document.jsonl").write_text("\n".join(per_doc_lines) + "\n", encoding="utf-8")
    text = render(report)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return (1 if identity["status"] == "FAIL" else 0), report


def _fmt_ci(g: dict) -> str:
    lo, hi = g["bits_per_byte_ci95"]
    if math.isnan(lo):
        return "      n/a       "
    return f"[{lo:.4f}, {hi:.4f}]"


def render(r: dict) -> str:
    ov = r["results"]["overall"]
    tk = r["tokenizer"]
    cov = r["coverage"]
    lines = [
        f"EVALUATION REPORT — harness v{r['harness_version']} — {r['label']}",
        f"checkpoint : {r['checkpoint']['dir']} (step {r['checkpoint']['step']}, "
        f"{r['checkpoint']['n_params']:,} params, model sha256 {r['checkpoint']['model_sha256'][:16]}…)",
        f"tokenizer  : {tk['id'] or tk['spec']} (vocab {tk['vocab_size']}, "
        f"dir sha256 {str(tk['dir_sha256'])[:16]}…, frozen v1: {tk['matches_frozen_v1']})",
        f"suite      : {r['suite']['id']} ({r['suite']['documents']:,} docs, fingerprint "
        f"{r['suite']['fingerprint'][:16]}…)",
        f"data identity: {r['data_identity']['status']} — {r['data_identity'].get('detail', '')}",
        f"coverage   : {cov['scored_tokens']:,} of {cov['stream_tokens']:,} tokens scored "
        f"(every token except the first of the stream; 1 context-only document)",
        "",
        f"OVERALL bits-per-byte {ov['bits_per_byte']:.4f}  95% CI {_fmt_ci(ov)}  | bits-per-char "
        f"{ov['bits_per_char']:.4f} | bits-per-token {ov['bits_per_token']:.4f}",
    ]
    est = r["checkpoint"]["training_estimate"]
    if est:
        lines.append(f"(training-time sampled estimate: best_bpb {est['best_bpb']:.4f})")
    lines += ["", f"{'language':10}{'script':18}{'docs':>6}{'bytes':>10}{'bpb':>9}  "
                  f"{'95% CI':18}{'bits/char':>10}"]
    for lang, g in r["results"]["per_language"].items():
        lines.append(f"{lang:10}{script_of(lang):18}{g['documents']:>6}{g['bytes']:>10,}"
                     f"{g['bits_per_byte']:>9.4f}  {_fmt_ci(g):18}{g['bits_per_char']:>10.4f}")
    lines += ["", f"{'script':18}{'docs':>6}{'bytes':>10}{'bpb':>9}  95% CI"]
    for sc, g in r["results"]["per_script"].items():
        lines.append(f"{sc:18}{g['documents']:>6}{g['bytes']:>10,}{g['bits_per_byte']:>9.4f}  {_fmt_ci(g)}")
    c = r["contamination"]
    lines.append("")
    if c["status"] == "CHECKED":
        frac = c["ngram_overlap_fraction"]
        lines.append(f"contamination vs train side: exact duplicates {c['exact_duplicates']} | "
                     f"docs sharing a {c['ngram_n']}-gram: {c['ngram_overlap_documents']} of "
                     f"{c['ngram_eligible_documents']} eligible"
                     + (f" ({frac:.2%})" if frac is not None else "")
                     + f" | {c['ngram_too_short_documents']} docs too short for {c['ngram_n']}-grams")
    else:
        lines.append(f"contamination: {c['status']} — {c.get('detail', '')}")
    lines.append(f"domain: {r['results']['domain']}")
    lines.append(f"scores sha256: {r['reproducibility']['scores_sha256']}")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    args.label = args.label or _default_label(Path(args.ckpt))
    args.out = args.out or f"out/eval/{args.exp_id}/{args.label}"
    if args.no_record:
        code, _ = evaluate(args)
        return code

    outcome: dict = {}

    def body() -> dict:
        code, report = evaluate(args)
        outcome["code"] = code
        ov = report["results"]["overall"]
        return {
            "bits_per_byte": ov["bits_per_byte"],
            "bits_per_byte_ci95": ov["bits_per_byte_ci95"],
            "bits_per_char": ov["bits_per_char"],
            "per_language_bpb": {k: v["bits_per_byte"] for k, v in report["results"]["per_language"].items()},
            "data_identity": report["data_identity"]["status"],
            "scores_sha256": report["reproducibility"]["scores_sha256"],
            "suite_fingerprint": report["suite"]["fingerprint"],
            "tokenizer_dir_sha256": report["tokenizer"]["dir_sha256"],
            "model_sha256": report["checkpoint"]["model_sha256"],
        }

    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            seed=args.bootstrap_seed,
            name=f"Evaluation report (harness v{HARNESS_VERSION}): {args.label}",
            output_dir=str(args.out),
            params={"ckpt": args.ckpt, "tokenizer": args.tokenizer, "suite": args.suite,
                    "batch_size": args.batch_size, "device": args.device, "bootstrap": args.bootstrap},
            data_paths=[str(Path(args.frontier_dir) / "manifest.json"), args.suite],
            command=list(sys.argv),
            tags=["evaluation", "harness-v1", "frontier-heldout-v1"],
            notes=args.notes,
        )

    recorded = run_self_recorded(build_spec, body)
    if recorded.failed:
        print(f"[eval] record FAILED — {recorded.error}", file=sys.stderr)
        return 1
    return outcome.get("code", 1)


if __name__ == "__main__":
    raise SystemExit(main())
