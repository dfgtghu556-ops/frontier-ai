#!/usr/bin/env python3
"""EXP-045: the pre-registered evaluation of the EXP-043 model (tests 1, 2, 3 and 5) + fp16 copy.

    python scripts/eval_exp045.py --weights out/kaggle/EXP-043/.../model_final.pt \\
        --heldout-jsonl out/eval/heldout-v1.jsonl --contamination evals/results/EXP-045/contamination.json \\
        --out out/eval/EXP-045-eval --export-fp16 out/eval/EXP-045-eval/model_fp16.pt

Tests (fixed in EXPERIMENTS.md before the model existed; protocol strings are in the report):
  1. protected suite ``frontier-heldout-v1``: bits per byte per language (needs ``--heldout-jsonl``,
     written on the founder's PC by ``scripts/export_heldout_text.py``; every document is checked
     against the suite's hashes first). Without the file, test 1 is reported as NOT RUN.
  2. Belebele: accuracy per language with a 95% Wilson interval; with ``--contamination`` also
     without the questions flagged by test 4.
  3. same passages in every language: total bits and bits per byte (and total bits relative to
     English).
  5. samples: the 5 fixed everyday prompts per language (``evals/prompts/everyday-v1.json``),
     greedy and temperature 0.8 (seed 1).

Writes ``summary.json``, ``SUMMARY.txt``, ``samples.jsonl`` and ``belebele_items.jsonl`` (per
question: key, chosen option, correct or not; no Belebele text), and with ``--export-fp16`` the
weights-only fp16 copy (not published; for download).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from frontier_ai.evaluation import belebele as bb  # noqa: E402

SCHEMA = "frontier-exp045-eval-v1"
PROMPTS = ROOT / "evals" / "prompts" / "everyday-v1.json"
SUITE = ROOT / "evals" / "suites" / "frontier-heldout-v1" / "SUITE.json"
SETUP = {"sample_new_tokens": 64, "sample_temperature": 0.8, "sample_seed": 1, "batch_size": 16}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(args, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.weights:
        from frontier_ai.engine.weights import load_weights

        model, info = load_weights(args.weights, device)
        info = {
            **info,
            "source": "weights",
            "file": Path(args.weights).name,
            "sha256": sha256_file(Path(args.weights)),
        }
        return model, info
    from frontier_ai.config import ExperimentConfig
    from frontier_ai.engine import checkpoint as ckpt

    cfg = ExperimentConfig.load(Path(args.ckpt) / "config.json")
    model = ckpt.build_model_from_config(cfg, device)
    ckpt.load_checkpoint(Path(args.ckpt), model, map_location=str(device))
    model.eval()
    return model, {"source": "checkpoint dir", "file": Path(args.ckpt).name}


def ratio(num: float, den: float) -> float | None:
    return (num / den) if den else None


def test_heldout(model, tok, args, block, device, amp) -> dict[str, Any]:
    if not args.heldout_jsonl:
        return {"status": "NOT RUN", "reason": "no --heldout-jsonl (the texts live on the founder's PC)"}
    from frontier_ai.evaluation.suite import load_suite, verify_docs_against_suite

    rows = [
        json.loads(x) for x in Path(args.heldout_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    docs = [
        SimpleNamespace(doc_id=r["doc_id"], language=r["language"], source_id=r["source_id"], text=r["text"])
        for r in rows
    ]
    suite = load_suite(args.suite)
    if not args.no_verify:
        verify_docs_against_suite(docs, suite)  # raises SuiteError: no number without identity
    bits, windowed = bb.document_bits(model, tok, [d.text for d in docs], block, device, args.batch_size, amp)
    per: dict[str, Any] = {}
    for d, b in zip(docs, bits):
        e = per.setdefault(d.language, {"documents": 0, "bits": 0.0, "bytes": 0})
        e["documents"] += 1
        e["bits"] += float(b)
        e["bytes"] += len(d.text.encode("utf-8"))
    for e in per.values():
        e["bpb"] = ratio(e["bits"], e["bytes"])
    tb, ty = sum(e["bits"] for e in per.values()), sum(e["bytes"] for e in per.values())
    return {
        "status": "RUN",
        "suite_id": suite["suite_id"],
        "suite_fingerprint": suite["fingerprint"],
        "identity": "SKIPPED (--no-verify)"
        if args.no_verify
        else "PASS (every document matches the suite hashes)",
        "protocol": bb.DOC_PROTOCOL,
        "documents": len(docs),
        "windowed_documents": windowed,
        "bpb_all": ratio(tb, ty),
        "per_language": dict(sorted(per.items())),
    }


def test_belebele(model, tok, items_by_lang, args, block, device, amp, flagged) -> tuple[dict, list[dict]]:
    per: dict[str, Any] = {}
    item_rows: list[dict] = []
    for lang, items in items_by_lang.items():
        t = time.monotonic()
        res = bb.belebele_language(model, tok, items, block, device, args.batch_size, amp)
        rep = res["report"]
        if flagged is not None:
            bad = set(flagged.get(lang, []))
            clean = [r["correct"] for r in res["rows"] if r["key"] not in bad]
            rep["without_flagged"] = {
                **bb.accuracy_report(clean),
                "flagged_removed": len(res["rows"]) - len(clean),
            }
        rep["seconds"] = round(time.monotonic() - t, 1)
        per[lang] = rep
        for r in res["rows"]:
            item_rows.append(
                {"language": lang, **{k: r[k] for k in ("key", "pred", "answer", "correct", "cut")}}
            )
        print(f"[eval] belebele {lang}: {rep['correct']}/{rep['questions']}", flush=True)
    return {
        "protocol": bb.PROTOCOL,
        "chance": bb.CHANCE,
        "contamination": "applied" if flagged is not None else "NOT AVAILABLE (no --contamination)",
        "per_language": per,
    }, item_rows


def test_parallel(model, tok, items_by_lang, block, device, args, amp) -> dict[str, Any]:
    texts = {lang: bb.passages(items) for lang, items in items_by_lang.items()}
    common = sorted(set.intersection(*(set(t) for t in texts.values()))) if texts else []
    per: dict[str, Any] = {}
    for lang, t in texts.items():
        chosen = [t[k] for k in common]
        bits, windowed = bb.document_bits(model, tok, chosen, block, device, args.batch_size, amp)
        nbytes = sum(len(x.encode("utf-8")) for x in chosen)
        per[lang] = {
            "bits": float(bits.sum()),
            "bytes": nbytes,
            "bpb": ratio(float(bits.sum()), nbytes),
            "windowed": windowed,
        }
    if "en" in per:
        for e in per.values():
            e["bits_vs_en"] = ratio(e["bits"], per["en"]["bits"])
    return {
        "protocol": bb.DOC_PROTOCOL,
        "passages": len(common),
        "passages_per_language": {k: len(v) for k, v in texts.items()},
        "per_language": per,
    }


def samples(model, tok, langs, device) -> list[dict[str, Any]]:
    spec = json.loads(PROMPTS.read_text(encoding="utf-8"))
    eot = tok.special_token_ids["<|endoftext|>"]
    rows = []
    for lang in langs:
        for i, prompt in enumerate(spec["prompts"][lang]):
            ids = [eot, *tok.encode_ordinary(prompt)]
            for mode in ("greedy", f"temperature {SETUP['sample_temperature']}"):
                g = torch.Generator(device=device).manual_seed(SETUP["sample_seed"])
                out = model.generate(
                    torch.tensor([ids], device=device),
                    SETUP["sample_new_tokens"],
                    temperature=1.0 if mode == "greedy" else SETUP["sample_temperature"],
                    top_k=1 if mode == "greedy" else None,
                    generator=g,
                    stop_at_eos=eot,
                )
                cont = [t for t in out[0, len(ids) :].tolist() if t != eot]
                rows.append(
                    {
                        "language": lang,
                        "prompt_set": spec["prompt_set"],
                        "prompt_index": i,
                        "mode": mode,
                        "prompt": prompt,
                        "continuation": tok.decode(cont),
                        "new_tokens": len(cont),
                    }
                )
    return rows


def render(s: dict) -> str:
    out = [
        f"EXP-045 evaluation of {s['model'].get('file')} (step {s['model'].get('step')})",
        f"code commit {s['environment'].get('code_commit')} | device {s['device']} | amp {s['amp']}",
    ]
    if s.get("limit"):
        out.append(f"LIMITED RUN: {s['limit']} questions per language - not a result")
    h = s["tests"]["1_heldout"]
    out += ["", "1. Protected suite frontier-heldout-v1 (bits per byte, lower is better)"]
    if h["status"] != "RUN":
        out.append(f"   NOT RUN: {h['reason']}")
    else:
        out.append(f"   identity: {h['identity']}; all languages: {h['bpb_all']:.4f}")
        out.append("   " + "  ".join(f"{k} {v['bpb']:.3f}" for k, v in h["per_language"].items()))
    b = s["tests"]["2_belebele"]
    out += ["", "2. Belebele reading comprehension (chance 25%; 'above' = whole 95% interval above 25%)"]
    for lang, r in b["per_language"].items():
        lo, hi = r["wilson95"]
        line = f"   {lang:3} {100 * r['accuracy']:5.1f}%  [{100 * lo:4.1f}, {100 * hi:4.1f}]"
        line += "  above chance" if r["above_chance"] else "  not above chance"
        if "without_flagged" in r:
            w = r["without_flagged"]
            line += (
                f" | without {w['flagged_removed']} flagged: {100 * w['accuracy']:5.1f}%"
                if w["questions"]
                else ""
            )
        if r["passages_cut"]:
            line += f" | {r['passages_cut']} passages cut"
        out.append(line)
    out.append(f"   contamination: {b['contamination']}")
    p = s["tests"]["3_parallel"]
    out += ["", f"3. The same {p['passages']} passages in every language (total bits; relative to English)"]
    for lang, r in p["per_language"].items():
        rel = f"{r['bits_vs_en']:.2f}x" if r.get("bits_vs_en") else "-"
        out.append(f"   {lang:3} {r['bits']:12,.0f} bits  {rel:>6}  ({r['bpb']:.3f} bits/byte)")
    out += ["", f"5. Samples: {s['tests']['5_samples']['rows']} continuations in samples.jsonl (for reading)"]
    if s.get("fp16_copy"):
        f = s["fp16_copy"]
        out.append(f"fp16 copy: {f['file']} ({f['bytes'] / 1e9:.2f} GB) sha256 {f['sha256']}")
    if s["smoke"]:
        out.append("SMOKE RUN: tiny model / synthetic data, not a result.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", help="weights-only file (EXP-043 model_final.pt)")
    src.add_argument("--ckpt", help="checkpoint directory (model.pt + config.json; tests)")
    p.add_argument("--out", required=True)
    p.add_argument("--belebele-dir", default="/tmp/belebele", help="download folder (never committed)")
    p.add_argument("--heldout-jsonl", default=None)
    p.add_argument("--suite", default=str(SUITE))
    p.add_argument("--contamination", default=None, help="contamination.json from test 4")
    p.add_argument("--export-fp16", default=None, help="write the fp16 weights copy here")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", default="auto", choices=["auto", "fp16", "none"], help="auto = fp16 on CUDA")
    p.add_argument("--batch-size", type=int, default=SETUP["batch_size"])
    p.add_argument("--languages", default=",".join(bb.LANGUAGES))
    p.add_argument("--limit", type=int, default=0, help="questions per language (tests only)")
    p.add_argument("--no-verify", action="store_true", help="skip hash checks (tests only)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    import gpu_bringup as gb
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    started = time.monotonic()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    amp = torch.float16 if (args.amp == "fp16" or (args.amp == "auto" and device.type == "cuda")) else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    langs = [x for x in args.languages.split(",") if x]

    model, model_info = load_model(args, device)
    tok = load_frontier_tokenizer_v2()
    if tok.vocab_size != model.cfg.vocab_size:
        raise SystemExit(f"tokenizer vocabulary {tok.vocab_size} != model vocabulary {model.cfg.vocab_size}")
    block = model.cfg.block_size

    items_by_lang = {}
    for lang in langs:
        if not args.no_verify:
            bb.download(lang, args.belebele_dir)
        items = bb.load_language(
            args.belebele_dir,
            lang,
            verify=not args.no_verify,
            expected=None if args.no_verify else bb.QUESTIONS_PER_LANGUAGE,
        )
        items_by_lang[lang] = items[: args.limit] if args.limit else items
    flagged = None
    if args.contamination:
        c = json.loads(Path(args.contamination).read_text(encoding="utf-8"))
        if c.get("belebele_revision") != bb.REVISION:
            raise SystemExit("contamination.json was made for another Belebele revision")
        flagged = c["flagged"]

    tests: dict[str, Any] = {}
    t = time.monotonic()
    tests["1_heldout"] = test_heldout(model, tok, args, block, device, amp)
    tests["1_heldout"]["seconds"] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    tests["2_belebele"], item_rows = test_belebele(
        model, tok, items_by_lang, args, block, device, amp, flagged
    )
    tests["2_belebele"]["seconds"] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    tests["3_parallel"] = test_parallel(model, tok, items_by_lang, block, device, args, amp)
    tests["3_parallel"]["seconds"] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    rows = samples(model, tok, langs, device)
    tests["5_samples"] = {
        "prompt_set": "everyday-v1",
        "rows": len(rows),
        **SETUP,
        "seconds": round(time.monotonic() - t, 1),
    }

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "exp_id": "EXP-045",
        "smoke": bool(args.smoke),
        "complete": True,
        "limit": args.limit or None,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": gb.environment_record(),
        "device": str(device),
        "amp": "fp16" if amp is not None else "none (float32)",
        "model": {k: v for k, v in model_info.items() if k in ("source", "file", "sha256", "step", "exp_id")}
        | {"n_params": sum(q.numel() for q in model.parameters()), "block_size": block},
        "belebele": {"repo": bb.REPO, "revision": bb.REVISION, "license": bb.LICENSE},
        "tests": tests,
        "numpy": np.__version__,
    }
    if args.export_fp16:
        from frontier_ai.engine.weights import export_fp16

        summary["fp16_copy"] = export_fp16(
            model,
            args.export_fp16,
            exp_id=model_info.get("exp_id"),
            step=model_info.get("step"),
            source_sha256=model_info.get("sha256"),
        )
    summary["seconds"] = round(time.monotonic() - started, 1)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "SUMMARY.txt").write_text(render(summary), encoding="utf-8")
    (out / "samples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    (out / "belebele_items.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in item_rows), encoding="utf-8", newline="\n"
    )
    print(render(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
