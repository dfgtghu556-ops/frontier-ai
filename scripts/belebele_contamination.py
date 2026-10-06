#!/usr/bin/env python3
"""EXP-045 test 4: are Belebele texts inside the EXP-037 training tokens? (Kaggle CPU, no GPU)

    python scripts/belebele_contamination.py --data-dir DIR --out out/eval/EXP-045

1. Checks the 13 EXP-037 token files against their manifest sha256 (skipped with --no-verify).
2. Downloads the 13 pinned Belebele files (``--belebele-dir``; verified by git blob hash and size;
   never committed) and tokenizes every passage, question and answer option with Frontier
   Tokenizer v2.
3. Streams all training tokens (the ``train`` part of every file, about 2.78 B tokens) and finds
   every 13-token run that also occurs in a Belebele text (hash hits confirmed token by token).
4. Writes ``summary.json`` + ``SUMMARY.txt`` (published) and ``contamination.json`` (the flagged
   question keys, read by ``scripts/eval_exp045.py`` to report Belebele with and without them).

A question is *flagged* if its passage, its question or any of its options shares at least one
13-token run with the training data. Nothing is removed from the training data: the overlap is
measured and reported (EXP-045). Only keys, counts and offsets are written; no Belebele text.

``--coverage`` (EXP-045 coverage follow-up, approved 2026-10-06): also measures *how much* of each
text is in the training data, i.e. the share of its distinct 13-token runs that occur there. A text
is **substantially present** when that share is at least ``COVERAGE_THRESHOLD`` (70%, PaLM's rule
for a contaminated example, applied to our 13-token runs), and a question is substantially present
when its passage, its question or any option is. The rule was fixed on 2026-10-06, before any model
accuracy existed. Belebele accuracy is then also reported without these questions. In this mode
the run writes ``coverage.json`` instead of ``contamination.json`` and checks that its flags
reproduce test 4's committed ``contamination.json`` exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from frontier_ai.evaluation import belebele as bb  # noqa: E402
from frontier_ai.evaluation import token_ngrams as tn  # noqa: E402

SCHEMA = "frontier-exp045-contamination-v1"
COVERAGE_THRESHOLD = 0.70  # fixed 2026-10-06 before any model accuracy existed (PaLM's 70% rule)
COVERAGE_BINS = 10  # histogram of passage coverage in steps of 10%
TEST4 = ROOT / "evals" / "results" / "EXP-045" / "contamination.json"
MANIFEST = ROOT / "evals" / "results" / "EXP-037" / "manifest.json"


def build_index(tok: Any, items_by_lang: dict[str, list[bb.Item]]) -> tn.NgramIndex:
    index = tn.NgramIndex(tn.N)
    for lang, items in items_by_lang.items():
        for key, text in bb.passages(items).items():
            index.add(f"{lang}|passage|{key}", tok.encode_ordinary(text))
        for it in items:
            index.add(f"{lang}|question|{it.key}", tok.encode_ordinary(it.question))
            for j, option in enumerate(it.options, 1):
                index.add(f"{lang}|option{j}|{it.key}", tok.encode_ordinary(option))
    index.freeze()
    return index


def flag_items(items_by_lang: dict[str, list[bb.Item]], hit_refs: set[str], too_short: set[str]) -> dict:
    per_lang: dict[str, Any] = {}
    flagged: dict[str, list[str]] = {}
    for lang, items in items_by_lang.items():
        pkey = {k: k for k in bb.passages(items)}
        passage_of = {q: k for k in pkey for q in k.split("|")}
        by = {"passage": 0, "question": 0, "option": 0}
        keys = []
        short = {"passage": 0, "question": 0, "option": 0}
        for it in items:
            parts = {
                "passage": f"{lang}|passage|{passage_of[it.key]}",
                "question": f"{lang}|question|{it.key}",
            }
            opts = [f"{lang}|option{j}|{it.key}" for j in range(1, 5)]
            hit = False
            for name, ref in parts.items():
                if ref in hit_refs:
                    by[name] += 1
                    hit = True
                short[name] += ref in too_short
            if any(r in hit_refs for r in opts):
                by["option"] += 1
                hit = True
            short["option"] += sum(r in too_short for r in opts)
            if hit:
                keys.append(it.key)
        flagged[lang] = keys
        per_lang[lang] = {
            "questions": len(items),
            "flagged": len(keys),
            "flagged_by": by,
            "too_short_to_check": short,
        }
    return {"per_language": per_lang, "flagged": flagged}


def render(s: dict) -> str:
    lines = [
        f"EXP-045 test 4 - Belebele vs EXP-037 training tokens ({s['ngram_n']}-token runs)",
        f"code commit {s['environment'].get('code_commit')} | Belebele {bb.REPO}@{bb.REVISION[:8]}",
        f"training tokens scanned: {s['train_tokens']:,} in {len(s['files'])} files "
        f"({s['seconds']['scan']:.0f} s); Belebele 13-grams: {s['index_keys']:,}",
        "",
        "lang  questions  flagged  (passage / question / option)",
    ]
    for lang, r in s["per_language"].items():
        b = r["flagged_by"]
        parts = f"({b['passage']} / {b['question']} / {b['option']})"
        lines.append(f"{lang:4}  {r['questions']:9}  {r['flagged']:7}  {parts}")
    t = s["totals"]
    lines += [
        "",
        f"TOTAL: {t['flagged']} of {t['questions']} questions flagged "
        f"({100 * t['flagged'] / max(t['questions'], 1):.1f}%)",
        "Flagged questions are listed in contamination.json; Belebele accuracy is reported with and",
        "without them. Texts shorter than 13 tokens cannot be checked (counts in summary.json).",
    ]
    c = s.get("coverage")
    if c:
        thr = int(round(100 * c["threshold"]))
        lines += [
            "",
            "COVERAGE: share of each text's distinct 13-token runs found in the training tokens;",
            f"'substantially present' = at least {thr}% (PaLM's rule), fixed before any model accuracy.",
            f"reproduces test 4's flags exactly: {c.get('reproduces_test4')}",
            "",
            "lang  flagged  substantially  (passage / question / option)  remaining   passage coverage:",
            "                 present                                              none | 0-10% .. 90-100%",
        ]
        for lang, r in c["per_language"].items():
            b = r["substantially_present_by"]
            parts = f"({b['passage']} / {b['question']} / {b['option']})"
            hist = " ".join(f"{x}" for x in r["passage_coverage_hist"])
            flagged = s["per_language"][lang]["flagged"]
            lines.append(
                f"{lang:4}  {flagged:7}  {r['substantially_present']:13}  {parts:<29}  "
                f"{r['not_substantially_present']:9}   {r['passages_no_overlap']} | {hist}"
            )
        t = c["totals"]
        lines.append(
            f"TOTAL: {t['substantially_present']} of {t['questions']} questions substantially present "
            f"({100 * t['substantially_present'] / max(t['questions'], 1):.1f}%); listed in coverage.json"
        )
    if s["smoke"]:
        lines.append("SMOKE RUN: synthetic data, not a result.")
    return "\n".join(lines) + "\n"


def _parts(lang: str, items: list[bb.Item]) -> list[tuple[bb.Item, dict[str, list[str]]]]:
    passage_of = {q: k for k in bb.passages(items) for q in k.split("|")}
    return [
        (
            it,
            {
                "passage": [f"{lang}|passage|{passage_of[it.key]}"],
                "question": [f"{lang}|question|{it.key}"],
                "option": [f"{lang}|option{j}|{it.key}" for j in range(1, 5)],
            },
        )
        for it in items
    ]


def substantial_items(
    items_by_lang: dict[str, list[bb.Item]], cov: dict[str, tuple[int, int]], threshold: float
) -> dict[str, Any]:
    """Questions with a part whose coverage is at least ``threshold``, plus passage coverage histograms."""
    per_lang: dict[str, Any] = {}
    keys_by_lang: dict[str, list[str]] = {}

    def strong(ref: str) -> bool:
        found, total = cov.get(ref, (0, 0))
        return total > 0 and found / total >= threshold

    for lang, items in items_by_lang.items():
        by = {"passage": 0, "question": 0, "option": 0}
        keys = []
        for it, parts in _parts(lang, items):
            hit = False
            for name, refs in parts.items():
                if any(strong(r) for r in refs):
                    by[name] += 1
                    hit = True
            if hit:
                keys.append(it.key)
        hist = [0] * COVERAGE_BINS
        none = 0
        for key in bb.passages(items):
            found, total = cov.get(f"{lang}|passage|{key}", (0, 0))
            if total == 0:
                continue
            if found == 0:
                none += 1
            else:
                hist[min(int(COVERAGE_BINS * found / total), COVERAGE_BINS - 1)] += 1
        keys_by_lang[lang] = keys
        per_lang[lang] = {
            "questions": len(items),
            "substantially_present": len(keys),
            "not_substantially_present": len(items) - len(keys),
            "substantially_present_by": by,
            "passages_no_overlap": none,
            "passage_coverage_hist": hist,
        }
    return {"per_language": per_lang, "substantially_present": keys_by_lang}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with the 13 EXP-037 .bin files")
    p.add_argument("--out", required=True)
    p.add_argument("--belebele-dir", default="/tmp/belebele", help="download folder (never committed)")
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--chunk", type=int, default=50_000_000, help="tokens per scan chunk")
    p.add_argument(
        "--no-verify", action="store_true", help="skip the sha256 and Belebele hash checks (tests)"
    )
    p.add_argument("--smoke", action="store_true", help="mark the output as a smoke run (tests)")
    p.add_argument("--languages", default=",".join(bb.LANGUAGES))
    p.add_argument("--coverage", action="store_true", help="also measure coverage (EXP-045 follow-up)")
    p.add_argument("--test4", default=str(TEST4), help="test 4's contamination.json, to check reproduction")
    args = p.parse_args(argv)

    import numpy as np

    import gpu_bringup as gb
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    langs = [x for x in args.languages.split(",") if x]
    manifest = {
        f["language"]: f for f in json.loads(Path(args.manifest).read_text(encoding="utf-8"))["files"]
    }
    data_dir = Path(args.data_dir)
    files = []
    for lang in langs:
        f = manifest[lang]
        path = data_dir / f["path"]
        if not args.no_verify:
            got = gb.sha256_file(path)
            if got != f["sha256"]:
                raise SystemExit(f"{path}: sha256 {got} != manifest {f['sha256']}")
        n_total = path.stat().st_size // 2
        if n_total != f["n_train"] + f["n_val"]:
            raise SystemExit(f"{path}: {n_total} tokens != n_train + n_val in the manifest")
        files.append((lang, path, int(f["n_train"])))
    t_verify = time.monotonic() - t0

    tok = load_frontier_tokenizer_v2()
    items_by_lang = {}
    for lang in langs:
        if not args.no_verify:
            bb.download(lang, args.belebele_dir)
        items_by_lang[lang] = bb.load_language(
            args.belebele_dir, lang, verify=not args.no_verify, expected=None if args.no_verify else 900
        )
    index = build_index(tok, items_by_lang)
    t_index = time.monotonic() - t0 - t_verify
    matched = np.zeros(len(index.keys), dtype=bool) if args.coverage else None

    hits: dict[str, dict[str, Any]] = {}
    file_rows = []
    train_total = 0
    for lang, path, n_train in files:
        ts = time.monotonic()
        found = tn.scan(index, tn.train_tokens(path, n_train), args.chunk, matched)
        for ref, e in found.items():
            h = hits.setdefault(ref, {"windows": 0, "files": {}})
            h["windows"] += e["windows"]
            h["files"][lang] = {"windows": e["windows"], "first_offset": e["first_offset"]}
        train_total += n_train
        file_rows.append(
            {
                "language": lang,
                "file": path.name,
                "train_tokens": n_train,
                "refs_hit": len(found),
                "seconds": round(time.monotonic() - ts, 1),
            }
        )
        print(f"[contam] {lang}: {n_train:,} tokens, {len(found)} Belebele texts hit", flush=True)
    t_scan = time.monotonic() - t0 - t_verify - t_index

    flags = flag_items(items_by_lang, set(hits), set(index.too_short))
    totals = {
        "questions": sum(r["questions"] for r in flags["per_language"].values()),
        "flagged": sum(r["flagged"] for r in flags["per_language"].values()),
    }
    summary = {
        "schema": SCHEMA,
        "exp_id": "EXP-045",
        "test": "4 (contamination)",
        "smoke": bool(args.smoke),
        "complete": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": gb.environment_record(),
        "belebele": {
            "repo": bb.REPO,
            "revision": bb.REVISION,
            "license": bb.LICENSE,
            "files": {lang: bb.FILES[lang][0] for lang in langs},
        },
        "ngram_n": tn.N,
        "method": tn.__doc__.strip().splitlines()[0],
        "index_keys": len(index),
        "index_texts": index.texts,
        "too_short_texts": len(index.too_short),
        "train_tokens": train_total,
        "files": file_rows,
        "per_language": flags["per_language"],
        "totals": totals,
        "seconds": {"verify": round(t_verify, 1), "index": round(t_index, 1), "scan": round(t_scan, 1)},
        "numpy": np.__version__,
    }
    contamination = {
        "schema": SCHEMA + "-items",
        "belebele_revision": bb.REVISION,
        "flagged": flags["flagged"],
        "hits": {ref: hits[ref] for ref in sorted(hits)},
    }
    if matched is not None:
        cov = tn.coverage(index, matched)
        sub = substantial_items(items_by_lang, cov, COVERAGE_THRESHOLD)
        test4 = Path(args.test4)
        repro: dict[str, Any] = {
            "test4_file": str(test4.relative_to(ROOT)) if test4.is_relative_to(ROOT) else str(test4)
        }
        if test4.exists():
            old = json.loads(test4.read_text(encoding="utf-8"))
            same = old.get("belebele_revision") == bb.REVISION and all(
                sorted(old["flagged"].get(lang, [])) == sorted(flags["flagged"][lang]) for lang in langs
            )
            repro["reproduces_test4"] = bool(same)
        else:
            repro["reproduces_test4"] = None
        summary["coverage"] = {
            "rule": (
                "coverage = distinct 13-token runs of a text found in the training tokens / all of its "
                "distinct 13-token runs; a text is substantially present at coverage >= threshold; a "
                "question is when its passage, question or any option is"
            ),
            "threshold": COVERAGE_THRESHOLD,
            "threshold_source": (
                "PaLM (Chowdhery et al.): an example is contaminated if >= 70% of its n-grams occur in the "
                "training data; fixed 2026-10-06 before any model accuracy existed"
            ),
            "per_language": sub["per_language"],
            "totals": {
                "questions": totals["questions"],
                "substantially_present": sum(
                    r["substantially_present"] for r in sub["per_language"].values()
                ),
            },
            **repro,
        }
        coverage_file = {
            "schema": SCHEMA + "-coverage",
            "belebele_revision": bb.REVISION,
            "threshold": COVERAGE_THRESHOLD,
            "substantially_present": sub["substantially_present"],
            "texts": {ref: list(v) for ref, v in sorted(cov.items()) if v[0] > 0},
        }
        (out / "coverage.json").write_text(json.dumps(coverage_file, indent=1) + "\n", encoding="utf-8")
    else:
        (out / "contamination.json").write_text(json.dumps(contamination, indent=1) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "SUMMARY.txt").write_text(render(summary), encoding="utf-8")
    print(render(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
