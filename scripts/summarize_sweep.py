#!/usr/bin/env python3
"""Verify and summarize a finished EXP-A tokenizer sweep (read-only over the run).

Consumes the sweep framework's records (``sweep.json`` + one ``experiment.json``
per configuration), re-checks the internal consistency of every run, and prints
the full comparison tables — including the per-language held-out table and the
top-2 candidates for EXP-B. This script *reports*; it never selects a tokenizer
and never modifies the run directory (it only writes ``summary.txt`` beside
``sweep.json``).

Consistency checks (any failure -> exit 1):
* every run in ``sweep.json`` has status ``success`` and its record exists;
* the recorded headline metric equals the run's own ``heldout.overall.chars_per_token``;
* the losslessness gate passed on BOTH sides (0 failures, checked == documents);
* the per-language document counts of each run sum to the held-out total;
* the per-language metrics are present for every language in the run's held-out docs.

Multiple sweep dirs can be merged (e.g. the main 15-configuration run plus a
supplemental run of the 5 bpe_hf + mark_aware cells in its own directory):
configuration names are unioned, and for a name present in several dirs the
LATER dir wins (its record is the one summarized).

Examples
--------
    python scripts/summarize_sweep.py --sweep-dir out/experiments/EXP-028
    python scripts/summarize_sweep.py --sweep-dir out/experiments/EXP-028 --no-write
    python scripts/summarize_sweep.py --sweep-dir out/experiments/EXP-028 \\
        out/experiments/EXP-028-hf-mark-aware
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_INCONSISTENT = 1
EXIT_MISSING = 2


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, width: int = 10, decimals: int = 4) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep-dir", nargs="+", default=["out/experiments/EXP-028"],
                        metavar="DIR",
                        help="dir(s) containing sweep.json, merged in the given order "
                             "(default: %(default)s)")
    parser.add_argument("--no-write", action="store_true",
                        help="do not write summary.txt (print only)")
    args = parser.parse_args()

    sweep_dirs = [Path(d) for d in args.sweep_dir]
    problems: list[str] = []
    sweeps: list[tuple[Path, dict]] = []
    for sweep_dir in sweep_dirs:
        sweep_path = sweep_dir / "sweep.json"
        if not sweep_path.is_file():
            print(f"[summarize] no sweep.json at {sweep_path} — did the sweep finish? "
                  "exit 2 means the input was not produced.", file=sys.stderr)
            return EXIT_MISSING
        sweeps.append((sweep_dir, _load(sweep_path)))

    # collect runs across all dirs; a configuration present in several dirs is
    # taken from the LATER dir (that run supersedes the earlier one)
    runs_by_name: dict[str, dict] = {}
    run_order: list[str] = []
    for sweep_dir, sweep in sweeps:
        for run in sweep.get("runs", []):
            name = run.get("configuration", "?")
            record_rel = run.get("record")
            if not record_rel:
                problems.append(f"{sweep_dir}: run {name}: no record path")
                continue
            record_path = sweep_dir / record_rel
            if not record_path.is_file():
                problems.append(
                    f"{sweep_dir}: run {name}: record file missing ({record_path})")
                continue
            results = _load(record_path).get("results") or {}
            if name not in runs_by_name:
                run_order.append(name)
            if name in runs_by_name and run.get("status") == "success":
                problems.append(
                    f"{name}: present in more than one sweep dir — using "
                    f"{sweep_dir} (later dir wins)")
            runs_by_name[name] = {"entry": run, "results": results, "dir": sweep_dir}
    runs = [runs_by_name[name] for name in run_order]

    for sweep_dir, sweep in sweeps:
        section = sweep.get("sweep", {})
        if section.get("status") != "success":
            problems.append(f"{sweep_dir}: sweep status is {section.get('status')!r} "
                            "(not 'success')")

    # ---- per-run consistency checks -----------------------------------------
    for item in runs:
        run, r = item["entry"], item["results"]
        name = run.get("configuration", "?")
        if run.get("status") != "success":
            problems.append(f"{name}: run status {run.get('status')!r} "
                            f"(error: {run.get('error')})")
            continue
        metric = run.get("metric_value")
        recorded = (r.get("heldout") or {}).get("overall", {}).get("chars_per_token")
        if recorded is None or metric is None or abs(float(metric) - float(recorded)) > 1e-9:
            problems.append(f"{name}: recorded metric {metric!r} != results "
                            f"heldout.overall.chars_per_token {recorded!r}")
        gate = r.get("gate") or {}
        train_gate = gate.get("train") or {}
        held_gate = gate.get("heldout") or {}
        n_train = r.get("train", {}).get("documents")
        n_held = r.get("heldout", {}).get("documents")
        if not gate.get("lossless", False):
            problems.append(f"{name}: losslessness gate FAILED")
        if train_gate.get("checked") != n_train or train_gate.get("failures", 0) != 0:
            problems.append(f"{name}: train gate mismatch (checked={train_gate.get('checked')}, "
                            f"failures={train_gate.get('failures')}, docs={n_train})")
        if held_gate.get("checked") != n_held or held_gate.get("failures", 0) != 0:
            problems.append(f"{name}: heldout gate mismatch (checked={held_gate.get('checked')}, "
                            f"failures={held_gate.get('failures')}, docs={n_held})")
        overall = (r.get("heldout") or {}).get("overall") or {}
        if overall.get("round_trip_failures", 0) != 0:
            problems.append(f"{name}: overall round_trip_failures = {overall.get('round_trip_failures')}")
        per_lang = (r.get("heldout") or {}).get("per_language") or {}
        lang_docs = sum(b.get("examples", 0) for b in per_lang.values())
        if lang_docs != n_held:
            problems.append(f"{name}: per-language examples sum {lang_docs} != heldout docs {n_held}")
        if not per_lang:
            problems.append(f"{name}: no per-language metrics recorded")

    # ---- tables ---------------------------------------------------------------
    lines = [
        "EXP-A sweep — verified summary",
        "sweep: " + " + ".join(
            f"{sweep_dir} (status {s.get('sweep', {}).get('status')})" for sweep_dir, s in sweeps
        ),
        "",
        "configurations (ranked by held-out chars_per_token, higher = denser):",
        f"{'rank':<5} {'configuration':<24} {'pretoken':<13} {'vocab':>6} "
        f"{'gate':<6} {'chars/tok':>10} {'tok/char':>9} {'tok/word':>9} "
        f"{'train s':>9} {'train docs':>11}",
    ]
    ranked = []
    for item in runs:
        run, r = item["entry"], item["results"]
        cfg = r.get("config") or {}
        overall = (r.get("heldout") or {}).get("overall") or {}
        train = r.get("train") or {}
        gate_ok = bool((r.get("gate") or {}).get("lossless"))
        ranked.append({
            "name": run.get("configuration", "?"),
            "pretoken": cfg.get("pretoken", ""),
            "vocab": cfg.get("vocab_size", 0),
            "gate": "PASS" if gate_ok else "FAIL",
            "chars_per_token": overall.get("chars_per_token"),
            "tokens_per_char": overall.get("tokens_per_char"),
            "tokens_per_word": overall.get("tokens_per_word"),
            "train_seconds": train.get("seconds"),
            "train_docs": train.get("documents"),
            "train_file_truncated": bool(train.get("file_truncated")),
            "results": r,
        })
    ranked.sort(key=lambda row: (-(row["chars_per_token"] or 0), row["name"]))
    for i, row in enumerate(ranked, start=1):
        lines.append(
            f"{i:<5} {row['name']:<24} {row['pretoken']:<13} {row['vocab']:>6} "
            f"{row['gate']:<6} {_fmt(row['chars_per_token'])} {_fmt(row['tokens_per_char'])} "
            f"{_fmt(row['tokens_per_word'])} {_fmt(row['train_seconds'], 9, 1)} "
            f"{row['train_docs']:>11}"
            + ("  [SMOKE: train truncated]" if row["train_file_truncated"] else "")
        )

    # per-language held-out density (docs/chars are identical across configs)
    langs: list[str] = []
    lang_docs: dict[str, dict] = {}
    for item in runs:
        per_lang = ((item["results"].get("heldout") or {}).get("per_language")) or {}
        if per_lang:
            langs = sorted(per_lang)
            lang_docs = {lang: per_lang[lang] for lang in langs}
            break
    if langs:
        lines += ["", "held-out documents / chars (same for every configuration):"]
        lines.append("  " + "  ".join(f"{lang}: {lang_docs[lang].get('examples', 0)} docs / "
                                      f"{lang_docs[lang].get('chars', 0):,} chars" for lang in langs))
        lines += ["", "held-out chars_per_token by configuration x language (higher = denser):"]
        lines.append(f"{'configuration':<24}" + "".join(f"{lang:>10}" for lang in langs))
        for row in ranked:
            per_lang = ((row["results"].get("heldout") or {}).get("per_language")) or {}
            lines.append(
                f"{row['name']:<24}"
                + "".join(_fmt(per_lang.get(lang, {}).get("chars_per_token"), 10) for lang in langs)
            )
        # per-language winner (context for EXP-B; not a selection)
        winner_lines = []
        for lang in langs:
            best = max(
                (row for row in ranked
                 if ((row["results"].get("heldout") or {}).get("per_language") or {})
                 .get(lang, {}).get("chars_per_token") is not None),
                key=lambda row: (row["results"]["heldout"]["per_language"][lang]
                                 ["chars_per_token"]),
                default=None,
            )
            if best is not None:
                value = best["results"]["heldout"]["per_language"][lang]["chars_per_token"]
                winner_lines.append(f"{lang} -> {best['name']} ({value:.4f})")
        lines += ["", "per-language best configuration (context only — NOT a selection):"]
        lines += [f"  {w}" for w in winner_lines]

    if ranked:
        top2 = ranked[:2]
        lines += [
            "",
            "TOP-2 CANDIDATES FOR EXP-B (by overall held-out chars_per_token; EXP-B trains these",
            "with a small language model, >= 3 seeds, and compares bits-per-character; the",
            "selection is made AFTER EXP-B, per the approved scope):",
        ]
        for i, row in enumerate(top2, start=1):
            lines.append(f"  {i}. {row['name']}  (chars/token {row['chars_per_token']:.4f})")

    if problems:
        lines += ["", "CONSISTENCY PROBLEMS (exit 1):"]
        lines += [f"  - {p}" for p in problems]

    summary = "\n".join(lines) + "\n"
    print(summary)
    if not args.no_write:
        out = sweep_dirs[0] / "summary.txt"
        out.write_text(summary, encoding="utf-8", newline="\n")
        print(f"[summarize] wrote {out}")
    if problems:
        return EXIT_INCONSISTENT
    if not ranked:
        print("[summarize] no usable runs found", file=sys.stderr)
        return EXIT_MISSING
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
