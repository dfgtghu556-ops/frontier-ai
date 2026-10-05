#!/usr/bin/env python
"""EXP-043 (step 12): train the D-049 model (14 x 896, about 190 M parameters) over one pass of the
2.78 B-token EXP-037 corpus, one Kaggle session at a time.

Every launch is ONE session; the committed ``<prev-dir>/session-<n>/summary.json`` files say where
the run stands, so the same command is typed once per session:

1. **Learning-rate check** (``--part lrcheck``): 6,104-step runs (the first 100 M tokens of the
   one-pass order) at 2.5e-4, 5e-4 and 1e-3, plus at most one extension run at the edge
   (1.25e-4 or 2e-3); the pre-registered rule picks the learning rate.
2. **Main run** (``--part main``): 169,911 steps (every 512-token window once, in a fixed
   shuffled order; :class:`frontier_ai.data.multi.OnePassDataset`). Each session resumes from
   the checkpoint the previous session left in its Kaggle output (mounted through
   ``kernel_sources``), but only if its sha256 values equal those in the last committed summary;
   it trains until its time budget, saves a checkpoint and records its sha256.
3. **End** (the session that finishes the pass): full validation per language, the three
   pre-registered checks, ``samples.jsonl`` and the weights-only ``model_final.pt``.

``--part auto`` (default, what the Kaggle kernel uses) does whichever of the two is next and starts
the main run in the same session if the learning-rate check finishes early.

Stop rules (the session stops, keeps its last good checkpoint and reports; going on then needs a
decision): a NaN/inf loss; more than 5% skipped fp16 steps in any 2,000-step window; a sampled
validation loss more than 0.1 nats above the best so far at 3 evaluations in a row.

Two GPUs (D-050, approved 2026-10-04 after EXP-044 passed): with ``--gpus 2`` the main run's
training is done by a ``torchrun`` worker with two processes (this same script, ``--ddp-worker``)
running the same :func:`run_main`: 16 windows per GPU per step instead of 16 x 2 on one GPU, so
the same 32 windows per step, order, learning rate and rules. Everything else (Part 0, the chain
check, the learning-rate state) stays in this one process. Per-step decisions (stop rules,
evaluations, checkpoints, time budget) are made on process 0 and shared; the end-of-session
evaluation and final files are made by process 0 alone after the process group is closed. If the
worker fails before its first training step, the session continues on one GPU. Without
``--gpus 2`` nothing of this runs and the one-GPU path is unchanged.

CPU test of the whole flow: ``--smoke`` (tiny model, tiny data; see tests/test_pretrain.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gpu_bringup as gb  # noqa: E402
import gpu_lr_arch as la  # noqa: E402
from frontier_ai.data.multi import MultiTokenDataset, OnePassDataset, full_split_loss  # noqa: E402
from frontier_ai.engine import distributed as dd  # noqa: E402
from frontier_ai.engine.trainer import Trainer  # noqa: E402

ROOT = gb.ROOT
SCHEMA = "frontier-pretrain-v1"
EXP_ID = "EXP-043"
CHAIN_FILE = "chain.json"
CKPT_FILES = ("model.pt", "optimizer.pt", "trainer_state.pt", "meta.json", "config.json")
# D-050: the two-GPU worker. Not part of SETUP_* on purpose: earlier sessions are refused when their
# setup differs, and the GPU count does not change the pre-registered plan.
DDP_DIR = "ddp"  # under --scratch: in.json, worker.log, first_step.json, done.json
DDP_GRACE_MIN = 90.0  # after the training deadline: session-end evaluation, final files, start-up
DDP_NCCL_TIMEOUT_MIN = 60.0
_FIRST_STEP_MARKER: Path | None = None  # set in the worker on process 0

MODEL_FULL = {"n_layer": 14, "n_head": 14, "n_embd": 896, "block_size": 512, "pos": "rope", "n_kv_head": 2}
MODEL_SMOKE = {"n_layer": 2, "n_head": 4, "n_embd": 32, "block_size": 16, "pos": "rope", "n_kv_head": 2}

SETUP_FULL: dict[str, Any] = {
    "model": MODEL_FULL,
    "windows_per_step": 32,  # 32 x 512 = 16,384 tokens per step
    "expected_windows": 5_437_162,  # EXP-043 section 1, from the EXP-037 manifest
    "expected_steps": 169_911,
    "seed": 1,
    "order_seed": 1,
    "lr_grid": [2.5e-4, 5e-4, 1e-3],
    "lr_extension": {"low": 1.25e-4, "high": 2e-3},
    "lr_check_steps": 6104,  # 100 M tokens
    "warmup": 1000,
    "min_lr_ratio": 0.1,
    "eval_every": 1000,
    "eval_iters": 20,
    "eval_batch": 16,
    "save_every": 5000,
    "skip_window": 2000,
    "max_skipped": 0.05,
    "diverge_nats": 0.1,
    "diverge_evals": 3,
    "mem_limit_gb": 14.5,
    "micro_options": [[16, 2], [8, 4]],
    "total_cap_hours": 100.0,
    "reserve_minutes": 30.0,  # session end: save + full validation (+ final files)
    "min_train_minutes": 30.0,  # do not start/resume the main run for less than this
    "noise": 0.0174,  # EXP-042 seed noise
    "reference_run": "C3-s3-lr0.001",  # best EXP-042 run, 0.7736
    "reference_bpb": 0.7736,
    "predicted_bpb": 0.683,  # EXP-042 post-hoc secondary fit at this size and 2.78 B tokens
    "consistent_band": 0.03,
    "sample_prompt_tokens": 32,
    "sample_new_tokens": 128,
    "sample_prompts_per_language": 3,
    "sample_temperature": 0.8,
}
SETUP_SMOKE: dict[str, Any] = {
    **SETUP_FULL,
    "model": MODEL_SMOKE,
    "windows_per_step": 4,
    "expected_windows": None,
    "expected_steps": None,
    "lr_check_steps": 10,
    "warmup": 2,
    "eval_every": 10,
    "eval_iters": 2,
    "eval_batch": 8,
    "save_every": 25,
    "skip_window": 20,
    "micro_options": [[2, 2], [1, 4]],
    "reserve_minutes": 0.0,
    "min_train_minutes": 0.0,
    "sample_prompt_tokens": 6,
    "sample_new_tokens": 6,
}


class StopRule(Exception):
    """A pre-registered stop rule fired: end the session WITHOUT saving the current state."""


# ----------------------------------------------------------------- report --
class Report(la.Report):
    def save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.data["elapsed_min"] = round((time.time() - self.t0) / 60, 1)
        tmp = self.out / "summary.json.tmp"
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        tmp.replace(self.out / "summary.json")
        (self.out / "SUMMARY.txt").write_text(render(self.data), encoding="utf-8", newline="\n")


def _f(v: Any, nd: int = 4) -> str:
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render(d: dict[str, Any]) -> str:
    out = [f"{d['exp_id']} - train the step-12 model (D-049), session {d.get('session')}"]
    if d.get("smoke"):
        out.append("SMOKE TEST, not a result (tiny model, tiny data, CPU)")
    out.append(f"started {d.get('started_at')}  finished {d.get('finished_at', '(running)')}")
    out.append(
        f"elapsed {d.get('elapsed_min')} min; GPU-hours before this session {_f(d.get('hours_before'), 2)}"
    )
    if d.get("stopped"):
        out.append(f"STOPPED: {d['stopped']}")
    if d.get("stop_rule"):
        out.append(f"STOP RULE FIRED: {d['stop_rule']} (continuing needs a decision)")
    for key, label in (
        ("two_gpu_fallback", "TWO-GPU START FAILED"),
        ("two_gpu_failure", "TWO-GPU RUN FAILED"),
    ):
        if d.get(key):
            fb = d[key]
            out.append(f"{label}: {fb.get('note')} (exit code {fb.get('exit_code')})")
            out += [f"    {k}: {v}" for k, v in fb.get("errors", {}).items()]
            out += [f"    | {line}" for line in fb.get("log_tail", [])[-8:]]
    p0 = d.get("part0", {})
    if p0:
        osc = p0.get("one_step_check", {})
        out.append(
            f"Part 0: data files {p0.get('files_ok')}/{p0.get('files_expected')} "
            f"(sha256 {'checked' if p0.get('sha256_checked') else 'NOT checked'}); "
            f"tests {p0.get('tests_line')}; "
            f"one-step float64 check pass={osc.get('pass')}"
        )
    ch = d.get("chain_check")
    if ch:
        out.append(f"Checkpoint chain: {ch.get('result')}")
    if d.get("order"):
        o = d["order"]
        out.append(
            f"One-pass order: {o['n_windows']:,} windows, {o['steps_per_pass']:,} steps, "
            f"fingerprint {o['fingerprint'][:16]}..."
        )
    if d.get("micro"):
        out.append(f"Micro-batch: {d['micro'][0]} x {d['micro'][1]} accumulation ({d.get('micro_note', '')})")
    runs = d.get("lr_runs", [])
    if runs:
        out.append("")
        out.append("Learning-rate check (6,104 steps each, same first 100 M tokens):")
        out.append(f"  {'lr':>9} {'session':>7} {'status':<28} {'mean bpb':>9} {'minutes':>8}")
        for r in runs:
            out.append(
                f"  {r['lr']:>9g} {r.get('session', ''):>7} {r.get('status', '')[:28]:<28} "
                f"{_f(r.get('bpb_mean')):>9} {_f(r.get('minutes'), 1):>8}"
            )
    lc = d.get("lr_choice")
    if lc:
        if lc.get("decided"):
            out.append(f"  Chosen learning rate: {lc['lr']:g}  ({lc['reason']})")
        else:
            out.append(f"  Not decided yet: {lc.get('reason')}")
    m = d.get("main")
    if m:
        out.append("")
        out.append(
            f"Main run: steps {m.get('start_step'):,} -> {m.get('end_step', m.get('start_step')):,} "
            f"of {m.get('total_steps'):,} at lr {m.get('lr'):g}; session end: {m.get('session_end')}"
        )
        if m.get("tokens_per_s"):
            out.append(f"  {m['tokens_per_s']:,.0f} tokens/s; peak memory {_f(m.get('peak_mem_gb'), 2)} GB")
        out.append(f"  skipped fp16 steps this session: {m.get('skipped_steps_session')}")
        mpg = m.get("micro_per_gpu")
        if mpg:
            per = m.get("peak_mem_gb_per_gpu")
            out.append(
                f"  trained on {m.get('gpus', 1)} GPU(s): {mpg[0]} x {mpg[1]} per GPU"
                + (f"; peak memory per GPU {per} GB" if per else "")
                + (f"  [{d['gpus_plan']}]" if d.get("gpus_plan") else "")
            )
        curve = m.get("val_curve", [])
        if curve:
            out.append(
                "  sampled validation loss: "
                + ", ".join(f"{c['step']}: {c['val_loss']:.4f}" for c in curve[-12:])
            )
        se = m.get("session_eval")
        if se:
            out.append(
                f"  full validation at session end: mean {se['bpb_mean']:.4f} bpb "
                f"(token-weighted {se['bpb_token_weighted']:.4f}, pooled {se['bpb_pooled']:.4f})"
            )
    c = d.get("checkpoint")
    if c:
        out.append(f"Checkpoint for the next session: step {c['step']:,} (made in session {c['session']})")
    f = d.get("final")
    if f:
        out.append("")
        out.append("FINAL (one pass complete):")
        s = f["scores"]
        out.append(
            f"  mean {s['bpb_mean']:.4f} bpb, token-weighted {s['bpb_token_weighted']:.4f}, "
            f"pooled {s['bpb_pooled']:.4f}"
        )
        for lang, v in s["bpb"].items():
            ref = f["checks"]["per_language_reference"].get(lang)
            out.append(f"    {lang:<6} {v:.4f}   (C3-s3 {_f(ref)})")
        for k, v in f["checks"].items():
            if k != "per_language_reference":
                out.append(f"  {k}: {v}")
        if f.get("model_final"):
            out.append(f"  model_final.pt sha256 {f['model_final']['sha256']}")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------- previous state --
def load_sessions(prev_dir: Path, setup: dict, smoke: bool) -> list[dict[str, Any]]:
    """The committed earlier sessions; refuses summaries made with another plan."""
    sessions = []
    for f in sorted(prev_dir.glob("session-*/summary.json"), key=lambda p: int(p.parent.name.split("-")[1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("schema") != SCHEMA or bool(d.get("smoke")) != smoke or d.get("setup") != setup:
            raise ValueError(f"{f} was made with a different EXP-043 plan; it cannot be continued")
        sessions.append(d)
    return sessions


def state_from(previous: list[dict[str, Any]]) -> dict[str, Any]:
    st: dict[str, Any] = {
        "lr_runs": [],
        "lr_choice": None,
        "micro": None,
        "fingerprint": None,
        "checkpoint": None,
        "best_val": float("inf"),
        "bad_evals": 0,
        "stop_rule": None,
        "complete": False,
    }
    for d in previous:
        st["lr_runs"] += [r for r in d.get("lr_runs", []) if r.get("session") == d["session"]]
        if d.get("lr_choice", {}) and d["lr_choice"].get("decided"):
            st["lr_choice"] = d["lr_choice"]
        st["micro"] = st["micro"] or d.get("micro")
        st["fingerprint"] = st["fingerprint"] or (d.get("order") or {}).get("fingerprint")
        if d.get("checkpoint"):
            st["checkpoint"] = d["checkpoint"]
        m = d.get("main") or {}
        if m.get("best_val") is not None:
            st["best_val"] = m["best_val"]
            st["bad_evals"] = m.get("bad_evals", 0)
        if d.get("stop_rule"):
            st["stop_rule"] = (d["session"], d["stop_rule"])
        st["complete"] = st["complete"] or bool(d.get("complete"))
    return st


# ---------------------------------------------------------- learning rate --
def choose_lr(runs: list[dict[str, Any]], setup: dict) -> dict[str, Any]:
    """The pre-registered rule (EXP-043 section 2). Returns the decision or the next run(s)."""
    grid, ext, noise = setup["lr_grid"], setup["lr_extension"], setup["noise"]
    tried = {r["lr"] for r in runs if r.get("attempted")}
    missing = [lr for lr in grid if lr not in tried]
    if missing:
        return {"decided": False, "next": missing, "reason": f"grid runs still to do: {missing}"}
    ok = [r for r in runs if r.get("status") == "done" and not r.get("failed")]
    if not ok:
        return {"decided": False, "next": [], "stop": True, "reason": "no run can be chosen (all failed)"}
    best = min(ok, key=lambda r: r["bpb_mean"])
    lower = [r for r in ok if r["lr"] < best["lr"] and r["bpb_mean"] <= best["bpb_mean"] + noise]
    chosen = min(lower, key=lambda r: r["lr"]) if lower else best
    reason = f"lowest mean bpb {best['bpb_mean']:.4f} at {best['lr']:g}" + (
        f"; {chosen['lr']:g} is within the noise {noise} so the lower one is chosen" if lower else ""
    )
    extension_done = any(lr in tried for lr in ext.values())
    if not extension_done:
        edge = ext["low"] if chosen["lr"] == min(grid) else ext["high"] if chosen["lr"] == max(grid) else None
        if edge is not None:
            return {
                "decided": False,
                "next": [edge],
                "reason": f"{reason}; {chosen['lr']:g} is at the edge of the grid, "
                f"one extension run at {edge:g}",
            }
    return {"decided": True, "lr": chosen["lr"], "best_lr": best["lr"], "reason": reason}


def run_lrcheck(
    args: argparse.Namespace, rep: Report, ops: OnePassDataset, setup: dict, st: dict, deadline: float
) -> None:
    runs = rep.data["lr_runs"]
    runs[:] = list(st["lr_runs"])
    micro, accum = rep.data["micro"]
    plan = {
        "warmup": setup["warmup"],
        "eval_iters": setup["eval_iters"],
        "eval_batch": setup["eval_batch"],
        "min_lr_ratio": setup["min_lr_ratio"],
        "eval_interval": setup["eval_every"],
    }
    new = 0
    while True:
        decision = choose_lr(runs, setup)
        rep.data["lr_choice"] = decision
        rep.save()
        if decision["decided"] or decision.get("stop") or not decision["next"]:
            if decision.get("stop"):
                rep.data["stopped"] = "learning-rate check: " + decision["reason"]
            return
        done_min = [r["minutes"] for r in runs if r.get("status") == "done"]
        need = 1.1 * max(done_min) * 60 if done_min else 0.0
        if args.max_runs is not None and new >= args.max_runs:
            return
        if time.time() + need > deadline:
            rep.data["lr_note"] = "the next learning-rate run does not fit this session; it runs in the next"
            return
        lr = decision["next"][0]
        gb._say(f"learning-rate check run at {lr:g} ({setup['lr_check_steps']} steps)")
        r = la.one_run(
            args,
            ops,
            None,
            plan,
            "rope_gqa2",
            lr,
            setup["seed"],
            "lrcheck",
            model=setup["model"],
            steps=setup["lr_check_steps"],
            micro_batch=micro,
            accum=accum,
            eval_interval=setup["eval_every"],
            name=f"lrcheck-lr{lr:g}",
        )
        r.update(session=rep.data["session"], attempted=not r.get("status", "").startswith("error"))
        runs.append(r)
        new += 1
        rep.save()
        if not r["attempted"]:
            rep.data["stopped"] = f"learning-rate run at {lr:g} hit an error: {r['status']}"
            return


# --------------------------------------------------------------- chain I/O --
def checkpoint_hashes(ckpt_dir: Path) -> dict[str, str]:
    return {name: gb.sha256_file(ckpt_dir / name) for name in CKPT_FILES if (ckpt_dir / name).exists()}


def find_chain(root: Path, written_by: int) -> Path | None:
    """The folder (under the mounted inputs) whose chain.json was written by session `written_by`."""
    if not root.exists():
        return None
    hits = []
    for f in root.rglob(CHAIN_FILE):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("exp_id") == EXP_ID and d.get("written_by_session") == written_by:
            hits.append(f.parent)
    return sorted(hits)[0] if hits else None


def check_chain(
    args: argparse.Namespace, previous: list[dict], st: dict
) -> tuple[dict[str, Any], Path | None]:
    """Part 0 (EXP-043): find the previous session's output and verify the checkpoint sha256 values."""
    if not previous:
        return {"result": "first session, nothing to check", "ok": True}, None
    last = previous[-1]["session"]
    found = find_chain(Path(args.chain_in), last)
    expected = st["checkpoint"]
    res: dict[str, Any] = {"looked_in": str(args.chain_in), "expected_from_session": last}
    if found is None:
        res.update(ok=False, result=f"the output of session {last} was not found under {args.chain_in}")
        return res, None
    res["found"] = str(found)
    if expected is None:
        res.update(ok=True, result=f"session {last}'s output is mounted (no checkpoint expected yet)")
        return res, None
    marker = json.loads((found / CHAIN_FILE).read_text(encoding="utf-8")).get("checkpoint")
    actual = checkpoint_hashes(found / "last")
    if marker != expected or actual != expected["files"]:
        bad = sorted(
            k for k in set(expected["files"]) | set(actual) if expected["files"].get(k) != actual.get(k)
        )
        res.update(ok=False, result=f"REFUSED: the checkpoint does not match the committed sha256 ({bad})")
        return res, None
    res.update(
        ok=True, result=f"checkpoint of step {expected['step']:,} verified (sha256 of {len(actual)} files)"
    )
    return res, found


def write_chain(chain_out: Path, session: int, checkpoint: dict | None) -> None:
    chain_out.mkdir(parents=True, exist_ok=True)
    marker = {"exp_id": EXP_ID, "written_by_session": session, "checkpoint": checkpoint}
    (chain_out / CHAIN_FILE).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------- evaluation --
def score_all(model: torch.nn.Module, ds: MultiTokenDataset, block: int, batch: int, device, amp) -> dict:
    bpb: dict[str, float] = {}
    nats: dict[str, float] = {}
    total_bits, total_bytes = 0.0, 0
    for lang, part in ds.parts.items():
        loss, _ = full_split_loss(model, part, block, batch, device, "val", amp)
        nats[lang] = loss
        bpb[lang] = loss / math.log(2) * part.tokens_per_byte("val")
        total_bits += loss / math.log(2) * part.n_val
        total_bytes += part.split_lengths("val")[0]
    return {
        "nats": nats,
        "bpb": bpb,
        "bpb_mean": sum(bpb.values()) / len(bpb),
        "bpb_token_weighted": sum(bpb[x] * ds.parts[x].n_val for x in bpb) / ds.n_val,
        "bpb_pooled": total_bits / total_bytes,
    }


def reference_bpb(setup: dict) -> dict[str, float]:
    for f in sorted((ROOT / "evals" / "results" / "EXP-042").glob("session-*/summary.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for r in d.get("runs", []):
            if r.get("name") == setup["reference_run"] and r.get("session") == d["session"]:
                return r["bpb"]
    return {}


def final_checks(scores: dict, setup: dict) -> dict[str, Any]:
    ref = reference_bpb(setup)
    mean = scores["bpb_mean"]
    gate = setup["reference_bpb"] - setup["noise"]
    consistent = abs(mean - setup["predicted_bpb"]) <= setup["consistent_band"]
    common = [x for x in scores["bpb"] if x in ref]
    return {
        "check1_gate": f"{'PASS' if mean < gate else 'FAIL'}: mean {mean:.4f} vs < {gate:.4f} "
        f"(C3-s3 {setup['reference_bpb']} minus noise {setup['noise']})",
        "check1_pass": mean < gate,
        "check2_prediction": f"{'consistent' if consistent else 'not consistent'}: {mean:.4f} vs predicted "
        f"{setup['predicted_bpb']} (+/- {setup['consistent_band']})",
        "check2_consistent": consistent,
        "check3_languages": (
            f"{sum(scores['bpb'][x] < ref[x] for x in common)} of {len(common)} languages below C3-s3"
            if common
            else "C3-s3 reference not found"
        ),
        "per_language_reference": ref,
    }


def make_samples(model: torch.nn.Module, ds: MultiTokenDataset, setup: dict, device) -> list[dict[str, Any]]:
    """3 fixed prompts per language (the opening tokens of the first validation documents that are long
    enough), continued greedily and at temperature 0.8. For reading only; not scored."""
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    tok = load_frontier_tokenizer_v2()
    eot = tok.special_token_ids["<|endoftext|>"]
    n_p, n_new = setup["sample_prompt_tokens"], setup["sample_new_tokens"]
    model.eval()
    rows = []
    for lang, part in ds.parts.items():
        val = np.asarray(part.split("val"))
        starts = []
        for e in np.flatnonzero(val == eot):
            s = int(e) + 1
            if s + n_p <= len(val) and not (val[s : s + n_p] == eot).any():
                starts.append(s)
            if len(starts) == setup["sample_prompts_per_language"]:
                break
        for i, s in enumerate(starts):
            ids = val[s : s + n_p].astype(np.int64).tolist()
            for mode in ("greedy", f"temperature {setup['sample_temperature']}"):
                g = torch.Generator(device=device).manual_seed(1)
                with torch.no_grad():
                    out = model.generate(
                        torch.tensor([ids], device=device),
                        n_new,
                        temperature=1.0 if mode == "greedy" else setup["sample_temperature"],
                        top_k=1 if mode == "greedy" else None,
                        generator=g,
                        stop_at_eos=eot,
                    )
                cont = out[0, n_p:].tolist()
                rows.append(
                    {
                        "language": lang,
                        "prompt_index": i,
                        "mode": mode,
                        "prompt": tok.decode(ids),
                        "continuation": tok.decode([t for t in cont if t != eot]),
                        "new_tokens": len(cont),
                    }
                )
    return rows


# --------------------------------------------------------------- memory --
def choose_micro(
    args: argparse.Namespace, base: MultiTokenDataset, setup: dict
) -> tuple[list[int], str, Any]:
    """16 x 2 if a few real training steps peak below 14.5 GB, otherwise 8 x 4 (EXP-043 section 3)."""
    if not torch.cuda.is_available():
        return setup["micro_options"][0], "no GPU (smoke): first option", None
    probes = []
    for micro, accum in setup["micro_options"]:
        out = Path(args.scratch) / f"memprobe-{micro}x{accum}"
        cfg = gb.make_cfg(
            setup["model"],
            base.path,
            out,
            max_steps=3,
            batch_size=micro,
            accum_steps=accum,
            lr=1e-4,
            warmup_steps=1,
            precision="fp16",
            compile=True,
            seed=setup["seed"],
            eval_interval=0,
            log_interval=1,
        )
        torch.cuda.reset_peak_memory_stats()
        peak, err = None, None
        try:
            Trainer(cfg, base).fit()
            peak = torch.cuda.max_memory_allocated() / 1024**3
        except Exception as exc:  # noqa: BLE001 - an out-of-memory error means "does not fit"
            err = gb._err(exc)
        gb.reset_backend()
        shutil.rmtree(out, ignore_errors=True)
        probes.append(
            {"micro": [micro, accum], "peak_gb": None if peak is None else round(peak, 3), "error": err}
        )
        if peak is not None and peak < setup["mem_limit_gb"]:
            return [micro, accum], f"peak {peak:.2f} GB < {setup['mem_limit_gb']} GB", probes
    return [], "no micro-batch fits in memory", probes


# --------------------------------------------------------------- main run --
def run_main(
    args: argparse.Namespace,
    rep: Report,
    ops: OnePassDataset,
    setup: dict,
    st: dict,
    deadline: float,
    resume_dir: Path | None,
) -> None:
    gpu = torch.cuda.is_available()
    micro, accum = rep.data["micro"]
    world = dd.world()
    if world > 1:  # D-050: the same windows per step, split over the processes
        if accum % world:
            raise ValueError(f"accumulation {accum} cannot be split over {world} processes")
        accum //= world
    lr = rep.data["lr_choice"]["lr"]
    total = ops.steps_per_pass(setup["windows_per_step"])
    chain_out = Path(args.chain_out)
    cfg = gb.make_cfg(
        setup["model"],
        ops.path,
        chain_out,
        max_steps=total,
        batch_size=micro,
        accum_steps=accum,
        lr=lr,
        warmup_steps=setup["warmup"],
        precision="fp16" if gpu else "fp32",
        compile=gpu,
        seed=setup["seed"],
        data_seed=setup["seed"],
        eval_interval=0,  # evaluations and checkpoints are made by the callback below
        eval_iters=setup["eval_iters"],
        log_interval=100 if not args.smoke else 1,
        save_interval=0,
        save_best=False,
        resume=str(resume_dir) if resume_dir else "",
    )
    cfg.optim.min_lr_ratio = setup["min_lr_ratio"]
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(cfg, ops)
    start = trainer.state.step
    if st["checkpoint"] is not None and start != st["checkpoint"]["step"]:
        raise RuntimeError(f"resumed at step {start}, expected {st['checkpoint']['step']}")
    m: dict[str, Any] = {
        "lr": lr,
        "total_steps": total,
        "start_step": start,
        "end_step": start,
        "val_curve": [],
        "periodic_saves": [],
        "best_val": None if math.isinf(st["best_val"]) else st["best_val"],
        "bad_evals": st["bad_evals"],
        "gpus": world,
        "micro_per_gpu": [micro, accum],
    }
    rep.data["main"] = m
    rep.save()
    best = st["best_val"]
    skips: deque[int] = deque(maxlen=setup["skip_window"])
    last_skipped = trainer.state.skipped_steps
    t_train_end = deadline - setup["reserve_minutes"] * 60
    t0 = time.time()

    def callback(tr: Trainer) -> str | None:
        return dd.run_on_main(lambda: decide(tr))

    def decide(tr: Trainer) -> str | None:
        nonlocal best, last_skipped
        s = tr.state.step
        if _FIRST_STEP_MARKER is not None and s == start + 1:
            _FIRST_STEP_MARKER.write_text(json.dumps({"step": s}) + "\n", encoding="utf-8")
        if args.smoke and args.test_stop_at is not None and s == args.test_stop_at:
            raise StopRule(f"test stop rule at step {s}")
        if args.smoke and os.environ.get("FRONTIER_TEST_DDP_FAIL") == f"step:{s}" and dd.world() > 1:
            raise RuntimeError(f"test: the two-GPU worker fails at step {s}")
        skips.append(tr.state.skipped_steps - last_skipped)
        last_skipped = tr.state.skipped_steps
        m["end_step"] = s
        if sum(skips) > setup["max_skipped"] * setup["skip_window"]:
            raise StopRule(f"{sum(skips)} skipped fp16 steps in the last {len(skips)} steps (step {s})")
        if s % setup["eval_every"] == 0 or s == total:
            val = tr.evaluate()
            tr.model.train()
            m["val_curve"].append(
                {"step": s, "val_loss": round(val, 5), "minutes": round((time.time() - t0) / 60, 2)}
            )
            if not math.isfinite(val):
                raise StopRule(f"non-finite sampled validation loss at step {s}")
            if val < best:
                best, m["bad_evals"] = val, 0
                tr.state.best_val = val
            elif val > best + setup["diverge_nats"]:
                m["bad_evals"] += 1
            else:
                m["bad_evals"] = 0
            m["best_val"] = best
            if m["bad_evals"] >= setup["diverge_evals"]:
                raise StopRule(
                    f"sampled validation loss more than {setup['diverge_nats']} nats above the best "
                    f"({best:.4f}) at {m['bad_evals']} evaluations in a row (step {s})"
                )
            rep.save()
        if s % setup["save_every"] == 0 and s < total:
            tr.save("last")
            m["periodic_saves"].append(s)
            rep.save()  # the record names the newest checkpoint even if the process dies later
        if args.session_steps is not None and s - start >= args.session_steps:
            return "session step limit (test)"
        if time.time() > t_train_end:
            return "time budget of this session"
        return None

    try:
        res = trainer.fit(should_stop=callback)
        m["session_end"] = res["stopped"] or "one pass complete"
    except StopRule as exc:
        rep.data["stop_rule"] = str(exc)
        m["session_end"] = "stop rule"
    except FloatingPointError as exc:
        rep.data["stop_rule"] = f"non-finite training loss ({gb._err(exc)})"
        m["session_end"] = "stop rule"
    train_s = time.time() - t0
    trained = trainer.state.step - start
    m.update(
        end_step=trainer.state.step,
        tokens_seen=trainer.state.tokens_seen,
        train_minutes=round(train_s / 60, 2),
        tokens_per_s=trained * setup["windows_per_step"] * cfg.model.block_size / max(train_s, 1e-9),
        peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 3) if gpu else None,
        skipped_steps_total=trainer.state.skipped_steps,
        skipped_steps_session=trainer.state.skipped_steps - (rep.data.get("skipped_before") or 0),
        sampler_position=ops.position,
    )
    if trainer.world > 1:  # D-050: both processes get here together (decisions are shared)
        m["peak_mem_gb_per_gpu"] = dd.all_gather_object(m["peak_mem_gb"])
        dd.barrier()
        dd.cleanup()  # from here on process 0 works alone: no exchange can hang during evaluation
        if trainer.rank != 0:
            del trainer
            gb.reset_backend()
            return
    rep.save()
    model = trainer.raw_model if trainer.world > 1 else getattr(trainer.model, "_orig_mod", trainer.model)
    amp = trainer.spec.amp_dtype if trainer.spec.amp else None
    if not rep.data.get("stop_rule"):
        t_eval = time.time()
        m["session_eval"] = score_all(
            model, ops.base, cfg.model.block_size, setup["eval_batch"], trainer.device, amp
        )
        m["session_eval_minutes"] = round((time.time() - t_eval) / 60, 2)
        rep.save()
    if trainer.state.step >= total and not rep.data.get("stop_rule"):
        final: dict[str, Any] = {
            "scores": m["session_eval"],
            "checks": final_checks(m["session_eval"], setup),
        }
        rep.data["final"] = final
        rows = make_samples(model, ops.base, setup, trainer.device)
        (rep.out / "samples.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
        )
        final["samples"] = len(rows)
        fdir = Path(args.final_dir)
        fdir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_config": asdict(cfg.model),
                "state_dict": {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
                "exp_id": EXP_ID,
                "step": trainer.state.step,
            },
            fdir / "model_final.pt",
        )
        info = {
            "file": "model_final.pt",
            "sha256": gb.sha256_file(fdir / "model_final.pt"),
            "bytes": (fdir / "model_final.pt").stat().st_size,
            "model_config": asdict(cfg.model),
            "n_params": model.n_params(),
            "step": trainer.state.step,
        }
        (fdir / "model_final.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        final["model_final"] = info
        rep.data["complete"] = True
    del trainer, model
    gb.reset_backend()


# --------------------------------------------------------------- session --
def run_session(
    args: argparse.Namespace, rep: Report, previous: list[dict], setup: dict, deadline: float
) -> None:
    st = state_from(previous)
    rep.data["environment"] = gb.environment_record()  # Part 0 records it again; here for early stops
    chain_out = Path(args.chain_out)
    if chain_out.exists():
        shutil.rmtree(chain_out)
    rep.data["checkpoint"] = st["checkpoint"]  # carried forward unless this session makes a new one
    resume_dir: Path | None = None
    try:
        if st["complete"]:
            rep.data["stopped"] = "the one pass was already completed in an earlier session"
            rep.data["complete"] = True
            return
        if st["stop_rule"]:
            n, why = st["stop_rule"]
            rep.data["stopped"] = f"a stop rule fired in session {n} ({why}); continuing needs a decision"
            return
        needs_chain = args.part != "lrcheck" and (st["lr_choice"] is not None or st["checkpoint"] is not None)
        chain, resume_dir = check_chain(args, previous, st)
        rep.data["chain_check"] = chain
        rep.save()
        if not chain["ok"] and (needs_chain or st["checkpoint"] is not None):
            rep.data["stopped"] = "Part 0 checkpoint chain: " + chain["result"]
            return
        base = la.part0(
            args,
            rep,
            la.CHECK_SHAPE_SMOKE if args.smoke else la.CHECK_SHAPE_FULL,
            la_plan(args),
            device_check=False,
            extra_tests=("tests/test_onepass_sampler.py",),
        )
        if base is None:
            return
        gb._say("Part 0: one-step float64 check, D-048 architecture")
        osc = la.one_step_check(
            args,
            base,
            la.CHECK_SHAPE_SMOKE if args.smoke else la.CHECK_SHAPE_FULL,
            la_plan(args),
            "rope_gqa2",
            la.STATES_SMOKE if args.smoke else la.STATES_FULL,
        )
        rep.data["part0"]["one_step_check"] = osc
        rep.save()
        if osc.get("pass") is False:
            rep.data["stopped"] = "the one-step float64 check failed (D-048 point 3); nothing trained"
            return
        ops = OnePassDataset(base, setup["model"]["block_size"], setup["order_seed"])
        order = {
            "n_windows": ops.n_windows,
            "steps_per_pass": ops.steps_per_pass(setup["windows_per_step"]),
            "dropped_windows": ops.n_windows % setup["windows_per_step"],
            "fingerprint": ops.fingerprint(),
        }
        rep.data["order"] = order
        rep.save()
        if setup["expected_windows"] is not None and (
            order["n_windows"] != setup["expected_windows"]
            or order["steps_per_pass"] != setup["expected_steps"]
        ):
            rep.data["stopped"] = (
                f"{order['n_windows']:,} windows / {order['steps_per_pass']:,} steps, expected "
                f"{setup['expected_windows']:,} / {setup['expected_steps']:,}"
            )
            return
        if st["fingerprint"] and st["fingerprint"] != order["fingerprint"]:
            rep.data["stopped"] = "the one-pass order differs from earlier sessions (fingerprint mismatch)"
            return
        if st["micro"]:
            rep.data["micro"], rep.data["micro_note"] = st["micro"], "chosen in an earlier session"
        else:
            micro, note, probes = choose_micro(args, base, setup)
            rep.data.update(micro=micro or None, micro_note=note, memory_probes=probes)
            if not micro:
                rep.data["stopped"] = note
                return
        rep.save()
        if st["lr_choice"] is None:
            if args.part == "main":
                rep.data["stopped"] = "the learning-rate check is not finished; run --part lrcheck first"
                return
            run_lrcheck(args, rep, ops, setup, st, deadline)
        else:
            rep.data["lr_runs"] = list(st["lr_runs"])
            rep.data["lr_choice"] = st["lr_choice"]
        if rep.data.get("stopped") or args.part == "lrcheck" or not rep.data["lr_choice"].get("decided"):
            return
        need = (setup["reserve_minutes"] + setup["min_train_minutes"]) * 60
        if time.time() + need > deadline:
            rep.data["main_note"] = "not enough time left in this session to start the main run"
            return
        if st["checkpoint"] is not None and resume_dir is None:
            rep.data["stopped"] = "a checkpoint is expected but was not verified"
            return
        rep.data["skipped_before"] = 0
        if resume_dir is not None:
            meta = json.loads((resume_dir / "last" / "meta.json").read_text(encoding="utf-8"))
            rep.data["skipped_before"] = meta.get("skipped_steps", 0)
        gpus, note = gpus_for_main(args, st)
        rep.data["gpus_plan"] = note
        if gpus > 1:
            run_main_two_gpus(args, rep, ops, setup, st, deadline, resume_dir)
        else:
            run_main(args, rep, ops, setup, st, deadline, resume_dir)
    finally:
        finish_chain(rep, chain_out, resume_dir)


# ------------------------------------------------------------ two GPUs --
def gpus_for_main(args: argparse.Namespace, st: dict) -> tuple[int, str]:
    """How many processes train the main run (D-050) and why.

    D-050: the first main-run session (session 2) runs on one GPU; the switch to two happens at a
    checkpoint, i.e. only once the main run has a sha256-checked checkpoint from an earlier session.
    """
    if args.gpus <= 1:
        return 1, "one GPU (--gpus 1)"
    if st["checkpoint"] is None:
        return 1, (
            "one GPU: the main run has no checkpoint yet (D-050: the first main-run session runs on "
            "one GPU; the switch to two GPUs happens at a checkpoint)"
        )
    if args.smoke:
        return 2, "two CPU processes (smoke test of the two-GPU path, gloo)"
    n = torch.cuda.device_count()
    if n < 2:
        return 1, f"--gpus 2 was asked, but {n} GPU(s) are visible: one GPU"
    return 2, f"two GPUs ({torch.cuda.get_device_name(0)} x {n}; D-050)"


def run_main_two_gpus(
    args: argparse.Namespace,
    rep: Report,
    ops: OnePassDataset,
    setup: dict,
    st: dict,
    deadline: float,
    resume_dir: Path | None,
) -> None:
    """Run :func:`run_main` in a two-process torchrun worker; fall back to one GPU if it fails to start."""
    wdir = Path(args.scratch) / DDP_DIR
    shutil.rmtree(wdir, ignore_errors=True)
    wdir.mkdir(parents=True)
    snapshot = json.loads(json.dumps(rep.data))
    payload = {
        "args": vars(args),
        "setup": setup,
        "st": st,
        "deadline": deadline,
        "resume_dir": str(resume_dir) if resume_dir else None,
        "rep_data": rep.data,
        "rep_t0": rep.t0,
        "rep_out": str(rep.out),
    }
    (wdir / "in.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rep.save()
    gb.reset_backend()
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes", "1"]
    cmd += ["--nproc_per_node", "2", str(SCRIPTS / "gpu_pretrain.py"), "--ddp-worker", str(wdir / "in.json")]
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    if args.smoke:  # the CPU test stays on CPU even on a GPU machine
        env["CUDA_VISIBLE_DEVICES"] = ""
    timeout = max(deadline - time.time(), 0.0) + DDP_GRACE_MIN * 60
    gb._say(f"main run on two processes: {' '.join(cmd[-6:])}")
    timed_out = False
    with open(wdir / "worker.log", "w", encoding="utf-8") as log:
        try:
            code = subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            code, timed_out = -9, True
    tail = (wdir / "worker.log").read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
    print("\n".join(tail), flush=True)
    done, started = (wdir / "done.json").exists(), (wdir / "first_step.json").exists()
    summary = rep.out / "summary.json"
    if done or started:  # process 0 kept the session summary up to date: take it over
        rep.data = json.loads(summary.read_text(encoding="utf-8"))
    if done:
        return
    errors = {
        f.stem: (f.read_text(encoding="utf-8", errors="replace").strip().splitlines() or [""])[-1]
        for f in sorted(wdir.glob("error_rank*.txt"))
    }
    failure = {"exit_code": code, "timed_out": timed_out, "errors": errors, "log_tail": tail[-20:]}
    if started:
        failure["note"] = "the two-GPU worker failed after training had started"
        rep.data["two_gpu_failure"] = failure
        rep.data["stopped"] = (
            f"the two-GPU training failed after it started (exit code {code}"
            f"{', timed out' if timed_out else ''}); the newest complete checkpoint is kept"
        )
        return
    rep.data = snapshot
    failure["note"] = (
        "the two-GPU start failed before the first training step; this session continues on one GPU"
    )
    rep.data["two_gpu_fallback"] = failure
    rep.data["gpus_plan"] = snapshot.get("gpus_plan", "") + " -> FALLBACK: one GPU"
    rep.save()
    run_main(args, rep, ops, setup, st, deadline, resume_dir)


class _QuietReport:
    """The report of processes other than 0: they train, process 0 reports."""

    def __init__(self, out: Path, data: dict) -> None:
        self.out, self.data = out, data

    def save(self) -> None:
        pass


def ddp_worker(path: Path) -> int:
    """One process of the two-GPU main run (started by torchrun from :func:`run_main_two_gpus`)."""
    global _FIRST_STEP_MARKER
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = argparse.Namespace(**payload["args"])
    dd.init_from_env(timeout_minutes=DDP_NCCL_TIMEOUT_MIN)
    rank = dd.rank()
    out = Path(payload["rep_out"])
    if rank == 0:
        _FIRST_STEP_MARKER = path.parent / "first_step.json"
        rep: Any = Report(out, args.exp_id, args.smoke)
        rep.data, rep.t0 = payload["rep_data"], payload["rep_t0"]
    else:
        rep = _QuietReport(out, payload["rep_data"])
    setup, st = payload["setup"], payload["st"]
    code = 0
    try:
        if args.smoke and os.environ.get("FRONTIER_TEST_DDP_FAIL") == "start":
            raise RuntimeError("test: the two-GPU start fails")
        if dd.world() != 2:
            raise RuntimeError(f"expected 2 processes, got {dd.world()}")
        manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
        base = MultiTokenDataset([Path(args.data_dir) / f["path"] for f in manifest["files"]])
        ops = OnePassDataset(base, setup["model"]["block_size"], setup["order_seed"])
        if ops.fingerprint() != payload["rep_data"]["order"]["fingerprint"]:
            raise RuntimeError("the worker's one-pass order differs from the session's")
        gb.reset_backend()
        resume = Path(payload["resume_dir"]) if payload["resume_dir"] else None
        rep.data["precision"] = rep.data["precision"] + "; DistributedDataParallel on two GPUs (D-050)"
        run_main(args, rep, ops, setup, st, payload["deadline"], resume)
    except Exception:  # noqa: BLE001 - the parent reads the exit code, these files and the log
        traceback.print_exc()
        (path.parent / f"error_rank{rank}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        code = 1
    if code == 0 and rank == 0:
        rep.save()
        (path.parent / "done.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    dd.cleanup()
    return code


def finish_chain(rep: Report, chain_out: Path, resume_dir: Path | None) -> None:
    """Leave exactly one valid checkpoint (or none yet) plus chain.json in this session's output."""
    session = rep.data["session"]
    last = chain_out / "last"
    if (last / "model.pt").exists():
        meta = json.loads((last / "meta.json").read_text(encoding="utf-8"))
        rep.data["checkpoint"] = {"session": session, "step": meta["step"], "files": checkpoint_hashes(last)}
    elif resume_dir is not None:  # nothing new: carry the verified input checkpoint forward
        shutil.copytree(resume_dir / "last", last)
        if checkpoint_hashes(last) != rep.data["checkpoint"]["files"]:
            rep.data["stopped"] = (
                rep.data.get("stopped") or ""
            ) + "; the carried-forward checkpoint copy differs"
        rep.data["checkpoint_carried_forward"] = True
    for p in chain_out.glob("*") if chain_out.exists() else []:
        if p.is_dir() and p.name.endswith((".tmp", ".old")):
            shutil.rmtree(p, ignore_errors=True)
    write_chain(chain_out, session, rep.data.get("checkpoint"))


def la_plan(args: argparse.Namespace) -> dict:
    return la.PLAN_SMOKE if args.smoke else la.PLAN_FULL


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--ddp-worker"]:  # D-050: one process of the two-GPU main run
        return ddp_worker(Path(argv[1]))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with the 13 EXP-037 .bin + .meta.json files")
    p.add_argument("--manifest", default=gb.DEFAULT_MANIFEST)
    p.add_argument("--part", choices=["auto", "lrcheck", "main"], default="auto")
    p.add_argument("--out", default="out/gpu/EXP-043", help="results: <out>/session-<n>/")
    p.add_argument("--prev-dir", default="evals/results/EXP-043", help="the committed earlier sessions")
    p.add_argument(
        "--chain-in", default="/kaggle/input", help="where the previous session's output is mounted"
    )
    p.add_argument("--chain-out", default="out/gpu/EXP-043-chain", help="this session's checkpoint (output)")
    p.add_argument("--final-dir", default="out/gpu/EXP-043-final", help="model_final.pt (last session)")
    p.add_argument("--scratch", default="out/gpu/EXP-043-scratch")
    p.add_argument("--exp-id", default=EXP_ID)
    p.add_argument("--max-hours", type=float, default=9.0, help="time budget of this session")
    p.add_argument("--smoke", action="store_true", help="tiny model and data (CPU test of the script)")
    p.add_argument(
        "--gpus", type=int, choices=[1, 2], default=1, help="processes for the main run (2: D-050, torchrun)"
    )
    p.add_argument("--test-stop-at", type=int, default=None, help=argparse.SUPPRESS)  # smoke tests only
    p.add_argument("--max-runs", type=int, default=None, help=argparse.SUPPRESS)  # tests
    p.add_argument("--session-steps", type=int, default=None, help=argparse.SUPPRESS)  # tests
    p.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-data-hash", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    setup = json.loads(json.dumps(SETUP_SMOKE if args.smoke else SETUP_FULL))
    prev_dir = Path(args.prev_dir) if Path(args.prev_dir).is_absolute() else ROOT / args.prev_dir
    try:
        previous = load_sessions(prev_dir, setup, args.smoke)
    except ValueError as exc:  # nothing written: a bad session file would block every later launch
        print(f"STOPPED: {exc}", flush=True)
        return 1
    session = max((d["session"] for d in previous), default=0) + 1
    hours_before = sum(d.get("elapsed_min", 0) for d in previous) / 60
    session_hours = min(args.max_hours, setup["total_cap_hours"] - hours_before)
    rep = Report(Path(args.out) / f"session-{session}", args.exp_id, args.smoke)
    rep.data.pop("runs", None)
    rep.data.update(
        schema=SCHEMA,
        part=args.part,
        session=session,
        setup=setup,
        hours_before=round(hours_before, 3),
        session_hours=round(session_hours, 3),
        previous_sessions=[d["session"] for d in previous],
        lr_runs=[],
        precision="fp16 + GradScaler + torch.compile (D-047)",
        primary_metric="mean of per-language validation bits per byte, equal weight",
    )
    deadline = time.time() + max(session_hours, 0) * 3600
    try:
        if session_hours <= 0:
            rep.data["stopped"] = (
                f"the {setup['total_cap_hours']:g} GPU-hour cap over all sessions is used up"
            )
        else:
            run_session(args, rep, previous, setup, deadline)
    except Exception as exc:  # noqa: BLE001 - keep the evidence gathered so far
        traceback.print_exc()
        rep.data["stopped"] = f"unexpected error: {gb._err(exc)}"
    rep.data["finished_at"] = gb._now()
    rep.save()
    print(render(rep.data), flush=True)
    attention = rep.data.get("stopped") or rep.data.get("stop_rule")
    return 0 if not attention or (rep.data.get("complete") and not rep.data.get("stop_rule")) else 1


if __name__ == "__main__":
    sys.exit(main())
