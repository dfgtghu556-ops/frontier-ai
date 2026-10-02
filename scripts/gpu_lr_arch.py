#!/usr/bin/env python3
"""EXP-040 (step 11, phase 1): learning rate and RoPE + GQA-2 at GPU scale on all 13 languages.

    python scripts/gpu_lr_arch.py --data-dir /kaggle/input/.../ --out /kaggle/working/EXP-040 \\
        --scratch /tmp/exp040

Runs the pre-registered EXP-040 plan (EXPERIMENTS.md) on the first CUDA GPU and rewrites
``summary.json`` and ``SUMMARY.txt`` in ``--out`` after every run, so a session that is cut off
still leaves every finished run.

* Part 0: environment; every file of the EXP-037 manifest must be in ``--data-dir`` with the
  recorded sha256; model/trainer tests; the D-047 float64 CPU = GPU check (<= 1e-8 per step,
  50 steps) for the RoPE + GQA-2 maths (if it fails, candidate runs are skipped).
* Grid A: baseline (D-043) and candidate (RoPE + GQA-2) x peak lr {5e-4, 1e-3, 2e-3, 4e-3}, seed 1.
* Grid B: seed 2 for both architectures at the two learning rates where the baseline did best in
  grid A.
* Every run: 100 M tokens (6,104 steps x 32 x 512), fp16 + GradScaler + torch.compile (D-047),
  then each language's FULL validation split is scored. Primary metric: the mean of the 13
  per-language bits per byte with equal weight.
* Rules 1-4 (learning rate, seed noise, architecture verdict, stability) are evaluated exactly as
  pre-registered and written into the summary.

``--part followup`` (EXP-041) re-asks EXP-040's architecture question without changing EXP-040:
* Part A (gate): a one-step float64 check for BOTH architectures. Parameter states are taken from
  a 50-step CPU run (before steps 1, 11, 21, 31, 41, 50); at each, identical parameters and the
  identical batch give one loss and one gradient on the CPU and on the GPU. Nothing carries over
  between steps, so nothing is amplified. Pass: |loss diff| <= 1e-12 and
  ||grad diff|| / ||grad|| <= 1e-10 at every state. The baseline is the control: if it fails,
  the check is wrong and the experiment stops. The EXP-040-style 50-step trajectory of the
  baseline on this data is reported (not a gate).
* Part B (only if RoPE + GQA-2 passes Part A): its 6 runs EXP-040 skipped, then one baseline
  control run (lr 1e-3, seed 1). EXP-040's rules 1-4 are evaluated unchanged on EXP-040's 6
  baseline runs (read from ``--prev``) plus these 6 candidate runs.

Data: the languages are sampled in proportion to their training tokens (the corpus as it is; not
a chosen mixture), see ``frontier_ai.data.multi``. ``--smoke`` shrinks everything for a CPU test.
"""

from __future__ import annotations

import os

# like gpu_bringup: required for the deterministic float64 check, set before torch touches CUDA
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402

import gpu_bringup as gb  # noqa: E402
from frontier_ai.data.multi import MultiTokenDataset, full_split_loss  # noqa: E402
from frontier_ai.engine.trainer import Trainer  # noqa: E402

ROOT = gb.ROOT
SCHEMA = "frontier-lr-arch-v1"
LRS = [5e-4, 1e-3, 2e-3, 4e-3]
ARCHS = {
    "baseline": {"pos": "learned", "n_kv_head": 6},  # D-043 (6 query heads = 6 key-value heads)
    "rope_gqa2": {"pos": "rope", "n_kv_head": 2},  # the candidate
}
SHAPE_FULL = {"n_layer": 8, "n_head": 6, "n_embd": 384, "block_size": 512}  # EXP-038 "M"
SHAPE_SMOKE = {"n_layer": 2, "n_head": 6, "n_embd": 48, "block_size": 32}
CHECK_SHAPE_FULL = {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 128}  # EXP-038 "S"
CHECK_SHAPE_SMOKE = {"n_layer": 2, "n_head": 4, "n_embd": 32, "block_size": 32}
PLAN_FULL = {
    "steps": 6104,
    "batch": 32,
    "warmup": 200,
    "eval_interval": 1000,
    "eval_iters": 20,
    "eval_batch": 16,
    "check_steps": 50,
    "min_lr_ratio": 0.1,
}
PLAN_SMOKE = {
    "steps": 12,
    "batch": 4,
    "warmup": 2,
    "eval_interval": 6,
    "eval_iters": 2,
    "eval_batch": 8,
    "check_steps": 6,
    "min_lr_ratio": 0.1,
}
TOL_FP64 = 1e-8
MAX_SKIPPED = 0.05
# EXP-041 Part A (pre-registered): one step from identical states, nothing compounds
TOL_ONE_STEP_LOSS = 1e-12
TOL_ONE_STEP_GRAD = 1e-10
STATES_FULL = [1, 11, 21, 31, 41, 50]  # "before step k" of the 50-step CPU run
STATES_SMOKE = [1, 3, 6]
CONTROL = ("baseline", 1e-3, 1)  # EXP-041 Part B: one baseline rerun, reported (not a gate)


def run_name(arch: str, lr: float, seed: int) -> str:
    return f"{arch}-lr{lr:g}-s{seed}"


def render_text(d: dict[str, Any]) -> str:
    env = d.get("environment", {})
    lines = [
        (
            f"{d['exp_id']}: follow-up to {d.get('previous', {}).get('exp_id', 'EXP-040')}"
            " - one-step device check, then the RoPE + GQA-2 runs (step 11)"
            if d.get("part") == "followup"
            else f"{d['exp_id']}: learning rate and RoPE + GQA-2 at GPU scale, 13 languages"
            " (step 11, phase 1)"
        )
        + (" [SMOKE TEST, not a result]" if d.get("smoke") else ""),
        f"complete: {d['complete']}   started {d['started_at']}   finished {d.get('finished_at', '-')}"
        f"   elapsed {d.get('elapsed_min', 0)} min",
        f"device: {env.get('gpu_name', 'none')}   torch {env.get('torch', '-')}"
        f"   CUDA {env.get('cuda', '-')}",
    ]
    if d.get("stopped"):
        lines.append(f"STOPPED: {d['stopped']}")
    p0 = d.get("part0", {})
    if p0:
        lines += [
            "",
            "Part 0",
            f"  data: {p0.get('files_ok', '-')} of {p0.get('files_expected', '-')} files match the manifest"
            f" (sha256 checked: {p0.get('sha256_checked')})",
            f"  model/trainer tests: {gb._fmt_pass(p0.get('tests_pass'))} ({p0.get('tests_line', '')})",
        ]
        c = p0.get("fp64_check")
        if c:
            detail = c.get("error") or f"max |loss diff| {c['max_abs_diff']:.3e} over {c['steps']} steps"
            lines.append(f"  float64 CPU = GPU, RoPE + GQA-2: {gb._fmt_pass(c.get('pass'))}  {detail}")
    pa = d.get("part_a")
    if pa:
        lines += [
            "",
            "Part A: one-step float64 check, CPU vs GPU from identical states (pre-registered gate)",
            f"  pass = |loss diff| <= {TOL_ONE_STEP_LOSS:g} and |grad diff|/|grad| <= {TOL_ONE_STEP_GRAD:g}"
            " at every state",
        ]
        for arch, c in pa.items():
            if c.get("error"):
                lines.append(f"  {arch:<10} {gb._fmt_pass(c.get('pass'))}  {c['error']}")
            else:
                lines.append(
                    f"  {arch:<10} {gb._fmt_pass(c.get('pass'))}  max |loss diff| {c['max_loss_diff']:.3e}"
                    f"   max relative grad diff {c['max_grad_rel_diff']:.3e}   states {c['states']}"
                )
    t = d.get("trajectory_baseline")
    if t:
        detail = t.get("error") or (
            f"max |loss diff| {t['max_abs_diff']:.3e} over {t['steps']} steps"
            f" (the EXP-040 criterion {TOL_FP64:g} would say: {gb._fmt_pass(t.get('pass'))})"
        )
        lines.append(f"  reported, not a gate - baseline 50-step float64 trajectory on this data: {detail}")
    runs = d.get("runs", [])
    if runs:
        langs = d.get("languages", [])
        lines += [
            "",
            f"Runs (final validation bits per byte; mean = {len(langs)} languages with equal weight,"
            " tok-w = weighted by validation tokens)",
        ]
        if any(r.get("source") for r in runs):
            lines.append(
                f"  runs marked * were measured in {d.get('previous', {}).get('exp_id', 'the earlier run')}"
            )
        lines.append(f"  {'run':<26}{'grid':>5}{'mean':>8}{'tok-w':>8}{'tok/s':>9}{'min':>6}  status")
        for r in runs:
            if r.get("status") == "done":
                star = "*" if r.get("source") else ""
                lines.append(
                    f"  {r['name'] + star:<26}{r['grid']:>5}{r['bpb_mean']:>8.4f}"
                    f"{r['bpb_token_weighted']:>8.4f}"
                    f"{r['tokens_per_s']:>9,.0f}{r['minutes']:>6.1f}  done"
                )
            else:
                lines.append(
                    f"  {r['name']:<26}{r['grid']:>5}{'-':>8}{'-':>8}{'-':>9}{'-':>6}  {r['status']}"
                )
        done = [r for r in runs if r.get("status") == "done"]
        if done and langs:
            lines += [
                "",
                "  per-language bits per byte:",
                "  " + f"{'run':<26}" + "".join(f"{x:>7}" for x in langs),
            ]
            for r in done:
                lines.append("  " + f"{r['name']:<26}" + "".join(f"{r['bpb'][x]:>7.3f}" for x in langs))
    ctl = d.get("control")
    if ctl:
        if ctl.get("status") == "done":
            lines += [
                "",
                f"Control run (reported, not a gate): {ctl['name']}  mean {ctl['bpb_mean']:.4f}"
                f"  vs {ctl.get('previous_bpb_mean', float('nan')):.4f} in the earlier session"
                f"  (difference {ctl.get('difference_vs_previous', float('nan')):+.4f})",
            ]
        else:
            lines += ["", f"Control run: {ctl['name']}  {ctl.get('status')}"]
    rules = d.get("rules")
    if rules:
        lines += ["", "Pre-registered rules"]
        for arch, r1 in rules.get("rule1_best_lr", {}).items():
            lines.append(f"  1. best learning rate, {arch}: {r1}")
        noise = rules.get("rule2_noise")
        lines.append(
            f"  2. seed noise: {'not computable' if noise is None else f'{noise:.4f} bits per byte'}"
        )
        for lr, v in rules.get("rule3_per_lr", {}).items():
            lines.append(f"  3. at lr {lr}: {v}")
        lines.append(f"  3. RoPE + GQA-2 adopted for phase 2: {rules.get('rule3_adopt')}")
        lines.append(
            f"  4. failed runs (NaN/inf or > 5% skipped steps): {rules.get('rule4_failed') or 'none'}"
        )
    return "\n".join(lines) + "\n"


class Report:
    def __init__(self, out: Path, exp_id: str, smoke: bool) -> None:
        self.out = out
        self.t0 = time.time()
        self.data: dict[str, Any] = {
            "schema": SCHEMA,
            "exp_id": exp_id,
            "smoke": smoke,
            "complete": False,
            "started_at": gb._now(),
            "runs": [],
        }

    def save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.data["elapsed_min"] = round((time.time() - self.t0) / 60, 1)
        tmp = self.out / "summary.json.tmp"
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        tmp.replace(self.out / "summary.json")
        (self.out / "SUMMARY.txt").write_text(render_text(self.data), encoding="utf-8", newline="\n")


# ------------------------------------------------------------------ part 0 --
def part0(
    args: argparse.Namespace, rep: Report, check_shape: dict, plan: dict, device_check: bool = True
) -> MultiTokenDataset | None:
    rep.data["environment"] = gb.environment_record()
    p0: dict[str, Any] = {}
    rep.data["part0"] = p0
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir)
    files = manifest["files"]
    p0["files_expected"] = len(files)
    p0["sha256_checked"] = not args.skip_data_hash
    p0["packed_id"] = manifest.get("packed_id")
    bad = []
    for f in files:
        path = data_dir / f["path"]
        if not path.exists() or not (data_dir / f.get("meta", Path(f["path"]).stem + ".meta.json")).exists():
            bad.append(f"{f['path']}: missing")
        elif not args.skip_data_hash and gb.sha256_file(path) != f["sha256"]:
            bad.append(f"{f['path']}: sha256 differs")
    p0["files_ok"] = len(files) - len(bad)
    p0["problems"] = bad
    rep.save()
    if bad:
        rep.data["stopped"] = "data files do not match the EXP-037 manifest: " + "; ".join(bad)
        return None
    if not args.smoke and not torch.cuda.is_available():
        rep.data["stopped"] = "no CUDA GPU found (on Kaggle: the accelerator must be a GPU)"
        return None
    if args.skip_tests:
        p0["tests_pass"], p0["tests_line"] = None, "skipped"
    else:
        p0["tests_pass"], p0["tests_line"] = gb.run_repo_tests()
        if not p0["tests_pass"]:
            rep.data["stopped"] = "the model/trainer tests failed on this machine"
            return None
    ds = MultiTokenDataset([data_dir / f["path"] for f in files])
    if ds.vocab_size != 32896:
        rep.data["stopped"] = f"vocabulary {ds.vocab_size}, expected 32896 (Frontier Tokenizer v2)"
        return None
    rep.data["languages"] = ds.names
    rep.data["sampling_weights"] = {n: round(float(w), 6) for n, w in zip(ds.names, ds.weights("train"))}
    rep.save()
    if device_check:  # EXP-040: the D-047 check for maths that has never run on a GPU (RoPE + GQA-2)
        p0["fp64_check"] = trajectory_check(args, ds, check_shape, plan, "rope_gqa2")
        rep.save()
    return ds


# ----------------------------------------------------------- device checks --
def _check_cfg(
    args: argparse.Namespace, ds: MultiTokenDataset, check_shape: dict, plan: dict, arch: str, d: str
):
    # the check shape (EXP-038 "S") has 4 query heads: the baseline is full multi-head attention
    # there (as in EXP-039), the candidate keeps 2 key-value heads
    over = dict(ARCHS[arch])
    if arch == "baseline":
        over["n_kv_head"] = check_shape["n_head"]
    return gb.make_cfg(
        {**check_shape, **over},
        ds.path,
        Path(args.scratch) / f"fp64-{arch}-{d}",
        max_steps=plan["check_steps"],
        device=d,
        precision="fp32",
        deterministic=True,
        eval_iters=2,
    )


def trajectory_check(
    args: argparse.Namespace, ds: MultiTokenDataset, check_shape: dict, plan: dict, arch: str
) -> dict[str, Any]:
    """D-047 / EXP-040: train 50 steps in float64 on the CPU and on the GPU, compare every loss."""
    gpu = torch.cuda.is_available()
    dev = "cuda" if gpu else "cpu"
    try:
        n = plan["check_steps"]
        runs = {}
        for d in ("cpu", dev):
            runs[d], dtype = gb.exact_losses(_check_cfg(args, ds, check_shape, plan, arch, d), ds, fp64=True)
            if dtype != "float64":
                raise RuntimeError(f"model ran in {dtype}, not float64")
        diffs = [abs(a - b) for a, b in zip(runs["cpu"], runs[dev])]
        if len(diffs) != n:
            raise RuntimeError(f"expected {n} losses per run")
        out = {
            "pass": (max(diffs) <= TOL_FP64) if gpu else None,
            "steps": n,
            "max_abs_diff": max(diffs),
            "diffs": diffs,
            "tolerance": TOL_FP64,
            "devices": ["cpu", dev],
            "arch": arch,
        }
    except Exception as exc:  # noqa: BLE001
        out = {"pass": False, "error": gb._err(exc), "arch": arch}
    shutil.rmtree(Path(args.scratch), ignore_errors=True)
    return out


def capture_states(
    cfg, ds: MultiTokenDataset, wanted: list[int]
) -> list[tuple[int, dict, torch.Tensor, torch.Tensor]]:
    """Train on the CPU in float64 and keep (parameters, batch) at the START of the wanted steps."""
    out = Path(cfg.train.out_dir)
    if out.exists():
        shutil.rmtree(out)
    caps: list[tuple[int, dict, torch.Tensor, torch.Tensor]] = []
    count = 0

    def pre(module, inputs, kwargs):
        nonlocal count
        if not (torch.is_grad_enabled() and module.training):
            return
        count += 1  # accum_steps = 1: one training forward = one optimizer step
        if count in wanted:
            x = inputs[0]
            y = kwargs["targets"] if "targets" in kwargs else inputs[1]
            state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            caps.append((count, state, x.detach().clone(), y.detach().clone()))

    with gb.math_attention(True):
        trainer = Trainer(cfg, ds)
        trainer.model.double()
        handle = trainer.model.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            trainer.fit()
        finally:
            handle.remove()
    del trainer
    gb.reset_backend()
    shutil.rmtree(out, ignore_errors=True)
    return caps


def _loss_and_grad(model: torch.nn.Module, state: dict, x: torch.Tensor, y: torch.Tensor, device: str):
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    with gb.math_attention(True):
        loss = model(x.to(device), targets=y.to(device)).loss
        loss.backward()
    grad = torch.cat([p.grad.detach().reshape(-1).to("cpu") for p in model.parameters()])
    return float(loss.detach()), grad


def one_step_check(
    args: argparse.Namespace,
    ds: MultiTokenDataset,
    check_shape: dict,
    plan: dict,
    arch: str,
    states: list[int],
) -> dict[str, Any]:
    """EXP-041 Part A: CPU vs GPU, one loss + gradient from identical parameters and batch."""
    from frontier_ai.model.gpt import GPT

    gpu = torch.cuda.is_available()
    dev = "cuda" if gpu else "cpu"
    try:
        cfg = _check_cfg(args, ds, check_shape, plan, arch, "cpu")
        caps = capture_states(cfg, ds, states)
        if [c[0] for c in caps] != states:
            raise RuntimeError(f"captured states {[c[0] for c in caps]}, expected {states}")
        models = {}
        for d in ("cpu", dev):
            m = GPT(cfg.model).double().to(d)
            m.train()
            models[d] = m
        if str(next(models[dev].parameters()).dtype) != "torch.float64":
            raise RuntimeError("the GPU model is not float64")
        rows = []
        for step, state, x, y in caps:
            lc, gc = _loss_and_grad(models["cpu"], state, x, y, "cpu")
            lg, gg = _loss_and_grad(models[dev], state, x, y, dev)
            rows.append(
                {
                    "state": step,
                    "loss_cpu": lc,
                    "loss_gpu": lg,
                    "loss_diff": abs(lc - lg),
                    "grad_rel_diff": float((gc - gg).norm() / gc.norm()),
                    "grad_norm": float(gc.norm()),
                }
            )
        max_l = max(r["loss_diff"] for r in rows)
        max_g = max(r["grad_rel_diff"] for r in rows)
        ok = max_l <= TOL_ONE_STEP_LOSS and max_g <= TOL_ONE_STEP_GRAD
        out = {
            "pass": ok if gpu else None,  # CPU vs CPU (smoke) is not a result
            "arch": arch,
            "devices": ["cpu", dev],
            "states": states,
            "max_loss_diff": max_l,
            "max_grad_rel_diff": max_g,
            "tolerance_loss": TOL_ONE_STEP_LOSS,
            "tolerance_grad_rel": TOL_ONE_STEP_GRAD,
            "rows": rows,
        }
        del models
    except Exception as exc:  # noqa: BLE001 - a crash is a failed check, recorded
        out = {"pass": False, "error": gb._err(exc), "arch": arch}
    gb.reset_backend()
    shutil.rmtree(Path(args.scratch), ignore_errors=True)
    return out


# -------------------------------------------------------------------- runs --
def one_run(
    args: argparse.Namespace,
    ds: MultiTokenDataset,
    shape: dict,
    plan: dict,
    arch: str,
    lr: float,
    seed: int,
    grid: str,
) -> dict[str, Any]:
    name = run_name(arch, lr, seed)
    out_dir = Path(args.scratch) / name
    r: dict[str, Any] = {
        "name": name,
        "grid": grid,
        "arch": arch,
        "lr": lr,
        "seed": seed,
        "status": "running",
    }
    gpu = torch.cuda.is_available()
    cfg = gb.make_cfg(
        {**shape, **ARCHS[arch]},
        ds.path,
        out_dir,
        max_steps=plan["steps"],
        batch_size=plan["batch"],
        lr=lr,
        warmup_steps=plan["warmup"],
        precision="fp16" if gpu else "fp32",  # CPU only in --smoke
        compile=gpu,
        seed=seed,
        data_seed=seed,
        eval_interval=plan["eval_interval"],
        eval_iters=plan["eval_iters"],
        log_interval=50,
    )
    cfg.optim.min_lr_ratio = plan["min_lr_ratio"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    failure = None
    try:
        trainer = Trainer(cfg, ds)
        result = trainer.fit()
    except FloatingPointError as exc:
        failure = f"failed: non-finite loss ({gb._err(exc)})"
    except Exception as exc:  # noqa: BLE001
        failure = f"error: {gb._err(exc)}"
    if failure is not None:
        gb.reset_backend()
        shutil.rmtree(out_dir, ignore_errors=True)
        r["status"] = failure
        r["failed"] = failure.startswith("failed")
        return r
    train_seconds = time.time() - t0
    model = getattr(trainer.model, "_orig_mod", trainer.model)
    amp = trainer.spec.amp_dtype if trainer.spec.amp else None
    bpb: dict[str, float] = {}
    nats: dict[str, float] = {}
    total_bits, total_bytes = 0.0, 0
    for lang, part in ds.parts.items():
        loss, _n_pred = full_split_loss(
            model, part, shape["block_size"], plan["eval_batch"], trainer.device, "val", amp
        )
        tpb = part.tokens_per_byte("val")
        nats[lang] = loss
        bpb[lang] = loss / math.log(2) * tpb
        total_bits += loss / math.log(2) * part.n_val
        total_bytes += part.split_lengths("val")[0]
    log = gb.read_log(out_dir)
    end = next((x for x in reversed(log) if x.get("event") == "run.end"), {})
    skipped = result["skipped_steps"]
    r.update(
        status="done",
        n_params=model.n_params(),
        steps=result["steps"],
        tokens=trainer.state.tokens_seen,
        minutes=(time.time() - t0) / 60,
        train_minutes=train_seconds / 60,
        tokens_per_s=float(end.get("mean_tps") or trainer.state.tokens_seen / train_seconds),
        peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 3) if gpu else None,
        skipped_steps=skipped,
        failed=skipped > MAX_SKIPPED * result["steps"],
        val_curve=[{"step": e["step"], "val_loss": e["val_loss"]} for e in log if e.get("event") == "eval"],
        nats=nats,
        bpb=bpb,
        bpb_mean=sum(bpb.values()) / len(bpb),
        bpb_token_weighted=sum(bpb[x] * ds.parts[x].n_val for x in bpb) / ds.n_val,
        bpb_pooled=total_bits / total_bytes,  # all validation bits / all validation bytes
    )
    if not math.isfinite(r["bpb_mean"]):
        r["failed"] = True
        r["status"] = "failed: non-finite validation loss"
    elif r["failed"]:
        r["status"] = f"failed: {skipped} of {result['steps']} steps skipped by the fp16 scaler"
    del trainer, model
    gb.reset_backend()
    shutil.rmtree(out_dir, ignore_errors=True)
    return r


def evaluate_rules(runs: list[dict[str, Any]], candidate_ok: bool) -> dict[str, Any]:
    ok = {r["name"]: r for r in runs if r.get("status") == "done" and not r.get("failed")}
    rules: dict[str, Any] = {"rule1_best_lr": {}, "rule3_per_lr": {}}
    for arch in ARCHS:
        cands = [(ok[run_name(arch, lr, 1)]["bpb_mean"], lr) for lr in LRS if run_name(arch, lr, 1) in ok]
        if not cands:
            rules["rule1_best_lr"][arch] = "not measured"
            continue
        best = min(cands)[1]
        edge = " (EDGE of the range: the optimum may lie outside it)" if best in (LRS[0], LRS[-1]) else ""
        rules["rule1_best_lr"][arch] = f"{best:g}{edge}"
    b_lrs = sorted({r["lr"] for r in runs if r["grid"] == "B"})
    pairs = []
    for arch in ARCHS:
        for lr in b_lrs:
            a, b = ok.get(run_name(arch, lr, 1)), ok.get(run_name(arch, lr, 2))
            if a and b:
                pairs.append(abs(a["bpb_mean"] - b["bpb_mean"]))
    if not candidate_ok:
        rules["rule2_noise"] = None
        rules["rule3_adopt"] = "NOT TESTED (the float64 check for RoPE + GQA-2 failed; baseline stays)"
    elif len(b_lrs) != 2 or len(pairs) != 4:
        rules["rule2_noise"] = None
        rules["rule3_adopt"] = "NOT DECIDED (grid B incomplete; baseline stays)"
    else:
        noise = max(pairs)
        rules["rule2_noise"] = noise
        verdicts = []
        for lr in b_lrs:
            mb = (
                ok[run_name("baseline", lr, 1)]["bpb_mean"] + ok[run_name("baseline", lr, 2)]["bpb_mean"]
            ) / 2
            mc = (
                ok[run_name("rope_gqa2", lr, 1)]["bpb_mean"] + ok[run_name("rope_gqa2", lr, 2)]["bpb_mean"]
            ) / 2
            diff = mb - mc
            v = "BETTER" if diff > noise else ("WORSE" if diff < -noise else "NO DETECTABLE DIFFERENCE")
            verdicts.append(v)
            rules["rule3_per_lr"][f"{lr:g}"] = (
                f"{v} (baseline {mb:.4f} - candidate {mc:.4f} = {diff:+.4f}; noise {noise:.4f})"
            )
        rules["rule3_adopt"] = "YES" if all(v == "BETTER" for v in verdicts) else "NO (baseline stays)"
    rules["rule4_failed"] = [r["name"] for r in runs if r.get("failed")]
    return rules


class RunQueue:
    """Starts runs one after another while they still fit the session's time budget."""

    def __init__(
        self,
        args: argparse.Namespace,
        ds: MultiTokenDataset,
        shape: dict,
        plan: dict,
        rep: Report,
        deadline: float,
    ) -> None:
        self.args, self.ds, self.shape, self.plan, self.rep, self.deadline = (
            args,
            ds,
            shape,
            plan,
            rep,
            deadline,
        )
        self.last_minutes: float | None = None

    def go(
        self,
        arch: str,
        lr: float,
        seed: int,
        grid: str,
        skip_reason: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        stub = {"name": name or run_name(arch, lr, seed), "grid": grid, "arch": arch, "lr": lr, "seed": seed}
        if skip_reason:
            return {**stub, "status": f"skipped ({skip_reason})"}
        if time.time() + 1.1 * (self.last_minutes or 0) * 60 > self.deadline:
            return {**stub, "status": "skipped (time budget)"}
        gb._say(f"run {stub['name']} (grid {grid})")
        r = one_run(self.args, self.ds, self.shape, self.plan, arch, lr, seed, grid)
        r["name"] = stub["name"]
        if r.get("minutes"):
            self.last_minutes = r["minutes"]
        return r


def run_grid(
    args: argparse.Namespace, rep: Report, ds: MultiTokenDataset, shape: dict, plan: dict, deadline: float
) -> None:
    """EXP-040: Part 0's trajectory check gates the candidate; grids A and B; rules 1-4."""
    candidate_ok = rep.data["part0"]["fp64_check"].get("pass") is not False
    runs: list[dict[str, Any]] = rep.data["runs"]
    q = RunQueue(args, ds, shape, plan, rep, deadline)

    def go(arch: str, lr: float, seed: int, grid: str) -> None:
        skip = "float64 check failed" if arch == "rope_gqa2" and not candidate_ok else None
        runs.append(q.go(arch, lr, seed, grid, skip))
        rep.save()

    for lr in LRS:
        for arch in ARCHS:
            go(arch, lr, 1, "A")
    base = [r for r in runs if r["arch"] == "baseline" and r.get("status") == "done" and not r.get("failed")]
    best_two = sorted(base, key=lambda r: r["bpb_mean"])[:2]
    rep.data["grid_b_lrs"] = sorted(r["lr"] for r in best_two)
    for lr in rep.data["grid_b_lrs"]:
        for arch in ARCHS:
            go(arch, lr, 2, "B")
    rep.data["rules"] = evaluate_rules(runs, candidate_ok)
    rep.data["complete"] = len(runs) == 12 and all(r.get("status") == "done" for r in runs)


def load_previous(path: Path, shape: dict, plan: dict, smoke: bool) -> dict[str, Any]:
    """The EXP-040 summary whose baseline runs EXP-041 reuses; refuses a mismatched one."""
    prev = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if prev.get("schema") != SCHEMA:
        problems.append(f"schema {prev.get('schema')!r}")
    if bool(prev.get("smoke")) != smoke:
        problems.append("smoke flag differs")
    if prev.get("plan", {}).get("shape") != shape:
        problems.append("model shape differs")
    if prev.get("plan", {}).get("steps", {}).get("steps") != plan["steps"]:
        problems.append("steps per run differ")
    if prev.get("plan", {}).get("steps", {}).get("batch") != plan["batch"]:
        problems.append("batch differs")
    if len(prev.get("grid_b_lrs", [])) != 2:
        problems.append("grid B learning rates missing")
    if problems:
        raise ValueError(f"{path} cannot be combined with this run: " + "; ".join(problems))
    return prev


def run_followup(
    args: argparse.Namespace,
    rep: Report,
    ds: MultiTokenDataset,
    shape: dict,
    check_shape: dict,
    plan: dict,
    deadline: float,
    prev: dict[str, Any],
) -> None:
    """EXP-041: Part A (one-step check, both architectures), then Part B if the candidate passes."""
    states = STATES_SMOKE if args.smoke else STATES_FULL
    pa: dict[str, Any] = {}
    rep.data["part_a"] = pa
    for arch in ARCHS:  # the baseline (the control) first
        gb._say(f"Part A: one-step float64 check, {arch}")
        pa[arch] = one_step_check(args, ds, check_shape, plan, arch, states)
        rep.save()
    gb._say("reported: baseline 50-step float64 trajectory on this data")
    rep.data["trajectory_baseline"] = trajectory_check(args, ds, check_shape, plan, "baseline")
    rep.save()
    base_ok = pa["baseline"].get("pass") is not False
    cand_ok = pa["rope_gqa2"].get("pass") is not False
    prev_id = prev.get("exp_id", "previous")
    prev_base = [dict(r, source=prev_id) for r in prev["runs"] if r.get("arch") == "baseline"]
    runs: list[dict[str, Any]] = rep.data["runs"]
    runs.extend(prev_base)
    q = RunQueue(args, ds, shape, plan, rep, deadline)
    if not base_ok:
        rep.data["stopped"] = (
            "the BASELINE failed Part A: the check itself is wrong; no candidate runs (pre-registered)"
        )
        skip = "Part A failed for the baseline (control)"
    elif not cand_ok:
        skip = "Part A failed for RoPE + GQA-2"
    else:
        skip = None
    plan_runs = [(lr, 1, "A") for lr in LRS] + [(lr, 2, "B") for lr in prev["grid_b_lrs"]]
    for lr, seed, grid in plan_runs:
        runs.append(q.go("rope_gqa2", lr, seed, grid, skip))
        rep.save()
    arch, lr, seed = CONTROL
    ctl = q.go(arch, lr, seed, "ctl", skip, name=run_name(arch, lr, seed) + "-control")
    before = next(
        (r for r in prev_base if r["name"] == run_name(arch, lr, seed) and r.get("status") == "done"), None
    )
    if ctl.get("status") == "done" and before:
        ctl["previous_bpb_mean"] = before["bpb_mean"]
        ctl["difference_vs_previous"] = ctl["bpb_mean"] - before["bpb_mean"]
    rep.data["control"] = ctl
    rep.data["rules"] = evaluate_rules(runs, base_ok and cand_ok)
    if not base_ok:
        rep.data["rules"]["rule3_adopt"] = "NOT TESTED (the baseline control failed Part A; baseline stays)"
    elif not cand_ok:
        rep.data["rules"]["rule3_adopt"] = (
            "NOT TESTED (RoPE + GQA-2 failed the one-step check; baseline stays)"
        )
    new = [r for r in runs if not r.get("source")]
    rep.data["complete"] = (
        base_ok
        and cand_ok
        and len(new) == 6
        and all(r.get("status") == "done" for r in new)
        and ctl.get("status") == "done"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with the 13 EXP-037 .bin + .meta.json files")
    p.add_argument("--manifest", default=gb.DEFAULT_MANIFEST)
    p.add_argument("--out", default="out/gpu/EXP-040")
    p.add_argument("--scratch", default="out/gpu/EXP-040-scratch")
    p.add_argument("--exp-id", default="EXP-040")
    p.add_argument(
        "--part",
        choices=["lr-arch", "followup"],
        default="lr-arch",
        help="lr-arch = EXP-040 (default); followup = EXP-041 (needs --prev)",
    )
    p.add_argument(
        "--prev",
        default="evals/results/EXP-040/summary.json",
        help="followup: the EXP-040 summary whose baseline runs are reused",
    )
    p.add_argument("--max-hours", type=float, default=9.0, help="hard time budget for the whole session")
    p.add_argument("--smoke", action="store_true", help="tiny model and steps (CPU test of the script)")
    p.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-data-hash", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    deadline = time.time() + args.max_hours * 3600
    shape, check_shape, plan = (
        (SHAPE_SMOKE, CHECK_SHAPE_SMOKE, PLAN_SMOKE)
        if args.smoke
        else (SHAPE_FULL, CHECK_SHAPE_FULL, PLAN_FULL)
    )
    followup = args.part == "followup"
    rep = Report(Path(args.out), args.exp_id, args.smoke)
    rep.data["part"] = args.part
    rep.data["plan"] = {
        "shape": shape,
        "archs": ARCHS,
        "lrs": LRS,
        "steps": plan,
        "max_hours": args.max_hours,
        "tokens_per_run": plan["steps"] * plan["batch"] * shape["block_size"],
        "precision": "fp16 + GradScaler + torch.compile (D-047)",
        "primary_metric": "mean of per-language validation bits per byte, equal weight",
    }
    try:
        prev = None
        if followup:
            prev_path = Path(args.prev) if Path(args.prev).is_absolute() else ROOT / args.prev
            prev = load_previous(prev_path, shape, plan, args.smoke)
            rep.data["previous"] = {
                "exp_id": prev.get("exp_id"),
                "file": args.prev,
                "code_commit": prev.get("environment", {}).get("code_commit"),
                "grid_b_lrs": prev["grid_b_lrs"],
            }
            rep.data["grid_b_lrs"] = prev["grid_b_lrs"]
        ds = part0(args, rep, check_shape, plan, device_check=not followup)
        if ds is None:
            rep.data["finished_at"] = gb._now()
            rep.save()
            print(render_text(rep.data), flush=True)
            return 1
        if followup:
            run_followup(args, rep, ds, shape, check_shape, plan, deadline, prev)
        else:
            run_grid(args, rep, ds, shape, plan, deadline)
    except Exception as exc:  # noqa: BLE001 - keep the evidence gathered so far
        traceback.print_exc()
        rep.data["stopped"] = f"unexpected error: {gb._err(exc)}"
    rep.data["finished_at"] = gb._now()
    rep.save()
    print(render_text(rep.data), flush=True)
    return 0 if rep.data["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
