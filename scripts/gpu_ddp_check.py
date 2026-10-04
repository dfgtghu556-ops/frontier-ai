#!/usr/bin/env python
"""EXP-044 (step 13): check two-GPU data-parallel training before using it.

Kaggle attaches two T4s to every GPU session; until now one was used (D-047). This script trains the
real D-049 model (14 x 896, fresh weights, seed 1) for the first 300 steps of the EXP-043 one-pass
order twice, once on one GPU and once on two (DistributedDataParallel, ``torchrun``), and applies
the pre-registered rules (EXPERIMENTS.md, EXP-044):

* **agreement:** the full-validation mean bits per byte after 300 steps differ by at most 0.01; no
  NaN/inf; at most 5% skipped fp16 steps in either run;
* **speed:** two GPUs process at least 1.4x the tokens per second of one GPU over steps 101-300;
* **memory:** peak memory per GPU under 14.5 GB.

Each step uses the same 32 windows of 512 tokens in both runs. One GPU: 16 x 2 accumulation (8 x 4
if 16 does not fit). Two GPUs: the same micro-batch per GPU, half the accumulation (16 x 1 or
8 x 2). Every run is a separate process (``--worker``), so GPU memory starts clean and a crash in
one run cannot spoil the other. Results: ``<out>/summary.json`` and ``SUMMARY.txt``.

CPU test of the whole flow: ``--smoke`` (tiny model, tiny data, gloo backend; tests/test_ddp.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402

import gpu_bringup as gb  # noqa: E402
import gpu_lr_arch as la  # noqa: E402
import gpu_pretrain as gp  # noqa: E402
from frontier_ai.data.multi import MultiTokenDataset, OnePassDataset  # noqa: E402
from frontier_ai.engine import distributed as dd  # noqa: E402
from frontier_ai.engine.trainer import Trainer  # noqa: E402
from frontier_ai.model.gpt import GPT  # noqa: E402
from frontier_ai.utils.seed import set_seed  # noqa: E402

ROOT = gb.ROOT
SCHEMA = "frontier-ddp-check-v1"
EXP_ID = "EXP-044"

SETUP_FULL: dict[str, Any] = {
    "model": gp.MODEL_FULL,
    "windows_per_step": 32,
    "steps": 300,
    "speed_from_step": 100,  # tokens/s is measured over steps 101-300 (after compile warm-up)
    "lr": 5e-4,  # the centre of the EXP-043 grid; the same in both runs
    "warmup": 100,
    "min_lr_ratio": 0.1,
    "seed": 1,
    "order_seed": 1,  # the EXP-043 one-pass order
    "eval_batch": 16,
    "micro_options_one": [[16, 2], [8, 4]],
    "gpus": 2,
    "agree_bpb": 0.01,
    "min_speedup": 1.4,
    "mem_limit_gb": 14.5,
    "max_skipped": 0.05,
    "worker_timeout_min": 50,
}
SETUP_SMOKE: dict[str, Any] = {
    **SETUP_FULL,
    "model": gp.MODEL_SMOKE,
    "windows_per_step": 4,
    "steps": 30,
    "speed_from_step": 10,
    "lr": 1e-3,
    "warmup": 5,
    "eval_batch": 8,
    "micro_options_one": [[2, 2], [1, 4]],
    "worker_timeout_min": 10,
}


def setup_for(smoke: bool) -> dict[str, Any]:
    return json.loads(json.dumps(SETUP_SMOKE if smoke else SETUP_FULL))


# ------------------------------------------------------------------ report --
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
    out = [f"{d['exp_id']} - two-GPU data-parallel check (step 13)"]
    if d.get("smoke"):
        out.append("SMOKE TEST, not a result (tiny model, tiny data, CPU)")
    out.append(f"started {d.get('started_at')}  finished {d.get('finished_at', '(running)')}")
    out.append(f"elapsed {d.get('elapsed_min')} min")
    if d.get("stopped"):
        out.append(f"STOPPED: {d['stopped']}")
    p0 = d.get("part0", {})
    if p0:
        out.append(
            f"Part 0: data files {p0.get('files_ok')}/{p0.get('files_expected')} "
            f"(sha256 {'checked' if p0.get('sha256_checked') else 'NOT checked'}); "
            f"tests {p0.get('tests_line')}"
        )
    env = d.get("environment", {})
    if env:
        out.append(f"GPUs attached: {env.get('device_count')} x {env.get('gpu_name')}")
    o = d.get("order")
    if o:
        out.append(
            f"One-pass order fingerprint {o['fingerprint'][:16]}... (same as EXP-043: {o['same_as_exp043']})"
        )
    for key, title in (("one", "One GPU"), ("two", "Two GPUs")):
        r = d.get(key)
        if not r:
            continue
        out.append("")
        out.append(
            f"{title}: micro-batch {r.get('micro')} x {r.get('accum')} per GPU, {r.get('world')} process(es)"
        )
        if r.get("error"):
            out.append(f"  ERROR: {r['error']}")
            continue
        out.append(
            f"  {_f(r.get('tokens_per_s'), 0)} tokens/s (steps {r.get('speed_steps')}); peak memory per GPU "
            f"{r.get('peak_mem_gb')} GB; skipped fp16 steps {r.get('skipped_steps')}"
        )
        s = r.get("scores")
        if s:
            out.append(f"  validation after {r.get('steps')} steps: mean {s['bpb_mean']:.4f} bpb")
    v = d.get("verdict")
    if v:
        out.append("")
        out.append("Pre-registered rules:")
        for k in ("agreement", "speed", "memory"):
            out.append(f"  {k}: {v[k]}")
        out.append(f"VERDICT: {v['result']}")
        if v.get("saving"):
            out.append(f"  {v['saving']}")
    out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------ worker --
def _weights_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_base(args: argparse.Namespace) -> MultiTokenDataset:
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    return MultiTokenDataset([Path(args.data_dir) / f["path"] for f in manifest["files"]])


def train_once(args: argparse.Namespace, setup: dict) -> dict[str, Any]:
    """One run in this process (one of `world` processes). Every process returns the same dict."""
    gpu = torch.cuda.is_available()
    world = dd.world()
    base = load_base(args)
    block = setup["model"]["block_size"]
    ops = OnePassDataset(base, block, setup["order_seed"])
    steps = args.steps or setup["steps"]
    out = Path(args.out)
    cfg = gb.make_cfg(
        setup["model"],
        ops.path,
        out / "ckpt",
        max_steps=steps,
        batch_size=args.micro,
        accum_steps=args.accum,
        lr=setup["lr"],
        warmup_steps=setup["warmup"],
        precision="fp16" if gpu else "fp32",
        compile=gpu,
        seed=setup["seed"],
        data_seed=setup["seed"],
        eval_interval=0,
        log_interval=50 if not args.smoke else 1000,
        save_interval=0,
        save_best=False,
        resume=args.resume or "",
    )
    cfg.optim.min_lr_ratio = setup["min_lr_ratio"]
    model = None
    if args.float64:  # the D-047 method: compare devices/process counts in float64
        set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
        model = GPT(cfg.model).double()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    tr = Trainer(cfg, ops, model=model)
    start = tr.state.step
    if args.micro * args.accum * world != setup["windows_per_step"]:
        raise ValueError(
            f"{args.micro} x {args.accum} x {world} processes != {setup['windows_per_step']} windows per step"
        )
    losses: list[float] = []
    marks: dict[int, float] = {}
    speed_from = min(setup["speed_from_step"], steps - 1)

    def callback(t: Trainer) -> str | None:
        s = t.state.step
        shares = dd.all_gather_object(t.last_loss)
        losses.append(sum(shares) / len(shares))  # each process's mean over its equal share
        if s in (speed_from, steps):
            if gpu:
                torch.cuda.synchronize()
            dd.barrier()
            marks[s] = time.time()

        def decide() -> str | None:
            if args.raise_at and s == args.raise_at:
                raise gp.StopRule(f"test stop rule at step {s}")
            if args.session_steps and s - start >= args.session_steps:
                return "session step limit (test)"
            return None

        return dd.run_on_main(decide)

    res: dict[str, Any] = {"world": world, "micro": args.micro, "accum": args.accum, "start_step": start}
    try:
        fit = tr.fit(should_stop=callback)
        res["end"] = fit["stopped"] or "done"
    except gp.StopRule as exc:
        res["stop_rule"] = str(exc)
    except FloatingPointError as exc:
        res["stop_rule"] = f"non-finite training loss ({gb._err(exc)})"
    res.update(
        steps=tr.state.step,
        losses=losses,
        skipped_steps=tr.state.skipped_steps,
        tokens_seen=tr.state.tokens_seen,
        sampler_position=ops.position,
    )
    if speed_from in marks and steps in marks and marks[steps] > marks[speed_from]:
        n = steps - speed_from
        res["speed_steps"] = f"{speed_from + 1}-{steps}"
        res["tokens_per_s"] = n * setup["windows_per_step"] * block / (marks[steps] - marks[speed_from])
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if gpu else None
    res["peak_mem_gb"] = dd.all_gather_object(peak)
    raw = tr.raw_model
    res["weights_sha256"] = _weights_sha(raw)
    if args.save_weights and dd.is_main():
        torch.save({k: v.detach().cpu() for k, v in raw.state_dict().items()}, out / "weights.pt")
    if not res.get("stop_rule") and not args.no_eval:
        amp = tr.spec.amp_dtype if tr.spec.amp else None
        res["scores"] = dd.run_on_main(
            lambda: gp.score_all(raw, base, block, setup["eval_batch"], tr.device, amp)
        )
    return res


def worker(args: argparse.Namespace) -> int:
    dd.init_from_env(timeout_minutes=10.0)
    setup = setup_for(args.smoke)
    try:
        res = train_once(args, setup)
        code = 0
    except Exception as exc:  # noqa: BLE001 - recorded for the driver
        traceback.print_exc()
        res, code = {"error": gb._err(exc), "world": dd.world(), "micro": args.micro, "accum": args.accum}, 1
    if dd.is_main():
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "worker.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    if code == 0:
        dd.barrier()
    dd.cleanup()
    return code


# ------------------------------------------------------------------ driver --
def worker_cmd(args: argparse.Namespace, procs: int, micro: int, accum: int, out: Path) -> list[str]:
    script = [str(SCRIPTS / "gpu_ddp_check.py"), "--worker", "--micro", str(micro), "--accum", str(accum)]
    script += ["--data-dir", str(args.data_dir), "--manifest", str(args.manifest), "--out", str(out)]
    if args.smoke:
        script.append("--smoke")
    if procs == 1:
        return [sys.executable, *script]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(procs),
        *script,
    ]


def run_worker(args: argparse.Namespace, setup: dict, procs: int, micro: int, accum: int, tag: str) -> dict:
    out = Path(args.scratch) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "worker.json").unlink(missing_ok=True)
    cmd = worker_cmd(args, procs, micro, accum, out)
    gb._say(f"run '{tag}': {' '.join(cmd[-14:])}")
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    if args.smoke:  # the CPU test must stay on CPU even on a GPU machine (Part 0 on Kaggle runs it)
        env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    with open(out / "worker.log", "w", encoding="utf-8") as log:
        try:
            code = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=setup["worker_timeout_min"] * 60,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -9
    tail = (out / "worker.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
    if (out / "worker.json").exists():
        res = json.loads((out / "worker.json").read_text(encoding="utf-8"))
    else:
        res = {"error": f"worker exit code {code} without a result", "micro": micro, "accum": accum}
    if code != 0 and not res.get("error"):
        res["error"] = f"worker exit code {code}"
    res["minutes"] = round((time.time() - t0) / 60, 2)
    if res.get("error"):
        res["log_tail"] = tail
        print("\n".join(tail), flush=True)
    if not args.smoke:
        res.pop("losses", None)  # 300 numbers per run; the smoke test compares them
    return res


def verdict(one: dict, two: dict, setup: dict, smoke: bool) -> dict[str, Any]:
    v: dict[str, Any] = {}
    ok_runs = not one.get("error") and not two.get("error") and one.get("scores") and two.get("scores")
    if not ok_runs:
        v.update(agreement="not measured (a run failed)", speed="not measured", memory="not measured")
        v.update(agreement_pass=False, speed_pass=False, memory_pass=False)
    else:
        a, b = one["scores"]["bpb_mean"], two["scores"]["bpb_mean"]
        diff = abs(a - b)
        lim = setup["max_skipped"] * setup["steps"]
        finite = math.isfinite(a) and math.isfinite(b)
        skips_ok = one["skipped_steps"] <= lim and two["skipped_steps"] <= lim
        v["agreement_pass"] = finite and skips_ok and diff <= setup["agree_bpb"]
        v["agreement"] = (
            f"{'PASS' if v['agreement_pass'] else 'FAIL'}: |{a:.4f} - {b:.4f}| = {diff:.4f} bpb "
            f"(limit {setup['agree_bpb']}); "
            f"skipped fp16 steps {one['skipped_steps']} / {two['skipped_steps']} "
            f"(limit {lim:g})"
        )
        t1, t2 = one.get("tokens_per_s"), two.get("tokens_per_s")
        if t1 and t2:
            ratio = t2 / t1
            v["speedup"] = round(ratio, 3)
            v["speed_pass"] = ratio >= setup["min_speedup"]
            v["speed"] = (
                f"{'PASS' if v['speed_pass'] else 'FAIL'}: {t2:,.0f} vs {t1:,.0f} tokens/s = {ratio:.2f}x "
                f"(at least {setup['min_speedup']}x)"
            )
        else:
            v["speed_pass"], v["speed"] = False, "not measured"
        peaks = [p for p in two.get("peak_mem_gb") or [] if p is not None]
        if peaks:
            v["memory_pass"] = max(peaks) < setup["mem_limit_gb"]
            v["memory"] = (
                f"{'PASS' if v['memory_pass'] else 'FAIL'}: peak {max(peaks):.2f} GB per GPU "
                f"(under {setup['mem_limit_gb']} GB)"
            )
        else:
            v["memory_pass"], v["memory"] = smoke, "not measured on CPU" if smoke else "not measured"
    passed = v["agreement_pass"] and v["speed_pass"] and v["memory_pass"]
    v["pass"] = passed
    v["result"] = (
        "PASS - two-GPU training may be proposed for the rest of EXP-043 (a separate decision)"
        if passed
        else "FAIL - EXP-043 continues on one GPU; nothing else changes"
    )
    if smoke:
        v["result"] = "SMOKE (no verdict): " + v["result"]
    if passed and v.get("speedup"):
        v["saving"] = (
            f"at {v['speedup']:.2f}x the rest of a run needs "
            f"{100 / v['speedup']:.0f}% of its one-GPU GPU-hours"
        )
    return v


def exp043_fingerprint() -> str | None:
    for f in sorted((ROOT / "evals" / "results" / "EXP-043").glob("session-*/summary.json")):
        fp = (json.loads(f.read_text(encoding="utf-8")).get("order") or {}).get("fingerprint")
        if fp:
            return fp
    return None


def driver(args: argparse.Namespace) -> int:
    setup = setup_for(args.smoke)
    rep = Report(Path(args.out), EXP_ID, args.smoke)
    rep.data.pop("runs", None)
    rep.data.update(
        schema=SCHEMA,
        setup=setup,
        precision="fp16 + GradScaler + torch.compile (D-047), DistributedDataParallel for two GPUs",
    )
    try:
        base = la.part0(
            args,
            rep,
            la.CHECK_SHAPE_SMOKE if args.smoke else la.CHECK_SHAPE_FULL,
            la.PLAN_SMOKE if args.smoke else la.PLAN_FULL,
            device_check=False,
            extra_tests=("tests/test_onepass_sampler.py", "tests/test_ddp.py"),
        )
        if base is None:
            return finish(rep)
        if not args.smoke and torch.cuda.device_count() < setup["gpus"]:
            rep.data["stopped"] = f"only {torch.cuda.device_count()} GPU(s) attached; the check needs two"
            return finish(rep)
        ops = OnePassDataset(base, setup["model"]["block_size"], setup["order_seed"])
        ref = exp043_fingerprint()
        fp = ops.fingerprint()
        rep.data["order"] = {
            "fingerprint": fp,
            "n_windows": ops.n_windows,
            "same_as_exp043": "unknown (no EXP-043 session committed)"
            if ref is None
            else ("yes" if ref == fp else "NO"),
        }
        rep.save()
        del ops, base
        one: dict[str, Any] = {}
        for micro, accum in setup["micro_options_one"]:
            one = run_worker(args, setup, 1, micro, accum, f"one-{micro}x{accum}")
            peak = (one.get("peak_mem_gb") or [None])[0]
            if not one.get("error") and (peak is None or peak < setup["mem_limit_gb"]):
                break
        rep.data["one"] = one
        rep.save()
        if one.get("error"):
            rep.data["stopped"] = f"the one-GPU run failed: {one['error']}"
            rep.data["verdict"] = verdict(one, {"error": "not run"}, setup, args.smoke)
            return finish(rep)
        micro = one["micro"]
        accum2 = setup["windows_per_step"] // (setup["gpus"] * micro)
        two = run_worker(args, setup, setup["gpus"], micro, accum2, f"two-{micro}x{accum2}")
        rep.data["two"] = two
        rep.data["verdict"] = verdict(one, two, setup, args.smoke)
        rep.data["complete"] = not two.get("error")
    except Exception as exc:  # noqa: BLE001 - keep the evidence gathered so far
        traceback.print_exc()
        rep.data["stopped"] = f"unexpected error: {gb._err(exc)}"
    return finish(rep)


def finish(rep: Report) -> int:
    rep.data["finished_at"] = gb._now()
    rep.save()
    print(render(rep.data), flush=True)
    return 0 if rep.data.get("complete") else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with the 13 EXP-037 .bin + .meta.json files")
    p.add_argument("--manifest", default=gb.DEFAULT_MANIFEST)
    p.add_argument("--out", default="out/gpu/EXP-044", help="results (driver) or this run's folder (worker)")
    p.add_argument("--scratch", default="out/gpu/EXP-044-scratch", help="the runs' working folders")
    p.add_argument("--smoke", action="store_true", help="tiny model and data (CPU test of the script)")
    p.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-data-hash", action="store_true", help=argparse.SUPPRESS)
    # one run (started by the driver, or by tests directly)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--micro", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--accum", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--resume", default="", help=argparse.SUPPRESS)
    p.add_argument("--session-steps", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--raise-at", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--float64", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--save-weights", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-eval", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    return worker(args) if args.worker else driver(args)


if __name__ == "__main__":
    sys.exit(main())
