#!/usr/bin/env python3
"""EXP-038 (step 10): first GPU training bring-up — correctness, speed/memory, first real-data run.

    python scripts/gpu_bringup.py --data /kaggle/input/.../hi.bin --out /kaggle/working/EXP-038 \\
        --scratch /tmp/exp038

Runs the pre-registered parts of EXP-038 (EXPERIMENTS.md) on the first CUDA GPU and writes
``summary.json``, ``SUMMARY.txt`` and ``samples.jsonl`` to ``--out`` (rewritten after every part,
so a session that is cut off still leaves the evidence gathered so far). Checkpoints and logs go to
``--scratch`` and are not published.

* Part 0: environment record; the data file must match its sha256 in the EXP-037 manifest; the
  model/trainer tests must pass (otherwise STOP).
* Part 1: correctness with pre-registered tolerances — CPU = GPU (fp32), fp16 vs fp32, checkpoint
  resume in a fresh process, ``torch.compile`` vs eager.
* Part 2: tokens/s, step time, peak memory and MFU for the S/M/L bring-up sizes.
* Part 3: model M on the training split of the data file for one pass or 90 minutes, whichever is
  first; validation curve, bits per byte, 5 sample completions (look only; no quality claim).

``--smoke`` shrinks everything so the whole script runs on a CPU in a minute (tests use it).
Single GPU only (no distributed training). ``--max-hours`` is a hard budget for the session.
"""

from __future__ import annotations

import os

# Deterministic cuBLAS for the Part 1 comparisons; must be set before CUDA is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from frontier_ai.config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from frontier_ai.data.dataset import TokenDataset  # noqa: E402
from frontier_ai.engine.trainer import Trainer  # noqa: E402

SCHEMA = "frontier-gpu-bringup-v1"
DEFAULT_MANIFEST = "evals/results/EXP-037/manifest.json"
EOT_ID = 32768
TOL_CPU_GPU = 1e-3  # max |loss difference| per step, fp32
TOL_REL_VAL = 0.02  # fp16 vs fp32 and compile vs eager: final validation loss, relative
TOL_SKIPPED = 0.05  # share of optimizer steps the fp16 scaler may skip
TOL_RESUME = 1e-5  # max |loss difference| per step after resume, fp32 deterministic
PEAK_TFLOPS = {  # dense tensor-core peaks from the vendor data sheets (no sparsity)
    "Tesla T4": {"fp16": 65.0, "fp32": 8.1},
    "Tesla P100-PCIE-16GB": {"fp16": 18.7, "fp32": 9.3},
}
PROMPTS = [
    "भारत की राजधानी",
    "आज सुबह से बारिश",
    "किसानों के लिए सरकार ने",
    "शिक्षा का महत्व",
    "एक समय की बात है,",
]

SIZES_FULL = {
    "S": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 128},
    "M": {"n_layer": 8, "n_head": 6, "n_embd": 384, "block_size": 512},
    "L": {"n_layer": 12, "n_head": 12, "n_embd": 768, "block_size": 1024},
}
SIZES_SMOKE = {
    "S": {"n_layer": 2, "n_head": 2, "n_embd": 32, "block_size": 32},
    "M": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64},
    "L": {"n_layer": 2, "n_head": 4, "n_embd": 96, "block_size": 64},
}
# steps and caps; the smoke plan only shrinks them
PLAN_FULL = {
    "cpu_gpu_steps": 50,
    "precision_steps": 300,
    "resume_steps": 200,
    "bench_warmup": 10,
    "bench_warmup_compile": 25,
    "bench_steps": 100,
    "run_minutes": 90.0,
    "run_eval_interval": 500,
    "run_eval_iters": 50,
    "batches": [64, 32, 16, 8, 4, 2, 1],
    "run_batch_cap": 32,
    "eval_iters": 20,
    "margin_minutes": 10.0,
}
PLAN_SMOKE = {
    "cpu_gpu_steps": 10,
    "precision_steps": 20,
    "resume_steps": 20,
    "bench_warmup": 2,
    "bench_warmup_compile": 2,
    "bench_steps": 5,
    "run_minutes": 0.5,
    "run_eval_interval": 10,
    "run_eval_iters": 2,
    "batches": [8, 4],
    "run_batch_cap": 8,
    "eval_iters": 2,
    "margin_minutes": 0.0,
}


# ------------------------------------------------------------------ helpers --
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _say(msg: str) -> None:
    print(f"[bringup {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def make_cfg(size: dict[str, int], data_path: Path, out_dir: Path, **train: Any) -> ExperimentConfig:
    """The D-043 architecture (EXP-B baseline) at a bring-up size, Frontier Tokenizer v2 vocab."""
    model = ModelConfig(
        vocab_size=32896,
        dropout=0.0,
        norm="rmsnorm",
        ffn="swiglu",
        pos="learned",
        tie_embeddings=True,
        **size,
    )
    optim = OptimConfig(
        lr=train.pop("lr", 3e-3),
        warmup_steps=train.pop("warmup_steps", 20),
        min_lr_ratio=0.1,
        weight_decay=0.1,
        grad_clip=1.0,
    )
    batch = train.pop("batch_size", 16)
    base = dict(
        out_dir=str(out_dir),
        accum_steps=1,
        eval_interval=0,
        eval_iters=20,
        log_interval=1,
        save_interval=0,
        save_best=False,
        device="auto",
        precision="fp32",
        seed=1337,
        deterministic=False,
        compile=False,
    )
    base.update(train)
    return ExperimentConfig(
        model=model,
        data=DataConfig(path=str(data_path), batch_size=batch, seed=1337),
        optim=optim,
        train=TrainConfig(**base),
    )


def reset_backend() -> None:
    """Undo the deterministic settings of a Part 1 run, so later timings are realistic."""
    torch.use_deterministic_algorithms(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    gc.collect()


@contextlib.contextmanager
def math_attention(enabled: bool):
    """Force the deterministic math attention kernel (Part 1 fp32 comparisons)."""
    if not enabled:
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
    except ImportError:  # older torch
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield


def read_log(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "train.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def step_losses(out_dir: Path) -> dict[int, float]:
    return {r["step"]: r["loss"] for r in read_log(out_dir) if r.get("event") == "train"}


def eval_points(out_dir: Path) -> list[dict[str, Any]]:
    return [r for r in read_log(out_dir) if r.get("event") == "eval"]


def run_fit(
    cfg: ExperimentConfig, ds: TokenDataset, *, math_attn: bool = False, schedule_steps: int | None = None
) -> dict[str, Any]:
    """Train with the repository Trainer; returns its result plus the per-step losses."""
    out = Path(cfg.train.out_dir)
    if out.exists():
        shutil.rmtree(out)
    with math_attention(math_attn):
        trainer = Trainer(cfg, ds)
        if schedule_steps is not None:
            trainer.scheduler.max_steps = schedule_steps
        result = trainer.fit()
    losses = step_losses(out)
    del trainer
    reset_backend()
    return {"result": result, "losses": losses, "evals": eval_points(out)}


def flops_per_token(model_cfg: ModelConfig, n_params: int) -> float:
    """6 N (weights, forward + backward) + 12 L d T (attention scores), N without position table."""
    n = n_params - (model_cfg.block_size * model_cfg.n_embd if model_cfg.pos == "learned" else 0)
    return 6.0 * n + 12.0 * model_cfg.n_layer * model_cfg.n_embd * model_cfg.block_size


def n_params_of(cfg: ExperimentConfig) -> int:
    from frontier_ai.model.gpt import GPT

    return GPT(cfg.model).n_params()


# ------------------------------------------------------------------ report --
class Report:
    def __init__(self, out: Path, exp_id: str, smoke: bool) -> None:
        self.out = out
        self.data: dict[str, Any] = {
            "schema": SCHEMA,
            "exp_id": exp_id,
            "smoke": smoke,
            "complete": False,
            "started_at": _now(),
            "parts": {},
        }
        self.samples: list[dict[str, Any]] = []
        self.t0 = time.time()

    def save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.data["elapsed_min"] = round((time.time() - self.t0) / 60, 1)
        tmp = self.out / "summary.json.tmp"
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        tmp.replace(self.out / "summary.json")
        (self.out / "SUMMARY.txt").write_text(render_text(self.data), encoding="utf-8", newline="\n")
        with open(self.out / "samples.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for s in self.samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def _fmt_pass(v: Any) -> str:
    return {True: "PASS", False: "FAIL", None: "n/a"}.get(v, str(v))


def render_text(d: dict[str, Any]) -> str:
    env = d.get("environment", {})
    lines = [
        f"{d['exp_id']}: GPU training bring-up (step 10)"
        + (" [SMOKE TEST, not a result]" if d.get("smoke") else ""),
        f"complete: {d['complete']}   started {d['started_at']}   finished {d.get('finished_at', '-')}   "
        f"elapsed {d.get('elapsed_min', 0)} min",
        f"device: {env.get('gpu_name', 'none')} ({env.get('gpu_memory_gb', '-')} GB)"
        f"   torch {env.get('torch', '-')}"
        f"   CUDA {env.get('cuda', '-')}   python {env.get('python', '-')}",
    ]
    if d.get("stopped"):
        lines.append(f"STOPPED: {d['stopped']}")
    p = d["parts"]
    if "part0" in p:
        p0 = p["part0"]
        lines += [
            "",
            "Part 0 - environment",
            f"  data file sha256 matches EXP-037 manifest: {_fmt_pass(p0.get('data_sha256_ok'))}",
            f"  model/trainer tests: {_fmt_pass(p0.get('tests_pass'))} ({p0.get('tests_line', '')})",
        ]
    if "part1" in p:
        lines += ["", "Part 1 - correctness (pre-registered tolerances)"]
        for key, label in (
            ("cpu_equals_gpu", "1. CPU = GPU (fp32)"),
            ("fp16_vs_fp32", "2. fp16 vs fp32"),
            ("resume", "3. checkpoint resume"),
            ("compile", "4. torch.compile vs eager"),
        ):
            r = p["part1"].get(key)
            if r is None:
                continue
            lines.append(f"  {label}: {_fmt_pass(r.get('pass'))}  {r.get('detail', '')}")
    if "part2" in p:
        steps = d.get("plan", {}).get("steps", {}).get("bench_steps", "?")
        lines += [
            "",
            f"Part 2 - speed and memory ({steps} timed steps after warm-up)",
            f"  {'model':<6}{'params':>12}{'precision':>16}{'batch':>7}{'tokens/s':>12}{'step ms':>9}"
            f"{'peak GB':>9}{'MFU':>7}  status",
        ]
        for r in p["part2"]:
            mfu = f"{100 * r['mfu']:.1f}%" if r.get("mfu") is not None else "-"
            tps = f"{r['tokens_per_s']:,.0f}" if r.get("tokens_per_s") else "-"
            step = f"{r['step_ms']:.0f}" if r.get("step_ms") else "-"
            peak = f"{r['peak_mem_gb']:.2f}" if r.get("peak_mem_gb") is not None else "-"
            lines.append(
                f"  {r['model']:<6}{r['n_params']:>12,}{r['precision']:>16}{r.get('batch') or '-':>7}"
                f"{tps:>12}{step:>9}{peak:>9}{mfu:>7}  {r['status']}"
            )
    if "part3" in p:
        r = p["part3"]
        lines += ["", "Part 3 - first real-data run (model M, Hindi training split)"]
        if r.get("status") != "done":
            lines.append(f"  status: {r.get('status')}")
        else:
            lines += [
                f"  steps {r['steps']:,} x {r['tokens_per_step']:,} tokens = {r['tokens_seen']:,} tokens "
                f"({r['share_of_one_pass']:.0%} of one pass) in {r['minutes']:.1f} min"
                f" ({r['tokens_per_s']:,.0f} tokens/s incl. evals)",
                f"  validation loss {r['initial_val_loss']:.3f} -> {r['final_val_loss']:.3f} nats/token; "
                f"{r['final_bits_per_byte']:.3f} bits per byte (not comparable to earlier models)",
                f"  no NaN/inf: {_fmt_pass(r['finite'])}"
                f"   final below initial: {_fmt_pass(r['val_decreased'])}"
                f"   fp16 skipped steps: {r['skipped_steps']}",
                "  validation curve (step: loss): "
                + ", ".join(
                    f"{e['step']}: {e['val_loss']:.3f}"
                    for e in r["val_curve"][:: max(1, len(r["val_curve"]) // 12)]
                ),
                "  5 sample completions are in samples.jsonl (for a look only; no quality claim).",
            ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ parts --
def part0(args: argparse.Namespace, rep: Report, data_path: Path) -> bool:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            gpu_name=props.name,
            gpu_memory_gb=round(props.total_memory / 1024**3, 2),
            compute_capability=f"{props.major}.{props.minor}",
            bf16_supported=torch.cuda.is_bf16_supported(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        )
        with contextlib.suppress(Exception):
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            env["driver"] = q.stdout.strip().splitlines()[0] if q.stdout.strip() else None
    with contextlib.suppress(Exception):
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        env["code_commit"] = head.stdout.strip() or None
    rep.data["environment"] = env
    p0: dict[str, Any] = {}
    rep.data["parts"]["part0"] = p0
    # data identity
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    entry = next((f for f in manifest["files"] if f["path"] == data_path.name), None)
    if entry is None:
        p0["data_sha256_ok"] = False
        rep.data["stopped"] = f"{data_path.name} is not listed in {args.manifest}"
        return False
    if args.skip_data_hash:
        p0["data_sha256_ok"] = None
    else:
        p0["data_sha256_ok"] = sha256_file(data_path) == entry["sha256"]
    p0["data"] = {
        "file": data_path.name,
        "language": entry["language"],
        "n_train": entry["n_train"],
        "n_val": entry["n_val"],
        "sha256": entry["sha256"],
        "packed_id": manifest.get("packed_id"),
    }
    rep.save()
    if p0["data_sha256_ok"] is False:
        rep.data["stopped"] = "the data file does not match the EXP-037 manifest"
        return False
    if not args.smoke and not torch.cuda.is_available():
        rep.data["stopped"] = "no CUDA GPU found (on Kaggle: the accelerator must be a GPU)"
        return False
    if args.skip_tests:
        p0["tests_pass"], p0["tests_line"] = None, "skipped"
    else:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "tests/test_model.py",
            "tests/test_engine.py",
            "tests/test_train_smoke.py",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
        p0["tests_pass"] = res.returncode == 0
        p0["tests_line"] = tail[-1] if tail else res.stderr.strip()[-200:]
        if res.returncode != 0:
            print(res.stdout[-4000:], res.stderr[-2000:], flush=True)
            rep.data["stopped"] = "the model/trainer tests failed on this machine"
            rep.save()
            return False
    rep.save()
    return True


def part1(args: argparse.Namespace, rep: Report, ds: TokenDataset, sizes: dict, plan: dict) -> dict[str, Any]:
    p1: dict[str, Any] = {}
    rep.data["parts"]["part1"] = p1
    scratch = Path(args.scratch) / "part1"
    data_path = Path(ds.path)
    gpu = torch.cuda.is_available()
    dev = "cuda" if gpu else "cpu"

    # 1. CPU = GPU
    _say("Part 1.1: CPU = GPU (model S, fp32)")
    try:
        n = plan["cpu_gpu_steps"]
        runs = {}
        for d in ("cpu", dev):
            cfg = make_cfg(
                sizes["S"],
                data_path,
                scratch / f"s-{d}",
                max_steps=n,
                device=d,
                precision="fp32",
                deterministic=True,
                eval_iters=plan["eval_iters"],
            )
            runs[d] = run_fit(cfg, ds, math_attn=True)["losses"]
        diffs = [abs(runs["cpu"][s] - runs[dev][s]) for s in range(1, n + 1)]
        p1["cpu_equals_gpu"] = {
            "pass": max(diffs) <= TOL_CPU_GPU,
            "steps": n,
            "max_abs_diff": max(diffs),
            "tolerance": TOL_CPU_GPU,
            "devices": ["cpu", dev],
            "detail": f"max |loss diff| {max(diffs):.2e} over {n} steps (tolerance {TOL_CPU_GPU})",
        }
    except Exception as exc:  # noqa: BLE001 - every failure is evidence
        p1["cpu_equals_gpu"] = {"pass": False, "error": _err(exc), "detail": f"error: {_err(exc)}"}
    rep.save()

    # 2. fp16 vs fp32
    _say("Part 1.2: fp16 autocast vs fp32 (model M)")
    n = plan["precision_steps"]
    try:
        res = {}
        for prec in ("fp32", "fp16"):
            cfg = make_cfg(
                sizes["M"],
                data_path,
                scratch / f"m-{prec}",
                max_steps=n,
                precision=prec,
                eval_interval=n,
                eval_iters=plan["eval_iters"],
            )
            res[prec] = run_fit(cfg, ds)
        v32, v16 = res["fp32"]["evals"][-1]["val_loss"], res["fp16"]["evals"][-1]["val_loss"]
        rel = abs(v16 - v32) / v32
        skipped = res["fp16"]["result"]["skipped_steps"]
        applicable = gpu
        ok = rel <= TOL_REL_VAL and skipped <= TOL_SKIPPED * n
        p1["fp16_vs_fp32"] = {
            "pass": ok if applicable else None,
            "steps": n,
            "val_fp32": v32,
            "val_fp16": v16,
            "rel_diff": rel,
            "skipped_steps": skipped,
            "tolerance_rel": TOL_REL_VAL,
            "tolerance_skipped": TOL_SKIPPED,
            "detail": (
                f"final val {v32:.4f} (fp32) vs {v16:.4f} (fp16), {100 * rel:.2f}% apart; "
                f"{skipped} of {n} steps skipped"
                if applicable
                else "not applicable without a GPU (fp16 autocast is CUDA-only)"
            ),
        }
    except FloatingPointError as exc:
        p1["fp16_vs_fp32"] = {"pass": False, "error": _err(exc), "detail": f"non-finite loss: {_err(exc)}"}
    except Exception as exc:  # noqa: BLE001
        p1["fp16_vs_fp32"] = {"pass": False, "error": _err(exc), "detail": f"error: {_err(exc)}"}
    rep.save()

    # 3. checkpoint + resume in a fresh process
    _say("Part 1.3: checkpoint and resume in a fresh process (model M, fp32, deterministic)")
    n = plan["resume_steps"]
    half = n // 2
    try:
        common = dict(max_steps=n, precision="fp32", deterministic=True, eval_iters=plan["eval_iters"])
        straight = run_fit(
            make_cfg(sizes["M"], data_path, scratch / "resume-straight", **common), ds, math_attn=True
        )["losses"]
        first_cfg = make_cfg(sizes["M"], data_path, scratch / "resume-split", save_interval=half, **common)
        half_cfg = copy.deepcopy(first_cfg)
        half_cfg.train.max_steps = half
        run_fit(half_cfg, ds, math_attn=True, schedule_steps=n)
        resume_cfg = copy.deepcopy(first_cfg)
        resume_cfg.train.init_from = "resume"
        cfg_path = scratch / "resume-child.json"
        resume_cfg.save(cfg_path)
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child-resume", str(cfg_path)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if child.returncode != 0:
            raise RuntimeError(f"resume process failed: {child.stderr.strip()[-500:]}")
        resumed = step_losses(scratch / "resume-split")
        diffs = [abs(straight[s] - resumed[s]) for s in range(half + 1, n + 1)]
        p1["resume"] = {
            "pass": max(diffs) <= TOL_RESUME,
            "steps": n,
            "resumed_at": half,
            "max_abs_diff": max(diffs),
            "tolerance": TOL_RESUME,
            "detail": f"steps {half + 1}-{n}: max |loss diff| {max(diffs):.2e} (tolerance {TOL_RESUME})",
        }
    except Exception as exc:  # noqa: BLE001
        p1["resume"] = {"pass": False, "error": _err(exc), "detail": f"error: {_err(exc)}"}
    rep.save()

    # 4. torch.compile vs eager
    _say("Part 1.4: torch.compile vs eager (model M, fp16)")
    n = plan["precision_steps"]
    if not gpu:
        p1["compile"] = {
            "pass": None,
            "available": False,
            "detail": "not applicable without a GPU (the trainer compiles on CUDA only)",
        }
    else:
        try:
            res = {}
            for comp in (False, True):
                cfg = make_cfg(
                    sizes["M"],
                    data_path,
                    scratch / f"m-compile-{comp}",
                    max_steps=n,
                    precision="fp16",
                    eval_interval=n,
                    eval_iters=plan["eval_iters"],
                    compile=comp,
                )
                t0 = time.time()
                res[comp] = run_fit(cfg, ds)
                res[comp]["seconds"] = time.time() - t0
            ve, vc = res[False]["evals"][-1]["val_loss"], res[True]["evals"][-1]["val_loss"]
            rel = abs(vc - ve) / ve
            p1["compile"] = {
                "pass": rel <= TOL_REL_VAL,
                "available": True,
                "val_eager": ve,
                "val_compile": vc,
                "rel_diff": rel,
                "seconds_eager": round(res[False]["seconds"], 1),
                "seconds_compile": round(res[True]["seconds"], 1),
                "tolerance_rel": TOL_REL_VAL,
                "detail": f"final val {ve:.4f} (eager) vs {vc:.4f} (compiled), {100 * rel:.2f}% apart",
            }
        except Exception as exc:  # noqa: BLE001 - compile problems are recorded, not fatal
            p1["compile"] = {
                "pass": None,
                "available": False,
                "error": _err(exc),
                "detail": f"torch.compile did not work here ({_err(exc)}); eager mode is used",
            }
    rep.save()
    shutil.rmtree(scratch, ignore_errors=True)
    return p1


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ''}"


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def part2(
    args: argparse.Namespace,
    rep: Report,
    ds: TokenDataset,
    sizes: dict,
    plan: dict,
    compile_ok: bool,
    deadline: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rep.data["parts"]["part2"] = rows
    gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_properties(0).name if gpu else "cpu"
    peaks = PEAK_TFLOPS.get(gpu_name)
    scratch = Path(args.scratch) / "part2"
    variants = [("fp32", False), ("fp16", False)] + ([("fp16", True)] if compile_ok else [])
    for name in ("S", "M", "L"):
        for prec, comp in variants:
            base = make_cfg(sizes[name], Path(ds.path), scratch / "run", precision=prec, compile=comp)
            row: dict[str, Any] = {
                "model": name,
                **sizes[name],
                "n_params": n_params_of(base),
                "precision": prec + (" + compile" if comp else ""),
                "status": "not run",
            }
            rows.append(row)
            if time.time() > deadline:
                row["status"] = "skipped (time budget)"
                continue
            _say(f"Part 2: {name} {row['precision']}")
            warm = plan["bench_warmup_compile"] if comp else plan["bench_warmup"]
            for batch in plan["batches"]:
                cfg = make_cfg(
                    sizes[name],
                    Path(ds.path),
                    scratch / "run",
                    precision=prec,
                    compile=comp,
                    batch_size=batch,
                    max_steps=warm + plan["bench_steps"],
                )
                failure = None
                try:
                    if gpu:
                        torch.cuda.reset_peak_memory_stats()
                    out = run_fit(cfg, ds)
                except Exception as exc:  # noqa: BLE001
                    failure = ("oom" if _is_oom(exc) else "error", _err(exc))
                if failure is not None:
                    # outside the except block, so the traceback (and the tensors it holds) is released
                    reset_backend()
                    if failure[0] == "oom":
                        row["status"] = f"out of memory at batch {batch}"
                        continue
                    row["status"] = f"error: {failure[1]}"
                    break
                tokens_per_step = batch * sizes[name]["block_size"]
                recs = [
                    r
                    for r in read_log(Path(cfg.train.out_dir))
                    if r.get("event") == "train" and r["step"] > warm and r.get("tps")
                ]
                seconds = sum(tokens_per_step / r["tps"] for r in recs)
                tps = len(recs) * tokens_per_step / seconds if seconds > 0 else None
                row.update(
                    batch=batch,
                    tokens_per_step=tokens_per_step,
                    timed_steps=len(recs),
                    tokens_per_s=tps,
                    step_ms=1000 * seconds / len(recs) if recs else None,
                    peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 3) if gpu else None,
                    flops_per_token=flops_per_token(cfg.model, row["n_params"]),
                    status="ok",
                    final_loss=out["losses"].get(max(out["losses"])) if out["losses"] else None,
                )
                peak = (peaks or {}).get(prec)
                row["mfu"] = (tps * row["flops_per_token"] / (peak * 1e12)) if (peak and tps) else None
                row["peak_tflops"] = peak
                break
            rep.save()
    shutil.rmtree(scratch, ignore_errors=True)
    return rows


def part3(
    args: argparse.Namespace,
    rep: Report,
    ds: TokenDataset,
    sizes: dict,
    plan: dict,
    bench: list[dict[str, Any]],
    compile_ok: bool,
    deadline: float,
) -> None:
    r: dict[str, Any] = {"status": "not run"}
    rep.data["parts"]["part3"] = r
    use_compile = compile_ok
    ref = next(
        (
            b
            for b in bench
            if b["model"] == "M"
            and b["precision"] == ("fp16 + compile" if use_compile else "fp16")
            and b["status"] == "ok"
        ),
        None,
    )
    if ref is None and use_compile:
        use_compile = False
        ref = next(
            (b for b in bench if b["model"] == "M" and b["precision"] == "fp16" and b["status"] == "ok"), None
        )
    if ref is None:
        r["status"] = "skipped: no Part 2 measurement for model M in fp16"
        rep.save()
        return
    batch = min(ref["batch"], plan["run_batch_cap"])
    tokens_per_step = batch * sizes["M"]["block_size"]
    tps = ref["tokens_per_s"] * batch / ref["batch"]  # conservative: smaller batches are not faster per token
    minutes = min(plan["run_minutes"], (deadline - time.time()) / 60 - plan["margin_minutes"])
    if minutes < min(2.0, plan["run_minutes"]):
        r["status"] = "skipped (time budget)"
        rep.save()
        return
    one_pass = math.ceil(ds.n_train / tokens_per_step)
    by_time = max(1, int(0.9 * minutes * 60 * tps / tokens_per_step))
    steps = min(one_pass, by_time)
    _say(
        f"Part 3: model M, fp16{' + compile' if use_compile else ''}, batch {batch}, {steps:,} steps "
        f"(one pass = {one_pass:,}; time cap {minutes:.0f} min)"
    )
    out_dir = Path(args.scratch) / "part3"
    cfg = make_cfg(
        sizes["M"],
        Path(ds.path),
        out_dir,
        precision="fp16",
        compile=use_compile,
        batch_size=batch,
        max_steps=steps,
        lr=1e-3,
        warmup_steps=min(200, max(1, steps // 10)),
        eval_interval=min(plan["run_eval_interval"], steps),
        eval_iters=plan["run_eval_iters"],
        save_best=True,
        log_interval=50,
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t0 = time.time()
    finite = True
    try:
        trainer = Trainer(cfg, ds)
        initial = trainer.evaluate()
        result = trainer.fit()
    except FloatingPointError as exc:
        finite = False
        r.update(status="failed: non-finite loss", error=_err(exc))
        rep.save()
        return
    seconds = time.time() - t0
    evals = eval_points(out_dir)
    final = evals[-1]["val_loss"] if evals else float("nan")
    report = trainer.loss_report(final)
    r.update(
        status="done",
        model="M",
        precision="fp16" + (" + compile" if use_compile else ""),
        batch=batch,
        tokens_per_step=tokens_per_step,
        steps=result["steps"],
        tokens_seen=trainer.state.tokens_seen,
        one_pass_steps=one_pass,
        share_of_one_pass=result["steps"] / one_pass,
        minutes=seconds / 60,
        tokens_per_s=trainer.state.tokens_seen / seconds,
        lr=cfg.optim.lr,
        warmup=cfg.optim.warmup_steps,
        initial_val_loss=initial,
        final_val_loss=final,
        final_bits_per_byte=report["bits_per_byte"],
        best_val_loss=result["best_val"],
        skipped_steps=result["skipped_steps"],
        finite=finite,
        val_decreased=final < initial,
        val_curve=[{"step": e["step"], "val_loss": e["val_loss"]} for e in evals],
        n_params=trainer.model.n_params() if hasattr(trainer.model, "n_params") else ref["n_params"],
    )
    # sample completions (for a look only)
    from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

    tok = load_frontier_tokenizer_v2()
    model = getattr(trainer.model, "_orig_mod", trainer.model)
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(38)
    for i, prompt in enumerate(PROMPTS):
        ids = tok.encode_ordinary(prompt)
        with torch.no_grad():
            out = model.generate(
                torch.tensor([ids], dtype=torch.long, device=device),
                max_new_tokens=100,
                temperature=0.8,
                top_k=40,
                generator=gen,
                stop_at_eos=EOT_ID,
            )
        new = [t for t in out[0].tolist()[len(ids) :] if t != EOT_ID]
        rep.samples.append(
            {
                "id": i,
                "prompt": prompt,
                "completion": tok.decode(new),
                "new_tokens": len(new),
                "sampling": {"temperature": 0.8, "top_k": 40, "seed": 38, "max_new_tokens": 100},
            }
        )
    del trainer, model
    reset_backend()
    rep.save()


def child_resume(cfg_path: str) -> int:
    cfg = ExperimentConfig.load(cfg_path)
    ds = TokenDataset(cfg.data.path)
    with math_attention(True):
        Trainer(cfg, ds).fit()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="the EXP-037 token file (e.g. hi.bin; its .meta.json next to it)")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--out", default="out/gpu/EXP-038")
    p.add_argument("--scratch", default="out/gpu/EXP-038-scratch")
    p.add_argument("--exp-id", default="EXP-038")
    p.add_argument("--max-hours", type=float, default=5.5, help="hard time budget for the whole session")
    p.add_argument("--smoke", action="store_true", help="tiny sizes and steps (CPU test of the script)")
    p.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-data-hash", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--child-resume", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.child_resume:
        return child_resume(args.child_resume)
    if not args.data:
        p.error("--data is required")
    deadline = time.time() + args.max_hours * 3600
    sizes, plan = (SIZES_SMOKE, PLAN_SMOKE) if args.smoke else (SIZES_FULL, PLAN_FULL)
    rep = Report(Path(args.out), args.exp_id, args.smoke)
    rep.data["plan"] = {
        "sizes": sizes,
        "steps": plan,
        "max_hours": args.max_hours,
        "tolerances": {
            "cpu_equals_gpu": TOL_CPU_GPU,
            "rel_val": TOL_REL_VAL,
            "skipped": TOL_SKIPPED,
            "resume": TOL_RESUME,
        },
        "architecture": "D-043 (EXP-B baseline: learned positions, SwiGLU, RMSNorm, tied embeddings)",
        "tokenizer": "frontier-tokenizer-v2 (vocab 32,896)",
    }
    data_path = Path(args.data)
    try:
        if not part0(args, rep, data_path):
            rep.data["finished_at"] = _now()
            rep.save()
            print(render_text(rep.data), flush=True)
            return 1
        ds = TokenDataset(data_path)
        if ds.meta.vocab_size != 32896:
            rep.data["stopped"] = f"vocab {ds.meta.vocab_size} in {data_path.name}.meta.json, expected 32896"
            rep.save()
            return 1
        p1 = part1(args, rep, ds, sizes, plan)
        compile_ok = bool(p1.get("compile", {}).get("pass"))
        bench = part2(args, rep, ds, sizes, plan, compile_ok, deadline)
        part3(args, rep, ds, sizes, plan, bench, compile_ok, deadline)
        rep.data["complete"] = (
            all(k in rep.data["parts"] for k in ("part0", "part1", "part2", "part3"))
            and rep.data["parts"]["part3"].get("status") == "done"
        )
    except Exception as exc:  # noqa: BLE001 - keep the evidence gathered so far
        traceback.print_exc()
        rep.data["stopped"] = f"unexpected error: {_err(exc)}"
    rep.data["finished_at"] = _now()
    rep.save()
    print(render_text(rep.data), flush=True)
    return 0 if rep.data["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
