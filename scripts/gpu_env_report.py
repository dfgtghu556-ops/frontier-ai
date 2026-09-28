#!/usr/bin/env python3
"""GPU machine readiness report (read-only): is this computer ready for frontier-ai GPU work?

    python scripts/gpu_env_report.py            # writes out/gpu_env/REPORT.txt + report.json

Written for a new collaborator's machine (see GPU_COLLABORATOR_START_HERE.md). It records the
facts the step-10 GPU bring-up plan needs and checks that the GPU path of THIS repository works:

1. machine: OS, Python, CPU cores, RAM, free disk (no hostname, user name or environment
   variables, the same privacy policy as the experiment records);
2. PyTorch: version, whether it is a CUDA build, the CUDA runtime it ships;
3. every visible NVIDIA GPU: name, memory, compute capability, bf16 support, plus
   ``nvidia-smi`` driver facts when the tool exists;
4. smoke checks on the first GPU (seconds, random numbers only, no data, nothing saved):
   a float32 matmul against the CPU, an autocast matmul, fused attention, and 10 training
   steps of this repo's tiny model (``configs/cpu_smoke.json`` shape) with the precision the
   trainer would pick.

It never modifies the repository, installs nothing and trains on no real data. Numbers
it prints are measurements of this machine only; they are not benchmarks.

Exit codes: 0 = ready (CUDA GPU found and every smoke check passed); 1 = not ready (the
report says why); 2 = the report itself failed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 2**30, 1)
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2**20, 1)
    except Exception:  # noqa: BLE001 - a missing fact is reported as null, never guessed
        return None
    return None


def _cmd(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30, cwd=REPO_ROOT)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def machine_facts() -> dict:
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor() or None,
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "disk_free_gb": round(disk.free / 2**30, 1),
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _cmd(["git", "rev-parse", "--short", "HEAD"]),
    }


def torch_facts() -> dict:
    import torch

    facts = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
             "cuda_available": torch.cuda.is_available(), "gpus": []}
    if facts["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            facts["gpus"].append({
                "index": i, "name": p.name, "memory_total_gb": round(total / 2**30, 2),
                "memory_free_gb": round(free / 2**30, 2), "compute_capability": f"{p.major}.{p.minor}",
                "multiprocessors": p.multi_processor_count,
            })
        facts["bf16_supported"] = torch.cuda.is_bf16_supported()
    smi = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv,noheader"])
    facts["nvidia_smi"] = smi.splitlines() if smi else None
    return facts


def smoke_checks(device: str = "cuda:0") -> list[dict]:
    """Seconds-long checks on GPU 0. Each check is independent; a failure is recorded, not raised.
    ``device="cpu"`` runs the same code on the CPU (used by the tests; the sandbox has no GPU)."""
    import torch
    import torch.nn.functional as F

    from frontier_ai.config import ExperimentConfig
    from frontier_ai.engine import checkpoint as ckpt
    from frontier_ai.utils.device import resolve_spec

    dev = torch.device(device)
    cuda = dev.type == "cuda"
    spec = resolve_spec(dev.type, "auto")
    sync = (lambda: torch.cuda.synchronize(dev)) if cuda else (lambda: None)
    checks: list[dict] = []

    def check(name: str, fn) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn()
            sync()
            checks.append({"check": name, "status": "PASS", "detail": detail,
                           "seconds": round(time.perf_counter() - t0, 3)})
        except Exception as exc:  # noqa: BLE001
            checks.append({"check": name, "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})

    def matmul_fp32() -> str:
        g = torch.Generator().manual_seed(0)
        a, b = torch.randn(512, 512, generator=g), torch.randn(512, 512, generator=g)
        ref = a.double() @ b.double()
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 would (correctly) lose precision
        try:
            got = (a.to(dev) @ b.to(dev)).double().cpu()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        err = float((got - ref).abs().max())
        if not err < 1e-2:
            raise AssertionError(f"GPU float32 matmul differs from float64 reference by {err:.3g}")
        return f"max |diff| vs float64 reference {err:.2e}"

    def autocast_matmul() -> str:
        a = torch.randn(1024, 1024, device=dev)
        with torch.autocast(dev.type, dtype=spec.amp_dtype, enabled=spec.amp):
            out = a @ a
        if not torch.isfinite(out).all():
            raise AssertionError("non-finite autocast result")
        used = str(spec.amp_dtype).replace("torch.", "") if spec.amp else "float32 (no autocast)"
        return f"precision the trainer picks here: {used}"

    def fused_attention() -> str:
        dtype = spec.amp_dtype if spec.amp else torch.float32
        q = torch.randn(2, 4, 256, 64, device=dev, dtype=dtype)
        out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        flash = "n/a (not a GPU)"
        if cuda:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    F.scaled_dot_product_attention(q, q, q, is_causal=True)
                flash = True
            except Exception:  # noqa: BLE001 - flash kernel unavailable on this GPU / build
                flash = False
        if not torch.isfinite(out.float()).all():
            raise AssertionError("non-finite attention output")
        return f"scaled_dot_product_attention OK; flash-attention kernel available: {flash}"

    def tiny_training() -> str:
        cfg = ExperimentConfig.load(REPO_ROOT / "configs" / "cpu_smoke.json").with_overrides(
            ["model.vocab_size=256"])
        torch.manual_seed(0)
        model = ckpt.build_model_from_config(cfg, dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(dev.type, enabled=spec.amp and spec.amp_dtype == torch.float16)
        bs, t = 8, cfg.model.block_size
        # one fixed batch, so a falling loss shows that the model really learns (memorises it)
        x = torch.randint(0, 256, (bs, t + 1), device=dev, generator=torch.Generator(dev).manual_seed(0))
        losses = []
        for _ in range(10):
            with torch.autocast(dev.type, dtype=spec.amp_dtype, enabled=spec.amp):
                loss = model(x[:, :-1], targets=x[:, 1:]).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        if not all(map(lambda v: v == v and abs(v) != float("inf"), losses)):
            raise AssertionError(f"non-finite loss: {losses}")
        if not losses[-1] < losses[0]:
            raise AssertionError(f"loss did not fall on a repeated batch: "
                                 f"{losses[0]:.3f} -> {losses[-1]:.3f}")
        peak = f", peak GPU memory {torch.cuda.max_memory_allocated(dev) / 2**20:.0f} MiB" if cuda else ""
        return (f"10 steps of the {sum(p.numel() for p in model.parameters()):,}-parameter smoke model, "
                f"loss {losses[0]:.3f} -> {losses[-1]:.3f}{peak}")

    check("float32 matmul matches CPU reference", matmul_fp32)
    check("mixed-precision (autocast) matmul", autocast_matmul)
    check("fused attention (scaled_dot_product_attention)", fused_attention)
    check("tiny model training steps", tiny_training)
    return checks


def verdict(tf: dict, checks: list[dict]) -> tuple[bool, list[str]]:
    reasons = []
    if tf.get("cuda_build") is None:
        reasons.append("PyTorch is a CPU-only build: reinstall it with the CUDA command from "
                       "pytorch.org (see GPU_COLLABORATOR_START_HERE.md, step 4).")
    elif not tf.get("cuda_available"):
        reasons.append("PyTorch has CUDA but sees no GPU: install/update the NVIDIA driver, "
                       "then check that `nvidia-smi` works.")
    reasons += [f"smoke check failed: {c['check']} ({c['detail']})" for c in checks if c["status"] != "PASS"]
    return not reasons, reasons


def render(report: dict) -> str:
    m, tf = report["machine"], report["torch"]
    lines = ["FRONTIER-AI GPU MACHINE REPORT", f"created {report['created_at']}", "",
             "MACHINE"]
    lines += [f"  {k}: {v}" for k, v in m.items()]
    lines += ["", "PYTORCH / GPU",
              f"  torch: {tf['torch']}   CUDA build: {tf['cuda_build'] or 'no (CPU-only build)'}   "
              f"CUDA available: {tf['cuda_available']}"]
    for g in tf["gpus"]:
        lines.append(f"  GPU {g['index']}: {g['name']} | {g['memory_total_gb']} GB total, "
                     f"{g['memory_free_gb']} GB free | compute capability {g['compute_capability']}")
    if "bf16_supported" in tf:
        lines.append(f"  bf16 supported: {tf['bf16_supported']}")
    lines.append(f"  nvidia-smi: {'; '.join(tf['nvidia_smi']) if tf['nvidia_smi'] else 'not found'}")
    if report["checks"]:
        lines += ["", f"SMOKE CHECKS (device {report['checks_device']}; random numbers only, nothing saved)"]
        for c in report["checks"]:
            secs = f" [{c['seconds']} s]" if "seconds" in c else ""
            lines.append(f"  {c['status']}  {c['check']}: {c['detail']}{secs}")
    lines += ["", "VERDICT: " + ("READY for GPU work" if report["ready"] else "NOT READY")]
    lines += [f"  - {r}" for r in report["reasons"]]
    lines += ["", "Send this whole file to the founder (it contains no names, passwords or paths)."]
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a legacy Windows console code page
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="out/gpu_env", help="output dir (default %(default)s, git-ignored)")
    p.add_argument("--checks-device", default=None, help=argparse.SUPPRESS)  # tests: run checks on cpu
    args = p.parse_args()
    try:
        tf = torch_facts()
        if args.checks_device:
            checks = smoke_checks(args.checks_device)
        else:
            checks = smoke_checks() if tf["cuda_available"] else []
        ready, reasons = verdict(tf, checks)
        report = {"schema": "frontier-gpu-env-report-v1",
                  "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                  "machine": machine_facts(), "torch": tf, "checks": checks,
                  "checks_device": args.checks_device or ("cuda:0" if checks else None),
                  "ready": ready, "reasons": reasons}
        out = Path(args.out)
        out = out if out.is_absolute() else REPO_ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        text = render(report)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (out / "REPORT.txt").write_text(text + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu-env] the report itself failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(text)
    print(f"\n[gpu-env] written to {out / 'REPORT.txt'}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
