"""GPU machine readiness report: runs on a CPU-only machine, checks run on CPU, privacy, verdicts."""

from __future__ import annotations

import getpass
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gpu_env_report.py"
_spec = importlib.util.spec_from_file_location("gpu_env_report", SCRIPT)
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)


def _run(*args: str):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=300)


def test_report_on_a_machine_without_gpu_is_written_and_not_ready(tmp_path):
    proc = _run("--out", str(tmp_path))
    assert proc.returncode == 1, proc.stdout + proc.stderr  # this sandbox has no CUDA GPU
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "frontier-gpu-env-report-v1"
    assert report["ready"] is False and report["reasons"] and report["checks"] == []
    assert report["machine"]["cpu_logical_cores"] >= 1 and report["machine"]["disk_free_gb"] > 0
    text = (tmp_path / "REPORT.txt").read_text(encoding="utf-8")
    assert "VERDICT: NOT READY" in text and text.isascii()
    # privacy: same policy as the experiment records (no host or user names)
    for secret in {getpass.getuser(), platform.node()} - {""}:
        assert secret not in text and secret not in json.dumps(report)


def test_smoke_checks_pass_on_cpu():
    checks = ge.smoke_checks("cpu")
    assert [c["check"] for c in checks] == [
        "float32 matmul matches CPU reference", "mixed-precision (autocast) matmul",
        "fused attention (scaled_dot_product_attention)", "tiny model training steps"]
    assert all(c["status"] == "PASS" for c in checks), checks
    assert "->" in checks[-1]["detail"]  # the loss fell on the repeated batch (checked inside)


def test_verdict_explains_what_to_fix():
    ok, why = ge.verdict({"cuda_build": None, "cuda_available": False}, [])
    assert not ok and "CPU-only build" in why[0] and "pytorch.org" in why[0]
    ok, why = ge.verdict({"cuda_build": "13.0", "cuda_available": False}, [])
    assert not ok and "nvidia-smi" in why[0]
    ok, why = ge.verdict({"cuda_build": "13.0", "cuda_available": True},
                         [{"check": "x", "status": "FAIL", "detail": "boom"}])
    assert not ok and "x (boom)" in why[0]
    assert ge.verdict({"cuda_build": "13.0", "cuda_available": True},
                      [{"check": "x", "status": "PASS", "detail": ""}]) == (True, [])
