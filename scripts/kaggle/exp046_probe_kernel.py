"""EXP-046 Kaggle kernel, step 1: the 15-minute probe of a Kaggle CPU session (no GPU).

Do not run this file directly. ``scripts/run_kaggle_exp046_probe.ps1`` copies it to ``run.py``,
replaces the commit placeholder with the commit being tested and pushes it as a private Kaggle
script kernel (CPU only, internet on, the private 13-language token dataset attached so the probe
can also see how Kaggle mounts it). It reads nothing from EXP-043 and uses no GPU quota.

It PROCESSES NO DATA (EXP-046 as approved on 2026-10-04). It only measures what the slice-2 build
plan depends on, all marked NOT VERIFIED in EXP-046 until now:

* CPU count and model, RAM, and free disk space of every relevant folder (``/kaggle/working`` is
  the 20 GB output; ``/tmp`` is scratch);
* whether the build's Python dependencies import here (numpy, pyarrow, the frozen Tokenizer v2);
* how long ONE pinned slice-2 file takes to download from Hugging Face and to hash: the first file
  of ``corpora/frontier/v2/sangraha_slice2.json`` (asm/data-1, 326 MB), downloaded into ``/tmp``
  with the repository's own resumable, SHA-256-checked downloader, then deleted (Sangraha files
  are never written to the kernel output);
* the file's row count, read from the parquet footer only (no document is read);
* the size of the attached EXP-037 token files (slice 2 is deduplicated against them later).

Writes ``/kaggle/working/EXP-046/probe/summary.json`` and ``SUMMARY.txt`` (the names the shared
runner expects). Always exits 0: the files carry the outcome. No credentials are needed or used.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = "https://github.com/dfgtghu556-ops/frontier-ai.git"
COMMIT = "__PINNED_COMMIT__"
SRC = Path("/tmp/frontier-ai")
OUT = Path("/kaggle/working/EXP-046/probe")
INPUT = Path("/kaggle/input")
SCRATCH = Path("/tmp/sangraha")
PINS = "corpora/frontier/v2/sangraha_slice2.json"
MANIFEST = "evals/results/EXP-037/manifest.json"
PROBE_SCHEMA = "frontier-exp046-probe-v1"


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def machine() -> dict:
    info: dict = {"python": sys.version.split()[0], "platform": platform.platform()}
    info["cpu_count_os"] = os.cpu_count()
    try:
        info["cpu_count_usable"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        info["cpu_count_usable"] = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, val = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                mem[key] = int(val.split()[0]) * 1024
        info["ram_total_bytes"] = mem.get("MemTotal")
        info["ram_available_bytes"] = mem.get("MemAvailable")
    except (OSError, ValueError):
        pass
    try:  # a cgroup memory limit can be lower than what /proc/meminfo shows
        for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            if Path(p).is_file():
                info["cgroup_memory_limit"] = Path(p).read_text().strip()
                break
    except OSError:
        pass
    info["gpu_visible"] = shutil.which("nvidia-smi") is not None
    return info


def disks() -> dict:
    out = {}
    for p in ("/kaggle/working", "/tmp", "/kaggle/input", "/"):
        try:
            u = shutil.disk_usage(p)
            out[p] = {"total_bytes": u.total, "free_bytes": u.free}
        except OSError as exc:
            out[p] = {"error": str(exc)}
    try:  # which folders share one device (scratch and output may be the same disk)
        out["df"] = subprocess.run(
            ["df", "-B1", "/kaggle/working", "/tmp", "/kaggle/input"], capture_output=True, text=True
        ).stdout
    except OSError:
        pass
    return out


def dependencies() -> dict:
    out: dict = {}
    for name in ("numpy", "pyarrow"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception as exc:  # noqa: BLE001 - report, do not fail
            out[name] = f"IMPORT FAILED: {exc}"
    try:
        from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

        tok = load_frontier_tokenizer_v2(SRC / "tokenizers" / "frontier-tokenizer-v2")
        ids = tok.encode_ordinary("\u0928\u092e\u0938\u094d\u0924\u0947 Hello world.")
        out["tokenizer_v2"] = f"loaded and hash-verified; sample sentence -> {len(ids)} tokens"
    except Exception as exc:  # noqa: BLE001
        out["tokenizer_v2"] = f"FAILED: {exc}"
    return out


def token_dataset() -> dict:
    names = [f["path"] for f in json.loads((SRC / MANIFEST).read_text(encoding="utf-8"))["files"]]
    dirs = sorted({p.parent for p in INPUT.rglob("*.bin")}) if INPUT.is_dir() else []
    found = [d for d in dirs if all((d / n).exists() for n in names)]
    res: dict = {"folders_with_all_files": [str(d) for d in found]}
    if len(found) == 1:
        res["total_bytes"] = sum((found[0] / n).stat().st_size for n in names)
        res["files"] = len(names)
    return res


def download_one() -> dict:
    from frontier_ai.corpus.sangraha import download_verified, load_slice_pins, parquet_row_count, sha256_file

    _, files = load_slice_pins(SRC / PINS)
    pf = files[0]
    res: dict = {"source_id": pf.source_id, "size_bytes": pf.size, "sha256_pinned": pf.sha256}
    lines: list[str] = []
    t0 = time.monotonic()
    res["result"] = download_verified(pf, SCRATCH, log=lambda m: (lines.append(m), print(m, flush=True)))
    res["download_and_verify_seconds"] = round(time.monotonic() - t0, 1)
    path = pf.local_path(SCRATCH)
    t1 = time.monotonic()
    res["sha256_again"] = sha256_file(path) == pf.sha256
    res["hash_seconds"] = round(time.monotonic() - t1, 1)
    seconds = max(res["download_and_verify_seconds"], 1e-9)
    res["download_MB_per_s_incl_hash"] = round(pf.size / 1e6 / seconds, 1)
    t2 = time.monotonic()
    res["parquet_rows_from_footer"] = parquet_row_count(path)
    res["footer_seconds"] = round(time.monotonic() - t2, 2)
    res["log"] = lines[-10:]
    path.unlink()
    res["deleted_after_measuring"] = not path.exists()
    return res


def write(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (OUT / "summary.json").write_text(text, encoding="utf-8")
    m, d, dl = report.get("machine", {}), report.get("disks", {}), report.get("download", {})

    def gb(x) -> str:
        return f"{x / 1e9:.1f} GB" if isinstance(x, int) else "?"

    lines = [
        "EXP-046 probe (Kaggle CPU session; no data processed)",
        f"commit: {COMMIT}",
        f"finished: {report.get('finished_utc', '?')}  ok: {report.get('ok')}",
        f"CPU: {m.get('cpu_count_usable')} usable of {m.get('cpu_count_os')} ({m.get('cpu_model', '?')})",
        f"RAM: {gb(m.get('ram_total_bytes'))} total, {gb(m.get('ram_available_bytes'))} available; "
        f"cgroup limit {m.get('cgroup_memory_limit', '?')}; GPU visible: {m.get('gpu_visible')}",
    ]
    for p in ("/kaggle/working", "/tmp", "/kaggle/input"):
        x = d.get(p, {})
        lines.append(f"disk {p}: {gb(x.get('free_bytes'))} free of {gb(x.get('total_bytes'))}")
    lines.append(f"dependencies: {report.get('dependencies')}")
    td = report.get("token_dataset", {})
    lines.append(f"EXP-037 token files: {td.get('files', '?')} files, {gb(td.get('total_bytes'))}")
    if dl:
        lines += [
            f"download {dl.get('source_id')}: {dl.get('result')} "
            f"in {dl.get('download_and_verify_seconds')} s "
            f"(~{dl.get('download_MB_per_s_incl_hash')} MB/s including the hash check)",
            f"hash alone: {dl.get('hash_seconds')} s; rows in footer: {dl.get('parquet_rows_from_footer')}; "
            f"deleted afterwards: {dl.get('deleted_after_measuring')}",
        ]
    for e in report.get("errors", []):
        lines.append(f"ERROR: {e}")
    (OUT / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    if len(COMMIT) != 40:
        raise SystemExit("the commit placeholder was not filled in; use scripts/run_kaggle_exp046_probe.ps1")
    started = time.time()
    report: dict = {"schema": PROBE_SCHEMA, "experiment": "EXP-046", "step": "probe", "smoke": False}
    report.update(environment={"code_commit": COMMIT}, complete=False, ok=False, errors=[])
    report["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    report["machine"] = machine()
    report["disks"] = disks()
    write(report)  # something is saved even if a later step fails
    try:
        sh(["git", "clone", "--quiet", REPO, str(SRC)])
        sh(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT])
        sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC)])
        sys.path.insert(0, str(SRC / "src"))
        report["dependencies"] = dependencies()
        report["token_dataset"] = token_dataset()
        report["download"] = download_one()
        report["disks_after"] = disks()
    except Exception:  # noqa: BLE001 - the probe reports, it never crashes the kernel
        report["errors"].append(traceback.format_exc(limit=3))
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["seconds"] = round(time.time() - started, 1)
    dl = report.get("download", {})
    report["ok"] = (
        not report["errors"]
        and dl.get("result") in ("downloaded", "verified")
        and bool(dl.get("sha256_again"))
    )
    report["complete"] = True
    write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
