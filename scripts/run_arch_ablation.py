#!/usr/bin/env python3
"""Architecture ablation runner (MASTER_CONTEXT §37 step 9): pre-registered spec → train → grade → decide.

    python scripts/run_arch_ablation.py --spec configs/ablations/EXP-032.json --dry-run
    python scripts/run_arch_ablation.py --spec configs/ablations/EXP-032.json

The spec (schema ``frontier-arch-ablation-v1``) fixes, BEFORE any result exists, the base
config, data, tokenizer, step budget, seeds, the baseline, the one-at-a-time variants (as
``--set`` overrides), a learning-rate check, a reproducibility check and the decision rules.
The runner then:

1. plans every cell (baseline × seeds, variant × seeds, learning-rate cells, repro cell);
   baseline cells may reuse earlier runs, but only after their saved config is checked
   against the spec (otherwise the baseline is trained fresh);
2. trains each cell as its own self-recording ``scripts/train.py`` run (D-032) and grades
   its FINAL checkpoint (``last/``) with ``scripts/eval_report.py`` (harness v1, D-042)
   right away, so partial nights still leave usable, graded models;
3. compares each variant with the baseline with ``scripts/eval_compare.py``;
4. applies the pre-registered rules and writes ``SUMMARY.txt`` / ``summary.json``, then
   an outer experiment record.

Restartable: cells whose ``last/`` checkpoint and success record exist are not retrained;
reports whose checkpoint hash matches are not regraded. Outputs go to
``out/arch/<exp-id>/`` (git-ignored); ``scripts/publish_eval_results.py`` copies the small
files to ``evals/results/<exp-id>/``.

Exit codes: 0 = every planned cell trained, graded and compared; 1 = some cell failed
(the summary is still written); 2 = bad spec or input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from frontier_ai.config import ExperimentConfig  # noqa: E402
from frontier_ai.evaluation import make_console_safe  # noqa: E402

SPEC_SCHEMA = "frontier-arch-ablation-v1"
RULES = ("superiority", "non_inferiority")
RESERVED = {"baseline", "repro-baseline"}


class SpecError(Exception):
    """The ablation spec is invalid."""


# --------------------------------------------------------------------------- spec --

def _overrides(d: dict[str, Any]) -> list[str]:
    return [f"{k}={v}" for k, v in d.items()]


def load_spec(path: Path) -> dict:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read spec {path}: {exc}") from None
    if spec.get("schema") != SPEC_SCHEMA:
        raise SpecError(f"spec schema must be {SPEC_SCHEMA!r}, got {spec.get('schema')!r}")
    if not re.fullmatch(r"EXP-\d{3,}", str(spec.get("exp_id", ""))):
        raise SpecError("spec exp_id must look like EXP-032")
    for key in ("base_config", "data", "tokenizer", "max_steps", "seeds", "baseline", "variants"):
        if key not in spec:
            raise SpecError(f"spec is missing {key!r}")
    if spec.get("checkpoint", "last") != "last":
        raise SpecError("only checkpoint 'last' is supported "
                        "(the 'best' checkpoint is selected on the suite)")
    seeds = spec["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
        raise SpecError("spec seeds must be a non-empty list of distinct integers")
    base = ExperimentConfig.load(_resolve(spec["base_config"]))
    base_ov = _overrides(spec["baseline"].get("overrides", {}))
    names = set()
    for v in spec["variants"]:
        name = v.get("name", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name) or name in RESERVED or name.startswith("lr-"):
            raise SpecError(f"bad or reserved variant name {name!r}")
        if name in names:
            raise SpecError(f"duplicate variant name {name!r}")
        names.add(name)
        if v.get("rule") not in RULES:
            raise SpecError(f"variant {name}: rule must be one of {RULES}")
        if v["rule"] == "non_inferiority" and not isinstance(v.get("margin_bpb"), (int, float)):
            raise SpecError(f"variant {name}: non_inferiority needs a numeric margin_bpb")
        if not v.get("overrides"):
            raise SpecError(f"variant {name}: overrides must change something")
        try:
            base.with_overrides(base_ov + _overrides(v["overrides"]))
        except (ValueError, TypeError) as exc:
            raise SpecError(f"variant {name}: invalid override: {exc}") from None
    for lr in (spec.get("lr_check") or {}).get("values", []):
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise SpecError(f"lr_check values must be positive numbers, got {lr!r}")
    for key in ("lr_check", "repro_check"):
        if spec.get(key) and spec[key].get("seed") not in seeds:
            raise SpecError(f"{key}.seed must be one of the spec seeds")
    return spec


def plan_cells(spec: dict, out_root: Path) -> list[dict]:
    """Every cell in a fixed order; cell ids do not depend on reuse, so they are stable."""
    base_ov = _overrides(spec["baseline"].get("overrides", {}))
    cells: list[dict] = []

    def add(group: str, seed: int, overrides: list[str], role: str) -> None:
        run_dir = out_root / "runs" / group / f"seed-{seed}"
        cells.append({
            "id": f"{spec['exp_id']}{len(cells) + 1:02d}", "group": group, "seed": seed, "role": role,
            "overrides": overrides, "run_dir": run_dir, "ckpt": run_dir / "last",
            "eval_dir": out_root / "eval" / f"{group}-seed-{seed}", "reused_from": None,
        })

    for seed in spec["seeds"]:
        add("baseline", seed, base_ov, "baseline")
    for v in spec["variants"]:
        for seed in spec["seeds"]:
            add(v["name"], seed, base_ov + _overrides(v["overrides"]), "variant")
    lr_check = spec.get("lr_check") or {}
    for lr in lr_check.get("values", []):
        add(f"lr-{lr:g}", lr_check["seed"], base_ov + [f"optim.lr={lr}"], "lr_check")
    if spec.get("repro_check"):
        add("repro-baseline", spec["repro_check"]["seed"], base_ov, "repro_check")
    return cells


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def cell_overrides(spec: dict, cell: dict) -> list[str]:
    return cell["overrides"] + [
        f"data.path={spec['data']}", f"data.seed={cell['seed']}", f"train.seed={cell['seed']}",
        f"train.out_dir={cell['run_dir']}", f"train.max_steps={spec['max_steps']}",
    ]


def _norm_path(p: Any) -> str:
    return str(p).replace("\\", "/")


def check_reuse(spec: dict, cell: dict, ckpt: Path) -> list[str]:
    """Mismatches between a candidate checkpoint's saved config and what the spec requires."""
    problems = []
    for f in ("model.pt", "config.json", "meta.json"):
        if not (ckpt / f).is_file():
            problems.append(f"missing {ckpt / f}")
    if problems:
        return problems
    saved = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    want = ExperimentConfig.load(_resolve(spec["base_config"])).with_overrides(cell_overrides(spec, cell))
    for key, value in vars(want.model).items():
        if key != "vocab_size" and saved.get("model", {}).get(key) != value:
            problems.append(f"model.{key}: saved {saved.get('model', {}).get(key)!r} != spec {value!r}")
    for key, value in vars(want.optim).items():
        got = saved.get("optim", {}).get(key)
        if (list(got) if isinstance(got, (list, tuple)) else got) != (
                list(value) if isinstance(value, (list, tuple)) else value):
            problems.append(f"optim.{key}: saved {got!r} != spec {value!r}")
    for section, keys in (("data", ("batch_size", "seed")),
                          ("train", ("max_steps", "accum_steps", "seed", "precision", "deterministic"))):
        for key in keys:
            got, value = saved.get(section, {}).get(key), getattr(getattr(want, section), key)
            if got != value:
                problems.append(f"{section}.{key}: saved {got!r} != spec {value!r}")
    if _norm_path(saved.get("data", {}).get("path")) != _norm_path(spec["data"]):
        problems.append(f"data.path: saved {saved.get('data', {}).get('path')!r} != spec {spec['data']!r}")
    if meta.get("step") != spec["max_steps"]:
        problems.append(f"checkpoint step {meta.get('step')} != max_steps {spec['max_steps']}")
    return problems


# ---------------------------------------------------------------------- execution --

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spec_sha256(path: Path) -> str:
    """Spec fingerprint over LF-normalised bytes: Git for Windows checks text files out with CRLF,
    which changed the raw-byte hash of the same spec (EXP-032: PC 2a7048f2… vs repo f8c13dad…)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cell_trained(spec: dict, cell: dict) -> bool:
    ckpt, rec = cell["ckpt"], cell["run_dir"] / "experiment.json"
    if not (ckpt / "model.pt").is_file() or not rec.is_file():
        return False
    try:
        ok = json.loads(rec.read_text(encoding="utf-8")).get("execution", {}).get("status") == "success"
        meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return ok and meta.get("step") == spec["max_steps"]


def _run(cmd: list[str]) -> tuple[int, float]:
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sys.stdout.write(proc.stdout)
    sys.stdout.write(proc.stderr)
    sys.stdout.flush()
    return proc.returncode, time.monotonic() - t0


def train_cell(spec: dict, cell: dict) -> bool:
    if cell_trained(spec, cell):
        print(f"[ablation] {cell['group']} seed {cell['seed']}: already trained, skipping", flush=True)
        return True
    print(f"[ablation] TRAIN {cell['group']} seed {cell['seed']} ({cell['id']}) ...", flush=True)
    set_args = [a for ov in cell_overrides(spec, cell) for a in ("--set", ov)]
    code, wall = _run([sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
                       "--config", str(_resolve(spec["base_config"])), *set_args, "--exp-id", cell["id"]])
    if code != 0 or not cell_trained(spec, cell):
        print(f"[ablation] FAILED training {cell['group']} seed {cell['seed']} (exit {code})", flush=True)
        return False
    (cell["run_dir"] / "ablation_cell.json").write_text(
        json.dumps({"cell": cell["id"], "wall_seconds": round(wall, 1),
                    "seconds_per_step": round(wall / spec["max_steps"], 3)}, indent=2) + "\n",
        encoding="utf-8")
    print(f"[ablation]   trained in {wall:.0f}s", flush=True)
    return True


def grade_cell(spec: dict, cell: dict) -> bool:
    report = cell["eval_dir"] / "report.json"
    model_sha = _sha256(cell["ckpt"] / "model.pt")
    if report.is_file():
        try:
            if json.loads(report.read_text(encoding="utf-8"))["checkpoint"]["model_sha256"] == model_sha:
                print(f"[ablation] {cell['group']} seed {cell['seed']}: already graded, skipping", flush=True)
                return True
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    print(f"[ablation] GRADE {cell['group']} seed {cell['seed']} ...", flush=True)
    code, wall = _run([sys.executable, str(REPO_ROOT / "scripts" / "eval_report.py"),
                       "--ckpt", str(cell["ckpt"]), "--tokenizer", str(spec["tokenizer"]),
                       "--out", str(cell["eval_dir"]), "--label", f"{cell['group']}-seed-{cell['seed']}",
                       "--exp-id", spec["exp_id"], *spec.get("eval_args", [])])
    if code != 0:
        print(f"[ablation] FAILED grading {cell['group']} seed {cell['seed']} (exit {code})", flush=True)
        return False
    print(f"[ablation]   graded in {wall:.0f}s", flush=True)
    return True


def compare_variant(spec: dict, cells: list[dict], name: str, out_root: Path) -> bool:
    a = [str(c["eval_dir"]) for c in cells if c["group"] == "baseline"]
    b = [str(c["eval_dir"]) for c in cells if c["group"] == name]
    if not all((Path(d) / "report.json").is_file() for d in a + b):
        print(f"[ablation] compare {name}: incomplete reports, skipped", flush=True)
        return False
    code, _ = _run([sys.executable, str(REPO_ROOT / "scripts" / "eval_compare.py"), "--a", *a, "--b", *b,
                    "--label-a", "baseline", "--label-b", name, "--out", str(out_root / "compare" / name)])
    return code == 0


# ------------------------------------------------------------------------ analysis --

def weights_compare(a: Path, b: Path) -> dict:
    import torch

    sa = torch.load(a / "model.pt", map_location="cpu", weights_only=True)
    sb = torch.load(b / "model.pt", map_location="cpu", weights_only=True)
    if set(sa) != set(sb):
        return {"status": "DIFFERENT", "detail": "different parameter names"}
    max_diff, unequal = 0.0, 0
    for k in sa:
        if sa[k].shape != sb[k].shape:
            return {"status": "DIFFERENT", "detail": f"shape differs for {k}"}
        if not torch.equal(sa[k], sb[k]):
            unequal += 1
            max_diff = max(max_diff, float((sa[k].double() - sb[k].double()).abs().max()))
    return {"status": "IDENTICAL" if unequal == 0 else "DIFFERENT", "tensors": len(sa),
            "unequal_tensors": unequal, "max_abs_diff": max_diff}


def superiority(ci: list[float], base: list[float], var: list[float]) -> str:
    if ci[1] < 0 and max(var) < min(base):
        return "BETTER"
    if ci[0] > 0 and min(var) > max(base):
        return "WORSE"
    return "NO DETECTABLE DIFFERENCE AT THIS SCALE"


def _bpb(cell: dict) -> float | None:
    p = cell["eval_dir"] / "report.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["results"]["overall"]["bits_per_byte"]


def _arch_facts(cell: dict) -> dict:
    cfg = json.loads((cell["ckpt"] / "config.json").read_text(encoding="utf-8"))["model"]
    head_dim = cfg["n_embd"] // cfg["n_head"]
    n_kv = cfg.get("n_kv_head") or cfg["n_head"]
    emb = cfg["vocab_size"] * cfg["n_embd"] * (1 if cfg.get("tie_embeddings", True) else 2)
    pos = cfg["block_size"] * cfg["n_embd"] if cfg.get("pos") == "learned" else 0
    meta = json.loads((cell["ckpt"] / "meta.json").read_text(encoding="utf-8"))
    n = meta.get("n_params")
    timing = cell["run_dir"] / "ablation_cell.json"
    sps = json.loads(timing.read_text(encoding="utf-8"))["seconds_per_step"] if timing.is_file() else None
    return {"n_params": n, "body_params": (n - emb - pos) if n else None,
            "kv_cache_values_per_token": 2 * cfg["n_layer"] * n_kv * head_dim, "seconds_per_step": sps}


def summarize(spec: dict, cells: list[dict], out_root: Path, failures: list[str]) -> dict:
    by = {}
    for c in cells:
        by.setdefault(c["group"], []).append(c)
    base_vals = [v for v in (_bpb(c) for c in by["baseline"]) if v is not None]
    base_std = statistics.stdev(base_vals) if len(base_vals) > 1 else None
    rows, verdicts = [], {}
    for group, gcells in by.items():
        vals = [_bpb(c) for c in gcells]
        done = [v for v in vals if v is not None]
        facts = _arch_facts(gcells[0]) if (gcells[0]["ckpt"] / "config.json").is_file() else {}
        rows.append({"group": group, "seeds": [c["seed"] for c in gcells], "bpb": vals,
                     "mean": statistics.mean(done) if done else None,
                     "std": statistics.stdev(done) if len(done) > 1 else None,
                     "reused_from": gcells[0]["reused_from"], **facts})
    for v in spec["variants"]:
        comp = out_root / "compare" / v["name"] / "compare.json"
        var_vals = [x for x in (_bpb(c) for c in by[v["name"]]) if x is not None]
        n = len(spec["seeds"])
        if not comp.is_file() or len(var_vals) != n or len(base_vals) != n:
            verdicts[v["name"]] = {"verdict": "INCOMPLETE"}
            continue
        cj = json.loads(comp.read_text(encoding="utf-8"))
        delta, ci = cj["overall"]["delta_b_minus_a"], cj["overall"]["ci95"]
        langs = cj.get("per_language", {})
        entry = {"rule": v["rule"], "delta_bpb": delta, "ci95": ci,
                 "superiority": superiority(ci, base_vals, var_vals),
                 "languages_better": sorted(k for k, s in langs.items() if s["ci95"][1] < 0),
                 "languages_worse": sorted(k for k, s in langs.items() if s["ci95"][0] > 0)}
        if v["rule"] == "non_inferiority":
            entry["margin_bpb"] = v["margin_bpb"]
            entry["verdict"] = "ACCEPTABLE" if ci[1] <= v["margin_bpb"] else "NOT SHOWN"
        else:
            entry["verdict"] = entry["superiority"]
        verdicts[v["name"]] = entry
    lr = None
    if spec.get("lr_check"):
        seed = spec["lr_check"]["seed"]
        ref = next((_bpb(c) for c in by["baseline"] if c["seed"] == seed), None)
        alts = {g: _bpb(cs[0]) for g, cs in by.items() if g.startswith("lr-")}
        if ref is None or base_std is None or any(v is None for v in alts.values()):
            lr = {"verdict": "INCOMPLETE"}
        else:
            base_lr = ExperimentConfig.load(_resolve(spec["base_config"])).optim.lr
            gains = {g: ref - v for g, v in alts.items()}
            conf = [g for g, gain in gains.items() if gain > base_std]
            lr = {"seed": seed, "reference_lr": base_lr, "reference_bpb": ref, "baseline_seed_std": base_std,
                  "alternatives": {g: {"bpb": alts[g], "improvement": gains[g]} for g in alts},
                  "verdict": "LR-CONFOUNDED" if conf else "NOT CONFOUNDED", "better_lrs": conf}
    repro = None
    if spec.get("repro_check"):
        seed = spec["repro_check"]["seed"]
        rc = next(c for c in by["repro-baseline"] if c["seed"] == seed)
        bc = next(c for c in by["baseline"] if c["seed"] == seed)
        if (rc["ckpt"] / "model.pt").is_file() and (bc["ckpt"] / "model.pt").is_file():
            repro = {"reference": _norm_path(bc["ckpt"]), **weights_compare(bc["ckpt"], rc["ckpt"])}
            rb, bb = _bpb(rc), _bpb(bc)
            if rb is not None and bb is not None:
                repro["bpb_reference"], repro["bpb_retrained"] = bb, rb
        else:
            repro = {"status": "INCOMPLETE"}
    return {"exp_id": spec["exp_id"], "spec_sha256": spec["_sha256"], "max_steps": spec["max_steps"],
            "seeds": spec["seeds"], "rows": rows, "variants": verdicts, "lr_check": lr,
            "repro_check": repro, "failures": failures}


def render(s: dict) -> str:
    f = lambda x: "—" if x is None else f"{x:.4f}"  # noqa: E731
    lines = [f"ARCHITECTURE ABLATION {s['exp_id']} — {s['max_steps']} steps, seeds {s['seeds']}, "
             f"final checkpoints graded by harness v1 (bits-per-byte, lower = better)",
             f"spec sha256 {s['spec_sha256'][:16]}… (pre-registered rules)", "",
             f"{'group':16}{'mean':>9}{'std':>9}  per seed{'':16}"
             f"{'params':>11}{'body':>11}{'kv/tok':>8}{'s/step':>8}"]
    for r in s["rows"]:
        seeds = " ".join(f(v) for v in r["bpb"])
        lines.append(f"{r['group']:16}{f(r['mean']):>9}{f(r['std']):>9}  {seeds:24}"
                     f"{r.get('n_params') or 0:>11,}{r.get('body_params') or 0:>11,}"
                     f"{r.get('kv_cache_values_per_token') or 0:>8}"
                     f"{format(r['seconds_per_step'], '.2f') if r.get('seconds_per_step') else '—':>8}"
                     + (" (reused)" if r["reused_from"] else ""))
    lines += ["", "VERDICTS (delta = variant - baseline; negative = variant better; paired 95% CI)"]
    for name, v in s["variants"].items():
        if v["verdict"] == "INCOMPLETE":
            lines.append(f"  {name:12} INCOMPLETE")
            continue
        extra = (f" (margin {v['margin_bpb']}; superiority: {v['superiority']})"
                 if v["rule"] == "non_inferiority" else "")
        lines.append(f"  {name:12} delta {v['delta_bpb']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
                     f"=> {v['verdict']}{extra}")
        lines.append(f"  {'':12} languages better: {', '.join(v['languages_better']) or 'none'} | "
                     f"worse: {', '.join(v['languages_worse']) or 'none'}")
    lr = s["lr_check"]
    if lr:
        if lr["verdict"] == "INCOMPLETE":
            lines += ["", "LEARNING-RATE CHECK: INCOMPLETE"]
        else:
            alts = ", ".join(f"{g}: {a['bpb']:.4f} ({a['improvement']:+.4f})"
                             for g, a in lr["alternatives"].items())
            lines += ["", f"LEARNING-RATE CHECK (seed-{lr['seed']} baseline at lr {lr['reference_lr']}: "
                          f"{lr['reference_bpb']:.4f}; threshold = baseline seed std "
                          f"{lr['baseline_seed_std']:.4f})",
                      f"  {alts}  => {lr['verdict']}"]
    rp = s["repro_check"]
    if rp:
        detail = (f"{rp.get('unequal_tensors', '?')} of {rp.get('tensors', '?')} tensors differ, "
                  f"max |diff| {rp.get('max_abs_diff', 0):.3g}" if rp.get("status") != "INCOMPLETE" else "")
        lines += ["", "REPRODUCIBILITY CHECK (retrained baseline vs reference weights): "
                      f"{rp['status']} {detail}"]
    if s["failures"]:
        lines += ["", "FAILED CELLS: " + ", ".join(s["failures"])]
    lines += ["", "Toy scale (see spec purpose): screening evidence only; "
                  "a provisional decision needs founder review."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- main --

def main() -> int:
    make_console_safe()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", default=None, help="default out/arch/<exp-id>")
    p.add_argument("--dry-run", action="store_true", help="plan + reuse checks only; train nothing")
    p.add_argument("--summary-only", action="store_true", help="re-derive the summary from existing outputs")
    p.add_argument("--sec-per-step", type=float, default=7.3, help="for the dry-run time estimate")
    p.add_argument("--no-record", action="store_true")
    args = p.parse_args()

    spec_path = _resolve(args.spec)
    try:
        spec = load_spec(spec_path)
    except SpecError as exc:
        print(f"[ablation] SPEC ERROR: {exc}", file=sys.stderr)
        return 2
    spec["_sha256"] = spec_sha256(spec_path)
    out_root = _resolve(args.out or f"out/arch/{spec['exp_id']}")
    if not _resolve(spec["data"]).is_file():
        print(f"[ablation] INPUT ERROR: data file not found: {spec['data']}", file=sys.stderr)
        return 2
    cells = plan_cells(spec, out_root)

    reuse_dir = spec["baseline"].get("reuse_runs_dir")
    for c in cells:
        if c["group"] == "baseline" and reuse_dir:
            cand = _resolve(reuse_dir) / f"seed-{c['seed']}" / "last"
            problems = check_reuse(spec, c, cand)
            if problems:
                more = f" and {len(problems) - 1} more" if len(problems) > 1 else ""
                print(f"[ablation] baseline seed {c['seed']}: NOT reused ({problems[0]}{more}) -> will train",
                      flush=True)
            else:
                c["reused_from"], c["ckpt"], c["run_dir"] = _norm_path(cand), cand, cand.parent
                print(f"[ablation] baseline seed {c['seed']}: reusing {_norm_path(cand)} (config verified)",
                      flush=True)

    to_train = [c for c in cells if not c["reused_from"] and not cell_trained(spec, c)]
    print(f"[ablation] {spec['exp_id']}: {len(cells)} cells, {len(to_train)} to train "
          f"(~{len(to_train) * spec['max_steps'] * args.sec_per_step / 3600:.1f} h at "
          f"{args.sec_per_step} s/step, plus a few minutes of grading per model)", flush=True)
    for c in cells:
        state = "reused" if c["reused_from"] else ("trained" if cell_trained(spec, c) else "to train")
        change = " ".join(c["overrides"]) or "(baseline)"
        print(f"  {c['id']}  {c['group']:16} seed {c['seed']}  [{state}]  {change}")
    if args.dry_run:
        return 0

    failures: list[str] = []
    if not args.summary_only:
        for c in cells:
            ok = (c["reused_from"] is not None or train_cell(spec, c)) and grade_cell(spec, c)
            if not ok:
                failures.append(f"{c['group']}-seed-{c['seed']}")
        for v in spec["variants"]:
            if not compare_variant(spec, cells, v["name"], out_root):
                failures.append(f"compare-{v['name']}")
    summary = summarize(spec, cells, out_root, failures)
    text = render(summary)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_root / "SUMMARY.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text, flush=True)

    if not args.no_record and not args.summary_only:
        from frontier_ai.experiments import ExperimentSpec
        from frontier_ai.experiments.autowire import run_self_recorded

        def build_spec() -> ExperimentSpec:
            return ExperimentSpec(
                experiment_id=spec["exp_id"], seed=spec["seeds"][0],
                name=f"Architecture ablation {spec['exp_id']}", output_dir=str(out_root),
                params={"spec": _norm_path(args.spec), "spec_sha256": spec["_sha256"],
                        "max_steps": spec["max_steps"], "seeds": spec["seeds"],
                        "variants": [v["name"] for v in spec["variants"]]},
                data_paths=[str(spec_path), str(_resolve(spec["data"]))], command=list(sys.argv),
                tags=["architecture", "ablation", "step-9"], notes=spec.get("title", ""))

        recorded = run_self_recorded(build_spec, lambda: {
            "variants": {k: v.get("verdict") for k, v in summary["variants"].items()},
            "lr_check": (summary["lr_check"] or {}).get("verdict"),
            "repro_check": (summary["repro_check"] or {}).get("status"), "failures": failures})
        if recorded.failed:
            print(f"[ablation] record FAILED — {recorded.error}", file=sys.stderr)
            return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
