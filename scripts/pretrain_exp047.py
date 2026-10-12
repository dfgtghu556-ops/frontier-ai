#!/usr/bin/env python
"""EXP-047: train the next base model over one pass of v2-slice1 + v2-slice2 (8,159,265,692 tokens),
one Kaggle session at a time.

CODE ONLY (founder 2026-10-11: "approve EXP-047 code"). Running it needs "approve EXP-047", the
founder's compute choice (A/B/C) and the size chosen by the rule of EXP-047 section 4 after EXP-043's
final result. That is why ``--size`` and ``--cap-hours`` have **no default**: whoever launches the run
must type the approved values, and both are written into every session's ``setup`` (a later session
with another size or cap is refused).

It is the EXP-043 runner (``scripts/gpu_pretrain.py``) with the EXP-047 parts swapped in while this
script runs; EXP-043 itself is unchanged (its hooks keep their defaults, ``tests/test_pretrain*.py``):

- **Data:** the three pinned slices of :mod:`frontier_ai.data.slices` (EXP-037 slice 1; EXP-046 build A
  and B = slice 2, D-051), 26 files named ``<slice>/<lang>``, natural proportions, one pass in a fixed
  shuffled order: 15,936,053 windows of 512 tokens = 498,001 steps of 32 windows. Part 0 checks each
  manifest's pinned SHA-256 and each file's size, meta counts and full SHA-256.
- **Model:** a member of the EXP-042/043 family (depth = width / 64, head size 64, RoPE, GQA-2, D-048):
  14x896, 16x1024, 18x1152, 20x1280, 22x1408 (EXP-047 section 1).
- **Learning rate, memory, sessions, checkpoint chain, two GPUs (D-050), stop rules:** exactly as EXP-043
  (same 6,104-step check and rule, same 32 windows per step). The memory probe tries micro-batch 16, 8,
  then 4 (accumulation 2, 4, 8) under the same 14.5 GB limit.
- **Validation:** slice 1 is the main yardstick (EXP-047 section 5), so EXP-047 and EXP-043 compare on the
  same text. Each session ends with full slice-1 validation; the session that finishes the pass also
  scores slice 2 and every language pooled over both slices (reported, not gated).
- **Gate (proposed; numbers fixed in the approved version):** final slice-1 mean bpb below EXP-043's final
  minus the seed noise 0.0174, and no language worse than EXP-043's final by more than the noise.

Approved 2026-10-12 (D-052): option B, size 20x1280, gate 0.6109. The Kaggle kernel
(``scripts/kaggle/exp047_kernel.py``) and runner (``scripts/run_kaggle_exp047.ps1``) type those values.
A sharded optimizer is not built (EXP-047 note 2026-10-13: the one-GPU memory probe, learning-rate check
and first main session decide whether the size fits, and sharding cannot help them).

CPU test of the whole flow: ``--smoke`` (tests/test_pretrain_exp047.py).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402

import gpu_bringup as gb  # noqa: E402
import gpu_pretrain as gp  # noqa: E402
from frontier_ai.data import slices as sl  # noqa: E402
from frontier_ai.data.multi import MultiTokenDataset, full_split_loss  # noqa: E402

ROOT = gb.ROOT
EXP_ID = "EXP-047"
YARDSTICK = "s1"  # slice 1 (EXP-037): the main yardstick, EXP-047 section 5
TEST_SLICES_ENV = "FRONTIER_EXP047_TEST_SLICES"  # CPU tests only (honoured with --smoke only)

# EXP-047 section 1: (layers, width, predicted bpb at 8.16 B tokens [EXP-042 secondary fit, extrapolated,
# recomputed if section 4 refits the laws], estimated 2xT4 hours [NOT VERIFIED]).
# (layers, width, predicted bpb, 2xT4 session-hours): the section 4 refit of 2026-10-12, pre-registered method
# (evals/results/EXP-047/refit/refit.json, "pred_refit" and "t4.total_h"); it replaced the section 1 values
# (0.6715 ... 0.6552) before any EXP-047 session ran.
FAMILY: dict[str, tuple[int, int, float, int]] = {
    "14x896": (14, 896, 0.6333, 122),
    "16x1024": (16, 1024, 0.6254, 172),
    "18x1152": (18, 1152, 0.6190, 236),
    "20x1280": (20, 1280, 0.6138, 314),
    "22x1408": (22, 1408, 0.6095, 408),
}
EXPECTED_WINDOWS = 15_936_053  # sum over the 26 files of (n_train - 1) // 512 (pinned manifests)
EXPECTED_STEPS = 498_001  # EXPECTED_WINDOWS // 32

# The EXP-043 originals, taken before anything is swapped in.
_GP = {
    name: getattr(gp, name)
    for name in ("render", "make_samples", "EXP_ID", "WORKER_SCRIPT", "SETUP_FULL", "SETUP_SMOKE")
}
HOOKS = ("part0_data", "open_base", "score_all", "final_scores", "final_checks", "make_samples", "render")


def model_for(size: str) -> dict[str, Any]:
    layers, width, _, _ = FAMILY[size]
    return {
        "n_layer": layers,
        "n_head": width // 64,
        "n_embd": width,
        "block_size": 512,
        "pos": "rope",
        "n_kv_head": 2,
    }


def make_setup(size: str, cap_hours: float, smoke: bool, slices: Sequence[sl.Slice]) -> dict[str, Any]:
    """The pre-registered plan of one EXP-047 run (identical in every session of the run)."""
    if size not in FAMILY:
        raise ValueError(f"unknown size {size!r}; the family is {', '.join(FAMILY)}")
    if not cap_hours > 0:
        raise ValueError("the GPU-hour cap must be positive")
    setup = json.loads(json.dumps(_GP["SETUP_SMOKE" if smoke else "SETUP_FULL"]))
    for key in ("reference_run", "reference_bpb"):  # EXP-043's reference was EXP-042's C3-s3
        setup.pop(key)
    setup.update(
        size=size,
        total_cap_hours=float(cap_hours),
        slices=[
            {"name": s.name, "manifest": s.manifest, "manifest_sha256": s.manifest_sha256} for s in slices
        ],
        yardstick=f"{YARDSTICK} validation: mean of per-language bpb, equal weight",
        reference_dir="evals/results/EXP-043",  # the gate compares with EXP-043's final result
        predicted_bpb=FAMILY[size][2],
    )
    if smoke:  # CPU test only: 2 tiny sampled batches are too noisy for the 0.1-nat stop rule
        setup.update(eval_iters=8, diverge_nats=0.5)
    else:
        setup.update(
            model=model_for(size),
            expected_windows=EXPECTED_WINDOWS,
            expected_steps=EXPECTED_STEPS,
            micro_options=[[16, 2], [8, 4], [4, 8]],  # all split evenly over two GPUs
            reserve_minutes=45.0,  # session-end save + slice-1 validation of a model up to 3.5x EXP-043's
        )
    return setup


# ------------------------------------------------------------------ data --
def slice_defs(smoke: bool) -> tuple[sl.Slice, ...]:
    """The three pinned EXP-047 slices (CPU tests may give their own, with --smoke only)."""
    spec = os.environ.get(TEST_SLICES_ENV)
    if spec and not smoke:
        raise sl.SliceError(f"{TEST_SLICES_ENV} is for CPU tests (--smoke) only")
    if spec:
        return tuple(sl.Slice(**d) for d in json.loads(spec))
    return sl.EXP047_SLICES


def open_data(args: argparse.Namespace, full_sha256: bool) -> tuple[MultiTokenDataset, dict[str, Any]]:
    """Find each slice under --data-dir (e.g. /kaggle/input) and open all of them after the checks."""
    pairs = []
    for s in slice_defs(args.smoke):
        pairs.append((s, sl.locate(s, sl.load_manifest(s), Path(args.data_dir))))
    return sl.open_slices(pairs, full_sha256=full_sha256)


def part0_data(args: argparse.Namespace, rep: gp.Report) -> MultiTokenDataset | None:
    """Part 0: the pinned manifests and every data file, the GPU, the tests; then the training data."""
    rep.data["environment"] = gb.environment_record()
    p0: dict[str, Any] = {"sha256_checked": not args.skip_data_hash, "files_expected": None, "files_ok": 0}
    rep.data["part0"] = p0
    try:
        ds, record = open_data(args, full_sha256=not args.skip_data_hash)
    except (sl.SliceError, OSError, ValueError) as exc:
        p0["problems"] = [str(exc)]
        rep.data["stopped"] = f"data files do not match the pinned EXP-047 slice manifests: {exc}"
        rep.save()
        return None
    n = sum(s["files"] for s in record["slices"])
    p0.update(files_expected=n, files_ok=n, problems=[], slices=record["slices"])
    p0.update(train_tokens=record["train_tokens"], val_tokens=record["val_tokens"])
    rep.save()
    if not args.smoke and not torch.cuda.is_available():
        rep.data["stopped"] = "no CUDA GPU found (on Kaggle: the accelerator must be a GPU)"
        return None
    if args.skip_tests:
        p0["tests_pass"], p0["tests_line"] = None, "skipped"
    else:
        extra = ("tests/test_onepass_sampler.py", "tests/test_slices.py")
        p0["tests_pass"], p0["tests_line"] = gb.run_repo_tests(extra=extra)
        if not p0["tests_pass"]:
            rep.data["stopped"] = "the model/trainer tests failed on this machine"
            return None
    if ds.vocab_size != 32896:
        rep.data["stopped"] = f"vocabulary {ds.vocab_size}, expected 32896 (Frontier Tokenizer v2)"
        return None
    rep.data["languages"] = ds.names
    rep.data["sampling_weights"] = {n: round(float(w), 6) for n, w in zip(ds.names, ds.weights("train"))}
    rep.save()
    return ds


def open_base(args: argparse.Namespace) -> MultiTokenDataset:
    """The two-GPU worker opens the data again (Part 0 already checked every file)."""
    return open_data(args, full_sha256=False)[0]


# ---------------------------------------------------------------- scores --
def slice_parts(ds: MultiTokenDataset, yardstick: bool) -> list[str]:
    return [n for n in ds.names if (n.split("/", 1)[0] == YARDSTICK) == yardstick]


def part_nats(
    model, ds: MultiTokenDataset, names: Sequence[str], block: int, batch: int, device, amp
) -> dict:
    return {n: full_split_loss(model, ds.parts[n], block, batch, device, "val", amp)[0] for n in names}


def summarise(ds: MultiTokenDataset, nats: dict[str, float], names: Sequence[str]) -> dict[str, Any]:
    """EXP-043's four numbers for a set of parts with one part per language (one slice, or s2a + s2b)."""
    langs = [n.split("/", 1)[1] for n in names]
    if len(set(langs)) != len(langs):
        raise ValueError(f"a language appears twice in {list(names)}")
    bpb = {n.split("/", 1)[1]: nats[n] / math.log(2) * ds.parts[n].tokens_per_byte("val") for n in names}
    n_val = sum(ds.parts[n].n_val for n in names)
    bits = sum(nats[n] / math.log(2) * ds.parts[n].n_val for n in names)
    nbytes = sum(ds.parts[n].split_lengths("val")[0] for n in names)
    return {
        "bpb": bpb,
        "bpb_mean": sum(bpb.values()) / len(bpb),
        "bpb_token_weighted": sum(bpb[n.split("/", 1)[1]] * ds.parts[n].n_val for n in names) / n_val,
        "bpb_pooled": bits / nbytes,
    }


def score_all(model, ds: MultiTokenDataset, block: int, batch: int, device, amp) -> dict:
    """Session end: full slice-1 validation (the yardstick), per language as EXP-043 reports it."""
    names = slice_parts(ds, yardstick=True)
    nats = part_nats(model, ds, names, block, batch, device, amp)
    return {"nats": nats, **summarise(ds, nats, names), "validated": f"{YARDSTICK} (slice 1) only"}


def final_scores(
    model, ds: MultiTokenDataset, session_eval: dict, block: int, setup: dict, device, amp
) -> dict:
    """The finished model: slice 1 (from the session end) plus slice 2 and both pooled per language."""
    names = slice_parts(ds, yardstick=False)
    nats = {**session_eval["nats"], **part_nats(model, ds, names, block, setup["eval_batch"], device, amp)}
    pooled = sl.combine_bpb(ds, nats)["per_language"]
    out = {**session_eval, "nats": nats, "validated": "slice 1, slice 2 and both pooled"}
    out["slice2"] = summarise(ds, nats, names) if names else None
    out["pooled_by_language"] = pooled
    out["pooled_mean"] = sum(pooled.values()) / len(pooled)
    return out


def reference_final(ref_dir: Path) -> dict[str, Any] | None:
    """EXP-043's final scores (slice-1 validation), if its last session is in the repository."""
    files = sorted(ref_dir.glob("session-*/summary.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("complete") and not d.get("smoke") and (d.get("final") or {}).get("scores"):
            s = d["final"]["scores"]
            return {"session": d["session"], "bpb": s["bpb"], "bpb_mean": s["bpb_mean"]}
    return None


def final_checks(scores: dict, setup: dict) -> dict[str, Any]:
    ref = reference_final(ROOT / setup["reference_dir"])
    mean, noise = scores["bpb_mean"], setup["noise"]
    consistent = abs(mean - setup["predicted_bpb"]) <= setup["consistent_band"]
    out: dict[str, Any] = {}
    if ref is None or set(ref["bpb"]) != set(scores["bpb"]):
        why = "is not in the repository" if ref is None else "has other languages"
        out["check1_gate"] = f"NOT DECIDED: EXP-043's final result {why} ({setup['reference_dir']})"
        out["check1_pass"] = None
    else:
        gate = ref["bpb_mean"] - noise
        worse = {
            k: round(v - ref["bpb"][k], 4) for k, v in scores["bpb"].items() if v > ref["bpb"][k] + noise
        }
        ok = mean < gate and not worse
        out["check1_gate"] = (
            f"{'PASS' if ok else 'FAIL'}: slice-1 mean {mean:.4f} vs < {gate:.4f} (EXP-043 final "
            f"{ref['bpb_mean']:.4f} minus noise {noise}); languages worse than EXP-043 by more than the "
            f"noise: {worse or 'none'}"
        )
        out["check1_pass"] = ok
        out["check1_worse_languages"] = worse
    out["check2_prediction"] = (
        f"{'consistent' if consistent else 'not consistent'}: {mean:.4f} vs predicted "
        f"{setup['predicted_bpb']} (+/- {setup['consistent_band']})"
    )
    out["check2_consistent"] = consistent
    s2 = scores.get("slice2")
    out["check3_slice2"] = (
        f"reported, not gated: slice-2 mean {s2['bpb_mean']:.4f}; pooled over both slices "
        f"{scores['pooled_mean']:.4f}"
        if s2
        else "slice 2 not scored"
    )
    out["per_language_reference"] = ref["bpb"] if ref else {}
    return out


def make_samples(model, ds: MultiTokenDataset, setup: dict, device) -> list[dict[str, Any]]:
    """EXP-043's fixed-prompt samples, from the slice-1 validation text of each language."""
    view = types.SimpleNamespace(parts={n.split("/", 1)[1]: ds.parts[n] for n in slice_parts(ds, True)})
    return _GP["make_samples"](model, view, setup, device)


def render(d: dict[str, Any]) -> str:
    size = (d.get("setup") or {}).get("size")
    text = _GP["render"](d)
    text = text.replace(
        "train the step-12 model (D-049)", f"train the next base model ({size}, v2-slice1 + v2-slice2)", 1
    )
    text = text.replace("full validation at session end:", "slice-1 validation at session end:")
    text = text.replace("(C3-s3 ", "(EXP-043 ")
    extra = []
    p0 = d.get("part0") or {}
    for s in p0.get("slices", []):
        extra.append(f"  slice {s['name']}: {s['files']} files, {s['train_tokens']:,} training tokens")
    f = d.get("final") or {}
    s2 = (f.get("scores") or {}).get("slice2")
    if s2:
        extra.append("  slice-2 validation (reported, not gated):")
        extra += [f"    {k:<6} {v:.4f}" for k, v in s2["bpb"].items()]
        pooled = f["scores"]["pooled_by_language"]
        extra.append(
            "  both slices pooled per language: " + ", ".join(f"{k} {v:.4f}" for k, v in pooled.items())
        )
    return text.rstrip("\n") + ("\n" + "\n".join(extra) if extra else "") + "\n"


# ------------------------------------------------------------------ main --
@contextlib.contextmanager
def installed(setup_full: dict | None = None, setup_smoke: dict | None = None) -> Iterator[None]:
    """Swap the EXP-047 parts into the EXP-043 runner for the duration of one call, then restore."""
    names = (*HOOKS, "EXP_ID", "WORKER_SCRIPT", "SETUP_FULL", "SETUP_SMOKE")
    saved = {n: getattr(gp, n) for n in names}
    try:
        for n in HOOKS:
            setattr(gp, n, globals()[n])
        gp.EXP_ID = EXP_ID
        gp.WORKER_SCRIPT = Path(__file__).resolve()
        if setup_full is not None:
            gp.SETUP_FULL = setup_full
        if setup_smoke is not None:
            gp.SETUP_SMOKE = setup_smoke
        yield
    finally:
        for n, v in saved.items():
            setattr(gp, n, v)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--ddp-worker"]:  # D-050: the setup travels in the worker's in.json
        with installed():
            return gp.main(argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False
    )
    p.add_argument("--size", choices=list(FAMILY), help="the family member chosen by EXP-047 section 4")
    p.add_argument("--cap-hours", type=float, help="the approved GPU-hour cap over all sessions")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("-h", "--help", action="store_true")
    own, rest = p.parse_known_args(argv)
    if own.help:
        p.print_help()
        print("\nThe other options are those of scripts/gpu_pretrain.py (EXP-043).")
        return 0
    if own.size is None or own.cap_hours is None:
        print("STOPPED: --size and --cap-hours have no default; type the approved values", flush=True)
        return 2
    try:
        setup = make_setup(own.size, own.cap_hours, own.smoke, slice_defs(own.smoke))
    except (ValueError, OSError) as exc:
        print(f"STOPPED: {exc}", flush=True)
        return 2
    defaults = [
        "--out", "out/gpu/EXP-047",
        "--prev-dir", "evals/results/EXP-047",
        "--chain-out", "out/gpu/EXP-047-chain",
        "--final-dir", "out/gpu/EXP-047-final",
        "--scratch", "out/gpu/EXP-047-scratch",
        "--exp-id", EXP_ID,
        "--manifest", "(EXP-047: the pinned slice manifests)",
    ]  # fmt: skip
    with installed(setup_full=None if own.smoke else setup, setup_smoke=setup if own.smoke else None):
        return gp.main(defaults + rest + (["--smoke"] if own.smoke else []))


if __name__ == "__main__":
    sys.exit(main())
