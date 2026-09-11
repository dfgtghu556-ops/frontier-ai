#!/usr/bin/env python3
"""Train a small GPT model.

Identical code path on CPU and CUDA - pick a config (or override any field).

Examples
--------
    # 1-minute CPU smoke test (prepares synthetic data too if missing)
    python scripts/train.py --config configs/cpu_smoke.json

    # GPU run
    python scripts/train.py --config configs/gpu_1x.json --set train.out_dir=out/gpu \
        --set data.path=data/shakes.bin --set data.tokenizer=data/shakes.tokenizer.json

    # resume
    python scripts/train.py --config configs/cpu_smoke.json --set train.init_from=resume
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.config import ExperimentConfig  # noqa: E402
from frontier_ai.data.dataset import DataMeta, TokenDataset  # noqa: E402
from frontier_ai.engine.trainer import Trainer  # noqa: E402
from frontier_ai.experiments import (
    ExperimentInputError,  # noqa: E402
    ExperimentSpec,  # noqa: E402
)
from frontier_ai.experiments.autowire import run_self_recorded  # noqa: E402
from frontier_ai.model.gpt import GPT  # noqa: E402
from frontier_ai.utils.device import resolve_spec, threads_for  # noqa: E402
from frontier_ai.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/cpu_smoke.json", help="path to a JSON config")
    p.add_argument(
        "--set",
        nargs="+",
        action="extend",  # repeated --set flags accumulate instead of overwriting
        default=[],
        metavar="section.key=value",
        help="config overrides",
    )
    p.add_argument("--max-steps", type=int, default=None, help="shortcut for --set train.max_steps=N")
    p.add_argument("--device", default=None, help="shortcut for --set train.device=cpu|cuda")
    p.add_argument("--eval-only", action="store_true", help="evaluate the (resumed) model and exit")
    p.add_argument("--print-model", action="store_true", help="print the model and exit")
    p.add_argument("--exp-id", default="EXP-000",
                   help="experiment id for the automatic record (default: EXP-000)")
    p.add_argument("--no-record", action="store_true",
                   help="do not write an experiment record for this run")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.load(args.config) if Path(args.config).exists() else ExperimentConfig()
    overrides = list(args.set)
    if args.max_steps is not None:
        overrides.append(f"train.max_steps={args.max_steps}")
    if args.device:
        overrides.append(f"train.device={args.device}")
    cfg = cfg.with_overrides(overrides)

    # the prepared corpus defines the vocabulary and the block-split boundaries
    meta_path = Path(cfg.data.path).with_suffix(".meta.json")
    if not meta_path.exists():
        raise ExperimentInputError(
            f"[train] no prepared data at {meta_path} - run scripts/prepare_data.py first"
        )
    meta = DataMeta.load(meta_path)
    cfg.model.vocab_size = meta.vocab_size
    return cfg


def setup(args: argparse.Namespace) -> tuple[ExperimentConfig, object, TokenDataset, GPT, Path]:
    """Everything a training or inspection run needs, in the order the CLI always used.

    Returns ``(cfg, device_spec, dataset, model, out_dir)``. Raising
    :class:`ExperimentInputError` (a ``FileNotFoundError``) for missing prepared data
    keeps that failure inside the recorded lifecycle instead of exiting before it.
    """
    cfg = build_config(args)

    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    device_spec = resolve_spec(cfg.train.device, cfg.train.precision)
    threads_for(device_spec.device, cfg.train.num_threads or None)

    ds = TokenDataset(cfg.data.path)
    model = GPT(cfg.model)
    print(
        f"[train] {device_spec.describe()} | params={model.n_params():,} | "
        f"tokens(train)={ds.n_train:,} tokens(val)={ds.n_val:,} | "
        f"eff_batch={cfg.data.batch_size * cfg.train.accum_steps} x {cfg.model.block_size} tok"
    )

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")
    return cfg, device_spec, ds, model, out_dir


def _override_value(overrides: list[str], key: str) -> str | None:
    """Last ``key=value`` override for ``key``, or None (CLI wins over the file)."""
    found = None
    for item in overrides or []:
        if item.startswith(f"{key}="):
            found = item.split("=", 1)[1]
    return found


def peek_config(args: argparse.Namespace) -> dict:
    """Best-effort spec fields read straight from the config file and ``--set``.

    The record has to exist before the body runs, but building the real
    :class:`ExperimentConfig` belongs *inside* the body so that a broken config or a
    missing corpus produces a **failed record** instead of a crash with nothing recorded.
    This reader never validates and never raises: it only needs the identity of the run
    (seed, output directory, declared data), which the body then resolves properly.
    """
    raw: dict = {}
    try:
        if Path(args.config).exists():
            raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    train = dict(raw.get("train") or {})
    data = dict(raw.get("data") or {})

    out_dir = _override_value(args.set, "train.out_dir") or train.get("out_dir") or "out/cpu-smoke"
    seed_raw = _override_value(args.set, "train.seed")
    if seed_raw is None:
        seed_raw = train.get("seed", 1337)
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError):
        seed = 1337
    data_path = _override_value(args.set, "data.path") or data.get("path", "")
    deterministic_raw = _override_value(args.set, "train.deterministic")
    if deterministic_raw is None:
        deterministic = bool(train.get("deterministic", False))
    else:
        deterministic = str(deterministic_raw).strip().lower() in {"1", "true", "yes"}

    return {
        "seed": seed,
        "output_dir": str(out_dir),
        "data_paths": [str(data_path)] if data_path else [],
        "deterministic": deterministic,
    }


def main() -> int:
    args = parse_args()

    # Inspection runs produce no artifacts, so they keep their exact old behaviour and
    # write no record (D-032): --print-model prints the model, --eval-only scores it.
    if args.print_model:
        _, _, _, model, _ = setup(args)
        print(model)
        return 0

    if args.eval_only:
        cfg, device_spec, ds, model, _ = setup(args)
        trainer = Trainer(cfg, ds, model=model)
        val = trainer.evaluate()
        report = trainer.loss_report(val)
        # historical precision for the two pre-bits-per-byte fields
        report["val_loss"] = round(val, 5)
        report["val_ppl"] = round(math.exp(val), 3)
        print(json.dumps(report, indent=2))
        return 0

    def body() -> dict:
        cfg, device_spec, ds, model, out_dir = setup(args)
        trainer = Trainer(cfg, ds, model=model)
        result = trainer.fit()
        bpb = result.get("best_bpb")
        bpb_txt = (
            f" best_bpb={bpb:.4f}" if bpb is not None
            else " (bits/byte unknown: no corpus byte counts)"
        )
        print(f"[train] done. best_val={result['best_val']:.4f}{bpb_txt} checkpoints in {out_dir}")
        # no paths in `results`: two runs into different directories must fingerprint the same
        return {
            "best_val": result["best_val"],
            "best_bpb": bpb,
            "steps": result["steps"],
            "n_params": model.n_params(),
            "tokens_seen": trainer.state.tokens_seen,
            "device": device_spec.describe(),
        }

    if args.no_record:
        body()
        return 0

    # training produces artifacts (checkpoints, config, metrics) -> record it
    peeked = peek_config(args)
    def build_spec() -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=args.exp_id,
            seed=peeked["seed"],
            name=f"train {Path(args.config).stem}",
            output_dir=peeked["output_dir"],
            params={
                "config": args.config,
                "overrides": list(args.set),
                "max_steps": args.max_steps,
                "device": args.device,
            },
            data_paths=peeked["data_paths"],
            config_path=args.config if Path(args.config).exists() else "",
            command=list(sys.argv),
            tags=["project-001", "train"],
            deterministic_mode=peeked["deterministic"],
        )
    return run_self_recorded(build_spec, body).exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
