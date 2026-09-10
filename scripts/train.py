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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.config import ExperimentConfig  # noqa: E402
from frontier_ai.data.dataset import DataMeta, TokenDataset  # noqa: E402
from frontier_ai.engine.trainer import Trainer  # noqa: E402
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
        raise SystemExit(f"[train] no prepared data at {meta_path} - run scripts/prepare_data.py first")
    meta = DataMeta.load(meta_path)
    cfg.model.vocab_size = meta.vocab_size
    return cfg


def main() -> int:
    args = parse_args()
    cfg = build_config(args)

    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    spec = resolve_spec(cfg.train.device, cfg.train.precision)
    threads_for(spec.device, cfg.train.num_threads or None)

    ds = TokenDataset(cfg.data.path)
    model = GPT(cfg.model)
    print(
        f"[train] {spec.describe()} | params={model.n_params():,} | "
        f"tokens(train)={ds.n_train:,} tokens(val)={ds.n_val:,} | "
        f"eff_batch={cfg.data.batch_size * cfg.train.accum_steps} x {cfg.model.block_size} tok"
    )

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    if args.print_model:
        print(model)
        return 0

    trainer = Trainer(cfg, ds, model=model)
    if args.eval_only:
        val = trainer.evaluate()
        print(json.dumps({"val_loss": round(val, 5), "val_ppl": round(__import__("math").exp(val), 3)}))
        return 0

    result = trainer.fit()
    print(f"[train] done. best_val={result['best_val']:.4f} checkpoints in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
