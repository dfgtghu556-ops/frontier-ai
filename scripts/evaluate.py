#!/usr/bin/env python3
"""Evaluate a checkpoint: validation loss, perplexity, and a sample continuation.

    python scripts/evaluate.py --ckpt out/cpu-smoke/best --data data/synthetic.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frontier_ai.config import ExperimentConfig  # noqa: E402
from frontier_ai.data.dataset import TokenDataset  # noqa: E402
from frontier_ai.data.tokenizer import Tokenizer  # noqa: E402
from frontier_ai.engine import checkpoint as ckpt  # noqa: E402
from frontier_ai.engine.metrics import bits_from_nats, bpb_note, loss_summary  # noqa: E402
from frontier_ai.engine.trainer import Trainer  # noqa: E402
from frontier_ai.utils.device import resolve_spec  # noqa: E402
from frontier_ai.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint directory (contains model.pt)")
    p.add_argument("--data", default=None, help="token .bin (defaults to the config it was trained on)")
    p.add_argument("--tokenizer", default=None, help="tokenizer .json")
    p.add_argument("--eval-iters", type=int, default=0, help="0 = full validation split")
    p.add_argument("--batch-size", type=int, default=0, help="0 = use the training config value")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto")
    p.add_argument("--sample", type=int, default=0, help="print N sampled tokens from the val prompt")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.ckpt)
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"[eval] {cfg_path} not found - pass --data/--tokenizer manually")
    cfg = ExperimentConfig.load(cfg_path)
    cfg.train.device = args.device
    cfg.train.precision = args.precision
    if args.eval_iters:
        cfg.train.eval_iters = args.eval_iters
    if args.batch_size:
        cfg.data.batch_size = args.batch_size

    spec = resolve_spec(cfg.train.device, cfg.train.precision)
    set_seed(cfg.train.seed)

    ds = TokenDataset(args.data or cfg.data.path)
    cfg.model.vocab_size = ds.meta.vocab_size
    model = ckpt.build_model_from_config(cfg, spec.device)
    meta = ckpt.load_checkpoint(ckpt_dir, model, map_location=str(spec.device))

    trainer = Trainer(cfg, ds, model=model, logger=None)
    val = trainer.evaluate(max_iters=cfg.train.eval_iters)
    tokens_per_byte = ds.tokens_per_byte("val")
    tokens_per_char = ds.tokens_per_char("val")
    report = loss_summary(val, tokens_per_byte=tokens_per_byte, tokens_per_char=tokens_per_char)
    # keep the precision of the pre-bits-per-byte fields so historical records
    # (EXP-001 onwards) stay comparable with new ones
    report["val_loss"] = round(val, 5)
    report["bits_per_token"] = round(bits_from_nats(val), 3)
    report.update(
        {
            "ckpt": str(ckpt_dir),
            "step": meta.get("step"),
            "n_params": model.n_params(),
            "device": spec.describe(),
            "corpus": {
                "path": str(ds.path),
                "level": ds.meta.level,
                "n_val_tokens": ds.n_val,
                "n_val_bytes": ds.meta.n_bytes_val,
                "n_val_chars": ds.meta.n_chars_val,
            },
            "note": bpb_note(tokens_per_byte is not None),
        }
    )
    print(json.dumps(report, indent=2))

    if args.sample:
        tok_path = args.tokenizer or cfg.data.tokenizer
        tok = Tokenizer.load(tok_path)
        # leave room for the continuation inside the context window
        prompt = ds.val[: cfg.model.block_size // 2].astype("int64").tolist()
        text = tok.decode(prompt)
        print(f"\n--- prompt ---\n{text[-300:]}\n--- sample ---")
        import torch

        idx = torch.tensor([prompt], dtype=torch.long, device=spec.device)
        out = model.generate(
            idx,
            max_new_tokens=args.sample,
            temperature=cfg.gen.temperature,
            top_k=cfg.gen.top_k,
            top_p=cfg.gen.top_p,
        )
        print(tok.decode(out[0].tolist()[len(prompt) :]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
