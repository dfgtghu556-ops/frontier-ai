#!/usr/bin/env python3
"""Sample text from a trained checkpoint (KV-cached decoding).

    python scripts/generate.py --ckpt out/cpu-smoke/best --prompt "the quiet cat"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from frontier_ai.config import ExperimentConfig  # noqa: E402
from frontier_ai.data.tokenizer import Tokenizer  # noqa: E402
from frontier_ai.engine import checkpoint as ckpt  # noqa: E402
from frontier_ai.utils.device import resolve_spec  # noqa: E402
from frontier_ai.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint directory (contains model.pt)")
    p.add_argument("--tokenizer", default=None, help="override tokenizer path")
    p.add_argument("--prompt", default="", help="conditioning text (empty = newline)")
    p.add_argument("--max-new-tokens", type=int, default=0, help="0 = use config value")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = use config value")
    p.add_argument("--top-k", type=int, default=0, help="0 = use config value")
    p.add_argument("--top-p", type=float, default=-1.0, help="<0 = use config value")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0, help="0 = use config value")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.ckpt)
    cfg = ExperimentConfig.load(ckpt_dir / "config.json")

    gen = cfg.gen
    max_new = args.max_new_tokens or gen.max_new_tokens
    temp = args.temperature or gen.temperature
    top_k = args.top_k or gen.top_k
    top_p = gen.top_p if args.top_p < 0 else args.top_p
    seed = args.seed or gen.seed

    spec = resolve_spec(args.device, "fp32")  # sampling is cheap: stay in fp32
    set_seed(seed)

    tok = Tokenizer.load(args.tokenizer or cfg.data.tokenizer)
    model = ckpt.build_model_from_config(cfg, spec.device)
    ckpt.load_checkpoint(ckpt_dir, model, map_location=str(spec.device))
    model.eval()

    prompt = args.prompt if args.prompt else "\n"
    ids = tok.encode(prompt)
    if not ids:
        raise SystemExit("[gen] prompt produced no tokens")
    ids = ids[-cfg.model.block_size :]
    idx = torch.tensor([ids], dtype=torch.long, device=spec.device)

    for s in range(args.num_samples):
        generator = torch.Generator().manual_seed(seed + s)
        out = model.generate(
            idx,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=top_k or None,
            top_p=top_p,
            generator=generator,
        )
        print(tok.decode(out[0].tolist()))
        if args.num_samples > 1:
            print("-" * 20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
