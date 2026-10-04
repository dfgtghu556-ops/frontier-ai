#!/usr/bin/env python3
"""Sample text from a trained checkpoint (KV-cached decoding).

    python scripts/generate.py --ckpt out/cpu-smoke/best --prompt "the quiet cat"
    python scripts/generate.py --weights model_fp16.pt --interactive      # "try the model" (EXP-045)

``--weights`` reads a weights-only file (EXP-043's ``model_final.pt`` or the fp16 copy written by
the EXP-045 evaluation) with Frontier Tokenizer v2; the prompt is treated as the start of a
document (it follows ``<|endoftext|>``, as in training). ``--interactive`` keeps the model loaded:
type a beginning, press Enter, read the continuation; an empty line (or Ctrl+C) ends it.
"""

from __future__ import annotations

import argparse
import sys
import time
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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="checkpoint directory (contains model.pt)")
    src.add_argument("--weights", help="weights-only file (model_final.pt or the fp16 copy)")
    p.add_argument("--interactive", action="store_true", help="type beginnings in a loop")
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


WARNING = (
    "This is a BASE model: it continues text in the style of its training data. It does not answer\n"
    "questions or follow instructions, and what it writes can be wrong or nonsensical."
)


def load(args):
    """(model, tokenizer, generation settings, block size, prefix ids, device)."""
    spec = resolve_spec(args.device, "fp32")  # sampling is cheap: stay in fp32
    if args.weights:
        from frontier_ai.engine.weights import load_weights
        from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2

        model, _info = load_weights(args.weights, spec.device)
        tok = load_frontier_tokenizer_v2()
        if tok.vocab_size != model.cfg.vocab_size:
            raise SystemExit(f"[gen] tokenizer vocabulary {tok.vocab_size} != model {model.cfg.vocab_size}")
        gen = {"max_new_tokens": 200, "temperature": 0.8, "top_k": 0, "top_p": 1.0, "seed": 1}
        prefix = [tok.special_token_ids["<|endoftext|>"]]
        return model, tok, gen, model.cfg.block_size, prefix, spec.device
    ckpt_dir = Path(args.ckpt)
    cfg = ExperimentConfig.load(ckpt_dir / "config.json")
    tok = Tokenizer.load(args.tokenizer or cfg.data.tokenizer)
    model = ckpt.build_model_from_config(cfg, spec.device)
    ckpt.load_checkpoint(ckpt_dir, model, map_location=str(spec.device))
    model.eval()
    g = cfg.gen
    gen = {
        "max_new_tokens": g.max_new_tokens,
        "temperature": g.temperature,
        "top_k": g.top_k,
        "top_p": g.top_p,
        "seed": g.seed,
    }
    return model, tok, gen, cfg.model.block_size, [], spec.device


def continue_text(
    model, tok, prompt: str, prefix: list[int], block: int, device, settings: dict, sample: int
):
    """(all output ids, new ids, seconds). With a prefix (weights mode) the prompt is ordinary text."""
    ids = prefix + tok.encode_ordinary(prompt) if prefix else tok.encode(prompt)
    if len(ids) <= len(prefix):
        raise SystemExit("[gen] prompt produced no tokens")
    ids = ids[-block:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(settings["seed"] + sample)
    t = time.monotonic()
    out = model.generate(
        idx,
        max_new_tokens=settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_k=settings["top_k"] or None,
        top_p=settings["top_p"],
        generator=generator,
        stop_at_eos=prefix[0] if prefix else None,
    )
    new = [t for t in out[0, len(ids) :].tolist() if not prefix or t != prefix[0]]
    return out[0].tolist(), new, time.monotonic() - t


def main() -> int:
    args = parse_args()
    model, tok, gen, block, prefix, device = load(args)
    settings = {
        "max_new_tokens": args.max_new_tokens or gen["max_new_tokens"],
        "temperature": args.temperature or gen["temperature"],
        "top_k": args.top_k or gen["top_k"],
        "top_p": gen["top_p"] if args.top_p < 0 else args.top_p,
        "seed": args.seed or gen["seed"],
    }
    set_seed(settings["seed"])
    if args.weights:
        print(WARNING, file=sys.stderr)

    if args.interactive:
        print("Type the beginning of a text and press Enter (empty line = quit).", file=sys.stderr)
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt.strip():
                break
            _all, new, secs = continue_text(model, tok, prompt, prefix, block, device, settings, 0)
            print(prompt + tok.decode(new))
            print(
                f"[{len(new)} new tokens in {secs:.1f} s = {len(new) / max(secs, 1e-9):.1f} tokens/s]",
                file=sys.stderr,
            )
        return 0

    prompt = args.prompt if args.prompt else "\n"
    for s in range(args.num_samples):
        all_ids, new, _secs = continue_text(model, tok, prompt, prefix, block, device, settings, s)
        print(prompt + tok.decode(new) if prefix else tok.decode(all_ids))
        if args.num_samples > 1:
            print("-" * 20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
