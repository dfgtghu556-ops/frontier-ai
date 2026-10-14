"""Frontier AI research preview: the EXP-043 base model (190 M parameters) continues a text.

This file is the whole demo page of a Hugging Face Space. ``scripts/hf_demo.py`` puts it in a
bundle together with the ``frontier_ai`` package, Frontier Tokenizer v2 and the model weights in fp16
(``model_fp16.pt``). Generation runs on the Space's CPU in float32.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the bundled frontier_ai package

import torch  # noqa: E402

from frontier_ai.engine.weights import load_weights  # noqa: E402
from frontier_ai.tokenization.frozen import load_frontier_tokenizer_v2  # noqa: E402

WEIGHTS = HERE / "model_fp16.pt"
TOKENIZER = HERE / "tokenizers" / "frontier-tokenizer-v2"
MAX_NEW_TOKENS = 300

INTRO = """\
# Frontier AI - research preview (EXP-043, 190 M parameters)

A language model **trained from scratch** on 12 Indian languages and English (Assamese, Bengali,
Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu, English): one pass
over 2.78 billion tokens on free Kaggle T4 GPUs.

**What it does:** it *continues* the beginning of a text that you type.
**What it cannot do yet:** answer questions, follow instructions or chat. It is a small base model:
on the Belebele reading test it scores at chance level in 12 of 13 languages (EXP-045). It often
writes text that sounds fluent but is wrong; it may repeat itself or produce offensive text. Do not
rely on anything it writes.

Every experiment, decision and number: https://github.com/dfgtghu556-ops/frontier-ai
"""

EXAMPLES = [
    ["भारत एक विशाल देश है"],
    ["কলকাতা শহরের"],
    ["தமிழ்நாட்டின் தலைநகரம்"],
    ["తెలుగు భాష"],
    ["महाराष्ट्रातील शेतकरी"],
    ["اردو زبان"],
    ["The monsoon season in India"],
]


def load(weights: Path = WEIGHTS, tokenizer: Path = TOKENIZER):
    """(model, tokenizer, info). The tokenizer is checked against its freeze record."""
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model, info = load_weights(weights, "cpu")
    tok = load_frontier_tokenizer_v2(tokenizer)
    if tok.vocab_size != model.cfg.vocab_size:
        raise ValueError(f"tokenizer vocabulary {tok.vocab_size} != model {model.cfg.vocab_size}")
    return model, tok, info


def continue_text(
    model, tok, prompt: str, max_new_tokens: int = 120, temperature: float = 0.8, seed: int = 1
):
    """The model's continuation of ``prompt`` (the prompt is the start of a new document)."""
    eot = tok.special_token_ids["<|endoftext|>"]
    body = tok.encode_ordinary(prompt or "")
    if not body:
        return ""
    max_new_tokens = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))
    keep = model.cfg.block_size - max_new_tokens - 1  # room for the new tokens
    ids = [eot] + body[-keep:]
    idx = torch.tensor([ids], dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=max(float(temperature), 1e-3),
            top_k=None,
            top_p=1.0,
            generator=generator,
            stop_at_eos=eot,
        )
    new = [t for t in out[0, len(ids) :].tolist() if t != eot]
    return tok.decode(new)


def build_ui(model, tok, info: dict):
    import gradio as gr

    def run(prompt, length, temperature, seed):
        if not (prompt or "").strip():
            return "Type the beginning of a text first."
        return prompt + continue_text(model, tok, prompt, length, temperature, seed)

    source = info.get("source", {})
    footer = (
        f"Model: {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"training step {source.get('step', '?')}, weights sha256 (float32 original) "
        f"{str(source.get('sha256', '?'))[:16]}..."
    )
    with gr.Blocks(title="Frontier AI research preview") as demo:
        gr.Markdown(INTRO)
        prompt = gr.Textbox(label="Beginning of a text", lines=3, placeholder="भारत एक विशाल देश है")
        with gr.Row():
            length = gr.Slider(20, MAX_NEW_TOKENS, value=120, step=10, label="New tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature (randomness)")
            seed = gr.Number(value=1, precision=0, label="Seed (same seed = same text)")
        button = gr.Button("Continue the text", variant="primary")
        output = gr.Textbox(label="Model output (your text + the model's continuation)", lines=10)
        gr.Examples(EXAMPLES, inputs=[prompt])
        gr.Markdown(footer)
        button.click(run, inputs=[prompt, length, temperature, seed], outputs=output)
    return demo


def main() -> None:
    model, tok, info = load()
    extra = HERE / "MODEL_INFO.json"
    if extra.exists():
        info = {**info, **json.loads(extra.read_text(encoding="utf-8"))}
    build_ui(model, tok, info).queue(default_concurrency_limit=1).launch()


if __name__ == "__main__":
    main()
