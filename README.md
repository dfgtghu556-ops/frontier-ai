# frontier-ai

A small, readable, **GPU-ready** PyTorch stack for training decoder-only transformer
language models. The same code that runs a 10-second smoke test on 2 CPU cores runs a
multi-hour job on an A100 — you change a JSON config, not the code.

```bash
git clone https://github.com/dfgtghu556-ops/frontier-ai.git && cd frontier-ai
pip install -e ".[dev]"

make data                # prepare a synthetic corpus (no downloads)
make train               # ~10 s on a laptop CPU: 139k-param GPT, val loss 3.9 -> 1.3
make generate            # sample from the checkpoint
```

```text
[    0.1s] event=train  step=20   loss=6.2112  ppl=497.0    lr=2.9e-03  gnorm=0.50
[    2.0s] event=eval   step=50   val_loss=1.86  val_ppl=6.4   improved=True
[    7.6s] event=eval   step=200  val_loss=1.28  val_ppl=3.6   improved=True
[    7.6s] event=run.end  steps=200  best_val=1.2787  throughput=53.6k tok/s
```

---

## Why this exists

Most "train a GPT" repos are either 300-line single-file demos or 30k-line frameworks.
This sits in between: ~1,600 lines of typed, tested, dependency-light code
(`torch` + `numpy`) covering the whole loop — data → model → train → eval → checkpoint →
sample — with the parts that actually matter for GPU training done properly
(mixed precision, gradient accumulation, cosine LR, grad clipping, KV-cached sampling).

## Quickstart

### 1. Prepare data

```bash
# built-in synthetic corpus (zero downloads, deterministic, great for smoke tests)
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic

# or your own text file
python scripts/prepare_data.py --source data/raw/my_corpus.txt --level word --min-count 2 --out data/mine
```

This writes three files:

| file | contents |
| --- | --- |
| `data/<name>.bin` | flat `numpy` array of token ids (memory-mapped at train time) |
| `data/<name>.meta.json` | token count, vocab size, dtype, train/val split |
| `data/<name>.tokenizer.json` | the fitted vocabulary (char- or word-level) |

### 2. Train

```bash
python scripts/train.py --config configs/cpu_smoke.json          # ~10 s, CPU
python scripts/train.py --config configs/gpu_1x.json             # single GPU, ~124M params
python scripts/train.py --config configs/gpu_1x.json \
    --set data.path=data/mine.bin --set data.tokenizer=data/mine.tokenizer.json \
    --set train.out_dir=out/mine --set optim.lr=6e-4
```

Any config field can be overridden with `--set section.key=value`; repeat the flag as
often as you like. The resolved config is written to `<out_dir>/config.json`, metrics go
to `<out_dir>/train.jsonl` (one JSON object per line), and checkpoints land in
`<out_dir>/{best,last,step-N}/`.

### 3. Evaluate and sample

```bash
python scripts/evaluate.py --ckpt out/cpu-smoke/best --data data/synthetic.bin --eval-iters 20
python scripts/generate.py --ckpt out/cpu-smoke/best \
    --tokenizer data/synthetic.tokenizer.json --prompt "the quiet cat" --max-new-tokens 200
```

### 4. Resume

```bash
python scripts/train.py --config configs/cpu_smoke.json --set train.max_steps=500 --set train.init_from=resume
```

Weights, optimizer state, scheduler position, step counter and best score are all
restored, so the LR schedule continues where it left off.

## Layout

```
configs/            cpu_smoke.json (tiny, CI-safe) · gpu_1x.json (single-GPU)
scripts/            prepare_data.py · train.py · evaluate.py · generate.py
src/frontier_ai/
    config.py       dataclass configs + JSON load/save + `--set` overrides
    data/
        tokenizer.py    char / word tokenizers, JSON-serializable
        dataset.py      memmap token store, train/val split, batch sampling
        synthetic.py    deterministic pseudo-English corpus (smoke-test fixture)
    model/gpt.py    the model: RMSNorm · SwiGLU · RoPE · GQA · KV cache
    engine/
        trainer.py      the training loop
        optim.py        AdamW + warmup/cosine scheduler
        checkpoint.py   save / load / resume
    utils/          device+dtype resolution, seeding, JSONL logging
tests/              47 unit + end-to-end tests (CPU, ~10 s)
```

## Model

A LLaMA-flavoured decoder-only transformer, ~250 lines in
[`src/frontier_ai/model/gpt.py`](src/frontier_ai/model/gpt.py):

- pre-norm blocks, RMSNorm (or LayerNorm), SwiGLU (or GELU) feed-forward
- grouped-query attention (`n_kv_head < n_head`) with `scaled_dot_product_attention`,
  so flash / mem-efficient kernels are used automatically when available
- learned absolute positions or RoPE; tied embeddings; no biases by default
- KV-cached `generate()` with top-k / top-p / temperature sampling
- optional gradient checkpointing and `torch.compile`

Sizing knobs live in `ModelConfig`: `n_layer`, `n_head`, `n_embd`, `block_size`,
`n_kv_head`, `ffn_mult`, `norm`, `ffn`, `pos`, `tie_embeddings`.

## Training loop

[`src/frontier_ai/engine/trainer.py`](src/frontier_ai/engine/trainer.py) implements:

| feature | config |
| --- | --- |
| device selection (cuda → mps → cpu) | `train.device` |
| mixed precision (bf16 on Ampere+, fp16 fallback, fp32 on CPU) | `train.precision` |
| gradient accumulation (effective batch = `batch_size × accum_steps`) | `data.batch_size`, `train.accum_steps` |
| AdamW with decoupled weight decay + matrix-only decay groups | `optim.*` |
| warmup → cosine decay to `min_lr_ratio`, grad-norm clipping | `optim.*` |
| periodic eval, best-checkpoint tracking, periodic snapshots | `train.eval_interval`, `train.save_interval` |
| resume, seeding, `torch.compile`, gradient checkpointing | `train.*` |

Nothing in the loop is CPU-specific: `resolve_spec()` inspects the machine and picks the
device, autocast dtype and GradScaler policy (fp16 needs a scaler, bf16/fp32 do not).

### Running on a GPU

No code changes — just a config and, for multi-GPU, a launcher:

```bash
# single GPU
python scripts/train.py --config configs/gpu_1x.json

# 8 GPUs (data parallel): torchrun + DDP-friendly batch/accum settings
torchrun --standalone --nproc_per_node=8 scripts/train.py --config configs/gpu_1x.json \
    --set data.batch_size=16 --set train.accum_steps=4
```

`configs/gpu_1x.json` targets ~124M parameters (12 layers, 768 dim, 1024 context,
4 KV heads, RoPE) and reaches ~196k tokens per optimizer step at `batch_size=24,
accum_steps=8`. For a 4090/A100-class card, expect bf16 autocast to be selected
automatically; set `train.precision=fp16` if you are on older (pre-Ampere) hardware.

> Note: this repository was developed on a CPU-only sandbox, so the GPU numbers above
> are sizing guidance, not benchmarks. The code paths are device-agnostic and the
> CPU path is exercised end-to-end by CI.

## Using your own data

1. Drop raw text anywhere (e.g. `data/raw/book.txt`).
2. `python scripts/prepare_data.py --source data/raw/book.txt --level word --out data/book`
   (use `--level char` for small corpora or non-Latin scripts — it needs no vocabulary cutoff).
3. `python scripts/train.py --config configs/cpu_smoke.json --set data.path=data/book.bin
   --set data.tokenizer=data/book.tokenizer.json --set model.block_size=256`

Tips: the vocabulary is defined by the prepared corpus, so `model.vocab_size` is set
automatically at train time; `data.val_frac` (default 0.1) controls the held-out split;
word-level tokenizers accept `--min-count N` to cap the vocab.

## CI

A GitHub Actions workflow ([`docs/ci.yml.example`](docs/ci.yml.example)) installs CPU
PyTorch, lints with `ruff`, runs the test suite, and then drives the whole CLI
(prepare → train → evaluate → generate) on a 60k-character corpus. Enable it with:

```bash
mkdir -p .github/workflows && cp docs/ci.yml.example .github/workflows/ci.yml
```

## Tests

```bash
pytest -q        # 47 tests, ~10 s on CPU
ruff check .     # lint
make test lint
```

The suite covers the model (causality, architecture variants, KV-cache equivalence,
context-length clamping), data (tokenizer round-trips, split math, deterministic
corpus), the engine (schedules, checkpoint round-trips, config overrides), and full
train → checkpoint → resume → sample runs. Two tests exist specifically to catch bugs
found while building this: logged loss must be a *mean* over accumulation steps, and
repeated `--set` flags must accumulate.

## Roadmap

- [ ] DDP launcher + `torchrun` example config (code is DDP-friendly; not yet exercised)
- [ ] Optional HuggingFace BPE tokenizer behind the same `Tokenizer` interface
- [ ] WandB / TensorBoard metric sink behind `RunLogger`
- [ ] Throughput + MFU reporting against device peak FLOPs

## License

MIT — see [LICENSE](LICENSE).
