# PROJECT_CONTEXT.md

> Read this file first if you are an AI agent (or human) opening this repository in a
> new chat with no other context. It describes the mission, what exists today, what has
> been proven, what has not, and where to start.

---

## CURRENT POSITION — START HERE

**Project 001 is COMPLETE. The next development task is Project 002, which has NOT started yet.**

- Project 001 = the CPU-first, GPU-ready PyTorch GPT training pipeline in this repository
  (data → model → training loop → evaluation → checkpointing → sampling, plus 47 tests).
- It is committed on branch `arena/01a08a78-frontier-ai` and is open as **PR #1** against `main`.
- **Do not start Project 002 until the user explicitly asks for it.**
- The most likely shape of Project 002 is "make experiments trustworthy and measurable"
  (better tokenizer, real data pipeline, deterministic reproducible runs, an experiment
  runner) — see [ROADMAP.md](ROADMAP.md) Stages 1–3. Confirm the actual scope with the user
  before writing code; the roadmap is a proposal, not an approved plan.
- Before changing any code, read: this file, [ROADMAP.md](ROADMAP.md),
  [DECISIONS.md](DECISIONS.md), [EXPERIMENTS.md](EXPERIMENTS.md), and [README.md](README.md).
- Record any experiment you run in [EXPERIMENTS.md](EXPERIMENTS.md) using the standard
  template. Never record a number you did not actually observe.

---

## 1. Mission and long-term goal

The long-term goal is to build an **independent Indian frontier AI company/lab**: an
organization that can design, train, evaluate, and deploy its own foundation models,
with the research capability to push beyond published state of the art over time.

Concretely, "frontier lab" here means we eventually own the entire stack end to end:

| Layer | Must be ours |
|---|---|
| Tokenizer | trained on our own data, with coverage for Indian languages |
| Model architecture | our design choices, our code |
| **Weights** | **trained from random initialization by us — not a fine-tune of someone else's model as the core product** |
| Data pipeline | our collection, licensing, filtering, deduplication, mixing, and versioning |
| Pre-training system | our training loop, distributed strategy, observability |
| Post-training | our SFT / preference / reasoning / safety recipes |
| Evaluation | our benchmarks and harnesses, including Indic-language evaluation |
| Inference | our serving path, quantization, and distillation if used |
| Research | our own experiments, ablations, and publications |

## 2. What "ours" means: external infrastructure is allowed

Using external, open infrastructure is **allowed and expected**. The intent is not to
reimplement an operating system; it is to own the *intelligence*.

**Allowed (infrastructure / tooling):** Linux, Python, PyTorch, CUDA/ROCm, Triton,
NCCL, NumPy, Hugging Face *libraries* (`tokenizers`, `datasets`, `transformers` as
reference implementations and tooling), datasets with permissive or otherwise cleared
licenses, open-weight models as **baselines for comparison**, cloud GPUs, experiment
trackers, and standard DevOps tooling.

**Must be ours (core value):** the trained weights of our foundation models, the data
recipe and data pipeline that produced them, the tokenizer, the training/eval/inference
systems we run in production, and the research results we publish.

**Not acceptable as the foundation of the company:**

- Wrapping a proprietary model API (OpenAI/Anthropic/Google/etc.) and presenting it as
  our own model. APIs may be used for *tooling, data generation assistance, or
  evaluation baselines* — never as the core model intelligence.
- Shipping someone else's weights as our own foundation model. Fine-tuning an open-weights
  model is acceptable for *experiments, baselines, and short-term product features*, but
  the foundation-model track trains from random initialization.
- Using data whose license or provenance we cannot defend.

## 3. Two tracks: product track vs foundation-model track

These are separate tracks with different clocks. Do not let one block the other, and do
not confuse their success criteria.

**Foundation-model track (this repository).** Long-horizon research and training:
tokenizer, data pipeline, pre-training, evaluation, post-training, and eventually
frontier-scale models. Progress is measured in reproducible experiment results, not
shipping dates. This repository is the home of that work.

**Product track.** User-facing applications and the control-center UI (built in Lovable,
kept **out of** this repository — see [DECISIONS.md](DECISIONS.md) D-006). Products may
consume whatever model is best available at the time, including third-party APIs in the
short term, so that product iteration is never blocked on research. Product revenue and
usage can fund compute.

**The bridge.** As the foundation track produces models that beat the third-party option
for a given product surface, that product migrates onto our weights. Migration is a
deliberate, evaluated decision — not an automatic one.

## 4. Purpose of this repository

A single, well-tested PyTorch codebase for training decoder-only transformer language
models, starting small and growing by experiment. Its job is to be the **substrate** for
every future research step: the same training loop that runs a 10-second CPU smoke test
should, without modification, run a multi-GPU pre-training job.

It is intentionally *not* yet: a data company, a distributed-training framework, an
evaluation suite, or a serving stack. Those come later, built on top of this substrate.

## 5. What Project 001 built

Project 001 took this repository from an empty stub (`# frontier-ai`, one commit) to a
complete, tested training pipeline:

- **Data**: char-level and word-level tokenizers (JSON-serializable), a memory-mapped
  token store with train/val split and random-offset batch sampling, and a deterministic
  synthetic corpus generator so smoke tests need no downloads.
- **Model**: a ~250-line decoder-only transformer with RMSNorm, SwiGLU, optional RoPE,
  grouped-query attention, `scaled_dot_product_attention`, tied embeddings, KV-cached
  sampling, optional gradient checkpointing and optional `torch.compile`.
- **Engine**: training loop with gradient accumulation, mixed precision, warmup→cosine
  AdamW, grad-norm clipping, decay/no-decay parameter groups, best/periodic checkpoints,
  resume, and JSONL metric logging.
- **Interfaces**: `prepare_data.py`, `train.py`, `evaluate.py`, `generate.py`; JSON configs
  (`configs/cpu_smoke.json`, `configs/gpu_1x.json`) with `--set section.key=value` overrides.
- **Quality gates**: 47 tests (unit + end-to-end), `ruff` lint, and a GitHub Actions
  workflow example.

## 6. Current repository structure

```
configs/
    cpu_smoke.json            tiny config: 139k params, trains in ~10 s on CPU
    gpu_1x.json               single-GPU config (~124M params) — UNTESTED on real hardware
docs/
    ci.yml.example            GitHub Actions workflow (copy to .github/workflows to enable)
scripts/
    prepare_data.py           text or synthetic corpus -> tokens.bin + tokenizer.json + meta.json
    train.py                  build model from config, run training, save checkpoints
    evaluate.py               score a checkpoint: val loss, perplexity, bits/token, sample
    generate.py               sample text with the KV cache (temperature / top-k / top-p)
src/frontier_ai/
    config.py                 dataclass configs, JSON load/save, --set overrides, validation
    data/
        tokenizer.py          CharTokenizer / WordTokenizer, save/load
        dataset.py            memmap token store, train/val split, batch sampling
        synthetic.py          deterministic pseudo-English corpus (a test fixture)
    model/gpt.py              the model: norms, RoPE, attention, MLP, blocks, sampling
    engine/
        trainer.py            the training loop
        optim.py              AdamW builder + warmup/cosine LR scheduler
        checkpoint.py         save / load / find_latest / build model from config
    utils/
        device.py             device + dtype + GradScaler policy resolution
        seed.py               seeding helpers
        logging.py            console + JSONL logger
tests/
    test_model.py             architecture, causality, KV-cache equivalence, context limits
    test_data.py              tokenizer round-trips, split math, memmap integrity
    test_engine.py            schedules, checkpoint round-trips, config overrides
    test_train_smoke.py       end-to-end train / resume / sample / accumulation
PROJECT_CONTEXT.md            this file
ROADMAP.md                    staged plan from here to frontier scale
DECISIONS.md                  architectural decision records
EXPERIMENTS.md                experiment log (template + EXP-001)
README.md                     user-facing quickstart
Makefile, pyproject.toml, LICENSE (MIT), .gitignore
```

Ignored (never committed): `data/` (prepared corpora), `out/` (runs and checkpoints),
`.venv/`, caches.

## 7. Current model architecture and important implementation details

`src/frontier_ai/model/gpt.py`, configured by `ModelConfig` in `config.py`.

Defaults in the shipped smoke config (`configs/cpu_smoke.json`):

| Setting | Value |
|---|---|
| `n_layer` | 2 |
| `n_head` / `n_kv_head` | 2 / 2 (so: ordinary multi-head attention; GQA is available via `n_kv_head`) |
| `n_embd` | 64 (`head_dim` = 32) |
| `block_size` | 64 |
| `ffn` / `ffn_hidden` | SwiGLU / 256 |
| `norm` | RMSNorm |
| `pos` | learned absolute (RoPE available via `pos="rope"`) |
| `tie_embeddings` | true |
| `vocab_size` | 51 (set automatically from the prepared corpus) |
| **Total parameters** | **138,752** |

Parameter distribution for that config: MLP 98,304 (70.8%), attention 32,768 (23.6%),
embeddings 7,360 (5.3%), norms 320 (0.2%).

Implementation details that matter for future work:

- **Pre-norm blocks with residual adds**: `x = x + Attention(RMSNorm(x))`, then
  `x = x + MLP(RMSNorm(x))`, followed by a final RMSNorm.
- **Attention** uses `F.scaled_dot_product_attention` with `is_causal=True` during
  prefill, so PyTorch selects flash / memory-efficient kernels automatically on supported
  hardware. When a KV cache is passed, `is_causal=False` (a single query attends to all
  cached keys, so no mask is needed).
- **GQA** expands shared K/V heads with `repeat_interleave` when `n_kv_head < n_head`.
- **SwiGLU** does one big projection to `2 × ffn_hidden`, splits it, applies SiLU to one
  half and multiplies by the other (the gate).
- **RoPE** caches cos/sin tables and rebuilds them on demand, so decoding past
  `block_size` cannot crash; the table starts at `start_pos` for cached decoding.
- **Tied embeddings**: `lm_head.weight` *is* `token_emb.weight`, so it is stored and
  counted once.
- **Generation** truncates over-long prompts to `block_size`, caps `max_new_tokens` to the
  remaining context, and uses a per-block `(k, v)` KV cache with `start_pos`.
- **Initialization**: normal(0, 0.02) everywhere, with a scaled residual init
  (`0.02 / sqrt(2 · n_layer)`) on the output projections.
- **Optional** and off by default: gradient checkpointing, `torch.compile`.

## 8. Current training pipeline

`src/frontier_ai/engine/trainer.py`, driven by `configs/*.json` plus `--set` overrides.

One optimizer step:

```
zero_grad  →  for each of accum_steps micro-batches:
                 forward under autocast → loss / accum_steps → scale → backward
           →  unscale (fp16 only) → clip_grad_norm_(1.0) → optimizer.step
           →  scaler.update → scheduler.step
```

Smoke-config step economics: `batch_size=8 × block_size=64 × accum_steps=4`
= **2,048 tokens per optimizer step**; 200 steps = 409,600 tokens ≈ 2.3 passes over the
180,085-token training split.

- **Optimizer**: AdamW, betas (0.9, 0.95), weight decay 0.1 applied only to 2-D matrices
  (1-D norm scales and biases are excluded).
- **Schedule**: linear warmup over `warmup_steps` (20) to `lr` (3e-3), then cosine decay to
  `min_lr_ratio · lr` (10%) at `max_steps`.
- **Evaluation**: every `eval_interval` steps, average loss over `eval_iters` validation
  batches under `no_grad`; the best-scoring model is saved to `out/<run>/best`.
- **Checkpointing**: `last` at the end, `step-N` every `save_interval` if set.
- **Logging**: one JSON object per event in `out/<run>/train.jsonl`, plus a console line.
- **Metrics logged**: loss, perplexity, LR, gradient norm, tokens seen, tokens/sec, ETA.

## 9. Current tokenizer and data approach

- **Tokenizers**: `CharTokenizer` (one token per character, plus a `\ufffd` replacement
  token) and `WordTokenizer` (regex word/punctuation tokens with `<unk>` and an optional
  `--min-count` cutoff). Both are invertible and saved as JSON.
- **Storage**: token ids are written with NumPy `tofile()` as the smallest integer type
  that fits the vocabulary (`uint8` ≤ 256, `uint16` ≤ 65,536, else `int32`), accompanied by
  `<name>.meta.json` recording `n_tokens`, `vocab_size`, `dtype`, `level`, `n_train`, `n_val`.
  At training time the file is opened read-only through `np.memmap`, so corpora larger than
  RAM are possible.
- **Split**: `val_frac` (default 0.1) of the **tail** of the token stream — a contiguous
  split, not a shuffle and not document-aware.
- **Batching**: uniform random start offsets sampled with NumPy; `x = tokens[o : o+T]`,
  `y = tokens[o+1 : o+1+T]`. No packing, no document-boundary masking, no shuffling across
  boundaries.
- **Default corpus**: a deterministic synthetic pseudo-English corpus generated from a
  small grammar (`data/synthetic.py`). It exists so smoke tests and CI need no network
  access and so loss demonstrably falls in seconds. **It is a test fixture, not training
  data for any real model.**

## 10. Current evaluation and testing approach

**Testing (automated, 47 tests, ~10 s on CPU, `pytest -q`; `ruff check .` clean):**

- Model: shapes and initial loss near `ln(vocab)`; **causality** (changing tokens after
  position *t* must not change logits before *t*); every architecture variant
  (RMSNorm/LayerNorm × SwiGLU/GELU × learned/RoPE); GQA; **KV-cache equivalence**
  (incremental decoding reproduces a full-context forward); context-length clamping;
  tied-embedding storage sharing; parameter-group split; top-k masking.
- Data: tokenizer round-trips, `<unk>` handling, JSON save/load, dtype selection, split
  arithmetic, shift-by-one target construction, memmap integrity, corpus determinism.
- Engine: warmup+cosine schedule values, constant schedule, optimizer group stepping,
  checkpoint weight/optimizer round-trips, `find_latest`, config override validation.
- End-to-end (`test_train_smoke.py`): loss decreases over a real run; more steps beats
  fewer steps; resume restores step counter and continues improving; generated text is
  well-formed; **logged loss must be a mean over accumulation steps** (regression test for
  a bug found during development); gradient checkpointing runs; fp32/auto precision both run.

**Model evaluation (today):** validation loss, perplexity (`exp(loss)`), and
bits-per-token (`loss / ln 2`) on the held-out tail of the same corpus, plus a qualitative
text sample. There is **no** downstream task evaluation, no standardized benchmark, no
per-language evaluation, and no comparison against published models.

## 11. What Project 001 proves

- The complete pipeline works end to end: text → tokens → batches → model → loss →
  gradients → updates → checkpoints → resume → sampling.
- The shift-by-one next-token target construction is correct (loss would not fall otherwise).
- Causal masking is correct (proved by a dedicated test, not by eyeballing output).
- Learning actually happens: on the synthetic corpus, validation loss fell from **3.93**
  (chance, = ln 51) to **1.3776** (perplexity 3.965) in 200 steps / 409,600 tokens, in
  **7.7 seconds** at a mean **53.2k tokens/sec** on 2 CPU cores. Logged eval trajectory:
  2.326 (step 50) → 1.780 (100) → 1.472 (150) → 1.378 (200).
- Gradient flow, clipping, the LR schedule, and AdamW all behave sanely (gradient norms
  0.5–1.6 across the run).
- Checkpoint/resume is faithful: a test loads `last/` into a fresh model and requires the
  loss to match to 1e-5; resuming from step 200 continued to step 230 with the cosine LR
  picking up where it left off.
- The code is genuinely device-agnostic: the same path ran with `amp=off` on CPU and would
  run unchanged on CUDA.
- Two real bugs were caught by the tests during development: loss summed instead of
  averaged over accumulation steps (read 4× high), and repeated `--set` flags overwriting
  each other instead of accumulating.

## 12. What Project 001 does NOT prove

- **Nothing about GPU performance.** No CUDA device was available. bf16/fp16 autocast,
  flash kernels, the GradScaler path, and `torch.compile` have never executed. The
  "1–2 hours on an A100" note in the README is sizing arithmetic, not a measurement.
- **Nothing about scale.** 2 layers / 64 dim / 64 context says nothing about 12 layers /
  768 dim / 1024 context, let alone larger. Memory, numerics, and stability all change
  character with size.
- **Nothing about distributed training.** The `torchrun` example in the README is
  untested; there is no DDP/FSDP wrapping, no rank-aware data sharding or logging.
- **Nothing about real language.** The corpus is machine-generated pseudo-English with ~100
  words and repeating counting patterns; val loss ≈ train loss by construction.
- **Nothing about tokenization quality.** A 51-character vocabulary is not a BPE tokenizer;
  bits-per-character is not comparable to published bits-per-token numbers.
- **Nothing about hyperparameters.** lr=3e-3 suits a 139k model and is wrong for a 124M one.
  There are no sweeps and no multi-seed runs.
- **Nothing about long-run stability.** 200 steps cannot surface loss spikes, NaNs, data
  ordering bugs, or checkpoint corruption that appear after hours or days.
- **Nothing about data quality at scale.** No deduplication, filtering, mixing, licensing
  review, or provenance tracking exists yet.

## 13. Current limitations

| Area | Limitation |
|---|---|
| Tokenizer | char/word level only; no BPE, no special/EOS tokens, no multilingual or Indic-script coverage |
| Data | single in-memory-style memmap file; tail split; no packing, dedup, filtering, mixing, sharding, or streaming; `data.num_workers` and `data.shuffle_buffer` exist in config but are **unused placeholders** |
| Scale | largest verified model: 138,752 params, 64-token context |
| Hardware | verified only on 2 CPU cores / 3 GB RAM; no GPU, no multi-node |
| Precision | only fp32 has actually run; bf16/fp16 paths are written but unexercised |
| Sampling | temperature + top-k + top-p only; no repetition penalty, no batch generation, no stop tokens (no EOS in the tokenizer) |
| Evaluation | loss/perplexity on the training corpus distribution; no benchmarks, no task eval, no human eval |
| Experiment tooling | no sweeps, no ablation runner, no tracker (WandB/TensorBoard), no multi-seed support |
| CI | workflow committed as `docs/ci.yml.example`; it has never run on GitHub Actions (the App used to push lacks the `workflows` permission) |
| Post-training | none: no SFT, preference optimization, reasoning, or safety work |

## 14. CPU-first development was intentional

The tiny defaults (`configs/cpu_smoke.json`) exist so every change can be validated in
seconds on a laptop, with no GPU, no network, and no data downloads. That gives fast
feedback on *code correctness* — the thing most likely to break day to day.

It is a development strategy, **not** a statement about where training happens. Device
selection is resolved once at runtime (`utils/device.py`: cuda → mps → cpu, plus the
autocast dtype and GradScaler policy) and nothing in the model or loop branches on "am I
on a GPU". Real GPU training is a later stage (see [ROADMAP.md](ROADMAP.md) Stage 4+), and
it should require a config change, not a code change.

## 15. Scale discipline: what NOT to do next

Explicitly **out of scope until the earlier stages are done**. Jumping ahead produces a
large model nobody can debug, evaluate, or afford.

- Do not jump to 7B/70B-parameter training.
- Do not start RLHF / preference optimization before there is a base model worth aligning
  and an eval that can measure it.
- Do not add multimodality (vision/audio) before the text stack is solid.
- Do not build agent frameworks or tool-use systems before the underlying model can
  follow instructions reliably.
- Do not attempt frontier-scale pre-training before compute, data, distributed training,
  and evaluation are all proven at smaller scale.
- Do not fine-tune someone else's model and call it our foundation model.

Scale is earned by experiments, not declared. Every jump in size must be justified by
results at the current size (see [ROADMAP.md](ROADMAP.md)).

## 16. Exact current state of the GitHub project

- Repository: `https://github.com/dfgtghu556-ops/frontier-ai`
- Current branch (and the only branch this work happens on): **`arena/01a08a78-frontier-ai`**
- Base branch: `main` (contains one commit, `89b9a1c Initial commit`, a README stub)
- Project 001 commit: `e467eb0` — "Add CPU-first, GPU-ready PyTorch GPT training pipeline"
  (29 files, +2,688 / −1)
- **PR: #1** — OPEN, not a draft, mergeable:
  https://github.com/dfgtghu556-ops/frontier-ai/pull/1
- CI: no checks have ever run on the branch (workflow not installable under
  `.github/workflows` with the available App permission — see D-013)
- Verified locally in the development sandbox: Python 3.11.2, torch 2.14.0+cu130,
  `torch.cuda.is_available() == False`, 2 CPUs, 3 GB RAM

## 17. Commands a new agent should run first

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                       # CPU torch; the PyTorch CDN may be blocked (see D-012)
pytest -q                                     # 47 tests, ~10 s
ruff check .                                  # lint
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic
python scripts/train.py --config configs/cpu_smoke.json                     # ~10 s, val loss 3.9 -> ~1.38
python scripts/evaluate.py --ckpt out/cpu-smoke/best --data data/synthetic.bin
python scripts/generate.py --ckpt out/cpu-smoke/best \
    --tokenizer data/synthetic.tokenizer.json --prompt "the quiet cat"
```

## 18. Conventions for future agents

1. **Never invent results.** If a number is not in `train.jsonl`, a test, or a terminal
   you actually ran, it does not go in a document. "Not measured" is an acceptable answer.
2. **Log every experiment** in [EXPERIMENTS.md](EXPERIMENTS.md) with its config, seed, data
   version, and hardware. Unreproducible results do not count.
3. **Record decisions** in [DECISIONS.md](DECISIONS.md) when they constrain future work.
4. **Keep the CPU smoke test green.** If a change breaks `configs/cpu_smoke.json` on a
   laptop, it is not ready.
5. **Add tests with behavior changes**, especially around data handling, masking, caching,
   and loss normalization — those are where the bugs have been so far.
6. **Device-agnostic code only.** No `if cuda:` in the model or loop.
7. **Ask before big jumps** (new dependencies, new subsystems, large refactors, anything
   that changes Project 001's behavior).
