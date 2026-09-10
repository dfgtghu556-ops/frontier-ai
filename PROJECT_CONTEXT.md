# PROJECT_CONTEXT.md

> Read this file first if you are an AI agent (or human) opening this repository in a
> new chat with no other context. It describes the mission, what exists today, what has
> been proven, what has not, and where to start.

---

## CURRENT POSITION — START HERE

**Projects 001 and 002 are COMPLETE. Project 003 (ROADMAP Stage 1: reproducible experiment
infrastructure) is IN PROGRESS — Stage 1 infrastructure is built, tested and documented;
the rest of Stage 1 (multi-seed sweeps, comparable loss reporting, real licensed smoke-test
data) is NOT done.**

- **Project 001** = the CPU-first, GPU-ready PyTorch GPT training pipeline
  (data → model → training loop → evaluation → checkpointing → sampling, plus 47 tests).
- **Project 002** = the tokenizer research subsystem: a pluggable tokenizer framework,
  two byte-level BPE baselines (ours + HuggingFace `tokenizers`), a deterministic
  Indian-language evaluation fixture, and an evaluator/comparator that produce
  machine-readable metrics. Documented in [docs/tokenization.md](docs/tokenization.md);
  results recorded as **EXP-002** in [EXPERIMENTS.md](EXPERIMENTS.md).
- **Project 003** = reproducible experiment infrastructure (`src/frontier_ai/experiments/`):
  experiment specs, one seeding mechanism with documented limits, git/data/environment
  provenance, a JSON experiment record with a content fingerprint, a runner with a
  validate → record lifecycle, `scripts/experiment_record.py`, and sweeps over seeds **and
  configurations** (one record per run + a mean ± spread aggregate,
  `scripts/experiment_sweep.py`). Documented in
  [docs/experiments.md](docs/experiments.md); validation recorded as **EXP-003**,
  **EXP-004**, **EXP-005** and **EXP-006** (infrastructure/metric verification,
  *not* benchmarks).
- All three are committed on branch `arena/01a08a78-frontier-ai` and open as **PR #1**
  against `main`. PR #1 is **not merged** — the human merges it.
- The most likely next steps are finishing ROADMAP Stage 1 (multi-seed sweeps with
  mean ± spread, bits-per-byte reporting so char and BPE models compare fairly, real
  licensed smoke-test data) and continuing Stage 2/3 (vocabulary sweeps and real data).
  Confirm the actual scope with the user before writing code; the roadmap is a proposal,
  not an approved plan.
- **Project 002 did NOT choose a production tokenizer.** It built the framework for that
  decision and took the first measurements. Do not treat EXP-002 as a verdict.
- Before changing any code, read: this file, [ROADMAP.md](ROADMAP.md),
  [DECISIONS.md](DECISIONS.md), [EXPERIMENTS.md](EXPERIMENTS.md),
  [docs/tokenization.md](docs/tokenization.md), [docs/experiments.md](docs/experiments.md)
  and [README.md](README.md).
- Report any headline number as **mean ± spread over seeds** (D-028), not as a single
  lucky run: `python scripts/experiment_sweep.py --exp-id EXP-00N --seeds 1,2,3,4,5
  --metric <field> -- <command with {seed}>`. Spread is the sample standard deviation and
  is `null` (never zero) when fewer than two runs are usable.
- Record any new number through an experiment record (`scripts/experiment_record.py`) so
  the run carries its own code/data/seed/environment provenance.
- Record any experiment you run in [EXPERIMENTS.md](EXPERIMENTS.md) using the standard
  template. Never record a number you did not actually observe.

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

## 6. What Project 002 built (tokenizer research subsystem)

Project 002 added a modular tokenizer research framework under
`src/frontier_ai/tokenization/` and four CLI commands. **Project 001's tokenizers and
training pipeline were not modified** — they are wrapped by adapters so they can be
measured in the same harness.

- **Pluggable implementations** behind one interface (`SubwordTokenizer`) and a registry:
  - `bpe_python` — our own byte-level BPE, dependency-free, deterministic, with a
    **mark-aware pre-tokenizer** that keeps Indic combining marks attached to their base
    character (see §Unicode note below).
  - `bpe_hf` — byte-level BPE via HuggingFace `tokenizers` 0.23.2 (Apache-2.0), optional
    dependency, offline save/load.
  - `char`, `word` — adapters over the Project 001 tokenizers, kept as baselines.
- **Research corpus** (`corpus.py` + `scripts/tokenizer_prepare_corpus.py`): deterministic
  training text plus a hand-written evaluation fixture of 179 probes covering 14 languages
  (incl. Hinglish) and 9 orthography categories (numbers, dates, decimals, math,
  punctuation, URLs, technical terms, code-mixed tech, Unicode marks). The loader verifies
  a sha256 and refuses to run on a modified fixture.
- **Evaluator** (`evaluate.py`): per-tokenizer metrics overall, per language and per
  category — tokens, chars, UTF-8 bytes, words, tokens/char, chars/token, bytes/token,
  tokens/word, unknown rate, round-trip failures, plus special-token behaviour checks.
  Output is JSON (and optional JSONL per example).
- **Comparator** (`compare.py`): any number of artifacts scored on the same corpus version;
  refuses cross-corpus comparisons; disqualifies tokenizers that cannot round-trip their
  input from winning "best".
- **Artifacts** (`artifact.py`): tokenizer files + `manifest.json` recording experiment id,
  implementation and version, vocab size, special tokens, training params, corpus id/hash,
  seed, library versions, timestamp.
- **Tests**: 42 additional tests (89 total: 47 Project 001 + 38 tokenizer + 4 repo-hygiene) covering round trips for every language,
  determinism, save/load, special tokens, Unicode/Indic handling, evaluator arithmetic,
  comparison logic, invalid input, and an end-to-end CLI workflow.

**Unicode note (why this matters):** Python's `\w` and the GPT-2-style regex used by many
tokenizers match letters and numbers but **not** combining marks, so "मैं" is split into
"म" + "ैं". BPE merges cannot cross a pre-token boundary, so the fragment can never be
repaired. `bpe_python` treats marks as word characters; the HF baseline does not. This
difference is measured, not assumed — see EXP-002.

## 6A. What Project 003 built (experiment infrastructure)

**Status: Stage 1 infrastructure complete and tested; Stage 1 as a whole is not.**

- `src/frontier_ai/experiments/` — experiment spec, git/data/environment provenance,
  seeding with recorded limitations, the experiment record, and the runner
  (validate → config → git → data → seed → environment → execute → results → write).
- `scripts/experiment_record.py` — CLI wrapper that records provenance for any command and
  propagates its exit code.
- **No new dependencies**: stdlib plus the torch/numpy already required by Project 001.
  Data hashing re-uses Project 002's `sha256_file`/`sha256_text` (D-024), so there is one
  hashing implementation in the repository.
- Every run writes `experiment.json` (8 sections: experiment/code/data/configuration/
  randomness/environment/execution/results, `schema_version 1.0`) plus a rendered
  `experiment.txt`. `content_fingerprint()` hashes the record with timestamps, pid, paths
  and log tails removed, so identical runs have identical fingerprints.
- **What this proved about Project 001:** training was *not* reproducible despite a fixed
  seed, because `TokenDataset.get_batch` called `generator.seed()` — which re-seeds a torch
  generator from OS entropy instead of reading it. Fixed (and `Trainer.evaluate()` now
  passes its seeded generator); two identical runs now give bit-identical results
  (`final_val_loss 3.24484`, equal fingerprints). Reproducibility is only claimed under a
  recorded environment, not across machines, versions or thread counts.
- **Loss reporting per byte / per character (Stage 1 item 2):** per-token loss is not
  comparable across tokenizers, so `val_loss` (nats/token) is now reported alongside
  `bits_per_token`, **`bits_per_byte`** and **`bits_per_char`**
  (`bits_per_byte = bits_per_token × tokens_per_byte`). `prepare_data.py` measures the UTF-8
  byte and Unicode character length of every token's surface piece (`Tokenizer.tokenize()`)
  and stores the per-split sums in `*.meta.json`, so the conversion is measured, not
  estimated; the `char` and `word` corpora of the same text report the same 200,094 bytes.
  Corpora without counts report `null` — never `0` (D-030, EXP-006).
- **Stage 1B (multi-configuration sweeps):** the same sweeps now take **named
  configurations** (`--configs NAME:key=value`), each run across the whole seed list. One
  record per configuration × seed; every configuration is aggregated **separately** and
  configurations are never mixed (the top-level `statistics` says `aggregated: false` when
  there are two or more). The sweep schema stays `1.0` and the new fields appear only for
  named configurations, so seed-only sweeps stay byte-identical to Stage 1A (D-029).
- **Stage 1A (multi-seed sweeps):** `sweep.py` + `scripts/experiment_sweep.py` run one spec
  across an explicit seed list. Each seed keeps its own full record under
  `seed-<seed>/experiment.json`; the aggregate (`sweep.json`) carries the seed list,
  per-seed values and fingerprints, mean and **sample** standard deviation. Sample
  standard deviation is used because sweeps here are small (`n` of 3–5); it is `null` when
  undefined, is never called a confidence interval, and missing values are never replaced
  by zero. Status is `success` / `partial` / `failed`, so a lost seed can never be reported
  as a success. The aggregate is independent of the order seeds were supplied in.

Full details: [docs/experiments.md](docs/experiments.md) · decisions **D-024…D-030** ·
validation **EXP-003**, **EXP-004**, **EXP-005** and **EXP-006**.

## 7. Current repository structure

```
configs/
    cpu_smoke.json            tiny config: 139k params, trains in ~10 s on CPU
    gpu_1x.json               single-GPU config (~124M params) — UNTESTED on real hardware
docs/
    ci.yml.example            GitHub Actions workflow (copy to .github/workflows to enable)
    tokenization.md           Project 002: tokenizer research guide, metrics, limitations
    experiments.md            Project 003: experiment infrastructure, provenance, limits
scripts/
    prepare_data.py           text or synthetic corpus -> tokens.bin + tokenizer.json + meta.json
    train.py                  build model from config, run training, save checkpoints
    evaluate.py               score a checkpoint: val loss, perplexity, bits/token, sample
    generate.py               sample text with the KV cache (temperature / top-k / top-p)
    tokenizer_prepare_corpus.py   write the tokenizer research corpus (train + eval fixture)
    tokenizer_train.py            train a tokenizer from a local corpus -> versioned artifact
    tokenizer_evaluate.py         evaluate artifact(s) -> JSON metrics per tokenizer
    tokenizer_compare.py          compare 2+ artifacts on the same corpus -> JSON + table
    experiment_record.py       Project 003: record provenance for any experiment command
    experiment_sweep.py        Project 003: run one experiment across seeds/configurations
src/frontier_ai/
    config.py                 dataclass configs, JSON load/save, --set overrides, validation
    data/
        tokenizer.py          CharTokenizer / WordTokenizer, save/load
        dataset.py            memmap token store, train/val split, batch sampling
        synthetic.py          deterministic pseudo-English corpus (a test fixture)
    model/gpt.py              the model: norms, RoPE, attention, MLP, blocks, sampling
    tokenization/             Project 002: tokenizer research subsystem
        base.py                 pluggable SubwordTokenizer interface + TokenizerInfo
        registry.py             name -> implementation registry
        bpe_python.py           our own byte-level BPE (mark-aware pre-tokenizer)
        bpe_hf.py               HuggingFace `tokenizers` byte-level BPE (optional dep)
        adapters.py             Project 001 char/word tokenizers behind the same interface
        corpus.py               deterministic train text + Indic evaluation fixture
        evaluate.py             metrics: overall / per language / per category
        compare.py              multi-tokenizer comparison + rendered table
        artifact.py             tokenizer files + provenance manifest
        cli.py                  shared CLI helpers
    experiments/            Project 003: reproducible experiment infrastructure
        spec.py                 ExperimentSpec: validated, serializable experiment definition
        gitinfo.py              git commit/branch/dirty capture (never invents a SHA)
        hashing.py              input digests (re-uses Project 002's sha256_file)
        seeding.py              one master seed + derived seeds + documented limitations
        environment.py          selected environment metadata (no env dumps)
        record.py               ExperimentRecord: sections, fingerprint, save/load/render
        runner.py               run_experiment / run_command lifecycle
        sweep.py                run_sweep / run_command_sweep: one record per run + aggregate
        examples.py             reference experiments used by docs + determinism tests
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
    test_experiments.py       Project 003: spec, seeding, git, hashing, record, runner, CLI
    test_sweeps.py            Project 003: multi-seed sweeps, statistics, failures, CLI
    test_sweep_configs.py     Project 003: multi-configuration sweeps, compatibility, CLI
    test_repo_hygiene.py      no source file may be git-ignored (D-023)
PROJECT_CONTEXT.md            this file
ROADMAP.md                    staged plan from here to frontier scale
DECISIONS.md                  architectural decision records
EXPERIMENTS.md                experiment log (template + EXP-001)
README.md                     user-facing quickstart
Makefile, pyproject.toml, LICENSE (MIT), .gitignore
```

Ignored (never committed): `data/` (prepared corpora), `out/` (runs and checkpoints),
`.venv/`, caches.

## 8. Current model architecture and important implementation details

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

## 9. Current training pipeline

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

## 10. Current tokenizer and data approach

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
- **Tokenizer research is separate** (Project 002, §6): `src/frontier_ai/tokenization/`
  provides pluggable subword tokenizers and evaluation tooling, but the *training pipeline
  still uses the Project 001 char/word tokenizers.* Wiring a research tokenizer into
  training is a later step — the interface (`encode`/`decode`/`save`/`load`) was designed
  so that it is a small change, but it has not been made yet.

## 11. Current evaluation and testing approach

**Testing (automated, 194 tests, ~45 s on CPU, `pytest -q`; `ruff check .` clean):**

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
text sample. There is **no** downstream task evaluation, no standardized benchmark, and no
comparison against published models.

**Tokenizer evaluation (Project 002):** fertility and compression metrics
(tokens/char, chars/token, bytes/token, tokens/word), unknown rate, UTF-8 round-trip
correctness and special-token behaviour — reported overall, **per language** and per
orthography category, with multi-tokenizer comparison. See
[docs/tokenization.md](docs/tokenization.md).

## 12. What Project 001 proves

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

## 13. What Project 001 does NOT prove

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

## 14. Current limitations

| Area | Limitation |
|---|---|
| Tokenizer (training pipeline) | still char/word level in the training pipeline; the Project 002 subword tokenizers are **not** yet wired into training |
| Tokenizer (research) | measured only on a 179-example hand-written probe fixture; no vocabulary sweep, no Unigram comparison, no normalization ablation, no tokenizer→model-quality measurement |
| Data | single in-memory-style memmap file; tail split; no packing, dedup, filtering, mixing, sharding, or streaming; `data.num_workers` and `data.shuffle_buffer` exist in config but are **unused placeholders** |
| Scale | largest verified model: 138,752 params, 64-token context |
| Hardware | verified only on 2 CPU cores / 3 GB RAM; no GPU, no multi-node |
| Precision | only fp32 has actually run; bf16/fp16 paths are written but unexercised |
| Sampling | temperature + top-k + top-p only; no repetition penalty, no batch generation, no stop tokens (no EOS in the tokenizer) |
| Evaluation | loss/perplexity on the training corpus distribution; no benchmarks, no task eval, no human eval |
| Experiment tooling | **Project 003 infrastructure exists**: specs, seeding with recorded limits, git/data/env provenance, JSON records with fingerprints, a runner (`scripts/experiment_record.py`) and **sweeps over seeds and configurations** (mean ± sample standard deviation, `scripts/experiment_sweep.py`); large-scale sweep orchestration (Q-8), bits-per-byte reporting, an ablation runner and a tracker (WandB/TensorBoard) still do not exist; the Project 001/002 entry points do not write records themselves (the CLI wraps them) |
| CI | workflow committed as `docs/ci.yml.example`; it has never run on GitHub Actions (the App used to push lacks the `workflows` permission) |
| Post-training | none: no SFT, preference optimization, reasoning, or safety work |
| Reproducibility | runs are reproducible **under a recorded environment only** (same code, data, seed, command, library versions, thread count). cuDNN, thread-dependent FP order and RNGs outside python/numpy/torch are not covered — every record states this (D-025) |

## 15. CPU-first development was intentional

The tiny defaults (`configs/cpu_smoke.json`) exist so every change can be validated in
seconds on a laptop, with no GPU, no network, and no data downloads. That gives fast
feedback on *code correctness* — the thing most likely to break day to day.

It is a development strategy, **not** a statement about where training happens. Device
selection is resolved once at runtime (`utils/device.py`: cuda → mps → cpu, plus the
autocast dtype and GradScaler policy) and nothing in the model or loop branches on "am I
on a GPU". Real GPU training is a later stage (see [ROADMAP.md](ROADMAP.md) Stage 4+), and
it should require a config change, not a code change.

## 16. Scale discipline: what NOT to do next

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

## 17. Exact current state of the GitHub project

- Repository: `https://github.com/dfgtghu556-ops/frontier-ai`
- Current branch (and the only branch this work happens on): **`arena/01a08a78-frontier-ai`**
- Base branch: `main` (contains one commit, `89b9a1c Initial commit`, a README stub)
- Commits on the branch (oldest → newest):
  - `e467eb0` — Project 001: "Add CPU-first, GPU-ready PyTorch GPT training pipeline"
  - `4ede476` — "Add permanent project documentation for long-term development"
  - a third commit adds Project 002 (tokenizer research subsystem) — see `git log`
  - further commits on the same branch add Project 003 (experiment infrastructure) — see `git log`
- **PR: #1** — OPEN, not a draft, mergeable:
  https://github.com/dfgtghu556-ops/frontier-ai/pull/1
- CI: no checks have ever run on the branch (workflow not installable under
  `.github/workflows` with the available App permission — see D-013)
- Verified locally in the development sandbox: Python 3.11.2, torch 2.14.0+cu130,
  `tokenizers` 0.23.2, `torch.cuda.is_available() == False`, 2 CPUs, 3 GB RAM

## 18. Commands a new agent should run first

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                       # CPU torch; the PyTorch CDN may be blocked (see D-012)
pytest -q                                     # 194 tests, ~45 s
ruff check .                                  # lint
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic
python scripts/train.py --config configs/cpu_smoke.json                     # ~10 s, val loss 3.9 -> ~1.38
python scripts/evaluate.py --ckpt out/cpu-smoke/best --data data/synthetic.bin
python scripts/generate.py --ckpt out/cpu-smoke/best \
    --tokenizer data/synthetic.tokenizer.json --prompt "the quiet cat"

# tokenizer research (Project 002)
python scripts/tokenizer_prepare_corpus.py --out data/tokenizer/indic-v1
python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 --impl bpe_hf \
    --vocab-size 1024 --out artifacts/tokenizers/bpe_hf_1024 --exp-id EXP-002
python scripts/tokenizer_compare.py --corpus data/tokenizer/indic-v1 \
    --tokenizer artifacts/tokenizers/char artifacts/tokenizers/bpe_hf_1024 \
    --out out/tokenizer/compare.json

# experiment provenance (Project 003)
python scripts/experiment_record.py --exp-id EXP-003 --seed 1337 \
    --config configs/cpu_smoke.json --data data/synthetic.bin \
    --out out/experiments/EXP-003-train -- \
    python scripts/train.py --config configs/cpu_smoke.json --max-steps 50

# multi-seed sweep: mean +/- spread over seeds (Project 003 Stage 1A)
python scripts/experiment_sweep.py --exp-id EXP-004 --seeds 1,2,3,4,5 \
    --metric final_val_loss --config configs/cpu_smoke.json --data data/synthetic.bin \
    --out out/sweeps/EXP-004 -- \
    python scripts/train.py --config configs/cpu_smoke.json --max-steps 50 \
        --set train.seed={seed}

# several configurations, each across the same seeds (Project 003 Stage 1B)
python scripts/experiment_sweep.py --exp-id EXP-005 --seeds 1,2,3 \
    --configs "lr_low:params.lr=0.005" "lr_high:params.lr=0.05" \
    --metric final_val_loss --data data/synthetic.bin --out out/sweeps/EXP-005 -- \
    python my_experiment.py --config-name {config} --seed {seed}
```

## 19. Conventions for future agents

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
7. **Run experiments through `scripts/experiment_record.py`** (or
   `frontier_ai.experiments.run_experiment`) so the run carries its own provenance, and
   never claim reproducibility beyond the recorded limitations (D-025). Do not weaken a
   failing assertion to make it pass — if a test loses significance, add statistical power.
8. **Ask before big jumps** (new dependencies, new subsystems, large refactors, anything
   that changes Project 001's behavior).
9. **Never trust "tests pass" as proof the repository is complete.** An unanchored
   `.gitignore` rule once excluded `src/frontier_ai/data/` from git entirely: local tests
   passed, the branch was pushed, and only a fresh checkout showed `import frontier_ai.data`
   failing. `tests/test_repo_hygiene.py` now guards against it (D-023). Anchor any new
   ignore rule that could match a source path.
