# ROADMAP.md

> A staged path from today's 139k-parameter educational model to a genuine frontier lab.
> Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) first.

## How to read this roadmap

- **Stages are ordered by dependency, not by calendar.** A stage starts when the previous
  one's exit criteria are met and the resources exist. Several stages can run in parallel.
- **No stage commits to a parameter count.** Model size is an *output* of experiments,
  compute availability, data volume, and research results — never a target written in
  advance. Where a size appears below it is an example of how a stage *might* be exercised,
  not a promise.
- **Every stage ends in measurements, not vibes.** If a stage cannot produce numbers that
  a skeptical reader would accept, it is not finished.
- **This roadmap is a proposal.** The user approves what becomes the next project. An agent
  must not start a stage on its own.
- **Stage 0 (Project 001) and part of Stage 2 (Project 002) are done.** The tokenizer
  *framework* exists and has produced first measurements; the tokenizer *decision* is open.

---

## Stage 0 — Foundation plumbing ✅ COMPLETE (Project 001)

**Goal:** one tested, device-agnostic codebase that trains a small transformer end to end.
**Delivered:** data prep, tokenizers, memmap storage, GPT model, training loop, checkpointing,
resume, sampling, JSONL metrics, 47 tests, CI example.
**Exit criteria:** met — full CLI chain runs on CPU; tests and lint green; resume verified.
**Known gaps:** see "What Project 001 does NOT prove" in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

---

## Stage 1 — Make experiments trustworthy

**Goal:** before scaling anything, make results reproducible, comparable, and cheap to run.

- Deterministic runs: seed everything, pin versions, record the exact config + git SHA +
  data hash with every run.
- An **experiment runner** that takes a config (or a small sweep) and produces a run
  directory with config, metrics, environment, and a one-line summary.
- Fix loss/perplexity reporting to be comparable across tokenizers (bits per *byte* or
  per character, not just per token) so a char-level and a BPE model can be compared fairly.
- Multi-seed support: report mean ± spread for any headline number.
- Better smoke-test data: keep the synthetic corpus, add a small **real, clearly licensed**
  corpus for sanity checks.

**Exit criteria:** two runs with the same seed produce identical metrics; a 2-config sweep
runs unattended; every run directory is self-describing and diffable.

**Risk:** none technical — this is discipline work. It is the highest-leverage stage.

## Stage 2 — Tokenizer 🔶 FRAMEWORK BUILT, DECISION PENDING (Project 002)

**Goal:** a tokenizer trained on our own data, with coverage for Indian languages,
*chosen on measurements*.

**What Project 002 delivered (2026-09-10):**

- A pluggable tokenizer subsystem (`src/frontier_ai/tokenization/`): our own dependency-free
  byte-level BPE with a mark-aware pre-tokenizer, plus a HuggingFace `tokenizers` BPE
  baseline, plus adapters for the Project 001 char/word tokenizers as baselines.
- A deterministic research corpus: generated training text and a hand-written evaluation
  fixture of 179 probes across 14 languages (incl. Hinglish) and 9 orthography categories,
  sha256-verified on load.
- An evaluator (overall / per-language / per-category metrics, UTF-8 round-trip and
  special-token checks) and a comparator that scores any number of artifacts on the same
  corpus and refuses cross-corpus comparisons.
- 38 tests and full documentation ([docs/tokenization.md](docs/tokenization.md));
  first measurements recorded as **EXP-002** in [EXPERIMENTS.md](EXPERIMENTS.md).

**What Project 002 deliberately did NOT do:** select a production tokenizer, sweep
vocabulary size, compare BPE vs Unigram, ablate normalization, or measure how tokenizer
choice affects downstream model quality.

**Remaining before this stage can close:**

- Vocabulary-size sweep on real data (not just 512/1024 on a probe fixture).
- Pre-tokenization ablation: mark-aware vs GPT-2-style regex, whitespace attachment,
  script-aware initial alphabets (open question **Q-9**).
- Normalization policy (NFC/NFD) — **Q-11**.
- BPE vs Unigram comparison.
- **Tokenizer → model-quality measurement**: train the same small model with two
  tokenizers and compare (this is the only metric that ultimately matters).
- Train on real, licensed, Indic-heavy data (depends on Stage 3) — the current fixture is
  far too small to size a vocabulary for production.

**Exit criteria (unchanged):** a tokenizer trained on our data, versioned, with a
documented comparison against the current baselines **on representative data**, and an
explicit decision record superseding D-010. Until then: **no production tokenizer has been
selected.**

**Risk:** deciding from a 179-example fixture. The framework is explicitly designed to make
that mistake visible — every report carries the fixture caveat.

## Stage 3 — Real data pipeline

**Goal:** a defensible, versioned pipeline from licensed/public sources to training shards.

- Source inventory with **license and provenance recorded per source**; nothing enters
  training without a documented right to use it.
- Cleaning: language ID, encoding normalization, boilerplate/HTML stripping, PII scrubbing,
  quality filtering (heuristic and model-based), toxicity handling.
- Deduplication (exact + near-dup/MinHash) at document and shard level.
- Mixing/weighting per source, with the recipe stored as config so any dataset is
  reproducible from its version string.
- Sharded, shuffled storage that supports streaming and resume; document-boundary-aware
  packing so examples never straddle unrelated documents.
- **Indian-language data is a strategic priority**, not an afterthought: this is a core
  part of what makes the lab independent rather than a regional reskin.

**Exit criteria:** a reproducible dataset version built end to end, with per-source
statistics (tokens, documents, language distribution) and a held-out set that is genuinely
held out.

**Risk:** the largest single engineering effort in the plan, and the easiest to
underestimate. Legal review is part of the work, not paperwork after it.

## Stage 4 — Scaling ladder on real GPUs

**Goal:** establish that our training stack scales on accelerators, and learn the shape of
the scaling curve **for our setup**.

- Bring up single-GPU training: bf16/fp16 autocast, flash attention, GradScaler paths,
  `torch.compile`, gradient checkpointing — all written but unexercised today.
- Measure, don't assume: tokens/sec, memory headroom, MFU, step time at several model
  widths/depths/contexts.
- Run small scaling experiments: does loss scale predictably with parameters and tokens in
  our regime? Where does the compute-optimal frontier sit for the data we actually have?
- Establish checkpoint/restore discipline for long runs (periodic snapshots, corruption
  checks, restart drills).

**Exit criteria:** a documented single-GPU throughput and memory curve, and at least one
scaling-law sanity plot from runs we performed.

**Risk:** the first stage where hardware cost is real. Keep runs short and instrumented.

## Stage 5 — Distributed training

**Goal:** scale beyond one device without changing the training semantics.

- DDP first (simplest, matches current code), then FSDP/ ZeRO-style sharding when model
  state stops fitting; tensor/pipeline parallelism only if genuinely required.
- Rank-aware data sharding (no duplicate batches across ranks), rank-aware logging,
  deterministic checkpoint/restore across world sizes.
- Verify **numerical equivalence**: same loss curve at world size 1 and N within tolerance.

**Exit criteria:** throughput scaling efficiency measured at 2, 4, 8 devices; a resume from
a different world size works.

**Risk:** silent correctness bugs (data overlap, wrong loss reduction) that look like
"training is fine" until results are bad. Test for equivalence explicitly.

## Stage 6 — Evaluation

**Goal:** know whether a model is actually better, across capabilities and languages.

- Intrinsic: held-out perplexity per domain, per language, and per script.
- Downstream: a battery of standard benchmarks run through one harness, with the harness
  itself versioned and the prompting/decoding settings recorded.
- **Indic-language evaluation**: build or adopt evaluation sets for Indian languages
  (including transliteration-robust and code-mixed settings) rather than trusting English
  benchmarks to transfer.
- Calibration, robustness, and contamination checks (is the eval set in the training data?).
- Human evaluation protocol for qualities benchmarks miss.

**Exit criteria:** one command produces a full eval report for any checkpoint, and every
model comparison in this repo cites it.

**Risk:** benchmark-driven overfitting. Track a broad suite and treat any single number
with suspicion.

## Stage 7 — Post-training / instruction tuning

**Goal:** turn a base model into a useful assistant.

- SFT data: licensed, synthetic, and human-written instruction data; quality over volume;
  documented generation and filtering recipes.
- Chat formatting and role tokens baked into the tokenizer and model code.
- Preference optimization (DPO or similar) once a reliable preference dataset and reward
  signal exist.
- Evaluate instruction-following, helpfulness, honesty, and **multilingual/Indic** behavior
  specifically.

**Exit criteria:** instruction-tuned checkpoints measurably beat their base model on the
Stage 6 suite, with regressions tracked.

**Risk:** post-training can paper over base-model weakness. Keep investing in the base.

## Stage 8 — Reasoning

**Goal:** reliable multi-step problem solving, not just fluent answers.

- Data: worked solutions, verifiable tasks (math, code with tests, formal checks),
  self-consistency and rejection-sampling pipelines.
- Methods: chain-of-thought/distillation-style training, then RL-style optimization
  (GRPO/PPO family) **with verifiable rewards** where correctness is checkable.
- Instrument: pass@k, majority@k, length vs. accuracy trade-offs, and failure taxonomies.

**Exit criteria:** measurable gains on held-out reasoning benchmarks, with the cost of
inference-time reasoning (tokens, latency) reported alongside accuracy.

**Risk:** reward hacking and benchmark gaming. Verifiable rewards and held-out sets first.

## Stage 9 — Safety and alignment

**Goal:** a model that is safe to release, with evidence rather than assurances.

- Data-side: filtering, deduplication of harmful content, provenance.
- Training-side: refusal and harmlessness training that does not destroy usefulness;
  avoid over-refusal on benign cultural, religious, or regional content — an independent
  Indian lab will be judged on this.
- Evaluation-side: red-teaming (including in Indian languages and culturally specific
  contexts), jailbreak robustness, bias audits, misuse risk assessment.
- Governance: model cards, data statements, release process, incident response.

**Exit criteria:** documented red-team results, a release checklist, and a named owner for
release decisions.

**Risk:** safety theater. Prefer measurable evaluations and publish what they show.

## Stage 10 — Tool use and agents

**Goal:** models that act, not just talk — only after Stage 7–8 quality exists.

- Structured tool/function calling with schemas; reliable parsing and error recovery.
- Long-horizon task scaffolding: planning, memory, verification, and cost/latency control.
- Sandboxed execution environments for code and retrieval; strict permissioning.
- Evaluation on multi-step tasks with partial credit and failure analysis.

**Exit criteria:** agent success rates measured on held-out multi-step tasks, with cost per
task reported.

**Risk:** agent demos look impressive while being unreliable. Measure, then scale scope.

## Stage 11 — Multimodality

**Goal:** extend beyond text when the text stack is stable and there is a clear reason.

- Vision first (encoders + projection into the LM), then audio/speech — speech matters
  disproportionately for Indian-language accessibility.
- Multilingual multimodal data: OCR and speech in Indic scripts are strategically valuable
  and under-served.
- Reuse the evaluation harness; extend it per modality.

**Exit criteria:** multimodal checkpoints beat text-only baselines on the target tasks,
with per-language breakdowns.

**Risk:** cost and data complexity. Do not start this early; it will consume the team.

## Stage 12 — Research innovations

**Goal:** contribute, not just replicate.

- Choose bets where we have an edge: multilingual/Indic modeling, tokenizer design,
  data-efficiency, training stability at low compute, inference efficiency, evaluation
  methodology for low-resource languages.
- Run ablations that isolate one variable at a time; publish negative results internally.
- Build the habit: hypothesis → experiment → written conclusion (see
  [EXPERIMENTS.md](EXPERIMENTS.md)) for every idea, including the ones that fail.

**Exit criteria:** reproducible internal results that a researcher outside the team could
verify, and a written position on what we believe that others do not.

**Risk:** research without infrastructure is unpursuable; research without evaluation is
unfalsifiable. Stages 1–6 exist to prevent both.

## Stage 13 — Frontier scale

**Goal:** train models competitive with the best in the world, on our own weights and data.

**This stage has no plan yet, by design.** It is reached only when all of the following are
true, and the plan will be written at that time from the evidence accumulated above:

- Compute secured and costed, with a realistic training budget and failure budget.
- Data pipeline (Stage 3) producing enough high-quality, licensed, deduplicated tokens,
  with Indic coverage.
- Distributed training (Stage 5) proven at meaningful scale with measured efficiency.
- Evaluation (Stage 6) able to detect regressions before they reach users.
- Post-training, safety, and release processes (Stages 7–9) exercised on smaller models.
- Team and on-call capacity to run a long training job and respond to incidents.

**What will decide the size:** the scaling experiments from Stage 4, the token budget from
Stage 3, the efficiency measured in Stage 5, and the quality targets from Stage 6 — not a
number chosen today.

---

## Parallel track: product

The product/control-center work (Lovable; see D-006) runs on its own clock and is not
blocked by this roadmap. Products may use third-party models until our own beat them on a
measured task, at which point migration is evaluated case by case.

## Standing rules for every stage

1. Reproducible before big. If a result cannot be re-run from a config + seed + data
   version, it does not count.
2. Measure the thing you care about, not the thing that is easy to measure.
3. Write the failure down. Negative results are assets.
4. Never scale past your ability to debug and evaluate.
5. Own the weights: train from random initialization on the foundation-model track.
