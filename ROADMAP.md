# ROADMAP.md

> A staged path from today's 139k-parameter educational model to a genuine frontier lab.
> Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) and the founder's [MASTER_CONTEXT.md](MASTER_CONTEXT.md) first.

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
- **Status (2026-09-27) — synchronized with NEW_CHAT_START_HERE.md, EXPERIMENTS.md and
  DECISIONS.md.** Progress is tracked against the founder's 24-step sequence
  ([MASTER_CONTEXT.md](MASTER_CONTEXT.md) §37); the stages below are this document's own
  dependency grouping, mapped where the correspondence is clear.

  | §37 step | Status | ROADMAP stage | Evidence |
  |---|---|---|---|
  | 1. Sync documentation with the repository | ✅ complete | — | NEW_CHAT_START_HERE.md §8 |
  | 2. Freeze and verify the acquired corpus | ✅ complete | Stage 3 | EXP-025/026, D-035 (`indic-tokenizer/v2`) |
  | 3. Build the FoundationCorpus pipeline | ✅ complete | Stage 3 | D-036, D-037 |
  | 4. Produce FrontierCorpus v1 | ✅ complete **at pilot scale** | Stage 3 | EXP-027 |
  | 5. Production tokenizer sweep | ✅ complete | Stage 2 | EXP-028, D-038, D-039 |
  | 6. Tokenizer-vs-tokenizer small-model experiments | ✅ complete | Stage 2 | EXP-029 |
  | 7. Select Frontier Tokenizer v1 | ✅ complete (selected + frozen) | Stage 2 | D-040, EXP-030, D-041 |
  | **8. Build the evaluation harness** | **🔶 current task** | Stage 6 (intrinsic part) | harness v1, EXP-031 |
  | 9. Architecture ablations | not started | — (MASTER_CONTEXT §20; ablation discipline of Stage 12) | — |
  | 10. Actual GPU training bring-up | not started | Stage 4 | — |
  | 11. Scaling experiments | not started | Stage 4 | — |
  | 12. First serious Frontier base model | not started | Stages 4/13 prerequisites | — |
  | 13. Validate distributed training | not started | Stage 5 | — |
  | 14. Scale pretraining on evidence | not started | Stages 3–5 | — |
  | 15. Evaluate extensively | not started | Stage 6 (downstream + human) | — |
  | 16. SFT · 17. Preference optimization | not started | Stage 7 | — |
  | 18. Reasoning | not started | Stage 8 | — |
  | 19. Safety | not started | Stage 9 | — |
  | 20. Inference optimization | not started | — (MASTER_CONTEXT §29) | — |
  | 21. Multimodal | not started | Stage 11 | — |
  | 22. Tools/agents | not started | Stage 10 | — |
  | 23. Specialist ecosystem | not started | — (MASTER_CONTEXT §32) | — |
  | 24. Frontier research / larger models | not started | Stage 13 | — |

  What the completed steps do and do not establish:
  - **FrontierCorpus v1 is a completed *pilot* corpus** (~4.2 million characters, a few
    MB; EXP-027). It is sufficient for controlled tokenizer and evaluation research. It is
    **not** a claim of sufficient scale for foundation-model pretraining; building a
    substantially larger licensed corpus remains future work (Stage 3 at scale).
  - **Frontier Tokenizer v1 is selected and frozen** (D-040, D-041). It was trained and
    evaluated on the pilot corpus; D-041 allows a future **Frontier Tokenizer v2** once a
    substantially larger real training corpus exists, subject to founder approval and a
    new experiment.
  - **No serious foundation model has been trained yet**, and **no GPU training has been
    performed yet**; every model so far is a small CPU research model.
  - **GPU access becomes necessary at step 10** (real model training). The current work,
    step 8, runs on CPU and does not need a GPU.
  - Steps 9–24 remain not started until the founder explicitly approves each.

- **Update (2026-09-27, later):** step 8 is complete and was approved by the founder (EXP-031, D-042).
  **Step 9 (architecture ablations) is now the current step** (EXP-032, CPU screening on
  the frozen tokenizer; results provisional until step 11). Steps 10–24 are not started.
- **Update (2026-09-28):** step 9 has finished its experiments:
  - EXP-032 screening and the EXP-033 confirmation at two learning rates were run.
  - By the pre-registered rules, no architecture change is adopted at toy scale.
  - **D-043 is proposed**: keep the EXP-B baseline architecture for GPU bring-up, and make
    RoPE + GQA-2 the first candidate for the step-11 GPU-scale ablation. It is pending
    founder review.

  **Step 10 (GPU training bring-up) is next** and needs the founder's approval of its plan.
  A GPU machine is becoming available through a collaborator (a friend of the founder, with
  an NVIDIA RTX card; the exact model is NOT VERIFIED yet). The onboarding guide is
  [GPU_COLLABORATOR_START_HERE.md](GPU_COLLABORATOR_START_HERE.md), and the read-only
  readiness check is `scripts/gpu_env_report.py`. Real frontier-scale training will still
  need rented data-centre GPUs later.

  The current training data (about 1.8M tokens) is small, so growing the corpus remains on
  the critical path.
- **Update (2026-09-29):** the founder approved **D-043** ("approve D-043 and data plan"),
  so **step 9 is complete**: the EXP-B baseline is the provisional architecture for GPU
  bring-up, and RoPE + GQA-2 is the first step-11 candidate. Step 10 still waits for the
  GPU collaborator and a founder-approved plan.

  While it waits, the founder approved **data scale-up phase 1** (Stage 3 scale-up, no GPU):
  1. a source survey with license evidence for every candidate
     (`docs/frontier_corpus_v2_sources.md`);
  2. a data-rights policy for the founder to decide (proposed D-044);
  3. making the corpus pipeline ready for far more data (streaming, near-duplicate removal,
     decontamination against the protected suite at scale).

  No large download happens before the founder decides D-044. The corpus stays
  FrontierCorpus v1 until a v2 build is recorded as an experiment.
- **Update (2026-09-29, later):** the founder decided **D-044: option 2** (open sources plus
  curated collections), **starting with AI4Bharat's Sangraha Verified** (CC-BY-4.0), and asked
  for the best quality with no compromise. First slice: one pinned file per language (13 files,
  about 5.1 GB). **EXP-034** downloads it with fingerprint checks and measures it *before* any
  filtering (wrong-script text, duplicates, overlap with the protected suite, estimated tokens).
  Next after EXP-034: pre-register the v2 cleaning thresholds from those numbers, add the
  memory-bounded near-duplicate stage and PII scrubbing, then build and count tokens exactly.
- **North star (founder, 2026-09-27):** build, from scratch, a model family that can
  compete with the leading AI systems (e.g. ChatGPT, Claude) and become the best model in
  India (MASTER_CONTEXT §1). Every step is judged by how it moves us toward that goal, and
  progress is reported honestly. Today's models are small CPU research models, far from
  that goal, and the roadmap above is the evidence-driven path toward it.

- **Previous status (2026-09-26, superseded by the block above):**
  - Stages 0 and 1 are complete (Projects 001 and 003).
  - Stage 2 has its framework (Project 002) and a real, licensed research corpus
    (Project 004, 13 of 14 language slots). The tokenizer *decision* is open.
  - The founder's next step is Stage 3, FrontierCorpus v1
    ([MASTER_CONTEXT.md](MASTER_CONTEXT.md) §37–§38).

---

## Stage 0 — Foundation plumbing ✅ COMPLETE (Project 001)

**Goal:** one tested, device-agnostic codebase that trains a small transformer end to end.
**Delivered:** data prep, tokenizers, memmap storage, GPT model, training loop, checkpointing,
resume, sampling, JSONL metrics, 47 tests, CI example.
**Exit criteria:** met — full CLI chain runs on CPU; tests and lint green; resume verified.
**Known gaps:** see "What Project 001 does NOT prove" in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

---

## Stage 1 ✅ COMPLETE — Make experiments trustworthy

**Goal:** before scaling anything, make results reproducible, comparable, and cheap to run.

- Deterministic runs: seed everything, pin versions, record the exact config + git SHA +
  data hash with every run.
- An **experiment runner** that takes a config (or a small sweep) and produces a run
  directory with config, metrics, environment, and a one-line summary.
- Fix loss/perplexity reporting to be comparable across tokenizers (bits per *byte* or
  per character, not just per token) so a char-level and a BPE model can be compared fairly.
  **Delivered (2026-09-10):** `val_loss` (nats/token) is reported next to
  `bits_per_token`, **`bits_per_byte`** and **`bits_per_char`**; `prepare_data.py` records
  measured per-split byte/character counts, unknown counts report `null` (D-030, EXP-006).
- Multi-seed support: report mean ± spread for any headline number. **Delivered
  (2026-09-10, Stage 1A):** `run_sweep()` / `scripts/experiment_sweep.py` run one spec
  across an explicit seed list, keep one full record per seed, and aggregate as
  mean ± sample standard deviation (D-028).
- Better smoke-test data: keep the synthetic corpus, add a small **real, clearly licensed**
  corpus for sanity checks. **Delivered (2026-09-10):** `corpora/smoke/sources.json`
  declares three licensed sources (Project Gutenberg, public domain; Hindi and Bengali
  Wikisource, CC BY-SA 4.0 assumed conservatively), `scripts/fetch_smoke_corpus.py`
  acquires them with provenance, and `prepare_data.py --provenance` carries the licence
  into `*.meta.json`. Hashes are pinned only after a verified fetch, so the manifest ships
  unpinned and nothing was invented (D-031).

**Exit criteria:** two runs with the same seed produce identical metrics ✅; a 2-config sweep
runs unattended ✅ (Stage 1A and Stage 1B below); every run directory is self-describing and
diffable ✅. All met and re-verified from a fresh clone of the pushed branch (2026-09-11).

**Risk:** none technical — this is discipline work. It is the highest-leverage stage.

**What Project 003 delivered (2026-09-10) — the infrastructure half of this stage:**

- `src/frontier_ai/experiments/`: experiment spec; **one** seeding mechanism (master seed +
  derived component seeds) with its limitations recorded in every run; git provenance that
  never invents a SHA and flags dirty trees; data provenance re-using Project 002's SHA-256
  helpers; selected environment metadata (no env dumps); a JSON experiment record with
  eight separated sections, a schema version and a content fingerprint that ignores
  timestamps; and a runner implementing validate → config → git → data → seed →
  environment → execute → results → write record, which records failures and **re-raises**
  them instead of writing a misleading success.
- `scripts/experiment_record.py`: wraps any command and records its provenance, following
  the Project 001/002 CLI conventions.
- No new dependencies (stdlib + existing torch/numpy). 31 new tests; Project 001 and 002
  suites unchanged in behaviour and still green.
- Documented in [docs/experiments.md](docs/experiments.md); decisions **D-024…D-030**;
  verification recorded as **EXP-003** … **EXP-006** (infrastructure/metric verification,
  not benchmarks).
- **Bug found and fixed in Project 001:** batch sampling called `torch.Generator.seed()`,
  which *re-seeds* from OS entropy rather than reading the seed, so training was not
  reproducible despite a fixed seed. Fixed; two identical runs now match exactly
  (`final_val_loss 3.24484`, equal content fingerprints).

**What Project 003 Stage 1A added (2026-09-10): multi-seed sweeps**

- `src/frontier_ai/experiments/sweep.py` + `scripts/experiment_sweep.py`: one spec across an
  explicit seed list; **one full experiment record per seed** (own directory, own
  fingerprint), plus a `sweep.json` aggregate with the seed list, successful/failed seeds,
  per-seed metric values and fingerprints, mean and spread.
- Spread = **sample standard deviation** (`n − 1`), `null` when undefined, never called a
  confidence interval; missing values are never replaced by zero; status is
  `success` / `partial` / `failed` so a lost seed can never be reported as success; the
  aggregate is independent of seed order (D-028).
- 25 new tests; stdlib `statistics` only — no new dependencies.

**What Project 003 Stage 1B added (2026-09-10): multi-configuration sweeps**

- `sweep.py` + `scripts/experiment_sweep.py --configs NAME:key=value` run several **named
  configurations**, each across the full seed list: one normal experiment record per
  configuration × seed (`<config>/seed-<seed>/`, tagged `config:<name>`), and one
  `configurations[]` entry per configuration with its own status, seed lists, resolved spec
  and mean ± spread.
- **Configurations are never mixed**: with two or more configurations the top-level
  `statistics` section reports `aggregated: false` and publishes no mean; every mean and
  spread lives with the configuration that produced it (D-029).
- Configuration order, like seed order, does not affect the result or the fingerprint. The
  sweep schema stays `1.0`: the new fields appear only for named configurations, so
  seed-only sweeps remain byte-identical to Stage 1A and EXP-004's fingerprints still
  reproduce.
- 25 new tests in `tests/test_sweep_configs.py`; `tests/test_sweeps.py` (Stage 1A) is the
  backward-compatibility suite and is unchanged.

**What Project 003 added (2026-09-10): loss per byte and per character**

- `src/frontier_ai/engine/metrics.py`: one place that converts nats/token into
  `bits_per_token`, `bits_per_byte` and `bits_per_char`. Unknown denominators give `null`,
  never `0`.
- `data/tokenizer.py`: `Tokenizer.tokenize()` returns the surface pieces a tokenizer
  encodes; they concatenate back to the source text, which is what makes the counts exact.
  `data/dataset.py`: `write_tokens()` stores per-split `n_bytes_*` / `n_chars_*`, and
  `TokenDataset.tokens_per_byte()` / `tokens_per_char()` expose the ratios.
- `scripts/evaluate.py` prints all four numbers plus the corpus counts; the trainer logs
  `bits_per_byte` / `bits_per_char` on `eval` events and `best_bpb` on `run.end`;
  `scripts/train.py` prints `best_bpb`.
- 23 new tests in `tests/test_loss_reporting.py`; old corpora still load (D-030, EXP-006).
- **Found while testing (Q-13, not fixed):** `environment.torch.num_threads` is captured
  *before* the run body, so a run that changes torch's thread count makes a later
  in-process run's record differ; the tests pin threads to 1.

**Closed 2026-09-11:** the CLIs now record themselves (`scripts/train.py`,
`scripts/tokenizer_*.py`), with the outer run owning the record and nested runs publishing
their metrics instead of writing a second record (**D-032**, **D-033**, **D-034**;
verification **EXP-007**). Every item of Stage 1 is therefore delivered.

**Still open, deliberately out of scope for this stage** (not started):

- Larger sweep orchestration: many configurations, scheduling, resuming an interrupted
  sweep (**Q-8**). Two-configuration sweeps themselves are delivered.
- ~~Bits-per-byte / per-character loss reporting~~ **delivered 2026-09-10 (D-030, EXP-006)**
  for the char and word levels; re-check when a byte-level BPE enters the training path.
- ~~Real, clearly licensed smoke-test corpora~~ **delivered 2026-09-10 (D-031)** — with one
  caveat recorded rather than hidden: the corpus text itself is fetched, not committed, so
  a checkout has no corpus until `scripts/fetch_smoke_corpus.py --fetch --pin` has run in an
  environment that can reach gutenberg.org / wikisource.org.
- Wiring the existing `scripts/train.py` and `scripts/tokenizer_*.py` entry points to write
  experiment records themselves (today the CLI wraps them from outside).

## Stage 2 — Tokenizer ✅ COMPLETE — Frontier Tokenizer v1 selected and frozen (D-040, D-041)

**Update (2026-09-27):** closed by §37 steps 5–7. EXP-028 swept 20 configurations on the
frozen pilot corpus (all losslessness gates PASS; mark-aware pre-tokenization densest,
partially answering Q-9 and Q-10 for this corpus); EXP-029 compared the top two on identical small models
(held-out bits-per-byte, 3 seeds; `mark_aware-32768` 1.4463 vs `mark_aware-16384` 1.5779);
D-040 selected **`mark_aware-32768`**; EXP-030/D-041 froze it at `tokenizers/frontier-tokenizer-v1/` behind a
hash-verifying loader. Still open (recorded, not blocking): BPE vs Unigram (Q-1/Q-12),
special tokens (needed by later stages → v2), and retraining on a substantially larger
corpus (→ v2, founder approval). The historical text below is kept as written.

*Previous heading: "Stage 2 — Tokenizer 🔶 FRAMEWORK BUILT, RESEARCH CORPUS ACQUIRED,
DECISION PENDING (Projects 002, 004)".*

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

**What Project 004 delivered (2026-09-26):** a real, licensed tokenizer-research corpus,
`indic-tokenizer/v2`.
- It holds 59 public-domain works: Wikisource transcriptions under CC BY-SA, plus Project
  Gutenberg for English. Every source is verified and pinned by SHA-256.
- 13 of 14 language slots meet the 500-document / 200,000-character target: 34,684
  documents and about 4.2 million characters in total.
- `hi-en` has no lawful source yet.
- It is far too small for pretraining, and no tokenizer has been trained on it yet.
- See [docs/tokenizer_corpus_stage_a.md](docs/tokenizer_corpus_stage_a.md) and EXP-018 to
  EXP-024.

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

## Stage 3 — Real data pipeline ✅ PILOT COMPLETE (FrontierCorpus v1) — scale-up not started

**Update (2026-09-27):** §37 steps 2–4 are complete: the acquired corpus is frozen and
verified (`indic-tokenizer/v2`, D-035), the staged pipeline is built (D-036, D-037), and
**FrontierCorpus v1** was produced and verified (EXP-027: train 30,584 / held-out 3,427
documents, ~4.2 M characters). This is a **pilot-scale** corpus: it proves the pipeline
end to end and supports controlled research, but it is **not** a pretraining corpus of
sufficient scale. The exit criterion below is met for the pilot; producing a
substantially larger licensed corpus is future work and a prerequisite for serious
pretraining (§37 steps 12 and 14). The historical text below is kept as written.

*Previous heading: "Stage 3 — Real data pipeline ⏭ NEXT (FrontierCorpus v1)".*

**Goal:** a defensible, versioned pipeline from licensed/public sources to training shards.

**Founder's decision (2026-09-26):** this is the next project, FrontierCorpus v1 (see
[MASTER_CONTEXT.md](MASTER_CONTEXT.md) §12–§17). It starts with freezing and verifying
`indic-tokenizer/v2` (§37 step 2).

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

**Status (2026-09-27): not started.** This is where **GPU access becomes necessary**
(§37 step 10). No GPU training has been performed yet; nothing earlier in the sequence,
including the step 8 evaluation harness, requires a GPU.

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

## Stage 6 — Evaluation 🔶 IN PROGRESS — intrinsic harness v1 (§37 step 8)

**Update (2026-09-27):** §37 step 8 builds the *intrinsic* part of this stage for the
current small research models: `scripts/eval_report.py` (one command → versioned report
for any checkpoint; every held-out token scored once; bits-per-byte / per-character
overall, per language, per script and per source with bootstrap confidence intervals;
data-identity and contamination checks; full provenance), `scripts/eval_compare.py`
(paired comparisons), and the protected suite `frontier-heldout-v1` (fingerprints only).
First use: EXP-031 (re-scoring the EXP-B models). Per-domain results are not available
(the pilot corpus has no domain labels); code-mixed evaluation waits for a lawful Hinglish
source. **Downstream benchmarks, Indic task suites and human evaluation remain future work**
(§37 step 15) — they are not meaningful for few-million-parameter models.

**Update (2026-09-27, later):** EXP-031 completed and D-042 was recorded. Harness v1 scored the 6
EXP-B models exactly (every held-out token; reproducible bit for bit; 0 contamination) and
confirmed D-040 in all 13 languages. The step 8 exit criteria are met for intrinsic
evaluation. Step 9 waits for founder approval.

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
