# DECISIONS.md

> Architectural and strategic decision records for the frontier-ai research repository.
> Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) first.
>
> Format: **ID · date · status · decision · rationale · alternatives considered ·
> consequences · revisit when**. Statuses: `accepted` (in force), `provisional` (accepted
> for now, expected to change), `deferred` ( consciously postponed), `superseded`.

---

## D-001 — Train from random initialization; do not wrap someone else's model
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Foundation models are trained by us from random initialization. Our weights,
our tokenizer, our data recipe.

**Rationale:** The goal is an independent frontier lab. A wrapper around a proprietary API
or another party's weights has no defensible core asset, no research surface, and no path
to differentiation. Ownership of the trained weights is the asset.

**Alternatives considered:** (a) ship products on third-party APIs and skip research —
fine for the *product* track short term, fatal for the foundation track; (b) fine-tune
open-weights models as the foundation — acceptable for baselines and experiments, rejected
as the endpoint.

**Consequences:** Far longer path to a usable model; we must build data, tokenizer,
training, and eval systems ourselves.

**Revisit when:** never for the foundation track. Revisit only the *interim* use of
open-weights models for product features.

---

## D-002 — PyTorch is the core framework
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** PyTorch (with NumPy for data plumbing) is the implementation framework.

**Rationale:** De facto standard for research; largest ecosystem (kernels, distributed
training, tooling); easiest hiring and knowledge transfer; lets us spend effort on modeling
rather than framework plumbing.

**Alternatives considered:** JAX (excellent for TPU-centric scale, smaller ecosystem for
our needs), writing our own C++/CUDA framework (massive cost, no strategic benefit since
the model — not the autodiff engine — is the asset).

**Consequences:** Accept PyTorch's release cadence and some abstraction overhead. Keep the
model code in plain PyTorch so kernels remain swappable.

**Revisit when:** a framework shift is justified by a measured need (e.g. TPU-first
training at scale) and the migration cost is quantified.

---

## D-003 — CPU-first development, GPU-ready code, one device-agnostic path
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Default configs are small enough to train in seconds on 2 CPU cores. Device,
autocast dtype, and GradScaler policy are resolved **once** at runtime in
`src/frontier_ai/utils/device.py` (cuda → mps → cpu; bf16 if supported, else fp16, else fp32).
No `if cuda:` branches in the model or training loop.

**Rationale:** Development happened in a CPU-only sandbox (2 cores, 3 GB RAM). Fast, cheap
feedback on code correctness matters more than throughput while the architecture is moving.
Real GPU training must then require a config change, not a code change.

**Alternatives considered:** write GPU-first code and test only on GPUs (no GPU available;
slow, expensive feedback loop); maintain separate CPU and GPU code paths (drift and bugs —
explicitly rejected).

**Consequences:** bf16/fp16, flash attention, `torch.compile`, and GradScaler paths are
written but **unverified** until Stage 4. Some CPU-only simplifications (no pinned memory,
no non-blocking-heavy dataloading) are acceptable for now.

**Revisit when:** GPUs are available — then verify the untested paths first (Stage 4).

---

## D-004 — Config-driven, modular architecture
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Hyperparameters live in dataclasses (`config.py`) with validation in
`__post_init__`, serialized as JSON, overridable from the CLI via `--set section.key=value`.
Packages are split by responsibility: `data/`, `model/`, `engine/`, `utils/`.

**Rationale:** Every experiment is a config diff, which makes runs reproducible and
reviewable. Validation catches typos before a long run starts. Modularity lets the
tokenizer, data pipeline, and distributed strategy be replaced without touching the model.

**Alternatives considered:** YAML configs (adds a dependency for no benefit at this size);
hardcoded constants in scripts (not reproducible); a heavy experiment framework
(Hydra/Ray — premature).

**Consequences:** Some boilerplate in `config.py`; `--set` only handles simple scalar and
JSON-list values.

**Revisit when:** sweeps and nested overrides become common (Stage 1) — consider a runner
built **on top of** these dataclasses rather than replacing them.

---

## D-005 — Standard modern architecture, implemented from scratch
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Decoder-only transformer with pre-norm blocks, RMSNorm, SwiGLU FFN, tied
embeddings, no biases, `scaled_dot_product_attention`, optional RoPE and grouped-query
attention, KV-cached sampling.

**Rationale:** These choices are well-understood, stable, and cheap to train — the right
baseline to *learn the stack on*. Each is a config switch, so ablations are easy later.
Writing it ourselves (~250 lines, fully tested) means we understand and control every line
before adding research ideas.

**Alternatives considered:** start from an existing implementation (faster, but we would
inherit decisions we don't understand); add novel components immediately (no baseline to
compare against, and no way to tell if a change helped).

**Consequences:** No claim to novelty at this stage — that is intentional. Every "modern"
component is available but only some are exercised by the smoke config (learned positions,
not RoPE; no GQA at 2 heads).

**Revisit when:** ablations begin (Stage 4+) — then switch norms, position encodings, and
attention variants one at a time and measure.

---

## D-006 — Research repository is separate from the Lovable control-center UI
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** This repository holds research and training code only. The product
control-center UI lives in Lovable and is **not** in this repository.

**Rationale:** Different lifecycles, contributors, and review standards. Mixing them means
every UI tweak pollutes research history and every experiment complicates product deploys.
The product track must never be blocked by research, and research must not be constrained
by product release cadence.

**Alternatives considered:** monorepo (rejected for the coupling above); product repo
consuming this one as a git submodule (revisit if a shared model interface is needed).

**Consequences:** Any shared contract (model API, checkpoint format, eval outputs) must be
documented explicitly in both places.

**Revisit when:** the product needs to serve our own models — then define and version the
interface between serving and this repo.

---

## D-007 — No proprietary model API in the core model path
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Third-party model APIs may be used for tooling, data-generation assistance,
and evaluation baselines. They must never be the core model intelligence of a product we
present as ours.

**Rationale:** A dependency that can be repriced, rate-limited, or revoked is not a
foundation. It also leaks data and prevents the research loop from closing.

**Alternatives considered:** API-first product development (accepted **only** for the
product track as an interim measure, with migration planned).

**Consequences:** Short-term product capability is bounded by our own models; interim
products may run on third-party APIs with that fact disclosed.

**Revisit when:** a product surface is ready to migrate onto our own weights (Stage 7+).

---

## D-008 — Tests and reproducibility are part of the deliverable, not optional extras
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Behavior changes ship with tests. Every run records its config, seed, data
version, environment, and metrics. Two bugs found during Project 001 (loss summed instead
of averaged over accumulation steps; `--set` flags overwriting each other) got permanent
regression tests.

**Rationale:** In ML, silent bugs produce plausible-looking numbers. The failure mode is
not a crash, it is a wrong conclusion. Tests here target exactly the places that were
wrong: masking, KV-cache equivalence, target construction, loss normalization, resume
fidelity, config overrides.

**Alternatives considered:** move fast and test later (rejected: "later" never comes, and
debugging at scale is far more expensive).

**Consequences:** Slower initial feature velocity; much higher confidence in results.

**Revisit when:** never. Extend with determinism and multi-seed checks in Stage 1.

---

## D-009 — Incremental research; no premature scaling
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Explicitly out of scope until earlier stages are complete: 7B/70B training,
RLHF/preference optimization, multimodality, agents, and frontier-scale training. Model
size is an output of experiments, never a target set in advance.

**Rationale:** A large model trained on an unproven pipeline produces an expensive artifact
nobody can debug or evaluate. Each capability in the roadmap depends on the ones before it;
skipping ahead converts a research program into a fundraising slide.

**Alternatives considered:** go big early to attract attention (rejected: we could not
evaluate or improve what we built, and failure would be both public and expensive).

**Consequences:** Slower visible progress; much higher probability of eventually having
something real.

**Revisit when:** each stage's exit criteria are met — see [ROADMAP.md](ROADMAP.md).

---

## D-010 — Char/word tokenizer now; trained BPE later
**Date:** 2026-09-10 · **Status:** provisional

**Decision:** Ship `CharTokenizer` and `WordTokenizer` (both JSON-serializable, behind a
shared interface). Defer a trained BPE/Unigram tokenizer to Stage 2.

**Rationale:** A tiny vocabulary (51 tokens for the synthetic corpus) keeps the CPU smoke
test fast and the model small, and lets the rest of the pipeline be proven without
depending on tokenizer training. The `Tokenizer` interface (`encode`/`decode`/`save`/`load`)
was designed so the swap is contained.

**Alternatives considered:** integrate HuggingFace `tokenizers` immediately (adds a
dependency and a training step before we have data worth tokenizing).

**Consequences:** Current perplexity/bits numbers are **not** comparable to published
token-level numbers. No EOS/BOS/chat tokens exist yet, so `stop_at_eos` in `generate()` is
unused.

**Revisit when:** Stage 2 (tokenizer) begins — after a first pass on data (Stage 3), since
tokenizer design depends on the corpus.

---

## D-011 — Synthetic corpus as the default smoke-test data
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** `data/synthetic.py` generates a deterministic pseudo-English corpus from a
small grammar; it is the default data for smoke tests and CI.

**Rationale:** Zero network access, zero licensing questions, byte-identical across
machines, and structured enough that loss visibly falls within seconds — which is exactly
what a smoke test needs.

**Alternatives considered:** download a public corpus in CI (network dependency, licensing
review, non-determinism across versions).

**Consequences:** It is a **test fixture, not training data**. Its results say nothing about
language modeling quality. This must be stated wherever its numbers appear.

**Revisit when:** Stage 1 adds a small real, clearly licensed corpus for sanity checks;
synthetic data then stays as the fast CI fixture.

---

## D-012 — Dependency-light: torch + NumPy only
**Date:** 2026-09-10 · **Status:** provisional

**Decision:** Runtime dependencies are `torch` and `numpy`. Configs are JSON (no PyYAML).
No experiment tracker yet. Dev extras: `pytest`, `ruff`.

**Rationale:** Fewer dependencies means the codebase survives torch upgrades and installs
cleanly in constrained environments. Note also that the PyTorch CPU wheel CDN
(`download.pytorch.org`) was **blocked** in the development sandbox, while PyPI worked —
a reminder that install paths must have fallbacks (the CI example does).

**Alternatives considered:** add Hydra/WandB/accelerate early (rejected: premature, and
each is a lock-in surface we can add deliberately later).

**Consequences:** Manual sweep tooling for now; metrics live in `train.jsonl` until a
tracker is justified.

**Revisit when:** Stage 1 (experiment runner) and Stage 4 (GPU training at scale).

---

## D-013 — CI workflow shipped as `docs/ci.yml.example`, not `.github/workflows/`
**Date:** 2026-09-10 · **Status:** accepted (constraint-driven)

**Decision:** The GitHub Actions workflow is committed as `docs/ci.yml.example` and enabled
with `mkdir -p .github/workflows && cp docs/ci.yml.example .github/workflows/ci.yml`.

**Rationale:** The GitHub App available in the development environment lacks the
`workflows` permission, so pushing files under `.github/workflows/` is rejected by the
remote. This is an environment constraint, not a design choice.

**Alternatives considered:** drop CI entirely (rejected: tests need to run automatically);
commit it anyway (impossible — push rejected).

**Consequences:** CI has never actually run on GitHub; the local workflow (lint + tests +
CLI smoke) is verified manually only.

**Revisit when:** the App has the `workflows` permission, or a maintainer moves the file
manually — do this before relying on CI as a gate.

---

## D-014 — Device, dtype, and GradScaler policy resolved in one place
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** `utils/device.py` owns device selection (`resolve_device`), precision policy
(`resolve_spec`), and thread counts (`threads_for`). `Trainer` consumes a `DeviceSpec`;
GradScaler is enabled only for fp16 on CUDA (bf16 and fp32 need none).

**Rationale:** Mixed precision is where silent numerical bugs live. Centralizing the policy
means one place to audit, one place to test, and no scattered conditionals.

**Alternatives considered:** inline `torch.autocast` calls with local flags (rejected:
policy drift between train, eval, and generation).

**Consequences:** Adding a new backend (e.g. ROCm, XPU) is a change to one function.

**Revisit when:** a new accelerator backend is added.

---

## D-015 — Checkpoint = a self-describing directory
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Each checkpoint directory contains `model.pt` (weights only, fp32 master),
`optimizer.pt` (optimizer + scheduler state), `config.json` (full experiment config), and
`meta.json` (step, best_val, tokens_seen, n_params, torch version, timestamp). Runs also
write `train.jsonl` and the resolved config at the run root.

**Rationale:** A checkpoint that cannot be reproduced is not an artifact, it is a rumor.
Storing the config alongside the weights means any checkpoint can be re-instantiated and
evaluated without digging through history.

**Alternatives considered:** a single `.pt` file with everything pickled (opaque, and
breaks `weights_only=True` loading — see below).

**Consequences:** Slightly more files per checkpoint. Scheduler state must exclude
non-serializable config objects so `torch.load(weights_only=True)` keeps working (this was
fixed during Project 001).

**Revisit when:** distributed training arrives — checkpoint/restore must then be robust to
changing world size (Stage 5).

---

## D-016 — MIT license
**Date:** 2026-09-10 · **Status:** provisional

**Decision:** Code is MIT-licensed.

**Rationale:** Maximum freedom to use the tooling and to publish research code.

**Alternatives considered:** Apache-2.0 (adds a patent grant — arguably better for a
company that may contribute to standards); a research-only license (impedes adoption and
collaboration).

**Consequences:** Others may use this training code commercially. Model weights, data, and
research outputs can be licensed separately later — this license covers **code only**.

**Revisit when:** we begin releasing models or datasets, whose licenses should be decided
separately (and before any release).

---

## D-017 — One branch per Arena session; changes land via PR; no direct pushes to `main`
**Date:** 2026-09-10 · **Status:** accepted (workflow constraint)

**Decision:** Work happens on the session branch (currently
`arena/01a08a78-frontier-ai`), is pushed there, and is reviewed as a PR against `main`.
Agents must not create other branches, merge PRs, or push to `main`.

**Rationale:** Arena tracks each session by branch; work elsewhere is not associated with
the session. PRs keep a reviewable record of what each project changed.

**Alternatives considered:** commit directly to `main` (rejected: no review checkpoint, and
the user explicitly requires PRs).

**Consequences:** Long-term history is a sequence of project PRs, which is also a useful
narrative record for the lab.

**Revisit when:** the user changes the branching policy.

---

## Open items to decide later (not yet decisions)

| ID | Question | Deferred to |
|---|---|---|
| Q-1 | BPE vs Unigram tokenizer; vocabulary size | Stage 2 |
| Q-2 | Data source mix and licensing review process | Stage 3 |
| Q-3 | DDP vs FSDP as the primary distributed strategy | Stage 5 |
| Q-4 | Which benchmark suite becomes our primary gate | Stage 6 |
| Q-5 | SFT data sourcing: licensed vs synthetic vs human-written | Stage 7 |
| Q-6 | RL method for reasoning (GRPO/PPO family) | Stage 8 |
| Q-7 | Model/weights license for released checkpoints | Stage 9 |
| Q-8 | Whether to adopt an experiment tracker, and which | Stage 1/4 |

## How to add a decision

1. Next free ID (`D-0xx`) or a question ID (`Q-x`) if it is genuinely still open.
2. State the decision in one sentence, then the rationale, the alternatives you rejected,
   and the consequences you accept.
3. Set a status and a **revisit when** trigger — a decision without a trigger becomes
   invisible debt.
4. Never silently reverse an accepted decision in code: supersede it here with a new
   record that points at the old one.
