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

**Revisit when:** Stage 2 closes — after a first pass on real data (Stage 3), since
tokenizer design depends on the corpus. Project 002 (2026-09-10) built the evaluation
framework for this decision; it did **not** close it (see D-021).

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

**2026-09-10 update:** Project 002 added the first optional runtime dependency,
`tokenizers` (D-018), behind an extra (`pip install ".[tokenizer]"`). The core training
pipeline remains torch + NumPy.

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

---

## D-018 — Subword baseline: HuggingFace `tokenizers` BPE, with our own BPE as a dependency-free reference
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** The baseline subword tokenizer is byte-level BPE via the HuggingFace
`tokenizers` package (0.23.2, Apache-2.0), installed as an **optional** extra
(`pip install ".[tokenizer]"`). In parallel we implement and maintain our own byte-level
BPE (`bpe_python`) with no dependencies.

**Rationale:** `tokenizers` is established, permissively licensed, fast (Rust), supports
training from a local file, save/load, special tokens and vocab introspection, and works
fully offline (we only use `train`/`save`/`from_file`). Keeping our own implementation
means the research harness runs without the dependency, gives us a correctness oracle, and
lets us change pre-tokenization — which is where the interesting Indic questions live.

**Alternatives considered:** `sentencepiece` (Apache-2.0, Unigram/BPE, C++ extension,
weaker introspection, slower install path); `transformers` (far heavier than needed);
only our own implementation (no established reference to sanity-check against).

**Consequences:** One optional runtime dependency (~12 MB, plus ~7 MB transitive
`huggingface_hub`). If it is missing, `bpe_hf` is simply not registered and the subsystem
keeps working. **No network access is performed at any point.**

**Revisit when:** the vocabulary/pre-tokenization ablations (Q-9, Q-10) produce a design we
want to own end to end, or the dependency becomes a deployment problem.

---

## D-019 — Tokenizers are pluggable via an interface + registry
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** All tokenizers implement `SubwordTokenizer` (train / encode / decode /
vocab_size / special tokens / save / load) and register by name in `registry.py`. New
implementations are added without touching the evaluator, comparator, or CLIs. The
Project 001 char/word tokenizers are wrapped by adapters rather than modified.

**Rationale:** Tokenizer design is one of the most likely things to change. Hard-coding one
implementation would make every future comparison a refactor. The registry also lets us
keep weak baselines (char/word) in the same harness, which is what makes measurements
interpretable.

**Alternatives considered:** a single configurable tokenizer class (rigid; comparisons
require editing code); a heavy plugin framework (unnecessary at this scale).

**Consequences:** A small amount of interface boilerplate. Adapters must be updated if
Project 001's tokenizer API changes.

**Revisit when:** adding Unigram/WordPiece or a script-aware tokenizer — they should
implement the same interface.

---

## D-020 — Research corpus: deterministic, hash-verified, ours, and explicitly a fixture
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Tokenizer experiments use corpora generated by our own code
(`tokenization/corpus.py`): deterministic training text plus a hand-written evaluation
fixture (179 probes, 14 languages + 9 orthography categories). The loader verifies the
fixture sha256 against the manifest and **refuses to run** on a modified corpus. No
scraping, no private or copyrighted data, no third-party text.

**Rationale:** Reproducibility and legal safety are non-negotiable for a lab that intends
to release models. A hash check turns "someone edited the fixture" from a silent
measurement drift into a loud failure.

**Alternatives considered:** download a public corpus in CI (network + licensing +
version drift); ship a scraped sample (unacceptable legally).

**Consequences:** The fixture is tiny and shares vocabulary with the generated training
text, so compression numbers are **optimistic**. Every report carries that caveat, and the
comparator refuses cross-corpus comparisons.

**Revisit when:** Stage 3 produces real licensed data — then the same evaluator runs
against a real held-out set, and the fixture stays as a fast CI probe.

---

## D-021 — No production tokenizer is selected by Project 002
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Project 002 builds the framework and reports first measurements
(EXP-002). It does **not** designate a production tokenizer, and EXP-002 must not be cited
as evidence for one.

**Rationale:** The fixture is 179 hand-written examples. Sizing a vocabulary or choosing a
pre-tokenizer for Indian languages from that would be exactly the "scale past your ability
to evaluate" failure D-009 warns about.

**Alternatives considered:** pick the best-performing variant now (rejected: the ranking is
fixture-dependent and one of the two BPEs differs only by pre-tokenization, which is
itself an open ablation).

**Consequences:** Stage 2 stays open; D-010 remains in force for the training pipeline.

**Revisit when:** vocabulary sweeps, pre-tokenization ablations (Q-9) and a
tokenizer→model-quality measurement exist on representative data.

---

## D-022 — Losslessness is a gate, not a metric
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** A tokenizer with any round-trip failure (`decode(encode(x)) != x`) is
disqualified from "best" in comparisons and is reported separately, regardless of its
compression numbers.

**Rationale:** In our first measurement the word-level tokenizer had the best compression
while silently destroying 160 of 179 examples through unknown tokens. Ranking on
compression alone would have recommended it.

**Alternatives considered:** report failures as just another column (too easy to ignore);
exclude lossy tokenizers entirely (we still want to see their numbers for diagnosis).

**Consequences:** The comparison output has both a `best` block (lossless only) and a
`disqualified` block.

**Revisit when:** never — this is a standing rule for tokenizer comparisons.


---

## D-023 — Source code must never be git-ignored; artifact rules are anchored
**Date:** 2026-09-10 · **Status:** accepted

**Decision:** Ignore rules for generated artifacts are written anchored to the repository
root (`/data/`, `/out/`, `/artifacts/`), and `tests/test_repo_hygiene.py` fails the build if
any `.py` file under `src/`, `scripts/` or `tests/` is untracked or matched by
`git check-ignore`.

**Rationale:** The original `.gitignore` contained an unanchored `data/` rule intended for
prepared corpora at the repo root. Git matches such a pattern at *any* depth, so it also
matched `src/frontier_ai/data/` — the data package (tokenizers, token store, synthetic
corpus) was never committed. Everything worked locally, tests passed, and the branch was
pushed; only a **fresh checkout** revealed that `import frontier_ai.data` failed and four
test modules could not even be collected. The pushed PR was broken for anyone who cloned it.

**Alternatives considered:** remembering to be careful (no); relying on review (the diff
looked complete because the files were simply absent); negating rules with `!src/**/data/`
(harder to reason about than anchoring).

**Consequences:** Artifact directories at the root are still ignored; source directories
cannot be accidentally ignored. The hygiene tests depend on a git working tree and skip
cleanly when there isn't one.

**Revisit when:** never — this is a standing rule. Any new ignore pattern for a name that
could exist inside `src/` must be anchored.


## D-024 — One provenance mechanism, no new dependencies

**Status:** accepted (2026-09-10, Project 003)

**Decision:** Experiment infrastructure lives in `src/frontier_ai/experiments/` and is
stdlib-only apart from the torch/numpy Project 001 already requires. Data hashing
**re-exports** `sha256_file`/`sha256_text` from `frontier_ai.tokenization.corpus` (Project
002) rather than re-implementing them, and seeding goes through the existing
`frontier_ai.utils.seed.set_seed` rather than adding a parallel seeding helper.

**Rationale:** Duplicate hashing or seeding implementations would silently diverge — a
"reproducible" run whose digest disagrees with the corpus manifest is worse than no
provenance at all. A test asserts the new multi-file digest agrees with Project 002's
`sha256_file`, so divergence is caught. Standard-library `dataclasses`/`json`/`hashlib`
are sufficient for specs and records; a config framework or experiment tracker would add
a dependency and a migration burden before we know what we need (**Q-8**).

**Alternatives considered:** Hydra/pydantic for configuration (rejected: new dependency,
and Project 001 already owns `--set` override conventions); MLflow/W&B for tracking
(rejected for now: the record is a plain JSON file that is already diffable, and choosing
a tracker before running real experiments is premature — tracked as Q-8); a second hashing
module local to `experiments/` (rejected: two implementations of "the hash of our data").

**Consequences:** Adding a field to the record requires bumping `schema_version`; unknown
sections and mismatched schema versions are rejected on load rather than ignored.

**Revisit when:** we adopt a tracker (Q-8), or a record field proves insufficient for a
real experiment (then bump the schema, do not silently extend it).

## D-025 — Reproducibility claims are bounded by the recorded environment

**Status:** accepted (2026-09-10, Project 003)

**Decision:** The system claims reproducibility **under the recorded environment**, never
bit-identity across machines, library versions or thread counts. Known limits (cuDNN
nondeterminism, thread-dependent FP summation order, version sensitivity, RNGs outside
python/numpy/torch) are stored in every record as `randomness.limitations`, and the
content fingerprint deliberately excludes timestamps, pid, paths and log tails.

**Rationale:** A provenance system that over-claims is worse than none: it converts
"we don't know" into "this definitely reproduces". Recording the limits in the same file
as the claim makes the boundary impossible to miss, and excluding genuinely variable
metadata is what makes the fingerprint usable as a determinism test.

**Alternatives considered:** claiming full bit-reproducibility (false on CUDA and under
different thread counts); excluding nothing and comparing whole files (fingerprints would
differ between every run, so they could never be used as a regression test).

**Consequences:** Two runs are comparable through `content_fingerprint()`; a run on
different hardware or a different torch version is *documented* as not comparable, and the
environment section is what makes that judgement possible.

**Revisit when:** we run our first multi-machine or multi-seed experiment and learn which
of these limits actually bites.

## D-026 — Dirty trees and missing git are recorded, never hidden or guessed

**Status:** accepted (2026-09-10, Project 003)

**Decision:** Git provenance captures commit, branch, dirty flag, up to 50 dirty paths,
commit subject and remote. A dirty tree sets `reproducible_from_commit=false` and prints a
warning, but the run proceeds and is recorded. If git is unavailable (not a repository, no
commits, git binary missing, timeout), the record stores `available=false` with a
human-readable reason and **empty** commit/branch — we never invent or abbreviate a SHA
into a field that looks authoritative. Detached HEAD records `branch=None`.

**Rationale:** Most real research runs happen on dirty trees; refusing to run would push
people back to unrecorded runs, and silently marking them clean would make the record lie.
An empty-but-explicit "unavailable" is honest, whereas a fabricated or partially-derived
SHA is indistinguishable from a real one in a report.

**Alternatives considered:** refusing to run on a dirty tree (rejected: drives work
outside the system); recording only `git describe` output (rejected: ambiguous, and
fails in the detached/no-tags case); raising on unavailable git (rejected: provenance
capture must never break the experiment it is describing).

**Consequences:** A record from a dirty tree must not be cited as reproducible from its
commit — the flag says so. CI and release runs should still be clean.

**Revisit when:** we need release-grade provenance (then fail the run on dirty, as a
policy *above* this layer, not inside it).

## D-027 — Environment provenance is a selected field set, never an environment dump

**Status:** accepted (2026-09-10, Project 003; implemented since Stage 1 and recorded here
retroactively — `src/frontier_ai/experiments/environment.py` already refers to this ID)

**Decision:** `capture_environment()` records a deliberately small, stable set of fields:
python version and implementation, platform/system/release/machine, the versions of the
packages that can plausibly change a numerical result (`torch`, `numpy`, `tokenizers`,
`frontier-ai`), and a torch section (version, CUDA availability, device count, thread
count). Hostname, username, environment variables and full `pip freeze` output are
excluded, and the exclusions are themselves recorded in the artifact as
`excluded_by_policy` so that a reader can see they were deliberate.

**Rationale:** A record is read by humans and diffed by machines. Environment dumps are
noisy, change between runs on the same machine, leak information about the user, and make
two otherwise identical records look different — which destroys the value of a content
fingerprint. The fields that remain are exactly the ones needed to answer "could this
difference be a version or thread-count effect?".

**Alternatives considered:** recording `pip freeze` (rejected: hundreds of irrelevant lines
per record); recording the whole `os.environ` (rejected: leaks secrets and varies with the
shell); recording nothing (rejected: version and thread-count effects are the most common
cause of "it doesn't reproduce").

**Consequences:** If a result depends on something outside the recorded fields (a BLAS
library, a driver version, an env var), the record will not show it. That is accepted
boundedness, consistent with D-025 — and the fix is to add one more selected field, not to
start dumping environments.

**Revisit when:** a real reproducibility failure is traced to something outside the recorded
fields, or when GPU runs need driver/CUDA-runtime detail that `torch` does not expose.

## D-028 — Multi-seed sweeps: one record per seed, sample standard deviation as spread

**Status:** accepted (2026-09-10, Project 003 Stage 1A)

**Decision:** A sweep (`src/frontier_ai/experiments/sweep.py`,
`scripts/experiment_sweep.py`) runs one `ExperimentSpec` across an explicit list of seeds
and writes **both** one normal experiment record per seed (`seed-<seed>/experiment.json`,
with full git/data/environment/configuration provenance and its own content fingerprint)
**and** an aggregate `sweep.json` (schema `1.0`). Specifically:

* **Spread is the sample standard deviation** (`n − 1` denominator), computed with the
  standard library. It is recorded as `null` — never `0` — when fewer than two runs are
  usable, and it is **not** reported as a confidence interval, because none is computed.
* **A seed counts as usable only if it ran and produced a numeric value** for the requested
  metric. Missing or non-numeric values are never replaced by zero.
* **Sweep status is `success` / `partial` / `failed`**, so a sweep that lost a seed can never
  report success.
* **Seed order does not matter:** the seed list is validated, de-duplicated and sorted
  before anything runs, values are aggregated in seed order, and the template spec is
  recorded with the first canonical seed. `[1, 2, 3]` and `[3, 1, 2]` give the same
  aggregate and the same sweep fingerprint.
* Seeds run in **separate directories**, so they cannot overwrite each other's outputs.

**Rationale:** The roadmap asks for "mean ± spread" on headline numbers. Aggregating in a
way that hides individual runs would make the aggregate unverifiable; reporting a
population standard deviation would understate dispersion for small `n`; and calling it a
confidence interval would claim a coverage we never compute. Keeping one record per seed
also means a sweep costs nothing in provenance terms compared with running the seeds by
hand.

**Alternatives considered:** collapsing the runs into a single aggregate record (rejected:
loses per-seed provenance and the per-seed fingerprints that make the aggregate checkable);
population standard deviation (rejected: biased low for the 3–5 seed sweeps we can afford);
reporting a 95% CI from a t-distribution (rejected as over-claiming at these sample sizes —
a t-interval on 3 seeds assumes normality we have not tested); NumPy for the statistics
(rejected: `statistics.fmean`/`stdev` are sufficient and the project favours stdlib where
it is enough — D-024); stopping the sweep at the first failure (kept available as
`continue_on_error=False`, but not the default: an expensive sweep should not be lost to
one bad seed, as long as the failure is visible).

**Consequences:** The mean of a partial sweep is computed over a subset of the requested
seeds; readers must look at `successful_count` and `failed_seeds`. Spread from three seeds
is a rough dispersion estimate, not an inferential statistic — conclusions that need
tight error bars require more seeds than this repository's CPU budget allows so far.

**Revisit when:** we run sweeps large enough (`n >= 10`) for interval estimates to be
meaningful, or when a sweep over *configurations* (Q-8) needs a shared aggregation layer
with this one.

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
| Q-8 | Whether to adopt an experiment tracker (and sweep orchestration) on top of the JSON records, and which | Stage 1/4 |
| Q-9 | Pre-tokenization: mark-aware vs GPT-2-style regex, whitespace attachment, script-aware initial alphabets | Stage 2 (EXP-002 flagged this as the main difference between our BPE and the HF baseline) |
| Q-10 | Vocabulary sizing policy for Indic multilingual models | Stage 2, on real data |
| Q-11 | Unicode normalization policy (NFC/NFD/None) as a tokenizer-level decision | Stage 2 |
| Q-12 | BPE vs Unigram for our data mix | Stage 2 |

## How to add a decision

1. Next free ID (`D-0xx`) or a question ID (`Q-x`) if it is genuinely still open.
2. State the decision in one sentence, then the rationale, the alternatives you rejected,
   and the consequences you accept.
3. Set a status and a **revisit when** trigger — a decision without a trigger becomes
   invisible debt.
4. Never silently reverse an accepted decision in code: supersede it here with a new
   record that points at the old one.
