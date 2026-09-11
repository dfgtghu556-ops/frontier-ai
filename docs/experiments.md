# Experiment infrastructure (Project 003 — ROADMAP Stage 1)

**Purpose:** make every run answerable by one question — *exactly what code, configuration,
data, random seed, environment and command produced this result?*

This is **infrastructure only**. It changes no research direction, makes no production
tokenizer or model decision, and does not touch the numerical core of Project 001 or 002.
It is the mechanism that lets every later stage be trusted.

Package: [`src/frontier_ai/experiments/`](../src/frontier_ai/experiments/)
CLI: [`scripts/experiment_record.py`](../scripts/experiment_record.py)
Tests: [`tests/test_experiments.py`](../tests/test_experiments.py)

| | |
|---|---|
| Python | 3.9+ (`from __future__ import annotations`) |
| New dependencies | **none** — stdlib plus the torch/numpy already required by Project 001 |

---

## 1. Quickstart

### Python

```python
from frontier_ai.experiments import ExperimentSpec, run_experiment

spec = ExperimentSpec(
    experiment_id="EXP-003",          # must match ^EXP-\d{3,}$
    seed=1337,                        # one master seed, [0, 2**32-1]
    name="cpu-smoke",
    output_dir="out/experiments/EXP-003",
    data_paths=["data/synthetic.bin"],
    config_path="configs/cpu_smoke.json",
    params={"max_steps": 50, "lr": 3e-4},
    tags=["project-001"],
)

def my_experiment(ctx):
    ...                               # ctx.derived("data") -> a per-component seed
    return {"val_loss": 1.37}

outcome = run_experiment(spec, my_experiment)
print(outcome.path, outcome.record.content_fingerprint())
```

`run_experiment` returns a `RunOutcome(record, path, rendered)` and **never swallows an
exception**: on failure it writes a `status="failed"` record (error type, message,
traceback tail) and re-raises.

To run the same experiment across several seeds (and optionally several configurations)
and get `mean ± spread`, see [§8 Sweeps](#8-sweeps-seeds-and-configurations-mean--spread)
(`run_sweep` / `scripts/experiment_sweep.py`).

### CLI

```bash
. .venv/bin/activate

# wrap an existing Project 001 training run
python scripts/experiment_record.py --exp-id EXP-003 --seed 1337 \
    --config configs/cpu_smoke.json --data data/synthetic.bin \
    --out out/experiments/EXP-003-train --tag project-001 -- \
    python scripts/train.py --config configs/cpu_smoke.json --max-steps 50

# wrap a Project 002 tokenizer run
python scripts/experiment_record.py --exp-id EXP-003 --seed 1337 \
    --data data/tokenizer/indic-v1/train.txt --out out/experiments/EXP-003-tok -- \
    python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 \
        --impl bpe_hf --vocab-size 1024 --out artifacts/tokenizers/bpe_hf_1024

# provenance only (no command): describe the current code + data state
python scripts/experiment_record.py --exp-id EXP-003 --data data/synthetic.bin \
    --out out/experiments/inspect
```

Everything after `--` is the recorded command; its exit code is propagated. Flags follow the
Project 001/002 conventions: `--help`, `--exp-id`, `--seed`, `--name`, `--out`, `--config`,
`--data`, `--set key=value`, `--tag`, `--notes`, `--deterministic`, `--timeout`, `--quiet`.

Real output from this repository (wrapped command, dirty working tree):

```
Experiment EXP-003 — docs example
============================================================
status        : success
code          : 10e7aa9c on arena/01a08a78-frontier-ai (dirty)
data          : 2 file(s), 8b15610be1cc…
seed          : 1337 (deterministic mode: False)
environment   : python 3.11.2 / torch 2.14.0+cu130 / Linux
execution     : success
duration      : 0.011s
command       : python3 -c print('hello from the wrapped command')
tags          : docs

results:
  exit_code                0
  stdout_tail              ['hello from the wrapped command']
  stderr_tail              []

content fingerprint: c8af7658ca2deea1… (timestamps excluded)
WARNING: ran from a dirty working tree — not reproducible from the commit alone

{"experiment_id": "EXP-003", "record": "out/experiments/.../experiment.json",
 "fingerprint": "c8af7658ca2deea1f6520b7de579b8130efd24688732204b1c23a2b77435edea"}
```

---

## 2. The experiment record

One JSON file per run, `experiment.json` (plus a rendered `experiment.txt` twin), written
deterministically with sorted keys and fixed separators.

| section | contents |
|---|---|
| `schema_version` | `"1.0"` — bumped on any breaking section change |
| `record_type` | `"experiment"` |
| `experiment` | id, name, tags, notes, **status**, output dir |
| `code` | git commit / branch / dirty flag / dirty files / remote / reproducibility flag |
| `data` | input paths, SHA-256 digest, per-file digests, sizes, skipped symlinks |
| `configuration` | the full spec, CLI overrides, embedded config file |
| `randomness` | master seed, deterministic-mode flag, seeded RNGs, derived seeds, limitations |
| `environment` | python, platform, selected package versions, torch/device summary |
| `execution` | command argv, status, exit code, duration, stdout/stderr tails, error |
| `results` | whatever the experiment function returned (JSON-serializable) |

Loading validates the schema version and rejects unknown sections
(`ExperimentRecordError`), so a record written by a future version fails loudly instead of
being silently misinterpreted.

### Content fingerprint

`record.content_fingerprint()` is the SHA-256 of the canonical JSON with **variable
metadata removed**:

```
started_at, finished_at, duration_seconds, pid, cwd, output_dir, artifact_dir,
record_path, stdout/stderr tails
```

Two identical runs therefore produce the **same fingerprint** on different machines and at
different times; a change in code, data, seed, config or results changes it. This is what
the determinism tests assert.

---

## 3. Seeding — one master seed, honest limits

`seed_everything(seed, deterministic=False, components=("data", "model", "sampling"))`

* Seeds `random`, `numpy.random`, `torch` (CPU) and `torch.cuda` through the **existing**
  `frontier_ai.utils.seed.set_seed` — one mechanism, no parallel seeding code (D-024).
* Global RNGs receive the **master** seed itself, so results are stable if you already rely
  on `set_seed`.
* Each component gets an independent derived stream:
  `derive_seed(master, *components) = sha256("<master>|<components joined by />") mod 2**32`.
  Derivation is SHA-256 based, so it is stable across processes, machines and Python
  versions (Python's `hash()` is deliberately not used).
* `--deterministic` / `deterministic=True` additionally asks torch for deterministic
  algorithms (`warn_only`, since some ops have no deterministic kernel).

**Documented limitations** (stored in the record, not hidden in a comment):

1. CUDA/cuDNN kernels may stay nondeterministic even in deterministic mode.
2. CPU results depend on thread count — multithreaded reductions change FP summation order.
3. Library versions can change results for the same seed (hence the environment section).
4. Only Python/NumPy/Torch RNGs are seeded: subprocesses, third-party samplers and
   OS-level nondeterminism are not covered.

We claim *reproducibility under the recorded environment*, not bit-identity across
machines, versions or thread counts.

---

## 4. Git provenance

`capture_git_info()` runs `git` in the repository and returns
`GitInfo(available, commit, branch, dirty, dirty_files, commit_subject, remote, reason)`.

* **Never invents a SHA.** If git is missing, the directory is not a repository, there are
  no commits, or the call times out, it returns `available=False` with a human-readable
  `reason` and empty commit/branch.
* **Dirty trees are visible, not fatal:** `dirty=True` plus the first 50 changed paths, and
  `reproducible_from_commit=False`. The rendered record prints a warning; the runner keeps
  going. A result produced from uncommitted code is still recorded — just not claimable as
  reproducible from the commit.
* Detached HEAD yields `branch=None` rather than a fabricated name.

---

## 5. Data provenance

`hash_paths(paths, root=None)` → `DataDigest`. The byte-level helpers
(`sha256_file`, `sha256_text`) are **re-exported** from Project 002's
`frontier_ai.tokenization.corpus`; there is exactly one hashing implementation in the
repository (a test asserts the digests agree).

Documented rules:

* **Algorithm**: SHA-256 over raw file bytes, hex digest.
* **One input file** → the digest is that file's content digest, independent of its path.
* **Several files or a directory** → the digest covers a manifest of
  `relative path + size + content digest` lines, so names and structure are part of the
  identity.
* **Ordering**: manifest lines are sorted by relative POSIX path; argument order and
  filesystem enumeration order cannot change the digest.
* **Relative to**: `root` if given, else the **common ancestor** of the inputs, so digests
  never depend on absolute machine paths.
* **Directories**: expanded recursively, files only; empty dirs contribute nothing.
* **Symlinks**: skipped, never followed; listed under `skipped`.
* **Missing input**: `FileNotFoundError` naming the path — we never hash an incomplete
  dataset, and the runner's `validate_inputs` fails *before* any compute is spent.
* **Empty input list**: `ValueError` (say explicitly that there are no inputs).
* **Generated output is never hashed as a stand-in for source data.**

---

## 6. Environment provenance

`capture_environment()` records a **stable, selected** set of fields: python version and
implementation, platform/system/release/machine, versions of `torch`, `numpy`, `tokenizers`
and `frontier-ai`, plus a torch section (version, `cuda_available`, device count, thread
count). No hostname, username, environment-variable dump or full `pip freeze` —
`EXCLUDED_BY_POLICY` names what is deliberately left out so the omission is visible.

---

## 7. Runner lifecycle

`run_experiment(spec, fn)` and `run_command(spec, argv)` execute the same nine steps:

```
validate → config → git → data → seed → execute → environment → results → write record
```

* **validate** — inputs exist before anything runs (`ExperimentInputError`).
* **seed** — the experiment body runs only after seeding; `ExperimentContext` exposes
  `ctx.seed`, `ctx.derived_seeds`, `ctx.derived(name)` and `ctx.write_artifact()`.
* **execute** — exceptions are recorded *and re-raised*; a failed run writes
  `status="failed"` with the error, its type and a traceback tail, and the human summary
  shows it. No run ever ends with a misleading "success".
* **environment** — captured **after** the body, because the body owns torch's CPU thread
  count: only the value the run actually used is true provenance. The caller's value is
  restored immediately afterwards, so one run in a process cannot change the environment,
  fingerprint or thread count of the next (D-034, resolved Q-13 — see §10).
* **write record** — `experiment.json` + `experiment.txt` in the run directory.

Both entry points are usable from Python and from the CLI, with identical semantics.

### 7.1 CLIs that record themselves (D-032)

`scripts/train.py` and `scripts/tokenizer_*.py` write the same record when they are run
directly, through `frontier_ai.experiments.autowire.run_self_recorded()` — a thin adapter
over `run_experiment`, not a second lifecycle.

```bash
python scripts/train.py --config configs/cpu_smoke.json --exp-id EXP-007
# [record] out/... /experiment.json | fingerprint 98c1b858cfae3e9f…

python scripts/experiment_record.py --exp-id EXP-007 --out out/wrapped -- \
    python scripts/train.py --config configs/cpu_smoke.json
# one record, the wrapper's; the training metrics are in its stdout_tail
```

* **The outer run owns the record.** The runner exports `FRONTIER_AI_EXPERIMENT_DIR` while
  a body executes; subprocesses inherit it and a nested script writes no record, saying so
  on stderr. Wrapped runs therefore produce exactly one record, not two.
* **…but the outer run still gets the metrics (D-034).** A nested script publishes its
  results as one machine-readable JSON line on stdout:

  ```json
  {"frontier_ai_nested_results": {"schema": "1.0", "script": "train.py",
                                  "results": {"best_val": 3.4715, "steps": 10, "n_params": 138752}}}
  ```

  `run_command()` recognises only that marker and merges the results into the outer record's
  `results` (its own `exit_code`/`stdout_tail`/`stderr_tail` are never overwritten). Sweeps
  therefore aggregate a swept `train.py` on `best_val` with `metric_source: "results"` — no
  log scraping, no argv inspection, no second record.
* **Artifact-producing runs record by default.** Training and every tokenizer CLI that
  writes files records; `--print-model` and `--eval-only` do not (no artifacts); every
  script has `--no-record`. `--exp-id` defaults to `EXP-000`.
* **What the record contains** is the standard record: command and config (including the
  config file's contents), master seed, git commit/branch/dirty state, environment
  (Python/torch/numpy/tokenizers), the digest of the inputs it read (corpus, tokenizer
  artifacts), the metrics it produced (`best_val`, `best_bpb`, `steps`, `n_params`, …),
  and success/failure. Paths and timestamps are kept out of `results` so two identical
  runs fingerprint the same.
* **Failures never look successful.** A body that raises writes `status: "failed"` (error
  type, message, traceback tail) and the CLI exits 1; a declared input that is missing
  aborts before the lifecycle with `[record] FAILED — ExperimentInputError: …`, also exit
  1. An experiment id that is not `EXP-<3+ digits>` (legacy ids such as `--exp-id
  EXP-TEST`) is not recordable, so the run keeps working and reports `[record] SKIPPED`.
* **A record is not data (D-033).** Hashing an input *directory* skips the
  `experiment.json` / `experiment.txt` of earlier runs, so a downstream digest does not
  depend on when an upstream run happened.

Tests: [`tests/test_autowire.py`](../tests/test_autowire.py).

---

## 8. Sweeps: seeds and configurations (mean ± spread)

A sweep runs **one experiment specification across several seeds**, and optionally across
several **named configurations**, and answers: *how much does the measured result vary
across independent seeds, and between configurations?* It is the mechanism behind the
roadmap rule "report mean ± spread for any headline number" and the Stage 1 exit criterion
"a 2-config sweep runs unattended".

Python:

```python
from frontier_ai.experiments import ExperimentSpec, run_sweep

spec = ExperimentSpec(experiment_id="EXP-004", seed=1, name="cpu-smoke",
                      output_dir="out/sweeps/EXP-004", data_paths=["data/synthetic.bin"],
                      config_path="configs/cpu_smoke.json")

outcome = run_sweep(spec, [1, 2, 3, 4, 5], my_experiment_fn, metric="final_val_loss")
print(outcome.record.statistics["mean"], outcome.record.statistics["spread"])
```

CLI (follows the Project 001/002/003 conventions; everything after `--` is the per-seed
command, with `{seed}` substituted):

```bash
python scripts/experiment_sweep.py --exp-id EXP-004 --seeds 1,2,3,4,5 \
    --metric final_val_loss --config configs/cpu_smoke.json --data data/synthetic.bin \
    --out out/sweeps/EXP-004-train --tag project-001 -- \
    python scripts/train.py --config configs/cpu_smoke.json --max-steps 50 \
        --set train.seed={seed}
```

Exit codes: `0` every seed usable, `2` partial (some seeds failed), `1` no seed usable.
The sweep record is written in all three cases.

### Individual runs are preserved

Every seed gets its own **normal experiment record** under
`out/sweeps/<id>/seed-<zero-padded seed>/experiment.json` with the same sections as any
other run: experiment identity, its own seed, git provenance, data provenance,
configuration, environment, execution status, result and content fingerprint. The
aggregate never replaces them, and the per-seed directories are distinct by construction,
so two seeds cannot overwrite each other's outputs. Duplicate seeds in the list are
de-duplicated rather than run twice into the same directory.

### What the aggregate records (`sweep.json`, schema `1.0`)

| section | contents |
|---|---|
| `sweep` | id, name, tags, notes, **status**, metric, `seeds`, `successful_seeds`, `failed_seeds`, `requested_count`, `successful_count`, `failed_count` |
| `configuration` | the template spec (seed/output_dir replaced per run), `seeds_requested`, overrides, embedded config file |
| `code` / `data` / `environment` | the same provenance captured once for the whole sweep |
| `statistics` | `metric`, `n`, per-seed `values`, `mean`, `spread`, `spread_kind`, `spread_definition`, `min`, `max` |
| `runs` | per run: status, `seed_used`, `metric_value`, `metric_source`, record path (relative), fingerprint, error — plus `configuration` for multi-configuration sweeps |
| `configurations` | **multi-configuration sweeps only**: one entry per configuration with its own status, seed lists, counts, resolved spec and `statistics` |
| `execution` | start/finish, duration, cwd, pid |

`sweep.txt` holds a rendered human summary.

### Mean and spread

* **mean** — arithmetic mean of the values from the *usable* runs.
* **spread** — **sample standard deviation**, `sqrt(Σ(xᵢ − mean)² / (n − 1))`, computed with
  the standard library (`statistics`); NumPy is not needed for a mean and a stdev.
* Spread is **undefined, and recorded as `null`, for fewer than two usable runs** — never
  `0.0`. A single-run sweep reports `mean = that value`, `spread = null`.
* **Spread is not a confidence interval**, and is never reported as one: no interval is
  computed. The definition travels with the record in `statistics.spread_definition`.
* Missing values are **never substituted with zero**.

### Seed order does not matter

The seed list is normalised (integers, validated range, de-duplicated, **sorted ascending**)
before anything runs, statistics are computed over values sorted by seed, and the template
spec is recorded with the first canonical seed. So `[1, 2, 3]` and `[3, 1, 2]` produce the
same statistics and the same sweep fingerprint, while each run still records the seed it
actually used.

### Failure semantics

* A run that raises keeps its **failed experiment record** (written by the runner before it
  re-raises), is listed in `failed_seeds` with its error type and message, and does **not**
  stop the other runs.
* A seed counts as successful only if it ran **and** produced a numeric value for the
  requested metric. A run that succeeded but has no such value is marked
  `status="metric_missing"` and is counted as failed for aggregation.
* Sweep status: `success` (all runs usable), `partial` (some usable), `failed` (none
  usable). Multi-configuration sweeps report the same three states per configuration as
  well, and are `partial` when any configuration lost a run — see
  [Multiple configurations](#multiple-configurations). A partial sweep never reports `success`, and a failed sweep reports
  `mean = null`, `spread = null` instead of a misleading number.
* `run_sweep(..., continue_on_error=False)` stops at the first failure (the sweep record is
  still written, then the original exception is re-raised). The CLI always continues.

### Reproducibility

Running the same sweep twice with the same code, configuration, data, configuration list
and seed list reproduces the individual metric values, the per-run fingerprints, the
aggregate statistics and the sweep **content fingerprint** — which, like the run fingerprint, excludes
timestamps, duration, pid, cwd, output paths and log tails. Per-run record references in
the aggregate are stored relative to the sweep directory for the same reason.

### Multiple configurations

A sweep can evaluate several **named configurations**, each across the same seed list —
this is the roadmap's "2-config sweep runs unattended" case: *does the ranking of two
configurations survive seed noise?*

```bash
python scripts/experiment_sweep.py --exp-id EXP-005 --seeds 1,2,3 \
    --configs "lr_low:params.lr=0.005" "lr_high:params.lr=0.05" \
    --metric final_val_loss --data data/synthetic.bin --out out/sweeps/EXP-005 -- \
    python my_experiment.py --config-name {config} --seed {seed}
```

From Python:

```python
outcome = run_sweep(spec, [1, 2, 3], my_experiment_fn, "final_val_loss",
                    configurations={"lr_low": {"params": {"lr": 0.005}},
                                    "lr_high": {"params": {"lr": 0.05}}})
for cfg in outcome.record.configurations:
    print(cfg["name"], cfg["status"], cfg["statistics"]["mean"], cfg["statistics"]["spread"])
```

Rules (D-029):

* A configuration may override `params.<name>`, `config_path`, `name` and `notes`. Seed,
  output directory, experiment id, data paths and the command are sweep-level.
* Each configuration × seed gets its own record in `<sweep>/<config>/seed-<seed>/`, tagged
  `config:<name>`, so every run is independently identifiable.
* **Configurations are never mixed.** With two or more configurations the top-level
  `statistics` section is `{"aggregated": false, …}` and publishes no mean: every mean and
  spread lives in the `configurations` entry that produced it.
* Status is reported per configuration **and** for the sweep (`success` / `partial` /
  `failed`); the sweep is `partial` if any configuration lost a run.
* Configuration order and seed order both do not matter: configurations are sorted by name
  and seeds by value before anything runs, and the same sweep repeated (or written with the
  lists reversed) gives the same statistics and the same fingerprint.
* **Backward compatibility:** the extra fields (`configurations`, the per-run
  `configuration` key, `configuration_names` / `runs_requested` on the sweep) appear **only**
  when named configurations are used. A seed-only sweep is byte-identical to a Stage 1A
  sweep — same layout, same sections, same fingerprint — and the schema version stays `1.0`.

Per-configuration section (`configurations[]`): `name`, `status`, `metric`, `seeds`,
`successful_seeds`, `failed_seeds`, the three counts, the resolved `spec`, the `overrides`
that were applied, and that configuration's own `statistics` (mean, spread, per-seed
values, min/max). The flat `runs` list still contains every run, each carrying its
`configuration`, `seed`, status, `metric_value`, `metric_source`, relative record path and
fingerprint.

### Metric resolution

`--metric` / `metric=` is looked up in the run record's `results` first (dotted paths such
as `eval.val_loss` work), and — for wrapped commands, whose `results` only contain
`exit_code` and log tails — in the **last JSON object the command printed**, which is the
convention `scripts/train.py --eval-only` and `scripts/evaluate.py` already use. Which
source was used is recorded per run as `metric_source`. A command with no `{seed}`
placeholder runs identically for every seed; the CLI warns about it.

## 9. Loss reporting: per token, per byte, per character

Per-token loss is a property of the **tokenization** as much as of the model: predicting one
character at a time is an easier prediction than predicting a whole word, so a char-level
model can post a lower per-token loss while modelling the same text *worse*. The fix is to
divide by the amount of text a token represents:

```
bits_per_token = nats_per_token / ln 2          # "val_loss" is nats per token
bits_per_byte  = bits_per_token x tokens_per_byte
bits_per_char  = bits_per_token x tokens_per_char
```

`tokens_per_byte` and `tokens_per_char` come from the corpus, not from a guess:
`prepare_data.py` measures the UTF-8 byte length and the Unicode character length of every
token's **surface piece** (`Tokenizer.tokenize()`) and `write_tokens()` sums them per split
into `n_bytes_train/val` and `n_chars_train/val` in `*.meta.json`. The pieces concatenate
back to the source text for both supported levels, so the sums are exact — two corpora
built from the same text have identical byte totals whatever the tokenization.

### Reading the numbers

| field | meaning | comparable across tokenizers? |
|---|---|---|
| `val_loss` | nats per token — the training objective | **no** |
| `val_ppl` / `bits_per_token` | token-level perplexity / bits | **no** |
| `bits_per_byte` | bits per UTF-8 byte of source text | yes |
| `bits_per_char` | bits per Unicode character | yes |

`scripts/evaluate.py` prints all of them plus the corpus counts it used; the trainer logs
`bits_per_byte` / `bits_per_char` on every `eval` event and `best_bpb` on `run.end`;
`scripts/train.py` prints `best_bpb` with its final line.

### Unknown is null, never zero

Corpora prepared before this bookkeeping existed have no counts. Every consumer then reports
`bits_per_byte: null` and `bits_per_char: null` (never `0`, never an estimate), the CLI adds
a note telling you to re-run `prepare_data.py`, and the trainer simply omits the fields. Old
`.meta.json` files still load: the four fields are optional.

Measured 2026-09-10 (EXP-006): the same synthetic corpus trained as `char` (vocab 51) and as
`word` (vocab 141) with identical hyperparameters gave `val_loss` 1.37519 vs 2.16143 — the
char model looks 36 % better per token — but **1.983986 vs 1.546772 bits per byte**, i.e. the
word model is 22 % better per byte. The ranking inverts; quote bits per byte.

---

## 10. What determinism is actually verified here

Measured on this branch (CPU, torch 2.14.0+cu130), two runs of the same spec, same seed,
`tiny_training_experiment(steps=10)` — a real Project 001 `Trainer` run:

```
steps=10  n_params=35552  tokens_seen=640  final_val_loss=3.24484
step 1 loss 3.9337 · step 10 loss 3.2253 · content fingerprint fec05faa136aca33… (equal)
```

and two runs of `reference_experiment` (synthetic corpus + CharTokenizer + python/numpy
draws) → content fingerprint `b053048d589e2a50…` (equal).

Sweeps are held to the same standard (measured 2026-09-10, EXP-004): a 5-seed sweep of a
real Project 001 training run gave `mean 3.2626182`, `spread 0.0476365` over
`final_val_loss`, and repeating it reproduced every per-run fingerprint, the statistics and
the sweep fingerprint (`d8c2565c964ceb59…`). A 3-seed CLI sweep reproduced identically
(`d4f498b9977a6f94…`) and produced the **same** fingerprint when its seeds were supplied in
reverse order. A partial sweep (seed 2 exiting non-zero) reported `partial`, kept the failed
seed's record, and averaged only the two usable runs (`n=2`).

The same sweep, re-run from a **fresh clone of the pushed branch on a clean tree**, gives
`mean 3.1767848`, `spread 0.0534606` and fingerprint `c603596da493d50a…` — identical for a
second run and for the reversed seed order `5,4,3,2,1` (EXP-004).

**Thread-count provenance (D-034, resolved Q-13).** The body owns torch's CPU thread count,
so the environment is captured *after* it runs and the caller's value is restored immediately
afterwards. A run with `train.num_threads=2` records `num_threads: 2` — the value it used —
and identical in-process runs produce identical provenance sections and fingerprints. The
`pinned_threads` fixture in `tests/conftest.py` remains, but its job is now *numerical*
determinism (multi-threaded reductions can change floating-point summation order), not
provenance.

`pytest tests/test_experiments.py` covers: spec defaults/explicit/round-trip/invalid
values/unknown keys/overrides, seeding stability and recorded limitations, git provenance
(commit compared against `git rev-parse HEAD` in a fixture repo — **no hard-coded SHA**;
dirty vs clean; detached HEAD; unavailable), hashing (determinism, content sensitivity,
argument-order independence, directory expansion, missing input, agreement with Project
002's `sha256_file`), record sections/round-trip/schema rejection/fingerprint semantics,
runner success/failure/input validation/lifecycle order, and the CLI (`--help`, success,
failure, provenance-only).

### Bug found and fixed while building this

Building the infrastructure exposed a **real determinism bug in Project 001**:
`TokenDataset.get_batch` used `int(generator.seed() % 2**32)` to build its NumPy RNG, but
`torch.Generator.seed()` is *not* a getter — it **re-seeds the generator with fresh OS
entropy**. Every batch was therefore sampled randomly and training was not reproducible
despite a fixed seed (two identical runs: val loss 3.367474 vs 3.20499). It now draws an
integer *from* the generator, and `Trainer.evaluate()` passes the trainer's seeded
generator to `get_batch`. The fix surfaced a second issue: `test_resume_continues_from_
checkpoint` had been passing by luck on noisy 4-batch evaluations; it was re-powered
(`max_steps` 25 → 60, `eval_iters=8`) after measuring that ≥30 extra steps reliably
improve on that fixture. Nothing about the model or the training algorithm changed.

---

## 11. Module map

| file | responsibility |
|---|---|
| `spec.py` | `ExperimentSpec` — validated, serializable experiment definition |
| `gitinfo.py` | `capture_git_info()` — commit/branch/dirty, unavailable states |
| `hashing.py` | `hash_paths()` / `digest_paths()`, re-exporting Project 002's SHA-256 |
| `seeding.py` | `seed_everything()`, `derive_seed()`, `SEED_LIMITATIONS` |
| `environment.py` | `capture_environment()`, `EXCLUDED_BY_POLICY` |
| `record.py` | `ExperimentRecord` — sections, fingerprint, save/load/render |
| `runner.py` | `run_experiment()`, `run_command()`, `ExperimentContext`, `FRONTIER_AI_EXPERIMENT_DIR` nesting signal |
| `autowire.py` | `run_self_recorded()` — lets a CLI record itself through the same lifecycle (§7.1) |
| `sweep.py` | `run_sweep()`, `run_command_sweep()`, `SweepRecord`, `SweepConfiguration`, mean ± spread |
| `examples.py` | reference experiments used by the docs and determinism tests |

Loss normalisation lives one package over, because Project 001's trainer and `evaluate.py`
own it: `src/frontier_ai/engine/metrics.py` — `bits_from_nats()`, `perplexity()`,
`bits_per_unit()`, `loss_summary()` (§9).

`examples.py` is intentionally **not** imported by the package `__init__` (it pulls in the
training stack); import it directly when you need it.

## 12. Not done (Stage 1 is infrastructure only)

* Larger sweep orchestration (many configurations, scheduling, resuming a half-finished
  sweep) — Stage 1/4 work (Q-8). Seed sweeps and multi-configuration sweeps are implemented
  (§8).
* An external experiment tracker / dashboard; records are plain JSON files on disk.
* ~~Real licensed corpora for smoke tests~~ **delivered 2026-09-10 (D-031)** as a licensed
  manifest + acquisition script (`corpora/smoke/`); the corpus files themselves are fetched,
  not committed, and their hashes are pinned only after a verified fetch.
* ~~Automatic experiment-record wiring for `scripts/train.py` and
  `scripts/tokenizer_*.py`~~ **delivered 2026-09-11 (D-032, D-033)** — see §7.1: the
  wrappers still work unchanged, and the scripts now record themselves when run directly,
  with the wrapper's record winning when both apply.
* Bits-per-byte was on this list; it is implemented (§9, EXP-006) for the char and word
  levels, and will need re-checking when a byte-level BPE enters the training path.
* Anything that changes the model, the tokenizer, or the research direction.
