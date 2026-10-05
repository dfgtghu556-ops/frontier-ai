# EXPERIMENTS.md

> The experiment log for the frontier-ai foundation-model track.
> Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) first.
>
> **Rule zero: never write a number you did not observe.** If a metric was not produced by
> a run you can point to (a `train.jsonl`, a test, or a report file), write
> `not measured`. A log with invented numbers is worse than an empty log, because it will
> be trusted.

---

## 1. Rules for recording experiments

1. **Every run gets an entry** — including failures and "nothing changed" results. Negative
   results are assets; unrecorded results are lost.
2. **One experiment = one question.** If you are testing two things, log two experiments.
   Change one variable at a time relative to a named baseline.
3. **Reproducibility is part of the result.** An entry must contain enough to re-run it:
   config (or diff), data version/hash, seed, hardware, and code revision (git SHA).
4. **Metrics before conclusions.** Record the numbers first; interpret them afterwards, and
   mark interpretation clearly as interpretation.
5. **Name the artifacts.** Point at the run directory and the metrics file.
6. **End with a next action.** An experiment that does not change what you do next was not
   worth running.
7. **Update the status** if a later result changes the reading of an earlier one; never edit
   an old entry's numbers, add a follow-up note.

---

## 2. Standard entry format

Copy this block for each new experiment:

```markdown
### EXP-00N — <short title>

- **Status:** planned | running | complete | abandoned
- **Date:** YYYY-MM-DD
- **Objective:** the one question this experiment answers
- **Hypothesis:** what you expect to happen, and *why* (state it before running)
- **Baseline:** EXP-00X (or "none")

**Configuration**
- Model: n_layer=·, n_head=·, n_kv_head=·, n_embd=·, block_size=·, ffn=·, norm=·, pos=·, tie=·
- Parameters: <count>
- Tokenizer: char | word | bpe-<version> (vocab=<n>)
- Dataset: <name + version/hash>
- Data version: <dataset id>, n_train=·, n_val=·, split method=·
- Seed: <int>   ·   Deterministic: yes/no
- Hardware: <CPU/GPU model × count, RAM, torch version, precision>
- Training: max_steps=·, batch_size=·, block_size=·, accum_steps=·, tokens/step=·,
  optimizer=·, lr=·, schedule=·, warmup=·, weight_decay=·, grad_clip=·, precision=·
- Command: `<exact command>`
- Code revision: <git SHA>   ·   Config file: <path>

**Metrics**
| metric | value | notes |
|---|---|---|
| final val loss | | |
| best val loss | | |
| val perplexity | | |
| bits/token (or bits/byte) | | |
| tokens trained | | |
| wall time | | |
| throughput | | |
| peak memory | | |

**Results:** <what the numbers show>
**Observations:** <anything not captured by metrics — instabilities, surprises, artifacts>
**Conclusion:** <was the hypothesis right? one paragraph>
**Next action:** <the concrete next step this implies>
**Artifacts:** <run directory, train.jsonl, checkpoint path>
```

Additional fields to add when relevant: prompt/format for eval, decoding settings for
generation quality, per-language breakdowns, contamination check outcome, and cost
(GPU-hours or ₹/$).

---

## 3. Extra fields for tokenizer experiments (Project 002+)

In addition to the standard template, tokenizer experiments must record:

- **Tokenizer implementation** and **library version** (e.g. `bpe_python` /
  `frontier-bpe-python-1`, `bpe_hf` / `tokenizers-0.23.2`) — captured automatically in the
  artifact `manifest.json`.
- **Corpus identifier and version** plus the fixture's sha256 (the corpus loader refuses to
  run if the hash no longer matches, so the id is trustworthy).
- **Vocabulary size**, **special tokens**, **training parameters** (`vocab_size`,
  `min_frequency`, `min_count`), **seed**, and the **artifact directory**.
- **Per-language and per-category metrics**, not just overall: a single aggregate number
  hides language-specific regressions, which is the whole point for Indian languages.
- **Round-trip failures** and **unknown rate** — a tokenizer that cannot reconstruct its
  input is disqualified from "best" (D-022) no matter how good its compression is.

Reproduce a tokenizer experiment with:

```bash
python scripts/tokenizer_prepare_corpus.py --out data/tokenizer/indic-v1 --seed 1337
python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 --impl <impl> \
    --vocab-size <n> --out artifacts/tokenizers/<name> --exp-id EXP-00N
python scripts/tokenizer_compare.py --corpus data/tokenizer/indic-v1 \
    --tokenizer <artifact> [<artifact> ...] --out out/tokenizer/compare.json
```

---

## 4. Experiment log

### EXP-001 — CPU smoke test: end-to-end pipeline verification ✅

- **Status:** complete
- **Date:** 2026-09-10
- **Objective:** verify that the full pipeline (prepare → train → evaluate → checkpoint →
  resume → sample) works, and that a randomly-initialized model actually learns on a
  structured corpus.
- **Hypothesis:** validation loss will fall from ≈ ln(51) = 3.93 (chance) to well below 2.0
  within 200 steps, and resume will restore the step counter and continue the LR schedule.
- **Baseline:** none (first run in the repository)

**Configuration**
- Model: n_layer=2, n_head=2, n_kv_head=2, n_embd=64, block_size=64, ffn=swiglu,
  norm=rmsnorm, pos=learned, tie_embeddings=true, dropout=0.0
- Parameters: 138,752 (MLP 70.8%, attention 23.6%, embeddings 5.3%, norms 0.2%)
- Tokenizer: char (vocab=51, includes `\ufffd` replacement token)
- Dataset: synthetic pseudo-English corpus (`data/synthetic.py`, deterministic)
- Data version: `data/synthetic.bin` — n_tokens=200,094, n_train=180,085, n_val=20,009,
  split = **last 10% (tail split)**, dtype uint8
- Seed: 1337 (data and training) · Deterministic: not asserted for this run
- Hardware: 2 CPU cores, 3 GB RAM, Python 3.11.2, torch 2.14.0+cu130,
  `cuda=False`, precision fp32 (amp off)
- Training: max_steps=200, batch_size=8, block_size=64, accum_steps=4 → 2,048 tokens/step;
  AdamW, lr=3e-3, cosine to 10%, warmup=20 steps, weight_decay=0.1, grad_clip=1.0
- Command: `python scripts/train.py --config configs/cpu_smoke.json`
- Code revision: `e467eb0` · Config file: `configs/cpu_smoke.json`

**Metrics** (from `out/cpu-smoke/train.jsonl`)

| metric | value | notes |
|---|---|---|
| train loss, step 1 | 3.9837 | ≈ ln(51) = 3.93, i.e. untrained/chance |
| val loss @ step 50 | 2.32644 (ppl 10.241) | |
| val loss @ step 100 | 1.78034 (ppl 5.932) | |
| val loss @ step 150 | 1.47193 (ppl 4.358) | |
| val loss @ step 200 | 1.37759 (ppl 3.965) | final and best |
| best val loss | 1.37759 | |
| tokens trained | 409,600 | ≈ 2.3 epochs over the training split |
| wall time | 7.7 s (0.13 min) | |
| throughput | 53,198 tok/s (mean) | |
| peak memory | not measured | run is tiny; no profiler was used |

**Results:** validation loss fell monotonically at every eval point, 3.93 → 1.38
(perplexity 51 → 3.97). Gradient norms stayed in 0.5–1.6. Resume was separately verified:
loading `last/` at step 200 and continuing to step 230 restored the step counter and picked
the cosine LR up where it left off (≈3.0e-4), and a test asserts that a reloaded model
reproduces the trainer's loss to 1e-5.

**Observations:**
- Validation loss tracks training loss closely (final train ≈1.27 vs val 1.38). Expected:
  the synthetic corpus is highly repetitive, so the tail split is nearly the same
  distribution as the training split. **This measures fitting, not generalization.**
- Sampled text is word-shaped gibberish: correct spacing and "the _ the _" rhythm, some
  learned counting patterns, no real words. Consistent with 8 seconds of training on 200 KB.
- The tiny config is dominated by the MLP (70.8% of parameters), which is a consequence of
  `ffn_mult=4` plus the 64-multiple rounding for SwiGLU at `n_embd=64` — not a design
  statement about larger models.

**Conclusion:** hypothesis confirmed for the purpose it was meant to serve. The pipeline is
correct end to end, learning occurs, and checkpoint/resume is faithful. This is a
**plumbing verification, not a modeling result**: the corpus is synthetic, the vocabulary is
51 characters, and no GPU was involved.

**Next action:** Project 002 — not yet defined; confirm scope with the user. The likely
first step is Stage 1 from [ROADMAP.md](ROADMAP.md): make runs reproducible and comparable
(determinism, config+git+data-hash capture, per-byte normalization so tokenizer changes can
be compared), then a real tokenizer and data pipeline.

**Artifacts:** `out/cpu-smoke/train.jsonl`, `out/cpu-smoke/{best,last}/`,
`out/cpu-smoke/config.json` (regenerating this run overwrites them; timings are
machine-dependent). A later re-run on the same config (fresh environment, 200k-char
corpus, 200 steps) produced `best_val=1.3230` instead of `1.3776`; the corpus and seed are
fixed but thread scheduling is not, so small run-to-run differences like this are expected
and are why headline claims here are quoted from a specific recorded run.

---

---

### EXP-002 — Tokenizer baselines on the Indian-language probe fixture ✅

- **Status:** complete
- **Date:** 2026-09-10
- **Objective:** stand up the tokenizer research framework and take first measurements:
  compare character, word and byte-level BPE tokenizers on the same deterministic probe
  fixture, and check whether the framework can actually distinguish them.
- **Hypothesis:** (a) byte-level BPE will be lossless while char/word tokenizers will not
  (their tiny training text cannot cover the probe); (b) byte-level BPE will compress
  English better than Indic scripts at 512–1024 vocabulary, because Indic characters cost
  3 UTF-8 bytes each; (c) at equal vocabulary, the two BPE implementations will differ
  mostly because of pre-tokenization, not the merge algorithm.
- **Baseline:** none (first tokenizer experiment); char/word are Project 001 tokenizers
  used as reference points.

**Configuration**
- Corpus: `data/tokenizer/indic-v1` — `indic-eval-v1`, seed 1337, 73,033 training chars,
  179 evaluation examples (14 languages + 9 orthography categories; 4,108 code points,
  8,141 UTF-8 bytes, 680 whitespace words); sha256-verified on load
- Implementations and versions: `bpe_python` = `frontier-bpe-python-1`,
  `bpe_hf` = `tokenizers-0.23.2` (Apache-2.0), `char` = `project001-char-1`,
  `word` = `project001-word-1`
- Special tokens for every tokenizer: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- Training parameters: `bpe_python` vocab 512 and 1024; `bpe_hf` vocab 1024,
  `min_frequency=2`; `char`/`word` vocabularies determined by the corpus (365 / 1,085)
- Deterministic: yes (corpus seed 1337; BPE merge selection is frequency-then-pair ordered)
- Hardware: 2 CPU cores, 3 GB RAM, Python 3.11.2, torch 2.14.0+cu130, tokenizers 0.23.2
- Command: `python scripts/tokenizer_compare.py --corpus data/tokenizer/indic-v1
  --tokenizer artifacts/tokenizers/{char,word,bpe_py_512,bpe_py_1024,bpe_hf_1024}
  --out out/tokenizer/compare.json --exp-id EXP-002`
- Code revision: Project 002 branch commit; config: `data/tokenizer/indic-v1/manifest.json`

**Metrics** (179 examples, identical for every row)

| Tokenizer | Impl | Vocab | Tokens | chars/token | bytes/token | tokens/char | tokens/word | unk rate | round-trip failures |
|---|---|---|---|---|---|---|---|---|---|
| char | char | 365 | 4,108 | 1.000 | 1.982 | 1.000 | 6.041 | 9.74% | 128 |
| word | word | 1,085 | 3,111 | 1.320 | 2.617 | 0.757 | 4.575 | 18.61% | 160 |
| bpe_py_512 | bpe_python | 512 | 4,835 | 0.850 | 1.684 | 1.177 | 7.110 | n/a | 0 |
| bpe_py_1024 | bpe_python | 1,024 | 4,204 | 0.977 | 1.936 | 1.023 | 6.182 | n/a | 0 |
| bpe_hf_1024 | bpe_hf | 1,024 | 3,835 | 1.071 | 2.123 | 0.934 | 5.640 | n/a | 0 |

Per-language chars/token (higher = more compression), selected languages:

| Tokenizer | en | hi | hi-en | best language | worst language |
|---|---|---|---|---|---|
| bpe_py_512 | 1.390 | 0.777 | 1.214 | en 1.390 | or 0.636 |
| bpe_py_1024 | 1.507 | 0.893 | 1.376 | en 1.507 | pa 0.714 |
| bpe_hf_1024 | 1.825 | 0.967 | 1.690 | en 1.825 | pa 0.800 |

**Results:**
1. Both byte-level BPEs round-trip all 179 examples with zero unknown tokens. The char
   tokenizer altered 128/179 examples (9.74% of its tokens were the unknown-character
   token) and the word tokenizer altered 160/179 (18.61% unknown rate).
2. Byte-level BPE beats one-token-per-character on English (1.39–1.83 chars/token) but
   **not** on Indic scripts at these vocabulary sizes (0.64–0.97 chars/token vs 1.000 for
   the character baseline).
3. At equal vocabulary (1,024), `bpe_hf` produced 3,835 tokens vs 4,204 for `bpe_python`
   (8.8% fewer) on identical input.
4. Both char and word tokenizers are disqualified from "best" by the losslessness gate
   (D-022); among lossless tokenizers `bpe_hf_1024` leads on every compression metric.

**Observations:**
- Findings 1 and 2 confirm hypotheses (a) and (b). Finding 3 is consistent with (c): the
  merge rules are equivalent, and the difference is pre-tokenization — HuggingFace's
  ByteLevel pattern attaches a leading space to words so merges can absorb it, while our
  implementation emits whitespace runs as separate tokens. This is now open ablation
  **Q-9**, not a conclusion.
- The per-language spread is large: English reaches 1.39–1.83 chars/token while Odia and
  Punjabi sit at 0.71–0.86 for the same tokenizer — a >2× cost difference between
  languages, which is exactly the inequity this framework exists to quantify.
- `bpe_python`'s mark-aware pre-tokenizer keeps "मैं" whole, but that did not translate
  into fewer tokens here: Indic characters still cost 3 bytes each and the merges needed
  to reassemble them compete with word-level merges for the same vocabulary budget.
- The fixture's training text shares small word lists with its evaluation examples, so all
  compression numbers are optimistic. They are valid only as a like-for-like comparison on
  this fixture.

**Conclusion:** hypothesis confirmed on all three points. The framework works, is
deterministic, and produces discriminating measurements. **It does not select a
production tokenizer** — see D-021. The main substantive signal is that Indian-language
scripts need a substantially larger vocabulary budget (or a script-aware initialization)
before subword BPE pays off, which must be re-measured on real data.

**Next action:** do **not** pick a tokenizer yet. Next experiments: (1) vocabulary sweep
(2k / 4k / 8k / 16k / 32k) on the same fixture to find where Indic compression crosses
1.0 chars/token; (2) pre-tokenization ablation Q-9 (mark-aware vs GPT-2 pattern,
whitespace attachment); (3) repeat on real licensed Indic data once Stage 3 delivers it;
(4) eventually train the same small model with two tokenizers and compare quality.

**Artifacts:** `out/tokenizer/compare.json` (machine-readable comparison),
`out/tokenizer/compare.txt` (rendered table),
`out/tokenizer/eval/{bpe_hf_1024,bpe_py_1024}.json` (per-tokenizer reports),
`artifacts/tokenizers/*/manifest.json` (provenance). Artifacts are regenerable and are not
committed; regenerate with the commands in §3.


### EXP-003 — Experiment infrastructure: does the provenance system actually reproduce runs? ✅

> **This entry validates infrastructure, not a model.** There is no quality benchmark here,
> no tokenizer verdict and no scaling claim. It answers one question: *if we record a run,
> does the record let us reproduce it — and does it refuse to lie when it cannot?*

- **Status:** complete (infrastructure verification)
- **Date:** 2026-09-10
- **Objective:** verify that the Project 003 experiment infrastructure (ROADMAP Stage 1)
  captures code, data, configuration, seed, environment and command faithfully, that two
  identical runs produce identical results and identical record fingerprints, and that
  failures are recorded rather than swallowed.
- **Hypothesis:** (a) with a fixed seed and fixed inputs, two runs of the *same* spec
  produce bit-identical results and equal content fingerprints; (b) changing inputs or
  seed changes the fingerprint; (c) a failing command is recorded as `failed` **and**
  re-raised; (d) Project 001 and Project 002 results are unchanged by the new code, except
  where the new code fixes a genuine determinism bug.
- **Baseline:** EXP-001 (Project 001 pipeline) and EXP-002 (tokenizer baselines) as
  recorded above — their numbers must still reproduce.

**Configuration**

- Hardware: 2 CPU cores, 3 GB RAM, Python 3.11.2, torch 2.14.0+cu130, numpy (CPU only,
  `train.num_threads=1` for the determinism runs)
- Seed: 1337 (master); derived component seeds `data=1270167219`, `model=3236282298`,
  `sampling=1345220995` (`sha256("<master>|<component>") mod 2**32`)
- Deterministic: yes, under the recorded limitations — cuDNN, thread count, library
  versions and RNGs outside python/numpy/torch are **not** covered (D-025)
- Code revision: dirty working tree on `arena/01a08a78-frontier-ai` (commit `10e7aa9c`);
  the record flagged `reproducible_from_commit=false`, as designed (D-026)
- Commands (exact):

```bash
# 1. in-process: a reference experiment and a real Project 001 Trainer run, twice each
python3 - <<'SCRIPT'
from frontier_ai.experiments import ExperimentSpec, run_experiment
from frontier_ai.experiments.examples import reference_experiment, tiny_training_experiment
for i in range(2):
    spec = ExperimentSpec(experiment_id="EXP-003", seed=1337, name="reference",
                          output_dir=f"/tmp/expdemo/ref{i}", component="reference")
    print(run_experiment(spec, reference_experiment).record.content_fingerprint())
    spec = ExperimentSpec(experiment_id="EXP-003", seed=1337, name="tiny-train",
                          output_dir=f"/tmp/expdemo/tr{i}", component="training")
    out = run_experiment(spec, lambda ctx: tiny_training_experiment(ctx, steps=10))
    print(out.record.content_fingerprint(), out.record.results)
SCRIPT

# 2. the real Project 001 pipeline wrapped by the CLI, run twice into the same directory
python3 scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic
python3 scripts/experiment_record.py --exp-id EXP-003 --seed 1337 --name "identical runs" \
    --out out/experiments/EXP-003-same --config configs/cpu_smoke.json \
    --data data/synthetic.bin --tag project-001 -- \
    python3 scripts/train.py --config configs/cpu_smoke.json --max-steps 30 \
        --set train.num_threads=1
```

**Metrics**

| check | expected | observed | verdict |
|---|---|---|---|
| `reference_experiment` × 2 | equal fingerprints | `b053048d589e2a50…` both | pass |
| Project 001 `Trainer` run × 2 (10 steps) | equal fingerprints, equal results | `fec05faa136aca33…` both; `final_val_loss 3.24484`, step-1 loss 3.9337, step-10 loss 3.2253 | pass |
| CLI-wrapped `scripts/train.py` × 2 (30 steps) | equal fingerprints | `f54881c68513f2957600188b5912d89684a02a9a65149e3e414ed8aa810f1dbc` both; `best_val=2.9223` | pass |
| same run, different output dir | identical metrics, **different** fingerprint | loss 3.9706 / 3.1482 / 2.6112, `best_val=2.4691` in both; fingerprints differ because the command differs | pass (by design) |
| data change detected | digest changes | two-file digest `8b15610be1cc…`; editing a file changes the digest | pass |
| failing command | record `status="failed"`, error + traceback, exit code propagated, exception re-raised | observed on a deliberately invalid `scripts/train.py --out …` invocation | pass |
| missing declared input | fails before compute | `ExperimentInputError` naming the path | pass |
| dirty tree | flagged, not hidden | `dirty=true`, `reproducible_from_commit=false`, warning printed | pass |
| EXP-002 regression (`bpe_py_1024`) | reproduces recorded numbers | chars/token 0.977165 (recorded 0.977), tokens/word 6.182353 (6.182), en 1.507, hi 0.893, pa 0.714, 0 round-trip failures — all match | pass |
| Project 001 + 002 test suites | green | `pytest -q` → 120 passed; `ruff check .` clean | pass |

**Results:** The infrastructure does what it claims: identical specs give identical results
and identical fingerprints; any change to code, data, seed, config or command changes the
fingerprint; failures are visible instead of silent.

**Observations:**

- **A real determinism bug in Project 001 was found by this work.** `TokenDataset.get_batch`
  called `torch.Generator.seed()` to obtain a seed for its NumPy RNG, but that method
  *re-seeds* the generator from OS entropy — it is not a getter. Batch sampling was
  therefore random regardless of the configured seed (two identical runs gave
  `final_val_loss` 3.367474 vs 3.20499). It now draws an integer *from* the generator, and
  `Trainer.evaluate()` passes the trainer's seeded generator. No model or algorithm change.
- With that fixed, one pre-existing test failed: `test_resume_continues_from_checkpoint`
  had been passing on lucky 4-batch evaluations (3.17683 against a 3.14035 threshold).
  +15 steps cannot reliably beat a noisy estimate on that fixture; it was re-powered to
  `max_steps=60, eval_iters=8` after measuring that ≥30 extra steps reliably improve
  (3.1523 → 2.9035 at +50). Assertions were strengthened, not loosened.
- The two CLI runs that differ **only** in output directory produce different fingerprints.
  That is intended — the command is part of an experiment's identity — but it means
  "same fingerprint" is a statement about *identical* runs, not merely equivalent ones.
- All verification ran on a dirty tree, so these records honestly say
  `reproducible_from_commit=false`. Nothing here is claimed as reproducible from a commit.

**Conclusion:** Hypothesis confirmed on all four points. Reproducibility is now a property
we can *test*, and every record states its own limits (D-025, D-026). This makes later
stages' numbers comparable; it does not by itself make any model or tokenizer better.

**Next action:** finish ROADMAP Stage 1 — multi-seed sweeps with mean ± spread and an
unattended 2-config sweep (Q-8), plus bits-per-byte loss reporting so char-level and BPE
models can be compared fairly. Do not start Project 004 on the basis of this entry.

**Artifacts:** `out/experiments/EXP-003-same/experiment.json` (+ `experiment.txt`),
`out/experiments/EXP-003-train{1,2}/experiment.json`, `out/tokenizer/verify.json`,
`tests/test_experiments.py`, [docs/experiments.md](docs/experiments.md). Run directories
are git-ignored; the records are regenerable with the commands above.

### EXP-004 — Multi-seed sweeps: does the aggregate reproduce, and does it report spread honestly? ✅

> **This entry validates infrastructure, not a model.** The loss numbers below are from an
> 8–12 step run on a 35,552-parameter model over the synthetic smoke corpus. They are a
> *test signal* for the sweep machinery — they say nothing about model quality, no
> tokenizer or architecture decision is involved, and no scientific conclusion about
> training is drawn from them.

- **Status:** complete (infrastructure verification)
- **Date:** 2026-09-10
- **Objective:** verify the Stage 1A multi-seed sweep: that one specification executed
  across several seeds keeps a full record per seed, aggregates a metric as
  **mean ± spread**, is independent of seed order, reproduces when repeated, and reports
  failures without hiding them.
- **Hypothesis:** (a) identical sweeps reproduce exactly (per-run fingerprints, statistics
  and sweep fingerprint); (b) the aggregate does not depend on the order seeds are supplied
  in; (c) a failed seed keeps its failed record, is listed separately, and makes the sweep
  `partial` rather than `success`; (d) with zero usable runs no mean or spread is reported;
  (e) spread is the *sample* standard deviation and is `null` (not 0) when undefined.
- **Baseline:** EXP-003 (single-run provenance). EXP-001/EXP-002 numbers must be unaffected.

**Configuration**

- Hardware: 2 CPU cores, 3 GB RAM, Python 3.11.2, torch 2.14.0+cu130, numpy (CPU only,
  `train.num_threads=1`)
- Sweep: experiment id `EXP-004`, metric `final_val_loss`, seeds `1,2,3,4,5` (Python API)
  and `1,2,3` (CLI); statistics computed with the standard library `statistics` module
- Work per seed: a real Project 001 `Trainer` run (2 layers, 2 heads, `n_embd=32`,
  `block_size=32`, 8–12 steps, fp32, `num_threads=1`) on `data/synthetic.bin`
- Code revision: dirty working tree on `arena/01a08a78-frontier-ai` (commit `0e5521d`);
  records therefore show `reproducible_from_commit=false`, as designed (D-026)
- Commands (exact):

```bash
# 1. Python API: 5-seed sweep of a real training run, executed twice
python3 - <<'SCRIPT'   # (helper: /tmp/verify_sweep.py, not committed)
from frontier_ai.experiments import ExperimentSpec, run_sweep
from frontier_ai.experiments.examples import tiny_training_experiment
spec = ExperimentSpec(experiment_id="EXP-004", seed=1, name="5-seed training sweep",
                      output_dir="/tmp/exp004/sweep-a", data_paths=["data/synthetic.bin"],
                      config_path="configs/cpu_smoke.json")
run_sweep(spec, [1, 2, 3, 4, 5],
          lambda ctx: dict(tiny_training_experiment(ctx, steps=8)), "final_val_loss")
SCRIPT

# 2. CLI: 3-seed sweep of a per-seed training command, twice, and once in reverse order
python scripts/experiment_sweep.py --exp-id EXP-004 --seeds 1,2,3 \
    --metric final_val_loss --name "cli training sweep" --out /tmp/exp004/cli-a \
    --data data/synthetic.bin --tag project-001 --quiet -- \
    python3 /tmp/seed_train.py "{seed}"

# 3. CLI: partial failure (seed 2 exits 3)
python scripts/experiment_sweep.py --exp-id EXP-004 --seeds 1,2,3 \
    --metric final_val_loss --out /tmp/exp004/partial2 --data data/synthetic.bin --quiet -- \
    python3 -c "import sys, subprocess; seed=int(sys.argv[1]); sys.exit(3) if seed==2 else \
        subprocess.run([sys.executable, '/tmp/seed_train.py', str(seed)], check=True)" "{seed}"
```

`/tmp/seed_train.py` is a local (uncommitted) helper that trains the same tiny model with the
seed given on its command line and prints `{"final_val_loss": ...}` — it exists only to give
the CLI something real to sweep.

**Metrics**

| check | expected | observed | verdict |
|---|---|---|---|
| 5-seed training sweep (Python API) | 5 usable runs, mean ± spread reported | mean **3.2626182**, spread **0.0476365** (sample stdev), range 3.213519 … 3.328338 | pass |
| same sweep repeated | identical per-run fingerprints, statistics and sweep fingerprint | per-run fingerprints equal, statistics equal, sweep fingerprint `d8c2565c964ceb59…` both | pass |
| CLI sweep × 2 | identical aggregate | mean 3.1639953, spread 0.0653497, fingerprint `d4f498b9977a6f94…` both | pass |
| CLI sweep, seeds `3,2,1` vs `1,2,3` | identical aggregate | same mean, spread **and** sweep fingerprint `d4f498b9977a6f94…` | pass |
| per-seed values | each run records the seed it used | seed 1: 3.262067 · 2: 3.287883 · 3: 3.221284 · 4: 3.328338 · 5: 3.213519 | pass |
| individual records | one full record per seed | `seed-0000000001…0000000005/experiment.json`, each with git/data/environment/configuration/randomness and its own fingerprint | pass |
| partial failure (seed 2 exits 3) | `partial`, failed record kept, others continue | status `partial`, successful `[1, 3]`, failed `[2]`, `n=2`, mean 3.1636875, spread 0.0924153, CLI exit code **2**; `seed-0000000002/experiment.json` exists with `status=failed` and the `CalledProcessError` | pass |
| all seeds failing | no mean, no spread | status `failed`, `mean=null`, `spread=null`, note says "null, not zero"; CLI exit code **1** | pass (test suite + CLI run) |
| missing / non-numeric metric | never substituted with zero | run marked `metric_missing`, `metric_value=null`, counted as failed; sweep becomes `partial` | pass |
| one-seed sweep | mean = that value, spread undefined | `spread=null` with an explanatory note | pass |
| spread definition | `n − 1` denominator | equals `statistics.stdev`, differs from `statistics.pstdev`; definition text states it is not a confidence interval | pass |
| Project 001 / 002 regression | unchanged | `pytest -q` → **145 passed**; `ruff check .` clean; EXP-002 metrics re-verified unchanged | pass |
| fresh clone, clean tree, seeds `1,2,3,4,5` (CLI) | identical mean, spread and sweep fingerprint | mean 3.1767848, spread 0.0534606, fingerprint `c603596da493d50a…`, `reproducible_from_commit=true`, and identical for a second run and for `--seeds 5,4,3,2,1` | pass |

**Results:** The sweep machinery behaves as specified. Seed-to-seed variation in
`final_val_loss` on this 8-step toy run is small but real (spread ≈ 0.048 around a mean of
3.263, i.e. ~1.5%), which is exactly the kind of number a single-seed run cannot show.

**Fresh-clone re-verification (mandatory, and it caught a real thing):** the same CLI sweep
was repeated in a pristine clone of the pushed branch at `5cff3ee` with an independently
built environment (Python 3.11.2, torch 2.14.0+cu130). On a **clean** tree it gives
`mean 3.1767848`, `spread 0.0534606` over seeds 1–5 (3.229035 / 3.164611 / 3.098340 /
3.167123 / 3.224815), `reproducible_from_commit=true`, and the **same** sweep fingerprint
`c603596da493d50a…` for a second run and for `--seeds 5,4,3,2,1`. The first attempt at this
check reported a *different* fingerprint between runs for an innocent reason worth knowing:
the verification helper wrote its scratch files inside the clone, so the second run recorded
a dirty tree while the first had recorded a clean one. The statistics were identical in all
three runs — only the `dirty_files` list differed — which is exactly what D-026 is for.
Scratch output was moved outside the clone and the three runs then agreed exactly.

**Observations:**

- The original (local) sweeps ran on a **dirty** working tree, so their records honestly
  state `reproducible_from_commit=false`. The fresh-clone runs above are the clean-tree
  evidence; the numbers below the table are evidence about the *machinery*, not a claim
  that the dirty-tree runs can be reproduced from a commit.
- Order-independence required normalising the recorded template spec: the sweep embeds the
  base spec with the **first canonical** (smallest) seed. Before that, `[1,2,3]` and
  `[3,1,2]` differed in `configuration.spec.seed` and produced different fingerprints — the
  same class of bug as the Project 001 generator bug: metadata that varies without changing
  the experiment.
- Command-mode runs only record `exit_code` and log tails, so `--metric` additionally reads
  the last JSON object the command printed (the convention `scripts/train.py --eval-only`
  and `scripts/evaluate.py` already use); the source used is recorded per run as
  `metric_source` (`stdout_json` in these sweeps).
- Spread from `n = 2` or `n = 3` seeds is a rough dispersion estimate. The implementation
  deliberately does **not** convert it into a confidence interval, which would assume
  normality we have not tested.

**Conclusion:** Hypothesis confirmed on all five points. Multi-seed sweeps now make
seed-sensitivity measurable and reproducible, and the aggregate cannot silently hide a lost
seed. This is Stage 1 item 1 of 4; items 2–4 (2-config sweeps, bits-per-byte reporting, real
licensed corpora) are untouched.

**Next action:** use a sweep for any number that will be quoted as a headline result
(`--seeds 1,2,3,4,5`), and treat `spread` as a dispersion estimate, not an error bar. Then
continue Stage 1 with item 2 (unattended 2-config sweeps, **Q-8**). Do not start Project 004
on the basis of this entry.

**Artifacts:** `/tmp/exp004/sweep-a/sweep.json` (+ `sweep.txt` and one
`seed-*/experiment.json` per seed), `/tmp/exp004/cli-a/sweep.json`,
`/tmp/exp004/partial2/sweep.json`, the fresh-clone `out/sweeps/EXP-004-{a,b,rev}/sweep.json`,
`tests/test_sweeps.py`,
[docs/experiments.md §8](docs/experiments.md). Run directories are git-ignored; the sweeps
are regenerable with the commands above.

### EXP-005 — Multi-configuration sweeps: do two configurations separate, and does the sweep reproduce? ✅

> **This entry validates infrastructure, not a model.** The loss numbers come from 12-step
> runs of a ~35.6k-parameter model on the synthetic smoke corpus. They are a *test signal*
> for the sweep machinery: they are not a learning-rate recommendation, not a model-quality
> measurement, and no scientific conclusion about training is drawn from them.

- **Status:** complete (infrastructure verification)
- **Date:** 2026-09-10
- **Objective:** verify the Stage 1B extension: several named configurations, each run
  across the same seeds, keeping one full record per configuration × seed, aggregating each
  configuration separately, and refusing to mix configurations — with order independence,
  reproducibility, and the existing explicit failure semantics intact.
- **Hypothesis:** (a) two configurations that differ only in a parameter produce different,
  separately reported means; (b) the sweep reproduces exactly when repeated and when the
  configuration and seed lists are given in reverse order; (c) a configuration that fails
  for every seed leaves the sweep `partial`, keeps its failed records, and does not disturb
  the surviving configuration's aggregate; (d) a seed-only sweep is byte-compatible with
  Stage 1A (no new fields, EXP-004 fingerprints still reproduce).
- **Baseline:** EXP-004 (single-configuration seed sweep). EXP-001/EXP-002 numbers must be
  unaffected.

**Configuration**

- Sweep: `EXP-005`, metric `final_val_loss`, seeds `1,2,3`, configurations
  `lr_low` (`params.lr=0.005`) and `lr_high` (`params.lr=0.05`)
- Work per configuration × seed: a real Project 001 `Trainer` run — 2 layers, 2 heads,
  `n_embd=32`, `block_size=32`, 12 steps, batch 4, fp32, `num_threads=1` — on
  `data/synthetic.bin` (200,000 chars, seed 1337), driven through
  `scripts/experiment_sweep.py` with `{config}` and `{seed}` substituted into the command
- Hardware: 2 CPU cores, 3 GB RAM, Python 3.11.2, torch 2.14.0+cu130
- Code revision: dirty working tree on `arena/01a08a78-frontier-ai` (commit `c2c88cb`);
  records therefore show `reproducible_from_commit=false`, as designed (D-026)
- Commands (exact):

```bash
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic

# 2 configurations x 3 seeds, run three times (identical, and once with both lists reversed)
python scripts/experiment_sweep.py --exp-id EXP-005 --seeds 1,2,3 \
    --configs "lr_low:params.lr=0.005" "lr_high:params.lr=0.05" \
    --metric final_val_loss --name "lr two-config sweep" --out out/sweeps/EXP-005-a \
    --data data/synthetic.bin --tag project-001 -- \
    python3 /tmp/mc_train.py "{config}" "{seed}"

# partial failure: the lr_high command exits 4 for every seed
python scripts/experiment_sweep.py --exp-id EXP-005 --seeds 1,2,3 \
    --configs "lr_low:params.lr=0.005" "lr_high:params.lr=0.05" \
    --metric final_val_loss --out out/sweeps/EXP-005-partial --data data/synthetic.bin --quiet -- \
    python3 -c "import sys,subprocess; cfg,seed=sys.argv[1],sys.argv[2]; \
sys.exit(4) if cfg=='lr_high' else \
subprocess.run([sys.executable,'/tmp/mc_train.py',cfg,seed],check=True)" "{config}" "{seed}"
```

`/tmp/mc_train.py` is a local (uncommitted) helper that trains the tiny model with the
learning rate selected by the configuration name and prints `{"final_val_loss": …}`.

**Measured result (the real 2-configuration sweep)**

| configuration | seeds | final_val_loss per seed | mean | spread (sample stdev) | status |
|---|---|---|---|---|---|
| `lr_high` (lr 0.05) | 1, 2, 3 | 3.206224 · 3.190828 · 3.095739 | **3.164264** | **0.059841** | success |
| `lr_low` (lr 0.005) | 1, 2, 3 | 3.433670 · 3.365153 · 3.296912 | **3.365245** | **0.068379** | success |

Sweep status `success`, 6/6 runs usable, no cross-configuration mean published
(`statistics.aggregated = false`).

**Metrics**

| check | expected | observed | verdict |
|---|---|---|---|
| two configurations separate | different per-configuration means | 3.164264 vs 3.365245; the spread of each (≈0.06) is smaller than the gap (≈0.20) | pass |
| repeated sweep | identical fingerprints | `f7306163a37b7a83cecfe23b410d76c16eb1fdc4193764f5ea11f84d3bfe81e3` for run a, run b and the reversed run | pass |
| fresh clone, clean tree, commits `166c858`+`d15a25d` (CLI) | identical per-configuration means, spreads and sweep fingerprint | `lr_high` 3.164264 ± 0.059841, `lr_low` 3.365245 ± 0.068379, fingerprint `b397ba46addd7e7d…` — identical for a second run and for `--configs lr_high lr_low` + `--seeds 3,1,2`; `reproducible_from_commit=true`; `pytest` → 171 passed, `ruff check .` clean | pass |
| configuration order reversed | identical aggregate | `--configs lr_high lr_low` + `--seeds 3,1,2` → same means, spreads **and** fingerprint | pass |
| seed order reversed | identical aggregate | as above | pass |
| one record per configuration × seed | 6 records | `lr_low/seed-000000000{1,2,3}/experiment.json`, `lr_high/seed-000000000{1,2,3}/experiment.json`, each with git/data/environment/configuration/randomness and its own fingerprint | pass |
| configurations never mixed | no top-level mean | `statistics = {aggregated: false, note: …never mixed…}`; means only inside `configurations[]` | pass |
| partial failure | `partial`, exit 2, failed records kept | status `partial`, 3/6 runs, `lr_high` failed for seeds 1–3 with `CalledProcessError`, its records kept; `lr_low` still 3.365245 — the surviving aggregate is untouched | pass |
| complete failure / missing / non-numeric metrics | existing semantics | `failed` / `metric_missing` per run, per-configuration `failed` status, `mean=null`, `spread=null` (never 0) — covered by the test suite | pass |
| Stage 1A compatibility | seed-only sweep unchanged | no `configurations` section, no `configuration` key on runs, `seed-<seed>/` layout, top-level statistics aggregated as before; `tests/test_sweeps.py` (25 Stage 1A tests) passes unmodified | pass |
| Project 001 / 002 regression | unchanged | `pytest -q` → **171 passed**; `ruff check .` clean; EXP-001 `best_val=1.2816`, `val_loss 1.37519`, ppl 3.956, 138,752 params, `--init_from=resume` and `scripts/generate.py` all unchanged; EXP-002 metrics re-verified unchanged | pass |

**Results:** A two-configuration sweep now runs unattended, keeps every individual record,
reports each configuration's own mean ± spread, and refuses to publish a mixed number. On
this toy run the two learning rates separate by more than their seed spread (0.20 vs ≈0.06),
which is exactly the comparison a single-seed run could not support.

**Observations:**

- Reversing both lists initially produced a *different* fingerprint even though the
  statistics were identical: the per-configuration specs were resolved from the caller's
  spec, which carries whichever seed the CLI listed first. Configurations are now resolved
  from the canonicalised template (first *sorted* seed), the same class of order leak that
  D-028 fixed for the template spec. A regression test now varies configuration order and
  seed order together.
- A second, similar bug was found and fixed first: `run_experiment`'s new `configuration`
  parameter was shadowed by the local variable holding the record's configuration *section*,
  so `ctx.configuration` never reached the experiment body. Both bugs were invisible in
  "does it run" testing and only showed up in the order/repeat checks.
- All runs above were made on a **dirty** working tree, so their records honestly state
  `reproducible_from_commit=false`. The numbers are evidence about the machinery.
- After committing, the same 2-configuration × 3-seed sweep was re-run in a **fresh clone**
  of the pushed branch at `d15a25d` (clean tree, `reproducible_from_commit=true`): the
  per-configuration means and spreads are bit-identical to the dirty-tree run, and the three
  runs (a, b, reversed) share one fingerprint. The fingerprint itself differs from
  `f7306163…` because provenance now records a clean commit instead of a dirty tree and a
  different interpreter path — the *metrics* reproduce, which is what the machinery
  promises. A seed-only sweep in the same clone reproduced `fingerprint 03dfe74d…` twice
  and emitted no `configurations` section, confirming Stage 1A compatibility from the
  pushed branch as well.

**Conclusion:** Hypothesis confirmed on all four points. Stage 1B delivers the roadmap's
"2-config sweep runs unattended" exit criterion; Stage 1 as a whole is **not** complete
(bits-per-byte reporting and real licensed corpora remain).

**Next action:** use multi-configuration sweeps for any comparison that must survive seed
noise, and read per-configuration means (never a mixed mean) from `sweep.json`. Then
continue Stage 1 with bits-per-byte / per-character loss reporting. Do not start Project 004
on the basis of this entry.

**Artifacts:** `out/sweeps/EXP-005-{a,b,rev}/sweep.json` (+ `sweep.txt` and six
`<config>/seed-*/experiment.json` per sweep), `out/sweeps/EXP-005-partial/sweep.json`,
`tests/test_sweep_configs.py`, [docs/experiments.md §8](docs/experiments.md). The
fresh-clone re-verification sweeps live outside the repository (`/tmp/fresh-sweep-*`).
Run directories are git-ignored; the sweeps are regenerable with the commands above.

### EXP-006 — Bits per byte: does per-token loss misrank two tokenizations of the same text? ✅

> **This entry validates a metric, not a model.** Two tiny models (139k and 145k
> parameters) were trained for 200 steps on the templated synthetic corpus with a 141-type
> vocabulary. The numbers show how the *reporting* changes the ranking; they are not a
> statement about char-level vs word-level modelling in general, and no architecture or
> tokenizer is chosen on the basis of them.

- **Status:** complete (metric verification)
- **Date:** 2026-09-10
- **Objective:** verify Project 003 Stage 1 item 2: loss is now reported per **byte** and per
  **character** as well as per token, the byte/character counts are *measured* and exact
  rather than estimated, and a corpus without them says `null` instead of guessing.
- **Hypothesis:** (a) two corpora built from the same text report the same total bytes even
  though their token counts differ; (b) per-token loss and bits-per-byte **disagree about
  which model is better**, which is exactly the trap the roadmap item warns about; (c) the
  char-level run reproduces EXP-001's numbers exactly, so the change is a pure addition;
  (d) unknown counts are reported as `null`, never `0`.
- **Baseline:** EXP-001 (`val_loss 1.37519`, `val_ppl 3.956`, `bits_per_token 1.984`).

**Configuration**

- One corpus, two tokenizations (identical `prepare_data.py` source text, 200,094 chars,
  seed 1337):
  - `char` — 200,094 tokens, vocab 51, **1.0000 bytes/token**
  - `word` — 99,397 tokens, vocab 141, **2.0131 bytes/token**
  - both report **200,094 bytes and 200,094 characters** in total
- Model and schedule identical across runs (`configs/cpu_smoke.json`): 2 layers, 2 heads,
  `n_embd=64`, `block_size=64`, batch 8 × `accum_steps` 4, 200 steps, lr 3e-3 cosine,
  warmup 20, seed 1337, fp32, `num_threads=1`
- Third run: the word-level model at **100 steps** — because a word token carries ~2× the
  text, 100 word steps see roughly the same amount of text as 200 char steps
- Hardware: 2 CPU cores, 3 GB RAM, Python 3.11.2, torch 2.14.0+cu130

**Commands (exact)**

```bash
. .venv/bin/activate
python scripts/prepare_data.py --source synthetic --target-chars 200000 --level char --out data/synthetic-char
python scripts/prepare_data.py --source synthetic --target-chars 200000 --level word --out data/synthetic-word

# same hyperparameters, only the corpus (and, for the third run, the step count) changes
python scripts/train.py --config configs/cpu_smoke.json --set data.path=data/synthetic-char.bin \
    --set data.tokenizer=data/synthetic-char.tokenizer.json \
    --set train.out_dir=out/exp006-char  --set train.num_threads=1
python scripts/train.py --config configs/cpu_smoke.json --set data.path=data/synthetic-word.bin \
    --set data.tokenizer=data/synthetic-word.tokenizer.json \
    --set train.out_dir=out/exp006-word  --set train.num_threads=1
python scripts/train.py --config configs/cpu_smoke.json --set data.path=data/synthetic-word.bin \
    --set data.tokenizer=data/synthetic-word.tokenizer.json \
    --set train.out_dir=out/exp006-word100 --max-steps 100 --set train.num_threads=1

python scripts/evaluate.py --ckpt out/exp006-char/best     --data data/synthetic-char.bin --device cpu
python scripts/evaluate.py --ckpt out/exp006-word/best     --data data/synthetic-word.bin --device cpu
python scripts/evaluate.py --ckpt out/exp006-word100/best  --data data/synthetic-word.bin --device cpu
```

**Measured result** (full validation split, `scripts/evaluate.py`)

| run | level | steps | nats/token (`val_loss`) | `val_ppl` | bits/token | **bits/byte** | bits/char | params |
|---|---|---|---|---|---|---|---|---|
| `char` | char | 200 | **1.37519** | 3.956 | 1.984 | **1.983986** | 1.983986 | 138,752 |
| `word` | word | 200 | 2.16143 | 8.684 | 3.118 | **1.546772** | 1.546772 | 144,512 |
| `word100` | word | 100 | 2.30418 | 10.016 | 3.324 | **1.648925** | 1.648925 | 144,512 |

The two tokenizations disagree: **per-token loss ranks `char` 36 % "better" than `word`
(1.375 vs 2.161), while bits per byte ranks `word` 22 % better than `char` (1.547 vs
1.984).** The ranking inverts. It still inverts when the word model is given half the steps
and therefore sees roughly the same amount of text (1.649 vs 1.984).

**Metrics**

| check | expected | observed | verdict |
|---|---|---|---|
| byte counts are tokenizer-independent | same text ⇒ same bytes | `char` and `word` corpora both 200,094 bytes / 200,094 characters, despite 200,094 vs 99,397 tokens | pass |
| counts are exact, not estimated | per-token lengths sum to the source text | `tokenize()` pieces concatenate back to the text for both levels; `n_bytes_train + n_bytes_val == len(text.encode("utf-8"))` (test-pinned, incl. multi-byte UTF-8) | pass |
| per-token vs per-byte ranking | they must be able to disagree | they invert the ranking on this corpus (1.375 vs 2.161 per token; 1.984 vs 1.547 per byte) | pass |
| fresh clone at `e8abde6` (clean tree) | identical numbers | char `val_loss 1.37519` / `bits_per_byte 1.983986`, word `2.16143` / `1.546772` — bit-identical to the working-tree run; `pytest` 194 passed, `ruff` clean; seed-only sweep fingerprint `13e3f720…` twice; 2-config sweep `3.164264 ± 0.059841` / `3.365245 ± 0.068379` | pass |
| EXP-001 regression | char numbers unchanged | `val_loss 1.37519`, `val_ppl 3.956`, `bits_per_token 1.984` — identical to EXP-001; a second identical run reproduced `best_val=1.2816` / `bits_per_byte=1.983986` | pass |
| unknown counts | `null`, never `0` | a corpus prepared without counts reports `bits_per_byte: null` / `bits_per_char: null`, the CLI prints an explanatory note, and the trainer logs no `bits_per_byte` field | pass |
| old corpora still load | backward compatible | a pre-change `.meta.json` (no `n_bytes_*` keys) loads with `has_text_lengths=False` | pass |
| tokenizer honesty on ASCII | bits/char == bits/byte here | equal on this ASCII corpus (1.983986 / 1.983986); they differ on multi-byte text (covered by test) | pass |

**Results:** Per-token loss is a property of the tokenization as much as of the model, and
on this corpus it points the wrong way. Bits per byte is computed from byte counts that are
measured once, at prepare time, from the tokenizer's own pieces — so two corpora built from
the same text have identical byte totals by construction, whatever the tokenization. Every
consumer (`scripts/evaluate.py`, the trainer's `eval` and `run.end` events, `scripts/train.py`)
now reports both numbers, and the note in the output says which one is comparable.

**Observations:**

- The word-level model wins on bits per byte *even at half the steps*, so the effect is not
  simply "the word model saw more text". It is also not a recommendation: the synthetic
  corpus has 141 word types and templated sentences, and the two models differ in parameter
  count (138,752 vs 144,512) because the embedding table follows the vocabulary.
- `bits_per_char` equals `bits_per_byte` on this corpus only because it is ASCII. The
  distinction matters for the Indic text in Project 002's corpora, where a character can be
  three UTF-8 bytes.
- A sweep whose **every** run fails reports `status: failed`, exits 1, and keeps all six
  records with the error — seen for real in the fresh clone when the training command was
  pointed at a corpus that had not been prepared; the failure semantics needed no fixing,
  they simply worked.
- While adding the tests, a **pre-existing provenance leak** surfaced (recorded as Q-13):
  the environment section captures `torch.get_num_threads()` *before* the run body, so a run
  that changes torch's global thread count (any `Trainer` with `num_threads` set) makes a
  *later run in the same process* record a different environment and hence a different
  content fingerprint, even with identical metrics. The tests now pin threads to 1; the
  record shape was deliberately left unchanged so the fingerprints recorded in EXP-003,
  EXP-004 and EXP-005 still reproduce.

**Conclusion:** Hypothesis confirmed on all four points. Stage 1 item 2 (bits per byte /
per character) is implemented and measured. Stage 1 as a whole is **not** complete:
real licensed smoke corpora and automatic experiment-record wiring for `scripts/train.py` /
`scripts/tokenizer_*.py` remain.

**Next action:** quote **bits per byte** in any comparison that crosses tokenizers, and keep
quoting per-token loss as the training objective. Do not use this entry to pick a tokenizer.

**Artifacts:** `out/exp006-{char,word,word100}/` (configs, `train.jsonl`, checkpoints),
`data/synthetic-{char,word}.{bin,meta.json,tokenizer.json}`, `tests/test_loss_reporting.py`,
`src/frontier_ai/engine/metrics.py`, [docs/experiments.md §9](docs/experiments.md). Run
directories are git-ignored; the runs are regenerable with the commands above.

## 5. Log index

| ID | Title | Status | Date | Key metric |
|---|---|---|---|---|
| EXP-001 | CPU smoke test: end-to-end pipeline verification | complete | 2026-09-10 | val loss 3.93 → 1.378 (ppl 3.97) on synthetic corpus |
| EXP-002 | Tokenizer baselines on the Indian-language probe fixture | complete | 2026-09-10 | lossless byte-BPE (0 round-trip failures) vs char 128/179 and word 160/179 failures; en 1.83 vs pa 0.80 chars/token |
| EXP-003 | **Infrastructure verification** (not a benchmark): does the experiment record reproduce runs? | complete | 2026-09-10 | identical runs → identical fingerprints (`f54881c68513f295…` twice, `best_val=2.9223`); EXP-002 metrics reproduce exactly; a Project 001 determinism bug found and fixed |
| EXP-004 | **Infrastructure verification** (not a benchmark): do multi-seed sweeps reproduce and report spread honestly? | complete | 2026-09-10 | 5 seeds: mean 3.2626 ± 0.0476 (sample stdev); repeated and reversed-order sweeps give identical fingerprints `d4f498b9977a6f94…`; partial failure → `partial` + exit 2 |
| EXP-005 | **Infrastructure verification** (not a benchmark): do multi-configuration sweeps separate configurations and reproduce? | complete | 2026-09-10 | 2 configs × 3 seeds: `lr_high` 3.164264 ± 0.059841 vs `lr_low` 3.365245 ± 0.068379; identical fingerprint `f7306163a37b7a83…` across repeats and reversed order; one config failing → `partial`, 3/6 runs |
| EXP-006 | **Metric verification** (not a model benchmark): does per-token loss misrank two tokenizations of the same text? | complete | 2026-09-10 | char 1.37519 nats/token but **1.983986 bits/byte** vs word 2.16143 nats/token and **1.546772 bits/byte** — the ranking inverts; EXP-001 numbers reproduce exactly |
| EXP-007 | **Engineering verification** (not a model benchmark): do the CLIs record themselves, does a swept `train.py` keep its metrics, and is thread provenance stable? | complete | 2026-09-11 | 3-seed swept training run: `best_val` **mean 3.479525 ± 0.013175** (sample stdev), every run resolved from `results` (`metric_source: results`), 3 records and no inner ones, statistics and per-run fingerprints identical on repeat; in-process runs that set 3 threads record `3` and restore `1` |
| EXP-018 | P004B: fetch all 55 tokenizer-corpus sources (indic-tokenizer/v2) | complete | 2026-09-25 | build exit 0; 55/55 verified; 11 of 14 language slots EVALUATED (mr, ur, hi-en NOT_EVALUATED) |
| EXP-019 | P004B: lock all 55 tokenizer-corpus sources (indic-tokenizer/v2) | complete | 2026-09-25 | 55/55 pinned; the 47 earlier hashes unchanged, 8 new |
| EXP-020 | P004B: research Marathi and Urdu source candidates | complete | 2026-09-25 | No source declared; candidate scans failed the proofread and/or licence-evidence gates |
| EXP-021 | P004B: first Marathi and Urdu fetch; inspect transcription residues | complete | 2026-09-26 | 59/59 verified; prior 55 pins unchanged; new Urdu sources flagged, so none locked |
| EXP-022 | P004B: repair and re-fetch Marathi and Urdu sources | complete | 2026-09-26 | 59/59 verified; prior 55 pins unchanged; all four new sources clean, mr 256,165 and ur 208,621 characters |
| EXP-023 | P004B: lock Marathi and Urdu sources | complete | 2026-09-26 | 59/59 pinned; four new fingerprints match EXP-022; 55 earlier fingerprints unchanged |
| EXP-024 | P004B: complete the real tokenizer research corpus | complete | 2026-09-26 | 13 of 14 slots EVALUATED; 59 sources verified and pinned; hi-en honestly NOT_EVALUATED |
| EXP-025 | **Corpus freeze + verification** (not a benchmark): freeze and verify `indic-tokenizer/v2` (MASTER_CONTEXT §37 step 2) | complete | 2026-09-26 | 59/59 pins present and well-formed; all 59 hashes match the EXP-023 report prefixes; totals 34,684 docs / 4,211,707 chars; 415 passed, 1 skipped; manifest sha256 `aec3dfa0…` frozen (D-035); live re-fetch NOT YET VERIFIED from the sandbox |
| EXP-026 | P004B: live freshness re-fetch of frozen `indic-tokenizer/v2` | complete | 2026-09-26 | build exit 0; inspection: 59/59 sources verified and pinned, no flagged rows or REFUSED lines |
| EXP-027 | FrontierCorpus v1 frozen-corpus pilot | complete | 2026-09-26 | build/check exit 0; 34,684 input docs → 34,011 post-stages; train 30,584 / held_out 3,427; 4 train + 1 held-out shards |
| EXP-028 | **Production tokenizer sweep (EXP-A)** (MASTER_CONTEXT §37 step 5) | complete (PC run 2026-09-26) | 2026-09-26 | 20/20 cells gate PASS (15 grid + 5 supplement); mark_aware-32768 winner at 2.3054 held-out chars/token (+45 % over the 1.5971 baseline); hf-mark_aware ≡ py-mark_aware at every cell (cross-implementation validation); verified by scripts/summarize_sweep.py |
| EXP-029 | **Tokenizer-vs-tokenizer small-model comparison (EXP-B)** (MASTER_CONTEXT §37 step 6) | complete 2026-09-27 — winner `mark_aware-32768` (mean bpb 1.4463 vs 1.5779); founder approved → **D-040** (Frontier Tokenizer v1) | 2026-09-26 | mark_aware-32768 vs mark_aware-16384 × 3 seeds, fixed small model (configs/exp_b.json), held-out bits-per-byte; pre-registered decision rule (lower mean bpb wins; tie -> smaller vocab); selection becomes D-040 (Frontier Tokenizer v1) after review |
| EXP-030 | **Freeze Frontier Tokenizer v1** (D-040 follow-up; MASTER_CONTEXT §37 step 7) | complete 2026-09-27 — frozen at `fc7e8d8`, gates A–D PASS on the PC, Linux reproduces hashes + golden ids → **D-041** | 2026-09-27 | copy the EXP-028 `py-mark_aware-32768` artifact into tracked `tokenizers/frontier-tokenizer-v1/` behind 4 gates (EXP-029 fingerprint, structure, exact EXP-029 token counts, losslessness) + golden samples for cross-platform determinism; hash-verified loader `load_frontier_tokenizer()` |

*(Add one row per experiment as they are run. Do not add rows for planned experiments —
those belong in [ROADMAP.md](ROADMAP.md).)*

---

## 6. Reproducing experiments

**EXP-001 (model smoke test)**

```bash
. .venv/bin/activate
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic
python scripts/train.py --config configs/cpu_smoke.json
python scripts/evaluate.py --ckpt out/cpu-smoke/best --data data/synthetic.bin
cat out/cpu-smoke/train.jsonl
```

Expected on comparable hardware: val loss ≈ 1.3–1.4 after 200 steps in under a minute
(we have observed 1.3776 and 1.3230 on the same config across runs — see below). Exact
numbers vary with thread count and CPU; the corpus and seed are fixed, so loss curves should
be close but are **not** guaranteed bit-identical unless `train.deterministic=true`.

**EXP-002 (tokenizer baselines)**

```bash
. .venv/bin/activate
pip install ".[tokenizer]"                       # optional: enables the bpe_hf baseline
python scripts/tokenizer_prepare_corpus.py --out data/tokenizer/indic-v1 --seed 1337
for spec in "char:char:1024" "word:word:1024" \
            "bpe_python:bpe_py_512:512" "bpe_python:bpe_py_1024:1024" \
            "bpe_hf:bpe_hf_1024:1024"; do
  impl=${spec%%:*}; rest=${spec#*:}; name=${rest%%:*}; vs=${rest##*:}
  python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 --impl "$impl" \
      --vocab-size "$vs" --out "artifacts/tokenizers/$name" --exp-id EXP-002
done
python scripts/tokenizer_compare.py --corpus data/tokenizer/indic-v1 \
    --tokenizer artifacts/tokenizers/char artifacts/tokenizers/word \
                artifacts/tokenizers/bpe_py_512 artifacts/tokenizers/bpe_py_1024 \
                artifacts/tokenizers/bpe_hf_1024 \
    --out out/tokenizer/compare.json --exp-id EXP-002
```

The corpus is deterministic (seed 1337) and BPE merge selection is deterministic, so token
counts reproduce exactly. `bpe_hf` requires the optional `tokenizers` package; without it,
drop that row.

**EXP-003 (infrastructure verification)**

```bash
. .venv/bin/activate
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic
python scripts/experiment_record.py --exp-id EXP-003 --seed 1337 --name "identical runs" \
    --out out/experiments/EXP-003-same --config configs/cpu_smoke.json \
    --data data/synthetic.bin --tag project-001 -- \
    python scripts/train.py --config configs/cpu_smoke.json --max-steps 30 \
        --set train.num_threads=1
cat out/experiments/EXP-003-same/experiment.json
```

Run it twice: the `fingerprint` in the last line of the output must be identical across the
two runs.

**EXP-004 (multi-seed sweeps)**

```bash
. .venv/bin/activate
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic
python scripts/experiment_sweep.py --exp-id EXP-004 --seeds 1,2,3,4,5 \
    --metric final_val_loss --data data/synthetic.bin --out out/sweeps/EXP-004 -- \
    python my_per_seed_command.py --seed "{seed}"   # must print JSON with final_val_loss
cat out/sweeps/EXP-004/sweep.json                   # mean, spread, per-seed values
ls out/sweeps/EXP-004/seed-*/experiment.json        # one full record per seed
```

Run it twice (and once with the seeds in reverse order): `mean`, `spread` and the sweep
`fingerprint` must be identical every time. Add a seed that exits non-zero to see
`status=partial` with exit code 2 and the failed seed's own record preserved.

**EXP-005 (multi-configuration sweeps)**

```bash
. .venv/bin/activate
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic
python scripts/experiment_sweep.py --exp-id EXP-005 --seeds 1,2,3 \
    --configs "lr_low:params.lr=0.005" "lr_high:params.lr=0.05" \
    --metric final_val_loss --data data/synthetic.bin --out out/sweeps/EXP-005 -- \
    python my_experiment.py --config-name "{config}" --seed "{seed}"
cat out/sweeps/EXP-005/sweep.json        # per-configuration mean ± spread
ls out/sweeps/EXP-005/*/seed-*/experiment.json   # one full record per configuration x seed
```

Each configuration is aggregated **separately**; with two or more configurations the
top-level `statistics` reports `aggregated: false` and publishes no mixed mean. Reversing
the `--configs` order or the `--seeds` order must not change the means, the spreads or the
sweep fingerprint.

**EXP-006 (loss per byte instead of per token)**

```bash
. .venv/bin/activate
python scripts/prepare_data.py --source synthetic --target-chars 200000 --level char --out data/synthetic-char
python scripts/prepare_data.py --source synthetic --target-chars 200000 --level word --out data/synthetic-word
python scripts/train.py --config configs/cpu_smoke.json --set data.path=data/synthetic-char.bin \
    --set train.out_dir=out/exp006-char --set train.num_threads=1
python scripts/train.py --config configs/cpu_smoke.json --set data.path=data/synthetic-word.bin \
    --set train.out_dir=out/exp006-word --set train.num_threads=1
python scripts/evaluate.py --ckpt out/exp006-char/best --data data/synthetic-char.bin --device cpu
python scripts/evaluate.py --ckpt out/exp006-word/best --data data/synthetic-word.bin --device cpu
```

`evaluate.py` prints `val_loss` (nats per token), `bits_per_token`, **and** `bits_per_byte` /
`bits_per_char`. Only the per-byte and per-character numbers compare across tokenizers. If a
corpus predates this bookkeeping they are `null` and the output says to re-run
`prepare_data.py`. This is recorded here because the rules require every run to be logged — it
verifies the *infrastructure*, not model quality. Its own record will show `dirty=true`
whenever the working tree is dirty; that is the point, not a defect.

---

### EXP-007 — CLIs that record themselves: one record, metrics preserved, stable thread provenance ✅

> **This entry is engineering verification of infrastructure, not a model or tokenizer
> result.** No architecture, tokenizer or hyperparameter is chosen here, and the loss
> numbers below are a *signal that the plumbing works*, not a claim about model quality.
> They are reproducible with the commands given; the model is the 139k-parameter CPU smoke
> config trained for 10 steps on the synthetic corpus.

- **Status:** complete (infrastructure verification)
- **Date:** 2026-09-11
- **Objective:** verify Project 003 Stage 1 item 4 (and the two audit findings it closed):
  (a) `scripts/train.py` and `scripts/tokenizer_*.py` write the standard record when run
  directly; (b) a run that is wrapped or swept produces **exactly one** record, owned by the
  outer run; (c) the swept CLI's structured metrics (`best_val`) still reach the sweep;
  (d) torch thread provenance is stable and honest across in-process runs.
- **Hypothesis:** (a)–(c) hold by construction once a nested script publishes its results as
  one JSON line instead of writing a second record (D-034); (d) holds once the environment
  is captured *after* the body and the caller's thread count is restored (D-034, Q-13).
- **Baseline:** EXP-004 (multi-seed sweeps, mean ± sample spread) and EXP-006 (bits per
  byte). Before this work, a swept `train.py` reported `metric_missing` for `best_val`
  because the per-seed record contained only `exit_code`, `stdout_tail` and `stderr_tail`.

**Configuration**

- Model: `configs/cpu_smoke.json` (2 layers, 2 heads, `n_embd=64`, `block_size=64`),
  138,752 parameters, 10 steps, seed = sweep seed, `num_threads=1`
- Data: `data/synthetic.bin` (200,094 chars, vocab 51, seed 1337) — the same corpus as
  EXP-001, so nothing here is a new data claim
- Sweep: `scripts/experiment_sweep.py --seeds 1,2,3 --metric best_val`
- Hardware: 2 CPU cores, 3 GB RAM, Python 3.11.2, torch 2.14.0+cu130
- Code revision: branch `arena/01a08a78-frontier-ai`, the commit that adds D-034 (see
  `git log`); the runs below were made on a dirty tree, so their records carry
  `dirty=true` — that is D-026 working as intended, not a defect

**Commands (exact)**

```bash
. .venv/bin/activate
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic

# swept training runs: the CLI records nothing of its own, the sweep owns the records
python scripts/experiment_sweep.py --exp-id EXP-907 --seeds 1,2,3 --metric best_val \
    --data data/synthetic.bin --out out/sweeps/EXP-907 -- \
    python scripts/train.py --config configs/cpu_smoke.json --set train.seed={seed} \
        --set train.out_dir=out/sweeps/EXP-907-run-{seed} --max-steps 10

# wrapped run: one record, and the training metrics inside it
python scripts/experiment_record.py --exp-id EXP-907 --data data/synthetic.bin \
    --out out/experiments/EXP-907-wrapped -- \
    python scripts/train.py --config configs/cpu_smoke.json --set train.out_dir=out/... --max-steps 3
```

**Metrics**

| check | expected | observed | verdict |
|---|---|---|---|
| swept training run aggregates `best_val` | `metric_source: results`, no `metric_missing` | mean **3.479525**, spread **0.013175** (sample stdev, n=3); per-seed 3.471531 / 3.494731 / 3.472312; all three `metric_source: results` | pass |
| repeat of the same sweep | identical statistics and records | mean 3.479525, spread 0.013175, per-seed fingerprints `8a71d80d7e616dd9…`, `fa1d86389f710fd6…`, `62b293e2ca449dfb…` reproduced exactly | pass |
| one record per run | exactly 3 records, none in the training directories | 3 records under `seed-000000000{1,2,3}/experiment.json`; the three training directories contain no `experiment.json` | pass |
| wrapped run keeps the metrics | outer record carries the training results | `best_val`, `best_bpb`, `steps`, `n_params`, `tokens_seen` present; `exit_code` and the log tails untouched by the merge | pass |
| wrapped failure | one **failed** record, non-zero exit | record `status: failed`, error `CalledProcessError`, no inner record | pass |
| wrapped tokenizer run | one record, tokenizer metrics inside it | `impl`, `vocab_size`, `sample_round_trip_ok` present in the wrapper's record; artifact directory has no record | pass |
| in-process thread provenance (D-034) | the value the body used, restored afterwards | body setting 3 threads → `environment.torch.num_threads = 3` recorded, caller's `1` restored after the run | pass |
| repeated in-process runs with identical config | identical provenance and fingerprints | two runs that each set 3 threads: equal environments, equal fingerprints | pass |

**Results:** the plumbing is complete in both directions — a script records itself when it
owns the run, and publishes its metrics when it does not. The 3-seed numbers above are a
*signal* (spread ≈ 0.013, i.e. ~0.4% around the mean, on a 10-step toy run); they are not a
model result and nothing in Stage 2 should be decided from them.

**Fresh-clone re-verification (code commit `0104eb1`):** in a pristine clone of the pushed
branch with an independently built environment: `249 passed, 1 skipped`, `ruff check .`
clean, and the same swept numbers — `best_val` **mean 3.479524850845337**,
**spread 0.013175015328436988** over per-seed 3.471531 / 3.494731 / 3.472312, every run
`metric_source: results`. One-record guarantee: 8 records across the swept,
configuration-swept and wrapped runs, and **0** records inside the training/artifact
directories they wrapped. Thread provenance in process: three runs whose body sets 3 threads
recorded `[3, 3, 3]` — the old code, which captured before the body, would have recorded the
caller's `[1, 1, 1]` — the caller's `1` was restored after each run, and two identical
in-process runs produced the same fingerprint.

**Not verified here (recorded rather than hidden):** the D-031 licensed corpora. Their
manifest ships with `verified: false` and `sha256: null` by design, hashes are pinned only
after a real fetch, and `tests/test_smoke_corpus.py` skips its end-to-end case until
`scripts/fetch_smoke_corpus.py --fetch` has been run with network access. No corpus numbers
are claimed in this entry.

### EXP-018 — P004B: fetch all 55 tokenizer corpus sources (indic-tokenizer/v2)

- **Status:** complete
- **Date:** 2026-09-25
- **Objective:** Download and verify all declared sources for indic-tokenizer/v2, generating baseline hashes and coverage report
- **Hypothesis:** All 55 sources will verify successfully (licensed, proofread, accessible), establishing the baseline for locking
- **Baseline:** EXP-017 (Gujarati, Malayalam, Odia, Assamese sources locked)

**Configuration**
- Script: `scripts/build_tokenizer_corpus.py`
- Flags: `--fetch --exp-id EXP-018 --out data/tokenizer/indic-tokenizer-v2`
- Data sources: 55 declared sources from corpora/tokenizer/indic-tokenizer-v2/sources.json
- Hardware: Local machine (operator's computer)
- Seed: Not applicable (data acquisition)
- Deterministic: yes

**Metrics**
- Sources downloaded: 55/55
- Sources verified: 55/55
- Exit code: 0 (no pinned sources changed)
- Documents acquired: 28,451 train / 3,188 held out
- Evaluation status: 11 language slots EVALUATED (≥500 docs, ≥200k chars each)
- Inspection report: `corpora/tokenizer/indic-tokenizer-v2/reports/EXP-018-inspection.txt`

**Results:**
- First fetch of all 55 sources completed successfully
- All sources verified with proper licensing evidence
- Old project copy moved: `frontier-ai/` → `..\frontier-ai-old-copy`
- 4 sources flagged for genuine years in text (ml, kn, ta x2) — these are correct text content, not errors
- Cleaner removed wiki typos: 18 broken {{gap}} templates, missing template links, literal <poem> tag, stray }}
- Coverage shows 11 language slots now EVALUATED: en, hi, bn, gu, ml, or, as, pa, kn, te, ta
- 3 slots remain NOT_EVALUATED: mr, ur, hi-en (awaiting source declaration)

**Fresh-clone re-verification:** Not applicable (data acquisition experiment)

**Next action:** Lock all verified sources with cryptographic hashes (EXP-019), then record experiments and update status documentation.


### EXP-019 — P004B: lock all 55 tokenizer corpus sources (indic-tokenizer/v2)

- **Status:** complete
- **Date:** 2026-09-25
- **Objective:** Write cryptographic hashes for all verified sources to manifest, establishing the locked baseline
- **Hypothesis:** All 55 sources will lock successfully with hashes matching their verified content from EXP-018
- **Baseline:** EXP-018 (fetch of all 55 sources)

**Configuration**
- Script: `scripts/build_tokenizer_corpus.py`
- Flags: `--fetch --pin --exp-id EXP-019 --out data/tokenizer/indic-tokenizer-v2`
- Data sources: 55 declared sources from corpora/tokenizer/indic-tokenizer-v2/sources.json
- Hardware: Local machine (operator's computer)
- Seed: Not applicable (data acquisition)
- Deterministic: yes

**Metrics**
- Sources downloaded: 55/55
- Sources verified: 55/55
- Sources pinned: 55/55
- Exit code: 0 (no pinned sources changed from EXP-018 baseline)
- All hashes written: sha256, verified=true, retrieved_at updated
- Evaluation status unchanged: 11 language slots EVALUATED, 3 NOT_EVALUATED
- Hash prefixes match EXP-018 inspection report output

**Results:**
- All 55 sources locked with cryptographic hashes
- Manifest updated: only sha256, verified, and retrieved_at fields changed
- Each hash begins with prefix reported in EXP-018 inspection
- No text changes detected (exit code 0 confirms no pinned sources altered)
- Licensing compliance maintained: all sources properly attributed and verified
- Text integrity verified: cleaner repairs documented in inspection report

**Fresh-clone re-verification:** Not applicable (data acquisition experiment)

**Next action:** Record experiments in EXPERIMENTS.md, update status documentation (NEW_CHAT_START_HERE.md), then begin research for Marathi and Urdu sources as outlined in handover section 10.

### EXP-020 — P004B: research Marathi and Urdu source candidates

- **Status:** complete
- **Date:** 2026-09-25
- **Objective:** Identify lawful, sufficiently proofread Wikisource works for the remaining Marathi and Urdu tokenizer slots.
- **Method:** Inspected Wikisource API metadata, index pages, scan-page quality, author/year evidence, and site rights information. No corpus fetch or hash pin was run.

**Results:**
- Marathi `आईबापांचा मित्र` identifies Moro Ganesh Londhe (1911), but the scan-page review found only 5 level-3 pages among the first 110 pages and many pages without proofread status. `श्री एकनाथी भागवत` likewise did not expose a proofread range suitable for declaration in the inspected index response. The modern `'भारता'साठी` candidate was rejected because its author is Sharad Joshi.
- Urdu candidate scans including `Tota Kahani` (1801), `Ram Charcha in Urdu by Munshi Premchand`, and `Betal-pachcheesi` did not yet provide a sufficiently broad, fully proofread, and independently licensed range. The Urdu site rights endpoint reports CC BY-SA 4.0 for wiki contributions, but that does not establish the underlying scan work's public-domain status.
- Urdu-specific support was completed separately: sentence marks `۔` and `؟` are recognized, and the `سانچہ` Template namespace is registered.
- No source was declared, fetched, verified, or pinned. Both language slots remain honestly `NOT_EVALUATED`.

**Next action:** Find a Marathi or Urdu scan with explicit public-domain evidence and enough level-3/4 pages; do not declare a source until the complete licence and page-quality review succeeds.

### EXP-021 — P004B: first fetch of Marathi and Urdu scan ranges

- **Status:** complete
- **Date:** 2026-09-26
- **Objective:** Confirm the four declared scan ranges fetch, pass live rights verification and reach the corpus targets.
- **Baseline:** EXP-020 (candidate survey; no sources declared)
- **Command:** `python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-021 --out data/tokenizer/indic-tokenizer-v2`
- **Code revision:** `ce5739d`

**Results:**
- Build exit 0: 59/59 sources verified; all 55 existing pinned sources were unchanged.
- Marathi *स्फुट गोष्टी भाग तिसरा*: 1,667 documents / 256,165 corpus characters; source text 257,901 characters.
- Urdu *Ram Charcha*: 1,378 documents / 208,644 corpus characters; source text 210,517 characters.
- Both slots met the targets, but the initial inspection identified Urdu markup/text residues; no new source was pinned.

**Observations:** The inspection report flagged a malformed `xx-larger` heading and an unmatched `[[` in Urdu, `1004` embedded in page 62 despite its absence from the scan image, and U+200E direction marks. The saved EXP-021 report preserves the pre-repair evidence.

**Conclusion:** The live verification and coverage gates passed; human review of the Urdu text flags did not, so locking was deferred.

**Next action:** Repair only scan-confirmed artifacts with tests, then re-fetch and inspect all sources before any lock.
**Artifacts:** `corpora/tokenizer/indic-tokenizer-v2/reports/EXP-021-inspection.txt`; fetch output under `data/tokenizer/indic-tokenizer-v2`.

### EXP-022 — P004B: repair and re-fetch Marathi and Urdu scan ranges

- **Status:** complete
- **Date:** 2026-09-26
- **Objective:** Verify narrow, evidence-based Urdu cleanup and confirm Marathi/Urdu coverage without changing any earlier locked source.
- **Baseline:** EXP-021
- **Command:** `python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-022 --out data/tokenizer/indic-tokenizer-v2`
- **Code revision:** `edd2303`

**Results:**
- Build exit 0: 59/59 verified; all 55 previously pinned sources were byte-identical.
- Inspection exit 1 only because existing pinned ml, kn and ta sources retain documented digit flags; all four new sources are unflagged.
- Urdu repairs: malformed page-46 heading wrapper, exact page-62 scan-inconsistent number in its observed context, one unpaired page-248 `[[`, and two U+200E bidi marks. The cleaner counts each repair and retains surrounding prose.
- Coverage: mr 1,667 documents / 256,165 characters; ur 1,378 / 208,621 characters.
- EXP-022 hash prefixes: mr `e1e7dba1bfe5`; ur ranges `fb22db9983ef`, `36cd31410d2b`, `4835213b51c2`.
- Focused cleaner tests: 53 passed. `ruff check src tests scripts`: passed. Full suite: 393 passed, 1 skipped, 22 failed on pre-existing Windows encoding/line-ending behavior and missing `data/synthetic.bin`.

**Observations:** U+200E is a bidi-formatting control, not printed prose. The `1004` value was in the middle of a sentence, not a standalone line; the cleaner removes it only in the exact scan-inconsistent phrase confirmed against the scan image.

**Conclusion:** The new sources pass the text review and both slots remain `EVALUATED`; no fingerprint was written during either fetch.

**Next action:** Run a fresh `--fetch --pin` as EXP-023, compare its manifest diff with these hash prefixes, then record the final P004B status.
**Artifacts:** `corpora/tokenizer/indic-tokenizer-v2/reports/EXP-022-inspection.txt`; `data/tokenizer/indic-tokenizer-v2`.

### EXP-023 — P004B: lock Marathi and Urdu scan ranges

- **Status:** complete
- **Date:** 2026-09-26
- **Objective:** Lock the four reviewed Marathi and Urdu sources from a fresh fetch without changing any of the 55 existing fingerprints.
- **Baseline:** EXP-022
- **Command:** `python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-023 --out data/tokenizer/indic-tokenizer-v2`
- **Code revision:** `d9ddf6a`

**Results:**
- Build exit 0; inspection header: 59 sources, 59 verified, 59 pinned.
- The four new fingerprints match the EXP-022 report: mr `e1e7dba1bfe5`; ur `fb22db9983ef`, `36cd31410d2b`, `4835213b51c2`.
- The 55 previous fingerprints are unchanged. The manifest-only lock diff changed `sha256`, `verified` and `retrieved_at` for the four new sources and only `retrieved_at` for the 55 previously pinned.
- All four new sources remain free of inspection flags. The inspector's exit 1 is due to existing documented flags in the pinned Malayalam, Kannada and Tamil sources.
- Coverage remains mr 1,667 / 256,165 and ur 1,378 / 208,621 (documents / characters).

**Conclusion:** The fresh lock matched the reviewed fetch exactly; the four sources are now verified and fingerprinted.

**Next action:** Record P004B complete, with hi-en still honestly `NOT_EVALUATED`.
**Artifacts:** `corpora/tokenizer/indic-tokenizer-v2/reports/EXP-023-inspection.txt`; `corpora/tokenizer/indic-tokenizer-v2/sources.json`.

### EXP-024 — P004B: complete the real tokenizer research corpus

- **Status:** complete
- **Date:** 2026-09-26
- **Objective:** Close P004B with auditable coverage, legal-source and fingerprint status for all 14 language slots.
- **Baseline:** EXP-023; 11 slots were evaluated before the Marathi and Urdu acquisition.
- **Code revision:** `d9ddf6a`

**Results:**
- The manifest contains 59 verified, fingerprinted sources. Thirteen of 14 slots meet both corpus targets; mr has 1,667 documents / 256,165 characters and ur has 1,378 / 208,621.
- hi-en remains `NOT_EVALUATED`: the search found no lawful, attributable Hinglish/Romanised-Hindi source. No text was translated, synthesized, padded or borrowed from a related language.
- The new Marathi and Urdu sources have independent underlying-work public-domain evidence and live CC BY-SA evidence for their Wikisource transcriptions; all included scan pages passed the proofreading gate.
- Targeted MediaWiki cleaner tests pass (53/53) and Ruff passes. The full suite reports 393 passed, 1 skipped and 22 failures tied to missing synthetic fixture data and existing Windows encoding/line-ending assumptions; see EXP-022.

**Conclusion:** P004B is complete. The corpus is ready for the later tokenizer research work; an unevaluated slot is reported honestly rather than filled with a substitute.

**Next action:** The operator may merge `arena/01a0d31f-frontier-ai` into `main` when ready; no pull request was opened.
**Artifacts:** `corpora/tokenizer/indic-tokenizer-v2/reports/EXP-021-inspection.txt`, `EXP-022-inspection.txt`, `EXP-023-inspection.txt`, `corpora/tokenizer/indic-tokenizer-v2/sources.json`.

### EXP-025 — Freeze and verify `indic-tokenizer/v2` (MASTER_CONTEXT §37 step 2)

> **This entry freezes and verifies a corpus definition, not a model.** No tokenizer is
> trained, no vocabulary is sized and no claim about tokenizer quality is made. It answers
> one question: *is the corpus that P004B locked exactly the corpus downstream stages will
> use, and is that identity recorded so it cannot silently change?*

- **Status:** complete (corpus definition frozen; one live check intentionally out of reach, recorded below)
- **Date:** 2026-09-26
- **Objective:** freeze `indic-tokenizer/v2` at the state P004B locked (EXP-018…EXP-024), and verify that freeze from the committed repository state: every pin present and well-formed, every pin consistent with the independent inspection report, the recorded coverage totals correct, and the test suite green.
- **Baseline:** EXP-024 (P004B complete). All of its numbers must still hold.
- **Environment:** Arena sandbox, Linux (Python 3.11.2, torch 2.14.0+cu130, tokenizers 0.23.2). **No route to the corpus hosts** (wikisource.org, gutenberg.org fail TLS — re-checked 2026-09-25 and 2026-09-26), so everything below is measured from git, not from a live fetch.

**Commands (exact, all read-only with respect to the corpus):**

```bash
# 1. manifest identity and pin audit (python one-liner over sources.json)
# 2. offline build check, handover §4 — must exit 0, all source slots UNVERIFIED without text
python scripts/build_tokenizer_corpus.py --exp-id OFFLINE-CHECK --out /tmp/offline-check-corpus --no-record
# 3. full suite + lint
python -m pytest
ruff check src tests scripts
# 4. pin cross-check: every manifest sha256 vs the 12-hex prefix in the committed report
#    (python one-liner over sources.json + reports/EXP-023-inspection.txt)
```

**Metrics**

| check | expected | observed | verdict |
|---|---|---|---|
| sources declared / pinned / verified | 59 / 59 / 59 | 59 / 59 / 59, every pin a 64-hex sha256, every pinned source has `retrieved_at` | pass |
| licences within the allow-list | only CC0-1.0 / CC-BY-4.0 / CC-BY-SA-4.0 / PD-US | CC-BY-SA-4.0 × 57, PD-US × 2 | pass |
| licence evidence recorded | every non-Gutenberg source declares `license_evidence` | all 57 mediawiki-parse sources do | pass |
| pin ↔ report cross-check | every manifest hash starts with the prefix EXP-023's inspection reported for that source | 59 of 59 match; no mismatches, no duplicate rows | pass |
| coverage totals (summed from `reports/EXP-023-inspection.txt`) | 34,684 documents / 4,211,707 characters (the PROJECT_CONTEXT claim) | 34,684 / 4,211,707 exactly; all 13 `EVALUATED` slots above both targets (min: 1,077 documents, 201,209 characters) | pass |
| offline build check (handover §4) | exit 0; source slots `UNVERIFIED` without fetched text; hi-en `NOT_EVALUATED` | exit 0; 13 `UNVERIFIED` (en hi bn mr gu ta te kn ml pa or as ur), 1 `NOT_EVALUATED` (hi-en) | pass |
| test suite (Linux) | green | **415 passed, 1 skipped** in 161 s (skip = the smoke-corpus fetch test, needs a fetched smoke corpus); matches the state recorded in PROJECT_CONTEXT | pass |
| lint | clean | `ruff check src tests scripts`: All checks passed | pass |
| manifest untouched by this run | `sources.json` unmodified | git status: no change to the manifest | pass |

**Freeze identity**

- corpus id/version: `indic-tokenizer` / `v2` (schema 1.0)
- manifest: `corpora/tokenizer/indic-tokenizer-v2/sources.json`
- manifest sha256: `aec3dfa091370832e7f48f5fabb3d749716ef50a7cae70fc28b45ea272da67bf`
- corpus state commit: `76bb127` (main after PR #2)
- machine-readable record: `corpora/tokenizer/indic-tokenizer-v2/FREEZE.json`
- policy: D-035 — any manifest change needs the founder's approval, a new experiment, and
  produces `v3` with its own freeze record; v2 is never edited in place.

**Not yet verified (recorded rather than hidden):** the live re-fetch that proves no pinned
source's text has changed since the EXP-023 lock. The Arena sandbox cannot reach the hosts,
so the pins are as fresh as 2026-09-26 (EXP-023). On a network-enabled machine,
`python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-0NN --out
data/tokenizer/indic-tokenizer-v2` is that check: exit 0 means every pin still holds, a
`REFUSED:` line names a changed source (runbook §0.5), and the outcome must be appended to
`FREEZE.json`'s `not_yet_verified` note.

**Conclusion:** the corpus P004B acquired is frozen and its identity is recorded and
verifiable from git alone. Downstream stages (tokenizer sweeps, tokenizer-vs-tokenizer
model experiments) must cite `indic-tokenizer/v2` + the manifest sha256 above, train on its
train split and evaluate on its held-out split.

**Next action:** MASTER_CONTEXT §37 step 3 — design and build the FoundationCorpus
(FrontierCorpus v1) pipeline. Before any code, the same audit as before the last step
applies: what exists, what is missing, smallest useful step, founder approval.

**Artifacts:** `corpora/tokenizer/indic-tokenizer-v2/FREEZE.json`,
`corpora/tokenizer/indic-tokenizer-v2/reports/EXP-023-inspection.txt` (evidence),
`DECISIONS.md` D-035.

### EXP-026 — Live freshness re-fetch of frozen `indic-tokenizer/v2`

- **Status:** complete
- **Date:** 2026-09-26
- **Objective:** confirm from a network-enabled machine that each of the 59 pinned source texts still matches the frozen EXP-023 fingerprint.
- **Baseline:** EXP-025; frozen manifest SHA-256 `aec3dfa091370832e7f48f5fabb3d749716ef50a7cae70fc28b45ea272da67bf`.
- **Environment:** Windows, operator's machine; branch `arena/01a0dc16-frontier-ai`.
- **Command:** `python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-026 --out data/tokenizer/indic-tokenizer-v2`
- **Build exit code:** 0.

**Results**

| check | observed | verdict |
|---|---|---|
| Live source fetch | 59 sources ingested, 59 verified | pass |
| Pinned fingerprints | 59 sources pinned in the manifest | pass |
| Changed-source refusals | no `REFUSED:` lines | pass |
| Inspection flags | no flagged rows | pass |

**Conclusion:** All 59 sources were freshly fetched and verified against their existing pins; no fingerprint changed. No source was re-pinned and the manifest was not edited.

**Next action:** Arena agent starts the STEP 3 plan for the FrontierCorpus v1 pipeline.

**Artifacts:** `corpora/tokenizer/indic-tokenizer-v2/reports/EXP-026-refetch.txt`,
`corpora/tokenizer/indic-tokenizer-v2/FREEZE.json`.

### EXP-027 — FrontierCorpus v1 frozen-corpus pilot

- **Status:** complete
- **Date:** 2026-09-26
- **Objective:** build and verify the FrontierCorpus v1 pilot from the frozen `indic-tokenizer/v2` corpus, without fetching or changing sources.
- **Baseline:** EXP-026 freshness check; frozen manifest SHA-256 `aec3dfa091370832e7f48f5fabb3d749716ef50a7cae70fc28b45ea272da67bf`.
- **Environment:** Windows, operator's machine; branch `arena/01a0dc16-frontier-ai`.
- **Commands:** `python scripts/build_frontier_corpus.py --exp-id EXP-027`; `python scripts/build_frontier_corpus.py --check`.
- **Initial build exit code:** 2. The pilot compared raw file bytes with text hashes, while Windows had stored the same decoded text with CRLF newlines. All 59 cached sources' decoded text matched their pins; no fetch was performed. Fixed in `dbc67be` to validate the decoded text consumed by the pipeline, with a CRLF regression test.
- **Final build exit code:** 0. **`--check` exit code:** 0.

**Stage summary (documents in -> kept; removed with reasons; flagged kept):**

```text
  normalize     in= 34684  kept= 34684  removed=   0  flagged_docs=   0
  langid        in= 34684  kept= 34350  removed= 334  flagged_docs=   0
  quality       in= 34350  kept= 34350  removed=   0  flagged_docs=   1
  exact_dedup   in= 34350  kept= 34011  removed= 339  flagged_docs=   0
```

**Output:** train 30,584 documents / 3,778,463 chars; held_out 3,427 documents / 424,727 chars. Shards: 4 train, 1 held_out. Manifest SHA-256: `1c41beb9b1a399b2bb6d51421e6f1d7420cfebac57fb02e55724f44c6a05548f`.

**`exact_dedup` per-language statistics** (copied from `manifest.json`):

```json
{"as":{"in":2499,"kept":2460,"removed":39},"bn":{"in":4225,"kept":4012,"removed":213},"en":{"in":5913,"kept":5898,"removed":15},"gu":{"in":1965,"kept":1965,"removed":0},"hi":{"in":5290,"kept":5279,"removed":11},"kn":{"in":2632,"kept":2621,"removed":11},"ml":{"in":2052,"kept":2048,"removed":4},"mr":{"in":1658,"kept":1638,"removed":20},"or":{"in":1873,"kept":1869,"removed":4},"pa":{"in":2400,"kept":2392,"removed":8},"ta":{"in":1074,"kept":1070,"removed":4},"te":{"in":1401,"kept":1397,"removed":4},"ur":{"in":1368,"kept":1362,"removed":6}}
```

**Dedup cross-check:** EXP-023 counted 396 duplicate occurrences on the original text. NFC normalization yields 399 duplicate occurrences before langid; the langid stage removes 60 duplicate occurrences along with other rejected documents, leaving the observed 339 for exact dedup. Thus the difference is accounted for by pre-dedup stages (three additional NFC-equivalent duplicates, then 60 duplicate occurrences filtered by langid).

**Conclusion:** The frozen corpus built successfully into the pilot dataset, and `--check` confirmed every shard hash. No source was fetched or re-pinned.

**Next action:** Review the pilot statistics and proceed to the next approved FrontierCorpus step.

**Artifacts:** `corpora/frontier/v1/manifest.json`, `corpora/frontier/v1/reports/EXP-027-build.txt`, `scripts/build_frontier_corpus.py`, `tests/test_frontier_corpus_build.py`.

### EXP-028 — Production tokenizer sweep (EXP-A) — MASTER_CONTEXT §37 step 5

- **Status:** in progress — harness delivered and tested 2026-09-26; the full grid runs on the operator's PC (the corpus text and the frozen shards are git-ignored and PC-only)
- **Date:** 2026-09-26 (harness)
- **Approved:** founder, 2026-09-26 ("approve EXP-A")
- **Objective:** train the approved 15-configuration grid on FrontierCorpus v1, gate every configuration on losslessness (exact round-trip of EVERY train and held-out document), and score the held-out side with model-free token-density metrics. **No tokenizer is selected by this experiment** — the top-2 hand-off to EXP-B (small model, ≥ 3 seeds) happens after the sweep is reviewed.

**Grid (15 configurations — D-038):**

| implementation | pre-tokenization | vocab sizes | cells |
| --- | --- | --- | --- |
| `bpe_python` | `mark_aware` (default; combining marks stay attached) | 2048, 4096, 8192, 16384, 32768 | 5 |
| `bpe_python` | `gpt2_style` (GPT-2's `\p{L}`/`\p{N}`-based regex; marks shatter syllables) | 2048, 4096, 8192, 16384, 32768 | 5 |
| `bpe_hf` | built-in `ByteLevel` (the GPT-2-style variant) | 2048, 4096, 8192, 16384, 32768 | 5 |

The 5 missing `bpe_hf` + mark-aware combinations are a documented limitation (they need a custom HuggingFace `PreTokenizer` that cannot be tested until the PC run) — not silently skipped.

**Metrics (model-free by design):** losslessness gate (100 % required, both sides); held-out `tokens_per_char` / `chars_per_token` / `tokens_per_word` overall and per language (via `evaluate._measure` with documents mapped to examples); vocab size; training time. Bits-per-character needs a language model — that is EXP-B.

**Input identity (hard gates, exit 2):** the frozen `corpora/frontier/v1` is never modified. On-disk shards must hash to their manifest; the manifest's split parameters drive a deterministic re-derivation of the per-language documents from the frozen v2 sources; the re-derivation must reproduce the frozen shards exactly (per-language counts and the full shard line sequence, side by side).

**Harness (this delivery):**
- `src/frontier_ai/corpus/frontier_docs.py` — shared derivation (frozen-input verification, stages, split) used by both the builder and the sweep; the builder was refactored onto it with byte-identical output (its e2e test re-passes).
- `src/frontier_ai/tokenization/sweep.py` — grid, losslessness gate, doc-level measurement, per-config training.
- `scripts/run_tokenizer_sweep.py` — self-recording (D-032; one full record per configuration via the sweep framework + the script's own record), exit 0/1/2, `--vocab-sizes`, `--max-train-chars` smoke mode.
- `bpe_python` prerequisites (this experiment's code): the classic full-rescan BPE loop was infeasible at corpus scale (benchmark: 2048 vocab on 400k chars did not finish in 20 min), so `train()` now uses an **incremental pair-count loop that maintains exactly the pair counts the classic loop computes** (only words containing the merged pair are touched) with the same (frequency desc, pair asc) tie-break — regression tests assert merge-list equality against the classic loop re-implemented in the test file (6 random corpora + real Indic text). A `gpt2_style` pre-tokenizer (stdlib-only, faithful to GPT-2's regex boundaries) was added; default behavior is unchanged.

**Verification so far (sandbox):** suite 479 passed / 1 skipped, ruff clean; sweep e2e smoke on a fake frozen corpus (fake pins + FREEZE.json, no network) runs the real script end to end — all configurations gate PASS, per-language metrics present, tampered shard / manifest-count-mismatch / missing-dataset gates exit 2.

**Commands (PC):** see NEW_CHAT_START_HERE.md status log 2026-09-26 (harness entry).

- **Commands:** `python scripts/run_tokenizer_sweep.py --exp-id EXP-028` (full grid); smoke: `python scripts/run_tokenizer_sweep.py --exp-id EXP-028 --vocab-sizes 512,1024 --max-train-chars 200000 --no-record`
- **Expected:** exit 0 with `out/experiments/EXP-028/report.txt` (per-configuration table + per-language table) and `sweep.json` + 15 run records; exit 2 means the frozen input did not verify (stop and report, do not fix the corpus).
- **Stop condition:** report the exit code, the per-configuration table, and the per-language table. Do not choose a tokenizer.

**Results (2026-09-26, PC run; COMPLETE):** 15/15 grid cells + 5/5 supplement cells,
**all gate=PASS** (exit 0 on both runs). Verified by `scripts/summarize_sweep.py`
(per-run consistency checks over all 20 runs; merged summary at
`out/experiments/EXP-028/summary.txt`, input `6c43d12695f2ffaa…`).
Headline (held-out overall chars_per_token; higher = denser):

| Configuration | chars/token |
|---|---|
| `mark_aware-32768` (hf and py — identical) | 2.3054 |
| `mark_aware-16384` | 2.1108 |
| `mark_aware-8192` | 1.8909 |
| `mark_aware-4096` | 1.6620 |
| `mark_aware-2048` | 1.4333 |
| `hf-byte_level-32768` | 1.5971 |
| `py-gpt2_style-32768` | 1.5878 |

(small-vocab baseline cells: 1.5831 / 1.5464 / 1.4838 / 1.3891 for hf, 1.5696 / 1.5323 / 1.4702 / 1.3792 for py)

**Findings:**
- **Mark-aware is a clear winner:** +45 % held-out density over the GPT-2/ByteLevel
  baselines at 32k vocab (2.3054 vs 1.5971), consistent at every vocab size (matra
  hypothesis confirmed on the full corpus).
- **Cross-implementation validation:** `hf-mark_aware-X` and `py-mark_aware-X` produce
  *identical* held-out metrics at all 5 vocab sizes and across all 13 languages — two
  independent BPE implementations (pure Python; Rust `tokenizers`) agree on the full
  424,727-char held-out side. Consequence: the ranked top-2 list
  (hf-mark_aware-32768, py-mark_aware-32768) is one tokenizer in two implementations.
- **Per language:** mark_aware-32768 wins 11 of 13 languages; the baselines win `en`
  (3.9284 vs 2.2645 — English has no combining marks, GPT-2 boundaries are native) and
  `ur` (3.2352 vs 1.8791 — corpus Urdu is largely unvocalized, so there are few marks to
  protect).
- **Vocabulary size has diminishing returns:** 32768 vs 16384 = +9 % density.
- **Speed:** the HF path with the custom mark-aware pre-tokenizer trains each cell in
  ~27 s vs 142 s–2049 s for the pure-Python BPE (merging stays in Rust; only the
  boundary classification is Python).

**EXP-B hand-off (per D-038, no selection before EXP-B):** the literal top-2 is one
tokenizer twice, so the informative comparison is `mark_aware-32768` (winner) vs
`mark_aware-16384` (best *distinct* configuration; doubles as the vocab-size question),
each with ≥ 3 seeds, compared by bits-per-byte of an identical small model. The
hf/py implementation question is an engineering choice (encodings are identical;
HF format is the standard artifact). Plan pending founder approval.

**Artifacts (harness):** `scripts/run_tokenizer_sweep.py`, `src/frontier_ai/tokenization/sweep.py`, `src/frontier_ai/corpus/frontier_docs.py`, `tests/test_tokenizer_sweep.py`; pending: `out/experiments/EXP-028/` (PC).

**Supplement (D-039, implemented 2026-09-26 while the PC run was in flight):** the 5 previously out-of-grid `bpe_hf` + mark-aware cells are now implemented (`bpe_hf.MarkAwarePreTokenizer` via the `tokenizers` custom pre-tokenizer API; boundaries identical to `bpe_python`'s, byte-remapped by the built-in `ByteLevel(use_regex=False)`; save/load handled via a serializable placeholder + re-attach). To run them after the main sweep finishes (separate out dir, same frozen input):

    python scripts/run_tokenizer_sweep.py --exp-id EXP-028 --out out/experiments/EXP-028-hf-mark-aware --configs hf-mark_aware-2048,hf-mark_aware-4096,hf-mark_aware-8192,hf-mark_aware-16384,hf-mark_aware-32768

then merge both runs for review: `python scripts/summarize_sweep.py --sweep-dir out/experiments/EXP-028 out/experiments/EXP-028-hf-mark-aware`. Also new: `scripts/summarize_sweep.py` — a read-only verifier + ranked summarizer for a finished sweep (per-run consistency checks, ranked table, per-language matrix, TOP-2-for-EXP-B; exit 0/1/2).
* **Status update (2026-09-29, follow-up note; the record above is unchanged):**
  - **Status:** complete. The PC run finished on 2026-09-26 (see "Results … COMPLETE" above:
    15/15 grid cells + 5/5 supplement cells). Its top configurations went on to EXP-029, and
    it closed §37 step 5 (D-038, D-039).
  - This note exists because the status line at the top of this entry still said "in
    progress". The Lab OS snapshot exporter (`scripts/export_lab_state.py`) reads the last
    status line of each entry and caught the mismatch.

### EXP-029 — Tokenizer-vs-tokenizer small-model comparison (EXP-B) — MASTER_CONTEXT §37 step 6
**Date:** 2026-09-26 · **Status:** approved (founder: "approve EXP-B"), harness delivered, PC run pending

**Purpose:** EXP-028 (EXP-A) measured token *density* model-free. EXP-B answers the
remaining question before a selection: given the *same* small model, *same* data,
*same* budget, *same* seeds — which tokenizer's token sequence yields the better
held-out **bits-per-byte** (lower = better)? Per D-038, the tokenizer is selected
**after** this experiment.

**Cells (the informative top-2 from EXP-028):**
| Tokenizer | EXP-028 artifact | EXP-028 held-out chars/token |
|---|---|---|
| `mark_aware-32768` | `out/experiments/EXP-028/py-mark_aware-32768/seed-0000001337/tokenizer` | 2.3054 |
| `mark_aware-16384` | `out/experiments/EXP-028/py-mark_aware-16384/seed-0000001337/tokenizer` | 2.1108 |

The ranked top-2 of EXP-028 is `hf-mark_aware-32768` + `py-mark_aware-32768` — one
tokenizer in two implementations (identical held-out metrics at all 5 vocabs × 13
languages), so running both would double the cost for zero information. The second
cell is `mark_aware-16384`, the best *distinct* configuration; it doubles as the
vocab-size question (32k is +9 % denser at the token level — is it worth the larger
embedding table at the LM level?). The py artifacts are used (canonical loader, zero
dependency; hf ≡ py proven, so the implementation choice is an engineering decision
after the selection, not an experimental variable).

**Fixed for every cell (pre-registered, do not vary):**
* model: `configs/exp_b.json` — 4 layers / 4 heads / 128 embd / block 128 / RMSNorm /
  SwiGLU / learned positions / tied embeddings, dropout 0 (~9M params at 32k vocab);
* step budget: 1000 steps, **may be revised once, after the PC timing smoke, and the
  same budget is then used for all 6 cells**;
* seeds: 1337, 1338, 1339 (each varying both model init `train.seed` and data
  sampling `data.seed`);
* data: the frozen FrontierCorpus v1 — train side for training, held-out side for
  validation (identity re-gated by the same shared verification as EXP-A);
  documents in canonical doc_id order, encoded per-document and concatenated;
* determinism: `train.deterministic=true`, fp32, CPU.

**Metric:** held-out **bits-per-byte** (`best_bpb` of each cell; the trainer derives
it from the val loss × the exact UTF-8 byte count of the held-out side recorded in the
prepared data metadata). Per tokenizer: mean and std across its 3 seeds.

**Decision rule (pre-registered; the script applies it mechanically):**
1. winner = the tokenizer with the **lower mean** held-out bits-per-byte across seeds;
2. if `|mean_A − mean_B| < (std_A + std_B) / 2` (the gap is within seed noise) →
   **TIE → the smaller-vocabulary tokenizer wins** (deployment economy at equal quality;
   for this experiment that is `mark_aware-16384`);
3. the selection is recorded as **D-040 — Frontier Tokenizer v1** (MASTER_CONTEXT §37
   step 7) only after founder review of the table.

**Harness:**
* `scripts/prepare_exp_b_data.py` — input gates (shared
  `frontier_ai.corpus.verify_and_derive_frontier`, now also used by the sweep runner) +
  per-tokenizer encoding of both sides + document-aligned token files
  (`data.dataset.write_split_tokens`, exact per-split byte/char counts for bpb) + a
  data manifest with artifact fingerprints.
* `scripts/run_exp_b.py` — the 2×3 matrix as self-recording `scripts/train.py` runs
  (D-032: one record per cell, `EXP-0291`…`EXP-0296`), aggregate table, the
  pre-registered decision, `runs/report.txt`, outer record. `--smoke` runs one cell at
  300 steps and prints the estimated wall time for the full matrix on this CPU.

**Commands (PC, in order):**
1. `git pull --ff-only origin arena/01a0dc16-frontier-ai`
2. prepare:
   `.\.venv\Scripts\python scripts\prepare_exp_b_data.py --exp-id EXP-029 --tokens mark_aware-32768=out/experiments/EXP-028/py-mark_aware-32768/seed-0000001337/tokenizer,mark_aware-16384=out/experiments/EXP-028/py-mark_aware-16384/seed-0000001337/tokenizer`
3. timing smoke: `.\.venv\Scripts\python scripts\run_exp_b.py --smoke` → report the
   printed estimate; if ~2 h or less for the full matrix, continue; otherwise propose a
   smaller `--max-steps` (one revision, applied to all cells).
4. full matrix: `.\.venv\Scripts\python scripts\run_exp_b.py --seeds 1337,1338,1339 --max-steps <budget>`
   (paste back: the 6 `bpb=` lines + the final table + exit code).

**Path correction (2026-09-26, first PC attempt):** the seed directories written by
the sweep runner are zero-padded — `seed-0000001337`, per `SEED_DIR_TEMPLATE =
"seed-{seed:010d}"` (`src/frontier_ai/experiments/sweep.py`) — not `seed-1337`. The
first prep attempt on the PC failed on that path *after* the input gate passed
(`input verified: 6c43d12695f2ffaa…`), so only the `--tokens` paths above (and the
cells table) are corrected; the cells, fixed config, budget rule and decision rule
are unchanged.

**Verification (sandbox, pre-PC):** e2e smoke on the fake frozen corpus (prepare +
2-cell × 2-seed matrix with a tiny model) exits 0 with a decision line; a repeated cell
(same seed, fresh dir) reproduces the identical bits-per-byte (determinism); the
refactored sweep runner still passes all its e2e gates.

**Results (in progress — PC):**
- **Prep (2026-09-27):** gates passed (`input verified: 6c43d12695f2ffaa…`).
  Prepared data in `out/exp_b/EXP-029/`: `mark_aware-32768` → train 1,608,987
  tokens / val 184,233 tokens; `mark_aware-16384` → train 1,772,551 / val 201,215
  tokens; byte totals identical for both (train 9,368,283 / val 1,048,975 UTF-8
  bytes) — the bits-per-byte normalization base.
- **Timing smoke (2026-09-27):** `mark_aware-32768` seed 1337, 300 steps: 2205 s
  wall (~7.3 s/step steady state, ~1120 tok/s), held-out bpb 1.2747 at step 300
  (still improving). Runner estimate for the 1000-step budget: 6 cells ≈ 735 min
  (~12.3 h) — over the ~2 h stop condition.
- **Budget revision (the one permitted revision, applied to all cells):
  `--max-steps 150`** → ~1115 s/cell, 6 cells ≈ 111 min (~1.9 h). 150 steps ≈ 0.76
  passes over the train side (1 pass = 1,608,987 / 8192 ≈ 196 steps at
  eff_batch 8192); both tokenizers receive the identical budget, so the comparison
  stays symmetric.
- **Matrix results:** pending.
- **Matrix results (2026-09-27, completed — supersedes "pending" above):** run on the
  founder's PC, unattended, by the local agent following `PC_TASK_EXP_B.md` at the
  revised budget of 150 steps/cell (runner `scripts/run_exp_b_night.ps1` @ `c91ce6e`).
  Report `out/exp_b/EXP-029/runs/report.txt` (generated 2026-09-27T08:19:33Z):

  | Tokenizer | seed 1337 | seed 1338 | seed 1339 | mean bpb | std | tokens seen |
  |---|---|---|---|---|---|---|
  | `mark_aware-32768` | 1.4674 | 1.4410 | 1.4305 | **1.4463** | 0.0190 | 3,686,400 |
  | `mark_aware-16384` | 1.5903 | 1.5728 | 1.5707 | 1.5779 | 0.0108 | 3,686,400 |

  **DECISION (pre-registered rule, applied mechanically by the script):
  `mark_aware-32768`** — lower mean held-out bits-per-byte. Gap 0.1316 bpb vs the tie
  threshold (std_A + std_B)/2 = 0.0149 → the gap is ~8.8× the seed-noise band, so rule 2
  (tie → smaller vocab) does not apply. Every 32768 seed beats every 16384 seed (worst
  32768 cell 1.4674 < best 16384 cell 1.5707); 16384 is 9.1 % worse on mean bpb.
- **Verification (PC agent, from the files, not the log):** 6a night.log ends with
  `NIGHT RUN matrix COMPLETE`, exit code 0 — PASS; 6b report.txt has no FAIL and has
  the DECISION line — PASS; 6c all six per-cell records
  (`out/exp_b/EXP-029/runs/mark_aware-{32768,16384}/seed-{1337,1338,1339}/experiment.json`)
  have `execution.status == "success"` and `results.best_bpb` — PASS; 6d the six
  report values equal the six record values — PASS. Sandbox cross-check: mean, std, gap
  and tie threshold recomputed independently from the six values — identical.
- **Determinism evidence:** cell `mark_aware-32768` / seed 1337 produced best_bpb
  1.4674 in three separate processes (two aborted night attempts and the completed
  run) — identical to 4 decimals.
- **Scope notes (read with the decision; they do not change it):**
  1. *Undertrained budget.* 150 steps ≈ 0.76 passes over the train side; the 300-step
     smoke reached 1.2747, so curves were still falling. The result answers "which
     tokenizer is better at this equal budget", as pre-registered.
  2. *Equal steps = equal tokens, not equal bytes.* At 1,228,800 tokens per cell the
     32768 model saw ≈ 7.15 MB of text vs ≈ 6.49 MB for 16384 (5.82 vs 5.29 train
     bytes/token). Seeing more text per unit of compute is precisely the advantage a
     denser tokenizer offers, so this is part of what is measured, not a leak.
  3. *Embedding share at toy scale.* With tied embeddings at n_embd 128, the 32768
     model has 5,260,416 parameters vs ≈ 3,163,264 for 16384 (the non-embedding part,
     1,066,112, is identical). The pre-registered question explicitly included "is the
     larger embedding table worth it at the LM level" — at this scale, yes. At
     production width the embedding share is far smaller, which shrinks this cost.
- **Status:** complete. The selection becomes **D-040 — Frontier Tokenizer v1** only
  after founder review of this table (rule 3).
- **Selection (2026-09-27):** founder reviewed the table and approved →
  **D-040 — Frontier Tokenizer v1 = `mark_aware-32768`** (DECISIONS.md).

### EXP-030 — Freeze Frontier Tokenizer v1 (D-040 follow-up)
**Date:** 2026-09-27 · **Status:** in progress — harness delivered (sandbox); PC freeze pending
(founder: "approve freeze plan", including a one-folder commit + push by the PC agent)

**Purpose:** the D-040 tokenizer existed only on the founder's PC, under git-ignored
`out/`. EXP-030 moves it into a tracked, hash-verified location and proves the copy is
the exact artifact that won EXP-B, so every downstream consumer (evaluation harness,
training) loads one provable object instead of a loose path.

**Design (no experimental variable — an identity-preserving copy behind gates):**
* `scripts/freeze_tokenizer.py` — gates on the SOURCE artifact before anything is written:
  **A identity** — directory fingerprint == the `artifact_sha256` EXP-029 recorded in
  `out/exp_b/EXP-029/manifest.json` (same algorithm, test-locked);
  **B structure** — vocab 32,768, 32,512 merges, `mark_aware`, no special tokens;
  **C counts** — re-encoding the frozen FrontierCorpus v1 (identity re-gated as in
  EXP-A/B) reproduces EXP-029's train 1,608,987 / held-out 184,233 tokens on the same
  corpus fingerprint; **D lossless** — every document round-trips exactly.
  Only if A–D pass: byte-exact copy to `tokenizers/frontier-tokenizer-v1/tokenizer/` +
  `FREEZE.json` (hashes, lineage, gate results, 15 golden samples with their token ids,
  change policy), then re-verification of the copy through the downstream loader.
  Never overwrites an existing freeze (exit 2).
* `src/frontier_ai/tokenization/frozen.py` — `load_frontier_tokenizer()` refuses a
  tokenizer whose file set, byte hashes, directory fingerprint, structure or golden-sample
  encodings differ from `FREEZE.json`.
* `.gitattributes` — `tokenizers/** -text`: `save_json` writes CRLF on Windows and Git
  for Windows would convert it to LF on commit, silently breaking every raw-byte hash on
  Linux. Git now stores the folder byte-for-byte.
* PC handoff: `PC_TASK_TOKENIZER_FREEZE.md` (run, verify, commit + push exactly the 2
  frozen files, write `out/tokenizer_freeze/EXP-030/FREEZE_REPORT.txt`).

**Verification (sandbox):** `tests/test_freeze_tokenizer.py` — e2e freeze on the fake
frozen corpus with a CRLF (Windows-style) artifact: 4 gates PASS, frozen bytes ==
source bytes == EXP-029-style fingerprint, token counts equal, golden samples reproduce
and round-trip; tampered artifact → gate A FAIL, nothing written; wrong expected vocab →
gate B FAIL; existing freeze never overwritten; loader catches CRLF→LF conversion, a
stray file and altered golden ids; recorded (default) mode writes a successful
experiment record. Full suite: 502 collected, all pass except 2 expected skips (the
real-v1 test until the PC commit lands; the smoke-corpus fetch test). ruff clean.

**After the PC commit:** the skipped test `test_frontier_tokenizer_v1_is_frozen_and_verifies`
becomes active — Linux must reproduce the byte hashes and the PC-recorded golden ids
(cross-platform determinism). Then EXP-030 closes and **D-041** (tokenizer v1 frozen;
change policy) is recorded.

**Results (2026-09-27, completed — supersedes "PC freeze pending" above):** the PC agent
ran `PC_TASK_TOKENIZER_FREEZE.md`; commit **`fc7e8d8`** adds exactly 2 files
(`tokenizers/frontier-tokenizer-v1/FREEZE.json`, `tokenizer/bpe_python.json`).
* Gates on the PC (Windows 10, Python 3.13.15), from `FREEZE.json`:
  **A** directory fingerprint `b39afa08fa86b0d1…` == EXP-029 record — PASS;
  **B** vocab 32,768 / 32,512 merges / `mark_aware` / no special tokens — PASS;
  **C** train 1,608,987 == 1,608,987 and held-out 184,233 == 184,233 tokens on corpus
  `6c43d12695f2ffaa…` (same as EXP-029) — PASS; **D** 34,011 documents round-trip
  losslessly — PASS (encode + decode 41.8 s).
* Artifact: `bpe_python.json` 1,273,642 bytes, sha256
  `303db552912d4efa507196693ff2b039affc06aa06e8f22844ca3fec24584253`; dir_sha256
  `b39afa08fa86b0d1ccf7b7e1b59df305b5678bdd59061c6fae01245061321144`; lineage EXP-028
  `py-mark_aware-32768` seed 1337, trained on 3,809,046 chars of the FrontierCorpus v1
  train side.
* Byte-exact storage confirmed: `git ls-files --eol` → `i/crlf w/crlf attr/-text` for
  the artifact (Windows line endings preserved, as designed).
* **Cross-platform determinism (sandbox, Linux x86_64):** `sha256sum` of the checked-out
  file equals the PC value; `load_frontier_tokenizer()` accepts it (file set, byte hash,
  dir fingerprint, structure, all 15 golden samples); every golden sample's Linux token
  ids equal the ids recorded on Windows and round-trip exactly. The previously skipped
  `test_frontier_tokenizer_v1_is_frozen_and_verifies` now runs and passes.
* **Status:** complete → **D-041** (Frontier Tokenizer v1 frozen; change policy).

### EXP-031 — Evaluation harness v1: full held-out re-evaluation of the EXP-B models (MASTER_CONTEXT §37 step 8)
**Date:** 2026-09-27 · **Status:** in progress — harness built and tested (sandbox); PC run pending
(founder: "Approve eval harness plan + roadmap sync", 2026-09-27)

**Purpose:** EXP-029's bits-per-byte figures were *training-time sampled estimates*
(51,200 of the ~184k held-out tokens). Step 8 needs a harness that scores **every**
held-out token under identical, recorded conditions, so model comparisons (the D-040
tokenizer evidence first; architecture ablations later, with approval) rest on exact,
reproducible numbers. This experiment builds harness v1 and applies it to the 6 EXP-029
checkpoints. There is no new training and no new model.

**Design (harness v1.0.0, `src/frontier_ai/evaluation/`):**
* **Protected suite `frontier-heldout-v1`** (`scripts/build_eval_suite.py` →
  `evals/suites/frontier-heldout-v1/SUITE.json`): the FrontierCorpus v1 held-out side
  (3,427 docs, 13 languages) as **fingerprints only** (doc id, language, source, sha256,
  sizes; no text), in derived order, bound to the corpus `content_sha256`, with a
  self-checking suite fingerprint. It is built once and never replaced (exit 2).
  `find_exact_overlap(texts, suite)` guards future training data against the suite.
* **Scoring (`scripts/eval_report.py`):** documents are encoded individually and
  concatenated in suite order, exactly like the EXP-B validation stream. Non-overlapping
  windows of the model's `block_size` score every token except stream token 0 exactly
  once, in fp32 on CPU. Per-token bits are attributed to documents; document 0 is
  context-only and excluded from all statistics. Stream (not document-isolated) scoring
  is deliberate: isolated scoring gives each document a context-free first token whose
  cost depends on vocabulary size and would bias a tokenizer comparison.
* **Metrics:** bits-per-byte (primary; tokenizer-independent), bits-per-character and
  bits-per-token, overall, per language, per script and per source, with seeded
  document-level bootstrap 95% intervals (1,000 resamples, seed 0; not computed for
  per-source). Domain: not available (no domain labels in the pilot corpus).
* **Checks:** suite fingerprint and corpus binding (exit 2); tokenizer vocab == checkpoint
  vocab (exit 2); **data identity**, meaning the re-encoded stream is compared to the
  checkpoint's own dataset validation split → PASS / FAIL (exit 1) / NOT CHECKED;
  **contamination**: exact-hash and 13-word n-gram overlap of the evaluated documents
  against the FrontierCorpus v1 train side.
* **Provenance and reproducibility:** `report.json` (schema `frontier-eval-report-v1`)
  records model/config sha256, step, parameters, tokenizer dir fingerprint and
  frozen-v1 match, suite fingerprint, corpus fingerprint, protocol and thread count, and
  the training-time estimate for contrast. `scores_sha256` hashes the per-document
  scores, and a re-run must reproduce it bit for bit. Also written: `report.txt` (human),
  `per_document.jsonl`, and a self-recorded experiment record.
* **Comparison (`scripts/eval_compare.py`):** refuses to compare reports across harness
  versions, suites, protocols or document sets. It reports per-group mean ± std across
  seeds, and a paired document bootstrap on per-document bits (averaged within each
  group); delta = B − A with a verdict, overall and per language.
* **Limitations (v1):** no sliding-window/stride option (every token gets up to one
  block of context; the same for all compared models, so comparisons are fair while
  absolute numbers are protocol-specific); CPU fp32 results are reproducible for the
  same thread count; no downstream benchmarks or human evaluation (§37 step 15).

**Verification (sandbox):** `tests/test_eval_harness.py`, 13 tests on the fake frozen
corpus with tiny seeded models: per-token scoring equals an independent computation
through the model's own loss; per-document bits sum to that total; data identity PASS;
scored tokens = stream − 1; languages add up to overall; CI brackets the point; two runs
give byte-identical `per_document.jsonl` and equal `scores_sha256`; wrong-vocab tokenizer
→ exit 2; altered suite → exit 2; mismatched dataset → data identity FAIL (exit 1);
suite builder is idempotent and refuses replacement; compare delta = difference of the
group scores, and a self-comparison gives 0; contamination detects planted exact and
13-gram overlaps; recorded mode writes a successful record; the publisher copies only
the small files. Full suite: **514 passed, 1 skipped** (the smoke-corpus fetch test);
ruff clean.

**PC run (`PC_TASK_EVAL_EXP031.md`):** build the suite; score the 3 `mark_aware-32768`
checkpoints (`--tokenizer frozen`) and the 3 `mark_aware-16384` checkpoints (EXP-028
artifact); re-run one checkpoint, and also score it with the original EXP-028 32768
artifact, where both must reproduce `scores_sha256`; compare 32768 vs 16384; commit
only the suite file plus the small result files (`evals/results/EXP-031/`, via
`scripts/publish_eval_results.py`).

**Results:** pending the PC run.

**Results (2026-09-27, completed; supersedes "PC run pending" above).** The PC agent ran
`PC_TASK_EVAL_EXP031.md` (attempt 1 stopped at step 1 on an unrelated local edit to
`P004B-tokenizer-corpus-summary.html`, which was preserved to `out\eval\EXP-031\local-change-backup\`
and then restored; attempt 2 passed steps 0–4, then `eval_compare.py` crashed while *printing*
U+2212 on the Windows cp1252 console after its files were written, fixed in `ffbd4e6` with a
regression test; the resume ran steps 5–7). Commit **`0fa5b22`** adds exactly 25 files:
`evals/suites/frontier-heldout-v1/SUITE.json` and `evals/results/EXP-031/**` (report.json,
report.txt, experiment.json for each of the 6 models, 2 reproducibility reports, compare.json and
compare.txt). No `per_document.jsonl` is committed. Scoring ran at commit `d0fa01b`; the only
untracked item was `evals/`. Environment: Windows 10, Python 3.13.15, torch 2.14.0+cpu, 2 threads.
* **Suite:** `frontier-heldout-v1`, 3,427 docs, 13 languages, 1,048,975 bytes, 424,727 chars
  (document text only; the manifest's 1,052,401 / 428,153 also counts the 3,426 newlines joining
  the documents in the single held-out shard, so the two figures agree exactly). Fingerprint
  `a488b2aa41f9e861fa279a67e1fb6d98a610b4e8eebe554b9cfad262fc628a2f`, bound to corpus
  `6c43d12695f2ffaa…` (equals the manifest). It verifies on Linux through `load_suite`.
* **Coverage and identity:** every report has data identity **PASS**, i.e. the re-encoded stream
  equals the checkpoint's own validation split: 32768 → 184,233 tokens (184,232 scored), 16384 →
  201,215 tokens (201,214 scored). All 6 reports have the same suite fingerprint; the 32768 reports
  have `matches_frozen_v1: true`; all checkpoints are at step 150.
* **Exact held-out bits-per-byte** (document 0 context-only; 3,426 documents; 95% document
  bootstrap CI):

  | model | exact bpb [95% CI] | bits/char | bits/token | EXP-029 sampled estimate | exact − estimate |
  |---|---|---|---|---|---|
  | mark_aware-32768 seed 1337 | 1.4619 [1.4454, 1.4792] | 3.6106 | 8.3238 | 1.4674 | −0.0055 |
  | mark_aware-32768 seed 1338 | 1.4406 [1.4234, 1.4586] | 3.5580 | 8.2024 | 1.4410 | −0.0004 |
  | mark_aware-32768 seed 1339 | 1.4400 [1.4230, 1.4579] | 3.5565 | 8.1990 | 1.4305 | +0.0095 |
  | mark_aware-16384 seed 1337 | 1.5834 [1.5654, 1.6022] | 3.9107 | 8.2547 | 1.5903 | −0.0069 |
  | mark_aware-16384 seed 1338 | 1.5760 [1.5576, 1.5944] | 3.8924 | 8.2161 | 1.5728 | +0.0032 |
  | mark_aware-16384 seed 1339 | 1.5760 [1.5582, 1.5938] | 3.8923 | 8.2157 | 1.5707 | +0.0053 |

  Group means (3 seeds): **32768: 1.4475 ± 0.0125**, where the estimate was 1.4463 ± 0.0190;
  **16384: 1.5785 ± 0.0043**, where the estimate was 1.5779 ± 0.0108.
* **Paired comparison** (`compare.txt`): delta (16384 − 32768) = **+0.1310 bpb, 95% CI
  [+0.1280, +0.1339]**. The 16384 tokenizer is 9.05% worse, and the interval excludes 0. Every
  32768 seed (worst 1.4619) beats every 16384 seed (best 1.5760). **All 13 languages** favour
  32768 and every interval excludes 0. The smallest gap is hi (+0.0976) and the largest is en
  (+0.2979; English is a small, high-bpb slice at 3.38 vs 3.68).
* **Contamination** (evaluated docs vs the 30,584 train-side docs): 0 exact duplicates; 0 of
  1,545 eligible documents share a 13-word n-gram with training. 1,881 documents are too short
  to test at n = 13, which is a limitation of the n-gram check; the exact-hash check covers them.
* **Reproducibility:** rerunning 32768 seed 1337, and scoring it with the original EXP-028 artifact
  instead of the frozen copy, both give `scores_sha256`
  `7634729954028b33db50c827595222b8bbf5ac4e39d547666a1cc4966ee79f46`, identical to the original
  (bit-identical per-document scores; the frozen v1 is scoring-equivalent to the artifact that won EXP-B).
* **Sandbox verification:** full suite **515 passed, 1 skipped**; ruff clean. The numbers above
  were read by script from the committed JSON files.

**Conclusions:**
1. The exact, full held-out evaluation **confirms D-040**, now with a paired confidence interval
   and per-language evidence: `mark_aware-32768` is better overall and in every language.
2. The EXP-029 sampled estimates were unbiased on average (group means within +0.0012 / +0.0006)
   but noisy per model (up to ±0.0095). They also overstated the seed-to-seed spread (0.0190 vs
   0.0125; 0.0108 vs 0.0043). This is why comparisons from now on use the harness.
3. Caveats (unchanged from D-040): tiny CPU models (150 steps). The 32k model's larger embedding
   table is part of the comparison (5.26 M vs 3.16 M parameters). The evidence is on the pilot
   corpus only; per-domain results are not available (no labels). Absolute numbers are specific
   to this protocol (non-overlapping 128-token windows); comparisons are valid only within the
   same harness version, suite and protocol, which `eval_compare.py` enforces.
* **Status:** complete → **D-042**.

### EXP-032 — Step 9 architecture screening v1 (MASTER_CONTEXT §37 step 9)
**Date:** 2026-09-27 · **Status:** in progress: runner built and tested (sandbox); PC night run pending
(founder: "I approve step 9 plan", 2026-09-27)

**Purpose (why this serves the mission):** before any GPU money is spent (steps 10–12), choose a
*provisional* architecture for the first GPU bring-up from measured evidence on our own data
and tokenizer, instead of copying a popular design. Each change is tested one at a time on
the fixed EXP-B model and graded by harness v1 (D-042).

**Honest scale caveat (pre-stated):** the model has 5,260,416 parameters, of which 4,210,688
(80%) are the tied 32,768 × 128 embedding plus the 128-position table. The transformer
body, which the ablations change, is 1,049,728 parameters. Many transformer modifications
fail to transfer across implementations and scales (Narang et al., 2021; see the step 9
plan), so every result here is **screening evidence**, to be re-validated in the GPU
scaling experiments (step 11).

**Pre-registered spec:** `configs/ablations/EXP-032.json` (committed before any result).
It fixes: base `configs/exp_b.json`; data `out/exp_b/EXP-029/mark_aware-32768.bin`;
frozen tokenizer v1; 150 steps; seeds 1337/1338/1339; lr 3e-3. The **final** checkpoint
(`last/`) is graded, because the trainer's `best/` is selected on the held-out split, which is the
evaluation suite. For EXP-029 this made no difference, since every graded `best/` was step 150.

| group | change | body params | purpose |
|---|---|---|---|
| baseline | none (EXP-029 `mark_aware-32768` cells, reused only if their saved config verifies) | 1,049,728 | reference |
| rope | `model.pos=rope` | 1,049,728 (no position table) | positions that extend past `block_size` |
| gelu | `model.ffn=gelu`, `ffn_mult=6.0` | 1,049,728 (exact match) | confirm SwiGLU |
| layernorm | `model.norm=layernorm` | 1,050,880 | confirm RMSNorm |
| gqa2 | `model.n_kv_head=2` | 984,192 | halve the inference KV cache |
| lr-0.0015, lr-0.006 | learning rate, seed 1337 | 1,049,728 | is 3e-3 near-optimal? |
| repro-baseline | retrain baseline seed 1337 | 1,049,728 | identical weights? |

**Decision rules (pre-registered):**
- **Superiority:** a variant is BETTER only if the paired 95% CI of (variant − baseline) is
  below 0 **and** every variant seed beats every baseline seed. WORSE is the mirror image.
  Otherwise the verdict is NO DETECTABLE DIFFERENCE AT THIS SCALE.
- **GQA (efficiency change):** ACCEPTABLE if the CI upper bound is ≤ 0.010 bpb.
- **Learning rate:** LR-CONFOUNDED if an alternative LR improves the seed-1337 baseline by more
  than the baseline's seed standard deviation.
- **Reproducibility:** IDENTICAL if all weight tensors match exactly.
- **Adoption:** only BETTER/ACCEPTABLE results become candidates for **D-043** (provisional
  architecture for GPU bring-up), after founder review.

**Implementation:** `scripts/run_arch_ablation.py` validates the spec, checks the baseline's
saved config before reusing it, then trains each cell as its own self-recording `train.py`
run, grades each `last/` immediately with `eval_report.py`, and compares with `eval_compare.py`.
It applies the rules and writes `SUMMARY.txt`/`summary.json` plus an outer record. It is
restartable (finished cells are skipped and matching reports are not regraded).
`scripts/run_arch_ablation_night.ps1` is the unattended wrapper (logs to file, one automatic
retry). `publish_eval_results.py` now also publishes `summary.json`/`SUMMARY.txt`. No model
or trainer code changed: the trainer, `gpt.py`, `config.py` and `configs/exp_b.json` are
unchanged since EXP-029 (`b4f864d`), and the trainer has saved `last/` since `e467eb0`.
The reused baseline is therefore expected to reproduce EXP-031's per-model scores, and the
repro cell to reproduce its weights.

**Verification (sandbox):** `tests/test_arch_ablation.py`, 13 tests. The real EXP-032 spec
validates and plans 18 cells; GELU at `ffn_mult` 6.0 has exactly the baseline's 5,260,416
parameters; bad specs are rejected (rule, reserved name, unknown override, duplicate,
`best` checkpoint, missing margin, lr seed). Rule and weight-comparison unit tests pass.
End to end on the fake corpus with tiny models: every cell is trained and graded;
delta = the difference of group means; half the KV cache for MQA; retraining reproduces
weights **IDENTICAL**; the final checkpoint is graded; a restart retrains/regrades nothing; a
second experiment reuses the verified baseline; a mismatched budget is refused for reuse.

**PC run:** `PC_TASK_ARCH_EXP032.md`, about 5.5–6.5 h overnight (15 models to train if the
baseline is reused, plus 18 gradings).

**Results:** pending.

**Results (2026-09-28, completed; supersedes "PC night run pending" above).** The PC ran
`PC_TASK_ARCH_EXP032.md` unattended. The outer record succeeded at commit `788b751` (clean
tree). Commit **`4027c98`** adds exactly 80 files under `evals/results/EXP-032/`, with no
`per_document.jsonl`. Numbers below were read by script from the committed JSON.
* **Integrity:** all 3 baseline seeds were **reused** from EXP-029 after config verification,
  and their harness scores equal EXP-031 bit for bit (`scores_sha256` × 3). The retrained
  baseline (repro cell) has weights **IDENTICAL** to EXP-029 seed 1337 (36/36 tensors,
  max |diff| 0; bpb 1.4619 = 1.4619). All 18 reports have data identity PASS, the frozen v1
  tokenizer, and step 150. No failed cells.
* **Results** (final checkpoints, 150 steps, lr 3e-3, exact held-out bpb; delta = variant −
  baseline with paired 95% CI):

  | group | seeds 1337 / 1338 / 1339 | mean ± std | delta [95% CI] | pre-registered verdict |
  |---|---|---|---|---|
  | baseline | 1.4619 / 1.4406 / 1.4400 | 1.4475 ± 0.0125 | — | — |
  | rope | 1.4119 / 1.3650 / 1.3345 | 1.3705 ± 0.0390 | −0.0770 [−0.0787, −0.0753] (−5.3%) | **BETTER** (13/13 languages better) |
  | gelu (ffn_mult 6) | 1.3892 / 1.4221 / 1.4297 | 1.4137 ± 0.0215 | −0.0339 [−0.0346, −0.0332] (−2.3%) | **BETTER** (13/13 languages better) |
  | layernorm | 1.4798 / 1.4844 / 1.4547 | 1.4729 ± 0.0160 | +0.0254 [+0.0248, +0.0260] (+1.8%) | NO DETECTABLE DIFFERENCE (seeds overlap; 13/13 languages worse on CI) |
  | gqa2 | 1.3884 / 1.4744 / 1.4185 | 1.4271 ± 0.0437 | −0.0204 [−0.0210, −0.0197] (−1.4%) | **ACCEPTABLE** (margin 0.010; superiority: seeds overlap) |

  Learning-rate check (seed 1337, baseline): lr 1.5e-3 → 1.4916; **3e-3 → 1.4619**; **6e-3 →
  1.4049** (−0.0571, 4.6× the baseline seed std) ⇒ **LR-CONFOUNDED** under the pre-registered rule.
* **Interpretation (what the evidence does and does not support):**
  1. **The baseline learning rate is too low for this 150-step budget.** Doubling it gains
     about as much (−0.057 on seed 1337) as RoPE does (−0.077) and more than GELU (−0.034). In an
     under-tuned, short run, changes that simply make the model learn faster can look
     better. By the pre-registered rule, the BETTER verdicts hold **at lr 3e-3 only**, and the
     comparison must be repeated at a better learning rate before adoption.
  2. **RoPE** is the strongest result (every seed, every language). It agrees with prior
     large-scale evidence that relative position methods beat learned absolute positions
     (Narang et al., 2021, appendix), and it is needed for context beyond `block_size`. It is
     the leading candidate.
  3. **GELU beating SwiGLU contradicts the larger-scale literature**, where GLU variants
     (SwiGLU) improve over plain activations (Narang et al., 2021). It is treated as
     **suspect until re-tested** at a tuned learning rate. It is not adopted.
  4. **LayerNorm vs RMSNorm:** no detectable difference by the rule (it tends worse). The
     default RMSNorm stays (cheaper, not worse).
  5. **GQA (2 KV heads)** halves the inference KV cache (1,024 → 512 values per token) with no
     measurable quality cost. It is an efficiency candidate.
  6. **Seed variation dominates the uncertainty.** The paired document CIs are about ±0.001
     wide, but seed standard deviations are 0.012–0.044. The document bootstrap does not see
     training randomness, which is why the seed-separation criterion was pre-registered.
     Future ablations need that criterion, and more seeds where affordable.
  7. **Timing is not usable for efficiency conclusions.** Per-step times were confounded by
     run order, heat and a ~3 h pause of the PC after the lr-0.0015 cell (its training took
     18 min according to its record; the runner's wall clock includes the pause). Results are
     unaffected (deterministic; repro IDENTICAL).
* **Tooling fix:** the PC recorded spec sha `2a7048f2…` while the repo file hashes to
  `f8c13dad…`. The difference is Git for Windows' CRLF checkout (the CRLF version of the
  committed file hashes exactly to `2a7048f2…`), so the spec was unchanged.
  `run_arch_ablation.py` now fingerprints the spec over LF-normalised bytes, with a
  regression test.
* **Status:** complete. **D-043 is deferred**: under the pre-registered adoption rule no
  architecture change is adopted while the comparison is LR-confounded. The next step is a
  confirmation experiment at better learning rates (proposed as EXP-033; needs founder
  approval).

### EXP-033 — Step 9 architecture confirmation at two higher learning rates (MASTER_CONTEXT §37 step 9)
**Date:** 2026-09-28 · **Status:** in progress: spec, runner v2 and one-command PC script built
and tested (sandbox); PC night run pending (founder: "I approve EXP-033", 2026-09-28)

**Purpose (why this serves the mission):** EXP-032 was LR-CONFOUNDED (lr 6e-3 improved the
seed-1337 baseline by 0.0571 bpb, more than the seed spread), so its verdicts at lr 3e-3 may
reflect the learning rate rather than the architecture. EXP-033 re-tests the EXP-032 candidates
as one combined architecture at **two** higher learning rates and adopts a change only if it
wins at **both**, so the evaluation suite never selects a learning rate. The outcome is
**D-043**, a *provisional* architecture for the first GPU bring-up (step 10). The EXP-032
scale caveat applies unchanged: screening evidence only, to be re-validated at GPU scale
(step 11).

**Pre-registered spec:** `configs/ablations/EXP-033.json` (schema `frontier-arch-ablation-v2`,
committed before any result; LF-normalised sha256 `7f9e7af7901bdab9…`). Same base config, data,
frozen tokenizer, 150 steps, seeds 1337/1338/1339 and final-checkpoint grading as EXP-032.

| arm | change vs EXP-B | body params | KV values / token |
|---|---|---|---|
| baseline | none | 1,049,728 | 1,024 |
| rope-gqa2 | `model.pos=rope`, `model.n_kv_head=2` (SwiGLU, RMSNorm kept) | 984,192 | 512 |
| rope-gqa2-gelu | rope-gqa2 + `model.ffn=gelu`, `ffn_mult=6.0` | 984,192 (exact match) | 512 |

Each arm × lr ∈ {6e-3, 1e-2} × 3 seeds = 18 cells. The EXP-032 cell `lr-0.006/seed-1337` is
exactly `baseline` at lr 6e-3, seed 1337, and is reused only if its saved config verifies
(it is re-graded under EXP-033). So 17 cells are trained, about 7.5–8.5 h on the PC (CPU only).

**Comparisons and decision rules (pre-registered):**
- `candidate-vs-baseline` (rope-gqa2 vs baseline) and `gelu-vs-swiglu` (rope-gqa2-gelu vs
  rope-gqa2).
- **Per learning rate:** the EXP-032 superiority rule. BETTER needs the paired 95% CI of
  (b − a) below 0 **and** every b seed beating every a seed; WORSE is the mirror image;
  otherwise NO DETECTABLE DIFFERENCE AT THIS SCALE.
- **Divergence:** a cell whose training stops on a non-finite loss is DIVERGED. It is recorded
  as a result (never retrained, not a failure) and counts as the worst possible score. If only
  one arm of a comparison has diverged seeds at a learning rate, that arm loses there. If both
  arms do, that learning rate is UNSTABLE and gives no verdict.
- **Adoption:** ADOPT only if BETTER at every learning rate that is not UNSTABLE, with at least
  one stable learning rate; fewer than all is stated as weaker evidence. Otherwise NOT ADOPTED.
- **D-043 mapping:** candidate ADOPT → RoPE + GQA-2; otherwise keep learned positions and 4 KV
  heads (GQA may be revisited purely for inference efficiency). gelu-vs-swiglu ADOPT → GELU;
  otherwise SwiGLU stays (literature prior). RMSNorm stays (EXP-032). The learning-rate trend is
  descriptive only. The step-10 GPU runs must sweep the learning rate again at their own scale.

**Implementation:** `scripts/run_arch_ablation.py` now also accepts schema v2 (arms × learning
rates × seeds, pairwise comparisons, single-cell reuse, DIVERGED status). The v1 path used for
EXP-032 is unchanged. `scripts/publish_eval_results.py` also publishes `DIVERGED.json`.

The founder's PC agent (GitHub Copilot) was about to reach its usage limit, so the PC steps
(checks, preflight, run with one retry, publish, scoped commit of `evals/results/EXP-033/` only,
push, report) are in **one script, `scripts/run_night_unattended.ps1`**. It needs no agent: the
founder types one line in a terminal. `PC_TASK_ARCH_EXP033.md` documents it. There are 21 new
tests (spec, validation, the per-LR and adoption rules, a v2 end-to-end run with a forced
divergence, restart and reuse, and static safety checks of the PowerShell script). PowerShell
itself could not be run in the sandbox (downloads blocked). The script therefore uses only
constructs that already worked on the PC (EXP-029/EXP-032) and is ASCII-only.

**Results (2026-09-28, completed; supersedes "PC night run pending" above).** The founder ran
`scripts/run_night_unattended.ps1` with one typed line and no chat agent. It worked on its
first run: every check passed, and it published, committed and pushed on its own. The outer
record succeeded at commit `0eaf971` (clean tree). Commit **`5d7a0c7`** adds exactly 82 files
under `evals/results/EXP-033/`, with no `per_document.jsonl`. Numbers below were read by script
from the committed JSON.
* **Integrity:**
  - The EXP-032 cell `lr-0.006/seed-1337` was **reused** after config verification. Its
    EXP-033 re-grade has the same `model_sha256` and the same bpb (1.4049) as its EXP-032
    report.
  - All 17 trained cells succeeded at `0eaf971` (clean), with 1,047–1,950 s of training each
    (5.34 h in total; the whole run took 5 h 46 min).
  - All 18 reports have data identity PASS, the frozen v1 tokenizer and step 150.
  - No DIVERGED cells, no failures.
* **Results** (final checkpoints, 150 steps, exact held-out bpb, lower = better):

  | arm | lr 6e-3: seeds 1337 / 1338 / 1339 | mean ± std | lr 1e-2: seeds 1337 / 1338 / 1339 | mean ± std |
  |---|---|---|---|---|
  | baseline | 1.4049 / 1.5050 / 1.3779 | 1.4293 ± 0.0669 | 1.4566 / 1.4008 / 1.4269 | 1.4281 ± 0.0279 |
  | rope-gqa2 | 1.3851 / 1.3850 / 1.3719 | 1.3806 ± 0.0076 | 1.3960 / 1.3442 / 1.3466 | 1.3623 ± 0.0293 |
  | rope-gqa2-gelu | 1.3621 / 1.4080 / 1.3359 | 1.3687 ± 0.0365 | 1.3795 / 1.3936 / 1.3679 | 1.3803 ± 0.0129 |

  | comparison | lr 6e-3: delta [95% CI] → verdict | lr 1e-2: delta [95% CI] → verdict | decision |
  |---|---|---|---|
  | candidate-vs-baseline (rope-gqa2 − baseline) | −0.0486 [−0.0497, −0.0476] → NO DETECTABLE DIFFERENCE (13/13 languages better on CI; seeds overlap: baseline seed 1339, 1.3779, beats candidate seeds 1337 and 1338, 1.3851 and 1.3850) | −0.0658 [−0.0676, −0.0641] → **BETTER** (13/13 languages) | **NOT ADOPTED** |
  | gelu-vs-swiglu (rope-gqa2-gelu − rope-gqa2) | −0.0120 [−0.0127, −0.0113] → NO DETECTABLE DIFFERENCE (11 languages better, hi worse) | **+0.0181** [+0.0168, +0.0192] → NO DETECTABLE DIFFERENCE (11 languages worse, mr better) | **NOT ADOPTED** |

  The learning-rate trend (descriptive only) for mean bpb from lr 6e-3 to lr 1e-2:
  - baseline: 1.4293 → 1.4281, flat;
  - rope-gqa2: 1.3806 → 1.3623;
  - rope-gqa2-gelu: 1.3687 → 1.3803.
* **Interpretation (what the evidence does and does not support):**
  1. **By the pre-registered rule, nothing is adopted.** Under the pre-registered D-043
     mapping, the provisional architecture therefore stays the EXP-B baseline: learned
     positions, 4 KV heads, SwiGLU (ffn_mult 4), RMSNorm, tied embeddings. This is not
     overridden after seeing the results; that is the point of pre-registration.
  2. **RoPE (+GQA-2) is still the leading candidate, but it did not clear the bar.** Its mean
     bpb was lower than the baseline's in every comparison so far:
     - EXP-032 at lr 3e-3: RoPE alone −0.077, GQA-2 alone −0.020;
     - EXP-033 at lr 6e-3 and 1e-2: RoPE + GQA-2 −0.049 and −0.066.

     It was better in all 13 languages each time. At lr 6e-3 the rule failed only because one
     baseline seed (1339) was unusually good. That is supportive evidence, **not proof at this
     scale**, so RoPE + GQA-2 is the first architecture question to re-test at GPU scale
     (step 11).
  3. **The EXP-032 GELU result did not replicate.** The GELU − SwiGLU difference changed sign
     between learning rates (−0.012 and +0.018). That is consistent with EXP-032's GELU "win"
     being a learning-rate artefact, as suspected. SwiGLU stays, in line with the literature
     (Narang et al., 2021).
  4. **Seed noise decides everything at this scale.**
     - Seed standard deviations are 0.008–0.067 bpb, while the paired document CIs are about
       ±0.001 wide.
     - With 3 seeds, a single outlier seed can block a verdict.
     - The baseline at lr 6e-3 had the largest spread (seed 1338: 1.5050); the candidate the
       smallest (0.0076). That *hints* at better training stability with RoPE, but this was
       not pre-registered and is not claimed.
     - Lesson for step 11: use longer runs and/or more seeds before architecture claims.
  5. **The learning rate is still not tuned.** The optimum at this toy scale lies around
     6e-3–1e-2 and differs by architecture. That does not transfer to GPU scale, where the
     step-10/11 runs must sweep the learning rate again.
* **Step 9 exit summary (MASTER_CONTEXT §37 step 9, "Run architecture ablations"):**
  - Two pre-registered ablation experiments ran on the founder's CPU laptop: EXP-032
    screening and EXP-033 confirmation.
    - 32 models were trained; 4 earlier ones were reused after verification.
    - All were graded by harness v1 (D-042) on the protected suite.
    - Every run is deterministic and reproducible (EXP-032 repro IDENTICAL).
  - Built infrastructure that step 11 reuses:
    - the spec schemas v1 and v2;
    - the restartable runner with verified reuse and a DIVERGED status;
    - the agent-free unattended PC runner.
  - Outcome: **no architecture change is supported strongly enough at toy scale.** The
    proposed D-043 keeps the EXP-B baseline as the provisional architecture for GPU
    bring-up (step 10). RoPE + GQA-2 is recorded as the first candidate for the GPU-scale
    ablation (step 11). GELU and LayerNorm are dropped.
  - Limits: toy scale (a transformer body of about 1M parameters; embeddings are 80% of all
    parameters), 150 steps, 3 seeds, in-distribution suite. Step 9 cannot say anything
    about large-model architecture. It only says what did *not* earn a change here.
* **Status:** complete. **D-043 is proposed, pending founder review.** It is recorded in
  DECISIONS.md after approval, as with D-040.

* **Update (2026-09-29):** the founder approved D-043 ("approve D-043 and data plan"). It is
  recorded in DECISIONS.md as accepted, and step 9 is complete. The EXP-033 record above is
  unchanged.

### EXP-034 — Sangraha Verified slice 1: acquire and inspect before any filtering (D-044; data scale-up phase 1)
**Date:** 2026-09-29 · **Status:** pre-registered; code built and tested (sandbox); PC night run
pending (founder: "approve D-044 option 2 with Sangraha Verified", 2026-09-29)

**Purpose (why this serves the mission):** a model that can compete with the best needs far more
high-quality Indic text than the 4.2M-character v1 pilot. D-044 approves Sangraha *Verified*
(CC-BY-4.0) as the first large source. Before any filter threshold is chosen, we measure what the
files actually contain — so every later cleaning rule is justified by numbers from our own data,
not by guesses or by trust in someone else's pipeline.

**Inputs (pinned before any content was read):** `corpora/frontier/v2/sangraha_slice1.json` —
revision `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`, file `verified/<lang>/data-0.parquet` for
each of our 13 languages, 5,106,130,219 bytes total, size and SHA-256 per file (from the HF tree
API; the LFS object id is the SHA-256). Selection rule: file name only (`data-0`).

**Method:**
1. `scripts/fetch_sangraha_slice.py`: disk-space check (missing bytes + 3 GB margin), resumable
   download, accept a file only if size and SHA-256 equal the pin (else set aside, never used).
2. `scripts/inspect_sangraha_slice.py` (`corpus/slice_inspect.py`), one streaming pass per file,
   filtering nothing. Per file: documents; characters and bytes; length quantiles; `type` mix
   (web / OCR / speech etc.); documents with line breaks; normalization changes (NFC policy of
   D-036); **script check** with the pipeline's own langid rule (declared-script share ≥ 0.6 of
   letters; profile of the first 5,000 characters, documents longer than that are counted); exact
   duplicates within the file; hits of the default quality rules (`QualityPolicy`); **protected
   suite overlap** (exact text or any 13-gram shared with `frontier-heldout-v1`; the held-out
   shard is verified line by line against `SUITE.json` first); and a **token estimate** with the
   frozen Frontier Tokenizer v1 (every 50th document, ≤ 3M characters per file) reported with its
   sample size. A few text samples (≤ 160 characters, e-mails / URLs / long digit runs masked)
   are kept for a human read.
3. `scripts/run_data_night.ps1` runs 1–2 unattended on the PC and commits only
   `evals/results/EXP-034/` (`summary.json`, `SUMMARY.txt`).

**Pre-registered rules (fixed before any result):**
- This experiment **adopts nothing and removes nothing.** Its only outputs are measurements.
- The slice counts as **acquired** only if all 13 files verify against their pins.
- The inspection counts as **complete** only if every file was inspected in full (no
  `--max-docs`) and the suite check ran (`suite_status` starts with `CHECKED`). An inspection
  with `NOT CHECKED` is reported as partial and must be repeated before a build.
- Filter thresholds for the v2 build (script share, minimum/maximum length, quality rules,
  near-duplicate settings) are chosen **after** reading these numbers and pre-registered in the
  next experiment, together with the reason for each threshold.
- The token figure is an **estimate** from a sample. No corpus-size claim is made from it; the
  build experiment counts tokens exactly.
- Known limitation, stated in advance: `data-0` may not be a random sample of each language (the
  file order inside Sangraha is NOT VERIFIED). The `type` mix per file makes this visible; if one
  source type dominates, later slices must add files, not just more of the same file.

**Sandbox checks (2026-09-29):** 19 new tests (`tests/test_sangraha_slice.py`): pins complete
and tamper-proof; download verifies, resumes after a dropped connection, restarts when a server
ignores resume requests, sets aside a wrong-hash file; free-space refusal; parquet streaming with
provenance ids; the held-out loader refuses a wrong/partial shard and ignores CRLF; inspection
counts on a hand-built sample; end-to-end CLI including the `NOT CHECKED` path; static safety of
the PS runner. The langid script profile was made ~5× faster without changing its result (a
test compares it with the old per-character method, including key order). Sandbox throughput of
the full inspection pass on synthetic Hindi web-like text: about 2.5M characters/s; laptop speed
NOT VERIFIED.

**Results (PC night run 2026-09-29, 11:42–14:11 IST; published by the runner as commit `484b8a7`:
`evals/results/EXP-034/summary.json` and `SUMMARY.txt`; machine Windows 10, Python 3.13.15, 4
logical CPUs):**

*Pre-registered rules checked (by the agent, from `summary.json`):*
- **Acquired: yes.** All 13 files match their pinned size and SHA-256 (`pinned_sha256_verified`
  is true for 13/13). The download took 36 minutes.
- **Inspection complete: yes.** `max_docs_per_file` is null (every document read) and
  `complete` is true. Each file's document count equals the row count its parquet file declares.
  `suite_status` = "CHECKED against frontier-heldout-v1 (3427 documents, 38,297 13-grams; shard
  verified against SUITE.json)". The inspection pass took 1 h 53 min.
- **Adopted / removed: nothing.** Only the measurements below exist; no corpus was built.

*What the files contain (all 13 files, before any filtering):*

| lang | documents | characters (M) | script pass (docs) | exact dups | suite hits | PDF share of chars | est. tokens (M) |
|---|---|---|---|---|---|---|---|
| as | 110,238 | 318.9 | 94.6% | 0 | 5 | 50.2% | 175.3 |
| bn | 149,797 | 382.9 | 98.7% | 0 | 0 | 31.6% | 204.3 |
| en | 349,525 | 936.7 | 99.9% | 90 | 0 | 4.0% | 490.4 |
| gu | 149,797 | 362.0 | 99.1% | 0 | 0 | 29.3% | 205.3 |
| hi | 174,763 | 387.5 | 98.6% | 3,019 | 1 | 22.8% | 208.3 |
| kn | 174,762 | 360.1 | 98.5% | 0 | 4 | 16.7% | 174.2 |
| ml | 174,763 | 330.9 | 99.3% | 0 | 0 | 2.4% | 149.0 |
| mr | 174,763 | 379.5 | 99.5% | 0 | 0 | 11.5% | 185.1 |
| or | 189,757 | 358.8 | 99.2% | 0 | 8 | 41.5% | 191.4 |
| pa | 149,797 | 362.8 | 99.1% | 0 | 2 | 15.9% | 206.7 |
| ta | 149,796 | 375.1 | 98.9% | 0 | 0 | 13.4% | 165.2 |
| te | 174,762 | 356.0 | 98.8% | 0 | 0 | 19.4% | 200.7 |
| ur | 209,716 | 571.3 | 99.7% | 0 | 0 | 40.0% | 358.2 |
| **total** | **2,332,236** | **5,482.5** | 24,231 docs fail (1.0%) | 3,109 | **20** | 21.5% | **≈ 2,914 (estimate)** |

The token column is an **estimate**: the frozen tokenizer ran on every 50th document, up to 3M
characters per file (1,060–1,633 documents per file), and the result was scaled to the file. It
is not a corpus size (pre-registered rule). For scale only: the v1 pilot has 3.8M training
characters; this one slice, unfiltered, has about 1,400 times as many.

*Findings that later rules must address (observed; causes marked NOT VERIFIED where unknown):*
1. **Protected-suite overlap is real: 20 documents in 5 languages.** Every hit points to one of
   the Wikisource books our held-out suite comes from: Manomati (as), Godaan (hi), Ranganna (kn),
   Chha Mana Atha Guntha (or) and Satwant Kaur (pa). One Odia document shares 1,230 13-grams with
   a single suite document, so Sangraha contains copies of the same public-domain books.
   → Every build and every later slice must run the suite guard; this is already the design (D-044).
2. **Coverage gap in the suite guard.** 1,882 of the 3,427 suite documents are shorter than one
   13-gram. They are protected only by exact match, so a short held-out passage inside a longer
   Sangraha document would not be caught. → EXP-035 must close this gap before a build.
3. **The script gate is not a language check.** A document in the Urdu file that passed (share
   0.964) is Uyghur, which is also written in Arabic script. Assamese and Bengali also share a
   script. Script-mismatch samples show religious texts (Bible verses, a Quran translation) in
   non-Indic Latin-script languages inside the as, or and ur files, Uzbek in the pa file, and
   Gujarati and Telugu Bible text inside the English file.
   → The build needs a language-level check where scripts are shared; how often this happens is
   NOT VERIFIED.
4. **The 0.6 threshold also rejects code-mixed text.** 13,536 documents have a declared-script
   share of 0.4–0.6. In 7,931 rejected documents the declared script is still the most common
   one (hi 1,018, kn 1,082, bn 968, …). The samples are mostly real language mixed with English
   (Hindi news with "Google" and "Pixel"; site navigation text). → EXP-035 decides the threshold
   from a read sample. It is under 0.4% of documents either way.
5. **OCR text is a large share.** PDF-type documents are 21.5% of all characters, and 50.2% of
   Assamese, 41.5% of Odia and 40.0% of Urdu characters. Their OCR quality has not been measured.
   → EXP-035 needs an OCR-quality read before deciding how to treat them.
6. **Long documents.** The default quality rule `max_chars` (20,000) would drop 488–2,007
   documents per file (p99 document length is 10,872–27,615 characters; which types they are is
   NOT VERIFIED). The
   script check reads only the first 5,000 characters (8,507–35,727 documents per file are
   longer). → EXP-035 decides between splitting long documents and dropping them.
7. **Other quality-rule hits** (they count, remove nothing):
   - `digit_runs` fires on 546–6,813 documents per file. It may be over-triggering on news
     (dates, scores, phone numbers); NOT VERIFIED.
   - `repetition` fires 494 times in ml and at most 9 times in every other file; the cause is
     NOT VERIFIED.
   - `url_density` fires on 87–1,437 documents per file and `template_residue` on 8–112.
8. **Exact duplicates** are 1.7% of Hindi documents (3,019) and 90 English documents; every
   other file has 0, which suggests the source already removed exact duplicates there.
   Near-duplicates have not been measured (MinHash is planned).
9. **Normalization:** NFC (D-036) changes 43.1% of Marathi documents and 0.3–11.8% of the other
   files. The Marathi cause is NOT VERIFIED; the likely cause is a decomposed character sequence
   that is common in Marathi. NFC is applied anyway.
10. **Line structure:** 79–96% of documents in each file contain line breaks (41% in Urdu). The
    v2 format must keep document boundaries and must not use one line per document.
11. **Machine-translated text is visible in the samples.** For example, a Gujarati sample
    transliterates Turkish names letter by letter. How much there is is NOT VERIFIED.
12. **Known limitation (pre-registered):** web documents are 66.9–98.6% of each file. Whether
    `data-0` is a random sample of each language is still NOT VERIFIED; later slices should add
    other files (`data-1` …), not more of the same file.

**Status:** complete (acquired, fully inspected, suite check ran; adopts nothing).
Thresholds for the v2 build come in EXP-035, which is still to be proposed; the founder must
approve it before any implementation.

### EXP-035 — Calibrate the v2 build rules on the Sangraha slice: read samples, count, remove nothing (D-044; data scale-up phase 1)
**Date:** 2026-09-29 · **Status:** pre-registered; code built and tested (sandbox); PC night run pending (founder: "approve EXP-035", 2026-09-29)

**Purpose (why this serves the mission):** EXP-034 showed *what* is in the 13 files (twelve
findings: suite overlaps, code-mixed text, Uyghur passing as Urdu, 21.5% PDF/OCR text, long
documents, rule hits of unknown quality). A cleaning threshold chosen without reading the
documents it would remove is a guess. This experiment reads and counts, per problem group, so
that every v2 build rule (EXP-036) comes with a reason taken from our own data. Better rules
mean cleaner training text, which is the part of model quality we control on a laptop.

**Numbering note (no record changed):** the closing line of EXP-034 said the build thresholds
would come in EXP-035. In the plan the founder approved, EXP-035 is this calibration and the
thresholds + v2 build move to **EXP-036** (needs its own approval).

**Inputs:** the same 13 pinned files as EXP-034 (`corpora/frontier/v2/sangraha_slice1.json`,
revision `8b813c3f62d3…`), already on the PC; each file is re-checked (size + SHA-256) before it
is read. No new download. The protected suite `frontier-heldout-v1` and its PC-only held-out
shard (verified line by line against `SUITE.json`).

**Method:** `scripts/calibrate_sangraha_slice.py` (`corpus/slice_calibrate.py`), one streaming
pass per file, CPU only, **removes nothing**. Per file it writes:
1. **Samples for a human read** (reservoir sampling, seed = `EXP-035/<file>/<group>`, so the
   selection is reproducible): ordinary passing documents; script share 0.4–0.6 and 0.6–0.8
   (code-mixed candidates); long documents whose later text fails the script check; PDF/OCR
   documents; documents over 20,000 characters (start and middle excerpts); hits of the
   `digit_runs`, `repetition`, `url_density` and `template_residue` rules; documents that look
   like another language of the same script; Hindi documents with Marathi's `ळ`; up to 4
   near-duplicate pairs. At most 60 records per file, excerpts ≤ 400 characters; e-mails, URLs
   and digit runs of 7+ digits are masked.
2. **Counts the samples cannot give:** script-share bands split by whether the declared script is
   still the top script; start/middle/end script check for documents longer than the 5,000-
   character profile; letter markers in shared scripts (Assamese ৰ/ৱ vs Bengali র; Urdu ٹ ڈ ڑ ں ے ھ
   vs Uyghur, Sindhi, Pashto, Arabic letters); NFC changes vs whitespace-only changes, and which
   code points NFC changes (the Marathi 43.1% question); repeated lines (exact line hashes;
   boilerplate share); within-file near-duplicates (MinHash, 128 permutations, word 5-grams, LSH
   16 bands × 8 rows, estimated Jaccard ≥ 0.8 with the bucket representative); quality-rule hits by
   document type; documents over 20,000 characters by type.
3. **The short-suite gap (EXP-034 finding):** 1,882 suite documents are shorter than 13 words and
   were protected by exact match only. New `ShortSuiteIndex` (`corpus/decontaminate.py`) finds
   every suite document of 3–12 words that appears **word for word inside** a Sangraha document
   and counts hits by suite-document length (3–4, 5–6, 7–9, 10–12 words), so the minimum length
   of the v2 containment rule can be chosen from data. It is not yet part of the build guard.
4. **Suite text is never published.** A document that contains a short suite passage is counted
   but never sampled; documents kept by a reservoir and the near-duplicate excerpts are also
   screened with the EXP-034 13-gram guard, and screened-out samples are counted. Without the
   held-out shard, no samples are written at all (`samples_status: WITHHELD`).
5. `scripts/run_calibration_night.ps1` → `scripts/run_data_night.ps1 -Exp EXP-035 -Task
   calibrate`: same checks as EXP-034, then commits only `evals/results/EXP-035/`
   (`summary.json`, `SUMMARY.txt`, `samples.jsonl`) and pushes.

**Pre-registered rules (fixed before any result):**
- This experiment **adopts nothing and removes nothing.** No threshold is set here.
- The calibration counts as **complete** only if all 13 files were read in full (no `--max-docs`,
  no `--only`) and `short_suite_status` starts with `CHECKED`. Otherwise it is partial and is
  repeated before EXP-036.
- EXP-036 must state, for each rule it adopts, the threshold, the reason (a number or a sample
  from EXP-034/EXP-035), and what share of documents and characters it removes per language.
- Language markers are coarse letter counts used to pick samples and to size the problem; they
  are not a language classifier and are not used to remove anything here.
- Near-duplicate counts are within one file only; cross-file and cross-language duplicates are
  EXP-036 work.
- No corpus-size or token claim is made from this experiment.

**Sandbox checks (2026-09-29):** 11 new tests (`tests/test_slice_calibrate.py`): masking keeps
short numbers; reservoir determinism; short-suite containment (contained / one word different /
too short / 13+ words left to the 13-gram guard; `min_words` validation); MinHash (identical and
one-word-changed documents cluster, different documents do not; deterministic); exact repeated-
line counts; Assamese/Bengali and Urdu/Uyghur markers; the calibration on a hand-built file
(counts, suite-touching documents never sampled, reproducible); refusal when a file has more rows
than declared; end-to-end CLI with and without the held-out shard (without: `NOT CHECKED`, not
complete, no samples). The static test of the PS runner now also covers `-Task`. The quality
rule `control_chars` was made faster with a regex that gives the same result (a test compares
it with the old per-character method on random text). Sandbox throughput on synthetic Hindi
web-like text: about 1,000 documents/s (EXP-034's inspection pass: about 1,500/s on the same
text). Laptop runtime is **NOT VERIFIED**; estimate from EXP-034's 1.9 h inspection pass:
about 2.5–3.5 h.

**Results (PC night run 2026-09-29, 16:08–18:45 IST, 2 h 36 min; published by the runner as
commit `1be3c48`: `evals/results/EXP-035/summary.json`, `SUMMARY.txt`, `samples.jsonl` (603
masked records); machine Windows 10, Python 3.13.15, 4 logical CPUs; analysed by the agent from
those files):**

*Pre-registered rules checked:*
- **Complete: yes.** All 13 files were re-verified (size + SHA-256, 13/13) and read in full
  (`max_docs_per_file` null, `complete` true). `short_suite_status` = "CHECKED against
  frontier-heldout-v1: 1727 suite documents of 3-12 words indexed (155 shorter; shard verified
  against SUITE.json)". Throughput 160 (as) to 343 (gu) documents/s.
- **Removed / adopted: nothing.** No sample touches the suite: 45 documents with a short suite
  passage were withheld from sampling, and the 13-gram screen dropped 0 kept samples.
- The runner's last report line said "EXP-034 done" (the text was fixed in the script, not
  taken from `-Exp`). Fixed after the run; the static test now checks it. The results are
  not affected.

*Counts (all 13 files, before any filtering):*

| lang | docs | script share < 0.6 | near-dup removable | chars in lines seen ≥ 100× | docs > 20k chars (share of chars) | not NFC | short-suite hits |
|---|---|---|---|---|---|---|---|
| as | 110,238 | 6,003 (5.45%) | 8 | 0.60% | 1,983 (23.2%) | 0.3% | 7 |
| bn | 149,797 | 1,952 (1.30%) | 0 | 0.48% | 1,532 (15.6%) | 0.1% | 17 |
| en | 349,525 | 222 (0.06%) | 543 (0.16%) | 1.01% | 2,007 (8.3%) | 0.3% | 3 |
| gu | 149,797 | 1,364 (0.91%) | 0 | 0.29% | 709 (6.3%) | 0.1% | 0 |
| hi | 174,763 | 2,507 (1.43%) | 3,248 (1.86%) | 2.10% | 767 (9.0%) | 1.2% | 3 |
| kn | 174,762 | 2,664 (1.52%) | 1 | 1.44% | 501 (4.5%) | 0.1% | 1 |
| ml | 174,763 | 1,283 (0.73%) | 1 | 2.76% | 546 (6.4%) | 0.0% | 0 |
| mr | 174,763 | 878 (0.50%) | 1 | 1.09% | 594 (6.0%) | 42.8% | 2 |
| or | 189,757 | 1,478 (0.78%) | 0 | 0.61% | 1,183 (14.0%) | 0.0% | 12 |
| pa | 149,797 | 1,412 (0.94%) | 2 | 0.44% | 832 (8.5%) | 0.1% | 0 |
| ta | 149,796 | 1,692 (1.13%) | 0 | 1.17% | 1,063 (9.5%) | 0.2% | 0 |
| te | 174,762 | 2,131 (1.22%) | 0 | 1.04% | 488 (4.8%) | 0.1% | 0 |
| ur | 209,716 | 646 (0.31%) | 2 | 0.05% | 1,293 (7.8%) | 3.8% | 0 |

Script share is the declared script's share of letters in the first 5,000 characters (EXP-034
profile). Near-duplicates are within one file (MinHash, estimated Jaccard ≥ 0.8), counting all
but one document per cluster. "Chars in lines seen ≥ 100×" is the share of line characters in
exact lines that occur at least 100 times in the file.

*What the samples and counts show (samples read by the agent; excerpts ≤ 400 characters):*
1. **Long documents are the best text, not errors.** Of 52 samples over 20,000 characters, nearly
   all are books, novels, literary essays, scripture commentary, legislative debates and long
   articles (3 are weak: a mojibake English blog, a spammy Kannada page, a machine-translated-
   looking Punjabi page). They are 23.2% of Assamese, 15.6% of Bengali and 14.0% of Odia
   characters. The default rule `max_chars` 20,000 would delete them. Of the documents longer
   than 5,000 characters whose start passes the script check, 99.0–99.7% also pass in the middle
   and at the end (e.g. en 35,507 of 35,623); later-window failures in the samples are English tails, parallel
   English translations in debates, JavaScript in Kannada pages and one Malayalam page in an
   old ASCII font (mojibake).
2. **About 8.4% of the Urdu file is Persian.** 17,332 Urdu-file documents (48.1M characters)
   contain none of the letters ٹ ڈ ڑ ں ے ھ. All 6 samples are Persian (Iranian news agencies,
   e.g. Tasnim; Persian grammar and words), with Persian ه written as Urdu ہ. Also 209 documents
   (0.58M chars) have Uyghur-only letters (EXP-034 showed a Uyghur document passing as Urdu) and
   53 have Pashto letters (not sampled).
3. **Assamese vs Bengali is not a real problem.** Only 20 Assamese-file documents look
   Bengali-like and 1 Bengali-file document looks Assamese-like; 5 of the 6 as samples are old
   Assamese printed with Bengali র (one is Middle Bengali poetry). The as file's real problem is
   the 5,182 documents whose top script is not Bengali-Assamese at all (share < 0.2, already
   rejected by the 0.6 script gate; EXP-034 samples: Latin-script religious texts).
4. **Code-mixed band (0.4–0.6, 0.6–0.8):** most samples are real native-language text plus
   whole English lines: site furniture ("Welcome! / Forgot your password?", "Begin typing your
   search…"), product lists, embedded tweets, JSON-LD. Mixed words inside a sentence (Hinglish
   "Post Office recruitment की लिस्ट") are rare in these samples. A smaller part is junk (drug
   template pages, sex-story spam) or bilingual text (Telugu–English Bible in the en file). So
   the document-level share mostly measures English *lines*, and removing lines with no letter
   of the declared script would recover the native text.
5. **Rule hits:**
   - `digit_runs` (65 samples): ordinary news and books; the digit run is usually a phone
     number, ID or date. **Not a quality signal** → replace by masking phone numbers/e-mails.
   - `url_density` (39): mostly ordinary short articles with one or two web addresses; a few
     junk pages (search pages, "What's Japanese for …" template pages, JavaScript notices).
     **Not a good document-removal rule.**
   - `template_residue`: with **one** `{{`/`}}` marker, 14 of 16 samples are PDF literature or
     scripture with a stray OCR brace (good text); with **two or more**, 23 of 23 are web pages,
     mostly wiki edit histories and raw wiki markup (junk).
   - `repetition` (21): mostly junk — Malayalam pages that repeat their title/article many
     times (494 of ml's hits), empty lyric/speech pages ("- - -", "! ! !"), tables; 2 song
     lyrics and 1 repetitive scripture are the only losses.
6. **Boilerplate lines.** The most repeated lines are site furniture: "- First Published :"
   (×7,920 in as), News18/DNPA/"Don't Miss!"/"Follow us on Google News" footers, "Digitized by …"
   stamps in Odia PDFs, an RTI nodal-officer line (×2,638 in hi). Lines seen ≥ 100 times hold
   0.05% (ur) to 2.76% (ml) of line characters; ≥ 10 times, 0.10% to 5.18%.
7. **PDF/OCR text is mostly readable.** Of 104 passing PDF samples, 1 is unreadable (Bengali with
   stray Devanagari letters inside words); several have OCR letter errors but are readable (e.g.
   Gujarati `ેા` for `ો`, Odia letter swaps). Most are literature, scripture, history and
   legislative records — text that web crawls rarely have.
8. **Near-duplicates:** Hindi 3,248 removable (69 clusters: land-record forms, government press
   releases); English 543; every other file ≤ 8. Two of the 19 sampled pairs share the same
   `doc_id`, so Sangraha `doc_id` is **not unique** (provenance must add file + row).
9. **NFC is safe (D-036 stays).** The Marathi 42.8% is canonical composition: NFC joins र +
   nukta (U+0930 U+093C) into ऱ (U+0931) — 208,556 RA and 208,711 nukta code points disappear.
   Urdu: hamza above + waw → ؤ; Bengali: U+09DF is split by NFC (composition exclusion).
10. **Short-suite containment:** 45 documents contain a suite passage of 3–12 words word for word
    (as 7, bn 17, en 3, hi 3, kn 1, mr 2, or 12). 20 of them contain a passage of ≥ 5 words;
    25 only 3–4-word passages, and some 3–4-word passages recur in several documents (one Odia
    3-word passage in 6), which suggests common phrases or titles. Removing all 45 costs
    0.002% of documents, so the protection does not need a trade-off.
11. **Other observations (not measured, for later):** 249 Hindi-file documents with ≥ 3 ळ
    (10.8M chars): 2 of 4 samples are Rajasthani, 2 Hindi. Machine-translation artefacts are
    visible (English word fused into Gurmukhi: "architectਾਂਚਾਗਤ"). Their share is NOT VERIFIED.

**Proposed next step (not approved; nothing built): EXP-036 = the v2 build on this slice.**
Every rule below comes with its reason from EXP-034/EXP-035. Removal shares that EXP-035 could
not count (line-level rules, markers after cleaning) are measured by EXP-036 itself and
reported per rule and language before the corpus is accepted.

| # | rule | threshold | reason (EXP-034/035) | removes (per language, before other rules) |
|---|---|---|---|---|
| 1 | protected suite | 13-gram hit (EXP-034) **or** a suite passage of ≥ 3 words contained word for word | 20 + 45 docs; cost 0.002% | ≤ 65 docs (as 12, bn 17, en 3, hi 4, kn 5, mr 2, or 20, pa 2) |
| 2 | exact + near-duplicates | keep one per MinHash cluster (Jaccard ≥ 0.8), within file | hi 1.86% (templates, press releases) | hi 3,248, en 543, others ≤ 8 docs |
| 3 | boilerplate lines | drop exact lines seen ≥ 100 times in the file | top lines are site furniture / stamps | 0.05% (ur) – 2.76% (ml) of chars |
| 4 | foreign lines | drop lines with letters but **no** letter of the declared script | 0.4–0.8 band = native text + English lines | measured in EXP-036 |
| 5 | script gate | declared share ≥ 0.6 on the **whole cleaned** document (not the first 5,000 chars) | 0.6 stays; after rule 4 most code-mixed docs pass | ≤ docs < 0.6 today (0.06% en – 5.45% as) |
| 6 | Urdu language check | drop ur docs with none of ٹ ڈ ڑ ں ے ھ, and docs with Uyghur-only letters | 6/6 samples Persian; EXP-034 Uyghur sample | ≈ 17,541 docs, ≈ 8.5% of ur chars |
| 7 | long documents | **no** length limit (drop `max_chars`) | long docs are books/debates | 0 (keeps 4.5–23.2% of chars) |
| 8 | wiki markup | drop at ≥ 2 `{{`/`}}` markers (was ≥ 1) | 23/23 junk at ≥ 2; 14/16 good at 1 | ≤ 8–112 docs per file |
| 9 | repetition | drop (was flag), same threshold (distinct-bigram ratio < 0.3) | mostly junk; 3 of 21 losses | ml 494, others ≤ 9 docs |
| 10 | digits / URLs | no removal; mask e-mails and phone-like digit runs | `digit_runs` and `url_density` hit good text | 0 docs |

Kept as is: NFC (D-036); PDFs stay (rule 3 removes their stamps). Not adopted yet, only
measured in EXP-036: Pashto-letter docs (53), Hindi docs with ळ, a minimum length after
cleaning, OCR garble and machine-translation signals. Output: JSONL shards with document
boundaries and provenance (file, row, `doc_id`, type, revision), attribution file, **exact token
count** with the frozen tokenizer, and a final suite guard on the output that must find 0 hits.
The corpus itself stays on the PC (not in Git); only the manifest, counts and masked audit
samples are committed.

**Status:** complete (13/13 files verified and read in full; short-suite check CHECKED; adopts
and removes nothing). EXP-036 (above) needs founder approval before any build code.

### EXP-036 — FrontierCorpus v2 build on the Sangraha slice with the 10 approved rules (D-044; data scale-up phase 1)
**Date:** 2026-09-29 · **Status:** pre-registered; code built and tested (sandbox); PC night run pending (founder: "approve EXP-036", 2026-09-29)

**Purpose (why this serves the mission):** a model can only be as good as the text it learns
from. EXP-034 measured the 13 files and EXP-035 read samples of everything each rule would
remove. This experiment applies the resulting rules and produces the first cleaned, provenance-
tracked, suite-protected Indian-language corpus of this project with an **exact** token count —
the first real training-data number instead of an estimate.

**Inputs:** the 13 pinned files (`corpora/frontier/v2/sangraha_slice1.json`, revision
`8b813c3f62d3…`, already on the PC; each re-checked by size + SHA-256 before it is read); the
protected suite `frontier-heldout-v1` with its PC-only held-out shard (verified line by line
against `SUITE.json`; **mandatory** — without it the build stops before reading any data); the
frozen tokenizer `frontier-tokenizer-v1` (hash-checked by its loader).

**Method:** `scripts/build_sangraha_v2.py` (`src/frontier_ai/corpus/slice_build.py`), CPU only,
two files at a time (`--workers 2`, the laptop has 2 cores). Per file, three streaming passes:
(A) count every exact line; (B) apply the rules to each document and record the **first** reason
it fails; (C) near-duplicates, then write. The rules of the EXP-036 table (end of EXP-035), in the
order they are applied:

| order | rule (table #) | exact definition in the code |
|---|---|---|
| 1 | normalization (v1, D-036) | NFC, `\r` removed, runs of spaces/tabs → one space; empty → `empty` |
| 2 | protected suite (1) | on the normalized source text: exact suite text or a shared 13-gram (`SuiteGuard`) → `suite_exact`/`suite_ngram`; a suite document of 3–12 words contained word for word (`ShortSuiteIndex`, `min_words=3`) → `suite_short` |
| 3 | exact duplicates (2) | identical normalized text seen earlier in the same file → `exact_duplicate` |
| 4 | control characters (v1, unchanged) | > 2% invisible control characters → `control_chars` (0 hits in EXP-034/035) |
| 5 | boilerplate lines (3) | drop every line (stripped) whose exact text occurs ≥ 100 times in the file (counted in pass A over all documents) |
| 6 | foreign lines (4) | drop every line that has ≥ 1 letter (any script, including blocks the profiler does not know) and **no** letter of the declared script; lines without letters (numbers, punctuation) stay |
| 7 | — | lines are stripped; runs of blank lines become one blank line; nothing left → `empty_after_cleaning` |
| 8 | script gate (5) | on the **whole cleaned** document: no letters → `no_letters`; declared-script share < 0.6 → `script_share` |
| 9 | Urdu check (6, ur only) | on the cleaned document, EXP-035's `marker_label`: none of ٹ ڈ ڑ ں ے ھ among ≥ 50 letters → `urdu_persian_like`; ≥ 3 Uyghur-only letters and more than Urdu letters → `urdu_uyghur_like`; Pashto-/Sindhi-like documents are kept and counted |
| 10 | long documents (7) | no length limit |
| 11 | wiki markup (8) | ≥ 2 `{{`/`}}` markers in the cleaned document → `wiki_markup` |
| 12 | repetition (9) | ≥ 8 words and distinct word-bigram ratio < 0.3 → `repetition` |
| 13 | contacts (10) | nothing dropped: e-mail addresses → `[email]`; phone numbers → `[phone]` (`+CC` followed by 8–13 digits with optional spaces/hyphens; Indian mobile `98765 43210`/`98765-43210`; 10–13 contiguous digits; Indic digits count). Year lists (`1991 1992 1993`), PIN codes and amounts are not touched (tested) |
| 14 | protected suite again (1) | the **exact output text** is checked again (13-gram, exact, short passages) → `suite_after_cleaning`. Why: removing a line joins the words around it, which could create a 13-gram the source did not have |
| 15 | near-duplicates (2) | among documents that passed everything above: MinHash on the output text (128 permutations, word 5-grams, seed 35, LSH 16 × 8, estimated Jaccard ≥ 0.8 with the bucket representative); keep the first row of each cluster → `near_duplicate`. Done last so a cluster keeps a document that is really in the corpus |

**Output (PC only, git-ignored):** `data\frontier_v2\sangraha-slice1-v2\<lang>.jsonl.gz`, one JSON
document per line: `id` (`<source_id>#<row>`, unique; Sangraha `doc_id` is not), `text`, `lang`,
`source`, `row`, `doc_id`, `type`, `revision`, `tokens` (exact, frozen tokenizer). gzip with
`mtime=0`, so the same input gives the same bytes. Plus `ATTRIBUTION.txt` (Khan et al. 2024,
CC-BY-4.0, list of changes) and `manifest.json`. A finished file is recorded in
`<lang>.stats.json`; running the same line again reuses files built with the same code, rules,
pin, suite and tokenizer (checked by SHA-256), so an interrupted night is not lost.

**Committed evidence (runner, `evals/results/EXP-036/`):** `summary.json`, `SUMMARY.txt`,
`manifest.json` (file names, counts, tokens, SHA-256; no text) and `samples.jsonl`: per file up
to 4 masked excerpts (≤ 400 chars) per removal reason, 8 removed foreign lines, 6 kept documents
and 4 kept documents that lost lines, plus the top 25 removed boilerplate lines in the summary.
Suite-touching documents are never sampled as documents.

**Pre-registered rules (fixed before any result):**
- **Complete** only if all 13 files were built in full (no `--max-docs`/`--only`), every file
  matched its pin, the suite check was CHECKED, rows read = parquet row count, and kept + removed
  = rows for every file (the code refuses otherwise).
- **Suite gate: 0 hits in the output.** Every written text passed the 13-gram, exact and short-
  passage checks on its exact final form (step 14); tests re-check the output independently.
- **Per-rule cost is reported per language** (documents and share of characters, for removed
  documents and for removed lines). A rule is marked **REVIEW** in the report when it removes
  more than its bound (share of the file's characters): boilerplate lines 5%, foreign lines 10%,
  exact or near-duplicates 5%, script share 8%, no letters 2%, Urdu Persian-like 12%, Uyghur-like
  1%, wiki markup 2%, repetition 2%, empty after cleaning 2%, control chars 0.1%, suite after
  cleaning > 0. REVIEW does not stop the build; it means "read before accepting".
- **The corpus is not accepted automatically.** After the night the agent reads the samples per
  rule and the REVIEW lines and reports to the founder; the founder decides whether v2 is
  accepted. A rule that removes good text is changed only in a new, pre-registered experiment.
- **Token claims:** the exact count applies to this slice only (the first file `data-0` of each of
  13 languages), not to Sangraha. No model is trained on v2 without a separate approval.
- Measured only (nothing removed): Pashto-/Sindhi-like Urdu documents, Hindi/Marathi documents
  with ≥ 3 ळ, kept documents under 200 characters, Latin letters fused with Indic vowel signs
  ("architectਾਂ", a machine-translation/OCR artefact), kept documents that lost lines.
- Not done here (later work): cross-file and cross-language duplicates; OCR-garble detection.

**Sandbox checks (2026-09-29):** 20 new tests (`tests/test_slice_build.py`): the fast foreign-
line check equals its definition on 3,000 random mixed-script lines × 5 scripts; line cleaning
(boilerplate, foreign, paragraph breaks); 9 contact-masking cases (e-mail, +91, Indian mobile,
0-prefixed, Bengali digits; year lists, PIN, amounts, decimals untouched); the token counter
equals `len(encode(...))` with the frozen tokenizer (and with special tokens) and its cache is
bounded; the fused-Latin detector; a hand-built 16-document Hindi file that triggers every
removal reason exactly as designed (including a suite 13-gram created only by line removal),
with provenance, exact tokens, masked samples that contain no suite text, and an independent
suite re-check of the output; byte-identical output across two builds; the Urdu check (Persian,
Uyghur, short text kept); refusal on a row-count mismatch or a wrong short-suite index; the CLI
end to end (manifest hashes and token totals match the files; reuse on a second run; a damaged
output file is rebuilt; STOP without the held-out shard; STOP on too little disk space); and 1
vs 2 worker processes give byte-identical files. Runner: `-Task build` added to
`scripts/run_data_night.ps1` (step 1e stops without the held-out shard), wrapper
`scripts/run_build_night.ps1`, publisher also copies `manifest.json`; static test extended.

**Runtime and disk: NOT VERIFIED.** Sandbox, synthetic text built from EXP-035 samples: about
380–530 documents/s per worker, about 2× the time of the EXP-035 calibration pass on the same
text. EXP-035 took 2 h 36 min on the laptop, so one worker would need about 5 h; with two workers
the estimate is **about 3–6 h**. Output size is expected to be at most about the input size
(5.1 GB); the script stops before starting if less than 1.5 × the input size + 2 GB is free.

**Results (PC night run 2026-09-29/30, results commit `47eae55`; analysis 2026-09-30):**
`evals/results/EXP-036/` (`summary.json`, `SUMMARY.txt`, `manifest.json`, `samples.jsonl` with 471
masked excerpts). Founder's laptop (Windows 10, Python 3.13, 2 workers), 22:34 → 02:16
(**3 h 43 min**, inside the 3–6 h estimate). Config fingerprint `43ce349329a98770`, code
`24941a071255c303`, frozen tokenizer `b39afa08fa86b0d1`.

Pre-registered completeness checks:

| check | result |
|---|---|
| all 13 files built, row counts match the pins | yes (`complete: true`, `max_docs_per_file: null`) |
| suite check | CHECKED against frontier-heldout-v1 (3,427 docs, shard verified against SUITE.json) |
| suite hits in the written output | **0** (all 13 files) |
| suite removals before cleaning | 13-gram 20 docs, 3–12-word passage 36 docs, exact 0; created by line removal 0 |
| samples | 471, every one screened against the suite on its full text |
| REVIEW lines | 1: Assamese `empty_after_cleaning` 6.50% > 2% (explained below) |

Kept per language (exact token counts with the frozen tokenizer, **this slice only**):

| lang | input docs | kept docs | kept chars (share of input) | exact tokens | chars/token |
|---|---:|---:|---:|---:|---:|
| as | 110,238 | 104,386 | 292,524,670 (91.74%) | 155,735,335 | 1.88 |
| bn | 149,797 | 149,458 | 377,600,908 (98.61%) | 201,968,766 | 1.87 |
| en | 349,525 | 348,394 | 920,291,150 (98.25%) | 478,709,767 | 1.92 |
| gu | 149,797 | 149,466 | 357,391,570 (98.71%) | 201,772,881 | 1.77 |
| hi | 174,763 | 171,199 | 366,875,458 (94.69%) | 196,987,821 | 1.86 |
| kn | 174,762 | 173,555 | 346,614,900 (96.25%) | 166,057,000 | 2.09 |
| ml | 174,763 | 173,825 | 313,551,713 (94.75%) | 140,843,642 | 2.23 |
| mr | 174,763 | 174,485 | 372,982,531 (98.27%) | 181,004,073 | 2.06 |
| or | 189,757 | 188,737 | 352,233,374 (98.16%) | 188,496,005 | 1.87 |
| pa | 149,797 | 149,398 | 358,396,617 (98.79%) | 203,960,544 | 1.76 |
| ta | 149,796 | 149,525 | 367,312,056 (97.93%) | 161,146,485 | 2.28 |
| te | 174,762 | 174,147 | 347,777,028 (97.70%) | 196,080,611 | 1.77 |
| ur | 209,716 | 191,621 | 519,533,986 (90.94%) | 323,285,283 | 1.61 |
| **all** | **2,332,236** | **2,298,196 (98.54%)** | **5,293,085,961 (96.54%)** | **2,796,048,213** | 1.89 |

Compressed output: 2.91 GB (13 `.jsonl.gz` files, sha256 in `manifest.json`; the text stays on
the PC). **FrontierCorpus v2-slice1 = 2,796,048,213 tokens (≈ 2.80 B), exact.** This is the
first real token count of v2; it applies to the 13 `data-0` files only, not to Sangraha. For
scale: FrontierCorpus v1 training data is about 1.8 M tokens. Cross-check: EXP-034 *estimated*
about 2,914 M tokens for the uncleaned slice; 2,914 M × 96.54% (kept characters) ≈ 2,813 M, within
about 0.6% of the exact count, so the EXP-034 estimation method is confirmed for planning.

**The REVIEW line (Assamese `empty_after_cleaning`, 5,171 docs, 6.50% of characters).** All 4
sampled documents are Latin-script texts in non-Indian languages (Bible translations such as a
Mayan, a Guarani and two Papuan-like languages). These are the ~5,182 Assamese-file documents
whose main script was not Bengali in EXP-034 (and the first row seen through the datasets
server). Because the foreign-line rule (step 6) runs before the script gate (step 8), every line
of these documents was removed as foreign and the empty remainder was counted under
`empty_after_cleaning` instead of `script_share`. The documents are correctly removed; only the
label differs from the one the 2% bound was written for. Assamese removals by the two labels
together: 5,495 docs, versus 6,003 docs under the old document-level gate in EXP-035. **Verdict:
expected, no rule change needed.** Future bounds should count `empty_after_cleaning` together with
`script_share` (noted for a later experiment; the pre-registered bound is not changed here).

**What each rule removed, read against its samples** (removed share of the file's characters):

- **Foreign lines** (as 7.02%, kn 1.80%, all others ≤ 0.93%): 104 samples. Mostly site furniture
  in English ("Click to share on Twitter", "Follow Us On:"), Malagasy/Uzbek factory spam, and
  lines of other languages. **Known loss (small):** some good lines in another script are cut
  out of otherwise kept documents: Sanskrit verses (Devanagari) in Gujarati and Tamil religious
  books, English translation paragraphs in Bengali, Hindi lines in English parliamentary
  debates, Arabic Qur'an/hadith quotations in Bengali/Odia/Urdu books.
- **Script share < 0.6 after cleaning** (0.07–0.32% per language; 52 samples): medicine
  template pages ("X in Assamese ৰ ব্যৱহাৰ..."), spam, heavy code-mixing. Loss: some bilingual
  dictionaries (Odia–English) and English-learning books (Gujarati).
- **Line cleaning rescued documents.** Compared with the EXP-035 document-level gate (script
  share < 0.6: 24,232 docs), the build removed 3,583 docs by script share plus 7,622 as empty
  after cleaning. Roughly **13,000 documents that the old gate would have thrown away are now
  kept without their foreign lines** (approximate: EXP-035 measured the first 5,000 characters,
  the build measures the whole cleaned text).
- **Empty after cleaning, other languages** (kn 1.14%, or, ml, ur, te smaller): mostly non-Indian
  Latin-script Bible text and spam. Loss: whole documents in the wrong file (Hindi news in the
  English file, a Bengali hadith PDF in the Malayalam file, an English "Mann Ki Baat" transcript
  in the Marathi file) and romanised Hindi/Tamil/Urdu.
- **Exact duplicates:** hi 3,019 docs (3.64%, below the 5% bound): copies of one RTI disclosure
  page (4/4 samples); en 90 (JavaScript notices). **Near-duplicates** (last step, among survivors):
  en 791 (0.16%), as 334 (0.12%), hi 54, kn 70, ≤ 35 elsewhere. The Assamese number is higher
  than EXP-035's within-file count (8) on raw text; the samples are news articles, consistent with
  the same story republished by different sites becoming alike once their boilerplate lines
  (e.g. "- First Published :" × 7,920) were removed. NOT VERIFIED pair by pair (only the removed
  side is sampled).
- **Urdu check:** Persian-like 17,337 docs (8.43%, EXP-035 predicted ≈ 17,332); 4/4 samples are
  Persian (Iranian news, a Persian diwan). Uyghur-like 209 docs (0.10%): 4/4 are Uyghur or
  Sorani Kurdish, not Urdu. Pashto-/Sindhi-like 53 kept (measured only).
- **Wiki markup** (≤ 0.05% everywhere): mostly wiki edit histories and template source. Small
  loss: a few encyclopedia-style articles and OCR'd PDFs where stray braces counted as markup.
- **Repetition** (ml 494 docs, 1.46%, as EXP-035 predicted; ≤ 8 elsewhere): Malayalam news pages
  that repeat their own text; elsewhere tabular government gazettes and 2 Tamil song lyrics.
- **No letters:** 1 (a punctuation-only English transcript). **Control characters:** 0.
- **Boilerplate lines** (≤ 2.74% per language, ml highest): news-site footers, ad lines
  ("২ হাজাৰ টকা বিনিয়োগ কৰি..."), "This website follows the DNPA Code of Ethics." (hi × 5,237).
  A few short normal lines also repeat ≥ 100× ("ശരി." = "OK.", × 982); the loss is tiny.
- **Contact masking:** 4,120 e-mail addresses and 10,565 phone numbers replaced by placeholders.
- **Kept documents** (78 samples + 52 that lost lines): news, literature, government and
  religious books in the right language; no problem found in the samples.

Measured only (nothing removed): Latin letters fused with Indic vowel signs (a machine-
translation/OCR artefact) in pa 2,812 kept docs (1.9%), gu 1,347, bn 543; kept documents under
200 characters 10,116 in total. The ≥ 3 ळ count is meaningful only for Hindi (236 docs); for
Marathi (149,102) it only shows that ळ is an ordinary Marathi letter.

**Conclusion.** The build did what was pre-registered: every file is complete, the protected
suite is untouched (0 hits), each removal has a reason, and the one REVIEW line is a label shift,
not a wrong removal. The losses found in the samples are small (each ≤ 1% of a language) and are
listed for a later, separately approved experiment (not done here): cross-file routing of
documents that are in the wrong language file, keeping quoted verses/translations in another
script inside a document, counting the two script labels together, cross-file duplicates and
detecting machine-translation artefacts.

**Proposed decision (founder):** accept FrontierCorpus v2-slice1 (this build: 2,298,196
documents, 2,796,048,213 tokens, manifest `evals/results/EXP-036/manifest.json`) as the first v2
corpus, to be recorded as D-045. Accepting it does not start any training; training on v2 needs
its own approved plan (and step 10 still needs a GPU plan).

**Status:** complete (13/13 files built; 0 suite hits in the output; 2,796,048,213 tokens)
Suite check CHECKED; the exact token count is for this slice only. Acceptance of v2 is the
founder's decision (proposed D-045).

### EXP-037 — Make FrontierCorpus v2-slice1 training-ready: Frontier Tokenizer v2 (special tokens only) and packed token files
**Date:** 2026-09-30 · **Status:** proposed (needs the founder's "approve EXP-037" before any code)

**Purpose (why this serves the mission):** v2-slice1 (D-045) is clean text, but a model cannot
train on it yet. Two things are missing, and both are needed whichever GPU is used for step 10
(the collaborator's card or a rented one): (1) **a document-boundary token.** Frontier Tokenizer
v1 has no special tokens (D-041, consequence b), so documents would run into each other and the
model would learn false links between unrelated texts; D-041 already says such tokens must
arrive as `frontier-tokenizer-v2` with appended ids ≥ 32768; (2) **token files in the training
format** (`.bin` + `.meta.json`, read by `TokenDataset`) with a validation split to watch
training. Doing this now, while step 10 waits for the GPU report, means the first GPU run can
start as soon as its plan is approved.

**Part 1 — Frontier Tokenizer v2 = v1 plus special tokens, nothing else.**
- Same merges, same pre-tokenizer, same 32,768 ordinary ids. Appended: `<|endoftext|>` (32768),
  `<|pad|>` (32769) and 126 reserved slots `<|reserved_0|>`…`<|reserved_125|>`, so the vocabulary
  is 32,896 = 257 × 128 (a multiple of 128 is efficient on GPUs). Reserved slots keep model
  shapes stable when later stages need chat tokens; giving a slot a meaning is still a recorded
  change.
- **Ordinary text must encode exactly as in v1** (every id identical) — so every earlier token
  count and result stays valid. Checked on the protected suite texts, the EXP-036 samples and
  the v1 corpus.
- **Special tokens are never created from raw text.** A web page that contains the characters
  `<|endoftext|>` is encoded as ordinary text; the special id is only inserted by the packing
  code. (The current tokenizer code would match special strings inside text, so this is tested.)
- Frozen like v1 (`tokenizers/frontier-tokenizer-v2/`, own FREEZE record, verifying loader); v1
  stays unchanged. Recorded as a decision (D-046) after the result.

**Part 2 — pack v2-slice1 into training files (PC, one unattended night).**
- Input: the 13 files of v2-slice1; each must match its sha256 in
  `evals/results/EXP-036/manifest.json`, or the script stops.
- Validation split: a document goes to validation when the first bytes of sha256(its record
  id) put it in the lowest 0.5% (about 11,500 documents, about 14 M tokens). Deterministic and
  independent of file order. Near-duplicates were already removed within each file (EXP-036),
  so validation documents have no near-copy in training within the same language.
- Output (PC only, git-ignored): one `<lang>.bin` + `<lang>.meta.json` per language (uint16;
  `train || val`; exact bytes and characters per split for bits-per-byte), each document
  followed by one `<|endoftext|>`. **One file per language, so no mixing ratio is chosen here**
  (mixtures stay a later, measured comparison).
- Committed evidence: `evals/results/EXP-037/summary.json` + `SUMMARY.txt` (counts, sha256 of
  every output file). No text, no token files.

**Pre-registered checks (fixed before any result):**
- Per language: training tokens + validation tokens − number of documents = the EXP-036 exact
  count (2,796,048,213 in total). Any difference fails the run.
- Every document ends with exactly one `<|endoftext|>`; no other special id appears; decoding a
  sample of documents gives back the exact text.
- No record id is in both splits; the split is identical on a second run.
- Tokenizer v2 = v1 on all ordinary text (above); files byte-identical across two runs.

**Runtime and disk: NOT VERIFIED.** Output is about 2 bytes per token ≈ 5.6 GB; the script
checks free space first. Runtime will be measured in the sandbox before the night run (the
EXP-036 build, which also tokenized everything, took 3 h 43 min in total).

**Not in scope:** any training (needs the step-10 plan and its own approval), any change to the
v2-slice1 text, mixing ratios, a retrained tokenizer.

**Status:** proposed — founder approval needed before any code.

**Status:** approved and built — Frontier Tokenizer v2 frozen in the sandbox (all gates pass); packing code ready; the PC night run is next.
- Approved by the founder ("approve EXP-037", 2026-09-30). Code in `99a0cd5` and the commit after it.
- **Part 1 done (sandbox):** `tokenizers/frontier-tokenizer-v2/` frozen by
  `scripts/freeze_tokenizer_v2.py`; vocab 32,896 = the 32,768 v1 ids + `<|endoftext|>` (32768),
  `<|pad|>` (32769) and `<|reserved_0..125|>` (32770–32895). Merges and pre-tokenizer are v1's.
  Gates A–E pass: 1,074 texts (v1 golden samples, EXP-035 and EXP-036 samples, 4 adversarial
  strings with literal special-token strings) encode identically to v1 with `encode_ordinary`;
  every special encodes/decodes as itself; lossless round trip. Re-running the freeze script
  gives a byte-identical artifact (dir_sha256 `87bd6d31ae8723c2…`; tested); it refuses to
  overwrite an existing freeze. v1 is unchanged.
- **Part 2 built, not yet run:** `src/frontier_ai/corpus/pack.py` + `scripts/pack_sangraha_v2.py`,
  started on the PC by `scripts/run_pack_night.ps1` (`run_data_night.ps1 -Task pack`, no
  download). Every pre-registered check stops the run if it fails: input sha256 = EXP-036
  manifest (whose own sha256 must be the D-045 one); every document's v2 count = its EXP-036
  v1 count (so v2 = v1 is checked on all 2.3 M documents, not only samples); train + val − docs
  = the EXP-036 count per file and 2,796,048,213 in total; an independent re-read of each file
  (one `<|endoftext|>` per document, no other special id, both splits end with it); sampled
  documents decode to their exact text; record ids unique (so none can be in both splits).
  Split identical on a rerun and output byte-identical across runs and across 1 vs 2 workers:
  shown in tests (`tests/test_pack.py`, 11 tests); the PC run records a sha256 of the
  validation id list per file for later comparison.
- **Runtime estimate (NOT VERIFIED):** in the sandbox, packing took 17%–63% of the time the
  EXP-036 build code needed for the same synthetic documents (best and worst case for word
  caching). Scaled to the PC's 3 h 43 min build, that is roughly 40 min – 2 h 20 min. Disk:
  exactly 2 bytes per token = 5,596,692,818 bytes ≈ 5.6 GB; the script checks free space first.

**Results (PC night run 2026-09-30, results commit `3925cd7`; checked in the sandbox 2026-09-30):**
Laptop (Windows 10, Python 3.13.15, 2 workers), 17:02:52 → 19:38:04 = **2 h 35 min**, complete,
no file reused. Packer code `4c7423ce3d388ad7` (= `src/frontier_ai/corpus/pack.py` as committed in
`99a0cd5`), tokenizer `87bd6d31ae8723c2` (frontier-tokenizer-v2), input manifest sha256 = the
D-045 one. Evidence: `evals/results/EXP-037/{summary.json,SUMMARY.txt,manifest.json}`.

| lang | docs | val docs | train tokens | val tokens | min |
|---|---|---|---|---|---|
| as | 104,386 | 529 | 155,108,306 | 731,415 | 9.9 |
| bn | 149,458 | 790 | 200,991,341 | 1,126,883 | 18.5 |
| en | 348,394 | 1,727 | 476,579,709 | 2,478,452 | 15.2 |
| gu | 149,466 | 773 | 200,828,610 | 1,093,737 | 19.7 |
| hi | 171,199 | 889 | 196,159,928 | 999,092 | 13.8 |
| kn | 173,555 | 873 | 165,406,826 | 823,729 | 29.9 |
| ml | 173,825 | 878 | 140,315,042 | 702,425 | 37.8 |
| mr | 174,485 | 869 | 180,317,701 | 860,857 | 18.4 |
| or | 188,737 | 886 | 187,704,674 | 980,068 | 19.1 |
| pa | 149,398 | 746 | 203,046,650 | 1,063,292 | 10.4 |
| ta | 149,525 | 786 | 160,474,332 | 821,678 | 30.9 |
| te | 174,147 | 892 | 195,144,749 | 1,110,009 | 34.6 |
| ur | 191,621 | 1,022 | 321,752,220 | 1,724,684 | 12.0 |
| **all** | **2,298,196** | **11,660 (0.507%)** | **2,783,830,088** | **14,516,321** | 270 worker-min |

Token counts include one `<|endoftext|>` per document (2,798,346,409 in total).

Pre-registered checks, recomputed from `summary.json` (not only read from the report):
- **Input identity: PASS.** Manifest sha256 `73487435…caf39` (D-045). All 13 input files were
  re-hashed on the PC and matched it.
- **Token identity: PASS** for all 13 files: train + val − docs = the EXP-036 exact count;
  in total 2,798,346,409 − 2,298,196 = **2,796,048,213**, equal to the manifest; documents and
  characters per file also equal the manifest.
- **v2 = v1 on ordinary text: PASS on every document.** Each document's v2 `encode_ordinary`
  count equalled its recorded EXP-036 (v1) count (the run stops at the first difference).
- **File re-read: PASS.** Every file is exactly 2 bytes per token (5,596,692,818 bytes in
  total); `<|endoftext|>` count = documents; 0 other special ids; both splits end with
  `<|endoftext|>`.
- **Exact decoding: PASS** on 1,750 sampled documents (the first 100 of each file + every
  5,000th).
- **No record id in both splits: PASS** (record ids unique in every file).
- **Split and bytes identical on a rerun:** shown in tests (`tests/test_pack.py`); the PC ran
  once. The sha256 of each file's validation id list is recorded for later comparison.
- Validation share per language 0.469%–0.533% (target 0.5%; the rule depends only on the id).

**Runtime vs estimate:** 2 h 35 min, about 15 min above the upper end of the sandbox-based
estimate (40 min – 2 h 20 min). The Dravidian-script files were the slowest (ml 37.8, te 34.6,
ta 30.9, kn 29.9 min), consistent with more distinct words (fewer cache hits).

**Limits (unchanged from EXP-036):** validation documents are random documents of the same
languages and sources, so validation loss measures in-distribution fit only; the protected
held-out suite stays the benchmark. Near-duplicates were removed within each file, not across
files. The token files exist only on the founder's laptop (5.6 GB); `manifest.json` gives the
sha256 of each so a copy or a rebuild elsewhere can be verified.

**Proposed decision (founder):** D-046: Frontier Tokenizer v2 is the tokenizer for all training
on v2 data (v1 stays frozen for reproducing EXP-029–033), and the EXP-037 packed files
(`evals/results/EXP-037/manifest.json`) are the canonical training input for v2-slice1, with the
validation split fixed and never trained on. Accepting it starts no training.

**Status:** complete (13/13 files packed; all pre-registered checks pass; 2,796,048,213 tokens + 2,298,196 end-of-text markers)
D-046 is the founder's decision (proposed). No training on v2 without its own approved plan.

### EXP-038 — Step 10: first GPU training bring-up (free Kaggle T4): correctness, speed and memory, and a first real-data run
**Date:** 2026-09-30 · **Status:** proposed (needs the founder's "approve EXP-038" before any code)

**Purpose (why this serves the mission):** every model so far was trained on a CPU. Before any
serious model (step 12), the training code must be shown to run **correctly** on a GPU, and we
must **measure** its speed and memory, because every later cost estimate depends on those numbers
(ROADMAP stage 4: "measure, don't assume"). The mixed precision, `torch.compile` and GPU
checkpointing paths in `src/frontier_ai/engine/trainer.py` are written but have never run on a
GPU. The GPU is Kaggle's free NVIDIA T4 (16 GB), which the founder can use now
(`docs/compute_options_and_costs.md`). This does not wait for the collaborator's report, and the
same script can later run on his GPU. **This is bring-up, not a serious model.**

**Where it runs:** one private Kaggle script session, started from the founder's laptop by one
typed PowerShell line through the Kaggle API (a token he creates once; it stays on his PC and is
never put in the repo or the chat).
- The session runs on Kaggle's servers, so the laptop does **not** have to stay on.
- A second typed line the next day downloads the small results, commits
  `evals/results/EXP-038/` only and pushes.
- **One T4, single-GPU only** (no distributed training; that is step 13).
- Hard limit 6 GPU-hours of the ~30 free hours a week. Expected about 2.5–3 h (NOT VERIFIED).

**Data:** only the Hindi file of the EXP-037 packed data (`hi.bin`, 197,159,020 tokens ≈ 394 MB,
plus `hi.meta.json`), uploaded once as a **private** Kaggle dataset with the CC-BY-4.0
attribution.
- Its sha256 is checked against `evals/results/EXP-037/manifest.json` on the laptop before the
  upload, and again on Kaggle before training; any mismatch stops the run.
- One language only, so **no mixing ratio is chosen**.
- The validation split (D-046) is used only for evaluation.

**Model:** the D-043 architecture (the EXP-B baseline: learned positions, SwiGLU, RMSNorm, tied
embeddings) with Frontier Tokenizer v2 (vocab 32,896). Three bring-up sizes, used only to measure
the system; none is a chosen model size:
- **S** = 4 layers × 128 wide, context 128 (the EXP-B model, about 5.3 M parameters);
- **M** = 8 × 384, context 512 (about 32 M);
- **L** = 12 × 768, context 1,024 (about 139 M).

**Part 0 — environment (recorded):** GPU name and memory, compute capability, driver, CUDA and
PyTorch versions. The repository's model and trainer tests must pass on the Kaggle image before
anything else runs.

**Part 1 — correctness (pre-registered pass/fail; any failure is reported, and later parts still
run so the evidence is complete):**
1. **CPU = GPU.** Model S, fp32, the same seed and the same batches, 50 steps on Kaggle's CPU and
   on the T4. Pass if every step's training loss differs by ≤ 1e-3.
2. **Mixed precision is safe.** Model M, 300 steps, fp32 vs fp16 autocast with GradScaler (T4 has
   no bf16). Pass if the final validation loss differs by ≤ 2% (relative), no loss is NaN or
   infinite, and ≤ 5% of steps are skipped by the scaler.
3. **Checkpoint and resume.** Model M, fp32, deterministic mode: 200 steps straight, and 100 steps
   → save → load in a fresh process → 100 more. Pass if the losses of steps 101–200 differ by
   ≤ 1e-5.
4. **`torch.compile`.** Model M, fp16, 300 steps with and without compile. Pass if the final
   validation loss differs by ≤ 2%. If compile does not work on the T4, that is recorded (not a
   failure of the run); eager mode is then used.

**Part 2 — measurement (numbers only, no pass/fail):** for S, M and L in fp32 and fp16 (and fp16 +
compile where it works), 100 timed steps after warm-up with the largest power-of-two batch that
fits. Recorded:
- tokens/second, step time and peak GPU memory;
- model FLOPs utilisation (MFU), against the T4's 65 TFLOP/s fp16 peak (8.1 fp32).

These numbers replace the assumed 25% in `docs/compute_options_and_costs.md`.

**Part 3 — first real-data run:** model M, fp16, on the Hindi training split. It trains for one
pass (≈196 M tokens) or 90 minutes, whichever comes first, and evaluates on the Hindi validation
split every 500 steps (bits per byte from the exact byte counts in `hi.meta.json`).
- Checks: no NaN or infinite loss; the final validation loss is below the first one; the
  validation loss curve, bits per byte and tokens/second are recorded.
- 5 fixed Hindi prompts are completed by the model and saved. They are for a look only: no
  quality claim is made from them.
- This bits-per-byte figure is **not comparable** to earlier models (different data, tokenizer
  and split).

**Committed evidence:** `evals/results/EXP-038/`: `summary.json`, `SUMMARY.txt`, `samples.jsonl`
(model outputs only, no corpus text), and the environment record. Checkpoints stay on Kaggle and
are not published.

**Code to be built after approval** (tested in the sandbox on CPU; the Kaggle parts cannot be
tested here):
- `scripts/gpu_bringup.py` (Parts 0–3, device-agnostic, also runnable on the collaborator's GPU);
- a small Kaggle script that clones this branch at a pinned commit, installs it without changing
  Kaggle's PyTorch, and runs `gpu_bringup.py`;
- `scripts/run_kaggle_exp038.ps1`, which checks the file hash, creates or updates the private
  dataset, pushes the script session and later fetches, checks and publishes the results.

The Kaggle API details (token type, how a T4 is requested) come from its documentation and are
**NOT VERIFIED** against the founder's account until the first run.

**Not in scope:** a serious or long training run (step 12), choosing a model size or a data mix
(step 11 and later), multi-GPU or distributed training (step 13), the collaborator's GPU (the
same script, later), any spending.

**Status:** proposed — founder approval needed before any code.

**Approved (2026-09-30, founder: "approve EXP-038") and implemented (no GPU result yet):**
- Trainer fix found while building Part 1.3: a resumed run did NOT draw the same batches as an
  uninterrupted run (the batch generator state and the fp16 scaler state were not saved). Each
  checkpoint now also stores `trainer_state.pt`; older checkpoints still load. The new test
  failed on the old trainer and passes now. fp16 scaler-skipped steps are counted and saved.
- `scripts/gpu_bringup.py` (Parts 0–3); `--smoke` runs every part on a CPU in about a minute
  (`tests/test_gpu_bringup.py`). Correctness runs 1.1 and 1.3 use the deterministic math
  attention kernel; the timings in Part 2 use the normal kernels. MFU uses
  6·N + 12·layers·width·context FLOPs per token (N without the position table) against the T4
  data-sheet peaks (65 fp16 / 8.1 fp32 TFLOP/s). Part 3 uses compile only if Part 1.4 passed.
- `scripts/run_kaggle_exp038.ps1` + `scripts/kaggle/exp038_kernel.py` (static tests in
  `tests/test_kaggle_exp038.py`; the PowerShell runner could not be executed in the sandbox).
- Kaggle API behaviour (token, `machine_shape`, status texts) remains **NOT VERIFIED** until the
  first run on the founder's account.

**Status:** approved — code ready; waiting for the founder to launch the Kaggle run.

**Results (Kaggle run 2026-09-30, results commit `740fc16`; checked in the sandbox 2026-10-01).**
All numbers below are copied from `evals/results/EXP-038/summary.json`.
- **Where it ran:** a private Kaggle script session on commit `57a53ef`, 74.1 minutes, well
  under the 6 GPU-hour cap. It used a Tesla T4 (14.56 GB usable), driver 580.159.04, PyTorch
  2.10.0+cu128, Python 3.12.13. Kaggle attached **two** T4s (`device_count` 2); only the first
  was used, as planned. PyTorch reports `bf16_supported: true` on this sm75 card, but the T4 has
  no bf16 tensor cores, so fp16 stays the plan for T4s.
- **Part 0:** the sha256 of `hi.bin` matches the EXP-037 manifest (`9e9c7a4b…`); the model and
  trainer tests passed on Kaggle (40 passed).
- **Part 1, correctness (pre-registered tolerances):**

  | Check | Result | Measured | Tolerance |
  |---|---|---|---|
  | 1. CPU = GPU (S, fp32, 50 steps) | **FAIL** | max per-step loss difference 1.81e-3 | ≤ 1e-3 |
  | 2. fp16 vs fp32 (M, 300 steps) | PASS | final val 3.9879 (fp32) vs 4.0416 (fp16): 1.35%; 0 of 300 steps skipped | ≤ 2%; ≤ 5% skipped |
  | 3. Resume in a fresh process (M, fp32, deterministic) | PASS | steps 101–200 identical (difference 0.0) | ≤ 1e-5 |
  | 4. `torch.compile` vs eager (M, fp16) | PASS | final val 4.0206 vs 4.0600: 0.98% | ≤ 2% |

  Check 1 failed by a factor of 1.8. The script records only the largest difference, not the
  step it happened at. The other evidence (identical resume, fp16 and compile agreement,
  falling validation loss) is consistent with floating-point rounding differences between the
  CPU and GPU maths libraries that grow over training steps, but **that is an interpretation,
  not a measurement**. The pre-registered result stands as FAIL, and the tolerance is not
  changed after the fact. EXP-039 (below) is proposed to settle it.
- **Part 2, speed and memory** (100 timed steps after warm-up; largest power-of-two batch
  that fit; MFU = tokens/s × (6·N + 12·layers·width·context) ÷ data-sheet peak, 65 TFLOP/s fp16
  and 8.1 fp32):

  | Model (params) | Precision | Batch | Tokens/s | Step ms | Peak GB | MFU |
  |---|---|---|---|---|---|---|
  | S (5,276,800) | fp32 | 64 | 66,521 | 123 | 4.56 | 26.6% |
  | S | fp16 | 64 | 98,069 | 84 | 3.91 | 4.9% |
  | S | fp16 + compile | 64 | 233,820 | 35 | 1.84 | 11.6% |
  | M (31,709,568) | fp32 | 16 | 14,146 | 579 | 6.92 | 36.3% |
  | M | fp16 | 32 | 40,802 | 402 | 10.57 | 13.1% |
  | M | fp16 + compile | 64 | 69,127 | 474 | 11.11 | 22.1% |
  | L (139,315,968) | fp32 | 8 | 3,401 | 2,409 | 13.02 | 39.6% |
  | L | fp16 | 8 | 13,088 | 626 | 10.00 | 19.0% |
  | L | fp16 + compile | 16 | 17,870 | 917 | 11.79 | 26.0% |

  fp32 MFU is measured against the much lower fp32 peak, so the fastest setting at every size
  is fp16 + compile: 3.5× (S), 4.9× (M) and 5.3× (L) the fp32 tokens/s. Compile also lowered
  memory enough to double the batch for M and L. The 25% MFU assumed in
  `docs/compute_options_and_costs.md` is close to the measured 22% (M) and 26% (L) for one T4.
- **Part 3, first real-data run** (model M, fp16 + compile because check 4 passed; batch 32 ×
  context 512 = 16,384 tokens per step; lr 1e-3, warm-up 200):
  - 10,252 steps = 167,968,768 tokens (86% of one pass over the Hindi training split), in 48.6
    minutes (57,592 tokens/s including the evaluations). The run stopped at the pre-set time
    cap, not at the end of the data.
  - Validation loss **10.590 → 1.923 nats/token**; **0.586 bits per byte** (exact byte counts
    from `hi.meta.json`). It fell at every 500-step evaluation except two small upticks (7,000:
    2.017 after 2.007; 10,000: 1.934 after 1.925), which is within the noise of 50 evaluation
    batches. No NaN or infinite loss; the fp16 scaler skipped 1 step.
  - This bits-per-byte number is **not comparable** to earlier models (different data,
    tokenizer and split) or to published models (different data and evaluation).
  - The 5 samples (`samples.jsonl`, for a look only; no quality claim): the model writes
    fluent-looking Hindi sentences in a news style, but they carry wrong facts (it says both
    Delhi and Bhopal are the capital of India) and repeat themselves. That is expected of a 32 M
    model after 49 minutes of training.
- **Engineering found during the build:** the trainer did not restore the batch generator and
  the fp16 scaler on resume (fixed in `41fe338`; check 3 above confirms the fix on a GPU). The
  runner's final push was rejected once because the branch had moved while the GPU ran; the
  founder pulled with `--rebase` and pushed, and nothing was lost.

**Status:** complete — 3 of 4 correctness checks pass; check 1 (CPU = GPU) FAILED its tolerance (1.81e-3 > 1e-3), follow-up EXP-039 proposed; speed, memory and the first real-data run are recorded.

### EXP-039 — Step 10 follow-up: is the CPU-vs-GPU gap of EXP-038 rounding or a real difference?
**Date:** 2026-10-01 · **Status:** proposed (needs the founder's "approve EXP-039" before any code)

**Purpose (why this serves the mission):** EXP-038 check 1 failed: over 50 steps, the losses of
the same model on the CPU and on the T4 differed by up to 1.81e-3 (tolerance 1e-3). Before
step 11 spends GPU time on scaling experiments, we must know whether the GPU computes **the
same function** as the tested CPU code (differences then come only from rounding), or whether
something really differs. The other three checks passed, so rounding is the likely explanation,
but it is not yet measured.

**Method.** The EXP-038 check 1 setup is kept exactly (model S, the same seed, batches and
learning-rate schedule, 50 steps, math attention, deterministic mode), with two changes:
- **Part A: float64 on both devices.** The model, optimizer state and loss run in 64-bit
  floats on the CPU and on the T4. Rounding error in 64-bit is about 10⁹ times smaller than in
  32-bit, so if the gap comes from rounding it shrinks to about 1e-11 (a rough estimate from the
  measured fp32 gap), while a real difference in the computation (a different operation, a bug
  in a GPU code path) would still show at 1e-4 or more.
  **Pre-registered pass:** every one of the 50 steps differs by ≤ 1e-8.
- **Part B: the per-step curve in float32 (numbers only).** The same run in fp32, now recording
  the difference at **every** step, including step 1 (before any weight update, so it is pure
  forward-pass rounding). This shows whether the gap starts tiny and grows.

**What each outcome means (fixed now):**
- **Part A passes:** the GPU path computes the same function; the EXP-038 gap is rounding
  growth. EXP-038 check 1 stays recorded as FAIL. I would then propose a decision (D-047) to
  close step 10 and to use the float64 comparison (≤ 1e-8) for future device checks, with fp32
  differences recorded as numbers only.
- **Part A fails:** a real difference exists. No step-11 training until it is found and fixed,
  with a new pre-registered check.

**Where it runs:** Kaggle, one T4, through the same one-line runner; about 15 GPU-minutes
expected (NOT VERIFIED); hard cap 1 GPU-hour. The private dataset from EXP-038 is reused (the
sha256 check still runs); no new upload.

**Code to be built after approval** (tested in the sandbox on CPU):
- a `--part cpu-gpu-diagnostic` mode in `scripts/gpu_bringup.py` for Parts A and B (float64
  via `model.double()` inside the existing trainer, which keeps the optimizer's references);
- `scripts/run_kaggle_exp038.ps1` generalised with `-Exp EXP-039` (its own state, kernel
  `frontier-exp039` and kernel template); plus a fix for the push collision seen in EXP-038: if
  the push is rejected, it pulls with `--rebase` once and pushes again, and still commits only
  `evals/results/<EXP>/`.

**Committed evidence:** `evals/results/EXP-039/summary.json` and `SUMMARY.txt` (per-step
differences for Parts A and B, plus the environment record).

**Not in scope:** step 11 scaling experiments, changing any EXP-038 result or tolerance, the
collaborator's GPU, multi-GPU work, any spending.

**Status:** proposed — founder approval needed before any code.

**Approved (2026-10-01, founder: "approve EXP-039") and built (no GPU result yet):**
- `scripts/gpu_bringup.py --part cpu-gpu-diagnostic` runs Part 0 (environment, data sha256,
  tests) and then Parts A and B exactly as pre-registered: the same model S setup as EXP-038
  check 1 (seed, batches, schedule, 50 steps, math attention, deterministic mode).
- **Two things found while building it, both fixed before any run:**
  1. The training log rounds losses to 5 decimals, which would have hidden any float64
     difference below 1e-5 (Part A would have passed automatically). The diagnostic therefore
     reads the **exact** loss of every training step from the model output through a forward
     hook. EXP-038's check 1 was unaffected (its tolerance, 1e-3, is far above 1e-5).
  2. The model computed RMSNorm and the loss in float32 even when the model itself was float64,
     which would have left float32 rounding inside Part A. Both now keep float64 when the model
     is float64; fp16, bf16 and fp32 behave exactly as before (new tests).
- `scripts/run_kaggle_exp039.ps1` (the EXP-038 runner with `-Exp EXP-039`; same private
  dataset, no new upload) with a push retry: if the push is rejected, it runs
  `git pull --rebase` once and pushes again.
- On a CPU (smoke test) float64 really runs end to end: its losses differ from float32 by
  about 1e-6, and without a GPU the verdict is recorded as n/a, never PASS.

**Status:** approved — code ready; waiting for the founder to launch the Kaggle run.

**Results (Kaggle run 2026-10-01, results commit `d038f2a`; checked in the sandbox 2026-10-01).**
All numbers below are copied from `evals/results/EXP-039/summary.json`.
- **Where it ran:** private Kaggle script session on commit `e1df707`, Tesla T4, PyTorch
  2.10.0+cu128; 3.1 minutes of the 1 GPU-hour cap. Part 0: `hi.bin` sha256 matches; the
  model/trainer tests passed on Kaggle (40 passed).
- **Part A, float64 (pre-registered pass ≤ 1e-8 at every step): PASS.** Both devices ran in
  float64 (recorded dtypes float64 / float64). The largest per-step difference was **5.51e-10**
  (step 50). Step 1 (before any update): 1.78e-15. Through step 38 every difference was
  ≤ 2.7e-15 (12 of 50 steps exactly 0); then it grew: 5.8e-13 at step 39, 2.2e-12 at 40,
  6.7e-11 at 45, 5.5e-10 at 50.
- **Part B, float32 (numbers only):** largest difference **1.807e-3** (step 50), which reproduces
  EXP-038's 1.81e-3. Step 1: exactly 0. Through step 39 every difference was ≤ 5.7e-6; then
  1.1e-4 at step 40, 1.4e-4 at 45, 1.8e-3 at 50.
- **Reading (what the numbers support):**
  - In float64 the CPU and the T4 agree to 5.5e-10, about 3 million times closer than in
    float32 (1.8e-3). A real difference in what the GPU computes would not shrink like this, so
    the GPU path computes the same function as the tested CPU code. EXP-038's check 1 gap is
    rounding that the training run amplifies.
  - The amplification starts at almost the same point (steps 39–40) in both precisions. That
    points to a sensitive phase of this particular training run, not to the device. Which
    property of the run causes it was not measured.
  - EXP-038's check 1 stays recorded as FAIL against its pre-registered 1e-3 tolerance.
    Lesson: an fp32 step-by-step tolerance over 50 training steps was too tight to be a useful
    test, because ordinary rounding alone can exceed it.

**Status:** complete — Part A PASS (float64 CPU = GPU within 5.51e-10 ≤ 1e-8); Part B recorded (float32 1.807e-3); D-047 proposed to close step 10.

### EXP-040 — Step 11, phase 1: learning rate and the RoPE + GQA-2 question at GPU scale, on all 13 languages
**Date:** 2026-10-01 · **Status:** proposed (needs the founder's "approve EXP-040" before any code)

**Purpose (why this serves the mission):** step 11 has to tell us how big a model our data and
compute can support, which decides the size of the first serious model (step 12). Before a
scaling ladder (phase 2) is worth its GPU hours, two settings that every run in it shares have to
be measured at GPU scale:
- **the learning rate**: D-043 point 4 says the toy-scale optimum is not carried over, and
  EXP-038 used 1e-3 without testing;
- **the architecture**: D-043 point 2 makes RoPE + GQA-2 (rotary positions, 2 key-value
  heads) the first question at GPU scale. It was better in every comparison at toy scale but
  not decisively (EXP-033).

Running the ladder first and changing these afterwards would waste it. This is phase 1 of 2. It
does not pick a model size.

**Plan for step 11 (for orientation; only phase 1 is asked for now):**
- **Phase 1, EXP-040 (this proposal):** learning rate + RoPE + GQA-2 at one size. About 7 GPU-hours.
- **Phase 2, EXP-041 (later, its own approval):** a compute-optimal ladder in the style of
  Chinchilla's IsoFLOP profiles (Hoffmann et al., 2022): about 3 compute budgets × 4–5 model sizes,
  each run with its own learning-rate schedule. It estimates the best tokens-per-parameter for
  **our** data and, from that, which model size the 2.78 B training tokens support. This meets
  the Stage 4 exit criterion ("at least one scaling-law sanity plot from runs we performed").
  About 15–25 GPU-hours over one or two weeks (NOT VERIFIED).
- Then a decision record on the step-12 model size, justified by those measurements (MASTER_CONTEXT
  §22: no arbitrary target size).

**Where it runs:** one private Kaggle session on one T4, through the same runner (`-Exp
EXP-040`). Hard cap **9 GPU-hours** (sessions end at 12 h), out of about 30 free hours a week.
Expected about 7 hours (NOT VERIFIED). Each run's result is saved as soon as it ends, so a cut-off
session still leaves the finished runs.

**Data:** all 13 EXP-037 files (D-046; 2,783,830,088 training tokens, 5,596,692,818 bytes),
uploaded once from the laptop as a new **private** Kaggle dataset. Every file's sha256 is checked
against the manifest on the laptop and again on Kaggle.
- **Sampling:** each training sequence comes from a language chosen in proportion to its
  training tokens (English 17.1%, Urdu 11.6%, the other 11 languages 5.0–7.3% each), which is the
  same as reading the whole corpus uniformly. This is the corpus as it is, used as a neutral
  reference for this comparison only; it is **not** a chosen data mix. Mixtures are compared in
  their own experiment later.
- Validation splits (D-046) are used only for evaluation.

**Models** (D-047 settings: fp16 + GradScaler + `torch.compile`; the EXP-038 M shape: 8 layers ×
384 wide, 6 query heads, context 512, SwiGLU, RMSNorm, tied embeddings, vocab 32,896):
- **baseline** = D-043 (learned positions, 6 key-value heads): 31,709,568 parameters;
- **candidate** = RoPE + GQA-2 (rotary positions, 2 key-value heads): 29,940,096 parameters.
  It is 5.6% smaller, and that is part of the package being tested, as in EXP-033.

**Training (identical for every run):** 100 M tokens per run (6,104 steps × 32 sequences × 512
tokens, about 30 minutes each on the T4 by EXP-038's speed, NOT VERIFIED); warm-up 200 steps,
cosine decay to 10% of the peak, AdamW (betas 0.9/0.95, weight decay 0.1), gradient clipping
1.0. The batch order depends only on the seed, so both architectures see the same data.

**Grid:**
- **Grid A:** 2 architectures × 4 peak learning rates {5e-4, 1e-3, 2e-3, 4e-3} × seed 1 = 8 runs.
- **Grid B:** seed 2 for both architectures at the **two learning rates where the baseline did
  best in grid A** = 4 runs.

That is 12 runs in total; grid A runs first.

**Measurements:**
- Primary: final validation bits per byte, **mean over the 13 languages with equal weight**
  (each language's full validation split, exact byte counts). Equal weight is used because the
  goal is all Indian languages, not the largest ones.
- Also recorded: the token-weighted mean, per-language bits per byte (English included), the
  validation curve every 1,000 steps, tokens/s, peak memory, skipped fp16 steps and the GPU
  time per run.

**Part 0 (before any run):**
- the environment record; all 13 sha256 checks; the model and trainer tests;
- the D-047 float64 CPU = GPU check (≤ 1e-8 per step, 50 steps) for the **RoPE + GQA-2** maths,
  which has never run on a GPU. If that check fails, the candidate runs are skipped, the
  baseline runs still go ahead, and the failure is reported.

**Pre-registered rules (fixed before any result):**
1. **Learning rate:** each architecture's step-11 learning rate at this size is the one with the
   lowest primary metric in grid A. If that is 5e-4 or 4e-3 (the edge of the range), the
   optimum may lie outside the range: this is reported, and phase 2 widens the range.
2. **Seed noise:** the noise is the largest |seed 1 − seed 2| of the primary metric over the 4
   (architecture, learning rate) pairs of grid B.
3. **Architecture:** at each grid-B learning rate, the difference is baseline mean minus
   candidate mean over the 2 seeds. It counts as BETTER for the candidate if the difference is
   larger than the noise, WORSE if it is below −noise, and otherwise NO DETECTABLE DIFFERENCE.
   **RoPE + GQA-2 is adopted for phase 2 only if it is BETTER at both learning rates**;
   otherwise the D-043 baseline stays. Either way, a decision record follows.
4. **Stability:** a run with a NaN or infinite loss, or with more than 5% of steps skipped by the
   fp16 scaler, counts as failed at that learning rate and is reported.

**Code to be built after approval** (tested in the sandbox on CPU):
- a multi-file token dataset that samples languages in proportion to their training tokens,
  is deterministic per seed and reports per-language validation;
- `scripts/gpu_lr_arch.py` (Part 0, grids A and B, results after every run; reuses the
  `gpu_bringup.py` helpers);
- a kernel template, and `scripts/run_kaggle_exp038.ps1` extended with `-Exp EXP-040` and the
  13-file dataset (one upload of about 5.6 GB from the laptop, sha256-checked first).

**Committed evidence:** `evals/results/EXP-040/summary.json` and `SUMMARY.txt` (every run's
settings, curves, per-language bits per byte, the verdicts of rules 1–4, and the environment).
Checkpoints stay on Kaggle.

**Not in scope:** choosing a model size or a data mix, the phase-2 ladder, other architecture
changes (MoE, MLA, qk-norm and so on; still deferred by D-043), multi-GPU training, the protected
held-out suite (its text is on the laptop, in v1 tokens), English, math or coding benchmarks (none
exist in the repository yet; MASTER_CONTEXT §22 lists them as later measurements), and any
spending.

**Status:** proposed — founder approval needed before any code.

**Approved and built (2026-10-01):** the founder typed "approve EXP-040". Code as listed above,
no change to the pre-registered plan or rules:
- `src/frontier_ai/data/multi.py`: `MultiTokenDataset` (each sequence's language drawn in
  proportion to its training tokens, deterministic per seed, the Trainer's interface; per-language
  files in `.parts`) and `full_split_loss` (a whole validation split in non-overlapping windows).
- `scripts/gpu_lr_arch.py`: Part 0 (environment, all 13 sha256 checks, tests, the float64 CPU =
  GPU check for RoPE + GQA-2), grids A and B, the full per-language validation after every run
  (on the uncompiled model; bits per byte from each file's exact byte count), the primary metric
  (equal-weight mean of 13), the token-weighted mean, a pooled corpus figure, curves, tokens/s,
  peak memory, skipped steps; `summary.json` + `SUMMARY.txt` rewritten after every run; a run is
  started only if it still fits the 9-hour budget (otherwise "skipped (time budget)"); rules 1–4
  evaluated at the end ("not decided" if grid B is incomplete; the baseline then stays).
- `scripts/gpu_bringup.py`: `make_cfg` accepts architecture overrides and a data seed; the
  environment record and the test run are now shared functions. EXP-038/039 behaviour unchanged.
- `scripts/kaggle/exp040_kernel.py`, `scripts/run_kaggle_exp040.ps1` and `-Exp EXP-040` in
  `scripts/run_kaggle_exp038.ps1`: the 13 bin + meta files from the manifest, every fingerprint
  checked on the laptop, staged with hard links (copy only as a fallback), one upload as the
  private dataset `frontier-v2-tok2-13lang`, up to 90 minutes for Kaggle to process it, up to 11
  hours of waiting for the kernel.
- Tests: `tests/test_gpu_lr_arch.py` (dataset sums, sampling proportions, determinism, the rules,
  a CPU smoke run of all 12 runs on three tiny language files, the missing-file stop, the
  pre-registered plan and parameter counts) and EXP-040 checks in `tests/test_kaggle_exp038.py`.
  The CPU smoke numbers are not results.

Upload time for 5.6 GB from the laptop and the GPU time per run: NOT VERIFIED until the run.

**Status:** approved — code ready; waiting for the founder to launch the Kaggle run.

**Results (2026-10-02; the founder's Kaggle run, code `6f53b00`, results commit `7c6e5e2`,
`evals/results/EXP-040/`):** one Kaggle T4 session, Tesla T4, torch 2.10.0+cu128. The 5.6 GB
upload plus Kaggle's processing took about 31 minutes (22:39 → kernel launched 23:10, laptop
clock). The script ran for 178.1 minutes, within the 9-hour cap.

*Part 0:* 13 of 13 sha256 checks pass; tests pass (40 passed). **The float64 CPU = GPU check
for RoPE + GQA-2 FAILED its pre-registered tolerance: max |loss difference| 1.739e-07 at step 50
(tolerance 1e-8).** Following the plan, all 6 candidate runs were skipped and the baseline runs
went ahead. `complete: false` in the summary means exactly this.

What the per-step numbers show (recorded in `summary.json`; this is an observation, not a new
verdict): steps 1–27 agree to the last digits (differences 0 to 1.8e-15, the size of one
rounding step of a float64 number near 10). From step 28 the difference grows by roughly 1.5×
per step: 4.5e-12 at step 30, 9.7e-10 at step 39, 1.7e-07 at step 50. EXP-039's baseline check
had the same shape but started growing later (step 38) and reached only 5.5e-10. A wrong formula
on the GPU would show up from the first steps; this pattern looks like a last-digit rounding
difference being amplified by training, the effect EXP-039 already described. **That is an
interpretation, not a measurement.** This check also differed from EXP-039's in two ways at
once (the architecture AND the data: 13 languages instead of Hindi), so the earlier divergence
cannot be pinned on RoPE + GQA-2. The 50-step criterion mixes two things, "does the GPU compute
the same function" and "how fast does this training run amplify tiny differences", and we cannot
tell them apart with it.

*Baseline runs (D-043 architecture, 31,709,568 parameters, 100,007,936 tokens each):*

| run | grid | mean bpb (13 languages, equal weight) | token-weighted | minutes | tokens/s | fp16 steps skipped |
|---|---|---|---|---|---|---|
| lr 5e-4, seed 1 | A | 0.9439 | 1.0626 | 29.8 | 60,178 | 0 |
| lr 1e-3, seed 1 | A | **0.9213** | 1.0387 | 29.3 | 61,241 | 0 |
| lr 2e-3, seed 1 | A | 0.9548 | 1.0777 | 29.2 | 61,467 | 0 |
| lr 4e-3, seed 1 | A | 1.0195 | 1.1531 | 28.5 | 62,939 | 1 |
| lr 5e-4, seed 2 | B | 0.9503 | 1.0694 | 29.2 | 61,393 | 1 |
| lr 1e-3, seed 2 | B | **0.9388** | 1.0572 | 29.3 | 61,310 | 0 |

Peak GPU memory was 5.81 GB in every run, and no run failed (rule 4). The per-language table is
in `SUMMARY.txt`. Bits per byte can be compared between runs, **not between languages**: in UTF-8
English needs 1 byte per letter, Urdu's script 2 and the other Indian scripts 3, so English's
higher number (1.83–2.07) does not mean the model is worse at English.

*Pre-registered rules:*
1. **Learning rate, baseline: 1e-3** (lowest in grid A, not at the edge of the range). It is also
   the lowest in seed 2 (0.9388 vs 0.9503 at 5e-4). RoPE + GQA-2: not measured.
2. **Seed noise: not computable** as defined (it needs the 4 grid-B pairs). Observation only: the
   two baseline seed gaps are 0.0064 (lr 5e-4) and 0.0175 (lr 1e-3). That is as large as the gap
   between the two best learning rates (seed means 0.9301 vs 0.9471), so single-seed differences
   of about 0.02 at this scale are not reliable.
3. **RoPE + GQA-2: NOT TESTED; the D-043 baseline stays** until the question is answered.
4. **No failed runs.**

What this gives step 11: a measured learning rate (1e-3) and a measured seed noise scale for the
baseline at 32 M parameters / 100 M tokens, real T4 costs for 13-language training (about 29
minutes and 5.8 GB per run; 61k tokens/s including evaluations), and a working 13-language
pipeline on Kaggle. Still open: the architecture question, and a device check that separates the
GPU's maths from the run's amplification. Follow-up proposed as EXP-041 below. The IsoFLOP ladder,
called "EXP-041" in the plan above, moves to the next free number.

**Status:** complete — baseline learning rate 1e-3 measured (rule 1); the float64 device check for RoPE + GQA-2 FAILED its 1e-8 tolerance (1.739e-07 at step 50), so the candidate was not tested and the baseline stays (rule 3); no failed runs (rule 4); seed noise not computable as pre-registered.

### EXP-041 — Follow-up to EXP-040: a device check that does not compound, then the RoPE + GQA-2 runs
**Date:** 2026-10-02 · **Status:** proposed (needs the founder's "approve EXP-041" before any code)

**Purpose (why this serves the mission):** step 11 still has to settle the architecture before
the scaling ladder spends 15–25 GPU-hours on it. EXP-040 could not test RoPE + GQA-2 because its
float64 check failed. The step-by-step numbers suggest amplified rounding, not wrong maths, but
that is unproven, and the 50-step check cannot tell the two apart. This experiment answers the
maths question directly. Only if the maths passes does it run the 6 missing candidate runs, so
EXP-040's pre-registered comparison can be finished. **The EXP-040 result itself is not
changed.**

**Where it runs:** one private Kaggle T4 session through the same runner (`-Exp EXP-041`), with
the 13-language dataset that is already uploaded (no new upload). Hard cap **5 GPU-hours**;
about 3.8 expected (6 × 29.5 + 1 × 29.3 minutes measured in EXP-040, plus checks; NOT VERIFIED).

**Part A — the maths check (pre-registered gate):** the S-shape models of the EXP-040 check, on the
same 13-language data, in float64.
- Train 50 steps on the CPU and keep the parameters before steps 1, 11, 21, 31, 41 and 50.
- For each of those 6 states, load **identical** parameters on the CPU and on the GPU, feed the
  **identical** batch and compute the loss and every gradient once on each device. Nothing is
  carried from one step to the next, so nothing can be amplified.
- **Pass:** at all 6 states, |loss difference| ≤ 1e-12 and ‖gradient difference‖ / ‖gradient‖
  ≤ 1e-10. Float64 rounding is about 1e-16, and EXP-039/EXP-040 measured ≤ 1.8e-15 on losses near
  10 before any amplification. A real formula difference (a wrong rotation, wrong head grouping)
  would show up at 1e-3 or larger. Both limits leave a margin of 10,000 or more on both sides.
- Run for **both** architectures. The baseline is the control: **if the baseline fails Part A,
  the check itself is wrong and the experiment stops** without candidate runs.
- **Reported, not a gate:** the EXP-040-style 50-step trajectory for the baseline on the
  13-language data. This shows whether the baseline would also have "failed" 1e-8 on this data,
  i.e. whether the old criterion depends on the data.

**Part B — only if RoPE + GQA-2 passes Part A:** the 6 candidate runs EXP-040 skipped, exactly as
pre-registered there (lr 5e-4, 1e-3, 2e-3, 4e-3 with seed 1; 5e-4 and 1e-3 with seed 2; same
data order, settings and evaluation). Plus **one baseline control run** (lr 1e-3, seed 1), to
measure how much a rerun in another session differs (reported, not a gate). EXP-040's rules 1–4
are then evaluated **unchanged** on the 12 runs: 6 baseline runs from EXP-040 and 6 candidate runs
from here. The baseline runs from EXP-040 are reused because the baseline code path is unchanged
since `6f53b00`.

**If Part A fails for RoPE + GQA-2 (and passes for the baseline):** a real GPU/CPU difference in
the RoPE or GQA code is found. It is reported, the candidate stays untested, and the next step is
a code investigation, not more GPU runs.

**Decision afterwards:** if Part A behaves as expected, a short decision record (D-048) will
propose replacing D-047's "≤ 1e-8 over 50 steps" device check with Part A's non-compounding
check, with the trajectory kept as a reported diagnostic. That needs the founder's decision; it
is not decided by this experiment. Then the architecture decision follows from rule 3.

**Code to be built after approval** (tested on CPU in the sandbox): a `--part` switch in
`scripts/gpu_lr_arch.py` (one-step check, trajectory diagnostic, candidate-only runs and a
control run, merging with `evals/results/EXP-040/summary.json`), `scripts/kaggle/exp041_kernel.py`,
`scripts/run_kaggle_exp041.ps1` (`-Exp EXP-041`, reusing the dataset), and tests.

**Committed evidence:** `evals/results/EXP-041/summary.json` and `SUMMARY.txt`.

**Not in scope:** changing EXP-040's verdicts or rules, the IsoFLOP ladder, other architecture
changes, bf16, multiple GPUs, and any spending.

**Status:** proposed — founder approval needed before any code.

**Approved and built (2026-10-02):** the founder typed "approve EXP-041". Built as proposed, with
no change to the plan, limits or rules:
- `scripts/gpu_lr_arch.py --part followup`:
  - Part A: `one_step_check` keeps the parameters and batch before steps 1, 11, 21, 31, 41 and 50 of
    a float64 CPU run and computes one loss and gradient from each on the CPU and on the GPU (pass:
    ≤ 1e-12 and ≤ 1e-10 relative at every state). The baseline runs first, as the control, then
    RoPE + GQA-2.
  - The baseline's 50-step trajectory is reported, not used as a gate.
  - Part B runs only if both pass: the 6 candidate runs at the same seeds, so the batches are the
    same as EXP-040's baseline runs, then the control run (lr 1e-3, seed 1).
  - Rules 1–4 are computed from the code of EXP-040, unchanged, on EXP-040's 6 baseline runs, read
    from the committed `evals/results/EXP-040/summary.json`; a summary that does not match is
    refused.
- The default mode (EXP-040) behaves as before; its trajectory check became a shared function.
- On the S check shape (4 query heads), the baseline has 4 key-value heads (plain multi-head
  attention, as in EXP-039), and the candidate keeps 2.
- `scripts/kaggle/exp041_kernel.py` (cap 5 GPU-hours), `scripts/run_kaggle_exp041.ps1` and
  `-Exp EXP-041` in `scripts/run_kaggle_exp038.ps1`: the same private 13-language dataset (the
  fingerprints are checked again, nothing is uploaded), waiting up to 7 hours.
- Tests: identical states give identical CPU results. A change of one part in a million in a
  single weight matrix exceeds both limits, so the check can see a real difference. A CPU smoke
  run of the whole follow-up (it reuses a smoke EXP-040 summary), the refusal of a mismatched
  summary, the real EXP-040 summary being accepted, and static runner/kernel checks. The CPU smoke
  numbers are not results.

**Status:** approved — code ready; waiting for the founder to launch the Kaggle run.

**Results (2026-10-02; the founder's Kaggle run, code `f0738fe`, results commit `6ee0bd5`,
`evals/results/EXP-041/`):** one Kaggle T4 session, Tesla T4, torch 2.10.0+cu128. The script ran
for 207.1 minutes (cap 5 hours). `complete: true`. The laptop runner was stopped once by accident
(Ctrl+C) and started again. The GPU kernel was not affected: it had already been launched and the
restarted runner only waited for it.

*Part A, the pre-registered gate: PASS for both architectures.*

| architecture | max \|loss difference\| | max relative gradient difference | states |
|---|---|---|---|
| baseline (the control) | 8.9e-16 | 1.8e-15 | before steps 1, 11, 21, 31, 41, 50 |
| RoPE + GQA-2 | 1.8e-15 | 2.7e-15 | the same |

The limits were 1e-12 and 1e-10. Both architectures agree to about the size of one float64
rounding step, including at steps 31, 41 and 50, where EXP-040's trajectory had already drifted
apart. **The GPU computes the same RoPE + GQA-2 function as the CPU code.**

*Reported, not a gate:* the baseline's 50-step float64 trajectory on the 13-language data stayed
within 5.3e-15 (on Hindi in EXP-039 it reached 5.5e-10). So whether a 50-step trajectory drifts
depends on the architecture and the data; it says nothing about the device. This supports
EXP-040's interpretation (rounding amplified by training). **EXP-040's FAIL stays recorded as
it is.**

*Part B: the 6 RoPE + GQA-2 runs (29,940,096 parameters, 100,007,936 tokens each, the same seeds
and therefore the same batches as EXP-040's baseline runs).*

| lr | seed | baseline (EXP-040) | RoPE + GQA-2 | baseline − candidate | languages where the candidate is lower |
|---|---|---|---|---|---|
| 5e-4 | 1 | 0.9439 | 0.9114 | +0.0326 | 13 of 13 |
| 1e-3 | 1 | 0.9213 | **0.8706** | +0.0507 | 13 of 13 |
| 2e-3 | 1 | 0.9548 | 0.8801 | +0.0748 | 13 of 13 |
| 4e-3 | 1 | 1.0195 | 0.9715 | +0.0480 | 13 of 13 |
| 5e-4 | 2 | 0.9503 | 0.9059 | +0.0444 | 13 of 13 |
| 1e-3 | 2 | 0.9388 | **0.8605** | +0.0783 | 13 of 13 |

The values are the 13-language mean validation bits per byte, equal weight. Peak memory was 5.78 GB.
Speed was 61,946–64,628 tokens/s (the baseline: 60,178–62,939), 28.1–29.2 minutes per run. One
fp16 step was skipped at lr 4e-3; there were no failed runs.

*Control run (baseline lr 1e-3, seed 1, rerun in this session):* 0.9197 vs 0.9213 in EXP-040, a
difference of −0.0016. That is about a tenth of the seed noise, so combining EXP-040's baseline
runs with this session's runs is sound.

*Pre-registered rules (EXP-040's, unchanged):*
1. **Learning rate: 1e-3 for both architectures** (not at the edge of the range).
2. **Seed noise: 0.0174** bits per byte (the largest seed gap: the baseline at lr 1e-3; the
   candidate's gaps were 0.0055 and 0.0101).
3. **At lr 5e-4: BETTER** (+0.0385 over 2 seeds, noise 0.0174). **At lr 1e-3: BETTER** (+0.0645).
   **RoPE + GQA-2 is BETTER at both learning rates → adopted for phase 2.**
4. **No failed runs.**

**What this does and does not show:**
- It does show that at 30 M parameters and 100 M tokens on our 13-language corpus, RoPE + GQA-2 is
  better than the D-043 baseline by about 7% in bits per byte at the best learning rate. It was
  lower in every language and in every one of the 6 paired runs, while being 5.6% smaller and
  about 3% faster. Its key-value cache for generation is 3× smaller.
- It does not show which of the two parts gives the gain (they were tested as one package, as
  pre-registered). It does not show that the gain holds at larger sizes: that is what the phase-2
  ladder runs on, and it can show otherwise. The float64 check verifies the maths, not the fp16
  kernels; the fp16 runs above trained without failures.
- The decision record that rule 3 requires is proposed as D-048. The IsoFLOP ladder becomes
  EXP-042 and needs its own approval after D-048.

**Status:** complete — Part A PASS for both architectures (≤ 2.7e-15 against limits 1e-12 / 1e-10); RoPE + GQA-2 BETTER at both grid-B learning rates (+0.0385 and +0.0645 bits per byte, noise 0.0174) and adopted for phase 2 by rule 3; learning rate 1e-3 for both; no failed runs; D-048 proposed.

### EXP-042 — Step 11, phase 2: a compute-optimal ladder (IsoFLOP profiles) on all 13 languages
**Date:** 2026-10-02 · **Status:** proposed (needs the founder's "approve EXP-042" before any code)

**Purpose (why this serves the mission):** the first serious model (step 12) has to be the right
size for our data and compute: too big and it is starved of data, too small and it wastes the
corpus. MASTER_CONTEXT §22 forbids picking a size by fashion. This experiment measures, on **our**
corpus, how the best model size and the best number of training tokens grow with compute. That
growth law is what a size decision can rest on. It also meets the Stage 4 exit criterion: "at
least one scaling-law sanity plot from runs we performed". It does **not** choose the step-12
size; a decision record does that afterwards.

**Method:** the IsoFLOP approach of Hoffmann et al. (2022, "Chinchilla"):
- Fix a compute budget C. Train several model sizes, each for exactly the number of tokens that
  spends C.
- Find the size with the lowest validation loss at that budget.
- Repeat at three budgets and fit how that best size grows with C.

**Architecture:** D-048. RoPE, 2 key-value heads, SwiGLU, RMSNorm, tied embeddings, vocab 32,896,
context 512, head size 64. The sizes form one family, with depth = width / 64:

| size | layers × width (heads) | parameters | non-embedding parameters | training FLOPs per token |
|---|---|---|---|---|
| s1 | 2 × 128 (2) | 4.74 M | 0.52 M | 30 M |
| s2 | 4 × 256 (4) | 12.36 M | 3.93 M | 80 M |
| s3 | 6 × 384 (6) | 25.61 M | 12.98 M | 168 M |
| s4 | 8 × 512 (8) | 47.26 M | 30.42 M | 309 M |
| s5 | 10 × 640 (10) | 80.05 M | 59.00 M | 520 M |

These were computed with the repository's model code. FLOPs per token = 6 × parameters + 12 ×
layers × width × context, the formula already used for EXP-038's utilisation figures. The fits
use **non-embedding parameters** (Kaplan et al., 2020): with a 32,896-word vocabulary, embeddings
are most of a small model, and they cost no compute per token except through the output layer,
which the FLOP count includes. Total parameters are reported next to them.

**Budgets and runs** (one seed each). Each budget uses a window of 4 sizes. The windows were placed
around the size that the Chinchilla rule of thumb (about 20 tokens per parameter) would predict,
only so that the true optimum is likely to fall inside; that rule is **not** assumed in the
result.

| budget C | sizes | tokens per run | runs |
|---|---|---|---|
| C1 = 1e16 FLOPs | s1–s4 | 333 M (s1) … 32 M (s4) | 4 sizes × 3 learning rates = 12 |
| C2 = 3e16 FLOPs | s1–s4 | 1.0 B (s1) … 97 M (s4) | 4 |
| C3 = 1e17 FLOPs | s2–s5 | 1.24 B (s2) … 192 M (s5) | 4 |

- **Learning rate (D-048 point 2: 1e-3 is the centre of a sweep at each size):** at C1, every
  size runs at 5e-4, 1e-3 and 2e-3, and each size's best is used at C2 and C3. s5 is not in C1,
  so it uses s4's best (written down now, before any result). If a size's best is at the edge
  (5e-4 or 2e-3), this is reported.
- **Identical for every run:** 16,384 tokens per step (32 × 512; s5 as 16 × 2 accumulation steps
  for memory, which is the same maths); warm-up 200 steps; cosine decay to 10% over each run's own
  length; AdamW (0.9, 0.95; weight decay 0.1); clipping 1.0; fp16 + GradScaler + `torch.compile`
  (D-047).
- **Data:** the 13 languages in natural proportion, as in EXP-040/041; the same validation
  splits. The longest run reads 1.24 B of the 2.78 B training tokens, so no text is repeated.
- **Seed:** 1. The seed-noise scale is known from EXP-040/041: up to 0.0174 bits per byte at 100 M
  tokens.

**Measurement:** each run's final validation bits per byte, as the mean of the 13 languages with
equal weight (full validation splits, as in EXP-040). The token-weighted mean, the per-language
values, curves, tokens/s, memory and skipped steps are also recorded.

**Pre-registered analysis (fixed before any result):**
1. **Best size per budget:** fit a parabola to bits per byte against log(non-embedding
   parameters), using each size's best learning rate. Its minimum is N_opt(C), and D_opt(C) is
   the matching token count. A budget counts as **bracketed** only if the minimum lies strictly
   inside the window and the lowest measured point is not at an edge. Otherwise it is reported as
   "not bracketed" and left out of rule 2 (no extra runs in this experiment).
   - Also reported: the **flat region**, i.e. the sizes whose fitted value is within the seed
     noise (0.0174) of the minimum, because near the optimum the curve is flat.
2. **Growth law:** fit N_opt ∝ C^a and D_opt ∝ C^b over the bracketed budgets. At least 2 are
   needed, otherwise "not computable"; with 3, the leftover error is reported. Reported values: a
   and b (Chinchilla found about 0.5 and 0.5, Kaplan about 0.73 and 0.27) and the tokens per
   parameter at each budget, both non-embedding and total.
3. **What our 2.78 B tokens support:** the budget at which D_opt equals 2.78 B, and N_opt there,
   with a low–high range from leaving out one budget at a time. **This is an extrapolation about
   1–1.5 orders of magnitude beyond the largest measured budget and is labelled as such.** It is
   an input to the step-12 decision, not the decision.
4. **Secondary (reported, never a gate):** a fit of loss = E + A / N^α + B / D^β over all runs
   (Chinchilla's approach 3), with numpy only. If it does not converge, that is reported.
5. **Stability:** a run with a NaN or infinite loss, or with more than 5% skipped fp16 steps,
   counts as failed. A budget still gets a fit if at least 3 of its 4 sizes succeeded.
6. **The plot:** bits per byte against non-embedding parameters, one curve per budget, plus the
   N_opt line. It is made in the sandbox from the committed summary.

**Part 0 (every session):** the environment record, the 13 sha256 checks, the tests, and the
one-step float64 check for the D-048 architecture (D-048 point 3; minutes). If the check fails,
that session stops.

**Where it runs, cost and time:**
- **Where:** private Kaggle T4 sessions (single GPU, D-047) on the uploaded 13-language dataset,
  so nothing is uploaded again. The runner is `-Exp EXP-042`.
- **Sessions:** the runs do not fit in one 9-hour session, so the script works through them in
  order (C1, then C2, then C3) and stops a session when the next run would pass its 9-hour cap.
  The longest single run (s2 at C3) is about 3.6 hours (NOT VERIFIED), so no run spans two
  sessions. Each session's results are committed in their own folder, and nothing published is
  overwritten. The next session reads the earlier results from the repository and continues.
  The founder types the same line again for each session, about **3 sessions** in total.
- **Estimated GPU time:** about 19–21 GPU-hours. It is estimated from the measured speed at 30 M
  parameters (EXP-040/041: about 63,000 tokens/s). The other sizes' speeds are **NOT VERIFIED**;
  the smallest models probably use the GPU less efficiently. **Hard cap: 25 GPU-hours** over all
  sessions, enforced by the script.
- **Quota:** Kaggle's free quota is about 30 GPU-hours a week. About 7 hours were used this week
  by EXP-040/041, so the last session may have to wait for the weekly reset (the reset day is
  NOT VERIFIED). ₹0.

**Code to be built after approval** (tested on CPU):
- `scripts/gpu_lr_arch.py --part ladder`: the plan above; resumable across sessions; time-budget
  aware; the analysis with numpy.
- A kernel template and `-Exp EXP-042` in the runner: a new kernel version for each session,
  and committing only that session's results folder.
- Tests, including a CPU smoke run of a whole tiny ladder over 2 "sessions" and the fits on
  synthetic data with a known answer.

**Committed evidence:** `evals/results/EXP-042/session-*/summary.json` and `SUMMARY.txt`; the
final session's summary carries the full analysis. The plot goes in the same folder.

**Not in scope:** choosing the step-12 size (a decision record after this), data mixtures,
context length, other architecture changes, more than one seed per point, multiple GPUs, and
any spending.

**Status:** proposed — founder approval needed before any code.

**Approved and built (2026-10-02):** the founder typed "approve EXP-042". Built as proposed, with
no change to the sizes, budgets, learning rates, limits or rules:
- `scripts/gpu_lr_arch.py --part ladder`: each launch is one session. It reads the earlier
  sessions from `--prev-dir` (default `evals/results/EXP-042`; a session made with a different
  plan is refused) and writes `session-<n>/summary.json` + `SUMMARY.txt`; each summary lists every
  run so far.
  - Part 0 as in EXP-041 (environment, 13 sha256 checks, tests), then the one-step float64 check
    for the D-048 architecture; if it fails, that session runs nothing.
  - Runs go in plan order (C1, then C2, then C3). C2/C3 runs of a size wait until all three of its
    C1 learning rates were tried; s5 waits for s4. A size with no successful C1 run is reported as
    "not run" for C2/C3, so the ladder can still finish.
  - Time budget: a run starts only if 1.15 × its FLOPs ÷ the measured FLOP rate of that size
    (else of the slowest size measured so far) fits before the session limit, which is
    min(9 hours, 25 − hours used by the earlier sessions). A crashed run (not a failed one) is
    retried in the next session.
  - When nothing is left, the final session's summary carries the analysis (rules 1–5).
- `scripts/ladder_fit.py` (numpy only): the parabola fit with the bracketing rule and the flat
  region, the power-law fits, the growth law with the 2.78 B-token projection and its
  leave-one-budget-out range, and the parametric fit (grid over α and β, non-negative least
  squares for E, A, B). D_opt and the total-parameter count at N_opt are interpolated in log-log
  between the measured sizes.
- `scripts/kaggle/exp042_kernel.py` (9 GPU-hours per session, 25 in total),
  `scripts/run_kaggle_exp042.ps1` and `-Exp EXP-042` in `scripts/run_kaggle_exp038.ps1`: the same
  private 13-language dataset (fingerprints checked again, nothing uploaded), waiting up to 11
  hours, a fresh download folder per launched commit, committing only the new
  `evals/results/EXP-042/session-<n>/` files. After a successful push it forgets the launched
  commit, so the same line launches the next session.
- Tests (`tests/test_ladder.py`, `tests/test_kaggle_exp038.py`): the fits recover planted answers
  on synthetic data (minimum, flat region, exponents, the projection's closed form, the parametric
  constants); the full-size plan matches the table above (parameters, 16,384 tokens per step, 20
  runs, s2 at C3 ≈ 1.24 B tokens); the analysis finds a planted optimum in all three budgets; a CPU
  smoke ladder over two sessions (5 runs, then the other 15, no run twice, the analysis at the
  end) and a third launch that does nothing; the 25-hour cap; the refusal of a mismatched
  session; static runner and kernel checks. The CPU smoke numbers are not results.
- The plot (rule 6) is made in the sandbox from the final committed summary.

**Status:** approved — code ready; waiting for the founder to launch the Kaggle run.

**Results (2026-10-03; the founder's Kaggle runs, `evals/results/EXP-042/`):** three Kaggle T4
sessions, Tesla T4, torch 2.10.0+cu128, each running the commit it was launched with:

| session | code commit | results commit | script time | runs |
|---|---|---|---|---|
| 1 | `2e24422` | `39ddd65` | 405.7 min | 16 (all of C1 and C2) |
| 2 | `39ddd65` | `e82dc80` | 418.9 min | 3 (C3: s2, s3, s4) |
| 3 | `e82dc80` | `2e79552` | 133.9 min | 1 (C3: s5); `complete: true` |

Total 15.98 GPU-hours by the script's own clock (cap 25); Kaggle's session clock is a little
longer (start-up and download). Each session ended because the next run would not fit, as
designed; no run spanned two sessions and nothing was overwritten.

- **Part 0, every session:** 13 of 13 files match the manifest; 40 tests pass; the one-step
  float64 check PASSES for the D-048 architecture (max |loss difference| 1.776e-15, max relative
  gradient difference 2.654e-15; limits 1e-12 / 1e-10).
- **Stability (rule 5):** no failed runs. At most 0.04% of steps were skipped by the fp16 scaler
  (32 of 75,889, C3-s2); peak memory at most 6.7 GB of 16 GB.
- **Speed (training tokens/s):** s1 about 267,000; s2 about 135,000; s3 about 73,000;
  s4 about 44,000; s5 25,387. Including evaluations, s5 ran at 1.27e13 FLOPs per second.

**Final validation bits per byte** (mean of the 13 languages with equal weight; lower is better):

| budget | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| C1 = 1e16, lr 5e-4 / 1e-3 / 2e-3 | 1.0178 / 1.0013 / **0.9916** | 0.9592 / 0.8997 / **0.8916** | 0.9629 / **0.9093** / 0.9735 | **1.0005** / 1.0011 / 1.0288 | — |
| C2 = 3e16, best lr | 0.9772 | 0.8400 | **0.8253** | 0.8824 | — |
| C3 = 1e17, best lr | — | 0.8165 | **0.7736** | 0.7766 | 0.7845 |

**Pre-registered analysis** (computed by the script in session 3, `session-3/summary.json`):
1. **Best learning rate at C1:** s1 2e-3, s2 2e-3, s3 1e-3, s4 5e-4. Three are at an edge of the
   range, which is reported as required. The gaps to the middle value are within the seed noise
   for s1 (0.0097), s2 (0.0081) and s4 (0.0006). The trend is clear, though: the best learning
   rate falls as the model grows. s5 used s4's 5e-4, as written down in advance.
2. **Best size per budget:** all three budgets are **bracketed**.

   | budget | N_opt (non-embedding) | N_opt (total, interpolated) | D_opt | tokens per non-embedding / total parameter | flat region (within 0.0174) |
   |---|---|---|---|---|---|
   | C1 | 3.96 M | 12.41 M | 124 M | 31.3 / 10.0 | 1.77–8.87 M |
   | C2 | 6.83 M | 17.31 M | 265 M | 38.8 / 15.3 | 2.87–16.27 M |
   | C3 | 22.19 M | 37.66 M | 406 M | 18.3 / 10.8 | 7.48–65.79 M |

3. **Growth law (rule 2), from all three budgets:** N_opt ∝ C^0.752 (non-embedding), D_opt ∝
   C^0.513. Leftover error in log10: up to 0.080 for N_opt and 0.057 for D_opt, both at C2.
4. **What our 2.78 B tokens support (rule 3), an EXTRAPOLATION 1.6 orders of magnitude beyond the
   largest budget:** D_opt = 2.78 B tokens at C = 3.77e18 FLOPs, where N_opt = **311 M
   non-embedding parameters**. Leaving out one budget at a time gives anywhere from **37 M to
   4.6 B**. The point estimate is therefore weakly determined.
5. **Parametric fit (secondary, never a gate):** E 0.614, A 27.2, α 0.34, B 7.3e3, β 0.58, rmse
   0.0186 (about the seed noise), converged. It implies N_opt ∝ C^0.63.
6. **The plot:** `evals/results/EXP-042/isoflop.svg`, made by `scripts/plot_ladder.py` from the
   committed summary.

**Post-hoc checks (NOT pre-registered; labelled as such; `evals/results/EXP-042/posthoc.json`,
written by the same script):**
- **The two exponents do not add up.** If N and D are counted consistently, then C ∝ N × D, so
  a + b should be about 1. Here it is 0.752 + 0.513 = **1.27**. Pearce & Song (2024, "Reconciling
  Kaplan and Chinchilla Scaling Laws") show why: counting only non-embedding parameters at small
  scale, where the embeddings are a large part of the model, inflates the exponent to about
  0.74–0.78. That is close to Kaplan et al.'s 0.73, and to ours. Porian et al. (2024) find the
  output layer's compute to be one of the causes; our FLOP count already includes it.
- **The same fit with total parameters:** N_opt ∝ C^**0.485**, so a + b = 0.998, consistent and
  close to Chinchilla's 0.5. At C = 3.77e18 this gives **206 M total parameters** (leave one
  budget out: 48 M – 1.28 B), about 13.5 tokens per parameter.
- **Which model of the same family spends 3.77e18 FLOPs on 2.78 B tokens:** 1.36e9 FLOPs per
  token are needed. That lies between 14 layers × 896 (190.1 M total, 160.6 M non-embedding, 1.22e9
  FLOPs per token) and 16 × 1024 (272.8 M total, 239.1 M non-embedding, 1.74e9). This uses only the
  pre-registered D_opt law and our FLOP formula.
- **The secondary fit evaluated at all 2.78 B tokens (an extrapolation of a secondary fit):**
  predicted bits per byte 0.701 (10 × 640), 0.690 (12 × 768), 0.683 (14 × 896), 0.677 (16 × 1024),
  0.673 (18 × 1152). Near the optimum the curve is flat: doubling the model changes the
  prediction by less than the seed noise.
- **T4 time for 3.77e18 FLOPs** at s5's measured 1.27e13 FLOPs per second: about **82 GPU-hours**.
  Whether larger models reach the same speed on a T4 is NOT VERIFIED.

**Limits:** one seed per point; the best learning rates at the edges of the range (s1, s2 and s4,
and s5 by inheritance) may slightly understate those sizes; the step-12 question is 1.6 orders of
magnitude beyond the largest measured budget; one data mix (natural proportions) and one context
length (512).

**What this means:** our training stack scales predictably on a GPU, and the compute-optimal
size grows with compute as the literature expects once parameters are counted consistently. For
the 2.78 B tokens we have, the evidence points to a model of roughly **200–300 M total
parameters**, with a wide uncertainty. Stage 4's exit criteria are now met: a measured
throughput and memory curve (EXP-038, EXP-042) and a scaling-law plot from our own runs. The
step-12 size itself is decided separately (D-049, proposed).

**Status:** complete — all 20 runs done, none failed; all 3 budgets bracketed; N_opt ∝ C^0.75 (non-embedding; post-hoc with total parameters C^0.48), D_opt ∝ C^0.51; extrapolation for 2.78 B tokens: 311 M non-embedding (range 37 M – 4.6 B; post-hoc 206 M total); 15.98 GPU-hours; D-049 proposed.

**Correction note (2026-10-03, added after the results; nothing above is changed):** the
proposal says that the longest run "reads 1.24 B of the 2.78 B training tokens, so no text is
repeated". That is not exact. `MultiTokenDataset.get_batch` draws each 512-token window at a
random offset, with replacement, so windows can repeat or overlap within a run. Drawing 1.24 B
tokens this way from 2.78 B covers about 36% of the corpus at least once. That makes about 19% of
the tokens processed by C3-s2 repeats, and less for every other run. At under one epoch this
should matter little: Muennighoff et al. (2023) find a few repetitions nearly as good as fresh
data. The laws count the tokens processed, as pre-registered, and the results stand. For step 12
it matters more: one "pass" of random windows would show only about 63% of the corpus.
EXP-043 therefore proposes a true one-pass sampler.

### EXP-043 — Step 12: train the first serious Frontier base model (14 × 896, about 190 M parameters, D-049)
**Date:** 2026-10-03 · **Status:** proposed (needs the founder's "approve EXP-043" before any code)

**Purpose (why this serves the mission):** this is the first model trained on our whole verified
corpus at the size our own scaling evidence supports (D-049). It tests the whole pipeline at a
scale 34× beyond anything trained so far: tokenizer, 13-language data, architecture, GPU stack,
multi-session training and evaluation. Everything later (more data, larger models, post-training)
builds on a pipeline that has done this once, end to end. It is a **research base model**: it
should write fairly fluent short text in our languages; it will not follow instructions or reason
(D-049).

**Fixed by D-048/D-049 (not re-decided here):** 14 layers × width 896, 14 query heads, 2 key-value
heads, head size 64, RoPE, SwiGLU, RMSNorm, tied embeddings, vocabulary 32,896, context 512;
190.06 M parameters (160.59 M non-embedding); one pass over the 13 EXP-037 language files in
natural proportions. fp16 + GradScaler + `torch.compile` on one Kaggle T4 (D-047).

**1. A true one-pass sampler (new code).** The 2,783,830,088 training tokens are cut into
5,437,162 non-overlapping 512-token windows (each with its next-token targets). They are shuffled
once with a fixed seed, across all 13 files, and read in that order: batch i is windows 32·i to
32·i + 31. That gives **169,911 steps of 16,384 tokens (2,783,821,824 tokens, every window exactly
once)**; the last 10 windows, which do not fill a batch, are dropped. Natural proportions follow
automatically. The position in the order is the step number, so resuming is exact. Validation
stays as in EXP-040–042 (full validation splits, unchanged).

**2. Learning-rate check (one Kaggle session, before the main run).** EXP-042 showed the best
learning rate falling with size (1e-3 at 13 M non-embedding parameters, 5e-4 at 30 M), so 1e-3
is not reused blindly.
- Three runs of 6,104 steps (100 M tokens, the first 100 M of the one-pass order, identical for
  all three) at **2.5e-4, 5e-4 and 1e-3**: the centre is the best size-30 M value; one step down
  and one up. Warm-up 1,000 steps; cosine decay to 10% over each run's own length; seed 1;
  16,384 tokens per step.
- **Pre-registered rule:**
  - The learning rate with the lowest final validation bits per byte (mean of 13 languages) is
    chosen. If a lower learning rate is within 0.0174 (the seed noise) of the best, the lower one
    is chosen instead: the main run is 28× longer, and longer runs favour lower learning rates
    and are less likely to diverge.
  - If the chosen value is at an edge of the grid, one more run is made one step further out
    (1.25e-4 or 2e-3) and the rule is applied to all four. At most one extension.
  - A run with NaN/inf or more than 5% skipped fp16 steps cannot be chosen.
- **Warm-up:** 1,000 steps (16 M tokens, 0.6% of the main run) instead of the ladder's 200,
  because this model is 2.4× larger than any trained so far. The check runs use the same warm-up,
  so the chosen learning rate is tested with it.

**3. Main run (about 9 Kaggle sessions):** 169,911 steps at the chosen learning rate, warm-up
1,000 steps, cosine decay to 10% over all 169,911 steps, AdamW (0.9, 0.95; weight decay 0.1),
clipping 1.0, seed 1.
- **Memory:** the micro-batch is 16 × 2 accumulation steps if the measured peak memory is below
  14.5 GB in Part 0, otherwise 8 × 4. Both are 32 sequences per step, the same maths.
- **Carrying the run across sessions:** each session stops at its time budget, saves a checkpoint
  (model, optimizer, fp16 scaler, step; about 2.3 GB) into the Kaggle output, and records its
  sha256 in `session-<n>/summary.json`, which is committed. The next session gets that output as
  an input through Kaggle's `kernel_sources`. Two kernels, `frontier-exp043-a` and `-b`,
  alternate, each reading the other's latest output, so no kernel reads itself. A session refuses
  to train unless the checkpoint it finds has exactly the sha256 of the last committed session. A
  periodic checkpoint every 5,000 steps guards against a session that is killed early.
- **Monitoring:** a sampled validation loss every 1,000 steps (the curve), the full 13-language
  validation bits per byte at the end of each session, tokens/s, memory and skipped steps.
- **Stop rules** (the session stops, keeps its last good checkpoint, and reports; continuing then
  needs a decision):
  - a NaN or infinite loss;
  - more than 5% skipped steps in any 2,000-step window;
  - a sampled validation loss more than 0.1 nats above the best so far, at 3 evaluations in a row.

**4. At the end (last session):** the full validation bits per byte per language, the mean with
equal weight, the token-weighted mean and the pooled value; text samples for reading (3 fixed
prompts per language, greedy and temperature 0.8, `samples.jsonl`; not scored); a weights-only
file `model_final.pt` (fp32, about 760 MB) with its config and sha256.

**Pre-registered checks on the final model:**
1. **Sanity (gate):** the mean bits per byte must be lower than the best EXP-042 run (C3-s3,
   0.7736) by more than 0.0174, because this run uses 34× more compute. If not, the result is not
   used for anything until it is explained.
2. **Does the scaling evidence predict it? (reported, not a gate):** the result is compared with
   the secondary parametric fit's prediction for this size at 2.78 B tokens, **0.683 bits per
   byte** (EXP-042 post-hoc, `posthoc.json`). It counts as "consistent" if within ±0.03. The new
   sampler (more unique text) and the re-tuned learning rate could both make it a little better.
3. **Every language (reported):** each language's bits per byte against C3-s3's; all 13 are
   expected to be lower.

**Part 0 (every session):** environment record; the 13 sha256 checks; the tests; the one-step
float64 check for the D-048 architecture (D-048 point 3); from the second main-run session on, the
checkpoint sha256 check.

**Where it runs, cost and time:**
- **Where:** private Kaggle T4 kernels, single GPU (D-047), the 13-language dataset already
  uploaded. The runner is `-Exp EXP-043`; the founder types the same line once per session, as in
  EXP-042.
- **Estimated GPU time** (from EXP-042's measured 1.27e13 FLOPs per second; NOT VERIFIED at this
  size): learning-rate check about 8 hours (one session); main run about 74 hours (about 9
  sessions); about 85 GPU-hours in total. **Hard cap: 100 GPU-hours** over all sessions,
  enforced by the script.
- **Quota:** Kaggle's free quota is about 30 GPU-hours a week (NOT VERIFIED for this account), so
  this takes about **3 weeks** of calendar time. ₹0.
- **Downloads to the PC:** only the small result files each session (`kaggle kernels output
  --file-pattern`; if this CLI version does not support it, the full output, about 2.3 GB, is
  downloaded instead and that is reported), plus `model_final.pt` once at the end, into `out/`
  (not committed to git).
- **NOT VERIFIED on the founder's account:** the `kernel_sources` chaining, where Kaggle mounts
  another kernel's output, and the output size limit. Both are first used in main-run session 2.
  If the chain fails there, the session stops at Part 0 after a few minutes, the earlier
  checkpoint stays safe in the other kernel's output, and nothing is lost but those minutes.

**Code to be built after approval** (tested on CPU in the sandbox):
- The one-pass sampler in `src/frontier_ai/data/multi.py`, with exact resume.
- `scripts/gpu_pretrain.py`, reusing EXP-040–042's Part 0, one-step check, run and report code:
  `--part lrcheck` and `--part main`, multi-session, time-budget aware, checkpoint chain and
  sha256 checks, stop rules, final evaluation and samples.
- `scripts/kaggle/exp043_kernel.py`, `scripts/run_kaggle_exp043.ps1` and `-Exp EXP-043` in the
  runner: the alternating kernels with `kernel_sources`, the results-only download, the one-time
  model download.
- Tests:
  - the sampler covers every window exactly once and resumes exactly;
  - a CPU smoke of the whole thing with a tiny model: the learning-rate check, then a main run
    over 3 "sessions" passing the checkpoint through a folder that stands in for Kaggle's input;
  - the sha256 refusal;
  - the stop rules;
  - static runner and kernel checks.

**Committed evidence:** `evals/results/EXP-043/session-*/summary.json` and `SUMMARY.txt`, and at
the end `samples.jsonl`. Checkpoints and `model_final.pt` are not committed (too large); their
sha256 values are.

**Not in scope:** more data, longer context, other sizes, more than one seed, instruction tuning
or any post-training, multiple GPUs, publishing the model, and any spending.

**Status:** proposed — founder approval needed before any code.

**Approval and implementation note (2026-10-04).** Approved by the founder on 2026-10-03
("approve EXP-043"). Built and tested on CPU in the sandbox; nothing has run on a GPU yet.
- **Code:** `src/frontier_ai/data/multi.py` (`OnePassDataset`, `one_pass_order`);
  `src/frontier_ai/engine/trainer.py` (uses a dataset's own window order and `seek` on resume;
  `fit(should_stop=...)`; checkpoints are written to `<tag>.tmp` and swapped in only when complete);
  `scripts/gpu_pretrain.py` (`--part auto | lrcheck | main`); `scripts/kaggle/exp043_kernel.py`;
  `scripts/run_kaggle_exp043.ps1` (`-Exp EXP-043` in `scripts/run_kaggle_exp038.ps1`). Tests:
  `tests/test_onepass_sampler.py`, `tests/test_pretrain.py`, EXP-043 checks in
  `tests/test_kaggle_exp038.py`.
- **Details fixed while building (not in the proposal text, recorded here):**
  - **The order** is a shuffle built from integer hashing (SplitMix64 keys, stable sort), not
    numpy's random generator, whose streams may change between numpy versions during a run of
    several weeks. Its sha256 fingerprint is recorded in every session and must not change.
  - **Memory choice:** in the first session, 3 real compiled fp16 training steps are made with
    16 × 2 and, if needed, 8 × 4; the first with peak allocated memory below 14.5 GB is kept for
    all later sessions (also for the learning-rate check).
  - **Text samples:** the "3 fixed prompts per language" are the first 32 tokens of the first
    three validation documents (of at least 32 tokens) in each language, continued for 128 tokens:
    greedy (top-1) and temperature 0.8 (seed 1). Real corpus text was used rather than prompts
    written by the agent in 13 languages.
  - **Same-session start:** with `--part auto` (what the kernel uses), the main run starts in the
    learning-rate check's session if at least 1 hour is left: 30 minutes of training plus a
    30-minute reserve for the session end (save, full validation, and in the last session the
    final files).
  - **Stop rules** end the session without saving the current state. The checkpoint that remains
    is the last periodic one of that session or, if there is none, the verified input
    checkpoint. The best sampled validation loss and the count of evaluations above it carry over
    between sessions. The window of 2,000 steps for skipped fp16 steps restarts at each session
    start.
  - **Keeping the chain unbroken:** each session's output holds `exp043_chain/chain.json` (which
    session wrote it and its checkpoint's sha256). A session that makes no new checkpoint copies
    the verified input checkpoint into its own output. The runner does not publish a session
    whose chain check failed, so the session number does not advance and a retry reads the same
    checkpoint. The first main-run session already checks that the other kernel's output is
    mounted (the `kernel_sources` test) before any checkpoint depends on it.
- **Bug found and fixed:** text generation with the KV cache used the wrong RoPE positions
  (`GPT.forward` asked the RoPE table for `T + start_pos` positions instead of `T`), so cached
  generation failed for every RoPE model. Training and evaluation always use `start_pos = 0` and
  were not affected. The only committed text generation so far (EXP-038 Part 3) used learned
  positions. No earlier result changes. `tests/test_model.py` now checks cached generation
  against full recomputation for learned positions, RoPE and RoPE + GQA-2; the new cases fail on
  the old code.
- **CPU checks (tiny model, 3 tiny language files):**
  - The sampler reads every window exactly once and resumes exactly.
  - The learning-rate rule passes its table of cases.
  - The whole flow passes the CPU smoke test: learning-rate check, then the main run over two
    later "sessions" passing the checkpoint through folders.
  - The run split across sessions ends with **bit-identical final weights** (same sha256) to
    the same run done in one session.
  - A checkpoint with one changed byte is refused, and so is a missing one.
  - A rising validation loss fires the stop rule, keeps the last periodic checkpoint, and the
    next launch refuses to continue.
- **Still NOT VERIFIED (first seen on Kaggle):**
  - speed and memory at this size;
  - the `kernel_sources` chaining;
  - `--file-pattern` support in the founder's Kaggle tool (the fallback is the full download);
  - the output size limit.

**Status:** approved — code ready; next: session 1 (learning-rate check) via `scripts\run_kaggle_exp043.ps1`.

**Update (2026-10-04): session 1 (learning-rate check) complete.** Kaggle T4, commit `2f31c82`;
results in `evals/results/EXP-043/session-1/`.

| learning rate | mean bpb (13 languages) | skipped fp16 steps | minutes |
|---|---|---|---|
| 2.5e-4 | 0.8543 | 0 | 160.2 |
| 5e-4 | 0.8172 | 0 | 159.8 |
| 1e-3 | 0.8160 | 0 | 159.5 |

- **Rule applied as pre-registered:** 1e-3 has the lowest mean bpb. 5e-4 is 0.0012 behind, within the
  seed noise 0.0174, so the lower rate **5e-4** is chosen. 2.5e-4 is 0.0383 behind, outside the noise.
  5e-4 is not at an edge of the grid, so there is no extension run. No run had NaN or skipped steps.
- **Now measured (were NOT VERIFIED):** peak memory 9.27 GB with micro-batch 16 × 2 (kept for all
  sessions); speed about 11,100 tokens/s, i.e. 1.47 s per training step (1.57 s including validation);
  `--file-pattern` works in the founder's Kaggle tool; session 1 used 8.05 GPU-hours (483.2 min).
- The main run did not start in this session (less than 1 hour left, by the same-session rule). At
  1.47–1.57 s per step, its 169,911 steps need about 69–74 GPU-hours (about 9 sessions).
- **Still NOT VERIFIED:** the `kernel_sources` checkpoint chain (first used in session 2) and the output
  size limit with a checkpoint.
- Code check for session 2: the PC now runs a later commit (EXP-044/045/046 were added on top). A CPU
  smoke of the whole EXP-043 flow (learning-rate check, then 2 chained sessions) was run on `2f31c82`
  and on the current code. Model, optimizer and trainer-state files, and the final weights
  (`6b3eab46…`), are bit-identical. The only differences are wall-clock times and run folder paths,
  which also differ between two runs of the same code.

**Status:** running — session 1 (learning-rate check) complete: 5e-4 chosen by the pre-registered rule; next: session 2 starts the main run

**Update (2026-10-05): D-050 approved and built. The main run continues on both GPUs** from the first
session launched on the new code. The plan is unchanged: the same 32 windows per step (16 per GPU),
order, learning rate 5e-4, schedule, stop rules, evaluation, gate and 100-hour cap (counted in session
hours). Details and checks are in D-050's 2026-10-05 note:
- one-GPU path bit-identical;
- CPU tests of two processes, the fallback to one GPU and failures;
- full suite 748 passed, 1 skipped.
Session 2 was launched on one GPU before this; its result is not in the repository yet. Each session
summary now says whether it ran on 1 or 2 GPUs. **NOT VERIFIED:** whole-session speed on two T4s.

**Status:** running — session 1 complete (5e-4 chosen); session 2 launched on one GPU (result not yet in the repo); from the next session launched: two GPUs (D-050)

### EXP-044 — Step 13: train on both Kaggle GPUs at once (two-GPU data parallel), checked before use
**Date:** 2026-10-04 · **Status:** proposed (needs the founder's "approve EXP-044" before any code)

**Purpose (why this serves the mission):** every Kaggle session we have run had **two T4 GPUs**
attached (`device_count: 2` in the committed environment records of EXP-038, EXP-040, EXP-041 and
all three EXP-042 sessions), and we use one (D-047: "only one is used until step 13"). Kaggle's
weekly quota counts session hours, not GPUs (NOT VERIFIED for this account). So if two-GPU
training works, the same quota trains almost twice as much. Free compute is our tightest limit:
the EXP-043 main run needs about 74 GPU-hours (NOT VERIFIED), about 2.5 weeks of quota on one
GPU. Every later, larger model needs this ability too. This is step 13 of the roadmap.

**What changes (and what does not):**
- Two processes, one per GPU (PyTorch DistributedDataParallel, launched with `torchrun`). Each
  step still uses **the same 32 sequences of 512 tokens (16,384 tokens) in the same order**: GPU 0
  takes sequences 0–15 of the step's batch, GPU 1 takes 16–31. The gradients are averaged across
  the two GPUs. Mathematically this is the same update as one GPU doing all 32. Only the order of
  floating-point additions differs.
- Unchanged: model, data order, learning-rate schedule, optimizer, fp16 + GradScaler + compile,
  the stop rules and the checkpoint format. Only GPU 0 evaluates, writes checkpoints and writes
  the report. A stop rule that fires is sent to both processes, so neither keeps training alone.
- A checkpoint written by the two-GPU run loads in the one-GPU run and the other way round (the
  saved weights are stored without the wrapper prefixes).

**Checks before any GPU time (CPU, in the sandbox, tiny model):**
1. Two CPU processes (`gloo` backend) against one process with the same global batch, in float64
   (the D-047 method): every step's loss and the final weights agree to 1e-10.
2. Checkpoint crossing: save on two processes, resume on one, and the other way round. The
   result equals the uninterrupted run.
3. A stop rule raised on one process stops both; only process 0 writes files.

**GPU check (one short Kaggle session, kernel `frontier-exp044`, about 1.5 GPU-hours, hard cap 2):**
- The real 190 M model (D-049), fresh weights with seed 1, the first 300 steps of the EXP-043
  one-pass order. Run once on one GPU and once on two GPUs (after compile warm-up in each).
- **Pre-registered pass rules:**
  - **Agreement:** the validation bits per byte (mean of 13 languages) after 300 steps differ by
    at most 0.01 between the two runs (well under the 0.0174 seed noise). There is no NaN/inf, and
    skipped fp16 steps are at most 5% in both runs.
  - **Speed:** the two-GPU run processes at least **1.4×** as many tokens per second as the
    one-GPU run, measured over steps 101–300. Below 1.4× the saving is too small to justify a
    change during a running experiment.
  - **Memory:** peak memory per GPU is under 14.5 GB with 16 sequences per GPU and no gradient
    accumulation (otherwise 8 × 2, the same rule as EXP-043).
- **When:** only **between** EXP-043 sessions (the founder starts it after a session has finished
  and before starting the next). It never runs while an EXP-043 session runs, and it reads nothing
  from EXP-043's outputs.

**What a pass leads to (a separate decision, not decided here):** if all three rules pass, a new
decision proposes running **the rest of the EXP-043 main run on two GPUs**: same order, same
batch, same schedule, same pre-registered final rules. It would continue from the latest checkpoint
and add a dated note recording which steps ran on one GPU and which on two. If the speed-up is
about 1.8× (NOT VERIFIED; the gradient exchange is about 0.76 GB per step over PCIe), the remaining
main run would need roughly 55% of the GPU-hours. If any rule fails, EXP-043 simply continues on
one GPU and nothing is lost but the check's ~1.5 GPU-hours.

**Not in scope:** more than one machine, model or tensor parallelism, ZeRO/FSDP, mixed GPU
types, bf16. The founder's guardrail on distributed training is the reason this needs explicit
approval.

**Budget:** about 1.5 GPU-hours (cap 2), ₹0. Code: a `--gpus 2` path in `scripts/gpu_pretrain.py`
(rank-aware batch slicing in `OnePassDataset`, a rank-0-only report and checkpoint, a broadcast
stop flag), a small kernel and a runner line, and tests. No change to EXP-043's code until the
follow-up decision is approved.

**Update (2026-10-04): approved and implemented (code only; the GPU check has not run yet).**
The founder approved it on 2026-10-04 ("approve EXP-044, EXP-045, EXP-046"). Code (commits
`ff7aa2c`, `fbd1aea`):
- `src/frontier_ai/engine/distributed.py`: start-up from `torchrun`, rank-0-only saving and
  logging, a "did every GPU pass?" vote (so one GPU can never stop alone and leave the other
  waiting forever), and checkpoints always saved in the one-GPU key format.
- `trainer.py` and `multi.py`: DDP-aware, and **exact no-ops on one GPU**. Re-checked with the
  EXP-043 CPU smoke run: the final weights hash, every checkpoint file hash and the validation
  numbers are identical before and after the change. EXP-043 sessions therefore train exactly as
  before.
- `scripts/gpu_ddp_check.py` (the pre-registered check), `scripts/kaggle/exp044_kernel.py`,
  `scripts/run_kaggle_exp044.ps1`; `tests/test_ddp.py` (8 tests).

Measured on CPU (tiny model, float64, 12 steps of 4 windows, gloo backend):

| Comparison | Largest loss difference | Largest weight difference |
|---|---|---|
| 1 process (2 windows x 2 micro-steps) vs 2 processes (2 windows each) | 0.0 | 0.0 |
| 2 processes with gradient accumulation (1 x 2 each) vs 1 process | - | 9.7e-17 |

- A checkpoint written by 2 processes resumes on 1 process (and the reverse); 6 + 6 steps equal
  12 uninterrupted steps to 1e-10.
- A stop rule firing at step 5 on 2 processes stops both; no checkpoint is written.

Fixed during the build: the first 2-process loss read twice too high (each process already
reports its own mean, so the step loss is the average of the processes, not their sum). Later
review: when the CPU tests run on Kaggle's GPU machine (Part 0), they now hide the GPUs, so they
stay float64 on CPU.

These are CPU results only. Whether two T4s agree, how fast they are and how much memory they use
is what the Kaggle check measures (NOT VERIFIED until it runs).

**Status:** approved 2026-10-04; code ready; the Kaggle check (`scripts\run_kaggle_exp044.ps1`, about 1.5 GPU-hours) runs between two EXP-043 sessions

**Update (2026-10-04): Kaggle check complete — PASS on all three pre-registered rules.** Two T4s,
commit `818bdf8`, 32.6 minutes; results in `evals/results/EXP-044/`. The same 190 M model, data
order (fingerprint identical to EXP-043's) and learning rate 5e-4 were used for 300 steps each.

| | 1 GPU (16 × 2) | 2 GPUs (16 × 1 each) | rule | result |
|---|---|---|---|---|
| validation, mean bpb after 300 steps | 1.4319 | 1.4276 | differ by ≤ 0.01 | PASS (0.0042) |
| skipped fp16 steps | 0 | 0 | ≤ 5% | PASS |
| tokens/s (steps 101–300) | 12,225 | 21,882 | ≥ 1.4× | PASS (1.79×) |
| peak memory per GPU | 9.26 GB | 8.98 GB | < 14.5 GB | PASS |

- At 1.79×, the rest of a run needs about 56% of its one-GPU session hours.
- Repository tests on the Kaggle machine: 54 passed.
- As pre-registered, using two GPUs for the rest of EXP-043 is a separate decision: **D-050
  (proposed)**.

**Status:** complete — PASS (agreement 0.0042 bpb, speed 1.79×, memory 8.98 GB); D-050 proposed

### EXP-045 — Step 15 preparation: a pre-registered evaluation plan for the EXP-043 model, plus a "try the model" tool
**Date:** 2026-10-04 · **Status:** proposed (needs the founder's "approve EXP-045" before any code)

**Purpose (why this serves the mission):** "the best model in India" must be shown with fixed,
public, honest tests, not with impressions. EXP-043 already measures bits per byte on our own
validation split. This plan fixes, **before the model exists**, which outside tests we run, how
we score them and how we report them. Then the result cannot be tuned after the fact. The same
yardstick then measures every later model, so progress (or its absence) is visible.

**Tests (all on the EXP-043 final checkpoint):**
1. **Protected suite `frontier-heldout-v1`** (D-042; 3,427 documents, 13 languages): bits per byte
   per language with Frontier Tokenizer v2. EXP-043 does not measure it. It is the one set that
   has been protected since before any v2 data was built.
2. **Belebele reading comprehension** (`facebook/belebele`, pinned revision `7899cdfa…`, CC BY-SA
   4.0, not gated on Hugging Face):
   - 900 four-choice questions per language. All 13 of our languages are listed in its metadata;
     the exact configuration names are NOT VERIFIED until the download.
   - **Scoring (fixed now):** zero-shot; the text is the passage, a newline, the question, a
     newline, then the answer option. Each option's score is its summed log-probability divided
     by its UTF-8 bytes; the highest score is the model's answer. If passage + question + option
     exceed 512 tokens, the passage is cut from the left and the number of cut items is reported.
   - **Report:** accuracy per language with a 95% Wilson interval (about ±3 points at 900
     questions); chance is 25%.
   - **Honest expectation:** close to chance for a 190 M base model. A language is called "above
     chance" only if its whole interval is above 25%. The point is the baseline that later models
     must beat, not a headline.
3. **Same text in every language (Belebele's FLORES passages):** every Belebele passage is a
   FLORES-200 passage, translated by people into each language. We report **total bits needed to
   encode the same passages** in each language, and bits per byte. Because the content is
   identical, total bits compare languages fairly, which per-byte numbers cannot (scripts use
   different numbers of bytes per letter). This shows which languages the model knows least.
4. **Contamination check (Kaggle CPU session, no GPU):** 13-token n-gram overlap between the
   Belebele passages, questions and options (tokenized with Tokenizer v2) and all 2.78 B EXP-037
   training tokens, streamed from the existing Kaggle dataset. Every overlapping item is listed.
   Belebele results are reported **with and without** flagged items. The training data was built
   before this plan, so overlap is measured and reported, never silently removed.
5. **Samples for reading:** the EXP-043 samples, plus 5 fixed everyday prompts per language
   (written down before evaluation), generated greedily and at temperature 0.8 with seed 1. They
   are printed in the report so a reader can judge fluency.

**"Try the model" tool (laptop, CPU):** `scripts/generate.py` already samples from a checkpoint
directory. It gets:
- an `--interactive` loop (type a beginning, the model continues it);
- support for the EXP-043 final checkpoint file.

The evaluation session also writes a **weights-only fp16 copy** (about 0.38 GB instead of the
multi-GB training checkpoint) for the founder to download. Plain-language warning in the tool:
this is a base model that continues text. It does not answer questions or follow instructions.
Speed on the i3-4030U is NOT VERIFIED and is measured on first use.

**Where and when:**
- Code and tests are built now on CPU with tiny models (no GPU, no laptop load, no EXP-043 code
  touched).
- The contamination check (test 4) can run on a Kaggle CPU session at any time.
- Tests 1–3 and 5 run in one short GPU session (estimated under 0.5 GPU-hours, NOT VERIFIED; cap
  1) **after EXP-043 finishes**. Belebele needs about 11,700 passage+question encodings plus 4
  short options each, which is too slow on CPU.

**Not in scope:** comparing against other companies' models (that needs their weights and compute
and is a separate plan); any test whose licence we have not checked; generative benchmarks that a
base model of this size cannot meaningfully attempt.

**Budget:** ≤ 1 GPU-hour after EXP-043; Kaggle CPU time for test 4; ₹0. The Belebele data is
downloaded at evaluation time by pinned revision and **not committed** (ShareAlike). Only hashes,
scores and short excerpts in reports go into the repo.

**Update (2026-10-04): approved and implemented.** Approved together with EXP-044/046. Code:
`src/frontier_ai/evaluation/belebele.py` and `token_ngrams.py`, `src/frontier_ai/engine/weights.py`,
`scripts/belebele_contamination.py` (test 4), `scripts/eval_exp045.py` (tests 1, 2, 3 and 5, plus
the fp16 copy), `scripts/export_heldout_text.py`, `generate.py --weights/--interactive`,
`scripts/kaggle/exp045_kernel.py` + `scripts/run_kaggle_exp045.ps1` (a CPU session),
`evals/prompts/everyday-v1.json`, `tests/test_exp045.py` (15 tests) and 3 runner tests.

What the download check settled (Hugging Face API, revision `7899cdfa`): one JSON-lines file per
language under `data/`; ours are `asm_Beng`, `ben_Beng`, `eng_Latn`, `guj_Gujr`, `hin_Deva`,
`kan_Knda`, `mal_Mlym`, `mar_Deva`, `ory_Orya`, `pan_Guru`, `tam_Taml`, `tel_Telu` and
`urd_Arab`. Each is pinned by its git blob hash and size, and checked before use. The data README
says 900 questions and 488 passages per language.

Details fixed while building (recorded, not result-driven; no model exists yet):
1. Training documents end with `<|endoftext|>`, so every scored text starts with that token
   ("start of a document"). The exception is a Belebele text whose passage had to be cut.
2. Passage, question and option are tokenized separately and joined, so an option's tokens are
   the same in every context.
3. Texts longer than 512 tokens (same-text bits and test 1) are scored in overlapping windows.
   Every token is scored once, with at least 256 tokens of context.
4. On the GPU, scoring uses fp16 autocast, like EXP-043's own validation on the T4 (D-047). Reports
   say which precision was used.
5. Test 1 needs the protected texts. `SUITE.json` holds only hashes, and the texts are re-built from
   the frozen corpus on the founder's PC. `scripts/export_heldout_text.py` writes them (about 1 MB,
   every document checked against the suite) for a private Kaggle input. Without them, test 1 is
   reported as NOT RUN, never estimated.
6. The 5 everyday prompts per language were written by the agent from 5 English themes and are NOT
   VERIFIED by native speakers. A correction becomes a new set (`everyday-v2`), never an edit.
7. The contamination scan was timed on synthetic data in the sandbox (2 cores): 100 M tokens in
   40 s, so about 19 minutes for 2.78 B. The Kaggle time is NOT VERIFIED.

When: test 4 runs on a Kaggle CPU session (no GPU quota) between two EXP-043 sessions, with
`scripts\run_kaggle_exp045.ps1`. Tests 1, 2, 3 and 5 run after EXP-043 finishes. That session's
kernel will read EXP-043's final kernel output, which is known only at the end, so wiring that
session into the Kaggle runner is a small step done then.

**Status:** approved 2026-10-04; code ready; test 4 (Kaggle CPU) runs between two EXP-043 sessions; tests 1-3 and 5 run after EXP-043

### EXP-046 — Data for the next model: plan FrontierCorpus v2-slice2 (more Sangraha Verified, same recipe), built on Kaggle CPU sessions
**Date:** 2026-10-04 · **Status:** proposed (needs the founder's "approve EXP-046" before any code)

**Purpose (why this serves the mission):** EXP-042/D-049 showed that our 2.78 B training tokens
support only a ~190 M model. A better model needs more data before it needs anything else. This
plan measures how much more verified data exists and proposes the next build. It runs on Kaggle
CPU sessions, so it does not load the founder's laptop and does not use GPU quota.

**Measured on 2026-10-04 (Hugging Face API, pinned revision `8b813c3f…`):**
- Sangraha `verified/` totals **217.9 GB** of parquet over 23 language folders.
- Our 13 languages total **195.3 GB**. v2-slice1 used one ~0.33–0.60 GB file per language (5.1 GB
  in total).

| lang | full verified (GB) | ≈ files | estimated tokens if fully built (B) |
|---|---|---|---|
| as | 0.98 | 3 | 0.47 |
| bn | 29.90 | 76 | 15.4 |
| en | 29.92 | 50 | 24.1 |
| gu | 9.93 | 27 | 5.4 |
| hi | 37.34 | 99 | 19.5 |
| kn | 7.65 | 21 | 3.5 |
| ml | 12.17 | 36 | 5.1 |
| mr | 12.78 | 33 | 6.1 |
| or | 3.92 | 11 | 2.0 |
| pa | 4.14 | 12 | 2.4 |
| ta | 18.71 | 52 | 8.4 |
| te | 15.10 | 40 | 7.9 |
| ur | 12.78 | 26 | 8.4 |
| **total** | **195.3** | | **≈ 108** |

- The token column is an **estimate (NOT VERIFIED)**. It multiplies each language's measured
  slice-1 tokens-per-parquet-byte by the folder size. Later files may filter differently, and
  duplicates across files would lower it. The file counts are folder size ÷ slice-1 file size.
- So about **39× more verified data** exists than we use. Assamese is the exception: only about
  3× exists.

**How much to build next (sized to the compute we can actually get, not to what exists):**
- At EXP-043's measured T4 rate, one GPU does about 1.3e13 useful FLOP/s. If EXP-044 passes, two
  GPUs do roughly 2.3e13 (NOT VERIFIED). That is roughly **2.5e18 FLOP per week** of free quota.
- A next model with ~4× EXP-043's compute (~1.4e19, about 5–6 weeks of quota) would want about
  **2× the tokens**, by EXP-042's fit (tokens grow as compute^0.513): ~5.7 B tokens in one pass.
- With validation and a margin, the target is **v2-slice1 + v2-slice2 ≈ 8 B tokens**. Building all
  108 B now would be wasted work: we could not train on it with free compute. This plan does
  **not** fix the next model's size; that gets its own experiment.

**Selection rule (pinned before building):** files `data-1` and `data-2` of each of the 13
languages, pinned by size and SHA-256 in a new `corpora/frontier/v2/sangraha_slice2.json`.
Assamese has only three files, so this uses all of it. Every language grows by about the same
factor (≈ 3× in total), so **the language balance stays as in v2-slice1**, and the scaling
evidence from EXP-040–043 still applies. Changing the balance is a separate comparison experiment
(the standing rule: no fixed mixing percentages without a comparison). The estimate is about
+5.6 B tokens (NOT VERIFIED).

**Same recipe, plus four safeguards:**
1. The **unchanged v2 build rules** (D-045's configuration fingerprint): normalize, language ID,
   quality rules, exact dedup, protected-suite decontamination including the short-document
   index.
2. **No document from v2-slice1 twice:** slice-2 documents are compared with slice-1 by hashing
   each document's token sequence. Slice-1's tokens are in the existing EXP-037 Kaggle dataset,
   and the tokenizer is deterministic, so identical text gives identical tokens. Matches are
   removed and counted.
3. **Belebele protected too** (if EXP-045 is approved): its passages, questions and options join
   the protected set, so the next model is clean on the evaluation that EXP-045 fixes.
4. **Read before accepting:** the report prints random kept and removed documents per language,
   as EXP-036 did. Acceptance is a separate decision, as D-045 was for slice 1.

**Where it runs:**
- **Kaggle CPU sessions.** Official specification checked 2026-10-04 on kaggle.com/docs/notebooks:
  4 CPU cores, 30 GB RAM, 12 hours per session, 20 GB saved output.
- The work is split into two kernels (about 7 + 6 languages), so each output stays well under
  20 GB.
- Each output holds the packed tokens (uint16) and compressed filtered text, as a kernel output
  that training kernels mount with `kernel_sources`. That needs no upload from the founder's PC.
- NOT VERIFIED until a probe:
  - whether a CPU session can run while an EXP-043 GPU session runs;
  - whether CPU sessions use any quota;
  - scratch disk size;
  - download speed from Hugging Face.

  The **first step is a 15-minute probe kernel**: it reports CPU, RAM, disk and the download time
  of one pinned file, and processes nothing. If Kaggle refuses to start it while the GPU session
  runs, it simply waits for the next gap.
- For scale: EXP-036 built 5.1 GB on the laptop in 3 h 43 min.

**Not in scope:** sources other than Sangraha Verified (the `unverified` and `synthetic` subsets,
other datasets: each needs its own licence and quality review); near-duplicate (MinHash) removal
(proposed separately if slice-2 inspection shows near-duplicates matter); tokenizer changes
(Tokenizer v2 is frozen, D-035).

**Budget:** Kaggle CPU sessions only (estimated 2–4 sessions, NOT VERIFIED), no GPU, ₹0, no
laptop work.

**Update (2026-10-04): approved; step 1 implemented (pinned files + the 15-minute probe).** Approved
together with EXP-044/045. Code: `corpora/frontier/v2/sangraha_slice2.json`,
`scripts/kaggle/exp046_probe_kernel.py`, `scripts/run_kaggle_exp046_probe.ps1` (a CPU session through
the shared runner), `tests/test_exp046.py` (5 tests) and 2 runner tests.

The pinned files (Hugging Face API, revision `8b813c3f…`, read 2026-10-04):
- `data-1` and `data-2` of each of the 13 languages: 26 files, **9,787,246,071 bytes** (1.92× slice 1's
  5,106,130,219). Each is pinned by size and SHA-256, and the build checks both before use.
- Cross-check: the `data-0` SHA-256 values in the same listings equal all 13 slice-1 pins.
- **One thing seen in the listing:** `mal/data-1` is 151 MB, about half the usual file size. So
  Malayalam grows 2.45× while the others grow 2.8–3.0×. By parquet bytes, Malayalam's share goes from
  6.54% to 5.49%; every other language moves by at most 0.35 points. The rule was fixed by file name
  before any content was seen, so it is **kept as pinned**. The slice-2 report shows the real token
  shares, and a balance change, if wanted, is a separate comparison experiment.
- At slice 1's overall tokens per byte, the 26 files are roughly 5.3 B tokens (NOT VERIFIED; filtering
  and dedup against slice 1 will lower it).

The probe processes no data. It reports CPU, RAM, cgroup memory limit and free disk for
`/kaggle/working`, `/tmp` and `/kaggle/input`. It checks that numpy, pyarrow and the frozen Tokenizer
v2 load. It downloads `asm/data-1` (326 MB) into `/tmp` with the repository's resumable, SHA-256-checked
downloader and times the download and the hash. It reads the row count from the parquet footer only,
then deletes the file: Sangraha files never go into a kernel output. It writes
`EXP-046/probe/summary.json` and `SUMMARY.txt`, published to `evals/results/EXP-046/probe/`.

One change from the plan text: the probe runs **between** two EXP-043 sessions, under the same runner
guard as EXP-044/045. The plan said it could try to start while a GPU session runs. But two runners on
the laptop committing to git at the same time could stop EXP-043's publish step, and EXP-043 comes
first. So "can a CPU session run while a GPU session runs" stays **NOT VERIFIED**.

Next: the build kernels (two, about 7 + 6 languages, each under 20 GB of output) are written after
the probe's numbers are in. Disk, RAM and download speed decide how they are split.

**Status:** approved 2026-10-04; step 1 ready (pinned files + probe); the probe runs between two EXP-043 sessions; the build kernels follow its results
