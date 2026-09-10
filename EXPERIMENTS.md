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
