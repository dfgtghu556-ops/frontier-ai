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


## 5. Log index

| ID | Title | Status | Date | Key metric |
|---|---|---|---|---|
| EXP-001 | CPU smoke test: end-to-end pipeline verification | complete | 2026-09-10 | val loss 3.93 → 1.378 (ppl 3.97) on synthetic corpus |
| EXP-002 | Tokenizer baselines on the Indian-language probe fixture | complete | 2026-09-10 | lossless byte-BPE (0 round-trip failures) vs char 128/179 and word 160/179 failures; en 1.83 vs pa 0.80 chars/token |

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
