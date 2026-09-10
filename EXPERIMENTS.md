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

## 3. Experiment log

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
machine-dependent).

---

## 4. Log index

| ID | Title | Status | Date | Key metric |
|---|---|---|---|---|
| EXP-001 | CPU smoke test: end-to-end pipeline verification | complete | 2026-09-10 | val loss 3.93 → 1.378 (ppl 3.97) on synthetic corpus |

*(Add one row per experiment as they are run. Do not add rows for planned experiments —
those belong in [ROADMAP.md](ROADMAP.md).)*

---

## 5. Reproducing EXP-001

```bash
. .venv/bin/activate
python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic
python scripts/train.py --config configs/cpu_smoke.json
python scripts/evaluate.py --ckpt out/cpu-smoke/best --data data/synthetic.bin
cat out/cpu-smoke/train.jsonl
```

Expected on comparable hardware: val loss ≈ 1.38 after 200 steps in under a minute. Exact
numbers vary with thread count and CPU; the corpus and seed are fixed, so loss curves should
be close but are **not** guaranteed bit-identical unless `train.deterministic=true`.
