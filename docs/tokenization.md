# Tokenizer research — Project 002

> **This project does not select the final production tokenizer.**
> It builds the *research framework* that will let us select one later, and reports the
> first measurements taken with it. The final tokenizer decision must wait for larger and
> far more representative data experiments (ROADMAP Stages 2–4).

---

## 1. Why tokenization matters

A language model never sees characters or words — it sees a sequence of integer ids. The
tokenizer defines that mapping, and it quietly determines:

- **Cost.** Training and inference cost scales with the number of tokens, not characters.
  A tokenizer that needs 2× more tokens for the same text doubles the compute for the same
  amount of language.
- **Context length.** The window is measured in tokens. Poor compression means less text
  fits in the same context.
- **Quality.** Tokens are the units the model reasons over. If a word is split into
  fragments that never co-occur in training, the model struggles to use it. If digits are
  split unpredictably ("2026" → "20" + "26"), arithmetic gets harder. If a script is
  shattered into stray combining marks, nothing meaningful can be learned from it.
- **Multilingual fairness.** A tokenizer tuned on English can silently tax Indian
  languages: the *same sentence* costs several times more tokens in Hindi than in English,
  which means less context, higher cost, and worse service for those users.
- **Irreversibility.** You cannot change the tokenizer after training without retraining
  the model. It is one of the earliest and most permanent decisions in a model's life.

## 2. Character vs word vs subword

| Approach | Vocabulary | Unknowns | Sequence length | Typical failure |
|---|---|---|---|---|
| **Character** | tiny (hundreds) | only unseen characters | very long | poor compression; long-range dependencies become hard; unseen characters still break it (we measured 9.74% unknown-character rate on our probe) |
| **Word** | huge (100k+) | every unseen word | shortest | open-vocabulary disaster for morphologically rich languages; we measured 18.61% unknown rate |
| **Subword (BPE/Unigram)** | medium (8k–200k) | none if byte-level | intermediate | needs enough vocabulary to pay for multi-byte scripts |

BPE works by starting from a base alphabet and repeatedly merging the most frequent
adjacent pair. Byte-level BPE starts from the 256 possible byte values, so **any** UTF-8
string is encodable and decoding is lossless — there is no UNK token at all. That is why
both subword implementations in this repository are byte-level.

The catch: a byte-level tokenizer must *spend* vocabulary to reassemble multi-byte
characters. English letters cost 1 byte; a Devanagari character costs 3 bytes and often 2
code points (base consonant + vowel sign). This is directly visible in our measurements
(§7) and is the central reason Indian-language tokenization needs its own evaluation.

## 3. Why Indian-language tokenization needs specific evaluation

- **Script diversity.** Fourteen languages in our probe use seven distinct scripts
  (Devanagari, Bengali–Assamese, Gujarati, Tamil, Telugu, Kannada, Malayalam, Gurmukhi,
  Odia, and Perso-Arabic for Urdu). Byte cost per character differs by script.
- **Akshara structure.** The natural unit is often the *akshara* (base consonant + matra
  + virama + conjunct), not the Unicode code point. Tokenizers that split on code points
  or on `\p{L}` alone will fragment it.
- **Morphological richness.** Agglutinative word formation means word-level vocabularies
  explode; subword is mandatory.
- **Latin transliteration.** Hindi, Urdu (Roman Urdu), and Hinglish are widely written in
  Latin script; a tokenizer must handle both scripts for the same language.
- **Digit systems.** Both ASCII and Indic digits (०-९, ০-৯, ௦-௯) appear in real text.
- **Economics.** If Hindi costs 3× English per sentence, every downstream cost triples.
  Measuring *per-language* fertility is not academic — it is a product decision.

## 4. Unicode considerations

- **Code points ≠ characters.** "मैं" is 3 code points (म + ै + ं). Our evaluator reports
  both `chars` (code points) and `utf8_bytes` (encoded size) because they differ hugely
  across scripts.
- **Combining marks.** Categories Mn/Mc/Me (matras, virama, chandrabindu, nukta) are not
  letters or numbers, so regexes built on `\w` / `\p{L}` / `\p{N}` — including the
  GPT-2-style pattern used by many production tokenizers — split them away from their base
  character. **BPE merges never cross a pre-token boundary**, so the fragment can never be
  repaired. Our `bpe_python` implementation therefore treats marks as word characters
  (see `pretokenize`); the HF baseline does not. This is exactly the kind of difference
  the framework exists to measure.
- **Normalization.** NFC vs NFD changes code-point counts for the same text. This project
  does **not** normalize input (by design, for now): normalization is a tokenizer-level
  decision with real consequences, and it belongs in a measured ablation.
- **Grapheme clusters, emoji, ZWJ.** "👨‍👩‍👧" is one user-perceived character made of
  several code points joined by zero-width joiners. Byte-level tokenizers survive this;
  the probe corpus includes these cases to prove it.
- **Round-trip correctness is a hard requirement.** Any tokenizer that cannot reproduce
  its input is rejected by our comparator, no matter how good its compression looks.

## 5. Code-mixing / Hinglish

Hinglish (Hindi-English code-mixing in Latin script) is not a rare edge case in India — it
is how millions of people write. Implications:

- One sentence may mix scripts mid-word ("model को train करना है").
- English technical terms embed inside Indic sentences; splitting them character-by-character
  wastes context.
- A single-language tokenizer will treat the borrowed English words as rare fragments.

Our probe corpus has dedicated `hi-en` (Hinglish) and `code_mixed_tech` categories, and
the evaluator reports metrics per category, so code-mixing can be measured separately
instead of being averaged away.

## 6. What the metrics mean

| Metric | Definition | Better |
|---|---|---|
| `tokens` | ids produced for the corpus | fewer |
| `chars` | Unicode code points (`len(text)`) | context |
| `utf8_bytes` | size of the UTF-8 encoding | context |
| `words` | whitespace-separated tokens containing an alphanumeric; a heuristic, meaningless for scripts without spaces | context |
| `tokens_per_char` | fertility: tokens ÷ code points | lower |
| `chars_per_token` | compression: code points ÷ tokens | higher |
| `bytes_per_token` | UTF-8 bytes ÷ tokens; the fairest cross-script compression measure | higher |
| `tokens_per_word` | tokens ÷ whitespace words | lower |
| `unk_count` / `unk_rate` | tokens equal to the unknown id; `null` for byte-level tokenizers | lower |
| `round_trip_failures` | examples where `decode(encode(x)) != x` | 0 (required) |
| `special_tokens.all_ok` | every special token keeps its own id alone, inside text, and decodes to itself | true |

**Comparability rule.** These numbers are valid *only* between tokenizers measured on the
same corpus version. The comparator refuses to compare reports from different corpus
versions. They are **not** comparable to published figures computed on real corpora.

**Losslessness rule.** A tokenizer with round-trip failures is disqualified from winning
"best" in the comparison table: destroying text looks like excellent compression if you
only count tokens.

## 7. Architecture of the subsystem

```
src/frontier_ai/tokenization/
    base.py         SubwordTokenizer ABC + TokenizerInfo + special-token segmentation
    registry.py     name -> class registry (makes implementations pluggable)
    bpe_python.py   our own byte-level BPE (no dependencies, mark-aware pre-tokenizer)
    bpe_hf.py       HuggingFace `tokenizers` byte-level BPE (optional dependency)
    adapters.py     wraps the Project 001 char/word tokenizers behind the same interface
    corpus.py       deterministic train text + hand-written evaluation fixture
    evaluate.py     metrics (overall, per language, per category) + behavioural checks
    compare.py      side-by-side comparison, machine-readable + rendered table
    artifact.py     artifact = tokenizer files + manifest.json (provenance)
    cli.py          shared CLI helpers (corpus resolution, artifact loading)
```

**Pluggability.** To add a tokenizer: implement `SubwordTokenizer`, register it, and every
CLI (train / evaluate / compare) works with it unchanged. The Project 001 model and
training engine are untouched by this subsystem — a future production tokenizer only has
to satisfy the same interface to be usable by the training pipeline.

**Artifacts.** An artifact directory contains the implementation's files plus:

- `manifest.json` — experiment id, implementation and version, vocab size, special tokens,
  training parameters, corpus id/version/hash, seed, library versions, timestamp
- `vocab.json` — id → token dump for inspection

Artifacts are regenerable and are **not** committed (`artifacts/` is gitignored).

## 8. Commands

```bash
# 1. write the deterministic research corpus (train.txt + eval.jsonl + manifest.json)
python scripts/tokenizer_prepare_corpus.py --out data/tokenizer/indic-v1

# 2. train tokenizers (local corpus only — no network, no downloads)
python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 --impl bpe_hf \
    --vocab-size 1024 --out artifacts/tokenizers/bpe_hf_1024 --exp-id EXP-002
python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 --impl bpe_python \
    --vocab-size 1024 --out artifacts/tokenizers/bpe_py_1024 --exp-id EXP-002
python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 --impl char \
    --out artifacts/tokenizers/char --exp-id EXP-002

# 3. evaluate one or more artifacts -> JSON report(s) per tokenizer
python scripts/tokenizer_evaluate.py --corpus data/tokenizer/indic-v1 \
    --tokenizer artifacts/tokenizers/bpe_hf_1024 --out out/tokenizer/eval/bpe_hf.json \
    --jsonl out/tokenizer/per_example.jsonl

# 4. compare several artifacts on the same corpus -> comparison JSON + table
python scripts/tokenizer_compare.py --corpus data/tokenizer/indic-v1 \
    --tokenizer artifacts/tokenizers/char artifacts/tokenizers/word \
                artifacts/tokenizers/bpe_py_512 artifacts/tokenizers/bpe_py_1024 \
                artifacts/tokenizers/bpe_hf_1024 \
    --out out/tokenizer/compare.json --exp-id EXP-002
```

All four commands follow the Project 001 CLI conventions (argparse, `--set`-style
overrides are not needed here, JSON output, non-zero exit on error).

## 9. The research corpus — what it is and is not

Written by `scripts/tokenizer_prepare_corpus.py`, everything generated by our own code:

- **Training text** (`train.txt`): deterministic sentences generated from small curated
  word lists for 14 languages, plus numbers/dates/math variants, English technical lines,
  code snippets, and the Project 001 synthetic English corpus. ~73k characters, sha256
  recorded in the manifest.
- **Evaluation fixture** (`eval.jsonl`): 179 hand-written probes — 14 languages
  (English, Hindi, Hinglish, Bengali, Marathi, Gujarati, Tamil, Telugu, Kannada,
  Malayalam, Punjabi, Odia, Assamese, Urdu) and 9 orthography categories (numbers, dates,
  decimals, math, punctuation, URLs, technical terms, code-mixed technical text, Unicode
  marks). Totals: 4,108 code points / 8,141 UTF-8 bytes / 680 whitespace words.
- The loader verifies the fixture's sha256 against the manifest and refuses to run on a
  modified corpus.

**What it is NOT:**

- It is not a training corpus and not statistically representative of any language.
- It is not a benchmark. Numbers from it do not predict real-world performance.
- It is not licensed data, scraped data, or third-party data — it is ours, and tiny.
- Training text and evaluation text share small word lists, so compression measured here
  is **optimistic** for every tokenizer.

## 10. First measurements (EXP-002)

Full record in [EXPERIMENTS.md](../EXPERIMENTS.md). Headline numbers, 179 examples:

| Tokenizer | Vocab | Tokens | chars/token | bytes/token | unk% | round-trip failures |
|---|---|---|---|---|---|---|
| char (Project 001) | 365 | 4,108 | 1.000 | 1.982 | 9.74 | 128 |
| word (Project 001) | 1,085 | 3,111 | 1.320 | 2.617 | 18.61 | 160 |
| bpe_python @512 | 512 | 4,835 | 0.850 | 1.684 | n/a | 0 |
| bpe_python @1024 | 1,024 | 4,204 | 0.977 | 1.936 | n/a | 0 |
| bpe_hf @1024 | 1,024 | 3,835 | 1.071 | 2.123 | n/a | 0 |

Observations that matter more than the ranking:

1. **Both byte-level BPEs are lossless; the char/word baselines are not** on this fixture
   (128 and 160 of 179 examples altered), because their training text is too small to cover
   the characters and words in the probe.
2. **At 512–1024 vocabulary, byte-level BPE does not beat one-token-per-character for
   Indic scripts** (chars/token 0.65–1.0 vs 1.000 for char), while it clearly beats it for
   English (1.4–1.8). The vocabulary budget is being spent reassembling 3-byte characters.
3. **At equal vocabulary (1024), the HF baseline produced 8.8% fewer tokens than our
   Python BPE** (3,835 vs 4,204). The merge algorithms are equivalent; the difference comes
   from pre-tokenization — HF attaches a leading space to words so merges can absorb it,
   while our implementation emits whitespace runs as separate tokens. This is now an open
   ablation (Q-9), not a conclusion.
4. Per-language spread is large: English 1.5–1.8 chars/token vs Odia/Punjabi 0.71–0.86 for
   the same tokenizers — a >2× cost gap between languages.

None of this is a verdict. It is the framework working: it produced measurements, exposed a
vocabulary-budget problem specific to Indic scripts, and generated a concrete ablation.

## 11. Baseline dependency

| | |
|---|---|
| Package | `tokenizers` (HuggingFace), version 0.23.2 |
| License | Apache-2.0 |
| Why | Rust implementation, widely used, offline-capable (we only use `train` on a local file and `save`/`from_file`), byte-level BPE with special-token support and full introspection |
| Cost | ~12 MB wheel, plus its transitive dependency `huggingface_hub` (~7 MB) |
| Alternatives | `sentencepiece` (Apache-2.0; Unigram/BPE, C++ extension, less introspection), `transformers` (too heavy for our needs), writing only our own (slower, and no established reference to check against) |
| Mitigation | Declared as an **optional** extra (`pip install ".[tokenizer]"`). If it is missing, `bpe_hf` simply is not registered and the subsystem keeps working with `bpe_python`, `char`, and `word`. No network access is performed at any point. |

## 12. Limitations and open questions

- The corpus is a probe fixture (§9). No claim about real-world quality is made.
- Project 004 Stage A adds the licensed, versioned foundation for a real corpus
  (`indic-tokenizer/v2`): see [docs/tokenizer_corpus_stage_a.md](tokenizer_corpus_stage_a.md).
  It declares three sources, verifies none of them (no network to the source hosts), and
  therefore **does not** replace this fixture for anything yet.
- How those sources will later be acquired and verified is written up in
  [docs/tokenizer_corpus_stage_b_acquisition.md](tokenizer_corpus_stage_b_acquisition.md)
  (preflight → fetch → inspect → pin → re-verify; §0 is the five-command quick start). The
  code enforces the licence, index-page and pinned-hash gates and reports every source in
  `acquisition.json`. No acquisition has been performed.
- Vocabulary size was **not** swept: 512 and 1024 are demonstrations, not a recommendation.
- No normalization ablation (NFC/NFD), no Unigram vs BPE comparison, no script-aware
  initialization experiment yet.
- `bpe_python` is O(merges × corpus) and is a research reference, not a fast tokenizer.
- Tokenizer choice interacts with model quality, which we cannot measure until a model is
  actually trained with each tokenizer (Stage 4).

Open questions recorded in [DECISIONS.md](../DECISIONS.md): Q-9 (script-aware
pre-tokenization ablation), Q-10 (vocabulary sizing sweep), Q-11 (normalization policy).

## 13. The rule that governs this project

> **This project does not select the final production tokenizer.**
> It selects *how we will decide*. The decision itself requires: real licensed data at
> scale (Stage 3), a vocabulary-size sweep, an end-to-end quality measurement (train a
> model with each candidate tokenizer and compare), and cost/latency targets. Revisit
> after Stages 3–4.
