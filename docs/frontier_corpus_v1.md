# FrontierCorpus v1 — living plan and state

**Status (2026-09-26):** pipeline complete; the **pilot dataset is built and verified**
(EXP-027 on the PC, reviewed from the repository by the Arena agent): 34,684 frozen
documents in → 34,011 out (334 langid, 339 dedup), train 30,584 / held_out 3,427,
5 hash-recorded shards, `--check` exit 0. D-036 (NFC) and D-037 (staged pipeline) are
**accepted** (ratified by the founder's STEP 3 go-ahead). Governing context:
MASTER_CONTEXT §10–15 (what a pretraining corpus must be), §37 step 3 (freeze →
pipeline → tokenizer sweep). Next: MASTER_CONTEXT §37 step 5 — the production
tokenizer sweep (EXP-A), which needs this pilot dataset as its input.

## 1. What this is

FrontierCorpus v1 is the **reproducible, versioned, training-ready dataset pipeline** —
MASTER_CONTEXT §37 step 3, the milestone after the frozen tokenizer corpus (D-035) and
before any tokenizer experiment (steps 5–7). The pilot corpus is the frozen
`indic-tokenizer/v2`: 59 sources, 13 evaluated slots, 34,684 documents / 4,211,707
characters (hash-locked; EXP-023/025). The pilot adds **no new sources and no new
licensing surface** — every legal and availability question is already settled by the
freeze.

Output of a complete build: **pre-tokenization text shards** (plain UTF-8 text files,
one SHA-256 each) plus a **dataset manifest** recording corpus version, git SHA, seed,
normalization policy, per-stage and per-language statistics. No tokens are produced
here: no tokenizer is selected yet (§37 step 7), and tokenizing inside the pipeline
would force a corpus rebuild per tokenizer candidate.

## 2. Design (D-037)

One document model, one outcome contract, pure functions:

```
PipelineDocument(doc_id, source_id, language, text)      # frozen; sha256 = f(text)
StageOutcome(kept, removed, flagged, stats)              # frozen
```

Every stage is `list[PipelineDocument] -> StageOutcome`:

- **keeps** an ordered sequence of documents;
- **removes** nothing silently — every `Removal` carries a machine-readable `reason`
  plus the **measured values** that triggered it (MASTER_CONTEXT §15);
- **flags** kept documents a rule found suspicious (recorded, not dropped);
- reports **stats** — per rule, per language, in/out totals.

Stages have no global RNG and no dependence on input order beyond what they record.
Each stage is a pure function of its input, so the pipeline is reproducible given the
frozen corpus text and a recorded seed (the seed enters only in part 2, shuffle/pack).

Order: `normalize → langid → quality → exact_dedup → (mix → split → shuffle → pack →
shard → manifest)`.

The package **extends** the P004A/B infrastructure and does not fork it:
`frontier_ai.data.corpora` (fetch/clean/provenance, `sha256_text`),
`frontier_ai.data.mediawiki` (wiki source handling), `frontier_ai.tokenization.
research_corpus` (documents, content-hash split, leakage checks, stats). Split and
stats logic in part 2 reuses `research_corpus` directly.

## 3. Stages — part 1 (implemented)

### 3.1 `normalize` (`corpus/normalize.py`) — D-036

NFC composition + deterministic whitespace canonicalization (whitespace run → one
space, ends stripped; `\r` dropped; `\n` preserved — a document is one paragraph and
an internal newline is content, e.g. verse lines). Policies: `nfc` (default) and
`none` (ablation hook, Q-11). Policy + version (`"1"`) recorded in stats. **No
removals** — normalization transforms. Stats: `documents_changed`, in/out chars/bytes.

### 3.2 `langid` (`corpus/langid.py`)

A **script-profile gate, not a classifier**: each document's letter characters are
counted by Unicode script block; a document is removed when its declared script is
below `min_declared_share` (default 0.6) or when it has no letters at all. Reasons:
`langid_mismatch` (with expected/actual script and full profile), `langid_no_letters`.
Requires an explicit `language → script` map (a missing entry is a caller bug — fails
loudly). Documented limitation: same-script languages (hi/mr in Devanagari, bn/as in
Bengali) are **not** distinguished; a word-level classifier is an F3 option.
Deterministic, stdlib-only, fast.

### 3.3 `quality` (`corpus/quality.py`)

Measurable rules, one reason each — **not** a composite score (MASTER_CONTEXT §15):

| rule | action | fires when |
|---|---|---|
| `min_chars` | reject | chars < floor (default 1 — verse lines are short; F3 calibrates per language) |
| `max_chars` | reject | chars > ceiling (default 20,000 — truncation/parse accident) |
| `control_chars` | reject | non-whitespace C0 control chars > 2% of chars |
| `url_density` | flag | URL markers > 2 per 1,000 chars (navigation residue) |
| `template_residue` | flag | any `{{`/`}}` (defence in depth over the P004B cleaner) |
| `digit_runs` | flag | runs of ≥5 ASCII digits (phone/ID/scan accident; a year passes) |
| `repetition` | flag | distinct token-bigram ratio < 0.3 (documents with <8 tokens are recorded as `insufficient_data`, never silently passed) |

A reject removes the document (detail records **all** rules' measured values, not just
the guilty one); a flag keeps it and records the rule under `flagged`. First reject in
`RULE_ORDER` is the reason. `QualityPolicy` holds one default threshold table plus a
per-language override map; per-rule and per-language counts make systematic bias
visible. All thresholds are deliberately conservative for the pilot (verse lines).

### 3.4 `exact_dedup` (`corpus/dedup.py`)

Key = SHA-256 of the **normalized** text (so NFC variants deduplicate together).
Deterministic canonical order `(source_id, doc_id)`; first occurrence wins, later
copies removed with `reason="exact_duplicate"` and a `first_seen`/`first_source`
pointer. **Order-independent** (test: shuffled input → same kept set, same stats).
Reports within-source and cross-source duplicate pairs — cross-source duplication is
MASTER_CONTEXT §14's explicit case (the same text re-hosted on two wikis). Scope limit:
document-granularity *exact* dedup only; near-dedup (MinHash/LSH) is F3, deliberately
not approximated, because a silent "almost" would corrupt the honest exact count that
the frozen corpus's reports can cross-check against.

## 4. Part 2 (built)

1. **Registry import** (`corpus/registry.py`) — imports the frozen v2 manifest as the
   Frontier v1 registry *in memory* (no new file, nothing copied): refuses to run unless
   the manifest bytes hash to the `FREEZE.json` identity (D-035), assigns each source a
   `domain` (pilot rule: every current kind is `literature`; an unknown kind fails
   loudly), and derives the `language → script block` map the langid gate needs from the
   manifest's own script declarations.
2. **`split`** (`corpus/split.py`) — reuses `research_corpus.split_documents` (same
   deterministic content-hash rule, seed 1337 / 10% default), applied to the
   post-normalization, post-dedup text so byte-identical text can never straddle the
   split. The holdout is a second real output (its own shard set), not a removal.
3. **`mix`** — the pilot is single-domain (`literature`); the manifest records domains,
   and no recipe is applied (standing rule: a comparison framework, not fixed
   percentages — F2/F3).
4. **`shuffle`** (`corpus/shards.py`) — `random.Random(seed)` on a fresh generator, no
   global RNG; train uses `seed`, held-out uses `seed+1` (both recorded).
5. **`pack`/`shard`** (`corpus/shards.py`) — character-budget packing, documents atomic
   (one per line), LF newlines forced, one SHA-256 per shard.
6. **Manifest** (`corpus/manifest.py`) — `frontier-corpus` `1.0.0-pilot`: git SHA,
   freeze identity, split params, normalization policy + version (D-036), per-stage
   `StageOutcome` dicts, per-side totals, per-language counts, per-shard hashes.
   `content_sha256` hashes everything except `created_at`, so two builds of the same
   corpus at the same code state compare by identity even though their timestamps
   differ.
7. **`scripts/build_frontier_corpus.py`** — self-recording (D-032) via
   `run_self_recorded`, exit-code contract 0/1/2, `--check` mode (read-only shard
   verification against the manifest), human report under `corpora/frontier/v1/reports/`,
   and a hard input gate: the build refuses (exit 2) unless the freeze identity matches
   **and** every one of the 59 on-disk text files hashes to its pinned sha256.

Offline e2e coverage: `tests/test_frontier_corpus_build.py` builds and checks a real
build of a tiny two-source fake frozen corpus under `tmp_path` (real pins, real
FREEZE.json) — including `--check` success, tamper detection, drift refusal, freeze
mismatch refusal, and content-identity equality across two builds.

## 5. Verification plan

- **Unit tests** (`tests/test_frontier_corpus.py`, 32 tests, part 1;
  `tests/test_frontier_corpus_build.py`, 14 tests, part 2): golden set = the repo's
  own verified 14-language probe bank (`tokenization/corpus.EXAMPLE_BANK`) — every
  declared language passes its script gate; crafted fixtures per quality rule; dedup
  order-independence; stage composition; registry refusal paths; split content-keying;
  shard budget/determinism/tamper checks; manifest content identity; builder e2e on a
  fake frozen corpus.
- **Cross-check against the frozen corpus's own evidence:** EXP-023's inspection
  counts 396 identical documents total (as 46, bn 218, en 22, gu 0, hi 14, kn 11, ml 4,
  mr 20, or 27, pa 20, ta 4, te 4, ur 6). The pilot dedup removal must match that count
  **closely**; any delta must be explained (normalization can merge or split duplicate
  classes — e.g. composed vs decomposed twins). An unexplained delta fails the build
  review.
- **Reproducibility:** two builds on the same frozen text produce **byte-identical**
  shards and an identical manifest (modulo the build timestamp, which the manifest
  records separately from the content identity).
- Pilot run happens on a network-enabled machine (the Arena sandbox cannot reach the
  corpus hosts) and is recorded as **EXP-027**.

## 5a. Pilot results (EXP-027, run on the PC 2026-09-26, reviewed from the repo)

| Stage | in | kept | removed | note |
|---|---|---|---|---|
| normalize (NFC) | 34,684 | 34,684 | 0 | transforms only |
| langid (script gate) | 34,684 | 34,350 | **334** | largest: bn 161, or 29, pa 27, hi 33, as 20, gu 20 |
| quality (7 rules) | 34,350 | 34,350 | 0 | 1 kept doc flagged (`digit_runs`, kn) |
| exact dedup | 34,350 | 34,011 | **339** | per-source + cross-source pairs in the manifest |

Train 30,584 docs / held_out 3,427 docs (10.08%); 4 train shards + 1 held-out shard,
each with a SHA-256; `--check` exit 0. Manifest `content_sha256` `6c43d126…`; the
shard-file char counts equal document chars + joining newlines (checked exactly).

**396 cross-check (the verification-plan requirement):** EXP-023/026 counted 396
identical documents on the raw text; the pilot dedup removed 339. Reconciled:
NFC first merges 3 extra composed/decomposed twins (396 → 399, as reported by the
PC run), then the langid stage rejects 60 of those duplicate copies before dedup runs
(399 − 60 = 339). Every language's deficit (396→339) is ≤ that language's langid
removal count, and 7 languages (kn, ml, mr, ta, te, ur, gu) are unchanged — the
delta is fully attributable to the pre-dedup stages, as the plan requires. (The
per-language 399/60 sub-counts come from the PC's local recomputation; they are
internally consistent with the manifest and recomputable on the PC.)

**One fix made during the run** (`dbc67be`): the input gate compared raw file bytes
against the pin, but the v2 pin hashes the *decoded text* (CRLF on disk on Windows) —
the gate was corrected to validate the decoded text exactly as the v2 pipeline
pinned it, with a CRLF regression test; the refusal tests (drifted text, tampered
freeze) still pass, so the safety gate is unchanged in strength.

**Known pilot limitation (for F2):** the manifest records stage *counts* and the
dedup per-source/per-language breakdown, but not the per-document removal reasons of
the 334 langid rejections. The build is deterministic, so a reviewer can re-run it and
dump the `Removal` details at any time; if a deeper review of the 334 is wanted, that
dump is a small follow-up.

## 6. Out of scope (F2/F3)

Near-duplicate detection, PII removal, model-based quality scoring, language
*classification* beyond the script gate, data-mixing **recipe experiments** (framework
yes, recipe no), tokenization of any kind, new sources (F2 growth = live wiki research
on a network-enabled machine only — no from-memory source leads).
