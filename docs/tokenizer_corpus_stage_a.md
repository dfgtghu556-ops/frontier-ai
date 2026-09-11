# Tokenizer research corpus — Project 004, Stage A

**Status: foundation only. No tokenizer has been selected, trained or compared here.**

This document describes the corpus foundation the later Project 004 stages will use. It
covers structure, provenance, licensing, coverage, splitting, leakage, statistics and
reproducibility — and it is deliberately explicit about what could **not** be verified in
the environment where it was written.

## 1. Why this exists

Project 002's `indic-eval-v1` fixture is 179 hand-written probes (109 language examples +
70 orthography probes) across 14 languages, with generated training text. That is the
right thing for regression and the wrong thing for choosing a tokenizer: it is tiny, its
train and eval halves share word banks, and it is not licensed real text.

Stage A builds what the later stages need instead: a versioned, licensed, deterministic
corpus definition with real sources, a real train/held-out split, leakage diagnostics and
a coverage report that hides nothing.

Stage A does **not**: compare tokenizers, sweep vocabulary sizes, apply Unicode
normalization, choose a pre-tokenizer, or train anything. Those are Stages B–H.

## 2. Layout

| Path | What |
|---|---|
| `corpora/tokenizer/indic-tokenizer-v2/sources.json` | the corpus manifest (committed): corpus id/version, targets, split spec, policies, 14 language slots, 3 declared sources |
| `src/frontier_ai/tokenization/research_corpus.py` | manifest model + validation, ingestion, document splitting, split, leakage, statistics, coverage, build |
| `scripts/build_tokenizer_corpus.py` | the CLI (records itself; see §9) |
| `tests/test_tokenizer_corpus.py`, `tests/tokenizer_corpus_fixtures.py` | 47 tests over all of the above |
| `data/tokenizer/indic-tokenizer-v2/` | build output (git-ignored): `corpus.json`, `train.jsonl`, `heldout.jsonl`, `stats.json`, `coverage.json`, `leakage.json`, `sources/<id>.txt`, `sources/<id>.provenance.json` |

The manifest reuses `frontier_ai.data.corpora` for everything it already does well:
fetching (`fetch_text`), kind-specific cleaning (`prepare_source_text`), provenance
records (`write_provenance`) and hashing (`sha256_text` / `sha256_file`). Nothing was
forked. The smoke corpus (`corpora/smoke/sources.json`) is untouched and still works.

## 3. Corpus identity and manifest

```
corpus_id       indic-tokenizer
corpus_version  v2
schema_version  1.0
```

`v2` distinguishes it from the Project 002 fixture (`indic-eval`/`v1`) — the two are
different objects and are never mixed.

The manifest records, for every language slot: code, display name, script, targets,
declared source ids, optional candidate providers and a **reason** when no source is
declared. For every source it records: id, title, language, script, source URL, licence
id, licence URL, attribution text, `max_chars`, kind, `sha256`, `verified`,
`retrieved_at` and notes.

`validate_manifest` rejects: an unknown schema version, a corpus id that is not
`[a-z0-9-]`, a version that is not `v<digits>`, a language slot outside the researched
set, duplicate slots, unknown keys, a licence outside the allow-list, non-https URLs, an
attribution shorter than 16 characters, a `max_chars` outside `(0, 2_000_000]`, an unknown
kind, a malformed `sha256`, and `verified: true` without a pinned hash.

### Targets are goals, not quotas

```
sentences_per_language   500
characters_per_language  200,000
```

A slot below target is reported as `INSUFFICIENT` with the real amount and the real
percentage of target. Nothing is padded, duplicated or synthesized to reach a number.

## 4. Licensing

Allowed licences are the repository allow-list, unchanged:
`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `PD-US`. Anything else is rejected by validation.

Rules enforced in code, not just in prose:

1. **No source is `verified` until it has actually been fetched**, its licence marker has
   been found in the payload, and its SHA-256 has been computed from the cleaned text.
2. **No hash is ever invented.** All three declared sources carry `sha256: null` and
   `verified: false`. Hashes are pinned only by `--pin` after a verified fetch.
3. **Local text is never auto-verified.** Text supplied from disk is ingested as
   `local_unverified` and can never make a language slot count as `EVALUATED`.
4. **Unverified text is excluded from the corpus by default** (`--include-unverified`
   opts in, and still cannot make a slot `EVALUATED`).
5. Attribution text is mandatory and travels with the corpus into every provenance file.

Current declared sources (all three **unverified**):

| id | language | licence | kind | max_chars | sha256 |
|---|---|---|---|---|---|
| `en-gutenberg-alice-pd` | en | PD-US | gutenberg | 400,000 | `null` |
| `hi-wikisource-godan-ccbysa` | hi | CC-BY-SA-4.0 *(assumed, to be checked on first fetch)* | wikitext | 400,000 | `null` |
| `bn-wikisource-gitanjali-ccbysa` | bn | CC-BY-SA-4.0 *(assumed, to be checked on first fetch)* | wikitext | 400,000 | `null` |

The CC-BY-SA-4.0 tags for the two Wikisource works are the conservative default for
Wikimedia text. They are recorded as assumptions, not findings: on first fetch the page's
own licence tag must be read and the manifest corrected (a PD template would allow
`PD-US`). Gitanjali's author died in 1941, so the underlying work is public domain in
India, but the specific edition on Wikisource may carry its own tag.

## 5. Coverage: all 14 slots, honestly

Statuses are assigned per slot and reported for **every** slot:

| status | meaning |
|---|---|
| `EVALUATED` | has verified sources **and** meets both targets |
| `INSUFFICIENT` | has verified sources, below target (reason states the real shortfall) |
| `UNVERIFIED` | sources are declared but none verified (reason states the ingest status) |
| `NOT_EVALUATED` | no source declared at all (reason from the manifest) |

Current state of the build recorded as **EXP-008**:

| status | count | languages |
|---|---|---|
| `EVALUATED` | 0 | — |
| `INSUFFICIENT` | 0 | — |
| `UNVERIFIED` | 3 | en, hi, bn |
| `NOT_EVALUATED` | 11 | hi-en, mr, gu, ta, te, kn, ml, pa, or, as, ur |

Why the 11 are not evaluated:

* **hi-en** — no licensed, attributable Hinglish corpus was identified. The published
  Hinglish/Romanised-Hindi corpora are social-media derived, with unclear licences and
  privacy terms that the allow-list does not permit. Declaring one would require a licence
  review that has not happened.
* **mr, gu, ta, te, kn, ml, pa, or, as, ur** — a family-level candidate provider exists
  (the relevant Wikisource, CC BY-SA 4.0 assumed conservatively), but no concrete work has
  been declared. Selecting a work, confirming its licence tag and pinning its hash require
  network access and review that have not happened. They are recorded as `candidates`
  with a provider, not as sources, precisely so they cannot be mistaken for coverage.

No language is marked covered by proxy, and no data is synthesized to fill a slot.

## 6. Ingestion and normalization

`ingest_source` writes the cleaned text to `sources/<id>.txt` plus a
`sources/<id>.provenance.json` (reusing `write_provenance`), and returns an
`IngestedSource` with status, path, hash, char/byte counts and any error.

Statuses: `verified`, `local_unverified`, `fetch_failed`, `licence_marker_missing`,
`empty`, `not_attempted`.

**`normalization_policy` is `none`.** Stage A never applies NFC, NFD or NFKC, never
folds case, and never rewrites characters beyond the existing kind-specific cleaners
(Gutenberg header/footer stripping, wikitext markup reduction). Silently normalizing here
would make every later normalization experiment unreviewable — the comparison belongs to a
later stage, and the source text must survive untouched until then. A test asserts that
NFD text written through ingestion comes back as NFD.

Documents are paragraphs. A paragraph longer than 1,200 characters is cut on sentence
punctuation (`। ॥ . ! ?`) and only then hard-wrapped, so no text is ever dropped;
document ids are `<source_id>-<index:06d>` in document order.

## 7. Split

```
method            deterministic-hash
seed              1337
heldout_fraction  0.1
level             document
```

A document goes to the held-out set when
`int(sha256("<seed>:<doc_id>")[:16], 16) / 0xFFFF_FFFF_FFFF_FFFF < 0.1`.

There is no global RNG and no dependence on document order or on which sources were
ingested first, so:

* the same corpus and seed always yield the same split (verified by test);
* reordering the documents does not change any document's side (verified by test);
* changing the seed re-splits reproducibly;
* train and held-out are **disjoint by construction** — a document has one bucket.

The split is recorded in `corpus.json` and `stats.json` with method, seed, fraction,
level and the resulting counts.

## 8. Leakage diagnostics

`leakage.json` carries two cheap, deterministic checks:

* **Exact overlap** — SHA-256 of every held-out document compared against the train set.
  Disjointness is guaranteed by the split; this check exists so a duplicated paragraph
  *inside the source text* (which produces two different doc ids with identical text) is
  caught rather than assumed away.
* **n-gram overlap** — word 8-grams (character 8-grams as a fallback for documents with
  fewer than 8 whitespace-separated words) from the held-out set that also occur in train.
  The train side is sampled deterministically at the first 2,000 documents so the check
  stays bounded; `sampled: true` is recorded when that cap bites.

This is a diagnostic, not a proof: shared idiom, quotations and boilerplate produce
genuine overlap. It is here to catch accidental duplication, not to certify independence.

When the corpus is empty the report says so instead of implying a clean result:
`note: "no documents to compare…"`. The current build is exactly this case.

## 9. Statistics, reproducibility and experiment integration

`stats.json` reports, for **every** slot including the empty ones: examples, chars,
UTF-8 bytes, train/eval examples and chars/bytes, source count, source ids, licence ids,
licence status, verification status and the per-source ingest statuses. `corpus.json`
adds file paths, sizes and SHA-256s for every artifact, plus the manifest hash.

**Reproducible**: two builds of the same manifest produce byte-identical
`train.jsonl`, `heldout.jsonl`, `stats.json`, `coverage.json` and `leakage.json`
(verified by test and by two CLI runs; the five SHA-256s matched). `corpus.json` carries
a `built_at` timestamp and is therefore *not* byte-reproducible by design — that is the
only difference between two runs.

`load_corpus()` re-reads `corpus.json` and re-hashes every file it names, so a corpus
that was edited after the build is rejected rather than trusted.

**EXP-008** is the corpus experiment. The CLI records itself through the standard
lifecycle (`run_self_recorded`), so:

```
python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
```

* writes one record at `out/experiments/EXP-008/experiment.json`
  (fingerprint `f484beb4e8acf2aa…`, status `success`, inputs = the manifest);
* when nested inside another experiment it writes **no** record of its own and publishes
  one `frontier_ai_nested_results` line instead (D-032/D-034);
* `--no-record` disables recording for ad-hoc inspection.

Recorded `results` for the current build:

```
sources_declared   3     sources_verified     0
language_slots    14     slots_evaluated      0
slots_insufficient 0     slots_unverified     3
slots_not_evaluated 11   documents            0
exact_overlap_count 0    ngram_overlap_ratio  0.0
```

Those zeros are measurements of the truth, not placeholders: every declared source was
actually attempted and failed.

## 10. Commands

```
# validate the manifest and report coverage (no network, no record)
python scripts/build_tokenizer_corpus.py --no-record

# attempt a real fetch and pin whatever verifies
python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-008

# one source, longer timeout
python scripts/build_tokenizer_corpus.py --fetch --source hi-wikisource-godan-ccbysa \
    --timeout 60 --exp-id EXP-008

# print a built corpus manifest (re-hashing every file it names)
python scripts/build_tokenizer_corpus.py --print-json --out data/tokenizer/indic-tokenizer-v2

# include locally supplied (unverified) text in the split — still never EVALUATED
python scripts/build_tokenizer_corpus.py --include-unverified --no-record
```

Exit codes: `0` built, `1` build failed, `2` the manifest is invalid.

## 11. Limitations

1. **No source has been verified.** The environment where Stage A was implemented can
   reach GitHub and PyPI only; every corpus host (Project Gutenberg, all Wikimedia hosts
   including Wikisource) fails at the TLS handshake. The three declared sources were
   genuinely attempted and every attempt is recorded as `fetch_failed` with the error.
   Nothing was fetched, so nothing was hashed, so nothing was pinned.
2. **The corpus is therefore empty**: 0 documents, 0 train, 0 held-out. All 14 slots are
   `UNVERIFIED` (3) or `NOT_EVALUATED` (11). Every downstream tokenizer stage must
   re-run this build in an environment with network access before it can claim anything
   about a language.
3. **The leakage result is vacuous** until there is data. It reports `0` overlap over
   `0` documents and says so explicitly.
4. **Coverage targets are unmet for every slot** and reported as such.
5. **No vocabulary, no tokenizer, no normalization, no pre-tokenizer work.** No claim
   about tokenizer quality is made or implied anywhere in Stage A.
6. **Cleaning is the existing smoke-corpus cleaner** — a Gutenberg/wikitext reducer, not
   a general parser. Real Wikisource pages will need a review pass; anything exotic is
   left in the text for a human to notice.
7. **The `hi-en` slot may have no legal answer** at this scale. If no licensed source can
   be found, the honest outcome is for the slot to stay `NOT_EVALUATED` with the reason
   recorded, and for later stages to report tokenizer results without it.

## 12. What Stage B must add

* real, verified, licensed sources for as many of the 14 slots as possible (network
  access plus a licence review per work);
* pinned hashes and `verified: true` after those fetches;
* enough text per language to clear 500 sentences / 200k characters, or an honest
  `INSUFFICIENT` with the real number;
* only then: the vocabulary ladder and tokenizer-family comparisons, which are explicitly
  out of scope here.
