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
| `corpora/tokenizer/indic-tokenizer-v2/sources.json` | the corpus manifest (committed): corpus id/version, targets, split spec, policies, 14 language slots, 3 declared sources, licence-evidence endpoints |
| `src/frontier_ai/tokenization/research_corpus.py` | manifest model + validation, licence evidence, ingestion, document splitting, split, leakage, statistics, coverage, build |
| `scripts/build_tokenizer_corpus.py` | the CLI (records itself; see §9) |
| `tests/test_tokenizer_corpus.py`, `tests/tokenizer_corpus_fixtures.py` | 78 tests over all of the above |
| `data/tokenizer/indic-tokenizer-v2/` | build output (git-ignored): `corpus.json`, `train.jsonl`, `heldout.jsonl`, `stats.json`, `coverage.json`, `leakage.json`, `sources/<id>.txt`, `sources/<id>.provenance.json` |

The manifest reuses `frontier_ai.data.corpora` for everything it already does well:
fetching (`fetch_text`), kind-specific cleaning (`clean_text`/`trim_text`), provenance
records (`build_provenance`) and hashing (`sha256_text` / `sha256_file`). Nothing was
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
`retrieved_at`, notes, and — where the payload cannot carry its own licence — a
`license_evidence` block (§4).

`validate_manifest` rejects: an unknown schema version, a corpus id that is not
`[a-z0-9-]`, a version that is not `v<digits>`, a language slot outside the researched
set, duplicate slots, unknown keys, a licence outside the allow-list, non-https URLs, an
attribution shorter than 16 characters, a `max_chars` outside `(0, 2_000_000]`, an unknown
kind, a malformed `sha256`, `verified: true` without a pinned hash — **and** the three
reference errors that used to be silent:

* a slot referencing a source id that does not exist (it used to degrade quietly to
  `NOT_EVALUATED`);
* duplicate source ids;
* a declared source referenced by no slot (an orphan that could never contribute);

plus a malformed licence-evidence block (non-https URL, unknown kind/scope, no marker to
look for).

### Targets are goals, not quotas

```
sentences_per_language   500
characters_per_language  200,000
```

A slot below target is reported as `INSUFFICIENT` with the real amount and the real
percentage of target. Nothing is padded, duplicated or synthesized to reach a number.
(Note the unit: `actual_sentences` counts *documents* — paragraphs, or chunks of at most
1,200 characters — which is what the corpus is split into. For prose this target is met
well before the character target binds.)

## 4. Licensing

Allowed licences are the repository allow-list, unchanged:
`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `PD-US`. Anything else is rejected by validation.

Rules enforced in code, not just in prose:

1. **A source is `verified` only when its licence was proven by a recorded mechanism**,
   and the mechanism is written into the record (`licence_proof`):
   * `payload-marker` — the fetched payload itself carries a marker for the licence the
     manifest claims (the Project Gutenberg case); or
   * `licence-evidence` — a separately fetched, declared evidence endpoint proves it (§4.1).
2. **No hash is ever invented.** All three declared sources carry `sha256: null` and
   `verified: false`. Hashes are pinned only by `--pin` after a verified source.
3. **A hash is never licence proof.** Local text gets a content hash too; the provenance
   file says `hash_is_licence_proof: false` next to it.
4. **Local text is never auto-verified.** Text supplied from disk is ingested as
   `local_unverified` and can never make a language slot count as `EVALUATED`.
5. **Unverified text is excluded from the corpus by default** (`--include-unverified`
   opts in, and still cannot make a slot `EVALUATED`).
6. Attribution text is mandatory and travels with the corpus into every provenance file.

### 4.1 Licence evidence (Wikisource and friends)

Some hosts never put their licence in the payload we want. A Wikisource `?action=raw`
fetch returns bare wikitext: the CC BY-SA notice lives in the rendered page, not in the
text. Rather than weaken the licence check, a source may declare an auditable evidence
endpoint:

```json
"license_evidence": {
  "url": "https://hi.wikisource.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json",
  "kind": "mediawiki-api",
  "marker": "https://creativecommons.org/licenses/by-sa/4.0/",
  "scope": "site",
  "note": "…"
}
```

`kind` selects how the payload is read — `mediawiki-api` (parse JSON, search the
serialized document), `html` (search the page) or `plain` (search the text). `marker`
defaults to the source's `license_url`, so the usual check is "the host declares exactly
this licence".

The **entire result is recorded** — URL, kind, scope, marker, timestamp, status
(`ok` / `marker_not_found` / `fetch_failed` / `malformed`) and whether it was accepted —
in `corpus.json` and in the provenance file. Rules:

* the evidence is fetched over https only, and a fetch failure is a **failure**, never a
  fallback to trust;
* `scope` is recorded because site-level evidence is *weaker*: it proves
  `hi.wikisource.org` publishes under CC BY-SA 4.0, **not** that this specific page
  carries that tag. The page-level tag still needs a human review, recorded in the
  manifest, before the text is trusted for redistribution;
* no evidence configured and no payload marker ⇒ the source stays unverified. There is no
  path that silently bypasses licence verification.

Current declared sources (all three **unverified**):

| id | language | licence | evidence | `max_chars` | `sha256` |
|---|---|---|---|---|---|
| `en-gutenberg-alice-pd` | en | PD-US | payload marker (PG `.txt` files carry it) | 400,000 | `null` |
| `hi-wikisource-godan-ccbysa` | hi | CC-BY-SA-4.0 *(assumed)* | MediaWiki `meta=siteinfo` rightsinfo, **site** scope | 400,000 | `null` |
| `bn-wikisource-gitanjali-ccbysa` | bn | CC-BY-SA-4.0 *(assumed)* | MediaWiki `meta=siteinfo` rightsinfo, **site** scope | 400,000 | `null` |

The CC-BY-SA-4.0 tags for the two Wikisource works are the conservative default for
Wikimedia text. They are recorded as assumptions, not findings. Gitanjali's author died in
1941, so the underlying work is public domain in India, but the specific edition on
Wikisource may carry its own tag.

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

## 6. Ingestion, normalization and local files

`ingest_source` writes the cleaned text to `sources/<id>.txt` plus a
`sources/<id>.provenance.json`, and returns an `IngestedSource` with status, path, hash,
char/byte counts, truncation facts, the licence-evidence result and any error.

Statuses: `verified`, `local_unverified`, `fetch_failed`, `licence_marker_missing`,
`empty`, `not_attempted`.

**`normalization_policy` is `none`.** Stage A never applies NFC, NFD or NFKC, never
folds case, and never rewrites characters beyond the existing kind-specific cleaners
(Gutenberg header/footer stripping, wikitext markup reduction). Silently normalizing here
would make every later normalization experiment unreviewable — the comparison belongs to a
later stage, and the source text must survive untouched until then. A test asserts that
NFD text written through ingestion comes back as NFD.

**Truncation is visible.** When `max_chars` trims a work, `IngestedSource.truncated` is
`True`, `chars_before_trim` records the cleaned length before the cut, both are written
into the provenance file, and `stats.json` reports `truncated_sources` per language. A
researcher can always tell that a source is partial.

**Local files.** Lawfully obtained text can be supplied without any network:

```
python scripts/build_tokenizer_corpus.py --local-file hi-wikisource-godan-ccbysa=~/godan.txt
python scripts/build_tokenizer_corpus.py --local-dir data/local-sources   # <source_id>.txt
```

Local text is ingested as `local_unverified`: its hash is computed (it identifies the
bytes) and recorded with `hash_is_licence_proof: false`, the provenance records
`local_origin_path`, and the slot stays `UNVERIFIED` no matter how large the text is. It
enters the split only with `--include-unverified`. An unknown source id or a file that
matches no declared source is a hard error (exit 2), never a silently ignored input.

Documents are paragraphs. A paragraph longer than 1,200 characters is cut on sentence
punctuation (`। ॥ . ! ?`) and only then hard-wrapped, so no text is ever dropped;
document ids are `<source_id>-<index:06d>` in document order.

## 7. Split

```
method            deterministic-hash
seed              1337            (the manifest's; --seed overrides it)
heldout_fraction  0.1
level             document
```

A document goes to the held-out set when

```
sha256(f"{seed}:{sha256(text)}") → [0, 1]  <  heldout_fraction
```

The bucket key is the **content hash**, not the `doc_id`. That is what guarantees the
property the later stages rely on: byte-identical text cannot appear on both sides, even
when a source repeats a paragraph (refrains, chapter headers, boilerplate). Two copies of
the same text hash to the same bucket, so they land together. With `doc_id` bucketing they
did not — a real corpus with repeated lines leaked 4 identical documents across the split
before this change.

There is no global RNG and no dependence on document order, so the same corpus and seed
always yield the same split, reordering documents does not move any document, and changing
the seed re-splits reproducibly. Train and held-out are disjoint by construction.

The CLI uses the manifest seed unless `--seed` is given explicitly; the effective seed is
recorded in `corpus.json`, `stats.json` and the experiment record.

## 8. Leakage diagnostics

`leakage.json` carries two cheap, deterministic checks:

* **Exact overlap** — the SHA-256 of every held-out document compared against the train
  set. The split already makes this impossible; the check exists so that any regression in
  that guarantee (or any other path that assembles the two files) is caught.
* **n-gram overlap** — word 8-grams (character 8-grams as a fallback for documents with
  fewer than 8 whitespace-separated words) from the held-out set that also occur in train.
  **Sampling rule** (recorded as `sampling_rule` in the report): documents are ordered by
  `(language, doc_id)`, every language present gets an equal budget of
  `limit // n_languages` documents (limit = 2,000), and any budget left over because a
  language had fewer documents is filled in the same deterministic order. `per_language_sampled`
  reports what each language contributed. Sampling the *first* 2,000 sorted documents
  would have compared everything against whichever language sorts first.

This is a diagnostic, not a proof: shared idiom, quotations and boilerplate produce
genuine overlap. It is here to catch accidental duplication, not to certify independence.

When the corpus is empty the report says so instead of implying a clean result:
`note: "no documents to compare…"`. The current build is exactly this case.

## 9. Statistics, reproducibility and experiment integration

`stats.json` reports, for **every** slot including the empty ones: examples, unique texts,
duplicate documents, truncated sources, chars, UTF-8 bytes, train/eval examples and
chars/bytes, source count, source ids, licence ids, licence status, verification status
and the per-source ingest statuses.

**The per-language counts are computed from the already-split document lists**, never by
re-splitting. An earlier version recomputed the split from the manifest seed and silently
disagreed with the written artifacts whenever a seed was overridden (observed: `stats.json`
2711/290 vs the actual 2680/321). Statistics that cannot disagree with the data are the
only kind worth publishing; a test compares `stats.json` against `train.jsonl` /
`heldout.jsonl` line by line.

`corpus.json` adds file paths, sizes and SHA-256s for every artifact, plus the manifest
hash, and `load_corpus()` re-hashes every file it names, so a corpus edited after the build
is rejected rather than trusted.

**Reproducible**: two builds of the same manifest produce byte-identical `train.jsonl`,
`heldout.jsonl`, `stats.json`, `coverage.json` and `leakage.json` — verified with an empty
corpus and with ~9,000 documents of local text. `corpus.json` carries a `built_at`
timestamp and is therefore *not* byte-reproducible by design; that is the only difference
between two runs.

**EXP-008** is the corpus experiment. The CLI records itself through the standard
lifecycle (`run_self_recorded`), so:

```
python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
```

* writes one record at `out/experiments/EXP-008/experiment.json`
  (fingerprint `f9c73d7a5cc54980…`, status `success`, inputs = the manifest);
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

# ingest lawfully obtained local text (local_unverified; never EVALUATED)
python scripts/build_tokenizer_corpus.py --local-file hi-wikisource-godan-ccbysa=~/godan.txt \
    --include-unverified --no-record
python scripts/build_tokenizer_corpus.py --local-dir data/local-sources --no-record

# print a built corpus manifest (re-hashing every file it names)
python scripts/build_tokenizer_corpus.py --print-json --out data/tokenizer/indic-tokenizer-v2
```

Exit codes: `0` built, `1` build failed, `2` bad input (invalid manifest, unknown source
id, local file that matches no declared source).

`--pin` rewrites the manifest **only** when at least one source was genuinely verified;
pinning zero sources leaves the tracked file byte-identical. When it does write, it does
not add empty `candidates` / `reason` keys to slots that never had them.

## 11. Limitations

1. **No source has been verified.** The environment where Stage A was implemented can
   reach GitHub and PyPI only; every corpus host (Project Gutenberg, all Wikimedia hosts
   including Wikisource) fails at the TLS handshake. The three declared sources were
   genuinely attempted and every attempt is recorded as `fetch_failed` with the error.
   Nothing was fetched, so nothing was hashed, so nothing was pinned.
2. **The corpus is therefore empty**: 0 documents, 0 train, 0 held-out. All 14 slots are
   `UNVERIFIED` (3) or `NOT_EVALUATED` (11). Every downstream tokenizer stage must
   re-run this build where the network allows before it can claim anything about a
   language.
3. **The leakage result is vacuous** until there is data. It reports 0 overlap over 0
   documents and says so explicitly.
4. **The licence-evidence endpoints have never been reached**, so the CC BY-SA
   assumption is still an assumption. When the first fetch happens, the evidence result
   will be recorded verbatim — including a `scope: site` reminder that the *page's* own
   licence tag still needs a human review.
5. **Coverage targets are unmet for every slot** and reported as such.
6. **No vocabulary, no tokenizer, no normalization, no pre-tokenizer work.** No claim
   about tokenizer quality is made or implied anywhere in Stage A.
7. **Cleaning is the existing smoke-corpus cleaner** — a Gutenberg/wikitext reducer, not
   a general parser. Real Wikisource pages will need a review pass; anything exotic is
   left in the text for a human to notice.
8. **Documents are not de-duplicated**, only split apart: repeated paragraphs are kept
   (they are real text) but can no longer straddle the split. `stats.json` reports
   `duplicate_documents` per language so the amount of repetition is visible.
9. **No per-language balancing or genre control.** With one work per language, later
   cross-language `bytes_per_token` comparisons will be confounded by author, genre and
   era as well as by script. That has to be handled — by equal-sizing, or by stating the
   confound in every report — before any tokenizer comparison (Stage C), not before
   acquisition.
10. **The `hi-en` slot may have no legal answer** at this scale. If no licensed source can
    be found, the honest outcome is for the slot to stay `NOT_EVALUATED` with the reason
    recorded, and for later stages to report tokenizer results without it.

## 12. What Stage B must add

* real, verified, licensed sources for as many of the 14 slots as possible (network
  access plus a licence review per work); the first fetch must confirm that the
  `?action=raw` payload really is the work and not an index page, and that the licence
  evidence resolves;
* pinned hashes and `verified: true` after those fetches;
* enough text per language to clear 500 documents / 200k characters, or an honest
  `INSUFFICIENT` with the real number;
* only then: the vocabulary ladder and tokenizer-family comparisons, which are explicitly
  out of scope here.

## 13. Fixes made after the Stage A review

A read-only review of the first Stage A implementation found eight issues. All are fixed;
each fix has a regression test.

| Issue | What was wrong | Fix |
|---|---|---|
| H-1 | `language_statistics` re-split with the **manifest** seed while `build_corpus` used the effective seed; the CLI hard-coded `1337`, so the manifest seed was never honoured | statistics are computed from the already-split lists; `--seed` defaults to the manifest seed; tests compare `stats.json` against the JSONL files under a non-default seed |
| H-2 | `?action=raw` wikitext carries no CC BY-SA marker, so the Wikisource sources could never verify | `license_evidence` blocks (MediaWiki `meta=siteinfo`, `mediawiki-api`/`html`/`plain`), fetched and recorded with status, marker and scope; failure ⇒ unverified, never a bypass |
| H-3 | No way to ingest lawfully obtained files; `--include-unverified` was inert | `--local-file ID=PATH` / `--local-dir DIR`; local text is `local_unverified`, hash flagged `hash_is_licence_proof: false`, never `EVALUATED` |
| M-1 | Identical documents could straddle the split (4 leaked in a probe) | split buckets on `sha256(text)` instead of `doc_id`; duplicates now land together |
| M-2 | n-gram sample was the first 2,000 sorted documents — one language only | stratified per-language budget with deterministic fill; `per_language_sampled` recorded |
| M-5 | Slot→source references, duplicate ids and orphan sources were silent | all three are validation errors (and `build_corpus` refuses an invalid manifest) |
| M-6 | `max_chars` truncation was invisible | `truncated` + `chars_before_trim` in the ingest result, the provenance file and `stats.json` |
| M-7 | `--pin` rewrote the manifest even when it pinned nothing, adding empty keys | it is a no-op unless something was pinned, and it never invents empty `candidates` / `reason` keys |
