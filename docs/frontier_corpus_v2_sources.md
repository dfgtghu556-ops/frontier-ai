# FrontierCorpus v2 — source survey and data-rights options (data scale-up, phase 1)

**Status (2026-09-29):** research document; nothing has been downloaded, and no source
here is registered yet. The founder approved *data scale-up phase 1* on 2026-09-29
("approve D-043 and data plan"). Phase 1 has three parts:

1. this source survey;
2. a data-rights policy for the founder to decide (**D-044**, §6; proposed, not yet a
   decision);
3. making the corpus pipeline ready for far more data (§7).

**Governing records:**
- MASTER_CONTEXT §10–15 (what a pretraining corpus must be) and §37 (method);
- D-035 (the tokenizer corpus is frozen), D-036 (NFC), D-037 (staged pipeline),
  D-041 (tokenizer v1 frozen), D-042 (the protected suite `frontier-heldout-v1` must
  never be trained on);
- the standing rule: **no fixed mixing percentages**. Mixes are compared by experiment.

**Evidence rule:** every size and license below was read from the cited page on
2026-09-29 and is quoted in the unit the source uses (words, tokens, rows, bytes).
Anything not read from a source is marked **NOT VERIFIED**. Our own token counts exist only
after a build, measured with Frontier Tokenizer v1.

---

## 1. Why scale up now

| Fact | Value | Source |
|---|---|---|
| Current training data | ≈1.79M tokens (`mark_aware-32768.bin`, 3,586,440 bytes of uint16) | repo, EXP-029 |
| Current corpus | FrontierCorpus v1 pilot: 30,584 train / 3,427 held-out documents, ≈4.2M characters, **literature only** (57 Wikisource pages + 2 Project Gutenberg books) | `docs/frontier_corpus_v1.md`, `sources.json` |
| Rule of thumb for compute-optimal training | ≈20 training tokens per model parameter | Hoffmann et al. 2022 ("Chinchilla") |
| What that implies | 1.79M tokens ≈ enough for a ≈90K-parameter model; a 124M-parameter model would want ≈2.5B tokens | arithmetic |

Step 10 (GPU bring-up) can run on the current data because it tests the machinery, not
quality. From step 11 on (GPU-scale experiments), **data is the binding constraint**: a
bigger model on 1.8M tokens memorises instead of learning.

## 2. Legal background in India (context, not legal advice)

The founder must decide the policy with this background. It is summarised from the
sources, not interpreted beyond them. A lawyer should review before any commercial release
(open question Q-7 already covers the model license).

- **No text-and-data-mining (TDM) exception in Indian law today.** The DPIIT committee's
  *Working Paper on Generative AI and Copyright, Part 1* (December 2025) states that the
  Copyright Act 1957 has "no specific exception under copyright law for text and data
  mining". It proposes a **hybrid model**: a blanket licence to train on lawfully accessed
  works, with royalties payable only when an AI system is commercialised, collected by a
  new body (CRCAT). It is a proposal, **not law**; as of May 2026 no amendment had been made.
  Sources: DPIIT paper (dpiit.gov.in, Dec 2025); PIB release ID 2200741 (9 Dec 2025);
  intepat.com summary (13 May 2026).
- **First court ruling (provisional).** In *ANI Media v. OpenAI* (Delhi High Court,
  I.A. 45300/2024 in CS(COMM) 1028/2024), the court **refused an interim injunction** on
  24 July 2026. It found that training copies engaged the reproduction right, but were
  prima facie **fair dealing** under Section 52(1)(a). The findings are provisional, the
  suit continues, an appeal is possible, and the court left open whether the reasoning
  covers more expressive works (literature, music, film). Sources: intepat.com
  (6 Aug 2026), mondaq.com (12 Aug 2026).
- **What this means for us:** training on openly licensed or public-domain text is the
  low-risk path. Training on web crawls of copyrighted pages currently rests on an
  unsettled fair-dealing argument, and a future statutory royalty on commercial use is
  possible. Either way, **complete per-source provenance records** (which this repo
  already keeps) are what any future regime asks for (the DPIIT proposal includes an "AI
  Training Data Disclosure Form").

## 3. Candidate sources

Trust levels (used in §6):
- **A:** the authors themselves released the text openly, or it is in the public domain;
- **B:** a curated web-crawl collection whose *collection* license is permissive, but
  whose underlying pages keep their authors' copyright;
- **C:** machine-translated or synthetic text;
- **X:** excluded, or license not verified.

| Source | What it is | Our languages | Size (as reported) | License (evidence, checked 2026-09-29) | Level |
|---|---|---|---|---|---|
| **Wikipedia** (12 Indic editions + English) | encyclopedia articles written by volunteers | all 13 | 12 Indic editions: 1,750,823 articles, **≈552M words**; English: 7,245,707 articles, ≈5.29B words (per-language table in §4) | text under **CC BY-SA 4.0 + GFDL** (dumps.wikimedia.org/legal.html); attribution + share-alike | **A** |
| **Wikisource** (full editions) | public-domain and freely licensed books/texts | all 13 | e.g. hi 10.2M words, bn 17.4M words (siteinfo `cirrussearch-article-words`) | CC BY-SA 4.0 + GFDL for Wikimedia text (same legal page); hosted works are meant to be PD or freely licensed (per-work status NOT VERIFIED) | **A** (⚠ overlaps the protected suite, §5) |
| **`wikimedia/wikipedia`** on Hugging Face | pre-cleaned plain text of the Wikipedia dumps (snapshot `20231101`) | all 13 | per-language configs | card: **cc-by-sa-3.0, gfdl** | **A** (older snapshot; convenient: no wikitext parsing) |
| **Project Gutenberg** | public-domain books, mostly English | en (few Indic) | not measured | PD in the US; "verify local copyright terms" (gutenberg.org policy, already recorded in v1) | **A** (US-PD ≠ India-PD for every book; check per book) |
| **IndicCorp v2** (AI4Bharat) | web-crawled text (content mix NOT VERIFIED) | our 12 Indic languages (+ others; no English config) | 20.9B tokens, 24 languages, ≈275GB | card text: data under **CC0**; no license metadata field on the HF API | **B** |
| **Sangraha** (AI4Bharat) | three parts: *Verified* (human-verified websites, OCR'd PDFs, transcribed video/audio), *Unverified* (filtered from existing web corpora), *Synthetic* (English Wikimedia machine-translated into 14 languages) | all 13 | 251.3B tokens total: Verified 64.3B, Synthetic 162.7B, Unverified 24.3B; 705GB; Verified hi 12,617M, eng 12,760M, bn 10,604M tokens | HF metadata tag **license: cc-by-4.0** (HF API) | Verified: **B**; Unverified: **B/X** (source URLs not verifiable); Synthetic: **C** |
| **FineWeb-2** (Hugging Face) | filtered Common Crawl, 96 snapshots 2013–2024 | all 13 | ≈20TB, ≈5B documents, 1,868 language-script pairs; sizes in words | **ODC-By 1.0**, and also subject to the **Common Crawl Terms of Use** (card; HF metadata `odc-by`) | **B** |
| **HPLT v2 cleaned** | web crawl (Internet Archive + Common Crawl) | all 13 have configs | e.g. ben_Beng 11M rows, tam_Taml 6.11M rows | license **NOT VERIFIED** (not in the HF metadata we read) | **X** until verified |

Not surveyed yet (need their own check before any use): government publications
(PIB, Parliament debates; whether and how Indian government works may be reused is NOT
VERIFIED), textbooks (NCERT etc., copyrighted),
news sites (copyrighted), and Common Crawl directly (same issue as level B).

## 4. Level-A size in detail: Wikipedia by language

Read from each wiki's `action=query&meta=siteinfo&siprop=statistics` API on 2026-09-29.
`words` is the MediaWiki search index's article word count. It is **not a token count**,
and some wikis contain many short bot-created articles that quality filters may remove.

| Language | Articles | Words |
|---|---|---|
| bn | 191,666 | 115,528,250 |
| ur | 703,185 | 111,342,688 |
| te | 129,248 | 67,808,026 |
| hi | 171,757 | 64,353,978 |
| ta | 190,747 | 56,666,934 |
| ml | 89,100 | 30,546,028 |
| kn | 34,654 | 26,149,545 |
| pa | 59,734 | 24,708,701 |
| mr | 103,122 | 19,541,212 |
| as | 25,624 | 18,764,587 |
| gu | 30,913 | 9,558,688 |
| or | 21,073 | 7,138,072 |
| **12 Indic total** | **1,750,823** | **≈552M** |
| en | 7,245,707 | 5,291,088,501 |

Download size example: `hiwiki-latest-pages-articles.xml.bz2` is 240,367,815 bytes
(dump of 2026-09-01; `md5sums.txt` and `sha1sums.txt` are published next to it, so every
download can be hash-verified). The total for the 12 Indic dumps is NOT VERIFIED yet; it is
listed from the dump index before any PC run, so the download plan fits the laptop's disk
and connection.

**Honest reading:** Indic Wikipedia is about 552M words, a few hundred times our current
corpus. It is still far from the ≈2.5B tokens a 124M model wants in its Indic part.
Level A alone will not reach that for most Indic languages, so the D-044 choice matters.
English Wikipedia alone is larger than all Indic sources in level A, and **how much
English to use is a mixing question** answered by experiment, not fixed here.

## 5. The protected suite and new sources (hard rule)

`frontier-heldout-v1` (D-042) is built from the same Wikisource pages as the v1 pilot. A
full Wikisource dump **will contain those exact texts**, so they must be removed:

- The exact-hash guard (`evaluation.suite.find_exact_overlap`) only catches identical
  documents. A dump page is a whole page while the suite stores paragraphs/lines, so the
  hashes will not match.
- The **13-gram check** (`evaluation.contamination`) catches it. It currently builds a set
  of *all training* n-grams in memory, which cannot work at hundreds of millions of words.
  It must be inverted: hold the small suite's n-grams in memory and stream the training
  documents past them.
- Policy for v2: any training document sharing a 13-gram with a suite document is
  **removed** (with a recorded reason). The v1 rule "report, don't remove" was for the
  pilot's own split; for new external sources the suite is protected.

## 6. D-044 (proposed): data-rights policy — options for the founder

| Option | What goes in | Plain meaning | Risk | Scale reachable |
|---|---|---|---|---|
| **1. Open only** | level A only (Wikipedia, Wikisource, PD books) | only text whose authors released it openly | lowest | ≈552M Indic words + English; too small for later steps in most Indic languages |
| **2. Open + curated collections (recommended to start)** | A, plus level B from collections with a stated permissive license and per-document source URLs (Sangraha *Verified*, IndicCorp v2, FineWeb-2), each tagged as B in the manifest so it can be removed later | the approach of open collections such as FineWeb-2 and Sangraha, with every document traceable | medium: rests on fair dealing (ANI v. OpenAI, provisional) and a possible future royalty | tens of billions of Indic tokens available (Sangraha Verified alone reports 64.3B) |
| **3. Everything available** | A + B + C (machine-translated) + unverified | maximum volume | highest; machine translation can also teach "translationese" | largest |

Rules that apply under every option:
- **Level C (machine-translated) stays out of the main corpus.** It may only enter as a
  separately tagged experimental arm, compared by experiment.
- Level X is never used.
- Every document records source, license, level and retrieval date (the v1 manifest
  pattern). A source can be dropped later by rebuilding without it.
- CC BY-SA sources require attribution and share-alike. This affects how *datasets* are
  redistributed. Whether it affects model weights is part of Q-7 (model license), to be
  settled before any release.
- Personal data: a PII-scrubbing stage is required before level-B data is used (§7).
- Robots/opt-out signals recorded by a collection (e.g. HPLT's `robotstxt` field) are
  respected where available.

**Recommendation:** option 2, built in stages. First build the level-A pilot (v2-pilot,
Wikipedia), because it tests the whole scaled pipeline with the least legal risk. Then add
level-B collections one at a time, each as its own recorded experiment. The founder can
stop at level A at any time.

## 7. Pipeline readiness: what breaks at 100–1000× (verified from the code)

| Area | Today (v1 pilot) | Needed for v2 |
|---|---|---|
| Memory | every stage takes `list[PipelineDocument]` (whole corpus in memory; fine at 4.2M chars) | streaming stages over sharded input, bounded memory on the founder's laptop (RAM NOT VERIFIED) |
| Duplicates | exact document-hash dedup only (`corpus/dedup.py`; near-dup was deliberately deferred to F3) | **near-duplicate removal (MinHash + LSH)**, deterministic, with recorded reasons |
| Contamination | 13-gram check holds all training n-grams in memory | inverted streaming check against the suite; removal for new sources (§5). **Built 2026-09-29:** `corpus/decontaminate.py` (§7a) |
| Language ID | script-share gate; cannot separate hi/mr or bn/as | acceptable for level A (the wiki edition gives the language); a classifier is needed for level B |
| PII | none | email / phone / ID-number scrubbing before level-B data |
| Quality | 7 rule-based checks tuned for verse lines | per-source calibration; model-based quality stays later (F3) |
| Input formats | Wikisource via the API parse (HTML) and Gutenberg text | Wikipedia dumps (wikitext XML) or the pre-cleaned `wikimedia/wikipedia` parquet; streaming JSON/parquet for collections |

## 7a. Built so far: the protected-suite guard (`corpus/decontaminate.py`)

- `SuiteGuard.from_texts(...)` indexes the suite once: exact text hashes plus every 13-gram,
  using the same n-gram definition as `evaluation.contamination`.
- `iter_decontaminate(...)` is the streaming form (bounded memory over sharded input);
  `decontaminate(...)` is the pipeline-stage form (`list → StageOutcome`).
- Removal reasons are `suite_exact` / `suite_ngram`, recording the suite doc_id and the
  number of matched n-grams.
- Tests (`tests/test_suite_decontaminate.py`, 9 tests) include the dump case (a suite
  paragraph inside a whole page) and a check that `contamination_report` finds zero exact
  and zero 13-gram overlap after the stage.
- Sandbox micro-benchmark (synthetic text, 2-core sandbox, informal):
  - a 3,427-document / ≈137K-word suite builds in 0.26 s with ≈18 MB peak memory
    (95,956 n-grams);
  - checking runs at ≈3.2M training words per second.
  - The founder's laptop will be slower; its speed is NOT VERIFIED.

## 8. Phase-1 work order (after this survey)

1. **Founder decides D-044** (option 1, 2 or 3). Then it is recorded in DECISIONS.md.
2. Sandbox (no GPU, no downloads): streaming document interface, MinHash near-dup stage,
   inverted suite-contamination check. Each gets tests and a small measured benchmark.
3. PC (network, overnight, one typed line): download the level-A Wikipedia dumps with hash
   verification, build **FrontierCorpus v2-pilot**, tokenize with the frozen v1 tokenizer,
   and record per-language token counts as a new experiment.
4. Only then: level-B sources one at a time (if D-044 allows), and mixing experiments
   (comparison framework, no fixed percentages). A future tokenizer v2 would be considered
   only then, under D-041 (founder approval + new experiment).

## 9. Addendum (2026-09-29): the founder's AI4Bharat suggestion

The founder pointed to AI4Bharat's LLM page (ai4bharat.iitm.ac.in/areas/llm) and asked to
use their free resources because it is fast. Checked the same day:

- **Their GitHub repositories hold code, not data.** `AI4Bharat/IndicLLMSuite` (MIT, about
  69 KB) and `AI4Bharat/setu` (MIT) contain pipelines and download scripts. The data
  (Sangraha, 705 GB) is on Hugging Face. It cannot be "copied into our repository": GitHub
  rejects files over 100 MB, and this repository keeps data out of git by design (shards
  and tokens are git-ignored and hash-pinned, as in v1).
- **Sangraha Verified is published as per-language parquet files, each with a SHA-256.**
  Read from the HF tree API:
  - `verified/asm` is 3 files (328,035,291 + 326,255,363 + 325,164,324 bytes);
  - `verified/hin` is more than 30 files of about 345–378 MB each.

  So a download can be **pinned file by file**, exactly like the v1 pins. One file per
  language is a natural first slice for the laptop. Its token count under Frontier Tokenizer
  v1 is NOT VERIFIED until measured.
- **What we would use and not use:**
  - *Sangraha Verified*: yes, as level B (option 2).
  - *Sangraha Synthetic* (machine-translated): no (level C).
  - *Sangraha Unverified*: not now.
  - *IndicAlign* (instruction/alignment data): not now. That is fine-tuning (SFT) data, and
    SFT is outside the founder's current guardrails; it is recorded for a later stage.
  - *Setu* (Apache Spark pipeline): built for clusters, too heavy for the laptop. Its ideas
    can be borrowed with credit (MIT).
  - *AI4Bharat models* (Airavata etc.): not used. The goal is a model trained from scratch.
- Legal position unchanged (§2): the CC-BY-4.0 tag covers the collection; the scraped pages
  underneath keep their authors' copyright, which is why it is level B and every document
  stays tagged and removable.
