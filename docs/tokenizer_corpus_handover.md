# P004B handover — the tokenizer research corpus, now owned by the local assistant

**Date:** 2026-09-25. **From:** the Arena agent, which did P004B until this commit.
**To:** the assistant on the operator's machine (Claude Code in VS Code, through OmniRoute).
**Scope:** P004B only — acquiring and verifying the real text corpus of `indic-tokenizer/v2`.
Nothing else in the repository changes hands.

The operator has decided that you own this part **end to end**: choosing the books, the
licence review, the code, the downloads, reviewing the results, locking the hashes and
recording everything. Nobody else reviews your work before a lock any more, so the checks
below are yours to apply. Read this whole file before doing anything, then the three rules
at the top of [tokenizer_corpus_stage_b_acquisition.md](tokenizer_corpus_stage_b_acquisition.md)
(the runbook, referred to as "runbook §N" below).

---

## 1. What "done" means

* The corpus has 14 language slots: `en hi bn gu ml or as pa kn te ta mr ur hi-en`.
* A slot is `EVALUATED` when its verified, pinned sources give **at least 500 documents and
  at least 200,000 characters** (`targets` in the manifest). Otherwise it is `INSUFFICIENT`,
  `UNVERIFIED` or `NOT_EVALUATED`, with the reason recorded. Statuses are defined in
  [tokenizer_corpus_stage_a.md](tokenizer_corpus_stage_a.md).
* **Nothing is substituted** (runbook §9): no synthetic text, no machine translation, no
  related-language filler, no padding. A slot without a lawful source stays unevaluated.
  `hi-en` (Hinglish) will probably stay `NOT_EVALUATED`.
* When the work is finished, record P004B as an experiment in [EXPERIMENTS.md](../EXPERIMENTS.md)
  (the next `EXP-0NN` heading, in the style of the existing entries).

## 2. Where things stand (branch tip `cbab11c` plus this handover commit)

| Slot | Source(s) | State |
|---|---|---|
| en | Alice + Sadhana (Gutenberg) | `EVALUATED`, pinned (EXP-013) |
| hi | *Godaan*, 36 chapters (Wikisource) | `EVALUATED`, pinned (EXP-013) |
| bn | *Gitanjali* + *Devdas* | `EVALUATED`, pinned (EXP-013) |
| gu, ml, or, as | one novel each, 7 range sources (runbook §2.6) | pinned (EXP-017); characters above target (gu 297,628 · ml 399,168 · or 203,444 · as 241,025); **document counts never confirmed** — or (about 420 lines) may fall below 500 documents |
| pa, kn, te, ta | one work each, 8 range sources (runbook §2.7) | declared, **never fetched**; JOB-002 (EXP-018) is their first download |
| mr, ur | none | `NOT_EVALUATED` |
| hi-en | none | `NOT_EVALUATED`, probably permanently (runbook §9) |

* The manifest has 55 sources: 47 verified and pinned, 8 unverified with `sha256: null`.
* The pa/kn/te/ta works were chosen and licence-reviewed by the Arena agent on 2026-09-25;
  each source's `notes` lists what was checked. **You own them now: re-check that review
  yourself (§6) before you lock them**, and replace a work if you disagree.
* The last experiment number used is **EXP-017**; EXP-018 is reserved for JOB-002's fetch.
* The operator's folder `E:\frontier-ai` contains an old untracked second copy of the
  project in `E:\frontier-ai\frontier-ai`. JOB-002 step 2 moves it out (never delete it).

## 3. Rules

Non-negotiable:

1. **Never pin a hash you did not compute from a fetch you actually performed.** Only the
   build script's `--pin` writes `sha256`, `verified` and `retrieved_at`; never type, copy or
   edit those values.
2. **Never mark a source verified because the manifest says it is CC BY-SA.** Verification
   is what the build does with live evidence.
3. **If a source cannot be licensed, leave it unverified.** An honest `NOT_EVALUATED` is a
   result; a fabricated one invalidates every later tokenizer number.
4. **Nothing is substituted or padded** (§1).

How to work:

5. **Git.** Only branch `arena/01a0d31f-frontier-ai`. Start with
   `git pull origin arena/01a0d31f-frontier-ai`; push only with
   `git push origin arena/01a0d31f-frontier-ai`. Never push to `main`, never force-push, never
   create/switch/merge/rebase branches, never use VS Code's Sync or Publish buttons (one of
   them once pushed this work to `main` as a parentless commit). Stage exact paths — never
   `git add .`, `-A` or `commit -a`.
6. **Code changes come with tests.** Before every commit run the full suite,
   `python -m pytest -q -o addopts=""`, and `ruff check src tests scripts` (fix import order
   with `ruff check --fix`). Never weaken a check or a test to make a result pass. Write the
   expected value of a new test from a real observation, never a guess.
7. **Manifest edits only through a Python script** that loads
   `corpora/tokenizer/indic-tokenizer-v2/sources.json`, first asserts that
   `json.dumps(m, ensure_ascii=False, indent=2) + "\n"` reproduces the file byte for byte,
   makes the change, and writes the same form back as UTF-8 with LF line endings. Keep the
   existing key order. Never hand-edit it in an editor that may change line endings.
8. **Windows.** PowerShell 5.1's `>` writes UTF-16: use the scripts' `--output` options.
   Indic text is garbled in the console and non-ASCII text in PowerShell here-strings breaks:
   put Python code in `.py` files, run them with `python -X utf8`, and read results from
   UTF-8 files rather than from the console.
9. **Cleaning changes move hashes.** A change to the cleaner (`src/frontier_ai/data/mediawiki.py`,
   `src/frontier_ai/tokenization/research_corpus.py`) can change the text of sources that are
   already pinned; the next fetch then refuses them (`hash_mismatch`, exit 1). So a cleaner
   change needs a test, a fetch showing exactly which sources change and why, and a
   deliberate re-pin that you record. Never strip the Gutenberg "Produced by" line (it would
   change Alice's pinned hash).
10. **Record every run** where the next person will look: the manifest's top-level `notes`
    (a single string — append one sentence-group per EXP), the runbook's status header, its
    §2.x section and §8 checklist for the group, and the reports you save in
    `corpora/tokenizer/indic-tokenizer-v2/reports/`. Commit and push after each step, so the
    state can always be read from git alone.

The local-runner protocol ([tokenizer_corpus_local_runner.md](tokenizer_corpus_local_runner.md))
was written for a helper that only executes jobs. Its rules 1, 2, 4, 6, 7, 8 and 9 still
apply to you; rules 3 ("touch only what the job names") and 5 ("report, don't repair") are
replaced by rules 6–10 above.

## 4. The tools

| Path | What it is |
|---|---|
| `corpora/tokenizer/indic-tokenizer-v2/sources.json` | the manifest: slots, sources, targets, notes |
| `src/frontier_ai/data/mediawiki.py` | turns rendered Wikisource HTML into text: page anchors and proofreading levels (`_Extractor._observe`), red links to missing templates (`TEMPLATE_NAMESPACE`), repairs (`{{gap}}` residue, stray `}}`, literal `<poem>`), `quality_summary()` |
| `src/frontier_ai/tokenization/research_corpus.py` | documents (one per non-empty line; a line over 1,200 characters is split at sentence ends, `_SENTENCE_BOUNDARY`), coverage (`coverage_report`), the lock (`--pin`) |
| `scripts/build_tokenizer_corpus.py` | fetch, clean, verify, build; exit `0` = no pinned source changed (not "all verified"), `1` = a pinned source changed (`REFUSED:` line), `2` = bad input |
| `scripts/inspect_corpus_sources.py` | read-only report of the last build: header line, per-source lines, flags, repairs, samples, coverage per slot; exit `1` when anything is flagged |
| `tests/test_tokenizer_corpus_mediawiki.py`, `tests/test_corpus_fetch_and_inspection.py` | the tests closest to this work (fixtures in `tests/tokenizer_corpus_fixtures.py`) |
| `docs/tokenizer_corpus_stage_b_acquisition.md` | the runbook: procedures, the history of every run, checklists |
| `corpora/tokenizer/indic-tokenizer-v2/reports/` | `JOB.md` (a written-down job) and the saved reports `EXP-0NN-*.txt` |

Commands (from the repository root; use `.venv\Scripts\python.exe` if plain `python` cannot
`import frontier_ai`; if the project is not installed, `python -m pip install -e ".[dev,tokenizer]"`):

* Fetch: `python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-0NN --out data/tokenizer/indic-tokenizer-v2`
* Lock: the same command with `--pin` added. It pins **every** source verified in that run;
  already-pinned sources come back identical and only their `retrieved_at` moves.
* Report: `python scripts/inspect_corpus_sources.py --output corpora/tokenizer/indic-tokenizer-v2/reports/EXP-0NN-inspection.txt`
* Offline check of the manifest (no network): the build without `--fetch`, for example
  `python scripts/build_tokenizer_corpus.py --exp-id OFFLINE-CHECK --out <a scratch folder outside the repository>`;
  it must exit 0 and show every slot that has sources as `UNVERIFIED`.

The build's gates (nothing to switch on): an API error → `fetch_failed`; a redirect, a table
of contents or under 500 characters → `index_page_refused`; any page at level 1 or 2, a level
it cannot confirm, or no page anchors at all → `unproofread_refused`.

## 5. The cycle for a group of languages (3–4 at a time)

1. **Research and choose** (§6).
2. **Declare** (§7), with tests; commit and push.
3. **Fetch** (EXP-N) → **inspect** → **review** (§8). If the cleaner needs a fix, fix it
   with a test and fetch again, until the report is clean.
4. **Lock** (EXP-N+1) → check the diff (§9) → commit the manifest → record.

## 6. Research recipes

Use the MediaWiki API from Python (`urllib.request` with a descriptive `User-Agent`); every
wiki is `https://<code>.wikisource.org/w/api.php`, add `format=json&formatversion=2`.

* **The wiki's own facts:**
  `action=query&meta=siteinfo|proofreadinfo&siprop=rightsinfo|namespaces` gives the Template
  namespace (id 10) as the wiki spells it, the Page and Index namespaces, the quality-level
  category names, and the rights URL (all wikis so far: `…/by-sa/4.0/deed.<code>`, which
  contains the licence-evidence marker). Known: Page/Index ids gu, as, kn, te, mr 104/106;
  ml 106/104; or, pa, ta 250/252; hi's Page namespace is 250, bn's 104. mr's levels are
  1 `तपासणी करायचे साहित्य`, 2 `समस्या असलेले`, 3 `मुद्रितशोधन`, 4 `प्रमाणित`.
* **Survey:** `https://<code>.wikisource.org/w/index.php?title=Special:IndexPages&limit=500&filter=proofreadOrValidated`
  (add `&order=size` for the largest first). Counts on 2026-09-24: gu 204, mr 156, ta 500+,
  te 333, kn 60, ml 214, pa 198, or 21, as 400 fully proofread scans.
* **Index details:** `prop=revisions&rvprop=content&rvslots=main&titles=<Index:file>`
  (author, year, publisher, progress, transclusion) and `prop=imageinfo` on the file
  (`imagerepository`: `local` or `shared` = Commons; `pagecount`).
* **Proofreading levels:** `list=search&srnamespace=<Page ns>&srsearch=incategory:<level
  category> prefix:<Page ns name>:<file>/` — **`prefix:` must come last**: it takes the rest
  of the query, and put first it silently finds nothing. Category names may contain spaces
  (write them with underscores). Always confirm a zero with a positive count (level 3 or 4)
  and with the total (`prefix:` alone), so that "no page at level 1 or 2" really means it.
  Single pages: `prop=proofread`.
* **Chapter ranges:** the work page and its chapter subpages hold
  `<pages index="…" from=… to=… fromsection=… tosection=… />`. A range rendered without
  section limits contains every section of every page once, in order, so ranges only need
  page numbers. `from`/`to` are always ASCII numbers, whatever digits the page titles use.
* **Page titles:** local digits on hi, bn, gu, or, as, kn (`…/೧೬`); ASCII on ml, pa, te, ta.
  On gu and as ASCII-digit titles look "missing"; on or they are level-1 redirects.
* **Missing templates:** `generator=templates&gtlnamespace=10&gtllimit=max&prop=info&titles=<work or
  chapter pages, or sample Page titles>` — look for `"missing": true`. The cleaner removes and
  counts any red link to a missing template anyway, provided the wiki is in
  `TEMPLATE_NAMESPACE`.
* **How the render marks pages:** render one page with `action=parse&prop=text&contentmodel=wikitext
  &text=<pages index="…" from=N to=N />` and look at `<span class="pagenum …">`:
  `data-page-name` + `data-page-quality` (hi, bn, as, pa, kn, te, ta — levels read from the
  render); a `title` attribute only (gu, or — the build asks the API for each page's level,
  shown as `level_lookup` in the report); ml's `span#pr_page` with a `prp-pagequality-N` link.
  A new shape needs code and tests before its sources can verify.
* **Licence review** (runbook §4): the work page's licence tag; the scan file page's tag
  (local or Commons); the author's death date (Wikidata `P570`, e.g.
  `https://query.wikidata.org/sparql?format=json&query=…`); the edition, year and publisher
  from the scan's title page. **Public domain in India in 2026 = the author died in 1965 or
  earlier** (life + 60 years, counted from the next 1 January). A work first published after
  the author's death is protected for 60 years from publication, so check that the text
  appeared in the author's lifetime. Exclude everything not established as the author's:
  editors' and publishers' prefaces, biographies, notes, advertisements, price lists.
  Tamil Wikisource's "nationalised" books (rights bought by the state) are a policy, not a
  licence: use a Tamil book only if it is public domain by term.
* **Size:** wikitext bytes per page ÷ about 2.7 ≈ characters for Indic scripts (sample page
  sizes with `list=search … &srprop=size`); Malayalam showed estimates can be off by a fifth
  or more, so aim for 250,000+. Documents are paragraphs (non-empty lines); prose with short
  paragraphs reaches 500 documents easily, verse and very long paragraphs may not.
* **Range size:** keep each render at or below about 170 scan pages (the largest so far is
  178); split long works into ranges that meet where a chapter starts if possible. Every page
  must be in exactly one range.

## 7. Declaring sources

* Copy the shape of an existing `mediawiki-parse` source (for example
  `ta-wikisource-en-charithram-p022-101-ccbysa`): the same keys in the same order, `sha256`
  `null`, `verified` `false`, `retrieved_at` `null`, `max_chars` 400000, a site-level
  `license_evidence` block for the wiki, and a `source_url` of the form
  `https://<code>.wikisource.org/w/api.php?action=parse&format=json&formatversion=2&prop=text&disablelimitreport=1&disableeditsection=1&contentmodel=wikitext&title=<quoted work title>&text=<quoted <pages … /> tag>`
  (quote with `urllib.parse.quote(value, safe='')`).
* `notes` should let a stranger re-check everything: the range and why it starts and ends
  there, what is excluded and why, the level counts with their date, the template check,
  the anchor shape, the title digits, and the licence review (tags, edition, death date with
  its Wikidata id).
* In the language slot: set `sources` to the new ids and remove `candidates` and `reason`.
* **A new wiki** needs its Template namespace in `TEMPLATE_NAMESPACE`
  (`src/frontier_ai/data/mediawiki.py`); the test
  `test_every_rendered_wiki_in_the_manifest_has_its_template_namespace_listed` fails until it
  is there. mr (`साचा`) is already listed.
* **Sentence ends for a new script:** `_SENTENCE_BOUNDARY` in `research_corpus.py` knows
  `।` `॥` U+09F7 `.` `!` `?`. Urdu uses `۔` (U+06D4) and `؟` (U+061F): add them, with a test,
  before an Urdu source is built.

## 8. Reviewing an inspection report

* Header line: `N sources, N verified, P pinned in the manifest, built <time>` — all declared
  sources verified; any that did not, and why (the report names the gate).
* Exit code of the build (0 expected) and any `REFUSED:` line (a pinned source changed:
  find out why before anything else).
* Per new source: levels as expected (§6), the first and last lines of text match where the
  range should start and end (the source's `notes`), and no CSS, running headers, printed
  page numbers, front matter or advertisements in the samples.
* Flags and repairs: every flag explained (for example ml's `1033` is a genuine year),
  repair counts plausible.
* Coverage section: every slot's documents and characters; `EVALUATED` needs both targets.
  An `INSUFFICIENT` slot gets another work, never padding.
* The runbook's §8 checklist for the group (for pa/kn/te/ta, "Second group of new
  languages") lists the specific expectations; tick them in the runbook when they hold.

## 9. Locking

* Lock only after a clean review, with `--pin` on a fresh fetch.
* Then `git diff --stat` must show only the manifest, and `git diff` only `sha256`,
  `verified` and `retrieved_at` for the newly locked sources plus `retrieved_at` for the ones
  pinned before; each new hash must begin with the prefix the previous fetch's inspection
  report printed for that source. If anything else changed, undo with
  `git checkout -- corpora/tokenizer/indic-tokenizer-v2/sources.json` and investigate.
* Commit the manifest with the run's EXP number, record the run (rule 10), push.

## 10. Next steps

1. **JOB-002** ([reports/JOB.md](../corpora/tokenizer/indic-tokenizer-v2/reports/JOB.md)):
   the tidy-up, the EXP-018 fetch and its report. Review it yourself (§8), including the
   gu/ml/or/as coverage that has never been confirmed.
2. If a slot is short: te is the likeliest of the new four. Telugu leads found on 2026-09-25,
   none checked yet: Chilakamarti Lakshmi Narasimham's *Hemalatha* (he died in 1946),
   Panuganti's *Sakshi* essays (1940), Gurajada's *Kanyasulkam* (1915, a Digital Library of
   India scan). Avoid the 1966 edition of *Ganapati* (it has a modern editor) and *Malapalli*
   (not proofread). For or, the wiki has only 21 fully proofread scans.
3. Lock (EXP-019) after the review.
4. **mr:** mostly modern (in-copyright) works in the fully proofread list; look for authors
   who died in 1965 or earlier (for example Hari Narayan Apte 1919, Ram Ganesh Gadkari 1919,
   Lokmanya Tilak 1920, Sane Guruji 1950, N. C. Kelkar 1947, Jotiba Phule 1890, Keshavsut
   1905; not Savarkar, who died in 1966).
5. **ur:** survey ur.wikisource (small: about 12 active editors); namespaces and page
   anchors unknown yet; add the Urdu sentence marks (§7).
6. **hi-en:** keep the reason for `NOT_EVALUATED` current (runbook §9).
7. **Finish:** the EXPERIMENTS.md entry (§1).

## 11. Pitfalls already met

* An exit code of 0 only means no pinned source changed; read the report for the rest.
* Wikis change: a proofreader's edit to a page in a locked range changes that source's text
  and the next fetch refuses it. That is the lock working; look at the page history before
  re-pinning deliberately.
* `?action=raw` is useless for scanned works (their chapter pages hold only `<pages>` tags);
  `Special:WantedTemplates` is cached and misleading.
* The API can return a transient HTTP 500 — retry once. Keep request URLs short (split long
  title lists).
* When writing tests: `str.splitlines()` keeps empty lines; run the whole suite with
  `-o addopts=""`.

## 12. Working with the operator

The operator (the repository owner) works in VS Code's PowerShell terminal at `E:\frontier-ai` and is not a command-line expert. Report progress briefly and in
plain words — a small table per language (book, how much text, locked or not) works well;
say "fingerprint" for hash and "lock" for pin. If the operator must type something, give one
exact command at a time, say what they should see, and what to send back. URLs cannot be
pasted into PowerShell (it tries to run them): ask them to open URLs in a browser.

If you get stuck, say so plainly. The operator may take this part back to the Arena agent;
because every step is committed, pushed and recorded (rule 10), anyone can continue from git.
