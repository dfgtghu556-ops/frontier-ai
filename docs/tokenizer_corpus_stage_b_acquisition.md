# Stage B acquisition runbook — Project 004

**This is a procedure, not a report.** It was written in an environment that cannot reach
any corpus host (every endpoint fails with `TLS/SSL … EOF`); the fetches themselves run on
a network-enabled machine. **Status (2026-09-25):** the first live fetch (EXP-008) verified
Alice and refused both Stage A Wikisource roots — `hi` was a localized `#REDIRECT`, `bn` a
Wikidata/SPARQL infocard — and showed that both works are *scanned-book* transcriptions
whose chapter pages contain no text of their own. They are now declared as rendered pages
(§3.1). The second live fetch (EXP-009) verified 37 of 38 sources: all 36 Godaan chapters
(764,202 characters — the `hi` slot is `EVALUATED`) and Gitanjali (65,174 characters — `bn`
is `INSUFFICIENT`, as expected); Alice was lost to a dropped connection
(`IncompleteRead`), which the fetcher now retries. Because Alice (144,599 characters) and
Gitanjali are each below the 200,000-character target, a second work was declared for
each slot after EXP-009 — Tagore's *Sadhana* (§2.4) and Sarat Chandra's *Devdas* (§2.5) —
bringing the manifest to 40 sources. The third live fetch (EXP-010) verified **all 40**
(Alice 144,599 + Sadhana 206,965; Gitanjali 65,174 + Devdas 152,768 characters): `en`, `hi`
and `bn` are all `EVALUATED`. All 38 sources fetched before came back byte-identical (37
match EXP-009, Alice matches EXP-008), so repeated fetches are deterministic. The
inspection report then flagged 20 verbatim residues of mistyped `{{gap}}` templates on 19
Godaan scan pages (`{{Gap{}`, `{{Gap}]`, `<gap>` …, all located on the wiki) — the cleaner
now removes them (§3.1) — plus one leftover piece of wiki markup in Gitanjali. The fourth
fetch (EXP-011) confirmed that fix: exactly those five chapters changed and the other 35
sources were byte-identical. Its report's new *where to look* section located the rest: 8
red links to misspelt templates that do not exist (`साँचा:GaP`, `GP`, `Gpa`, `GAP` in
chapters 3, 21 and 24) and Gitanjali's stray `}}` (scan page ১৪৮). The cleaner now removes
both kinds too (§3.1); by the wikis' own template lists, no other missing template occurs in
any of the 38 rendered sources. The fifth fetch (EXP-012) confirmed both: all 40 verified,
exactly chapters 3, 21 and 24 and Gitanjali changed (chapter 3 lost exactly the 9
characters of `साँचा:GaP`, Gitanjali the 2 of `}}`), the other 36 sources were
byte-identical to EXP-011, the report's repair counts match every located residue, and
nothing was flagged.
**Pinned (EXP-013): all 40 `en`/`hi`/`bn` sources.** The operator's `--pin` run wrote their
hashes (retrieved 2026-09-24T17:04–17:05Z); each begins with the prefix EXP-012's inspection
reported, and Alice's equals the hash of EXP-008/010/011. Every later fetch compares against
these pins and refuses changed text (§0.5); the first re-verification (EXP-014) exited 0, so
no pinned source had changed. **First group of new languages:** Gujarati, Malayalam, Odia
and Assamese, 7 range sources from four public-domain novels (§2.6); researching them exposed
a proofreading-gate gap that is now closed (§3.1: a render that hides page levels used to pass
unchecked). Their first fetch (EXP-015) verified all 47 sources (the 40 pinned ones unchanged)
and put all four new slots above target: gu 297,628, ml 399,168, or 203,451, as 241,025
characters. Its inspection found one more wiki typo, a `<poem>` tag printed as text in the
Odia novel, which the cleaner now removes (§3.1). The next fetch (EXP-016) confirmed the fix:
all 47 verified, only the Odia text changed (by exactly the 7 characters of the tag and a
space, to 203,444), and nothing is flagged except a genuine year in the Malayalam novel.
**Pinned (EXP-017): the seven new sources**, each beginning with the prefix EXP-016 reported;
the 40 pinned before came back identical (hashes unchanged; only their `retrieved_at` moved,
to this run's time), so all 47 sources are now pinned. **Second group of new languages:**
Punjabi, Kannada, Telugu and Tamil, 8 range sources from four public-domain works (§2.7),
declared on 2026-09-25 and not fetched yet. hi-en, mr and ur still have no source (§9).
Run the steps where the network works, and record what you actually observe.

**Marathi and Urdu research (EXP-020, 2026-09-25):** the first candidate survey did not
produce a declaration. Marathi's `आईबापांचा मित्र` scan is not broadly proofread in the
inspected range, and `श्री एकनाथी भागवत` did not expose a usable proofread range. The
modern `'भारता'साठी` work was rejected on authorship grounds. Urdu candidates including
`Tota Kahani`, `Ram Charcha in Urdu by Munshi Premchand`, and `Betal-pachcheesi` did not
yet meet the combined proofread, coverage, and underlying-work licence gates. The Urdu
wiki's CC BY-SA rights endpoint is evidence for wiki contributions only, not for the
public-domain status of a scanned work. No Marathi or Urdu source was declared, fetched,
verified, or pinned; both slots remain `NOT_EVALUATED`.

**Current status (2026-09-26):** EXP-021 declared and fetched one Marathi work and one
Urdu work in four ranges. Its first inspection exposed malformed Urdu heading/link markup,
a scan-inconsistent `1004` embedded in page 62's prose, and two U+200E direction marks, so
nothing was pinned. The tested cleaner repairs were re-fetched as EXP-022: all 59 sources
verified, all 55 prior pins unchanged, and the four new sources have no inspection flags.
Marathi is `EVALUATED` at 1,667 documents / 256,165 characters; Urdu is `EVALUATED` at
1,378 / 208,621. EXP-023 then verified and pinned all 59 sources; its four new fingerprints
match EXP-022, all previous fingerprints are unchanged, and its manifest diff contains only
the permitted lock fields. P004B is complete: 13 slots are `EVALUATED`; hi-en remains
`NOT_EVALUATED` because no lawful, attributable corpus was identified.

Who this is for: whoever acquires the real tokenizer research corpus. Stage A built the
foundation (`indic-tokenizer/v2`) — 14 language slots, 3 declared sources, split, leakage
diagnostics, coverage reporting — and proved the machinery works. Stage B is the part that
turns it from an empty, honest corpus into a real one.

Read [tokenizer_corpus_stage_a.md](tokenizer_corpus_stage_a.md) first: it defines the
statuses (`EVALUATED` / `INSUFFICIENT` / `UNVERIFIED` / `NOT_EVALUATED`), the split, the
licence rules and the current state of every slot.

Three rules before you start:

1. **Never pin a hash you did not compute from a fetch you actually performed.**
2. **Never mark a source verified because the manifest says it is CC BY-SA.**
3. **If a source cannot be licensed, leave it unverified.** An honest `NOT_EVALUATED` is a
   result; a fabricated one invalidates every later tokenizer number.

**In a hurry?** §0 is the whole workflow in five commands (setup, preflight, fetch,
inspect, pin). Everything after it is the detail.

**Who does this work now:** since 2026-09-25 the assistant on the operator's machine
(Claude Code) owns P004B end to end — research, licence review, code, runs, review and
locks: [tokenizer_corpus_handover.md](tokenizer_corpus_handover.md). Jobs are still written
down in `corpora/tokenizer/indic-tokenizer-v2/reports/JOB.md` and reports saved in the same
folder ([tokenizer_corpus_local_runner.md](tokenizer_corpus_local_runner.md)); hashes are
still written only by `--pin`, and only after a clean review.

---

## 0. Quick start (Termux, or any network-enabled machine)

This is the whole workflow: **preflight → fetch → inspect → pin → (later) re-verify**.
Nothing is acquired until step 2 and nothing is pinned until step 4. Every command below
exists in the repository today; none of them are invented.

### 0.0 One-time setup (once per machine)

```
pkg install python git          # Termux only
cd ~/frontier-ai
git checkout arena/01a0d31f-frontier-ai
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"         # torch + numpy + pytest + ruff
```

* On Termux the interpreter is `python`; on a desktop it is usually `python3`. Use whichever
  one works — they are the same program.
* `torch` is a big download, but it is not optional here: the corpus CLI imports the
  experiments package, which imports torch.
* If you leave the shell, re-run `. .venv/bin/activate` before the commands below.
* **Windows PowerShell:** activate with `.venv\Scripts\Activate.ps1`, and type each command
  below on **one line** — PowerShell does not understand the trailing `\` continuation (use a
  backtick `` ` `` if you must split a line). A URL typed at the prompt is run as a command,
  not opened: open URLs in a browser, or use the scripts. When a script can write a file, use
  its `--output` option rather than `>` (PowerShell 5.1 redirection writes UTF-16).

### 0.1 Step 1 — preflight: can I reach the sources? (read-only)

```
python scripts/build_tokenizer_corpus.py --preflight --timeout 20
```

**What it does, in plain language:** knocks on every source URL and every licence-evidence
URL, reads at most the first 65,536 bytes of each, and prints what came back. It saves
nothing to disk, verifies nothing, pins nothing and never touches the manifest.

**Exit codes:** `0` = every endpoint answered · `1` = at least one did not ·
`2` = the manifest or your arguments are wrong.

**If an endpoint fails:** you have no route to that host (or it is down). Do not substitute
a mirror or a different URL without recording why — the source stays `unverified` and its
slot stays `UNVERIFIED`. That is an honest result, not a failure to hide.

### 0.2 Step 2 — fetch: acquire and check everything except the manifest

```
python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
```

**What it does, in plain language:** downloads each declared source, strips the
Gutenberg/wikitext wrapper, fetches the declared licence-evidence endpoint, computes the
SHA-256 of what it kept, refuses the text if it looks like a contents page, refuses a
pinned source whose bytes changed, then writes the text, its provenance and five reports
under `--out`. **It does not edit the manifest** — that is step 4, on purpose.

**While it runs** it prints one line as it starts on each source, so you can see how much
is left, e.g. `[corpus] downloading 12 of 40: hi-wikisource-godaan-ch10-ccbysa`. The report
below appears only once every source is done, followed by `[corpus] this run took 3 min 20 s`
(both illustrations, not observed values). A `retrying …` line means a download dropped and
is being tried again (3 attempts, 30 s timeout each): normal, nothing to do. One source makes
at most two requests (its text, plus its licence page if no earlier source fetched it), so if
the same `downloading` line stays on screen for more than about 5 minutes something is stuck:
press Ctrl+C and run the same command again. That is safe: this step never edits the
manifest, and the rerun overwrites the failed record of the interrupted one.

**What you must read in the output** — the `per-source acquisition` block, one line per
declared source plus its detail line:

```
[corpus] per-source acquisition
[corpus]   en-gutenberg-alice-pd            verified             PD-US        reachable=yes content=yes proof=payload-marker   chars=   152,089
[corpus]       sha256=1a2b3c4d5e6f7a8b…  slot=en EVALUATED sufficient=yes
[corpus]   hi-wikisource-godaan-ch01-ccbysa verified             CC-BY-SA-4.0 reachable=yes content=yes proof=licence-evidence chars=    17,480
[corpus]       sha256=9f8e7d6c5b4a3f2e…  evidence=site ok marker=found  slot=hi INSUFFICIENT sufficient=no
[corpus]   xx-wikisource-example-ch07-ccbysa unproofread_refused CC-BY-SA-4.0 reachable=yes content=yes proof=licence-evidence chars=    12,003
[corpus]       sha256=fcd858968746beff…  evidence=site ok marker=found  slot=xx UNVERIFIED sufficient=no
[corpus]       reason: 2 of the 9 scan pages rendered by … have not been proofread on the wiki: …
```

* `reachable` — did the host answer? (`n/a` = no fetch was attempted for this source)
* `content` — did the run get text?
* `proof` — how the licence was proven: `payload-marker`, `licence-evidence`, or `none`
* `sha256` — the hash of the cleaned text this run produced (`none` = nothing usable)
* `slot=… <status> sufficient=…` — whether the language slot met its targets
* `reason:` — only printed when something failed, and it always says what to do next

The same information is in `data/tokenizer/indic-tokenizer-v2/acquisition.json`
(machine-readable, one row per source, timestamp-free so two identical runs produce a
byte-identical file). The lines above are an *illustration of the format*, not observed
values.

### 0.3 Step 3 — inspect before you trust it

```
python scripts/inspect_corpus_sources.py
cat data/tokenizer/indic-tokenizer-v2/acquisition.json
head -c 600 data/tokenizer/indic-tokenizer-v2/sources/en-gutenberg-alice-pd.txt
python -c "import json;d=json.load(open('data/tokenizer/indic-tokenizer-v2/coverage.json'));[print(r['language'], r['status'], r['reason']) for r in d['languages']]"
```

`inspect_corpus_sources.py` (read-only) prints one line per source — characters, lines, the
writing system of its letters, the scan pages and proofreading levels of a rendered source,
a hash prefix — and **flags** anything cleaning should have removed (HTML, wiki markup,
CSS, `&…;` entities, invisible characters, U+FFFD, long ASCII digit runs in a non-Latin
text), then — under **where to look** — prints up to five places per kind of problem for
each flagged source, each with the text around it (`«…»` marks the spot), plus up to three
places with letters from another writing system than the language's own (information
only: an English word may belong in a Hindi novel); a cut list says so (`3 of 11 places
shown`). Under **repairs by the cleaner** it lists, per source, what the
cleaner removed because it was typed wrongly on the wiki (from the provenance), so a fix can
be checked by its counts. It lists every language slot's coverage (status, documents,
characters, and the reason when a slot is not `EVALUATED`), and its first line says when the
build ran and how many sources the manifest has pinned. Finally it shows how the first and
last source of every language start and end. It exits `1` when anything is flagged. It ignores a text file left over from
an earlier run (its hash must be the one this run recorded).

**What it does, in plain language:** shows you the report, the first lines of the text that
was actually acquired, and the status of all 14 language slots. This is the step that
catches "the page I fetched is not the work". The first-fetch checklists in §8 are the
human part of the inspection — the code cannot tell a work from a different work.

### 0.4 Step 4 — pin (only what passed, and only in this run)

```
python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
git diff corpora/tokenizer/indic-tokenizer-v2/sources.json
```

**What it does, in plain language:** repeats the fetch and, for sources that passed every
gate *in that run*, writes `sha256`, `verified: true` and `retrieved_at` into the manifest.
The `git diff` should show nothing but those three fields, for those sources only. A source
that was already pinned and came back identical keeps its hash; only its `retrieved_at`
moves (§5.1). If nothing verified, the manifest is byte-identical and there is nothing to
commit.

### 0.5 Step 5 — later, re-verify (this is what a pin is for)

```
python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
```

Identical to step 2, except that now every pinned source is compared against its pin:
identical bytes → `verified` again and exit `0`; changed bytes → `hash_mismatch`, exit `1`,
the text is not used and the old pin is kept. Nothing is silently re-pinned, ever.

---

Everything from §1 onwards is the detail behind those five commands.

## 1. Network preflight

Preflight answers "can I reach this endpoint, and does the payload look like what the
manifest claims?" It is **read-only**: it stores no text, computes no corpus hash,
verifies nothing, writes nothing into the manifest and creates no output directory.

```
python scripts/build_tokenizer_corpus.py --preflight
python scripts/build_tokenizer_corpus.py --preflight --print-json      # machine-readable
python scripts/build_tokenizer_corpus.py --preflight --timeout 10      # shorter timeout
python scripts/build_tokenizer_corpus.py --preflight --source hi-wikisource-godaan-ch01-ccbysa
```

Per endpoint it reads **at most the first 65,536 bytes** (`--sample-bytes` to change) using
a byte-range request, and reports:

| Field | Meaning |
|---|---|
| `ok` + `http_status` | the endpoint answered |
| `bytes_read`, `content_type` | how much came back and what the server says it is |
| `sample_lines`, `wiki_links`, `link_line_ratio` | shape of the sampled text |
| `looks_like_index_page` | **warning**: sampled lines are mostly wiki links — probably a contents page, not the work. Preflight only warns; a real `--fetch` refuses it (§3) |
| `gutenberg_marker_seen` | the sampled bytes contain `*** START OF THE PROJECT GUTENBERG EBOOK` |
| `payload_licence_marker_seen` | a licence marker occurs in the sampled bytes (indicative only) |
| `evidence.*` | the declared licence-evidence endpoint: reachability, `marker_seen_in_sample`, `scope` |

**Exit codes:** `0` = every endpoint reachable · `1` = at least one unreachable ·
`2` = bad input (invalid manifest, unknown `--source` id).

**What the results mean:**

* **Reachable** — the host answered. That is *connectivity*, nothing more: the payload may
  still be an error page, a redirect notice, the wrong work, or a contents page.
* **Unreachable** — you cannot acquire this source here. Do not work around it: no mirror,
  no alternate URL silently substituted, no hash invented. Record it and move on; the slot
  stays `UNVERIFIED`.
* **`marker_seen_in_sample: true` on an evidence endpoint** — promising, but preflight is
  not verification. Verification happens only during a real `--fetch` build, which records
  the full result in `corpus.json` and the provenance file.

There is **no** `--list-sources` flag. To list what is declared, read the manifest:

```
python3 -c "import json;d=json.load(open('corpora/tokenizer/indic-tokenizer-v2/sources.json'));\
[print(s['id'], s['language'], s['license_id'], s['verified'], s['sha256'], \
 'evidence' if 'license_evidence' in s else '-') for s in d['sources']]"
```

---

## 2. Per-source acquisition procedure

The order is fixed and matters: **fetch → inspect → licence verification → content-quality
check → hash → pin**. In practice you do it in two runs: first without `--pin`, inspect the
artifacts, then — only if you are satisfied — repeat with `--pin`.

Step A — acquire into the build directory (no pinning yet):

```
python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
```

This writes, under `--out`:

| File | What it tells you |
|---|---|
| `sources/<id>.txt` | the cleaned text actually ingested |
| `sources/<id>.provenance.json` | title, URL, licence, attribution, `truncated`, `chars_before_trim`, `licence_proof`, `license_evidence`, `hash_is_licence_proof` |
| `stats.json` | per-language examples/chars/bytes, `unique_texts`, `duplicate_documents`, `truncated_sources` |
| `coverage.json` | status + reason for all 14 slots |
| `leakage.json` | exact and n-gram overlap (vacuous while the corpus is empty) |
| `corpus.json` | per-source ingest status, error text, evidence result, file hashes |

Step B — inspect (details per source below). **Do not skip this.**

Step C — if and only if the source verified **and** the content is right, pin:

```
python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
```

`--pin` writes `sha256`, `verified: true` and `retrieved_at` **only for sources that
verified in that run**, and leaves the manifest byte-identical when nothing verified.

### 2.0 The gates the code enforces (nothing to switch on)

A `--fetch` run applies these automatically. Each one is a refusal, never a warning you
have to remember, and each refusal is recorded in `acquisition.json`, `corpus.json`, the
provenance file and the `reason:` line of the report.

| Gate | Refuses when | Status shown | Can it be bypassed? |
|---|---|---|---|
| **Licence** | no licence marker in the payload **and** the declared evidence endpoint does not confirm it | `licence_marker_missing` | No. A failed evidence fetch is a failure, not a fallback to trust |
| **Redirect / infocard** | a `wikitext` page whose first line is a `#REDIRECT` in any language (`#पुनर्प्रेषित`…), a Wikidata/SPARQL infocard, or a rendered redirect page | `index_page_refused` | No — name the page that holds the work |
| **Content shape** | ≥5 lines, of which at least half are link lines with <20 characters of prose left (wiki links, or links in a rendered page) | `index_page_refused` | No — acquire the chapter subpages (§3) or use `--local-file` (§6) |
| **Tiny stub** | the cleaned text is shorter than 500 characters (for `wikitext`: only when the raw page still carries markup, so a short poem passes) | `index_page_refused` | No |
| **Proofreading** (`mediawiki-parse`) | any rendered scan page is at ProofreadPage level 1 (not proofread) or 2 (problematic); or a page's level can be confirmed neither from the render nor through the wiki's API; or ProofreadPage content has no page anchor at all | `unproofread_refused` | No — wait for the wiki's proofreaders, or declare a range without those pages (§3.1). A level nobody could read is never taken as "proofread" |
| **Pinned hash** | the manifest pins a `sha256` and this fetch cleans to different bytes | `hash_mismatch` | Only by clearing `sha256`/`verified` in the manifest first, with a note saying why (§5) |
| **Empty text** | the cleaned text is empty | `empty` | No |
| **Fetch** | the host could not be reached, the response was not UTF-8, or (`mediawiki-parse`) the API returned an error or incomplete JSON. A connection dropped mid-body and HTTP 429/5xx are retried (3 attempts in total); timeouts, DNS and TLS failures are not — they mean no route | `fetch_failed` | No |

A refused source is **not** verified, is **not** part of the train/held-out split and is
**not** pinned. Its text file is still written under `sources/<id>.txt` so you can look at
what you got and decide what to do next (except for `fetch_failed`, where nothing usable
arrived). Its provenance says `verification: unverified` — even when its licence *was*
proven, which `licence_proof` still records — and records `content_check` (what the gates
saw) and `hash_is_licence_proof: false`.

What the code **cannot** check, and you must:

* that the text is *this* work and not a different one (read the first lines — §0.3);
* that the licence tag on the Wikisource *page* matches the site-level evidence (§2.2);
* that the volume is right for the corpus role (`coverage.json` says `INSUFFICIENT` with
  the real numbers, and never pads).

### 2.1 `en-gutenberg-alice-pd` (English, PD-US, kind `gutenberg`)

* **Fetch** — `https://www.gutenberg.org/cache/epub/11/pg11.txt`. Project Gutenberg `.txt`
  files carry their own licence header, so the payload marker check is expected to pass
  (`licence_proof: payload-marker`). No evidence endpoint is declared, and none is needed.
* **Inspect** — `clean_gutenberg_text` strips everything outside
  `*** START/END OF THE PROJECT GUTENBERG EBOOK … ***`. In `sources/en-gutenberg-alice-pd.txt`
  confirm: it starts with the book's own front matter, which sits inside the markers —
  `[Illustration]`, the title, `by Lewis Carroll`, `THE MILLENNIUM FULCRUM EDITION 3.0` and the
  contents list — then `CHAPTER I.` and "Alice was beginning to get very tired…" (checked in
  pg11.txt on 2026-09-24). No Gutenberg header or licence text ("This eBook is for the use of
  anyone anywhere…", "START: FULL LICENSE") may be left, and the text is English prose.
* **Volume** — Alice alone is below the 200,000-character target (EXP-008 kept 144,599
  characters), so the English slot also needs *Sadhana* (§2.4). `max_chars` is 400,000; if
  `truncated: true` appears, the source was cut and you should say so in the record rather
  than quietly accepting it.
* **Licence** — PD-US, with the attribution already written in the manifest. If the marker
  is missing, the file is not a Gutenberg text: stop, do not pin.
* **Hash + pin** — only after the above.

### 2.2 Godaan — `hi-wikisource-godaan-ch01-ccbysa` … `ch36` (Hindi, kind `mediawiki-parse`)

* **What changed and why.** Stage A declared `hi.wikisource.org/wiki/गोदान?action=raw`. The
  first live fetch returned one line, `#पुनर्प्रेषित [[गो-दान]]` — a Hindi-localized redirect —
  and the redirect target is no better: every chapter page `गो-दान/१ … /३६` holds only a
  ProofreadPage tag (`<pages index="गो-दान.djvu" from=… to=… />`). The prose lives on the
  scanned pages and exists only once MediaWiki renders the tag. So Godaan is now **36
  sources, one per chapter, each asking MediaWiki to render its page** (§3.1).
* **Fetch** — `https://hi.wikisource.org/w/api.php?action=parse&format=json&formatversion=2&prop=text|revid|categories&…&page=गो-दान/N`
  (percent-encoded in the manifest). The response is JSON; the cleaner keeps the rendered
  text and drops the header, page anchors and hidden metadata.
* **Deliberately not declared:** `गो-दान/ भाग 16` and `गो-दान/ भाग 17`. They render a
  *different* scan (`गोदान.pdf`, an older transcription) and would put the same chapters
  into the corpus twice — a train/held-out leak. `scripts/discover_wiki_chapters.py` flags
  exactly this case.
* **Inspect** — `sources/hi-wikisource-godaan-ch01-ccbysa.txt` must start with the novel:
  "होरीराम ने दोनों बैलों को सानी-पानी देकर…". Later chapters normally open with their number
  on a line of its own (`२`, `४` were checked). The provenance `content_check.page_quality` shows how many
  scan pages were rendered, their proofreading levels and the `first_page`/`last_page` —
  compare them with the range in the source notes. Chapters **23 and 24** carry the wiki's
  "missing scan pages" category (`छूटे स्कैन पृष्ठ`): read them and record what you find.
* **Volume** — the whole novel is far above the 200,000-character target; this is expected
  and fine (targets are minimums; balancing is a later stage).
* **Licence** — evidence endpoint
  `https://hi.wikisource.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json`,
  marker `https://creativecommons.org/licenses/by-sa/4.0/`, `scope: site`. It is fetched once
  per build and reused for the other chapters (`reused_within_run: true`). The work page
  `गो-दान` itself carries `{{PD-India}}`: record that page-level review in the notes before
  trusting the slot for redistribution.
* **Hash + pin** — only after content and licence checks, per chapter.

### 2.3 Gitanjali — `bn-wikisource-gitanjali-1913-ccbysa` (Bengali, kind `mediawiki-parse`)

* **What changed and why.** Stage A declared `bn.wikisource.org/wiki/গীতাঞ্জলি?action=raw`,
  which is a Wikidata/SPARQL infocard, not the work. The 1913 edition lives at
  `গীতাঞ্জলি (১৯১৩)`: a table of contents whose 157 poems are subpages, each rendering one or
  two scan pages. A single poem is far below the 500-character stub threshold, so the
  source renders the **whole poem range in one request** — the same `<pages>` tag the
  subpages use, from scan page 13 (poem ১) to 190 (poem ১৫৭). The URL's `text=` parameter is
  validated to contain exactly that one tag and nothing else.
* **Inspect** — the file must start with poem ১, "আমার মাথা নত করে দাও হে তোমার", one verse
  per line, poems separated by their number and the composition date (e.g. `১৩১৩`). No CSS
  (`.mw-parser-output …`) may appear — if it does, the cleaner missed a `<style>` block:
  stop and report it. `content_check.page_quality` should show 178 pages at level 4.
* **Licence** — same mechanism as Godaan, on `bn.wikisource.org`. Tagore died in 1941, so
  the work is public domain in India; record what the edition's page says. If it is PD, the
  `license_id` may be relaxed to `PD-US` with a note and a fresh verification run (the
  licence marker list in `src/frontier_ai/data/corpora.py` only knows the four allowed ids —
  do not add a new one without a decision record).
* **Volume** — Gitanjali is a short book of verse (65,174 characters in EXP-009), well under
  the 200,000-character target. The second Bengali work, *Devdas* (§2.5), is declared to
  close the gap; if it fails, the Bengali slot stays `INSUFFICIENT`. That is the honest
  result.

### 2.4 Sadhana — `en-gutenberg-sadhana-pd` (English, PD-US, kind `gutenberg`)

* **Why this book.** Alice alone is below target. *Sadhana: The Realisation of Life*
  (Rabindranath Tagore, 1913; the Gutenberg text follows the 1916 printing) adds a different
  genre — philosophical essays, non-fiction — and Indian English: Upanishadic terms, Indian
  names and romanised Sanskrit with diacritics (`Sādhanā`, `Daurbhikshāt …`), which an Indic
  tokenizer will meet in real English text.
* **Fetch** — `https://www.gutenberg.org/cache/epub/6842/pg6842.txt` (ebook 6842, checked on
  2026-09-24: "Public domain in the USA", plain text 226 kB). Same mechanism as Alice:
  `licence_proof: payload-marker`, no evidence endpoint.
* **Inspect** — the cleaned text starts with the producer's credit line, which sits inside
  the markers and is kept ("Produced by Chetan Jain at BharatLiterature", 43 characters),
  then the title page (`SĀDHANĀ`, `THE REALISATION OF LIFE`, `By`, `Rabindranath Tagore`, …,
  `1916`, `To`, `Ernest Rhys`), the Author's Preface ("Perhaps it is well for me to
  explain…"), and ends with chapter VIII ("…not distant, not anywhere else."). No Gutenberg
  licence text may be left.
* **Volume** — about 200,000 characters expected; with Alice the English slot should pass
  the target. `truncated: false` expected (it is far below `max_chars`).

### 2.5 Devdas — `bn-wikisource-devdas-ccbysa` (Bengali, kind `mediawiki-parse`)

* **Why this work.** Gitanjali is below target. *দেবদাস* (Devdas, Sarat Chandra
  Chattopadhyay, 1917) is prose by a different author, with literary narration and colloquial
  dialogue. Sarat Chandra died in 1938; the work page carries `{{PD-India}}`. The bare title
  `দেবদাস` is a disambiguation page; the other scan it lists
  (`দেবদাস - প্রচার পুস্তিকা (১৯৩৫).pdf`) is a 1935 film booklet, not the novel.
* **Fetch** — one range render, like Gitanjali: scan pages **5–110** of
  `দেবদাস - শরৎচন্দ্র চট্টোপাধ্যায়.pdf` under the work page
  `দেবদাস (শরৎচন্দ্র চট্টোপাধ্যায়)`. The 16 chapter subpages transclude exactly this range,
  split with `<section>` markers where two chapters share a page (chapter 1 = `from=5 to=12
  tosection=১`, chapter 16 = `from=100 fromsection=১৬ to=110`); rendering the range without
  section limits yields every section of every page once, in order. Pages 1–4 (cover, title,
  blank, a list of the author's other books) are excluded.
* **Inspect** — checked live on 2026-09-24: the render starts with the chapter number `এক`
  and the first line "একদিন বৈশাখের দ্বিপ্রহরে রৌদ্রেরও অন্ত ছিল না…", and ends with the last
  paragraph ("…দেখিয়া সে মরিতে পারে।") followed by `সমাপ্ত`. No CSS may remain.
  `content_check.page_quality` should show 106 pages, first `…pdf/৫`, last `…pdf/১১০`,
  levels 3 (pages 8–108) and 4 (5–7, 109–110) only.
* **Licence** — same mechanism as Gitanjali (site-level `rightsinfo` on `bn.wikisource.org`).
* **Volume** — about 150,000–185,000 characters estimated from the scan (≈1,700 per page);
  with Gitanjali the Bengali slot should pass the target. Report the real number.

### 2.6 First group of new languages — Gujarati, Malayalam, Odia, Assamese (kind `mediawiki-parse`)

Found by surveying each wiki's `Special:IndexPages` filtered to *proofread or validated* on
2026-09-24 (gu 204 fully proofread scans, ml 214, or 21, as 400; also mr 156, ta 500+, te 333,
kn 60, pa 198 for the next groups). All four works are novels that are public domain in India
(life + 60 years), rendered as ranges like Devdas; every source's `notes` has the details.

| Slot | Work (author, died) | Sources: scan pages | Estimate |
|---|---|---|---|
| gu | *સરસ્વતીચંદ્ર* part 1 (Govardhanram Tripathi, 1907) | `…-ch01-08`: 21–106 · `…-ch09-12`: 107–188 | ≈240–270k chars |
| ml | *രാമരാജാബഹദൂർ* (C. V. Raman Pillai, 1922) | `…-ch01-13`: 4–150 · `…-ch14-18`: 151–207 | ≈230–265k |
| or | *ଛମାଣ ଆଠଗୁଣ୍ଠ* (Fakir Mohan Senapati, 1918) | the whole novel: 4–162 | ≈190–215k — may fall just short |
| as | *মনোমতী* (Rajanikanta Bordoloi, 1940) | part 1: 5–131 · part 2: 133–267 | ≈220–255k |

* **Checked before declaring** (live, 2026-09-24): chapter boundaries from each work's
  chapter subpages; no scan page at level 1 or 2 in any of the four scans (the wikis' own
  quality categories), six sampled pages per book at level 3 or 4; missing templates via
  `prop=templates` (none in gu, ml chapters 1–18, or, and as part 1 chapters 1–15 — the rest
  of *Manomati* was not pre-checked; the cleaner removes and reports any red link); the
  Template namespaces are in `TEMPLATE_NAMESPACE`.
* **Licence review** (§4, human): gu work page `{{ઢાંચો:પ્રકાશન-ભારત}}` (gu's PD-India notice),
  scan 1887 `{{PD-1923}}`; ml scan file page (local) `{{PD-India}}`; or scan `{{cc-zero}}`,
  1903 edition, no tag on the work page; as work page `{{PD-India}}`, scan `{{PD-old-auto|1940}}`.
  *Manomati*'s scan is a posthumous 6th edition (1956), so only the chapters are used — its
  opening note, closing সামৰণি, notes and notice are excluded (authorship not established).
* **Why split into ranges.** One render per source stays at or below ~150 scan pages;
  the largest render used so far is Gitanjali's 178. Chapter-aligned where the wiki allows:
  *Saraswatichandra*'s chapters share pages (sections), and from page 106 on no chapter ends at
  the foot of a page, so the second gu range ends with the opening section of chapter 13.
* **Inspect:** gu and or sources will show `level_lookup` (the build asked the API: expect
  `confirmed` = pages and `error: null`); ml's anchors must leave no `[ n ]` page numbers in
  the text; every source should start with its first chapter's heading.
* **First fetch (EXP-015):** all 7 verified. gu 152,483 + 145,145; ml 288,952 + 110,216 (the
  sampled page sizes had underestimated Malayalam); or 203,451; as 114,802 + 126,223 characters.
  Levels: gu 168 and or 159 pages validated (confirmed through the API), ml 144 + 57 proofread
  and 3 validated, as 262 validated. Every source starts with its first chapter and ends where
  its range ends — the second gu range, as planned, in the middle of chapter 13's first page
  (`…સૂર્ય તનમનને`). Flags: or's literal `<poem>` (now removed: −7 characters, the tag and one
  space) and ml's `1033` (a Malayalam-era year, genuine). Information only: Sanskrit verse in
  the Gujarati novel, `--o--` ornaments between Odia chapters, an English gloss in Manomati.
  Manomati part 1 types U+09F7 for the danda, so the document splitter now splits there too
  and never cuts a sentence mid-word (document boundaries only; no hash changes).
* **Second fetch (EXP-016):** all 47 verified, exit 0. The 40 pinned sources and six of the
  new ones came back byte-identical (every prefix equals its pin or EXP-015's). Only or
  changed: 203,451 → 203,444 characters (prefix `6a0339ff3ea5` → `83850852f8f4`), its repairs
  line reads `literal <poem> tag -> removed (1)` and it is no longer flagged; the only flag
  left is ml's `1033`. Identical documents in as went from 44 to 46 (the new document
  splitting at U+09F7 and between words; no hash changed). Hash prefixes the pin (EXP-017)
  is expected to write: gu `c3fe2b7aa706`, `91e688351017`; ml `beb122a5db78`,
  `db88c28a02b9`; or `83850852f8f4`; as `9876b5c9940e`, `eb4b86b65401`.
* **Pinned (EXP-017):** exit 0. The operator's `--pin` run (retrieved 2026-09-24T18:59:41Z to
  19:00:14Z) wrote the seven hashes, each beginning with its EXP-016 prefix above; the 40
  earlier pins came back identical and only their `retrieved_at` moved. The pinned file was
  pushed from the operator's machine (`283eb6f`) and carried here with git, never retyped.

### 2.7 Second group of new languages — Punjabi, Kannada, Telugu, Tamil (kind `mediawiki-parse`)

Found on 2026-09-25 from the same survey (§2.6) and each wiki's own search. All four works
are public domain in India (life + 60 years) and rendered as ranges; every source's `notes`
has the details.

| Slot | Work (author, died) | Sources: scan pages | Estimate |
|---|---|---|---|
| pa | *ਸਤਵੰਤ ਕੌਰ* (Bhai Vir Singh, 1957) | `…-p007-164`: 7–164 · `…-p165-336`: 165–336 | ≈300–360k chars |
| kn | *ರಂಗಣ್ಣನ ಕನಸಿನ ದಿನಗಳು* (M. R. Srinivasamurthy, 1953) | `…-ch01-15`: 16–178 · `…-ch16-30`: 179–343 | ≈330–390k |
| te | *రాజశేఖర చరిత్రము* (Kandukuri Veeresalingam, 1919) | `…-ch01-07`: 15–104 · `…-ch08-15`: 105–217 | ≈220–260k — the tightest |
| ta | *என் சரித்திரம்* (U. V. Swaminatha Iyer, 1942), chapters 1–27 | `…-p022-101`: 22–101 · `…-p102-189`: 102–189 | ≈250–300k |

The estimates come from sampled page sizes (pa ≈2.9 KB of wikitext per page, kn ≈3.5,
te ≈3.4, ta ≈4.8) at about 2.7 bytes per character; ml showed such estimates can be off by
a fifth or more.

* **Checked before declaring** (live, 2026-09-25): every page of each scan exists; the wikis'
  own quality categories find no scan page at level 1 or 2 in any of the four (pa 336
  validated and 8 empty, kn 339 validated and 10 empty, te 216 of 220 validated, ta 803 of
  804 proofread at level 3, which the gate accepts); missing templates via
  `generator=templates` — none (kn for the whole book through its work page, pa on ten
  sampled pages, te chapters 1, 8 and 15, ta chapters 1, 14 and 27). The four Template
  namespaces are in `TEMPLATE_NAMESPACE` (and mr's, for later). All four wikis print
  `data-page-name` and `data-page-quality` in their page anchors, as hi, bn and as do, so
  the build reads the levels from the render: no `level_lookup` is expected.
  Search pitfall met on the way: CirrusSearch's `prefix:` takes the rest of the query, so it
  must come last (`incategory:X prefix:Page:…/`); put first, it silently finds nothing.
* **Licence review** (§4, human): pa scan (local file) `{{PD-India}}`, a 1973 printing of a
  novel published in 1900 and 1927, in the author's lifetime; the publisher's 1973 title page,
  imprint and advertisements (after `-ਇਤਿ-` on page 337, and 338–340) are excluded, so the
  second range stops at 336 and the novel's last paragraph is left out. kn scan (Commons)
  `{{PD-scan}}`, the 1951 edition, published in the author's lifetime; no tag on the work
  page. te work page `{{PD-old}}`, scan `{{PD-India}}`; the 1987 reprint's front matter
  (1–14) is excluded. ta work page `{{PD-India/ta}}`, scan `{{PD-India}}` (its "published
  1887" permission line is wrong and not relied on); the 1990 edition's front matter (1–21)
  is excluded, and footnotes never reach the text (the cleaner drops reference markup).
  Death dates (Wikidata): 1957-06-10, 1953-11-16 (the kn author page says 16-9-1953; the year
  agrees), 1919-05-27, 1942-04-28. Tamil Wikisource's many *nationalised* books (rights
  bought by the state) were passed over: that is a policy, not a licence; *En Charithram* is
  public domain by term.
* **Range sizes:** te and ta stay at or below 113 pages per render; pa and kn use renders of
  158–172 pages (below Gitanjali's 178) because their pages are small (≈3 KB of wikitext).
* **Where ranges meet:** te and kn split where a chapter starts on a new page; pa at page
  165, which carries its 18th chapter heading (perhaps mid-page); ta's chapters begin
  mid-page (sections), so its two ranges meet inside chapter 14. Every page is in exactly one
  range: nothing is lost or repeated.
* **First fetch and lock (EXP-018/019, JOB-002):** completed. EXP-018 verified all 55
  sources and put pa, kn, te and ta above target; EXP-019 locked the eight new sources after
  inspection. See the status header and experiment log for the recorded results.

### 2.8 Marathi and Urdu (kind `mediawiki-parse`)

**EXP-021 candidate review and declaration (2026-09-25):** after searching the named Marathi
authors, Hari Narayan Apte's *स्फुट गोष्टी भाग तिसरा* was the qualifying Marathi scan. Its
1928 Pune edition is identified in the Index and work-page metadata; Commons tags the scan
Public domain, Apte died in 1919 (Wikidata Q55687), and pages 13–124 are all level 3.
Pages 1–12 (title/front matter, contents and an unattributed publisher biography) and page
125 (level 0) are excluded.

For Urdu, *رام چرچا* (*Ram Charcha*) by Munshi Premchand uses the Lahore 1929 edition. The
scan's title page names Premchand and the 1929 publishers; Commons tags the scan Public
domain, and Premchand died in 1936 (Wikidata Q174152), during his lifetime of publication.
The three declared ranges are pages 7–170, 171–307 and 309–342. Pages 1–6 (title, preface
and contents), page 308 (level 1), and page 343 onward (back matter) are excluded. The Urdu
wiki's CC BY-SA rights endpoint covers its transcription, separately from the scan's
public-domain evidence.

EXP-021 fetched all 59 sources (build exit 0; the 55 pinned sources were unchanged).
Marathi produced 257,901 source characters; Urdu produced 210,517 across the three ranges.
The inspection flagged two raw markup residues in Urdu page 46, an ASCII `1004` embedded in
page 62's sentence despite being absent from the scan image, and U+200E bidi controls in
the first two ranges. Review also found one unmatched `[[` in the second range. No new
source was locked at this stage.

The cleaner now preserves the Urdu heading and prose while removing only the malformed
`{{|xx-larger...}}` wrapper and the unpaired link delimiter; it removes the page-62 `1004`
only in its observed surrounding phrase and strips U+200E as a direction-formatting
artifact. EXP-022 re-fetched all 59 sources with exit 0; all 55 existing pins were unchanged.
The four new sources have no inspection flags. Marathi has 1,667 documents / 256,165
characters; Urdu has 1,378 / 208,621. The EXP-022 report records source samples, repair
counts and the new hash prefixes.

EXP-023 was a separate fresh `--fetch --pin` run (exit 0): all 59 sources were verified and
pinned. The 55 previous fingerprints stayed identical, and the four new ones match EXP-022:
mr `e1e7dba1bfe5`; ur `fb22db9983ef`, `36cd31410d2b`, `4835213b51c2`. Its inspection report
has no flags for the four new sources. The manifest diff changed only `sha256`, `verified`
and `retrieved_at` for the four new sources, and only `retrieved_at` for the 55 existing
pins. All 14 slots are now accounted for: 13 `EVALUATED`, hi-en `NOT_EVALUATED` with no
licensed, attributable source found; no text was padded or substituted.

---

## 3. Wikisource-specific safety

**A Wikisource work-root URL may contain only navigation, a header and chapter links.**
Accepting it blindly would give you a "verified" source with a few hundred characters and
an `INSUFFICIENT` slot — or worse, a slot that looks fine because the character target was
somehow met by boilerplate.

**The code now refuses it for you.** If the fetched payload is `wikitext` and a majority of
its lines are wiki links carrying almost no prose, the source comes back
`index_page_refused`: not verified, not in the corpus, not pinned, with a `reason:` line
telling you how many lines looked like links. You do not have to remember to check. Short
lines of *verse* are prose, not navigation, so Gitanjali's Bengali poetry does not trip the
gate — only real link/template lines do.

Check, in this order:

1. **Preflight** — `looks_like_index_page: true`, a high `link_line_ratio`, or a large
   `wiki_links` count with few `sample_lines` means: probably a contents page.
2. **The ingested file** — `sources/<id>.txt` must be prose in the expected script, not a
   list of links. `wc -l` and a quick `head` are enough.
3. **The volume** — `stats.json` per-language `chars` and `examples` against the targets
   (500 documents / 200,000 characters).

If the root is an index page, the supported options are:

* **One manifest source per chapter** — the CLI ingests one URL per declared source, so N
  chapters means N sources (each with its own URL, licence id and evidence). Verbose, but
  fully supported today: every chapter is individually fetched, cleaned, hashed and
  attributed. Remember the manifest rules: every slot reference must exist, no duplicate
  ids, no orphan sources (§6).
* **Concatenate locally** — fetch the chapters yourself, concatenate them, and ingest with
  `--local-file` (§7). Honest and quick, but the source stays `local_unverified`, so the
  slot reports `UNVERIFIED` no matter how much text you supply.
* **Not implemented (future work)** — a manifest field listing several URLs whose cleaned
  texts are concatenated into one source. For scanned books it is not needed: a single
  `<pages>` render covers any page range (§3.1).

Do **not** change a source's `source_url` to "whatever happens to work" without updating
the attribution, the notes and the evidence endpoint to match.

### 3.1 Scanned books (ProofreadPage) — the `mediawiki-parse` kind

**How to recognise one.** Most Indic Wikisource works are transcriptions of scanned books.
Their chapter page's `?action=raw` is ~150 characters: a single tag such as
`<pages index="गो-दान.djvu" from=18 fromsection="1" to=२२ tosection="1" />`. The text is on
one `Page:` page per scan (`पृष्ठ:` in Hindi, `পাতা:` in Bengali) and only exists as a whole
after MediaWiki expands the tag — joining pages, splitting shared pages by section, reading
Devanagari digits (and even typos such as `from=1१`, which it reads as 11). Re-implementing
that would be a second, divergent copy of ProofreadPage, so we don't: a `mediawiki-parse`
source asks MediaWiki for the **rendered** page (`api.php?action=parse`, JSON) and the
cleaner turns the HTML into text (`src/frontier_ai/data/mediawiki.py`).

**Find the chapters** (read-only; writes nothing unless you ask it to):

```
python scripts/discover_wiki_chapters.py --lang hi --work "गो-दान"
python scripts/discover_wiki_chapters.py --lang mr --work "<work title>" --emit-sources mr-wikisource-<work> --output proposed.json
```

It lists the chapter subpages in numeric order (any script's digits), shows the scan and
page range each one renders, and flags subpages that render a **different scan** (legacy or
duplicate transcriptions — never declare them blindly) or no scan at all. `--emit-sources`
proposes unverified entries (`sha256: null`, `verified: false`) for review; nothing goes into
the manifest until a human puts it there.

**The two URL forms** (build them with `parse_page_url` / `parse_pages_range_url` from
`frontier_ai.data.mediawiki`; the manifest validator enforces the rules):

| Form | Use it for | Rule |
|---|---|---|
| `…api.php?action=parse&format=json&formatversion=2&prop=text\|revid\|categories&…&page=<title>` | one chapter page | `redirects` is forbidden: a redirect is refused, never followed |
| `…api.php?action=parse&…&contentmodel=wikitext&title=<work>&text=<pages index="…" from=N to=M />` | a range of scan pages (short poems) | `text=` must be exactly one `<pages>` tag — nothing but wiki content can enter |

**What the cleaner keeps and drops** (every rule is tested):

* drops everything Wikisource marks `ws-noexport` (the header ← previous · title · author ·
  next →, the hidden `ws-data` block, the inner page anchors), page-number anchors,
  `<style>`/`<script>` (TemplateStyles CSS is emitted inline!), footnote markers and lists,
  and `display:none` elements;
* keeps all other text: `<br>` = line break, blocks (`p`, `div`, headings, …) = paragraph
  breaks — so verse keeps one line per verse line and prose one line per paragraph;
* a `<br>` immediately after a page anchor, outside a `<poem>` block, becomes a space: many
  transcriptions start every page with `<br>` although the printed sentence continues;
* removes three invisible rendering artifacts — U+200B, U+2060 (from `{{gap}}`), U+FEFF (left
  by OCR imports) — and turns U+00A0 into a space; **keeps U+200C/U+200D** (ZWNJ/ZWJ), which
  decide how Indic conjuncts are written. No NFC/NFD/NFKC. All counts are in
  `content_check.artifacts_removed`;
* removes the verbatim residue of a **mistyped `{{gap}}`**: MediaWiki prints a template call
  it cannot parse as plain text, so `{{Gap{}`, `{{Gap}]`, `{{Gap]}`, `{Gap}}`, `{{Gap))`,
  `{{Gap@))`, `{{Gap` running into the text, or `<gap>` typed as a tag would otherwise
  reach the corpus (20 of them in Godaan, found by the inspection after EXP-010). A correct
  `{{gap}}` renders as a U+2060 spacer that is removed anyway, so the cleaned text is the
  same whether or not the typo is ever fixed on the wiki. Only a brace or `<` directly
  before the word triggers it; counted as `broken {{gap}} template -> removed`;
* drops a **red link to a missing template**. A misspelt template name (`{{GaP}}` for
  `{{Gap}}`) makes MediaWiki print a red link to the template that does not exist —
  `साँचा:GaP` in the middle of the text. Only links marked `redlink=1` whose target lies in
  the Template namespace are dropped (`साँचा:` on hi, `টেমপ্লেট:` on bn, `Template:` on every
  wiki — `TEMPLATE_NAMESPACE` in `mediawiki.py`); a red link to a missing article or author
  page keeps its text. Counted per template as `link to missing template <title> -> removed`.
  **Declaring a source from a new wiki? Add its Template namespace name there first** — a
  test fails until you do. A misspelt template that should have *shown* text (say
  `{{smallcaps|word}}`) loses that text on the wiki itself, so the provenance names every
  missing template; to list them for a source before fetching, open its `source_url` with
  `prop=templates` instead of `prop=text…` and look for `"exists": false`;
* removes a **`}}` that closes nothing** — on a line with no `{` and no `|`, where it carries
  nothing (Gitanjali's scan page ১৪৮ ends a poem with `২৬ আষাঢ় ১৩১৭}}`); counted as
  `stray }} -> removed`. A line with an opening brace or a `|` may be a broken template call
  and keeps its braces;
* removes the text of a **`<poem>` tag MediaWiki could not pair**. ଛମାଣ ଆଠଗୁଣ୍ଠ's scan page
  ୧୩୩ closes a verse block with `<poem/>` instead of `</poem>`; the opening tag then has no
  partner and is printed verbatim (`…ମସିହା । <poem> ଏଇଚ ଆରି; …`, found by the EXP-015
  inspection). A paired tag always renders as a poem block, never as text, so a literal
  `<poem>`, `</poem>` or `<poem …>` is residue; the lines it was meant to lay out stay as
  MediaWiki joined them. Counted as `literal <poem> tag -> removed`.

Any other stray markup is left in place for the inspection report to show.

**Known limitation (not fixed, on purpose):** a word split across two printed pages without
a hyphenation template renders with ProofreadPage's join space in the middle (`अधि कार` for
`अधिकार`). That is what the wiki shows; guessing where words continue would be inventing
text.

**Proofreading gate.** Every scan page has a ProofreadPage level (0 without text, 1 not
proofread, 2 problematic, 3 proofread, 4 validated). A source that renders any page at level
1 or 2 is `unproofread_refused`: unchecked OCR in an Indic script (broken conjuncts, wrong
matras) is exactly the noise a tokenizer comparison must not learn from. The text is kept
under `sources/` so you can see which pages; the `reason:` line names them. Preflight warns
about it (`WARN UNPROOFED`) from the sampled part of the page.

**Where the level comes from — three anchor shapes.** Each wiki renders ProofreadPage's page
anchor with its own template (all seen on 2026-09-24):

| Wiki | Anchor in the render | Level |
|---|---|---|
| hi, bn, as | `<span class="pagenum ws-pagenum" data-page-name="…" data-page-quality="4">` | read from the render |
| gu, or | the same span with only `title="<percent-encoded page title>"` (and `data-page-number`) | **not shown** — asked from the wiki's API |
| ml | an older anchor, `<span id="pr_page">[ <a class="prp-pagequality-4" title="താൾ:…/4">4</a> ]</span>` | read from the link's class |

None of them reaches the text (ml's bracketed `[ 4 ]` page number included). For every page
whose level the render does not show, a fetch asks the wiki itself
(`api.php?action=query&prop=proofread&titles=…`, in batches) and records the lookup in
`content_check.page_quality.level_lookup`. **If a level still cannot be confirmed** — the API
is unreachable, answers with an error, or knows no such page — the source is refused as
`unproofread_refused` ("could not be confirmed"); so is ProofreadPage content without a
single page anchor. Before this rule, such a render passed the gate unchecked: the gate only
refused levels it could *see*. Saved text (`--local-file`) cannot ask the API, so hidden
levels there are refused too. Preflight notes hidden levels and says the build will ask.

**Scan page titles use each wiki's digits.** On hi, bn, gu, or and as the page titles are
written in the script's own digits (`पृष्ठ:गो-दान.djvu/२८`, `પૃષ્ઠ:…pdf/૨૧`, `ପୃଷ୍ଠା:…pdf/୪`); on ml
in ASCII digits. The `<pages from=N to=M>` tag always takes plain numbers. When you look pages
up by hand, use the right digits: on or.wikisource the ASCII-digit titles (`…pdf/4`) exist
but are redirects left from a rename, marked level 1 — the render does not use them.

---

## 4. Licence verification

Three different things. Keep them separate:

| Concept | Question | Where it lives |
|---|---|---|
| **Licence evidence** | *may* we use this text? | `license_evidence` in the manifest; result recorded in `corpus.json` + provenance; `licence_proof` = `payload-marker` \| `licence-evidence` |
| **Content hash** | *which bytes* did we use? | `sha256` of the cleaned text; pinned only by `--pin` |
| **Content sufficiency** | *is there enough* of it? | `stats.json` per language; `coverage.json` status |

A hash proves **identity, never permission** — the provenance file says
`hash_is_licence_proof: false` next to every hash, including local ones. A source that has
plenty of text and a pinned hash but no licence evidence is still `UNVERIFIED`.

Strength of evidence, weakest to strongest:

1. `scope: site` — the host's default content licence (what the Wikisource sources declare
   today). Proves the *site*, not the work.
2. `scope: page` — evidence tied to the specific work. Prefer this whenever you can get it.
3. **Human review** — reading the work page's own licence template and recording the
   finding in the source `notes`. **Still required** for the Wikisource works even after
   the API evidence resolves, because the API cannot tell you that a particular edition is
   differently licensed.

The licence allow-list is unchanged and enforced in code:
`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `PD-US`. A licence outside it is rejected at
validation time, before any network call.

---

## 5. Pinning

Exact order — no shortcuts:

```
1. fetch      python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-008 --out <dir>
2. inspect    read sources/<id>.txt, provenance, stats.json, coverage.json, leakage.json
3. licence    confirm licence_proof and the recorded evidence result are what you expect
4. content    confirm language, script, volume, and that truncated is False or understood
5. hash       computed during step 1 over the cleaned text (sha256 in provenance)
6. pin        python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-008 --out <dir>
```

Notes:

* The hash exists from step 1, but it only enters the manifest at step 6. **Inspect before
  you pin.**
* `--pin` re-fetches, so pinning reflects the state at pin time, not at inspect time.
* `--pin` with nothing verified leaves the manifest byte-identical — it cannot smuggle in
  an unverified pin.
* After pinning, `git diff corpora/tokenizer/indic-tokenizer-v2/sources.json` should show
  only `sha256`, `verified` and `retrieved_at` changes for sources that verified.

### 5.1 Re-verification: what a pin buys you

Once a hash is pinned, every later `--fetch` compares against it:

| What happened | Status | Exit code | Manifest |
|---|---|---|---|
| same bytes as the pin | `verified` | `0` | re-pinned with the same hash (only `retrieved_at` moves) |
| different bytes | `hash_mismatch` | `1` | **unchanged**; the new bytes are not used and not pinned |
| could not fetch | `fetch_failed` | `0` | unchanged |

A `hash_mismatch` prints `REFUSED: <id> no longer matches its pinned sha256` on stderr and
the `reason:` line carries both hashes. This is deliberate: a pinned hash is the recorded
identity of a source, so new bytes have to be a decision, not a refresh.

To accept new bytes **deliberately**, edit the manifest first: set `sha256: null` and
`verified: false` for that source, add a note in `notes` saying why the bytes changed (new
edition, different URL, licence review outcome), then run the normal fetch → inspect → pin
sequence. Do it in a commit of its own so the change is reviewable.

---

## 6. Local-file fallback

Lawfully obtained text can be supplied with no network at all:

```
# one file for one declared source (in that source's raw format: a Gutenberg .txt,
# wikitext, or — for a mediawiki-parse source — the saved api.php?action=parse JSON)
python scripts/build_tokenizer_corpus.py \
    --local-file hi-wikisource-godaan-ch01-ccbysa=~/godaan-ch01.json \
    --include-unverified --no-record --out data/tokenizer/indic-tokenizer-v2

# a directory of <source_id>.txt files
python scripts/build_tokenizer_corpus.py --local-dir data/local-sources --no-record
```

* The source id **must** already be declared in the manifest. An unknown id, or a file in
  `--local-dir` matching no declared source, is a hard error (exit 2) — never a silent
  skip.
* Local text is ingested as **`local_unverified`**. Its hash is computed and recorded, but
  the provenance marks `hash_is_licence_proof: false`, `local_source: true` and
  `local_origin_path: <path>`.
* Without `--include-unverified` it is ingested and reported but **kept out of the split**.
  With the flag it enters the corpus — and the slot **still** reports `UNVERIFIED`, never
  `EVALUATED`, no matter how large the text is.
* Local text is the right tool when a human has lawfully obtained the work (for example,
  downloaded on another machine). It is **not** a way around licence review: the licence
  question is simply unanswered, and the coverage report says so.

---

## 7. Exact commands

All of these exist today; nothing here is invented.

```
# --- preflight (read-only) ------------------------------------------------
python scripts/build_tokenizer_corpus.py --preflight
python scripts/build_tokenizer_corpus.py --preflight --print-json
python scripts/build_tokenizer_corpus.py --preflight --source hi-wikisource-godaan-ch01-ccbysa --timeout 10

# --- list what is declared (no dedicated flag exists) ---------------------
python3 -c "import json;d=json.load(open('corpora/tokenizer/indic-tokenizer-v2/sources.json'));\
[print(s['id'], s['language'], s['license_id'], s['verified'], s['sha256']) for s in d['sources']]"

# --- validate + report coverage without touching the network --------------
python scripts/build_tokenizer_corpus.py --no-record

# --- fetch (acquire) ------------------------------------------------------
python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
python scripts/build_tokenizer_corpus.py --fetch --source hi-wikisource-godaan-ch01-ccbysa \
    --timeout 60 --no-record --out /tmp/probe-godaan

# --- check a built corpus (re-hashes every file it names) -----------------
python scripts/build_tokenizer_corpus.py --print-json \
    --out data/tokenizer/indic-tokenizer-v2
cat data/tokenizer/indic-tokenizer-v2/acquisition.json     # one row per source
python3 -c "import json;print(json.dumps(json.load(open('data/tokenizer/indic-tokenizer-v2/coverage.json'))['summary'],indent=1))"
python3 -c "import json;d=json.load(open('data/tokenizer/indic-tokenizer-v2/acquisition.json'));\
[print(r['source_id'], r['verification_status'], r['licence_proof'], r['sha256'], r['error'][:60]) for r in d['sources']]"
python3 -c "import json;d=json.load(open('data/tokenizer/indic-tokenizer-v2/corpus.json'));\
[print(s['id'], s['ingest']['status'], s['ingest'].get('truncated'), s['ingest'].get('licence_proof')) for s in d['sources']]"

# --- pin (only after inspection) ------------------------------------------
python scripts/build_tokenizer_corpus.py --fetch --pin --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
git diff corpora/tokenizer/indic-tokenizer-v2/sources.json   # only sha256/verified/retrieved_at

# --- re-verify later (exit 1 = a pinned source changed) -------------------
python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2

# --- local ingestion ------------------------------------------------------
python scripts/build_tokenizer_corpus.py --local-file hi-wikisource-godaan-ch01-ccbysa=~/godaan-ch01.json \
    --include-unverified --no-record
python scripts/build_tokenizer_corpus.py --local-dir data/local-sources --include-unverified --no-record
```

Chapter discovery (proposes entries for review, never edits the manifest):

```
python scripts/discover_wiki_chapters.py --lang hi --work "गो-दान"
```

Not implemented (future work, do not pretend otherwise): `--list-sources`; a multi-URL /
chapter-set source type (a single `<pages>` range render covers scanned books, §3.1);
automatic *addition* of discovered chapters to the manifest.

---

## 8. First-fetch checklist

Tick only what you have personally verified. If any box is unticked, the source stays
unverified and its slot does not become `EVALUATED`.

### Alice — `en-gutenberg-alice-pd`

- [ ] `sources/en-gutenberg-alice-pd.txt` starts with `[Illustration]`, the title and the contents list, then "Alice was beginning…" — no Gutenberg header/licence text
- [ ] Language is English; no leftover header/footer/licence block
- [ ] `truncated: false` (or explained); the en slot's ≥ 500 documents and ≥ 200,000 characters come from Alice + Sadhana together
- [ ] `licence_proof == "payload-marker"` (PD-US marker present in the fetched payload)
- [ ] Attribution present in `sources/en-gutenberg-alice-pd.provenance.json`
- [ ] `sha256` pinned **only** via a later `--pin` run, after all of the above

### Godaan — `hi-wikisource-godaan-ch01-ccbysa` … `ch36`

- [ ] All 36 chapters `verified` — none `index_page_refused`, `unproofread_refused` or `fetch_failed` (if one is, read its `reason:` line before doing anything else)
- [ ] `ch01` starts with "होरीराम ने दोनों बैलों को सानी-पानी देकर…"; a few later chapters open with their number
- [ ] No header/navigation text (`पीछे`, `आगे`, `प्रेमचंद` on a line of its own), no numbers like `38658`, no CSS anywhere
- [ ] Spot-check two chapters' `content_check.page_quality`: `first_page`/`last_page` match the range in the source notes; every level is 3 or 4 (or 0)
- [ ] Chapters 23 and 24 (wiki category "missing scan pages") read and the finding recorded in their notes
- [ ] Evidence reachable, `marker_found: true`, `scope: site` (chapters 2–36 show `reused_within_run: true`)
- [ ] **Human licence review of the work page recorded** (`गो-दान` carries `{{PD-India}}`)
- [ ] `sha256` pinned only after all of the above

### Gitanjali — `bn-wikisource-gitanjali-1913-ccbysa`

- [ ] Status `verified`, not `unproofread_refused` / `index_page_refused` / `fetch_failed`
- [ ] Starts with poem ১ "আমার মাথা নত করে দাও হে তোমার"; one verse per line; no CSS (`.mw-parser-output`) anywhere
- [ ] `content_check.page_quality`: 178 pages, first `…djvu/১৩`, last `…djvu/১৯০`, all level 4
- [ ] Evidence reachable, `marker_found: true`, `scope: site`
- [ ] **Human licence review recorded**: which licence template does this edition carry? (Tagore died 1941 — if PD, record it and update `license_id` with a note before re-verifying)
- [ ] Slot `EVALUATED` only together with Devdas; `INSUFFICIENT` if Devdas is not verified — expected, not a failure
- [ ] `sha256` pinned only after all of the above

### Sadhana — `en-gutenberg-sadhana-pd`

- [ ] Status `verified`; `licence_proof == "payload-marker"`
- [ ] Starts "Produced by Chetan Jain at BharatLiterature" then the title page and the Author's Preface; ends "…not distant, not anywhere else."; no Gutenberg licence text left
- [ ] `truncated: false`; en slot `EVALUATED` with Alice
- [ ] `sha256` pinned only after all of the above

### Devdas — `bn-wikisource-devdas-ccbysa`

- [ ] Status `verified`, not `unproofread_refused` / `index_page_refused` / `fetch_failed`
- [ ] Starts with `এক` and "একদিন বৈশাখের দ্বিপ্রহরে…"; ends "…দেখিয়া সে মরিতে পারে।" then `সমাপ্ত`; no CSS anywhere
- [ ] `content_check.page_quality`: 106 pages, first `…pdf/৫`, last `…pdf/১১০`, levels 3 and 4 only
- [ ] Evidence reachable, `marker_found: true`, `scope: site` (`reused_within_run: true` after Gitanjali)
- [ ] **Human licence review recorded** (the work page carries `{{PD-India}}`; Sarat Chandra died 1938)
- [ ] `sha256` pinned only after all of the above

### First group of new languages — gu, ml, or, as (§2.6)

- [ ] All 7 `verified`; none `unproofread_refused` (a "could not be confirmed" reason means the API lookup failed: rerun)
- [ ] gu and or: `content_check.page_quality.level_lookup` present with `confirmed` = `asked` and `error: null`; levels 3/4 only
- [ ] ml: page levels read from the render (no `level_lookup`), and no bracketed page numbers in the text
- [ ] Each starts with its first chapter (gu `સરસ્વતીચંદ્ર.` / `પ્રકરણ ૧.`, ml `അദ്ധ്യായം ഒന്ന്`, or `ପ୍ରଥମ ପରିଚ୍ଛେଦ`, as `মনোমতী` / `প্ৰথম খণ্ড`); no CSS anywhere
- [ ] Characters per slot: EXP-015 had gu 297,628, ml 399,168, or 203,451, as 241,025; with the `<poem>` fix only or changes, to 203,444 (all four `EVALUATED`) — EXP-016 showed exactly this
- [ ] or's repairs line shows `literal <poem> tag -> removed (1)`, and nothing is flagged for it any more
- [ ] `sha256` pinned only after all of the above
- [ ] After the pin, `git diff` shows `sha256`, `verified` and `retrieved_at` for these 7 and only `retrieved_at` for the 40 pinned before; each new hash begins with its EXP-016 prefix (§2.6) — EXP-017 showed exactly this

### Second group of new languages — pa, kn, te, ta (§2.7)

- [ ] Exit 0 and no `REFUSED:` line: the 47 pinned sources came back unchanged
- [ ] All 8 new sources `verified`; none `unproofread_refused`
- [ ] Levels read from the render (no `level_lookup` for these 8): pa, kn and te level 4 only; ta level 3 only
- [ ] pa starts with `੧. ਕਾਂਡ।` and has no advertisement text (no `ਮੈਨੇਜਰ`, no price list); ta starts with `என் சரித்திரம்` and `அத்தியாயம் 1`; kn shows no running headers (the book title with a page number); no CSS anywhere
- [ ] Coverage section: pa, kn, te and ta each `EVALUATED` (at least 500 documents and 200,000 characters); an `INSUFFICIENT` slot needs a second work, with the reason recorded — never padding
- [ ] Anything flagged is explained before any lock
- [ ] `sha256` pinned only after all of the above, in a later lock job

### Marathi and Urdu — mr and ur (§2.8)

- [x] EXP-022 verified all four sources; the previous 55 pinned hashes were unchanged
- [x] mr page qualities are 112 level-3 pages (13–124); ur has 298 level-3 and 37 level-4 pages, with level-1 page 308 excluded
- [x] Work/scan licensing reviewed independently: Apte's 1928 edition is public domain by term and Commons tag; Premchand's 1929 edition was published during his lifetime and the Commons scan is tagged Public domain; each wiki's live rights endpoint proves CC BY-SA for its transcription
- [x] Samples begin and end within the works; no front matter, back matter, CSS or running scan numbers appear in the selected text
- [x] EXP-022 inspection has no flags for the four new sources; Urdu's repairs preserve prose and are counted in the report
- [x] Both slots meet the 500-document / 200,000-character targets: mr 1,667 / 256,165; ur 1,378 / 208,621
- [x] EXP-023 fresh `--fetch --pin` succeeded; each new fingerprint matches its EXP-022 prefix and only permitted lock fields changed

---

## 9. Missing languages

As of EXP-023, all 59 sources are verified and pinned. Thirteen slots are `EVALUATED`,
including mr (1,667 documents / 256,165 characters) and ur (1,378 / 208,621). hi-en remains
`NOT_EVALUATED`: no licensed, attributable Hinglish/Romanised-Hindi source was identified.
No text was substituted or padded.

* **Nothing is substituted.** No synthetic text, no machine translation, no "close enough"
  corpus, no filling a slot from a related language. A language we cannot source legally
  stays `NOT_EVALUATED`, with the reason recorded in the manifest, and later tokenizer
  reports simply do not cover it.
* **`hi-en` (Hinglish) may stay `NOT_EVALUATED` permanently.** The available Hinglish /
  Romanised-Hindi corpora are social-media derived, with unclear licences and privacy
  terms; the allow-list does not permit them. If no lawful source appears, the honest
  outcome is a permanently unevaluated slot.
* The slots without sources have a **family-level candidate provider** recorded
  (`candidates` in the manifest, e.g. "Tamil Wikisource", CC BY-SA 4.0 assumed
  conservatively). A candidate is not a source and must not be read as coverage.
* **Adding a source** means: pick a concrete work, confirm its licence tag, add it to
  `corpora/tokenizer/indic-tokenizer-v2/sources.json` with attribution, `license_url`,
  `kind`, `max_chars` and (if the payload carries no licence) a `license_evidence` block,
  reference it from the language slot, then run validation and the tests. Duplicate ids,
  references to non-existent ids and orphan sources are rejected on purpose (§ M-5).

---

## 10. Research safety

Do **not** use:

* copyrighted books without explicit permission (a public-domain *author* is not a
  public-domain *edition*, and PD in one country is not PD everywhere);
* text scraped from random websites, forums or social media with no clear licence;
* private, personal or credential-adjacent data, however obtained;
* anything whose licence cannot be established — including "everyone knows this one is
  fine".

Also:

* no fabricated provenance, attribution, URLs, hashes or sizes — if a value was not
  observed, it stays `null`;
* no "we'll fix the licence later" — an unverified source is unverified until the evidence
  is recorded;
* respect `robots.txt`, rate limits and the source's terms; identify the client honestly
  (the fetcher sends a `frontier-ai-*` user agent);
* the corpus is a research fixture, not a redistribution: attribution travels with every
  copy, and CC BY-SA obligations (share-alike) apply to derivative distributions.

---

## 11. Coverage after acquisition

Re-read `coverage.json` after every run. The statuses mean:

| Status | What to do |
|---|---|
| `EVALUATED` | verified sources meeting both targets — usable for tokenizer research |
| `INSUFFICIENT` | verified but short: add more text for that language, or report it short |
| `UNVERIFIED` | declared but not verified (fetch failed, licence not evidenced, or local text) |
| `NOT_EVALUATED` | no source declared — record the reason; do not invent one |

An empty or partial corpus still produces a full report. That is the point: the report is
evidence of what exists, not a form to be filled in.

---

## 11.1 Reports a run produces

| File | Audience | Contains |
|---|---|---|
| standard output, `per-source acquisition` block | human | one line per source: reachable / content / licence proof / hash / slot sufficiency, plus `reason:` when it failed |
| `acquisition.json` | machine | the same rows, with all fields, plus a `summary` count. Timestamp-free: two runs over identical inputs are byte-identical |
| `coverage.json` | both | status + reason for all 14 slots (`EVALUATED` / `INSUFFICIENT` / `UNVERIFIED` / `NOT_EVALUATED`) |
| `stats.json` | both | per-language examples, characters, bytes, unique texts, duplicates, truncation |
| `leakage.json` | both | exact SHA-256 overlap and 8-gram overlap ratio |
| `corpus.json` | machine | everything above plus per-file hashes and the manifest hash after pinning |
| `sources/<id>.provenance.json` | human/audit | title, URL, licence, attribution, `licence_proof`, the verbatim evidence result, `content_check`, `truncated`, `hash_is_licence_proof: false` |
| `inspect_corpus_sources.py --output FILE` (a separate, read-only step) | human | the review report of §0.3: per-source lines, flags, repairs, coverage of all 14 slots, samples. A local runner pushes it as `corpora/tokenizer/indic-tokenizer-v2/reports/EXP-0NN-inspection.txt` |

## 12. Open items this runbook cannot close

* It has **still never been executed against live data** — the environment it was written
  and extended in cannot reach any corpus host (every endpoint fails TLS). Treat every
  "expect" above as a hypothesis to confirm.
* The gates in §2.0 (licence, index page, pinned hash, empty, fetch) are implemented and
  covered by offline tests, but they have only ever run against synthetic payloads. A real
  Wikisource page is messier than a fixture: if the index-page gate fires on genuine prose,
  report it (with the `link_line_ratio` from the `reason:` line) rather than working around
  it — the thresholds are deliberately conservative.
* ~~Whether `?action=raw` returns the work~~ — answered by EXP-008: it does not, for either
  wiki (a redirect and an infocard), and for scanned books it never can (§3.1).
* ~~Whether the `rightsinfo` evidence resolves~~ — answered by EXP-008: both wikis declare
  CC BY-SA 4.0.
* The `mediawiki-parse` cleaner ran on live payloads for the first time in EXP-009 (37
  sources verified). Checked against the live wiki on 2026-09-24: chapter 1 starts on scan
  page 11 with the novel's first line; the chapter 1→2 boundary is not duplicated (the
  shared page's `<section end="1"/><section begin="1"/>` markers split it exactly); chapter
  36 ends with the novel's last line on page 363 and no back matter follows. The human
  reading in §8 is still required before `--pin`.
* ~~Gitanjali: one leftover piece of wiki markup~~ — located by EXP-011's *where to look*: a
  `}}` typed after the date of poem 118 on scan page ১৪৮, a page that opens and closes its
  block with `{{Block center/s}}` … `{{block center/e}}`. It closes nothing; the cleaner now
  removes such braces (§3.1). The page could also be corrected on the wiki itself.
* **Typos in the transcription are kept.** A stray Latin `l` typed on पृष्ठ:गो-दान.djvu/८९
  (`पहुँची, lएक वन-पुष्प`) shows up in the report as one letter of another script in chapter
  7; like the OCR confusions below it is text as the wiki has it, and it stays.
* **"Proofread" is not error-free.** ProofreadPage level 3 means one volunteer checked the
  page; Godaan's level-3 pages still carry OCR confusions (e.g. `हीग` for `हीरा`, `ग्विलाते`
  for `खिलाते`, `वैठे` for `बैठे` in the opening pages of chapter 9, read on the live wiki on
  2026-09-24). The pipeline keeps the text as the wiki has it and never "corrects" words;
  this is a known noise floor for the Hindi slot, worth remembering when comparing
  tokenizers on it.
* No per-language balancing or genre control exists yet; one work per language will
  confound later cross-language comparisons (Stage C) even after acquisition succeeds.
