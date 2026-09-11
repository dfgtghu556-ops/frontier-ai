# Stage B acquisition runbook — Project 004

**This is a procedure, not a report.** It was written in an environment that cannot reach
any corpus host (every endpoint fails with `TLS/SSL … EOF`), so **none of the steps below
have been executed against live data yet**. Nothing here has been acquired, verified or
pinned. Run it where the network works, and record what you actually observe.

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

---

## 0. Quick start (Termux, or any network-enabled machine)

This is the whole workflow: **preflight → fetch → inspect → pin → (later) re-verify**.
Nothing is acquired until step 2 and nothing is pinned until step 4. Every command below
exists in the repository today; none of them are invented.

### 0.0 One-time setup (once per machine)

```
pkg install python git          # Termux only
cd ~/frontier-ai
git checkout arena/01a08a78-frontier-ai
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"         # torch + numpy + pytest + ruff
```

* On Termux the interpreter is `python`; on a desktop it is usually `python3`. Use whichever
  one works — they are the same program.
* `torch` is a big download, but it is not optional here: the corpus CLI imports the
  experiments package, which imports torch.
* If you leave the shell, re-run `. .venv/bin/activate` before the commands below.

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

**What you must read in the output** — the `per-source acquisition` block, one line per
declared source plus its detail line:

```
[corpus] per-source acquisition
[corpus]   en-gutenberg-alice-pd            verified             PD-US        reachable=yes content=yes proof=payload-marker   chars=   152,089
[corpus]       sha256=1a2b3c4d5e6f7a8b…  slot=en EVALUATED sufficient=yes
[corpus]   hi-wikisource-godan-ccbysa       index_page_refused   CC-BY-SA-4.0 reachable=yes content=yes proof=none             chars=       466
[corpus]       sha256=fcd858968746beff…  evidence=site marker_not_found  slot=hi UNVERIFIED sufficient=no
[corpus]       reason: …looks like a contents/index page…
```

* `reachable` — did the host answer? (`n/a` = no fetch was attempted for this source)
* `content` — did the run get text?
* `proof` — how the licence was proven: `payload-marker`, `licence-evidence`, or `none`
* `sha256` — the hash of the cleaned text this run produced (`none` = nothing usable)
* `slot=… <status> sufficient=…` — whether the language slot met its targets
* `reason:` — only printed when something failed, and it always says what to do next

The same information is in `data/tokenizer/indic-tokenizer-v2/acquisition.json`
(machine-readable, one row per source, timestamp-free so two identical runs produce a
byte-identical file).

### 0.3 Step 3 — inspect before you trust it

```
cat data/tokenizer/indic-tokenizer-v2/acquisition.json
head -c 600 data/tokenizer/indic-tokenizer-v2/sources/en-gutenberg-alice-pd.txt
python -c "import json;d=json.load(open('data/tokenizer/indic-tokenizer-v2/coverage.json'));[print(r['language'], r['status'], r['reason']) for r in d['languages']]"
```

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
The `git diff` should show nothing but those three fields, for those sources only. If
nothing verified, the manifest is byte-identical and there is nothing to commit.

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
python scripts/build_tokenizer_corpus.py --preflight --source hi-wikisource-godan-ccbysa
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
| **Content shape** | a `wikitext` payload where a majority of ≥20 lines are wiki links with <20 characters of prose left | `index_page_refused` | No — acquire the chapter subpages or use `--local-file` (§6) |
| **Pinned hash** | the manifest pins a `sha256` and this fetch cleans to different bytes | `hash_mismatch` | Only by clearing `sha256`/`verified` in the manifest first, with a note saying why (§5) |
| **Empty text** | the cleaned text is empty | `empty` | No |
| **Fetch** | the host could not be reached or the response was not UTF-8 | `fetch_failed` | No |

A refused source is **not** verified, is **not** part of the train/held-out split and is
**not** pinned. Its text file is still written under `sources/<id>.txt` so you can look at
what you got and decide what to do next. Its provenance says `verification: unverified`
and records `content_check` (what the shape gate saw) and `hash_is_licence_proof: false`.

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
  confirm: it starts at "Alice was beginning to get very tired…" (not at a header), there is
  no Gutenberg boilerplate left, and the text is English prose.
* **Volume** — expect well over 200,000 characters after cleaning (Alice is ~170 KB of
  text). `max_chars` is 400,000; if `truncated: true` appears, the source was cut and you
  should say so in the record rather than quietly accepting it.
* **Licence** — PD-US, with the attribution already written in the manifest. If the marker
  is missing, the file is not a Gutenberg text: stop, do not pin.
* **Hash + pin** — only after the above.

### 2.2 `hi-wikisource-godan-ccbysa` (Hindi, CC BY-SA 4.0 assumed, kind `wikitext`)

* **Fetch** — `https://hi.wikisource.org/wiki/गोदान?action=raw`. Expect **bare wikitext**:
  no licence notice (that is why the evidence endpoint exists).
* **Inspect — this is the risky one.** A Wikisource *work root* frequently contains only a
  header template and a list of chapter links. Before accepting it, check
  `sources/hi-wikisource-godan-ccbysa.txt`: is it prose in Devanagari, or just
  `[[गोदान/अध्याय १|अध्याय १]]` lines? Preflight's `looks_like_index_page` warning and the
  `wiki_links` / `sample_lines` counts are the first signal; the ingested file is the
  ground truth. See §3 for what to do if it is an index page.
* **Volume** — the target is ≥500 documents and ≥200,000 characters. Godaan is long enough
  if (and only if) you actually got the chapters.
* **Licence** — evidence endpoint:
  `https://hi.wikisource.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json`,
  marker `https://creativecommons.org/licenses/by-sa/4.0/`, `scope: site`. The result is
  recorded verbatim in `corpus.json` / the provenance file. **Site-level evidence proves
  the wiki publishes under CC BY-SA 4.0, not that this edition carries that tag.** Open the
  work page, read its own licence template (many Premchand works are PD-old), and record
  what you found in the source `notes` before treating the slot as trustworthy.
* **Hash + pin** — only after both content and licence checks.

### 2.3 `bn-wikisource-gitanjali-ccbysa` (Bengali, CC BY-SA 4.0 assumed, kind `wikitext`)

Same procedure as Godaan, with `bn.wikisource.org`, Bengali script, and this extra
licence wrinkle: **Rabindranath Tagore died in 1941**, so the underlying work is public
domain in India — but the specific edition on Wikisource may carry its own tag. Record
what the page actually says; if it is PD, the `license_id` may be relaxed to `PD-US` with
a note and a fresh verification run (the licence marker list in
`src/frontier_ai/data/corpora.py` only knows the four allowed ids — do not add a new one
without a decision record).

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
  texts are concatenated into one source. If Stage B needs it, add it deliberately with
  tests, not by editing URLs in place.

Do **not** change a source's `source_url` to "whatever happens to work" without updating
the attribution, the notes and the evidence endpoint to match.

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
# one file for one declared source
python scripts/build_tokenizer_corpus.py \
    --local-file hi-wikisource-godan-ccbysa=~/godan.txt \
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
python scripts/build_tokenizer_corpus.py --preflight --source hi-wikisource-godan-ccbysa --timeout 10

# --- list what is declared (no dedicated flag exists) ---------------------
python3 -c "import json;d=json.load(open('corpora/tokenizer/indic-tokenizer-v2/sources.json'));\
[print(s['id'], s['language'], s['license_id'], s['verified'], s['sha256']) for s in d['sources']]"

# --- validate + report coverage without touching the network --------------
python scripts/build_tokenizer_corpus.py --no-record

# --- fetch (acquire) ------------------------------------------------------
python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-008 \
    --out data/tokenizer/indic-tokenizer-v2
python scripts/build_tokenizer_corpus.py --fetch --source hi-wikisource-godan-ccbysa \
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
python scripts/build_tokenizer_corpus.py --local-file hi-wikisource-godan-ccbysa=~/godan.txt \
    --include-unverified --no-record
python scripts/build_tokenizer_corpus.py --local-dir data/local-sources --include-unverified --no-record
```

Not implemented (future work, do not pretend otherwise): `--list-sources`;
a multi-URL / chapter-set source type; automatic chapter enumeration.

---

## 8. First-fetch checklist

Tick only what you have personally verified. If any box is unticked, the source stays
unverified and its slot does not become `EVALUATED`.

### Alice — `en-gutenberg-alice-pd`

- [ ] `sources/en-gutenberg-alice-pd.txt` starts with the story, not with Gutenberg boilerplate
- [ ] Language is English; no leftover header/footer/licence block
- [ ] Volume ≥ 500 documents and ≥ 200,000 characters; `truncated: false` (or explained)
- [ ] `licence_proof == "payload-marker"` (PD-US marker present in the fetched payload)
- [ ] Attribution present in `sources/en-gutenberg-alice-pd.provenance.json`
- [ ] `sha256` pinned **only** via a later `--pin` run, after all of the above

### Godaan — `hi-wikisource-godan-ccbysa`

- [ ] Status is `verified`, not `index_page_refused` — if the gate fired, the page is a contents page and you need the chapter subpages (§3)
- [ ] The payload is Devanagari **prose**, not a contents page (`looks_like_index_page` false, or chapters acquired instead — §3)
- [ ] Language/script correct; wikitext markup removed; no navigation text left
- [ ] Volume ≥ 500 documents and ≥ 200,000 characters; `truncated: false` (or explained)
- [ ] Evidence endpoint reachable, `marker_found: true`, result recorded with `scope: site`
- [ ] **Human licence review of the work page recorded in the manifest notes** (site-level evidence alone is not page-level proof)
- [ ] Attribution present in the provenance file
- [ ] `sha256` pinned only after all of the above

### Gitanjali — `bn-wikisource-gitanjali-ccbysa`

- [ ] Status is `verified`, not `index_page_refused` (§3)
- [ ] The payload is Bengali **prose**, not a contents page (§3)
- [ ] Language/script correct; markup removed
- [ ] Volume ≥ 500 documents and ≥ 200,000 characters; `truncated: false` (or explained)
- [ ] Evidence endpoint reachable, `marker_found: true`, result recorded with `scope: site`
- [ ] **Human licence review recorded**: which licence template does this edition actually carry? (Tagore died 1941 — if PD, record it and update `license_id` with a note before re-verifying)
- [ ] Attribution present in the provenance file
- [ ] `sha256` pinned only after all of the above

---

## 9. Missing languages

Today: 3 slots `UNVERIFIED` (en, hi, bn) and 11 `NOT_EVALUATED`
(hi-en, mr, gu, ta, te, kn, ml, pa, or, as, ur).

* **Nothing is substituted.** No synthetic text, no machine translation, no "close enough"
  corpus, no filling a slot from a related language. A language we cannot source legally
  stays `NOT_EVALUATED`, with the reason recorded in the manifest, and later tokenizer
  reports simply do not cover it.
* **`hi-en` (Hinglish) may stay `NOT_EVALUATED` permanently.** The available Hinglish /
  Romanised-Hindi corpora are social-media derived, with unclear licences and privacy
  terms; the allow-list does not permit them. If no lawful source appears, the honest
  outcome is a permanently unevaluated slot.
* The ten others have a **family-level candidate provider** recorded (`candidates` in the
  manifest, e.g. "Tamil Wikisource", CC BY-SA 4.0 assumed conservatively). A candidate is
  not a source and must not be read as coverage.
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

## 12. Open items this runbook cannot close

* It has **still never been executed against live data** — the environment it was written
  and extended in cannot reach any corpus host (every endpoint fails TLS). Treat every
  "expect" above as a hypothesis to confirm.
* The gates in §2.0 (licence, index page, pinned hash, empty, fetch) are implemented and
  covered by offline tests, but they have only ever run against synthetic payloads. A real
  Wikisource page is messier than a fixture: if the index-page gate fires on genuine prose,
  report it (with the `link_line_ratio` from the `reason:` line) rather than working around
  it — the thresholds are deliberately conservative.
* Whether `?action=raw` on `hi.wikisource.org` / `bn.wikisource.org` returns the work or an
  index page is **unknown** until first fetch (§3).
* Whether the Wikisource `rightsinfo` evidence resolves is **unknown** until first fetch.
* No per-language balancing or genre control exists yet; one work per language will
  confound later cross-language comparisons (Stage C) even after acquisition succeeds.
