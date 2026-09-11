# Smoke-test corpora (real text, known licences)

The synthetic generator (`src/frontier_ai/data/synthetic.py`) remains the **default CI
fixture**: no network, no licence questions, byte-identical everywhere. This directory
holds the second fixture the roadmap asks for — a small amount of **real text with a
recorded licence and provenance** — so the pipeline is at least occasionally exercised on
something a human wrote.

**These files are a smoke-test / evaluation fixture, not training data.** Each source is
capped at 200,000 characters by default (`max_chars` in `sources.json`), which is far below
anything you would train on. Nothing here selects a tokenizer, a model or a data mix.

## Sources

`sha256` is `null` for every entry until somebody actually fetches the file: see
"Verification status" below.

| id | title | language / script | source | licence | cap |
|---|---|---|---|---|---|
| `en-alice-pd` | Alice's Adventures in Wonderland (Lewis Carroll) | en / Latin | `https://www.gutenberg.org/cache/epub/11/pg11.txt` | **PD-US** — public domain in the United States; [Project Gutenberg terms](https://www.gutenberg.org/policy/license.html) | 200,000 chars |
| `hi-godan-ccbysa` | गोदान (Godaan) — Munshi Premchand | hi / Devanagari | `https://hi.wikisource.org/wiki/गोदान?action=raw` | **CC BY-SA 4.0** (assumed conservatively; verify the page's own licence tag on first fetch) | 120,000 chars |
| `bn-gitanjali-ccbysa` | গীতাঞ্জলি (Gitanjali) — Rabindranath Tagore | bn / Bengali | `https://bn.wikisource.org/wiki/গীতাঞ্জলি?action=raw` | **CC BY-SA 4.0** (same caveat) | 120,000 chars |

The full machine-readable form (including the attribution string and notes) is
[`sources.json`](sources.json); `python scripts/fetch_smoke_corpus.py --list` prints it.

### Licence notes

- **PD-US / Project Gutenberg.** The English text is public domain in the United States
  (Carroll died 1898). Copyright terms outside the US differ; check them before using the
  text in a jurisdiction that matters to you. Project Gutenberg's terms permit copying and
  redistributing the plaintext, but **not** the use of the Gutenberg trademark, so the
  header/footer that carries their branding and licence text is stripped by
  `clean_gutenberg_text()` and the licence is recorded in the provenance file instead. We
  are redistributing the *work*, not the Gutenberg edition.
- **CC BY-SA 4.0 (Wikisource).** Attribution is required, the licence must be indicated,
  and adaptations must be shared under the same licence. The attribution string for each
  entry is in `sources.json` and is copied into every provenance record and into the
  prepared `*.meta.json`. If a Wikisource page turns out to carry a public-domain
  template, relax `license_id` to `PD-US` **and record what you saw** in `notes`.
- Only licences in `ALLOWED_LICENSES` (`src/frontier_ai/data/corpora.py`) may appear in
  the manifest: `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `PD-US`. No scraping, no
  copyrighted books, no private or unclear-licence data.

## Acquisition

```bash
. .venv/bin/activate

# 1. what is configured, and has it been verified?
python scripts/fetch_smoke_corpus.py --list

# 2. fetch (writes data/raw/smoke/<id>.txt + <id>.provenance.json)
python scripts/fetch_smoke_corpus.py --fetch                    # everything
python scripts/fetch_smoke_corpus.py --fetch en-alice-pd        # one source

# 3. once the licence marker was found and you have read the licence, pin the hash
python scripts/fetch_smoke_corpus.py --fetch --pin

# 4. verify what is on disk against the pinned hashes (CI-friendly)
python scripts/fetch_smoke_corpus.py --check
```

Files land in `data/raw/smoke/`, which is **git-ignored** — only the manifest, this
README and the code are committed. Exit codes: `0` ok · `1` `--check` found a missing or
mismatching file · `2` hash mismatch during `--fetch` (file left untouched) · `3` the
fetch failed, or `--pin` was refused because no licence marker was found.

### Then into the normal pipeline

```bash
python scripts/prepare_data.py --source data/raw/smoke/en-alice-pd.txt \
    --provenance data/raw/smoke/en-alice-pd.provenance.json \
    --level char --out data/smoke-en
python scripts/prepare_data.py --source data/raw/smoke/hi-godan-ccbysa.txt \
    --provenance data/raw/smoke/hi-godan-ccbysa.provenance.json \
    --level char --out data/smoke-hi
```

`--provenance` copies the record into the prepared `*.meta.json` (`source_provenance`), and
`prepare_data.py` re-hashes the source file and warns loudly if it no longer matches the
pinned digest — so "someone edited the corpus" becomes a visible failure instead of silent
measurement drift.

## What the provenance record contains

`data/raw/smoke/<id>.provenance.json` (schema `1.0`): `id`, `title`, `language`, `script`,
`source_url`, `license_id`, `license_url`, `attribution`, `kind`, `retrieved_at` (UTC),
`sha256` of the exact bytes stored, `n_chars`, `n_bytes`, `max_chars`, `verified`, `notes`.
It is copied verbatim into `*.meta.json` by `scripts/prepare_data.py --provenance`.

## Verification status (2026-09-10)

**No entry has been fetched or pinned yet.** The manifest therefore ships with
`sha256: null, verified: false` for all three sources, and `scripts/fetch_smoke_corpus.py`
refuses to pin a hash unless the download carries a marker for the licence we claim. This
is deliberate: the sandbox this was developed in can reach `pypi.org` only —
`gutenberg.org`, `wikisource.org`, `unicode.org` and `huggingface.co` are all unreachable
— so no hash could be measured here, and a hash asserted from memory would be a lie.

Consequences, stated plainly:

- The corpus does not exist in this checkout. Tests that need it skip with a clear message;
  the pipeline tests run against a locally authored sample instead.
- The Wikisource **page titles are unverified** for the same reason. If a title 404s or the
  page carries a different licence template, edit `sources.json` (title, `license_id`,
  `notes`) — the fetch script will say what it found and refuse to pin on an unverified
  licence.
- Run steps 2–4 above in an environment with network access to make the corpus real; commit
  the pinned `sources.json` afterwards.

## Adding a source

1. Only permissive/public-domain licences from the allow-list. When in doubt: don't.
2. Add an entry to `sources.json` with a resolvable HTTPS URL, SPDX `license_id`,
   `license_url`, an `attribution` string, and a `max_chars` cap small enough to stay a
   fixture. Leave `sha256` as `null`.
3. Fetch, read the licence, pin. Add a note if the page's licence tag differs from what the
   manifest assumed.
4. Do not commit the corpus text itself — `data/` is git-ignored on purpose and the
   provenance makes it reproducible from the URL plus the pinned hash.
