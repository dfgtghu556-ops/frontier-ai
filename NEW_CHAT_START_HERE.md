# New chat? Start here

Written on 2026-09-25 by the Arena agent working on branch `arena/01a0d31f-frontier-ai`. It
lets a **new Arena chat**, which has none of the earlier conversation, pick up the work.
Read all of it before doing anything. Whoever works on the project last keeps it current;
see the *Status log* (§8).

## 1. Get the work into your checkout first

A new Arena chat starts on its own new branch, made from `main`. `main` is old: `7fb7f49`,
from PR #1, merged 2026-09-10. All later work is on the branch
`arena/01a0d31f-frontier-ai`, unless someone has merged it into `main` since. Bring it in:

```
git fetch origin arena/01a0d31f-frontier-ai
git merge --ff-only FETCH_HEAD
```

* If the merge is refused, run `git merge --no-edit FETCH_HEAD` and tell the operator.
* If the fetch says the branch does not exist, it was merged into `main` and deleted.
  You already have everything.
* Always fetch that branch by name and use `FETCH_HEAD`. In the Arena sandbox a plain
  `git fetch origin` does not create `origin/arena/...` refs.
* The local assistant (§4) keeps pushing to `arena/01a0d31f-frontier-ai`. Before any work
  on P004B, fetch and merge it again.
* Push only your own working branch, as your system instructions say. If the operator's
  machine should move to your branch, give the operator the exact commands
  (`git fetch origin <your-branch>`, then `git switch <your-branch>`). Also update the
  branch name in [CLAUDE.md](CLAUDE.md) on your branch.

## 2. What this project is

* Background: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) covers projects 001–003 and the
  mission. Also read [ROADMAP.md](ROADMAP.md), [DECISIONS.md](DECISIONS.md), and
  [EXPERIMENTS.md](EXPERIMENTS.md), which records every run.
* The latest work is **P004B**: getting and verifying the real text corpus for the tokenizer
  `indic-tokenizer/v2`. It uses public-domain works hosted on Wikisource under CC BY-SA,
  plus Project Gutenberg for English. The detail lives in these files:
  * [docs/tokenizer_corpus_handover.md](docs/tokenizer_corpus_handover.md): rules, tools,
    research recipes, pitfalls, next steps.
  * [docs/tokenizer_corpus_stage_b_acquisition.md](docs/tokenizer_corpus_stage_b_acquisition.md):
    the runbook.
  * [corpora/tokenizer/indic-tokenizer-v2/sources.json](corpora/tokenizer/indic-tokenizer-v2/sources.json):
    the manifest.
  * [corpora/tokenizer/indic-tokenizer-v2/reports/](corpora/tokenizer/indic-tokenizer-v2/reports/):
    the reports.

## 3. Where P004B stands (2026-09-25, branch tip `2c31bfb`)

| Languages | State | Run |
|---|---|---|
| English, Hindi, Bengali | locked, 40 sources | EXP-013 |
| Gujarati, Malayalam, Odia, Assamese | locked, 7 sources | EXP-017 |
| Punjabi, Kannada, Telugu, Tamil | downloaded (EXP-018), locked, 8 sources | EXP-019 |
| Marathi, Urdu | not started (see note below) | — |
| Hindi–English mixed (hi-en) | expected to stay NOT_EVALUATED | — |

Note on Urdu: its sentence marks ۔ (U+06D4) and ؟ (U+061F) must be added to
`_SENTENCE_BOUNDARY` in `src/frontier_ai/tokenization/research_corpus.py`.

All 55 sources in the manifest are verified and locked. The Arena agent checked the EXP-019
lock mechanically:
* The 47 earlier fingerprints did not change. Only `retrieved_at` moved, which `--pin`
  always does.
* The 8 new sources gained a fingerprint and `verified`.
* Nothing else in the manifest changed.

## 4. Who does what

* **Operator:** the human owner. Uses Windows, with the project at `E:\frontier-ai`, and
  works in VS Code's terminal (PowerShell).
* **Local assistant:** Claude Code in VS Code, connected through OmniRoute, on the
  operator's machine. It has owned P004B end to end since 2026-09-25, by the operator's
  decision (see [CLAUDE.md](CLAUDE.md)). It runs the downloads and pushes to
  `arena/01a0d31f-frontier-ai`.
* **Arena chat (you):** cannot download from the corpus sites (§5). Do P004B work only if
  the operator asks you to take it back. In that case, first check in git everything the
  local assistant pushed: reports, the manifest's `notes`, and the runbook's status header.

## 5. Facts about the Arena sandbox

* **It cannot reach the corpus sites** (Wikisource, Gutenberg). Real downloads and builds
  run on the operator's machine.
  * The web-page tool can read those sites for research. It strips tags even inside JSON,
    so ask the MediaWiki API for `format=jsonfm`.
  * Research recipes are in handover §6.
* **It resets between turns.** HEAD falls back to the chat's starting commit, and `.venv`
  and `data/` disappear. At the start of each turn:
  1. Run `git fetch -q origin <your-branch> && git reset -q --mixed FETCH_HEAD`. Never use
     `--hard`.
  2. Run `git status --short`. Files changed by commits made elsewhere stay stale in the
     working tree.
  3. Restore those files with `git checkout -- <file>`, after checking that none of it is
     your own unsaved work.
* **Rebuilding the environment** takes about 2 minutes:
  1. `python3 -m venv .venv && . .venv/bin/activate && pip install -q --upgrade pip && pip install -q -e ".[dev,tokenizer]"`
  2. `python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic`
* **Tests:** `python -m pytest -q -o addopts=""` gave 413 passed, 1 skipped at `e74d002`.
* **Lint:** `ruff check src tests scripts`.
* **Tool quirks:**
  * `git rev-parse --short A B` fails; pass one ref per call.
  * `gh repo view --json visibility` is unsupported; use `isPrivate`.

## 6. Working with the operator

* The operator is not fluent with command lines. Use plain language.
* Report progress as a short table (language → book → status).
* Explain words like *fingerprint* (hash) and *lock* (pin).
* Give commands in PowerShell syntax, one at a time, as exact copy-paste text. Say what
  they should see and what to send back.
* URLs typed into PowerShell run as commands. Ask the operator to open URLs in a web
  browser instead.
* PowerShell pitfalls:
  * `>` writes UTF-16, so use the scripts' `--output`.
  * Non-ASCII here-strings break.
  * Indic text shows up garbled in the console.
* Never suggest VS Code's Sync or Publish buttons. One of them once pushed this work to
  `main` as a commit with no parent.
* The operator prefers the local assistant to do the machine work, so they do not have to
  copy outputs back and forth.
* A short summary in Hinglish at the end of a reply is welcome.
* Keep the operator's personal name out of the repository's files.

## 7. Rules that never change

These are the four rules in [CLAUDE.md](CLAUDE.md), plus handover §3:
1. Never pin a fingerprint you did not compute from a download you actually performed.
2. Never mark a source verified because the manifest says its licence is CC BY-SA.
3. If a source cannot be licensed, it stays unverified. Never pad or substitute text.
4. Follow the branch rules.

## 8. Status log

Newest entry last. At the end of each working session, whoever did the work (the local
assistant or an Arena chat) adds 2–4 lines: the date, what was done, the commits, and what
comes next. They also update the table in §3.

* **2026-09-25, Arena agent, `e74d002`:** handed P004B to the local assistant (handover
  doc, CLAUDE.md).
* **2026-09-25, local side, `d477974` and `2c31bfb`:** ran JOB-002. EXP-018 downloaded all
  55 sources (build exit 0) and wrote `reports/EXP-018-inspection.txt`. EXP-019 locked all
  55.
* **2026-09-25, Arena agent:** added EXP-018 and EXP-019 entries to EXPERIMENTS.md, updated
  the status table in this file, and prepared for Marathi and Urdu research. Next: research
  Marathi and Urdu sources as described in handover §10.