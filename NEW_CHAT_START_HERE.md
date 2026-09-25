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

## 3. Where P004B stands (2026-09-26; EXP-023 lock commit `d9ddf6a`)

| Languages | State | Run |
|---|---|---|
| English, Hindi, Bengali | locked, 40 sources | EXP-013 |
| Gujarati, Malayalam, Odia, Assamese | locked, 7 sources | EXP-017 |
| Punjabi, Kannada, Telugu, Tamil | downloaded (EXP-018), locked, 8 sources | EXP-019 |
| Marathi | *स्फुट गोष्टी भाग तिसरा* by Hari Narayan Apte; locked, 1,667 documents / 256,165 characters | EXP-023 |
| Urdu | *Ram Charcha* by Munshi Premchand; locked, 1,378 documents / 208,621 characters | EXP-023 |
| Hindi–English mixed (hi-en) | `NOT_EVALUATED`; no lawful, attributable source identified | EXP-024 |

Note on Urdu: its sentence marks ۔ (U+06D4) and ؟ (U+061F) are now recognized by
`_SENTENCE_BOUNDARY`, and the `سانچہ` Template namespace is registered in the MediaWiki
cleaner. The cleaner also removes the reviewed transcription residues in *Ram Charcha*;
the retained heading and prose are covered by tests and the EXP-022 report.

All 59 sources are now verified and locked. EXP-023 confirmed that the 55 existing
fingerprints stayed unchanged; the four new fingerprints match EXP-022, and the manifest
diff contains only the permitted lock fields. EXP-021–024 and their reports are recorded in
`EXPERIMENTS.md` and the runbook. Thirteen of 14 slots are `EVALUATED`; hi-en remains
`NOT_EVALUATED` without substitute text.

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
* **2026-09-25, Arena agent:** added this file. CLAUDE.md now asks the local assistant to
  keep this log. Next: record EXP-018 and EXP-019 in EXPERIMENTS.md and update the status
  docs. After that come Marathi and Urdu (handover §10).
* **2026-09-25, local assistant, `d5c4400`:** added the EXP-018 and EXP-019 entries and
  `P004B-tokenizer-corpus-summary.html`. By mistake it also deleted the rest of
  EXPERIMENTS.md (rules, EXP-001–007, index, reproduction guide) and §9 of this file.
* **2026-09-25, Arena agent:** restored both from git and kept the two new entries (now at
  the end of EXPERIMENTS.md and in its index). Added rule 5 to CLAUDE.md: add to shared
  files, never rewrite them. Next: Marathi and Urdu (handover §10).
* **2026-09-25, local assistant:** completed EXP-020 candidate research. No Marathi or Urdu
  source passed the proofread and underlying-work licence gates, so none was declared or
  fetched. Added Urdu sentence punctuation and Template-namespace support with regression
  coverage. Next: locate stronger public-domain scans; do not pin anything from this survey.
* **2026-09-26, local assistant, commits `ce5739d`, `edd2303`, `31c2c78`, `d9ddf6a`:**
  acquired, reviewed and locked Marathi and Urdu; 59 sources are verified and pinned, with
  13 slots evaluated and hi-en honestly unevaluated. EXP-021–024 and the handover/runbook
  completion updates are recorded. P004B is complete; next, the operator may merge this
  branch into `main` when ready (no pull request was opened).

* **2026-09-26, Arena agent:** checked the finished state from git.
  * All 59 sources are verified and pinned. The 55 earlier fingerprints are unchanged.
  * 13 of 14 slots are `EVALUATED`; hi-en is not.
  * In the Arena sandbox (Linux), the full test suite gives 415 passed, 1 skipped, and Ruff
    is clean. So the 22 failures that EXP-024 reports come from the Windows machine's setup,
    as EXP-024 itself says.
  * Next: the operator decides whether to merge this branch into `main`.

## 9. The prompt the operator pastes into a new Arena chat

```
We are continuing my project frontier-ai from an earlier Arena chat that you cannot see. Everything from that chat is saved on GitHub, on the branch arena/01a0d31f-frontier-ai. Before anything else:
1. Run: git fetch origin arena/01a0d31f-frontier-ai
   Then run: git merge --ff-only FETCH_HEAD
   If the merge is refused, run git merge --no-edit FETCH_HEAD and tell me. If the fetch says the branch does not exist, it has already been merged into main and you have everything.
2. Read NEW_CHAT_START_HERE.md completely and follow it. It links to everything else.
3. Then tell me in simple words where the project stands and what the next step is. Do not start any new work until I say so.
```
