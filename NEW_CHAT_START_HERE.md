# New chat? Start here

> **2026-09-26:** P004B is complete (EXP-024) and is in PR #2.
> * The founder's master context is [MASTER_CONTEXT.md](MASTER_CONTEXT.md).
> * The verified project state is in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md), section
>   "CURRENT POSITION".
> * The P004B detail below is kept as history.

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

* The founder's master context: [MASTER_CONTEXT.md](MASTER_CONTEXT.md), covering the
  mission, working rules, roadmap and the required answer format (§39).
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

## 3. Where P004B stands (2026-09-26; EXP-026 freshness check; EXP-023 lock commit `d9ddf6a`)

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

EXP-026 live freshness re-fetch exited 0: 59 sources verified and pinned, with no
`REFUSED:` lines or inspection flags. The freeze record now reflects this network-enabled
verification; see `FREEZE.json` and `reports/EXP-026-refetch.txt`.

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
* **2026-09-26, Arena agent:** saved the founder's [MASTER_CONTEXT.md](MASTER_CONTEXT.md).
  Synced PROJECT_CONTEXT.md (CURRENT POSITION, §7, §10, §17), ROADMAP.md, README.md and
  CLAUDE.md with the repository; this is MASTER_CONTEXT §37 step 1. Next: the founder
  merges PR #2, then step 2, freezing and verifying `indic-tokenizer/v2`.
* **2026-09-26, Arena agent, branch `arena/01a0dc16-frontier-ai`:** ran EXP-026 live
  freshness check: exit 0, 59/59 verified and pinned, no `REFUSED:` lines; report and freeze
  update committed as `989b669`. Next: Arena agent starts STEP 3 plan.
* **2026-09-26, Arena agent, branch `arena/01a0dc16-frontier-ai`:** MASTER_CONTEXT §37
  step 2 done, after the founder merged PR #2 (main is now `76bb127`). Verified the
  finished state from git: 59/59 pins present and well-formed, every manifest hash
  matching its EXP-023 report prefix, coverage totals exactly 34,684 documents /
  4,211,707 characters, offline build check exit 0, 415 passed / 1 skipped, ruff clean.
  Added `corpora/tokenizer/indic-tokenizer-v2/FREEZE.json` (manifest sha256
  `aec3dfa0…`), D-035 (frozen; any change means `v3` + founder approval) and EXP-025.
  Not yet verified from the sandbox: a live re-fetch (no route to the hosts). Next:
  STEP 3–4 — the FrontierCorpus v1 pipeline; audit first, smallest useful step, founder
  approval before code.
* **2026-09-26, Arena agent, branch `arena/01a0dc16-frontier-ai`:** MASTER_CONTEXT §37
  step 3 built (the founder directed "finish up the upcoming tasks quickly", then
  ratified the STEP 3 plan: "Arena agent starts the STEP 3 plan" — D-036 NFC and D-037
  staged pipeline are **accepted**): `src/frontier_ai/corpus/` — normalize (NFC,
  policy v1), langid (script-profile gate), quality (7 measurable rules, per-language
  overrides), exact dedup (cross-source, order-independent), split (reused P004B
  content-hash rule), seeded shuffle, character-budget shards (per-shard SHA-256),
  dataset manifest (`content_sha256` excludes the timestamp);
  `scripts/build_frontier_corpus.py` — self-recording (D-032), `--check`, exit 0/1/2,
  and a hard gate: refuses unless the freeze identity and all 59 pinned text hashes
  verify. 46 new unit tests incl. an offline e2e of the builder on a fake frozen
  corpus; full suite 458 passed / 4 skipped (environmental), ruff clean. Plan doc:
  `docs/frontier_corpus_v1.md`. EXP-026 was then verified from the repository (59/59
  pins unchanged, 396 identical documents exactly matching EXP-023) and merged with
  this work (merge `6dafd07`), then pushed. Not yet run on the real corpus (needs its
  text — a PC job). Next: EXP-027 pilot on the PC (`python scripts/build_frontier_corpus.py
  --exp-id EXP-027`) → cross-check dedup removals against the 396 identical documents.
* **2026-09-26, Arena agent, `dbc67be`, `6a8788d`:** fixed Windows CRLF hash validation,
  then built the FrontierCorpus v1 pilot from the frozen corpus; build and `--check` passed.
  Manifest/report and EXP-027 metrics recorded; shards remain git-ignored. Next: review the
  pilot results and proceed to the next approved FrontierCorpus step.
* **2026-09-26, Arena agent:** verified EXP-027 from the repository. Stage math matches the
  manifest (34,684 in → 34,011 post-stages; train 30,584 / held_out 3,427; shard chars
  = doc chars + joining newlines, checked exactly); the 339 dedup removals reconcile with
  the 396 baseline per language (every language's deficit ≤ its langid removals; the PC's
  399 NFC-equivalents − 60 langid-removed duplicates is internally consistent and
  recomputable on the PC); reviewed the `dbc67be` CRLF fix against the v2 pin derivation
  (`research_corpus` hashes the decoded text, not file bytes) — correct, not a weakening,
  and the refusal tests still pass. Suite 459 passed / 4 skipped, ruff clean. Next:
  MASTER_CONTEXT §37 step 5 — the production tokenizer sweep (plan needs founder approval).
* **2026-09-26, Arena agent:** the founder approved EXP-A ("approve EXP-A"). Delivered the
  sweep harness (commit `7f13a17` + this session's harness commit): `bpe_python` now has an
  **incremental BPE loop** (provably identical merges to the old loop — the old loop was
  infeasible at corpus scale: 2048 vocab on 400k chars did not finish in 20 min) and a new
  `gpt2_style` pre-tokenizer (GPT-2's regex, where Indic syllables shatter at their
  combining marks; the default `mark_aware` is unchanged). New: `src/frontier_ai/corpus/
  frontier_docs.py` (shared derivation, builder refactored onto it, output byte-identical),
  `src/frontier_ai/tokenization/sweep.py` (15-config grid, losslessness gate, per-language
  scoring), `scripts/run_tokenizer_sweep.py` (self-recording, exit 0/1/2, smoke mode).
  D-038 records the design (15 cells, model-free, re-derived input verified against the
  frozen shards). Suite 479 passed / 1 skipped, ruff clean. **Next: run the sweep on the
  PC** (the corpus text + shards are PC-only). PC handover:
  1. In `E:\frontier-ai` (PowerShell): `git fetch origin arena/01a0dc16-frontier-ai` then
     `git switch arena/01a0dc16-frontier-ai` (if not already there) then
     `git pull --ff-only origin arena/01a0dc16-frontier-ai`.
  2. Rebuild the venv (it must have the tokenizer extra): `python -m venv .venv` then
     `.\.venv\Scripts\pip install -q --upgrade pip` then
     `.\.venv\Scripts\pip install -q -e ".[dev,tokenizer]"` then
     `.\.venv\Scripts\python scripts\prepare_data.py --source synthetic`.
  3. Sanity (fast, ~4 min): `.\.venv\Scripts\python -m pytest -q` → expect
     **479 or 480 passed, 0 failed** (the count depends only on which optional packages
     the venv has; on the PC the git-tracking hygiene test also passes);
     `.\.venv\Scripts\python -m ruff check .` → "All checks passed!".
  4. The sweep (background-friendly, tens of minutes to a couple of hours, CPU only):
     `.\.venv\Scripts\python scripts\run_tokenizer_sweep.py --exp-id EXP-028`.
     Expected: input-verified lines, then one `gate=PASS` line per configuration (15),
     then a results table; exit code 0. Full report:
     `E:\frontier-ai\out\experiments\EXP-028\report.txt`.
  5. Stop condition: if the exit code is **2**, the frozen input did not verify — STOP, do
     not edit the corpus, report the `[sweep] INPUT GATE FAILED:` lines here. If **1**, a
     configuration failed — report the failing lines from `sweep.json`. If **0**, paste the
     two tables from `report.txt`. **Do not choose a tokenizer** — the top-2 go to EXP-B.
  6. Commit rule: the run writes only under `out\experiments\EXP-028\` (git-ignored).
     Nothing to commit unless the run says so.
* **2026-09-26, Arena agent (EXP-028 in flight):** the harness push had failed on expired
  auth and a sandbox reset had lost the commit; the same content was re-committed and
  pushed as `580d797` (the PC pulled it and re-ran the sweep). As of this entry the PC is
  mid-run: 12/15 cells done, **all gate=PASS**; the three remaining cells are the
  `py-mark_aware` ones (the longest — 30–60 min for 32768 from the PC's own timings).
  In parallel (founder directive: keep working while the PC job runs) three things were
  delivered and are in this commit: (1) `scripts/summarize_sweep.py` — a read-only
  verifier + ranked summarizer for a finished sweep (consistency checks per run, ranked
  table, per-language matrix, TOP-2-for-EXP-B; exit 0/1/2; multiple `--sweep-dir` to
  merge runs); (2) the previously out-of-grid 5 `bpe_hf` + mark-aware cells are now
  **implemented** (D-039): `bpe_hf.MarkAwarePreTokenizer` via the `tokenizers` custom
  pre-tokenizer API, boundaries identical to `bpe_python`'s, save/load via a
  serializable placeholder + re-attach; runner flags `--include-hf-mark-aware` /
  `--configs` let them run into their own out dir after the main sweep; sandbox
  cross-validation shows `hf-mark_aware` ≡ `py-mark_aware` on a fake corpus; (3) no EXP-B
  code yet — that plan comes after the sweep numbers, per the approved scope.
  **Next (PC, after the sweep finishes):** founder pastes the final table + exit code;
  then `git pull --ff-only`, then the supplement run
  (`python scripts/run_tokenizer_sweep.py --exp-id EXP-028 --out
  out/experiments/EXP-028-hf-mark-aware --configs hf-mark_aware-2048,hf-mark_aware-4096,
  hf-mark_aware-8192,hf-mark_aware-16384,hf-mark_aware-32768`), then
  `python scripts/summarize_sweep.py --sweep-dir out/experiments/EXP-028
  out/experiments/EXP-028-hf-mark-aware`; its TOP-2 section hands off to EXP-B.
* **2026-09-26, Arena agent (EXP-028 COMPLETE, commits `c11ba8f`, `3bae620`):** PC ran
  the full 15-cell sweep (all gate=PASS) and the 5-cell supplement (all gate=PASS;
  the runner's `--configs` auto-include fix in `3bae620` made the documented command
  work). Verified merged summary (20 runs): **mark_aware-32768 = 2.3054 held-out
  chars/token, +45 % over the GPT-2/ByteLevel baselines (1.5971)**, consistent at all
  vocab sizes; mark_aware-32768 wins 11 of 13 languages (baselines win `en` and
  unvocalized `ur`). Key validation: `hf-mark_aware-X` ≡ `py-mark_aware-X` — identical
  metrics at all 5 vocabs and 13 languages across two independent BPE implementations,
  so the ranked top-2 is one tokenizer in two implementations; the informative EXP-B
  pair is therefore `mark_aware-32768` vs `mark_aware-16384`. Results + findings
  recorded in EXPERIMENTS.md (EXP-028). **Next: founder approves the EXP-B plan
  (top-2 interpretation + pre-registered decision rule); no EXP-B code before
  approval.** Note: the sandbox suffered two more fresh-reclone resets today; both
  recovered via fetch + `git reset --mixed FETCH_HEAD` (tree intact; re-committed as
  `c11ba8f`).

* **2026-09-27, Arena agent (EXP-029 / EXP-B COMPLETE → D-040, commits `ce1b96f`,
  `c91ce6e`, `454bd28`):** two runner bugs were found on the PC and fixed (recorded-mode
  cells ran nested and skipped their records → `ce1b96f` + regression test; Windows
  PowerShell 5.1 treated the stderr `[record] … fingerprint` line as a fatal
  `NativeCommandError` → `c91ce6e`, cmd.exe merges stderr). The PC's local agent then
  ran the matrix unattended from `PC_TASK_EXP_B.md` (the file-based handoff that
  replaces copy-paste). Result at 150 steps/cell, 3 seeds: **mark_aware-32768 mean
  held-out bpb 1.4463 vs mark_aware-16384 1.5779** (gap 8.8× the tie band; every
  32768 seed beats every 16384 seed); 4/4 file-level checks PASS. Founder approved →
  **D-040: Frontier Tokenizer v1 = mark_aware-32768** (MASTER_CONTEXT §37 step 7 done).
  **Next (§37):** freeze the selected tokenizer artifact (PC-only, git-ignored) into a
  durable hashed location, then step 8 — the evaluation harness. Plan first; no code
  before founder approval.

* **2026-09-27, Arena agent (EXP-030 COMPLETE → D-041, commits `d103987`, `fc7e8d8`):**
  Frontier Tokenizer v1 frozen at `tokenizers/frontier-tokenizer-v1/` (PC gates A–D
  PASS: EXP-029 fingerprint, structure, exact token counts 1,608,987 / 184,233,
  34,011 docs lossless). Linux reproduces the raw-byte hash and all Windows-recorded
  golden ids. **Always load it with `load_frontier_tokenizer()`**; `tokenizers/**` is
  stored byte-exactly (`.gitattributes`). PC handoffs now go through `PC_TASK_*.md`
  files — put the `git pull` in the chat message itself, not only inside the file.
  **Next (§37 step 8):** plan the evaluation harness; no code before founder approval.

* **2026-09-27, Arena agent (§37 step 8 IN PROGRESS — evaluation harness v1 + ROADMAP
  sync; EXP-031):** founder approved the plan. Harness v1 built and tested in the sandbox
  (`scripts/eval_report.py`, `eval_compare.py`, `build_eval_suite.py`,
  `publish_eval_results.py`; `src/frontier_ai/evaluation/`; 13 tests; full suite 514
  passed / 1 skipped). ROADMAP.md now carries a dated 24-step status block (steps 1–7
  done, step 8 current, 9–24 not started; FrontierCorpus v1 = pilot; tokenizer v1 frozen,
  v2 possible; GPU needed from step 10). PC run pending via `PC_TASK_EVAL_EXP031.md`.
  **Next:** record EXP-031 results + D-042, report the step 8 exit criteria, then STOP
  for founder approval before step 9.

* **2026-09-27, Arena agent (EXP-031 COMPLETE → D-042; step 8 exit criteria met, commits
  `6e5df7b`, `ffbd4e6`, PC `0fa5b22`):** harness v1 scored all 6 EXP-B models on the
  protected suite `frontier-heldout-v1` (every held-out token; data identity PASS ×6;
  reruns and the EXP-028 artifact reproduce `scores_sha256` exactly; 0 contamination).
  The exact paired result confirms D-040: 32768 at 1.4475 vs 16384 at 1.5785 bpb, delta +0.1310
  [+0.1280, +0.1339], better in all 13 languages. **Rules now:** compare models only
  through `eval_report.py` / `eval_compare.py`; never train on `frontier-heldout-v1` (check
  with `find_exact_overlap`). Lessons: Windows consoles are cp1252, so every eval script now calls
  `make_console_safe()`; this sandbox keeps resetting its branch pointer and deleting `.venv`,
  so recover with `git fetch` + `git reset --mixed FETCH_HEAD` + `git checkout -- <paths>`.
  **Next:** STOP. Step 9 (architecture ablations) needs explicit founder approval of a plan.

* **2026-09-27, Arena agent (step 8 APPROVED by the founder; step 9 IN PROGRESS: EXP-032):**
  **North star (founder, 2026-09-27):** every step must move towards a model built from
  scratch that can compete with the leading systems (e.g. ChatGPT, Claude) and becomes the
  best model in India (MASTER_CONTEXT §1). Each report should say how the step serves that
  goal and state honestly how far we still are. The founder approved the step 9 plan. Built
  `scripts/run_arch_ablation.py` + `run_arch_ablation_night.ps1` + the pre-registered spec
  `configs/ablations/EXP-032.json` (rope / gelu / layernorm / gqa2 vs baseline, lr check,
  repro check; the FINAL checkpoint is graded because `best/` is selected on the suite).
  13 new tests. The founder's roadmap attachment has not reached the sandbox twice
  (`/home/user/uploads` absent); ask him to paste its text instead.
  **Next:** the PC night run via `PC_TASK_ARCH_EXP032.md`, then EXP-032 results and a
  proposed D-043. STOP for review.

* **2026-09-28, Arena agent (EXP-032 COMPLETE; PC commit `4027c98`):** step 9 screening ran
  overnight on the founder's i3-4030U (2 cores; ~7–11 s/step; no GPU needed). The baseline was
  reused from EXP-029 and reproduces EXP-031 exactly; the repro retrain has IDENTICAL weights.
  At lr 3e-3: **RoPE BETTER (−0.077 bpb, −5.3%)**, GELU BETTER (−0.034; contradicts the
  literature, so suspect), LayerNorm no detectable difference, **GQA-2 ACCEPTABLE** (half the
  KV cache). But the **learning-rate check is LR-CONFOUNDED** (lr 6e-3 improves the baseline by
  0.057), so nothing is adopted yet and D-043 is deferred. Spec hashes are now LF-normalised
  (CRLF checkout on Windows changed the raw hash). **Next:** propose EXP-033 (confirmation at
  better learning rates); STOP for founder approval.

* **2026-09-28, Arena agent (EXP-033 approved; infrastructure built):** the founder approved
  EXP-033 ("I approve EXP-033") and said his PC agent (Copilot) is about to reach its usage
  limit. What was built:
  - `configs/ablations/EXP-033.json`: schema v2, pre-registered. It tests baseline vs
    rope-gqa2 vs rope-gqa2-gelu at lr 6e-3 and 1e-2, over 3 seeds. A change is adopted only if
    it is BETTER at both learning rates. A DIVERGED seed counts as the worst score.
  - `run_arch_ablation.py` v2 support; the v1 path is unchanged.
  - **`scripts/run_night_unattended.ps1`**: a one-command PC runner that needs no agent. It
    checks, runs a preflight, runs the ablation with one retry, publishes, makes a scoped
    commit and push, and writes `NIGHT_REPORT.txt`, which it opens in Notepad.
  - `PC_TASK_ARCH_EXP033.md` and 21 tests.

  The founder types one line in the VS Code terminal (see the task file). **Next:** analyse
  the EXP-033 results, propose D-043 and the step 9 exit report, then STOP for approval
  before step 10.

* **2026-09-28, Arena agent (EXP-033 COMPLETE; PC commit `5d7a0c7`):** the founder ran the
  agent-free script with one typed line. It passed every check and pushed exactly 82 result
  files on its own (5 h 46 min). The reused EXP-032 cell re-grades identically, and there were
  no divergences or failures. Findings:
  - RoPE + GQA-2 vs baseline: −0.049 at lr 6e-3 (NO DETECTABLE DIFFERENCE, because one
    baseline seed overlaps) and −0.066 at lr 1e-2 (BETTER) ⇒ **NOT ADOPTED** by the
    pre-registered both-LR rule.
  - GELU vs SwiGLU: the sign flips (−0.012 / +0.018) ⇒ NOT ADOPTED. The EXP-032 GELU win did
    not replicate.

  The step 9 exit summary is in EXPERIMENTS (EXP-033 Results). **Proposed D-043:** keep the
  EXP-B baseline as the provisional architecture for GPU bring-up. RoPE + GQA-2 is the first
  candidate for the step 11 GPU-scale ablation. **Next:** founder reviews D-043; after
  approval, record D-043, update ROADMAP (step 9 complete), then propose the step 10 GPU
  bring-up plan with costs. STOP until approved.

* **2026-09-28, Arena agent (GPU collaborator onboarding):** the founder's GPU (an RTX card,
  model NOT VERIFIED) is on a **friend's computer**, and the friend will join the GitHub
  repository. The founder asked to update GitHub for him. Added:
  - `GPU_COLLABORATOR_START_HERE.md`: project status, working rules, Windows setup with CUDA
    PyTorch, the readiness report, and what happens next (including the git-ignored data he
    will receive with recorded hashes).
  - `scripts/gpu_env_report.py`: a read-only report on machine, PyTorch and GPU facts, plus
    seconds-long GPU smoke checks. It writes `out/gpu_env/REPORT.txt` and exits 0 / 1 / 2 for
    ready / not ready / error.
  - 3 tests.
  - Targeted README updates (status table, collaborator pointer, stale facts fixed) and a
    ROADMAP 2026-09-28 update.

  Notes:
  - The repository is **public**.
  - The agent's GitHub token cannot add collaborators, so the founder invites the friend
    himself.
  - D-043 is still pending founder review.

  **Next:** the friend's `REPORT.txt`, then a step-10 plan written for that GPU (founder
  approval), then a data zip plus a hash-verification command.
* **2026-09-29, Arena agent (D-043 accepted; data scale-up phase 1 started):** the founder
  wrote "approve D-043 and data plan".
  - D-043 is recorded in DECISIONS.md as accepted, so step 9 is complete (commit `97ad349`).
    ROADMAP has a 2026-09-29 update.
  - New `docs/frontier_corpus_v2_sources.md`: a source survey with license evidence
    (Wikipedia/Wikisource CC BY-SA 4.0 + GFDL, IndicCorp v2 CC0 per card, Sangraha
    cc-by-4.0 tag, FineWeb-2 ODC-By + Common Crawl terms, HPLT license NOT VERIFIED),
    per-language Indic Wikipedia sizes (≈552M words, not tokens), the India legal context
    (DPIIT working paper Dec 2025; ANI v. OpenAI interim ruling 24 Jul 2026), the
    protected-suite rule for new sources, the pipeline gaps, and **D-044 options**
    (1 open only / 2 open + curated collections, recommended / 3 everything).
  - Nothing has been downloaded.

  **Next:** the founder chooses a D-044 option. Then: the streaming interface, MinHash
  near-dup, and the inverted suite-contamination check in the sandbox (tests); then a PC
  overnight run for the level-A v2-pilot.
  - Same day: the protected-suite guard was built first, because it is needed under every
    D-044 option. It lives in `src/frontier_ai/corpus/decontaminate.py`, with 9 tests in
    `tests/test_suite_decontaminate.py`. It streams training documents past the suite's
    13-grams with bounded memory, and removals are recorded as `suite_exact` / `suite_ngram`.
    Still to build: the streaming interface for the other stages and MinHash near-dup.
  - **Founder decided D-044 (2026-09-29): option 2, starting with Sangraha Verified**
    (CC-BY-4.0), "the best, not compromising". Recorded in DECISIONS.md as accepted.
    - Pins: `corpora/frontier/v2/sangraha_slice1.json` (13 files `verified/<lang>/data-0.parquet`,
      5,106,130,219 bytes, revision `8b813c3f…`, size + SHA-256 each).
    - Code: `corpus/sangraha.py` (pins, resumable verified download, parquet streaming),
      `corpus/slice_inspect.py` (measures, filters nothing), `guard_from_heldout_shards` in
      `corpus/decontaminate.py`, `scripts/fetch_sangraha_slice.py`,
      `scripts/inspect_sangraha_slice.py`, PC runner `scripts/run_data_night.ps1`; tests in
      `tests/test_sangraha_slice.py`. Optional extra `data` (pyarrow).
    - **EXP-034** pre-registered in EXPERIMENTS.md. PC line:
      `git pull --ff-only origin arena/01a0dc16-frontier-ai; powershell -ExecutionPolicy Bypass -File scripts\run_data_night.ps1`
      Results come back in `out\data\EXP-034\NIGHT_REPORT.txt` and `evals/results/EXP-034/`.
    - Next after EXP-034: pre-register v2 build thresholds from the numbers; MinHash near-dup
      (memory-bounded) and PII scrub; build; exact token count.
  - **Founder approved "lab dashboard option B" (2026-09-29): Lab OS at `apps/lab-os/`.** It is a
    read-only dashboard of this repository (UI shell from his `frontier-labs-os` prototype, mock
    data removed). It is secondary to the model work.
    - Data: `apps/lab-os/src/data/lab_state.json`, written by `scripts/export_lab_state.py`
      (logic `src/frontier_ai/lab_state.py`). The only hand-kept input is `lab/registry.json`
      (step status, current work, baseline, compute, documents). The exporter rejects it if it
      contradicts EXPERIMENTS.md / DECISIONS.md / MASTER_CONTEXT §37.
    - **After changing EXPERIMENTS.md, DECISIONS.md, `lab/registry.json` or `evals/results/`, run
      `python scripts/export_lab_state.py` and commit the snapshot.** `tests/test_lab_state.py`
      fails while it is stale. When a roadmap step finishes, update its status in `lab/registry.json`.
    - Run: `cd apps/lab-os && npm ci && npm run dev`. See `apps/lab-os/README.md`.
  - **EXP-034 complete (PC night 2026-09-29, results commit `484b8a7`; analysis appended to
    EXPERIMENTS.md).** All 13 files verified; fully inspected; suite CHECKED. 2,332,236 docs,
    5,482.5M chars, ≈2,914M tokens (ESTIMATE). Nothing filtered. Key findings: 20 suite hits
    (same Wikisource books as the held-out suite); 1,882 short suite docs are only exact-match
    protected; the script gate is not a language check (Uyghur passed in the ur file); OCR/PDF
    = 21.5% of chars; `max_chars` 20k would drop long docs. **Next: EXP-035 (calibration:
    sample read + extra measurements, removes nothing) is PROPOSED in chat — founder approval
    needed before any code.** Dashboard snapshot refreshed (Sangraha table shows the inspection).
  - **EXP-035 approved (founder: "approve EXP-035", 2026-09-29) and built; PC night pending.**
    Calibration pass that removes nothing: `scripts/calibrate_sangraha_slice.py`
    (`corpus/slice_calibrate.py`), new `ShortSuiteIndex` + `heldout_pairs` in
    `corpus/decontaminate.py` (short suite docs contained word for word; not yet in the build
    guard). PC line: `git pull --ff-only origin arena/01a0dc16-frontier-ai; powershell
    -ExecutionPolicy Bypass -File scripts\run_calibration_night.ps1` (wrapper for
    `run_data_night.ps1 -Exp EXP-035 -Task calibrate`). Results: `evals/results/EXP-035/`
    (`summary.json`, `SUMMARY.txt`, `samples.jsonl`: masked excerpts, suite-touching docs never
    sampled). **Next: read the samples, then propose EXP-036 (v2 build rules, each with its
    reason) — founder approval needed before any build code.**
  - **EXP-035 complete (PC night 2026-09-29, 2 h 36 min, results commit `1be3c48`; analysis +
    EXP-036 proposal appended to EXPERIMENTS.md).** 13/13 files verified and read in full,
    short-suite CHECKED, nothing removed. Key findings: long docs (> 20k chars) are mostly books
    and debates, so `max_chars` must go; about 8.4% of the Urdu file is Persian (no ٹ ڈ ڑ ں ے ھ; 6/6
    samples); code-mixed rejects are mostly native text plus whole English lines; `digit_runs`
    and `url_density` hit good text; hi near-dups 1.86%; Marathi NFC = र+nukta → ऱ (safe);
    45 docs contain a 3–12-word suite passage; Sangraha `doc_id` is not unique. Runner fix:
    the final report line now uses `-Exp` (it said "EXP-034 done"). **Next: EXP-036 (v2
    build, 10 rules in the table at the end of EXP-035) is PROPOSED — founder must type
    "approve EXP-036" before any build code.**
  - **EXP-036 approved (founder: "approve EXP-036", 2026-09-29) and built; PC night pending.**
    The v2 build with the 10 rules from the EXP-035 table: `scripts/build_sangraha_v2.py`
    (`corpus/slice_build.py`); every removed document/line has a reason; provenance per
    document (`<source_id>#<row>`); exact token count with the frozen tokenizer; suite check on
    the exact output text (must be 0); near-duplicates last among survivors. Corpus →
    `data\frontier_v2\sangraha-slice1-v2\` on the PC (git-ignored, resumable per file).
    PC line: `git pull --ff-only origin arena/01a0dc16-frontier-ai; powershell -ExecutionPolicy
    Bypass -File scripts\run_build_night.ps1` (wrapper for `run_data_night.ps1 -Exp EXP-036
    -Task build`). Results: `evals/results/EXP-036/` (`summary.json`, `SUMMARY.txt`,
    `manifest.json`, `samples.jsonl`). Runtime NOT VERIFIED (estimate 3–6 h). **Next: read the
    per-rule costs, REVIEW lines and samples; report to the founder, who decides whether v2 is
    accepted.**
  - **EXP-036 complete (PC night 2026-09-29/30, 3 h 43 min, results commit `47eae55`; analysis
    appended to EXPERIMENTS.md).** 13/13 files built, suite CHECKED, **0 suite hits in the
    output**; 2,298,196 of 2,332,236 docs kept (96.54% of characters); **exact token count
    2,796,048,213 for this slice only** (frozen tokenizer; EXP-034's estimate was within about
    0.6%). The one REVIEW line (Assamese `empty_after_cleaning` 6.50%) is a label shift: the
    non-Indian Latin-script Bible texts in the Assamese file lose every line to the foreign-line
    rule before the script gate sees them. Small known losses (quoted Sanskrit verses, documents
    in the wrong language file) are listed for a later experiment. **Next: the founder decides
    whether v2-slice1 is accepted (proposed D-045: type "approve D-045"). Only after that: record
    D-045 in DECISIONS.md. No training on v2 without its own approved plan.**
  - **D-045 accepted (founder: "approve D-045", 2026-09-30).** FrontierCorpus v2-slice1 =
    the EXP-036 build (2,298,196 docs, 2,796,048,213 tokens, 0 suite hits); identity =
    `evals/results/EXP-036/manifest.json` (sha256 `73487435…`). Text on the founder's laptop
    only; rebuildable byte-identically from the pinned inputs. Recorded in DECISIONS.md, ROADMAP
    update block, `lab/registry.json`. **Next (critical path): step 10 GPU bring-up plan. It
    needs the GPU collaborator's `out\gpu_env\REPORT.txt` (GPU_COLLABORATOR_START_HERE.md §3,
    `scripts/gpu_env_report.py`); the founder was asked to get it. The plan must say how v2
    reaches the GPU machine (rebuild from pinned inputs there vs copy) and needs founder
    approval before any GPU run.**
  - **EXP-037 proposed (2026-09-30, while the GPU report is awaited; end of EXPERIMENTS.md).**
    Part 1: Frontier Tokenizer v2 = v1 merges unchanged + `<|endoftext|>` (32768), `<|pad|>`
    (32769), 126 reserved ids → vocab 32,896; ordinary text must encode exactly as v1; special
    ids never created from raw text (current `iter_segments` would match them — must be
    bypassed). Part 2: PC night packing v2-slice1 → per-language `.bin` (uint16, one
    `<|endoftext|>` per doc) with a 0.5% validation split by sha256(record id); check: train +
    val − docs = EXP-036 exact tokens. **Founder must type "approve EXP-037" before any code.**
  - **EXP-037 approved ("approve EXP-037", 2026-09-30) and built.** Tokenizer v2 frozen in
    `tokenizers/frontier-tokenizer-v2/` (gates pass; loader `load_frontier_tokenizer_v2()`;
    training text must use `encode_ordinary`, never `encode`). Packer
    `src/frontier_ai/corpus/pack.py` + `scripts/pack_sangraha_v2.py`; PC runner
    `scripts/run_pack_night.ps1` (no download; token files go to
    `data\frontier_v2\sangraha-slice1-v2-tok2\`, only `evals/results/EXP-037/` is committed).
    **Next: the founder runs `powershell -ExecutionPolicy Bypass -File scripts\run_pack_night.ps1`
    overnight and pastes NIGHT_REPORT.txt; then record the result and propose D-046.**
  - **EXP-037 complete (PC night 2026-09-30, 2 h 35 min, `3925cd7`; recorded in EXPERIMENTS.md).**
    All pre-registered checks pass: 2,783,830,088 train + 14,516,321 validation tokens
    (11,660 validation docs), train + val − docs = 2,796,048,213 = EXP-036; token files
    (5.6 GB) on the founder's laptop only, identity = `evals/results/EXP-037/manifest.json`.
    **Next: founder decides D-046 ("approve D-046": tokenizer v2 for all v2 training, packed
    files canonical, validation split never trained on). Then step 10 GPU plan (still waits for
    the GPU collaborator's REPORT.txt).**
  - **D-046 accepted ("approve D-046", 2026-09-30; DECISIONS.md, ROADMAP update block).**
    Tokenizer v2 for all v2 training; EXP-037 packed files canonical; validation split never
    trained on. The founder asked for free GPU options, costs and an honest ChatGPT comparison:
    answered in `docs/compute_options_and_costs.md` (Kaggle free 2×T4 ~30 h/week is enough for
    step 10–11 sized runs; ≥ 1 B params needs money/grants and more data; ChatGPT class is out of
    reach for a personal budget). **Next: the founder creates a Kaggle account and verifies his
    phone (free GPU access); then I write the step-10 bring-up plan for Kaggle (and/or the
    collaborator's GPU) for approval. No GPU run before that approval.**
  - **EXP-038 proposed (2026-09-30; end of EXPERIMENTS.md): step 10 GPU bring-up on one free
    Kaggle T4.** The founder logged into Kaggle (phone verification still needed for GPUs).
    Launch/fetch from the laptop via the Kaggle API (token stays on his PC); data = EXP-037
    `hi.bin` only (private Kaggle dataset, sha256-checked); Parts: 0 env, 1 correctness with
    pre-registered tolerances, 2 speed/memory/MFU for S/M/L bring-up sizes, 3 first real-data
    run (M on Hindi, ≤ 90 min). Cap 6 GPU-h. **Founder must type "approve EXP-038" before any
    code.** Kaggle API details are NOT VERIFIED until the first run.
  - **Founder verified his Kaggle phone (2026-09-30)** and asked how GPUs work. He was given a
    5-minute read-only GPU check (new notebook → GPU T4 → `!nvidia-smi`, then stop the session);
    not a project run. EXP-038 still needs "approve EXP-038" before any code.
  - **EXP-038 approved ("approve EXP-038", 2026-09-30) and implemented.** Trainer fix: resume now
    restores the batch generator and fp16 scaler (`trainer_state.pt`), so a resumed run draws the
    same batches (the new test failed before the fix). Built `scripts/gpu_bringup.py` (Parts 0–3,
    `--smoke` on CPU), `scripts/kaggle/exp038_kernel.py` and `scripts/run_kaggle_exp038.ps1`
    (one typed line: hidden one-time Kaggle token prompt stored in the Windows user env, private
    dataset with `hi.bin`, private one-T4 kernel at the pinned pushed commit, waits, commits only
    `evals/results/EXP-038/`). **No GPU result yet. Next: the founder creates a Kaggle API token
    (Kaggle → Settings → API; never pasted into chat) and runs the line; then I record EXP-038
    from the committed results.**
  - **EXP-038 complete (Kaggle T4, 2026-09-30; results `740fc16`, recorded 2026-10-01).** The
    first GPU training run: 74.1 min, commit `57a53ef`. fp16, resume and compile checks PASS;
    **CPU = GPU FAILED** (1.81e-3 > 1e-3). fp16 + compile: 69,127 tokens/s at 32 M (22% MFU),
    17,870 at 139 M (26%). A 32 M Hindi model reached 1.923 nats/token (0.586 bits per byte) in
    48.6 min; its samples are fluent-looking, factually wrong, and support no quality claim. The
    runner's push was rejected once because I pushed a doc during the run; the founder ran
    `git pull --rebase` and pushed. **EXP-039 proposed** (float64 CPU vs GPU, ≤ 1e-8, plus a
    runner push fix). **Next: the founder types "approve EXP-039"; no code before that.** The
    friend's GPU (RTX 3050, 4 GB) awaits his `REPORT.txt`.
  - **EXP-039 approved ("approve EXP-039", 2026-10-01) and built.** `gpu_bringup.py --part
    cpu-gpu-diagnostic` (exact per-step losses via a forward hook, because the log keeps 5
    decimals; float64 Part A ≤ 1e-8, float32 Part B numbers only). The model now keeps float64 in
    RMSNorm and the loss when it is float64 (other precisions unchanged). Runner:
    `scripts/run_kaggle_exp039.ps1` (same dataset, remembered username, push retry with
    `pull --rebase`). **Next: the founder runs `git pull` then that line; then I record EXP-039
    and, if Part A passes, propose D-047 to close step 10.**
  - **EXP-039 complete (Kaggle T4, 2026-10-01; results `d038f2a`).** Part A float64 CPU vs GPU
    PASS: 5.51e-10 ≤ 1e-8 (step 1: 1.8e-15; growth from step 39). Part B float32 1.807e-3
    (reproduces EXP-038; growth from step 40). The push retry was not needed. **D-047 proposed**
    (DECISIONS.md): step 10 complete for single-GPU training; float64 device check for every
    new GPU; fp16 + compile T4 default; EXP-038 throughput for step-11 planning. **Next: the
    founder types "approve D-047"; then I write the step-11 plan for approval (no training
    before).**
  - **D-047 accepted ("approve D-047", 2026-10-01): step 10 complete.** **EXP-040 proposed**
    (end of EXPERIMENTS.md): step 11 phase 1 on one Kaggle T4, all 13 languages (natural
    proportion, not a chosen mix), EXP-038 M shape, baseline vs RoPE + GQA-2 × learning rates
    {5e-4, 1e-3, 2e-3, 4e-3}, plus seed 2 at the 2 best baseline learning rates; equal-weight mean
    bits per byte over 13 languages; adoption only if BETTER at both learning rates. A 5.6 GB
    private dataset upload is needed. **Next: "approve EXP-040"; no code before that.**

  - **EXP-040 approved ("approve EXP-040", 2026-10-01) and built:** `src/frontier_ai/data/multi.py`
    (13 languages sampled in proportion to their tokens), `scripts/gpu_lr_arch.py` (Part 0 with
    the float64 check for RoPE + GQA-2, grids A and B, per-language validation, rules 1–4),
    `scripts/kaggle/exp040_kernel.py`, `scripts/run_kaggle_exp040.ps1` (`-Exp EXP-040` in the
    EXP-038 runner; one private 5.6 GB upload `frontier-v2-tok2-13lang`), tests in
    `tests/test_gpu_lr_arch.py`. **Next: the founder runs**
    `powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp040.ps1`; results come back to
    `evals/results/EXP-040/`. Don't push while that runner is going.

  - **EXP-040 ran (2026-10-02, results `7c6e5e2`):** baseline learning rate **1e-3** (rule 1);
    the float64 check for RoPE + GQA-2 **failed** 1e-8 (1.739e-07 at step 50; steps 1–27 agreed
    to ≤ 1.8e-15, then grew), so the candidate was not tested and the D-043 baseline stays.
    **EXP-041 proposed** (one-step non-compounding float64 check, then the 6 missing candidate
    runs only if it passes). The IsoFLOP ladder takes the next free number. **Next: "approve
    EXP-041"; no code before that.**

  - **EXP-041 approved ("approve EXP-041", 2026-10-02) and built:** `scripts/gpu_lr_arch.py
    --part followup` (Part A one-step float64 check for both architectures, baseline = control;
    Part B the 6 candidate runs + 1 baseline control only if both pass; EXP-040 rules unchanged on
    EXP-040's baseline runs), `scripts/kaggle/exp041_kernel.py`, `scripts/run_kaggle_exp041.ps1`
    (same 13-language dataset, no upload). **Next: the founder runs**
    `powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp041.ps1`. Don't push while it runs.

  - **EXP-041 ran (2026-10-02, results `6ee0bd5`, complete):** one-step float64 check PASS for
    both architectures (≤ 2.7e-15); RoPE + GQA-2 BETTER at both learning rates (+0.0385, +0.0645
    bits per byte; noise 0.0174), lower in all 13 languages; lr 1e-3 for both. **D-048 proposed**
    (adopt RoPE + GQA-2 for phase 2; device checks pass on the one-step check). The IsoFLOP ladder
    is now EXP-042. **Next: "approve D-048"; then propose EXP-042.**

  - **D-048 accepted ("approve D-048", 2026-10-02):** RoPE + GQA-2 is the architecture from
    phase 2 on; lr 1e-3 is the centre of each size's sweep; device checks pass on the one-step
    float64 check. **EXP-042 proposed** (IsoFLOP ladder: 3 budgets × 4 sizes, about 20 GPU-h over
    about 3 Kaggle sessions; no size chosen). **Next: "approve EXP-042"; no code before that.**

  - **EXP-042 approved and built ("approve EXP-042", 2026-10-02):** `scripts/gpu_lr_arch.py --part
    ladder` (one Kaggle session per launch, at most 9 GPU-h, 25 in total; resumes from
    `evals/results/EXP-042/session-*/`), `scripts/ladder_fit.py`, `scripts/kaggle/exp042_kernel.py`,
    `scripts/run_kaggle_exp042.ps1` (same dataset, no upload). **Next: the founder runs**
    `powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp042.ps1`, **and the same line
    again for each later session (about 3).** Don't push while a session runs.

  - **EXP-042 ran (2026-10-03, 3 Kaggle sessions, results `39ddd65`, `e82dc80`, `2e79552`,
    complete):** 20 runs, none failed, 15.98 GPU-h; all 3 budgets bracketed (N_opt 4.0 / 6.8 /
    22.2 M non-embedding); N_opt ∝ C^0.75 (non-embedding; post-hoc with total parameters C^0.48),
    D_opt ∝ C^0.51; for 2.78 B tokens: 311 M non-embedding (range 37 M – 4.6 B; post-hoc 206 M
    total). Plot and post-hoc checks: `scripts/plot_ladder.py` → `evals/results/EXP-042/`.
    Step 11 complete. **D-049 proposed:** step-12 model 14 × 896 (190 M parameters), one pass
    over all 2.78 B tokens. **Next: "approve D-049"; then propose EXP-043 (step-12 plan).**

  - **D-049 accepted ("approve D-049", 2026-10-03):** the step-12 model is 14 × 896 (190 M
    parameters), one pass over all 2.78 B tokens. Found while planning: `MultiTokenDataset` samples
    random windows with replacement (a correction note was added under EXP-042). **EXP-043
    proposed** (end of EXPERIMENTS.md): a one-pass sampler, a learning-rate check {2.5e-4, 5e-4,
    1e-3} × 100 M tokens, then the 169,911-step main run over about 9 Kaggle sessions with
    checkpoints chained through `kernel_sources` (kernels -a/-b); about 85 GPU-h, cap 100.
    **Next: "approve EXP-043"; no code before that.**

  - **EXP-043 approved and built ("approve EXP-043", 2026-10-03; code 2026-10-04):**
    `OnePassDataset` (`src/frontier_ai/data/multi.py`), trainer hooks (`seek`, `fit(should_stop)`,
    crash-safe checkpoint swap), `scripts/gpu_pretrain.py` (lr check, then the chained main run,
    sha256 refusal, stop rules, final evaluation), `scripts/kaggle/exp043_kernel.py`,
    `scripts/run_kaggle_exp043.ps1` (kernels `frontier-exp043-a`/`-b`, `kernel_sources`,
    `--file-pattern` download with a full-download fallback). On CPU, a split run ends with
    bit-identical weights to one uninterrupted run. Fixed: RoPE positions in KV-cached generation
    (training unaffected). **Next: the founder runs**
    `powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp043.ps1` **(session 1 = lr
    check), then the same line once per session (about 10).** Don't push while a session runs.
  - **Parallel plans proposed (2026-10-04, founder: "do most of all the work possible in
    parallel"):** EXP-044 two-GPU training (step 13; every Kaggle session had 2 T4s, we use 1),
    EXP-045 pre-registered evaluation plan + try-the-model tool, EXP-046 data slice 2 (Sangraha
    verified measured at 217.9 GB; ~108 B tokens estimated for our 13 languages, NOT VERIFIED)
    on Kaggle CPU sessions. **Next: the founder's "approve EXP-044 / EXP-045 / EXP-046"** (any
    subset). Pushing docs during a session is safe: the runner checks the commit only when it
    launches, and it rebase-retries a rejected results push (its commit touches only
    `evals/results/`).
  - **Parallel work approved and built (2026-10-04, "approve EXP-044, EXP-045, EXP-046"):**
    EXP-044 DDP code (`engine/distributed.py`, `scripts/gpu_ddp_check.py`); `trainer.py` and
    `multi.py` stay exact no-ops on one GPU (EXP-043 smoke hashes identical; re-check with a sha
    probe after ANY change there, because every EXP-043 session runs the pushed code). EXP-045 code
    (`evaluation/belebele.py`, `token_ngrams.py`, `scripts/eval_exp045.py`,
    `scripts/belebele_contamination.py`, `generate.py --weights/--interactive`). Founder lines,
    **only between two EXP-043 sessions** (the runner refuses otherwise):
    `scripts\run_kaggle_exp044.ps1` (two-GPU check, about 1.5 GPU-h) and
    `scripts\run_kaggle_exp045.ps1` (contamination check, CPU only). After EXP-043: export the
    held-out texts on the PC (`scripts/export_heldout_text.py`) and wire the EXP-045 GPU session.
    Next build: EXP-046 (probe kernel first).
  - **EXP-046 step 1 built (2026-10-04):** `corpora/frontier/v2/sangraha_slice2.json` pins
    `data-1` + `data-2` of the 13 languages (26 files, 9.79 GB; `mal/data-1` is half-size, kept as
    pinned). The probe kernel (`scripts/kaggle/exp046_probe_kernel.py`, CPU, processes no data)
    measures CPU/RAM/disk and one verified download. Founder line, **only between two EXP-043
    sessions**: `scripts\run_kaggle_exp046_probe.ps1`. Results land in
    `evals/results/EXP-046/probe/`. Next: write the two build kernels from the probe's numbers.
  - **EXP-043 session 1 done (2026-10-04):** learning-rate check, mean bpb 0.8543 / 0.8172 / 0.8160 at
    2.5e-4 / 5e-4 / 1e-3, 0 skipped steps, so **5e-4** is chosen by the pre-registered rule. 9.27 GB peak,
    about 11,100 tokens/s, 8.05 GPU-h. Results in `evals/results/EXP-043/session-1/`. A CPU smoke on
    `2f31c82` vs the current code gives bit-identical training state and final weights. Recommended order
    for the gap: EXP-044 (two-GPU check) first, then session 2 (the main run starts; first real
    `kernel_sources` chain test).
  - **EXP-044 PASSED (2026-10-04):** 1 vs 2 GPUs differ by 0.0042 bpb after 300 steps (limit 0.01),
    1.79x faster (rule 1.4x), 8.98 GB per GPU. **D-050 proposed:** the rest of the EXP-043 main run on
    both GPUs from the first session after it is built (fallback to one GPU if the two-GPU start
    fails; the cap counts session hours). Session 2 runs on one GPU now. Implement D-050 only after
    "approve D-050", keeping the one-GPU path bit-identical (CPU smoke hashes).
  - **D-050 approved (2026-10-04, "APPROVE D-50") and built (2026-10-05):** `gpu_pretrain.py --gpus 2`
    (torchrun worker, 16 x 1 per GPU, process 0 decides and writes; fallback to one GPU if the start
    fails); the EXP-043 kernel passes `--gpus 2`. One-GPU path bit-identical (CPU smoke hashes);
    `tests/test_pretrain_ddp.py` 6 passed; full suite 748 passed, 1 skipped. Every EXP-043 session
    launched from now on uses both GPUs (the same founder line). Check each two-GPU session against
    D-050's revisit rules: >5% skipped steps, a hang/NCCL error, or whole-session speed < 1.4x.
  - **Correction (2026-10-05):** session 2 had NOT been launched. The code now keeps the first
    main-run session (no checkpoint yet = session 2) on one GPU even with `--gpus 2` (D-050 point 2).
    Two GPUs start from session 3. The same founder line is used for every session.
  - **EXP-043 session 2 done (2026-10-06, results `1eeca41`):** main run steps 0 → 20,594 of 169,911
    on one GPU; 11,115 tokens/s (1.474 s/step); 9.27 GB; 7 skipped steps; checkpoint step 20,594
    (sha256 recorded). Session-end full validation 0.7727 bpb mean (mid-run, not a result; C3-s3
    final 0.7736). 16.72 of 100 session-h used. Next: session 3 = first checkpoint load from the other
    kernel AND the first two-GPU session (D-050). Check its chain line, `gpus_plan`, speed vs 1.474
    s/step (needs >= 1.4x) and skipped steps.
  - **EXP-043 session 3 done (2026-10-06, results `d0c3655`), the first two-GPU session:** checkpoint
    20,594 verified from kernel b and resumed; steps 20,594 -> 61,207; 0.744 s/step = 1.98x one GPU;
    8.97 GB per GPU; 16 skipped steps (0.04%); all D-050 revisit rules clear. Session-end full
    validation 0.7199 bpb mean (mid-run). 25.38 of 100 session-h used; about 3 more sessions (4-6).
    Kaggle's weekly GPU quota resets Saturday 00:00 UTC (Kaggle staff note, 2020; current behaviour
    NOT VERIFIED); about 26 h of it were used in the week starting 2026-10-03.
  - **EXP-043 session 4 refused by Kaggle (2026-10-06 19:14 IST):** "Maximum weekly GPU quota of 30.00
    hours reached" (EXP-042's last session on 2026-10-03 also counts in this quota week). Nothing ran;
    session 3's checkpoint (step 61,207, kernel a) is untouched. Session 4 = the same founder line
    after the weekly reset (Saturday 2026-10-10 00:00 UTC = 05:30 IST per Kaggle's 2020 staff note;
    NOT VERIFIED). Meanwhile the approved CPU-only work (EXP-046 probe, EXP-045 test 4) uses the gap.
    The runner prints Kaggle's refusal only to run.log, not REPORT.txt (possible small fix, needs
    approval).
  - **EXP-046 probe done (2026-10-06, results `1d3eabe`):** Kaggle CPU session = 4 cores Xeon 2.2 GHz,
    30 GiB RAM limit, 20.96 GB output, about 45 MB/s from Hugging Face; it started while the GPU quota
    was used up. Slice-2 output estimate 16.4 GB, so two build kernels (A: en ur as bn gu hi; B: kn ml
    mr or pa ta te). The build needs the held-out texts on Kaggle (export_heldout_text.py -> small
    private input). Next for the founder in the gap: EXP-045 test 4 (CPU).

## 9. The prompt the operator pastes into a new Arena chat

```
We are continuing my project frontier-ai from an earlier Arena chat that you cannot see. Everything from that chat is saved on GitHub, on the branch arena/01a0d31f-frontier-ai. Before anything else:
1. Run: git fetch origin arena/01a0d31f-frontier-ai
   Then run: git merge --ff-only FETCH_HEAD
   If the merge is refused, run git merge --no-edit FETCH_HEAD and tell me. If the fetch says the branch does not exist, it has already been merged into main and you have everything.
2. Read NEW_CHAT_START_HERE.md completely and follow it. It links to everything else.
3. Then tell me in simple words where the project stands and what the next step is. Do not start any new work until I say so.
```
