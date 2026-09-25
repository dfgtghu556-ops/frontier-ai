# Notes for Claude Code

General orientation for this repository is in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## P004B — the tokenizer research corpus (yours since 2026-09-25)

The operator has handed P004B — acquiring and verifying the real text corpus of
`indic-tokenizer/v2` — to you, end to end: choosing books, licence review, code, downloads,
reviewing results, locking hashes and recording each run. Before any work on it (anything
under `corpora/tokenizer/`, the tokenizer-corpus scripts, `src/frontier_ai/data/mediawiki.py`,
`src/frontier_ai/tokenization/research_corpus.py` or their docs), read
[docs/tokenizer_corpus_handover.md](docs/tokenizer_corpus_handover.md) in full and follow it.
The rules that matter most:

1. Never pin a hash you did not compute from a fetch you actually performed — only the
   build script's `--pin` writes hashes, and only after a clean review.
2. Never mark a source verified because the manifest says it is CC BY-SA.
3. If a source cannot be licensed, leave it unverified. Never pad or substitute text.
4. Work only on branch `arena/01a0d31f-frontier-ai`: pull first, push only there; never
   `main`, never force-push, never VS Code's Sync or Publish buttons.
5. Add to shared files, never rewrite them. EXPERIMENTS.md and NEW_CHAT_START_HERE.md keep
   everything that came before, so add new entries with an edit, never by writing the whole
   file anew. Before every commit check `git diff --stat`: many deleted lines mean
   something went wrong.

Where things stand: the handover's §2 and §10, the runbook's status header
([docs/tokenizer_corpus_stage_b_acquisition.md](docs/tokenizer_corpus_stage_b_acquisition.md)),
and the current job in `corpora/tokenizer/indic-tokenizer-v2/reports/JOB.md`.

## Keep the next chat informed

At the end of every working session on P004B, and when P004B is finished:
* add an entry to the status log at the end of [NEW_CHAT_START_HERE.md](NEW_CHAT_START_HERE.md)
  (date, what was done, the commits, what comes next);
* update its §3 table;
* push it with your work.

When P004B is finished, tell the operator the branch is ready to be merged into `main`.
Open that pull request (`gh pr create --base main --head arena/01a0d31f-frontier-ai`) only
if they ask. The operator merges it on GitHub; never push or merge to `main` yourself.

This file covers P004B only; other work in this repository is unaffected.
