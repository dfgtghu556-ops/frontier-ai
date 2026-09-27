# PC TASK — Freeze Frontier Tokenizer v1 (EXP-030)

**For:** the local agent (Claude Code / Copilot) on the founder's Windows PC, repo `E:\frontier-ai`.
**Mission:** copy the selected tokenizer (D-040: `mark_aware-32768`) into the tracked folder
`tokenizers/frontier-tokenizer-v1/`, prove it is the exact tokenizer that won EXP-B (4 gates),
then commit and push **only that folder**. Takes a few minutes. The founder approved this task,
including the one-folder commit + push (2026-09-27, "approve freeze plan").

## Context

- Branch: `arena/01a0dc16-frontier-ai`. Use the existing `.venv` (never create a new one).
- The tokenizer exists only on this PC:
  `out\experiments\EXP-028\py-mark_aware-32768\seed-0000001337\tokenizer\bpe_python.json`
  (git-ignored). EXP-029 recorded its fingerprint and token counts in
  `out\exp_b\EXP-029\manifest.json`.
- `scripts/freeze_tokenizer.py` (defaults = the EXP-030 values; no arguments needed) checks:
  **A** fingerprint == the EXP-029 record · **B** vocab 32768 / 32512 merges / mark_aware /
  no special tokens · **C** re-encoding the frozen FrontierCorpus v1 gives EXP-029's exact token
  counts (train 1,608,987 · held-out 184,233) · **D** every document round-trips losslessly.
  It writes nothing into the repo unless all four pass.
- `.gitattributes` (`tokenizers/** -text`) makes git store the folder byte-for-byte. This matters:
  the file has Windows line endings, and all hashes are over the raw bytes.

## Steps (in order — if any step fails, STOP and report; do not improvise fixes)

1. **Get the latest code:** `git pull --ff-only origin arena/01a0dc16-frontier-ai`
   Then confirm both: `Test-Path scripts\freeze_tokenizer.py` prints `True`, and
   `git check-attr text -- tokenizers/frontier-tokenizer-v1/tokenizer/bpe_python.json`
   prints `... text: unset`. If either is not so, STOP and report.
2. **Already done?** If `git ls-files tokenizers/frontier-tokenizer-v1` prints two files, the
   freeze is already committed — skip to step 6 and report that.
3. **Clean start:** `git status --short` must print nothing (`out\` is git-ignored, so
   experiment outputs never appear there). If anything is listed, STOP and report — do not
   stash, reset or commit it.
4. **Run the freeze** (stderr merged by cmd.exe so PowerShell cannot misreport normal log lines):
   `cmd /d /c ".venv\Scripts\python.exe -u scripts\freeze_tokenizer.py 2>&1"`
   then `echo $LASTEXITCODE`.
   Expected: four lines `[freeze] gate A_identity: PASS` … `gate D_lossless: PASS`, a token-count
   line, a line starting `[freeze] frozen: tokenizers\frontier-tokenizer-v1`, a `[record]` line,
   and exit code **0**. Exit 1 = a gate failed (nothing was written) → STOP and report the output.
   Exit 2 = an input is missing → STOP and report the output. Do NOT re-run with other arguments.
5. **Commit and push ONLY the frozen folder:**
   - `git status --short` must now show only `?? tokenizers/` — anything else: STOP and report.
   - `git add tokenizers/frontier-tokenizer-v1`
   - `git diff --cached --stat` must list exactly 2 files:
     `tokenizers/frontier-tokenizer-v1/FREEZE.json` and
     `tokenizers/frontier-tokenizer-v1/tokenizer/bpe_python.json`. Otherwise: STOP, run
     `git reset` (unstages only; deletes nothing) and report.
   - `git commit -m "EXP-030: freeze Frontier Tokenizer v1 (D-040) - PC gates A-D PASS"`
   - `git push origin arena/01a0dc16-frontier-ai`
     (if it fails with an authentication error: STOP and report — never force-push).
   - Byte-exactness check: `git ls-files --eol tokenizers/frontier-tokenizer-v1` — the
     `bpe_python.json` line must show `attr/-text`.
6. **Report** — write everything below to `out\tokenizer_freeze\EXP-030\FREEZE_REPORT.txt`
   (inside `out\`, git-ignored — allowed) and print it in the terminal:
   - the full output of step 4 and its exit code;
   - the commit hash (`git log --oneline -1`) and the push result line;
   - the output of `git ls-files --eol tokenizers/frontier-tokenizer-v1`;
   - a PASS/FAIL line for: gates A–D, "only 2 files committed", "push succeeded".
   Then STOP.

## Hard rules

- Do NOT edit, stage or commit any file other than the two in `tokenizers/frontier-tokenizer-v1/`.
- Do NOT change arguments of the freeze script, and do NOT hand-edit `FREEZE.json` or the tokenizer.
- Do NOT delete or modify anything under `out\experiments\EXP-028\` or `out\exp_b\EXP-029\`.
- Push only to `arena/01a0dc16-frontier-ai`; never force-push; never create branches.
- If a gate fails, commit nothing — a failed gate is a finding for the sandbox agent to analyse.
