# PC TASK — First evaluation-harness run (EXP-031)

**For:** the local agent (Claude Code / Copilot) on the founder's Windows PC, repo `E:\frontier-ai`.
**Mission:** build the protected evaluation suite, score the 6 EXP-B (EXP-029) models with
evaluation harness v1, prove reproducibility, compare the two tokenizers, then commit and push
the suite file plus the small result files. Expected time: roughly 30–45 minutes of CPU at 100 %
(normal). The founder approved this work on 2026-09-27 ("Approve eval harness plan + roadmap sync").

## Context

- Branch `arena/01a0dc16-frontier-ai`; existing `.venv` only. The founder's chat message already
  contained the `git pull`.
- Models: `out\exp_b\EXP-029\runs\mark_aware-{32768,16384}\seed-{1337,1338,1339}\best\`.
- Tokenizers: `mark_aware-32768` = the frozen Frontier Tokenizer v1 (`--tokenizer frozen`; EXP-030
  proved it byte-identical to the EXP-028 artifact). `mark_aware-16384` = the EXP-028 artifact
  `out\experiments\EXP-028\py-mark_aware-16384\seed-0000001337\tokenizer`.
- Run every Python command through `cmd /d /c "... 2>&1"` (Windows PowerShell 5.1 otherwise
  turns normal stderr log lines into errors), then check `echo $LASTEXITCODE`.

## Steps (in order — if any step fails, STOP and report; do not improvise fixes)

1. **Check the code:** `Test-Path scripts\eval_report.py`, `Test-Path scripts\build_eval_suite.py`,
   `Test-Path scripts\publish_eval_results.py` must all print `True`. `git status --short` must print
   nothing. Confirm all 6 `...\best\model.pt` files exist. Otherwise STOP and report.
2. **Build the protected suite:**
   `cmd /d /c ".venv\Scripts\python.exe -u scripts\build_eval_suite.py 2>&1"` → exit 0 and a line
   `[suite] wrote evals\suites\frontier-heldout-v1\SUITE.json: 3,427 documents, ...`.
3. **Score the 6 models** (one command each, exit code 0 each; each prints a report ending in
   `scores sha256: ...`; `data identity: PASS` is expected in every report):
   - `cmd /d /c ".venv\Scripts\python.exe -u scripts\eval_report.py --ckpt out\exp_b\EXP-029\runs\mark_aware-32768\seed-1337\best --tokenizer frozen 2>&1"`
   - same with `seed-1338`, then `seed-1339` (still `--tokenizer frozen`)
   - `cmd /d /c ".venv\Scripts\python.exe -u scripts\eval_report.py --ckpt out\exp_b\EXP-029\runs\mark_aware-16384\seed-1337\best --tokenizer out\experiments\EXP-028\py-mark_aware-16384\seed-0000001337\tokenizer 2>&1"`
   - same with `seed-1338`, then `seed-1339`
   Exit 1 means `data identity: FAIL` → STOP and report that report. Exit 2 = input problem → STOP
   and report the output.
4. **Reproducibility checks** (no records, separate folders):
   - a) same model again:
     `cmd /d /c ".venv\Scripts\python.exe -u scripts\eval_report.py --ckpt out\exp_b\EXP-029\runs\mark_aware-32768\seed-1337\best --tokenizer frozen --out out\eval\EXP-031\repro\rerun-32768-seed-1337 --no-record 2>&1"`
   - b) frozen tokenizer vs the original EXP-028 artifact:
     `cmd /d /c ".venv\Scripts\python.exe -u scripts\eval_report.py --ckpt out\exp_b\EXP-029\runs\mark_aware-32768\seed-1337\best --tokenizer out\experiments\EXP-028\py-mark_aware-32768\seed-0000001337\tokenizer --out out\eval\EXP-031\repro\exp028-artifact-32768-seed-1337 --no-record 2>&1"`
   Both must end with the **same `scores sha256`** as step 3's `mark_aware-32768-seed-1337` report.
5. **Compare the tokenizers:**
   `cmd /d /c ".venv\Scripts\python.exe -u scripts\eval_compare.py --a out\eval\EXP-031\mark_aware-32768-seed-1337 out\eval\EXP-031\mark_aware-32768-seed-1338 out\eval\EXP-031\mark_aware-32768-seed-1339 --label-a mark_aware-32768 --b out\eval\EXP-031\mark_aware-16384-seed-1337 out\eval\EXP-031\mark_aware-16384-seed-1338 out\eval\EXP-031\mark_aware-16384-seed-1339 --label-b mark_aware-16384 --out out\eval\EXP-031\compare 2>&1"`
6. **Publish + commit + push** (only these paths):
   - `cmd /d /c ".venv\Scripts\python.exe -u scripts\publish_eval_results.py --exp-id EXP-031 2>&1"` → exit 0.
   - `git status --short` must show only `?? evals/` — anything else: STOP and report.
   - `git add evals/suites/frontier-heldout-v1/SUITE.json evals/results/EXP-031`
   - `git diff --cached --stat` must list only files under `evals/suites/frontier-heldout-v1/` (1 file)
     and `evals/results/EXP-031/` (no `per_document.jsonl`). Otherwise: `git reset` (unstages only)
     and STOP and report.
   - `git commit -m "EXP-031: protected suite frontier-heldout-v1 + harness v1 results for the 6 EXP-B models"`
   - `git push origin arena/01a0dc16-frontier-ai` (auth error → STOP and report; never force-push).
7. **Report** — write to `out\eval\EXP-031\EVAL_SUMMARY.txt` and print: the step-2 line, the
   `OVERALL` line and `data identity` line of each of the 6 reports, the three `scores sha256`
   values of step 4 (original, rerun, EXP-028 artifact) with MATCH/MISMATCH, the full
   `compare.txt`, the commit hash and push result, and PASS/FAIL lines for steps 2–6. Then STOP.

## Hard rules

- Do NOT edit any repository file by hand; do NOT change script arguments beyond those above.
- Do NOT delete or modify anything under `out\experiments\EXP-028\` or `out\exp_b\EXP-029\`.
- Commit only `evals/suites/frontier-heldout-v1/SUITE.json` and `evals/results/EXP-031/**`.
- Push only to `arena/01a0dc16-frontier-ai`; never force-push; never create branches.
- Do NOT start any training run.
