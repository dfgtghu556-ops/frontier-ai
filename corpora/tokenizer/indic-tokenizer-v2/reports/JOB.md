# Current job: JOB-002 — download everything once (EXP-018) and report; nothing is locked

Written by the Arena agent, 2026-09-25. It replaces JOB-001, which was never run: this job's
report covers everything JOB-001's would have, and step 2 below is JOB-001's tidy-up. Read
[docs/tokenizer_corpus_local_runner.md](../../../../docs/tokenizer_corpus_local_runner.md)
first; its rules apply to every step.

**Since the handover ([docs/tokenizer_corpus_handover.md](../../../../docs/tokenizer_corpus_handover.md)),
the local assistant owns P004B:** run this job, then review the report yourself (handover
§8, runbook §2.7 and §8) and explain the result to the operator in plain words.

**Why.** Four more languages now have books declared in the manifest — Punjabi, Kannada,
Telugu and Tamil, 8 new sources (runbook §2.7). They have never been downloaded. This job
downloads all 55 sources once, so the new text can be reviewed and the 47 locked sources
checked for changes. It also tidies up the old second copy of the project
(untracked sub-folder `frontier-ai/`, downloaded 2026-09-24 17:28, not used since), which
must not stay inside the project, where it could be committed by accident.

**This job locks nothing:** it has no `--pin` command, and the manifest must not change.

**Steps**, from the repository root:

1. `git pull origin arena/01a0d31f-frontier-ai`
2. Tidy-up (JOB-001's steps 2–3). If there is no sub-folder `frontier-ai` in the repository
   root, skip this step and write "old copy: not found" in your final line. Otherwise:
   * Check that the old copy holds no changes: `git -C frontier-ai status --short`.
     Expected: no output at all. If it prints anything (or an error), do not move it; quote
     the output in your final line.
   * Move it next to the repository — **move it, never delete it**:
     PowerShell `Move-Item frontier-ai ..\frontier-ai-old-copy`, Git Bash
     `mv frontier-ai ../frontier-ai-old-copy`. If `frontier-ai-old-copy` already exists
     there, do not move anything and say so. If the move fails (for example "in use"),
     quote the error and go on with step 3.
3. Download and build (several minutes; one line per download):
   `python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-018 --out data/tokenizer/indic-tokenizer-v2`
   * Then note its exit code: PowerShell `echo $LASTEXITCODE`, Git Bash `echo $?`.
   * Expected: `0`, and no line starting with `REFUSED:`. Note any `REFUSED:` line word for
     word.
   * If downloads failed with a network error (for example a timeout or a dropped
     connection), you may rerun this command once. Do not change anything else.
4. Write the report (reads files only, downloads nothing):
   `python scripts/inspect_corpus_sources.py --output corpora/tokenizer/indic-tokenizer-v2/reports/EXP-018-inspection.txt`
   * Expected first line: `… 55 sources, 55 verified, 47 pinned in the manifest, built 2026-09-…`
   * Its exit code is expected to be `1` (at least one source is flagged for a genuine year
     in the text). Whatever it prints, the report is still committed in step 5.
5. Check, then commit and push only the report:
   * `git status --short` should list only
     `?? corpora/tokenizer/indic-tokenizer-v2/reports/EXP-018-inspection.txt`. If it lists
     anything else, do not stage it; mention it in your final line. (If
     `corpora/tokenizer/indic-tokenizer-v2/sources.json` shows as modified, which this job
     should never cause, undo it with
     `git checkout -- corpora/tokenizer/indic-tokenizer-v2/sources.json` and mention it.)
   * `git add corpora/tokenizer/indic-tokenizer-v2/reports/EXP-018-inspection.txt`
   * `git commit -m "P004B: JOB-002 report (EXP-018 fetch, build exit N)"` — with the exit
     code from step 3 in place of `N` (the last run's, if you reran it).
   * `git push origin arena/01a0d31f-frontier-ai`
6. Final line for the operator, for example:
   `JOB-002 done — build exit 0, pushed <commit id>; old copy moved to ..\frontier-ai-old-copy`
