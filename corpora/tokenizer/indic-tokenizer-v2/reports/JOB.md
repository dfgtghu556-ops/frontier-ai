# Current job: JOB-001 — tidy up, and report EXP-017's coverage (no downloads)

Written by the Arena agent, 2026-09-25. Read
[docs/tokenizer_corpus_local_runner.md](../../../../docs/tokenizer_corpus_local_runner.md)
first; its rules apply to every step.

**Why.** (1) The repository folder contains an old second copy of the whole project, in an
untracked sub-folder called `frontier-ai/` (downloaded 2026-09-24 17:28 and not used since).
It must not stay inside the project, where it could be committed by accident. (2) The Arena
agent needs the language coverage of the last build, EXP-017, which is already on disk; the
inspection report now includes it.

**Steps**, from the repository root:

1. `git pull origin arena/01a0d31f-frontier-ai`
2. Check that the old copy holds no changes: `git -C frontier-ai status --short`
   * Expected: no output at all. If it prints anything (or an error), skip step 3 and quote
     the output in your final line.
3. Move the old copy next to the repository — **move it, never delete it**:
   * PowerShell: `Move-Item frontier-ai ..\frontier-ai-old-copy`
   * Git Bash: `mv frontier-ai ../frontier-ai-old-copy`
   * If `frontier-ai-old-copy` already exists next to the repository, do not move anything;
     say so in your final line. If the move fails (for example "in use"), quote the error and
     go on with step 4.
4. Write the report of the build already on disk (reads files only, downloads nothing):
   `python scripts/inspect_corpus_sources.py --output corpora/tokenizer/indic-tokenizer-v2/reports/EXP-017-inspection.txt`
   * Expected first line: `… 47 sources, 47 verified, 47 pinned in the manifest, built 2026-09-24T…`
   * Its exit code is expected to be `1` (one source is flagged for a genuine year in the
     text). Anything else unexpected: still commit the report, and mention it.
5. Commit and push only that file:
   * `git add corpora/tokenizer/indic-tokenizer-v2/reports/EXP-017-inspection.txt`
   * `git commit -m "P004B: JOB-001 report (EXP-017 inspection with coverage)"`
   * `git push origin arena/01a0d31f-frontier-ai`
6. Final line for the operator, for example:
   `JOB-001 done — pushed <commit id>; old copy moved to ..\frontier-ai-old-copy`
