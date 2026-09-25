# Local runner protocol — running Stage B jobs on the operator's machine

**Who this is for:** an assistant (for example Claude Code in VS Code) or a person who runs
the network steps of the P004B corpus work on the operator's computer. The Arena agent plans
the work, writes the code, the manifest and the docs, and reviews every result. Its own
environment cannot download from the wikis, so the downloads run here. The two sides talk
only through git, on branch `arena/01a0d31f-frontier-ai`:

* **to you:** the current job, in `corpora/tokenizer/indic-tokenizer-v2/reports/JOB.md`;
* **back:** the files the job tells you to save in that folder, committed and pushed.

Read this whole file once per session. The job file says exactly what to run; this file
says what you may and may not do. The rules behind the work are the three at the top of
[tokenizer_corpus_stage_b_acquisition.md](tokenizer_corpus_stage_b_acquisition.md); you do
not need the rest of that runbook to run a job.

## Standing rules

1. **Folder.** Run everything from the repository root (the folder that holds
   `pyproject.toml`, `corpora/` and `scripts/`).
2. **Branch.** Work only on `arena/01a0d31f-frontier-ai`. Begin every job with
   `git pull origin arena/01a0d31f-frontier-ai`. Push only with
   `git push origin arena/01a0d31f-frontier-ai`. Never push to `main`, never force-push, never
   create, switch, merge or rebase branches, and never use VS Code's Sync or Publish buttons
   (one of them once pushed this work to `main`). If the pull reports a conflict or opens a
   merge message, stop and report.
3. **Touch only what the job names.** Run the job's commands as written. Stage only the
   exact paths the job lists (`git add <path>`); never `git add .`, `git add -A` or
   `git commit -a`. Never edit files under `src/`, `scripts/`, `tests/` or `docs/`, or the
   manifest `corpora/tokenizer/indic-tokenizer-v2/sources.json`, by hand.
4. **Hashes come only from `--pin`.** Never type, copy or edit a `sha256`, `verified` or
   `retrieved_at` value. Run a `--pin` command only when the job contains one, and commit the
   manifest only if every check the job lists passes. If a check fails, undo the pin with
   `git checkout -- corpora/tokenizer/indic-tokenizer-v2/sources.json` and report.
5. **Report, don't repair.** If a command fails or a result differs from what the job
   expects, do not change code, options or checks to make it pass, and do not look for other
   sources. A download that failed with a network error may be rerun once. Otherwise save
   what you saw, push it if the job says so, and stop.
6. **Python.** Use the interpreter the project is installed in. Check first:
   `python -c "import frontier_ai, sys; print(sys.executable)"`. If that fails and a `.venv`
   folder exists, use `.venv\Scripts\python.exe` (PowerShell) or `.venv/Scripts/python.exe`
   (Git Bash) wherever a job says `python`. Do not install or upgrade packages unless the
   job says so.
7. **Text files.** When a script can write a file (`--output`), use that option, never shell
   redirection (`>`): Windows PowerShell 5.1 redirection writes UTF-16.
8. **Nothing is deleted.** Never delete files or folders. `data/` and `out/` are the scripts'
   own git-ignored outputs; they change only by running the scripts.
9. **Finish** with one line for the operator: the job id, `done` or `stopped: <reason>`, and
   the commit id you pushed (if any).

## What the scripts do, so the output makes sense

* `python scripts/build_tokenizer_corpus.py --fetch --exp-id EXP-0NN --out data/tokenizer/indic-tokenizer-v2`
  downloads every declared source, cleans it, checks its licence and proofreading, and writes
  the results under `data/tokenizer/indic-tokenizer-v2/`. It prints one line per download and
  takes a few minutes. Exit code `0` means no pinned source changed — it does **not** mean
  every source verified (the report shows that); `1` means a pinned source came back
  different (`hash_mismatch`); `2` means bad input. With `--pin` it also writes the hashes of
  the sources that verified into the manifest.
* `python scripts/inspect_corpus_sources.py --output FILE` reads that build — read-only, no
  network — and writes the report the Arena agent reviews: one line per source, flags,
  repairs, samples of the text, and every language slot's coverage. It exits `1` when
  something is flagged, which a job may expect.
