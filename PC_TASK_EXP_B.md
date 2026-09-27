# PC TASK — EXP-B night matrix (EXP-029)

**For:** the local agent (Claude Code) running on the founder's Windows PC, repo `E:\frontier-ai`.
**Mission:** run the pre-registered EXP-B tokenizer comparison matrix to completion, verify it from the files, and report the result. The founder will sleep; work unattended and stop at the first real failure.

## Context

- Repo: `E:\frontier-ai`, branch `arena/01a0dc16-frontier-ai`. Use the existing `.venv` (never create a new one).
- The experiment is **PRE-REGISTERED** in `EXPERIMENTS.md` (search "EXP-029") and implemented in `scripts/run_exp_b.py`:
  - 2 tokenizers: `mark_aware-32768`, `mark_aware-16384` (prepared data already exists in `out\exp_b\EXP-029\`)
  - 3 seeds: 1337, 1338, 1339
  - fixed model/training config: `configs/exp_b.json`
  - budget: **150 steps per cell** (the one permitted budget revision — do not change it)
  - metric: held-out bits-per-byte (lower is better)
  - decision rule (in `decide()` of `scripts/run_exp_b.py`): lower mean bpb wins; if the gap is smaller than the average of the two within-tokenizer stds → TIE → the smaller-vocabulary tokenizer wins.
- The night runner is `scripts/run_exp_b_night.ps1`: it refuses to start if a matrix is already running, runs the matrix, logs every line to `out\exp_b\EXP-029\night.log`, and writes a completion marker with the exit code.

## Steps (in order — if a step fails, stop and report what failed; do not improvise fixes)

1. **Get the latest code (always):** `git pull --ff-only origin arena/01a0dc16-frontier-ai`, then confirm the fix is present: `Select-String -Path scripts\run_exp_b_night.ps1 -Pattern "Invoke-Matrix"` must print at least one line. If it prints nothing, STOP and report (the PowerShell fix is missing).
2. **Stop any stale matrix run — including orphaned cell processes** (a crashed runner can leave a `train.py` cell running on its own):
   `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_exp_b.py*' -or ($_.CommandLine -like '*train.py*' -and $_.CommandLine -like '*EXP-029*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped PID $($_.ProcessId)" }`
   (Printing nothing is fine — it just means nothing was running.)
3. **Keep the PC awake for the night:** `powercfg /change standby-timeout-ac 0`
4. **Skip check:** if `out\exp_b\EXP-029\runs\report.txt` already exists AND all six cell records from step 6 already exist AND the report contains no `FAIL`, skip to step 6 (a finished run must not be re-run).
5. **Run the matrix:** `powershell -ExecutionPolicy Bypass -File scripts\run_exp_b_night.ps1`
   - Takes **~2 hours**. It logs everything to `out\exp_b\EXP-029\night.log`.
   - Quiet stretches of ~20 minutes between cells are NORMAL (each cell trains silently, then prints its log at once) — not a hang.
   - If your tooling cannot block for 2 h, launch it in a background terminal and poll `night.log` every 5 minutes until the completion marker appears.
   - **History (why the script changed in `run_exp_b_night.ps1` after `8feb98f`):** the first attempt died right after cell 1 (bpb 1.4674) with a PowerShell `NativeCommandError` on the normal `[record] ... | fingerprint ...` line. That line is printed on stderr, and Windows PowerShell 5.1 treats any stderr line as fatal under `$ErrorActionPreference = "Stop"`. The script now lets `cmd.exe` merge stderr into stdout, so this can no longer happen. Make sure step 1 pulled the fix (the script must contain `Invoke-Matrix`).
   - **Fallback — only if the night script itself fails again with a PowerShell error (not a Python error):** run the same pre-registered matrix without PowerShell stream handling, from the repo root:
     `cmd /d /c ".venv\Scripts\python.exe -u scripts\run_exp_b.py --seeds 1337,1338,1339 --max-steps 150 > out\exp_b\EXP-029\night_direct.log 2>&1"`
     then check `echo $LASTEXITCODE` (must be 0) and use `night_direct.log` in place of `night.log` for check 6a (it has no `NIGHT RUN` marker; instead it must end with the report and an `[exp-b] report:` line). This fallback counts as the single allowed retry.
6. **Verify from the files** (do not trust the log alone):
   - **a.** `night.log` ends with `NIGHT RUN matrix COMPLETE` (or `NIGHT RUN COMPLETE`) and the exit-code line says `0`.
   - **b.** `out\exp_b\EXP-029\runs\report.txt` contains **no** `FAIL` and has a `DECISION` line.
   - **c.** All six per-cell records exist —
     `out\exp_b\EXP-029\runs\mark_aware-32768\seed-{1337,1338,1339}\experiment.json` and
     `out\exp_b\EXP-029\runs\mark_aware-16384\seed-{1337,1338,1339}\experiment.json` —
     each with `execution.status == "success"` and `results.best_bpb` present.
   - **d.** Cross-check: the six bpb values printed in `report.txt` exactly match the six `results.best_bpb` values in the `experiment.json` files.
7. **Report to the founder** (this is your final output):
   - First, write the report to a single easy-to-open file: copy `out\exp_b\EXP-029\runs\report.txt` to `out\exp_b\EXP-029\NIGHT_REPORT.txt`, then append to that same file a blank line and your four `6a`–`6d` PASS/FAIL lines (plus the six per-cell bpb values with their record paths). Writing files under `out\` is allowed — it is experiment output, not the repo.
   - Then also print the same content in the terminal so the founder can read it.
   - Then STOP and wait for the founder to wake up.

## Hard rules

- **Do NOT change any pre-registered parameter** (seeds, 150 steps, `configs/exp_b.json`, the decision rule) and do NOT re-run with different ones.
- If the night script exits non-zero, you may re-run it **ONE** more time (the run is deterministic; re-runs are safe and simply overwrite the cell records). If it fails twice, STOP and report the last 40 lines of `night.log`.
- **Do NOT edit any file in the repo, do NOT commit, do NOT push.** Recording the results is done by the sandbox agent after the founder reviews the table.
- **Do NOT re-run the matrix "to be sure"** beyond the single allowed retry.
- **Optional, only if the founder explicitly says so:** add `-Verify` to the night script for an independent second pass (~2 h more) that proves cross-process reproducibility.
