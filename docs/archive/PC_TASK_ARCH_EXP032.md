# PC TASK — Step 9 architecture screening, overnight (EXP-032)

**For:** the local agent (GitHub Copilot) on the founder's Windows PC, repo `E:\frontier-ai`.
**Mission:** run the pre-registered step 9 architecture ablation overnight, then publish,
commit and push only the small result files. The founder approved the step 9 plan on
2026-09-27 ("I approve step 9 plan").
**Expected time:** about 5.5–6.5 hours of CPU at 100 %, which is normal. Add about 1 hour if
the EXP-029 baseline cannot be reused (step 2 tells you).

## Context

- Branch `arena/01a0dc16-frontier-ai`; existing `.venv` only. The founder's chat message
  already contained the `git pull`.
- Spec (do not edit): `configs/ablations/EXP-032.json`. It defines baseline + 4 variants
  (`rope`, `gelu`, `layernorm`, `gqa2`) × seeds 1337/1338/1339, 2 learning-rate cells and 1
  reproducibility cell, at 150 steps on `out\exp_b\EXP-029\mark_aware-32768.bin` with the
  frozen tokenizer.
- Runner: `scripts/run_arch_ablation.py`. Night wrapper:
  `scripts/run_arch_ablation_night.ps1`, which logs to `out\arch\EXP-032\night.log` and retries
  once automatically. The run is restartable: finished models are skipped.
- Run every Python command through `cmd /d /c "... 2>&1"` and then check `echo $LASTEXITCODE`.

## Steps (if a step fails, STOP and report; do not improvise fixes)

1. **Check:** `git status --short` prints nothing. `Test-Path configs\ablations\EXP-032.json`,
   `Test-Path scripts\run_arch_ablation.py` and `Test-Path scripts\run_arch_ablation_night.ps1`
   all print `True`. `Test-Path out\exp_b\EXP-029\mark_aware-32768.bin` prints `True`.
2. **Preflight (trains nothing, about 1 minute):**
   `cmd /d /c ".venv\Scripts\python.exe -u scripts\run_arch_ablation.py --spec configs\ablations\EXP-032.json --dry-run 2>&1"`
   → exit 0. Save the whole output to `out\arch\EXP-032\preflight.txt`. For each baseline seed
   it prints either `reusing ... (config verified)` or `NOT reused (...) -> will train`. Both
   are fine; note which. It also prints the number of models to train and a time estimate.
3. **Keep the PC awake:** `powercfg /change standby-timeout-ac 0`
4. **Start the night run in its own window, so it keeps running even if this chat session ends:**
   `Start-Process powershell -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','scripts\run_arch_ablation_night.ps1' -WindowStyle Minimized`
   Then poll every 10 minutes:
   `Get-Content out\arch\EXP-032\night.log -Tail 5`
   until the log contains `NIGHT RUN COMPLETE` or `NIGHT RUN FAILED`. Quiet stretches of
   about 20 minutes between models are normal. Do NOT start a second run. If your session
   ends, the run continues on its own; the founder will later ask you to resume at step 5.
5. **If `NIGHT RUN FAILED`:** do not commit anything. Write the last 60 lines of `night.log`
   and the contents of `out\arch\EXP-032\SUMMARY.txt` (if it exists) into
   `out\arch\EXP-032\NIGHT_REPORT.txt`, print it, and STOP.
6. **If `NIGHT RUN COMPLETE`: publish, commit, push** (only these paths):
   - `cmd /d /c ".venv\Scripts\python.exe -u scripts\publish_eval_results.py --exp-id EXP-032 --src out\arch\EXP-032 2>&1"` → exit 0.
   - `git status --short` must show only `?? evals/results/EXP-032/`. Anything else: STOP and report.
   - `git add evals/results/EXP-032`
   - `git diff --cached --stat` must list only files under `evals/results/EXP-032/`. Otherwise
     run `git reset` (unstages only), then STOP and report.
   - `git commit -m "EXP-032: step 9 architecture screening results (pre-registered spec)"`
   - `git push origin arena/01a0dc16-frontier-ai` (on an auth error, STOP and report; never force-push).
7. **Report:** write `out\arch\EXP-032\NIGHT_REPORT.txt` containing the step-2 reuse lines, the
   full `out\arch\EXP-032\SUMMARY.txt`, the commit hash, the push result, and PASS/FAIL lines
   for steps 1–6. Print it, then STOP.

## Hard rules

- Do NOT edit any repository file, the spec, or script arguments.
- Do NOT delete or modify anything under `out\exp_b\EXP-029\` (the baseline may be reused from it).
- Commit only `evals/results/EXP-032/**`. Push only to `arena/01a0dc16-frontier-ai`.
- Do NOT start any other training run.
