# PC TASK — Step 9 architecture confirmation, overnight (EXP-033)

**For:** the founder's Windows PC, repo `E:\frontier-ai`. **No chat agent is needed.**
Everything is in one script, because the PC agent (GitHub Copilot) was about to hit its
usage limit. The founder approved EXP-033 on 2026-09-28 ("I approve EXP-033").
**Expected time:** about 7.5–8.5 hours of CPU at 100 %, which is normal on the i3-4030U. Keep the
laptop plugged in, lid open, on a hard surface, and do not use it during the run.

## What the founder does (the only step)

1. Open VS Code on `E:\frontier-ai`, then **Terminal → New Terminal** (a PowerShell terminal
   opens in the repo folder).
2. Type this line and press Enter:

   ```
   git pull --ff-only origin arena/01a0dc16-frontier-ai; powershell -ExecutionPolicy Bypass -File scripts\run_night_unattended.ps1
   ```

3. Leave VS Code open (minimized is fine) and go to sleep. In the morning Notepad shows
   `out\arch\EXP-033\NIGHT_REPORT.txt`. Its last line starts with `RESULT:`. Tell the Arena
   chat "EXP-033 done", or paste the report there.

If the laptop restarted during the night, type the same line again. Finished models are
skipped and only the missing work runs.

## What the script does (`scripts/run_night_unattended.ps1`, default `-Exp EXP-033`)

1. **Checks** (it stops if any check fails):
   - no other ablation is running;
   - the repo is on branch `arena/01a0dc16-frontier-ai`;
   - `git status --porcelain` is empty;
   - the spec, runner, publisher, `.venv` and training data exist.
2. **Preflight:** a dry run (trains nothing), saved to `out\arch\EXP-033\preflight.txt`. It
   shows whether the EXP-032 cell `lr-0.006/seed-1337` is reused (`config verified`) or
   will be trained (+~25 min). Both are fine.
3. **Keeps the PC awake** while plugged in (`powercfg`, sleep and hibernate off on AC).
4. **Runs** `configs/ablations/EXP-033.json` with `scripts/run_arch_ablation.py`:
   - every line is logged to `out\arch\EXP-033\night.log`;
   - it retries ONCE automatically, and finished models are skipped.

   A model that blows up at the high learning rate is recorded as DIVERGED. That is a result
   under the pre-registered rules, not a crash.
5. **On success:**
   - publishes the small files to `evals\results\EXP-033\`;
   - checks that the only changes are new files there, stages only them, and re-checks the
     staged list;
   - commits `EXP-033: step 9 architecture confirmation results (pre-registered spec)` and
     pushes to `arena/01a0dc16-frontier-ai` (never force).

   If the push fails, the commit stays local and the report says so.
6. **Writes** `out\arch\EXP-033\NIGHT_REPORT.txt`: PASS/FAIL lines, reuse lines, the full
   `SUMMARY.txt`, the commit hash and the push result. It then opens the report in Notepad.

## If a chat agent is available instead

Run exactly the line in step 2 above and report the contents of `NIGHT_REPORT.txt`. Do NOT
edit any repository file, the spec or script arguments. Do NOT delete anything under
`out\arch\EXP-032\` or `out\exp_b\EXP-029\` (inputs and reused cell). Do NOT start any other
training run.
