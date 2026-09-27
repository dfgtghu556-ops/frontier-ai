# scripts/run_exp_b_night.ps1
#
# EXP-B (EXP-029) overnight runner for the PC (Windows / PowerShell 5.1+).
#
# What it does, in order:
#   1. Refuses to start if a run_exp_b.py is already running (avoids stacked jobs).
#   2. Runs the pre-registered 2-tokenizer x 3-seed matrix at the revised
#      150-step budget (~1.9 h on the founder's CPU), echoing every line to the
#      console AND appending it to out\exp_b\EXP-029\night.log.
#      Note: each cell runs silently for ~19 min, then prints its log all at
#      once - quiet stretches are normal, not a hang.
#   3. Optional (-Verify, ~2 h more): re-runs the identical matrix in a fresh
#      process and checks that the second independent run reproduces the same
#      table (cross-process reproducibility proof; deterministic mode must make
#      all 6 cells bit-identical).
#   4. Writes a completion marker with the exit code into night.log.
#
# Usage (from anywhere; the script cd's to the repo root itself):
#   powershell -ExecutionPolicy Bypass -File scripts\run_exp_b_night.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_exp_b_night.ps1 -Verify
#
# Before running: keep the PC awake for the night and keep this window open
# (minimized is fine, do NOT close it):
#   powercfg /change standby-timeout-ac 0
#
# Safe to interrupt (Ctrl+C) and re-run: the matrix is deterministic, so a
# restart reproduces the same numbers and simply overwrites the cell records.

param([switch]$Verify)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$log = "out\exp_b\EXP-029\night.log"
$runsDir = "out\exp_b\EXP-029\runs"
New-Item -ItemType Directory -Force "out\exp_b\EXP-029" | Out-Null

function Add-Log($text) { $text | Out-File $log -Append -Encoding utf8 }

# Run the matrix and tee every output line to the console + night.log.
# WHY cmd /c "... 2>&1": Windows PowerShell 5.1 turns every stderr line of a
# native program into a NativeCommandError, and under $ErrorActionPreference
# "Stop" the FIRST such line kills the whole pipeline. train.py prints its normal
# success line "[record] ... | fingerprint ..." on stderr, so the first finished
# cell aborted the night run (observed on the PC 2026-09-27). Letting cmd.exe
# merge stderr into stdout at the OS level means PowerShell only ever sees plain
# stdout text. "Continue" is a second safety net for the duration of the call.
function Invoke-Matrix {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & cmd.exe /d /c ".venv\Scripts\python.exe -u scripts\run_exp_b.py --seeds 1337,1338,1339 --max-steps 150 2>&1" |
            ForEach-Object { $line = "$_"; $line | Out-File $log -Append -Encoding utf8; Write-Host $line }
        $rc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    # Only the exit code leaves this function (console echo uses Write-Host so
    # the output lines do not leak into the return value).
    return $rc
}

# --- 1) refuse to stack a second matrix on top of a running one -------------
$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*run_exp_b.py*" }
if ($running) {
    $ids = ($running.ProcessId -join ", ")
    Write-Host "STOP: a run_exp_b.py process is already running (PID $ids)."
    Write-Host "Do NOT start another matrix. If that run is stale, close it"
    Write-Host "(Ctrl+C in its window, or Stop-Process on that PID) and re-run."
    exit 2
}

# --- 2) the matrix (pre-registered cells + revised 150-step budget) ----------
"=== EXP-B night run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Log
$code = Invoke-Matrix
"=== matrix finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | exit code: $code ===" | Add-Log

if ($code -ne 0) {
    "NIGHT RUN FAILED at the matrix step (exit $code)." | Add-Log
    "Paste the last 30 lines of night.log to Arena." | Add-Log
    Write-Host ""
    Write-Host "FAILED (exit $code). The log is kept at: $log"
    exit $code
}
"NIGHT RUN matrix COMPLETE - table at $runsDir\report.txt" | Add-Log

# --- 3) optional: independent second run + table comparison -----------------
if ($Verify) {
    Copy-Item -Recurse -Force $runsDir "out\exp_b\EXP-029\runs_pass1"
    "=== verification pass 2 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Log
    $code2 = Invoke-Matrix
    "=== verification pass 2 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | exit code: $code2 ===" | Add-Log
    if ($code2 -ne 0) {
        "VERIFICATION FAILED at the second run (exit $code2)." | Add-Log
        "The pass-1 table is safe in out\exp_b\EXP-029\runs_pass1\report.txt" | Add-Log
        exit $code2
    }
    # compare the two reports, ignoring the 'generated:' timestamp line
    $p1 = Get-Content (Join-Path $runsDir "report.txt") | Where-Object { $_ -notmatch "^generated:" }
    $p2 = Get-Content "out\exp_b\EXP-029\runs_pass1\report.txt" | Where-Object { $_ -notmatch "^generated:" }
    $diff = Compare-Object $p1 $p2
    if ($diff) {
        "VERIFICATION: tables DIFFER between the two independent runs - see night.log" | Add-Log
        $diff | Add-Log
        exit 1
    }
    "VERIFICATION PASSED: an independent second run reproduced the identical table." | Add-Log
}

"NIGHT RUN COMPLETE - success. Final table: $runsDir\report.txt" | Add-Log
Write-Host ""
Write-Host "Done. The final table is at the end of $log and in $runsDir\report.txt"
Write-Host "In the morning, run:  Get-Content $log -Tail 40   and paste it to Arena."
