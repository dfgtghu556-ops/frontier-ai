# scripts/run_arch_ablation_night.ps1
#
# Step 9 (EXP-032) overnight runner for the PC (Windows / PowerShell 5.1+).
#
# What it does, in order:
#   1. Refuses to start if a run_arch_ablation.py is already running (no stacked jobs).
#   2. Runs the pre-registered ablation (configs/ablations/EXP-032.json): trains each
#      model, grades its final checkpoint with the evaluation harness, compares every
#      variant with the baseline and writes SUMMARY.txt. Every line goes to the console
#      AND to out\arch\EXP-032\night.log. Each model trains silently for ~19 min, then
#      prints its log at once; quiet stretches are normal, not a hang.
#   3. If step 2 fails, retries ONCE automatically. That is safe because finished models
#      are skipped and only unfinished work runs again.
#   4. Writes a completion marker with the exit code into night.log.
#
# Usage (the script cd's to the repo root itself):
#   powershell -ExecutionPolicy Bypass -File scripts\run_arch_ablation_night.ps1
#
# Keep the PC awake and this window open (minimized is fine):
#   powercfg /change standby-timeout-ac 0

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$spec = "configs\ablations\EXP-032.json"
$outDir = "out\arch\EXP-032"
$log = "$outDir\night.log"
New-Item -ItemType Directory -Force $outDir | Out-Null

function Add-Log($text) { $text | Out-File $log -Append -Encoding utf8 }

# WHY cmd /c "... 2>&1": Windows PowerShell 5.1 turns every stderr line of a native
# program into a NativeCommandError, and under "Stop" the first such line kills the
# pipeline (observed on the PC in EXP-029). cmd.exe merges stderr into stdout first.
function Invoke-Ablation {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & cmd.exe /d /c ".venv\Scripts\python.exe -u scripts\run_arch_ablation.py --spec $spec 2>&1" |
            ForEach-Object { $line = "$_"; $line | Out-File $log -Append -Encoding utf8; Write-Host $line }
        $rc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    return $rc
}

# --- 1) refuse to stack a second run -----------------------------------------------
$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*run_arch_ablation.py*" }
if ($running) {
    Write-Host "STOP: a run_arch_ablation.py process is already running (PID $($running.ProcessId -join ', '))."
    Write-Host "Do NOT start another one. If it is stale, close it and re-run this script."
    exit 2
}

# --- 2) the ablation, with one automatic retry ---------------------------------------
"=== EXP-032 night run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Log
$code = Invoke-Ablation
"=== attempt 1 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | exit code: $code ===" | Add-Log
if ($code -eq 1) {
    "=== retrying once (finished models are skipped) ===" | Add-Log
    $code = Invoke-Ablation
    "=== attempt 2 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | exit code: $code ===" | Add-Log
}

if ($code -ne 0) {
    "NIGHT RUN FAILED (exit $code). The summary of what finished is in $outDir\SUMMARY.txt" | Add-Log
    Write-Host "FAILED (exit $code). Log: $log"
    exit $code
}
"NIGHT RUN COMPLETE - success. Summary: $outDir\SUMMARY.txt" | Add-Log
Write-Host ""
Write-Host "Done. Summary: $outDir\SUMMARY.txt   Log: $log"
