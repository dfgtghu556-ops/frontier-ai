# scripts/run_night_unattended.ps1
#
# ONE-COMMAND overnight runner for the founder's Windows PC. It needs NO chat agent
# (written for EXP-033, when the PC agent's usage limit ran out). It does everything the
# PC task file used to ask the agent to do, in a fixed order, and STOPS on any problem:
#
#   1. checks: right branch, clean git tree, spec / runner / data / .venv exist, and no
#      other ablation is already running;
#   2. preflight: a dry run that trains nothing (saved to preflight.txt);
#   3. keeps the PC awake while plugged in;
#   4. runs the pre-registered ablation (configs\ablations\<EXP>.json), logging every line
#      to out\arch\<EXP>\night.log, with ONE automatic retry (finished models are skipped);
#   5. if it succeeded: publishes the small result files to evals\results\<EXP>\, checks
#      that nothing else changed, commits ONLY those files and pushes to this branch;
#   6. writes out\arch\<EXP>\NIGHT_REPORT.txt and opens it in Notepad.
#
# Usage (from the repo root, e.g. the VS Code terminal):
#   powershell -ExecutionPolicy Bypass -File scripts\run_night_unattended.ps1
#   (optional: -Exp EXP-033; the default is EXP-033)
#
# Safe to run again after an interruption (power cut, restart): finished models are
# skipped and only the missing work runs. It never deletes anything, never edits a
# tracked file, never force-pushes, and pushes only to arena/01a0dc16-frontier-ai.
#
# This file is ASCII-only on purpose: Windows PowerShell 5.1 reads BOM-less scripts
# with the ANSI code page.

param([string]$Exp = "EXP-033")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$branch = "arena/01a0dc16-frontier-ai"
$spec = "configs\ablations\$Exp.json"
$outDir = "out\arch\$Exp"
$log = "$outDir\night.log"
$reportPath = "$outDir\NIGHT_REPORT.txt"
$python = ".venv\Scripts\python.exe"
$resultsPrefix = "evals/results/$Exp/"
New-Item -ItemType Directory -Force $outDir | Out-Null

$report = New-Object System.Collections.Generic.List[string]
function Add-Report([string]$text) { $report.Add($text); Write-Host $text; $text | Out-File $log -Append -Encoding utf8 }
function Save-Report {
    $report | Set-Content -Path $reportPath -Encoding UTF8
    Write-Host ""
    Write-Host "Report written to $reportPath (opening it in Notepad)."
    try { Start-Process notepad.exe -ArgumentList $reportPath } catch { }
}
function Stop-Night([string]$why, [int]$code) {
    Add-Report "RESULT: STOPPED - $why"
    Add-Report "Nothing was committed or pushed by this step. Open the report and paste it into the Arena chat."
    Save-Report
    exit $code
}

# WHY cmd /c "... 2>&1": Windows PowerShell 5.1 turns every stderr line of a native
# program (python, git) into a NativeCommandError, and under "Stop" the first such line
# kills the script (observed on the PC in EXP-029). cmd.exe merges stderr into stdout.
function Invoke-Logged([string]$cmdline) {
    $lines = New-Object System.Collections.Generic.List[string]
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & cmd.exe /d /c "$cmdline 2>&1" |
            ForEach-Object { $line = "$_"; $lines.Add($line); $line | Out-File $log -Append -Encoding utf8; Write-Host $line }
        $rc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    return New-Object PSObject -Property @{ Code = $rc; Lines = $lines }
}

Add-Report "NIGHT REPORT $Exp - started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Add-Report ""

# --- 1) checks ----------------------------------------------------------------------
$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*run_arch_ablation.py*" }
if ($running) {
    Write-Host "STOP: a run_arch_ablation.py process is already running (PID $($running.ProcessId -join ', '))."
    Write-Host "Do NOT start another one. Wait for it, or close it if it is stale, then run this again."
    exit 2
}

$r = Invoke-Logged "git rev-parse --abbrev-ref HEAD"
if ($r.Code -ne 0 -or ($r.Lines -join "").Trim() -ne $branch) {
    Stop-Night "the repo is not on branch $branch (got: $($r.Lines -join ' '))" 2
}
Add-Report "PASS 1a: on branch $branch"

$r = Invoke-Logged "git status --porcelain"
if ($r.Code -ne 0 -or ($r.Lines | Where-Object { $_.Trim() -ne "" })) {
    Stop-Night "the git tree is not clean: $($r.Lines -join ' | ')" 2
}
Add-Report "PASS 1b: git tree clean"

foreach ($f in @($spec, "scripts\run_arch_ablation.py", "scripts\publish_eval_results.py", $python)) {
    if (-not (Test-Path $f)) { Stop-Night "missing file: $f" 2 }
}
$dataPath = (Get-Content $spec -Raw -Encoding UTF8 | ConvertFrom-Json).data
if (-not (Test-Path $dataPath)) { Stop-Night "missing training data: $dataPath" 2 }
Add-Report "PASS 1c: spec, runner, publisher, .venv and data ($dataPath) found"

# --- 2) preflight (trains nothing) ----------------------------------------------------
$r = Invoke-Logged "$python -u scripts\run_arch_ablation.py --spec $spec --dry-run"
$r.Lines | Set-Content -Path "$outDir\preflight.txt" -Encoding UTF8
if ($r.Code -ne 0) { Stop-Night "preflight dry run failed (exit $($r.Code)); see $outDir\preflight.txt" 2 }
Add-Report "PASS 2: preflight dry run (exit 0)"
foreach ($line in $r.Lines) {
    if ($line -match "reusing|NOT reused|cells, .* to train") { Add-Report "  $line" }
}

# --- 3) keep the PC awake while plugged in (non-fatal) ------------------------------
try {
    & powercfg /change standby-timeout-ac 0
    & powercfg /change hibernate-timeout-ac 0
    Add-Report "PASS 3: sleep and hibernate disabled while plugged in"
} catch {
    Add-Report "WARN 3: could not change power settings ($_); keep the laptop awake manually"
}

# --- 4) the ablation, with one automatic retry -----------------------------------------
Add-Report ""
Add-Report "=== $Exp run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (each model trains silently for ~20-30 min; quiet stretches are normal) ==="
$code = (Invoke-Logged "$python -u scripts\run_arch_ablation.py --spec $spec").Code
Add-Report "=== attempt 1 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | exit code: $code ==="
if ($code -eq 1) {
    Add-Report "=== retrying once (finished models are skipped) ==="
    $code = (Invoke-Logged "$python -u scripts\run_arch_ablation.py --spec $spec").Code
    Add-Report "=== attempt 2 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | exit code: $code ==="
}
Add-Report ""
Add-Report "----- SUMMARY.txt -----"
if (Test-Path "$outDir\SUMMARY.txt") {
    foreach ($line in (Get-Content "$outDir\SUMMARY.txt" -Encoding UTF8)) { Add-Report $line }
} else {
    Add-Report "(no SUMMARY.txt)"
}
Add-Report "-----------------------"
Add-Report ""
if ($code -ne 0) {
    Add-Report "FAIL 4: ablation run failed (exit $code). Last 60 log lines:"
    foreach ($line in (Get-Content $log -Tail 60 -Encoding UTF8)) { $report.Add("  $line") }
    Stop-Night "the ablation did not finish cleanly" 1
}
Add-Report "PASS 4: ablation complete (exit 0)"

# --- 5) publish, check, commit and push ONLY evals/results/<EXP>/ -----------------------
$r = Invoke-Logged "$python -u scripts\publish_eval_results.py --exp-id $Exp --src $outDir"
if ($r.Code -ne 0) { Stop-Night "publishing failed (exit $($r.Code))" 1 }
Add-Report "PASS 5a: published to $resultsPrefix"

$r = Invoke-Logged "git status --porcelain --untracked-files=all"
$changed = @($r.Lines | Where-Object { $_.Trim() -ne "" })
if ($changed.Count -eq 0) {
    Add-Report "PASS 5b: nothing new to commit (results were already committed earlier)"
} else {
    $bad = @($changed | Where-Object { -not $_.StartsWith("?? $resultsPrefix") })
    if ($bad.Count -gt 0) { Stop-Night "unexpected changes outside ${resultsPrefix}: $($bad -join ' | ')" 1 }
    Add-Report "PASS 5b: only new files under $resultsPrefix ($($changed.Count) files)"

    $r = Invoke-Logged "git add $resultsPrefix"
    if ($r.Code -ne 0) { Stop-Night "git add failed" 1 }
    $r = Invoke-Logged "git diff --cached --name-only"
    $staged = @($r.Lines | Where-Object { $_.Trim() -ne "" })
    $badStaged = @($staged | Where-Object { -not $_.StartsWith($resultsPrefix) })
    if ($r.Code -ne 0 -or $staged.Count -eq 0 -or $badStaged.Count -gt 0) {
        Invoke-Logged "git reset -q" | Out-Null
        Stop-Night "staged files are not exactly $resultsPrefix (unstaged again): $($badStaged -join ' | ')" 1
    }
    Add-Report "PASS 5c: staged $($staged.Count) files, all under $resultsPrefix"

    # The message goes through a file: PowerShell 5.1 mangles embedded quotes passed to cmd.exe.
    $msgFile = "$outDir\commit_message.txt"
    "${Exp}: step 9 architecture confirmation results (pre-registered spec)" |
        Set-Content -Path $msgFile -Encoding ASCII
    $r = Invoke-Logged "git commit -q -F $msgFile"
    if ($r.Code -ne 0) { Stop-Night "git commit failed: $($r.Lines -join ' | ')" 1 }
    $hash = ((Invoke-Logged "git rev-parse --short HEAD").Lines -join "").Trim()
    Add-Report "PASS 5d: committed $hash"
}

$r = Invoke-Logged "git push origin $branch"
if ($r.Code -ne 0) {
    Add-Report "FAIL 5e: git push failed (the results ARE committed locally; nothing is lost):"
    foreach ($line in $r.Lines) { Add-Report "  $line" }
    Add-Report "RESULT: run complete, push FAILED - paste this report into the Arena chat."
    Save-Report
    exit 1
}
Add-Report "PASS 5e: pushed to origin/$branch"
Add-Report ""
Add-Report "RESULT: COMPLETE - finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'). Tell the Arena chat: EXP-033 done (or paste this report)."
Save-Report
exit 0
