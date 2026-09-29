# scripts/run_data_night.ps1
#
# ONE-COMMAND overnight runner for FrontierCorpus v2 data work on the founder's Windows PC
# (D-044, EXP-034). It needs NO chat agent. In a fixed order, stopping on any problem:
#
#   1. checks: right branch, clean git tree, .venv, pin file, protected suite; installs the
#      one extra library it needs (pyarrow, to read parquet files) if it is missing;
#   2. keeps the PC awake while plugged in;
#   3. downloads the pinned Sangraha Verified slice (13 files, ~5.1 GB) into data\sangraha\,
#      checking free disk space first and the SHA-256 fingerprint of every file after
#      (scripts\fetch_sangraha_slice.py); a dropped connection resumes, with ONE extra retry;
#   4. inspects every file WITHOUT filtering anything (scripts\inspect_sangraha_slice.py):
#      documents, characters, wrong-script share, duplicates, overlap with the protected
#      evaluation suite, and a token estimate with the frozen v1 tokenizer;
#   5. publishes the small result files to evals\results\<EXP>\, checks that nothing else
#      changed, commits ONLY those files and pushes to this branch;
#   6. writes out\data\<EXP>\NIGHT_REPORT.txt and opens it in Notepad.
#
# Usage (from the repo root, e.g. the VS Code terminal):
#   powershell -ExecutionPolicy Bypass -File scripts\run_data_night.ps1
#
# Safe to run again after an interruption: verified files are not downloaded again and a
# half-downloaded file continues where it stopped. It never deletes anything, never edits a
# tracked file, never force-pushes, and pushes only to arena/01a0dc16-frontier-ai. The
# downloaded data stays in data\ (git-ignored); only the small reports are committed.
#
# This file is ASCII-only on purpose: Windows PowerShell 5.1 reads BOM-less scripts
# with the ANSI code page.

param([string]$Exp = "EXP-034")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$branch = "arena/01a0dc16-frontier-ai"
$pins = "corpora\frontier\v2\sangraha_slice1.json"
$suite = "evals\suites\frontier-heldout-v1\SUITE.json"
$outDir = "out\data\$Exp"
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

Add-Report "NIGHT REPORT $Exp (Sangraha Verified slice: download + inspect) - started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Add-Report ""

# --- 1) checks ----------------------------------------------------------------------
$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*sangraha_slice.py*" }
if ($running) {
    Write-Host "STOP: a Sangraha download/inspection is already running (PID $($running.ProcessId -join ', '))."
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

foreach ($f in @($pins, $suite, "scripts\fetch_sangraha_slice.py", "scripts\inspect_sangraha_slice.py",
                 "scripts\publish_eval_results.py", $python)) {
    if (-not (Test-Path $f)) { Stop-Night "missing file: $f" 2 }
}
Add-Report "PASS 1c: pin file, protected suite, scripts and .venv found"

# No embedded quotes on purpose (PowerShell 5.1 mangles them on the way to cmd.exe):
# "pip show" exits 1 when the package is missing. The version is the one the tests ran with.
$r = Invoke-Logged "$python -m pip show --disable-pip-version-check pyarrow"
if ($r.Code -ne 0) {
    Add-Report "INFO 1d: pyarrow (the parquet reader) is not installed yet; installing it now"
    $r = Invoke-Logged "$python -m pip install --disable-pip-version-check pyarrow==25.0.1"
    if ($r.Code -ne 0) { Stop-Night "could not install pyarrow (see the log lines above)" 2 }
    $r = Invoke-Logged "$python -m pip show --disable-pip-version-check pyarrow"
    if ($r.Code -ne 0) { Stop-Night "pyarrow still missing after installing it" 2 }
}
$ver = @($r.Lines | Where-Object { $_ -like "Version:*" }) -join " "
Add-Report "PASS 1d: pyarrow installed ($ver)"

if (Test-Path "corpora\frontier\v1\shards\heldout") {
    Add-Report "PASS 1e: held-out shard folder found (the protected-suite check will run)"
} else {
    Add-Report "WARN 1e: corpora\frontier\v1\shards\heldout not found - the suite check will say NOT CHECKED"
}

# --- 2) keep the PC awake while plugged in (non-fatal) ------------------------------
try {
    & powercfg /change standby-timeout-ac 0
    & powercfg /change hibernate-timeout-ac 0
    Add-Report "PASS 2: sleep and hibernate disabled while plugged in"
} catch {
    Add-Report "WARN 2: could not change power settings ($_); keep the laptop awake manually"
}

# --- 3) download with fingerprint check (one extra retry; downloads resume) -----------
Add-Report ""
Add-Report "=== download started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (about 5.1 GB; progress is printed once a minute) ==="
$code = (Invoke-Logged "$python -u scripts\fetch_sangraha_slice.py --pins $pins").Code
if ($code -eq 1) {
    Add-Report "=== download attempt 1 did not finish (exit 1); retrying once in 2 minutes - it resumes ==="
    Start-Sleep -Seconds 120
    $code = (Invoke-Logged "$python -u scripts\fetch_sangraha_slice.py --pins $pins").Code
}
Add-Report "=== download finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | exit code: $code ==="
if ($code -eq 2) { Stop-Night "the download did not start (disk space or pin file problem; see the lines above)" 2 }
if ($code -ne 0) { Stop-Night "the download did not complete; run the same line again tomorrow - it resumes" 1 }
Add-Report "PASS 3: all files downloaded and their SHA-256 fingerprints match the pins"

# --- 4) inspection (measures only; filters nothing) ----------------------------------
Add-Report ""
Add-Report "=== inspection started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (CPU near 100% is normal; progress every 50,000 documents) ==="
$code = (Invoke-Logged "$python -u scripts\inspect_sangraha_slice.py --exp-id $Exp --pins $pins --out $outDir").Code
Add-Report "=== inspection finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | exit code: $code ==="
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
    Add-Report "FAIL 4: inspection failed (exit $code). Last 60 log lines:"
    foreach ($line in (Get-Content $log -Tail 60 -Encoding UTF8)) { $report.Add("  $line") }
    Stop-Night "the inspection did not finish cleanly" 1
}
Add-Report "PASS 4: inspection complete (exit 0)"

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
    "${Exp}: Sangraha Verified slice-1 inspection results (pre-registered, measures only)" |
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
Add-Report "RESULT: COMPLETE - finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'). Tell the Arena chat: EXP-034 done (or paste this report)."
Save-Report
exit 0
