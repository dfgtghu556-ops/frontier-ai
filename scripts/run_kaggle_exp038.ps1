# scripts/run_kaggle_exp038.ps1
#
# ONE-COMMAND runner for EXP-038 (step 10: first GPU training bring-up on ONE free Kaggle T4 GPU)
# and, with -Exp EXP-039, for its follow-up (see below).
# It needs NO chat agent and can be started again at any time: it remembers where it stopped
# (out\kaggle\EXP-038\state.json) and continues from there.
#
#   1. checks: right branch, clean git tree, this PC's commit is the one on GitHub (Kaggle clones
#      the PUBLIC repository at exactly this commit), .venv, the Hindi token file hi.bin from
#      EXP-037 with its SHA-256 fingerprint;
#   2. Kaggle access: installs the Kaggle command-line tool into .venv if it is missing; the FIRST
#      time only it asks for your Kaggle API token (typed input is hidden; it is stored in YOUR
#      Windows user settings, never in the repository and never shown) and your Kaggle username;
#   3. uploads hi.bin + hi.meta.json (about 400 MB) ONCE as a PRIVATE Kaggle dataset
#      <username>/frontier-v2-hi-tok2 and waits until Kaggle says it is ready;
#   4. launches a PRIVATE Kaggle script kernel <username>/frontier-exp038 on one T4 GPU. It clones
#      the repository at the pinned commit and runs scripts\gpu_bringup.py (tests first, then the
#      pre-registered EXP-038 parts; hard limit 5.5 hours);
#   5. waits (checks every 10 minutes, up to 9 hours; the PC may stay idle but must stay on and
#      online) until the kernel has finished, downloads its small result files, checks them,
#      publishes them to evals\results\EXP-038\, commits ONLY that folder and pushes;
#   6. writes out\kaggle\EXP-038\REPORT.txt and opens it in Notepad.
#
# Usage (from the repo root, e.g. the VS Code terminal):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp038.ps1
# If the laptop was closed while the GPU run was going: run the same line again - it only checks
# on the kernel and continues. Add -Relaunch only if the report tells you to (a failed kernel).
#
# EXP-039 (the float64 CPU-vs-GPU follow-up) reuses this runner with -Exp EXP-039 (see
# scripts\run_kaggle_exp039.ps1): its own state folder out\kaggle\EXP-039, its own kernel
# <username>/frontier-exp039 from scripts\kaggle\exp039_kernel.py, the SAME private dataset (no new
# upload; the fingerprint is still checked) and the Kaggle username remembered from EXP-038.
# EXP-040 (step 11 phase 1: learning rate + RoPE/GQA-2 on all 13 languages; scripts\run_kaggle_exp040.ps1)
# also reuses it with -Exp EXP-040, but needs ALL 13 EXP-037 token files: each one's fingerprint is
# checked against the manifest, they are staged with hard links (no 5.6 GB copy; a plain copy only
# if hard links are impossible) and uploaded ONCE as a second PRIVATE dataset
# <username>/frontier-v2-tok2-13lang (about 5.6 GB; the upload time depends on the internet line).
# Its kernel has a hard limit of 9 GPU-hours, so the runner waits up to 11 hours.
# EXP-041 (the EXP-040 follow-up: one-step float64 check, then the RoPE + GQA-2 runs;
# scripts\run_kaggle_exp041.ps1) uses -Exp EXP-041 with the SAME 13-language dataset (fingerprints
# checked again, no new upload) and a hard limit of 5 GPU-hours, so the runner waits up to 7 hours.
# EXP-042 (the IsoFLOP ladder; scripts\run_kaggle_exp042.ps1) uses -Exp EXP-042 with the SAME
# 13-language dataset. It is MULTI-SESSION: each launch is one kernel of at most 9 GPU-hours that
# writes EXP-042\session-<n>\; the runner downloads into a folder named after the launched commit,
# publishes that session folder to evals/results/EXP-042/session-<n>/, pushes, and then forgets the
# launched commit so that the SAME line launches the next session (which reads the earlier sessions
# from the repository). It waits up to 11 hours per session.
# EXP-043 (step 12: train the D-049 model; scripts\run_kaggle_exp043.ps1) uses -Exp EXP-043 with the
# SAME 13-language dataset and works like EXP-042 (one launch = one session of at most 9 GPU-hours),
# with three differences. (1) Two private kernels take turns, <username>/frontier-exp043-a for odd
# session numbers and -b for even ones; each one gets the OTHER kernel's latest output mounted as an
# input (Kaggle "kernel_sources"), which carries the checkpoint (about 2.3 GB) from one session to
# the next without ever coming to this PC. (2) Only the small result files are downloaded
# (kaggle kernels output --file-pattern); if this Kaggle tool cannot do that, the whole output is
# downloaded instead and the report says so. At the end, model_final.pt (about 760 MB) is downloaded
# once into out\kaggle\EXP-043\ (never committed) and its sha256 checked. (3) If a session's
# checkpoint-chain check failed, its results are NOT published (the session number does not
# advance, so a retry reads the same checkpoint again).
# If the final push is rejected because the branch moved meanwhile (seen once in EXP-038), it pulls
# with --rebase once and pushes again; the results commit still touches only evals/results/<EXP>/.
#
# It never deletes anything, never edits a tracked file, never force-pushes, and pushes only to
# arena/01a0dc16-frontier-ai. Checkpoints stay on Kaggle's temporary disk and vanish with the
# session; only summary.json, SUMMARY.txt and samples.jsonl come back. No money is spent: Kaggle's
# free GPU quota (about 30 GPU-hours a week) is used, this run at most about 6.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([ValidateSet("EXP-038", "EXP-039", "EXP-040", "EXP-041", "EXP-042", "EXP-043")][string]$Exp = "EXP-038", [switch]$Relaunch)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$expSlug = $Exp.ToLower().Replace("-", "")
$branch = "arena/01a0dc16-frontier-ai"
$python = ".venv\Scripts\python.exe"
$kaggle = ".venv\Scripts\kaggle.exe"
$dataDir = "data\frontier_v2\sangraha-slice1-v2-tok2"
$manifestPath = "evals\results\EXP-037\manifest.json"
$template = "scripts\kaggle\${expSlug}_kernel.py"
$outDir = "out\kaggle\$Exp"
$statePath = "$outDir\state.json"
$log = "$outDir\run.log"
$reportPath = "$outDir\REPORT.txt"
$resultsPrefix = "evals/results/$Exp/"
$datasetSlug = "frontier-v2-hi-tok2"
$kernelSlug = "frontier-$expSlug"
$userFile = "out\kaggle\username.txt"
$pollSeconds = 600
$maxWaitHours = 9
# which token files go into the private dataset (EXP-038/039: Hindi only; EXP-040: all 13 languages)
$dataFiles = @("hi.bin", "hi.meta.json")
$datasetSubtitle = "Hindi training tokens for the frontier-ai project (EXP-037)"
$datasetText = "Hindi part of FrontierCorpus v2-slice1"
$uploadSize = "about 400 MB"
$expectedRun = "EXP-038 took about 80 minutes, EXP-039 should be shorter"
if ($Exp -eq "EXP-040" -or $Exp -eq "EXP-041" -or $Exp -eq "EXP-042" -or $Exp -eq "EXP-043") {
    $datasetSlug = "frontier-v2-tok2-13lang"
    $dataFiles = @()
    foreach ($f in (Get-Content $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json).files) { $dataFiles += @($f.path, $f.meta) }
    $datasetSubtitle = "13-language training tokens for the frontier-ai project (EXP-037)"
    $datasetText = "All 13 language files of FrontierCorpus v2-slice1"
    $uploadSize = "about 5.6 GB"
    $expectedRun = "EXP-040 is expected to take about 7 hours, at most 9"
    $maxWaitHours = 11
}
if ($Exp -eq "EXP-041") {
    $expectedRun = "EXP-041 is expected to take about 4 hours, at most 5"
    $maxWaitHours = 7
}
if ($Exp -eq "EXP-042") {
    $expectedRun = "one EXP-042 session takes at most 9 hours; the ladder needs about 3 sessions"
    $maxWaitHours = 11
}
if ($Exp -eq "EXP-043") {
    $expectedRun = "one EXP-043 session takes at most 9 hours; about 10 sessions in total"
    $maxWaitHours = 11
}
New-Item -ItemType Directory -Force $outDir | Out-Null

$report = New-Object System.Collections.Generic.List[string]
function Add-Report([string]$text) { $report.Add($text); Write-Host $text; $text | Out-File $log -Append -Encoding utf8 }
function Save-Report {
    $report | Set-Content -Path $reportPath -Encoding UTF8
    Write-Host ""
    Write-Host "Report written to $reportPath (opening it in Notepad)."
    try { Start-Process notepad.exe -ArgumentList $reportPath } catch { }
}
function Stop-Run([string]$why, [int]$code) {
    Add-Report "RESULT: STOPPED - $why"
    Add-Report "Nothing was committed or pushed by this step. Open the report and paste it into the Arena chat."
    Save-Report
    exit $code
}

# WHY cmd /c "... 2>&1": Windows PowerShell 5.1 turns every stderr line of a native program into a
# NativeCommandError, and under "Stop" the first such line kills the script. cmd.exe merges them.
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

function Read-State {
    if (Test-Path $statePath) { return (Get-Content $statePath -Raw -Encoding UTF8 | ConvertFrom-Json) }
    return New-Object PSObject -Property @{ username = ""; dataset_ready = $false; pinned_commit = ""; pushed_at = "" }
}
function Save-State($s) { $s | ConvertTo-Json | Set-Content -Path $statePath -Encoding ASCII }
# Kaggle's tool reads its JSON files with Python; they must not start with a byte-order mark.
function Write-Ascii([string]$path, [string]$text) { [System.IO.File]::WriteAllText((Join-Path (Get-Location) $path), $text) }

Add-Report "REPORT $Exp (GPU run on one free Kaggle T4) - started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Add-Report ""
$state = Read-State

# --- 1) checks ------------------------------------------------------------------------------
$r = Invoke-Logged "git rev-parse --abbrev-ref HEAD"
if ($r.Code -ne 0 -or ($r.Lines -join "").Trim() -ne $branch) { Stop-Run "the repo is not on branch $branch (got: $($r.Lines -join ' '))" 2 }
Add-Report "PASS 1a: on branch $branch"

$r = Invoke-Logged "git status --porcelain"
if ($r.Code -ne 0 -or ($r.Lines | Where-Object { $_.Trim() -ne "" })) { Stop-Run "the git tree is not clean: $($r.Lines -join ' | ')" 2 }
Add-Report "PASS 1b: git tree clean"

foreach ($f in @($python, $template, $manifestPath, "scripts\gpu_bringup.py", "scripts\publish_eval_results.py")) {
    if (-not (Test-Path $f)) { Stop-Run "missing file: $f (run git pull first?)" 2 }
}
Add-Report "PASS 1c: .venv and scripts found"

$launching = (-not $state.pinned_commit) -or $Relaunch
if ($launching) {
    $r = Invoke-Logged "git fetch -q origin $branch"
    if ($r.Code -ne 0) { Stop-Run "git fetch failed - is the PC online?" 2 }
    $head = ((Invoke-Logged "git rev-parse HEAD").Lines -join "").Trim()
    $remote = ((Invoke-Logged "git rev-parse FETCH_HEAD").Lines -join "").Trim()
    if ($head -ne $remote) {
        Stop-Run "this PC's commit ($head) is not the one on GitHub ($remote). Run: git pull   then run this line again" 2
    }
    Add-Report "PASS 1d: this PC's commit $($head.Substring(0, 7)) is on GitHub (Kaggle will run exactly this commit)"
}

if (-not $state.dataset_ready) {
    foreach ($f in $dataFiles) {
        if (-not (Test-Path "$dataDir\$f")) { Stop-Run "missing $dataDir\$f - the EXP-037 token files must be on this PC" 2 }
    }
    $manifestFiles = (Get-Content $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json).files
    $bins = @($dataFiles | Where-Object { $_ -like "*.bin" })
    foreach ($b in $bins) {
        $expected = ($manifestFiles | Where-Object { $_.path -eq $b }).sha256
        Write-Host "  checking the fingerprint of $b ..."
        $actual = (Get-FileHash -Algorithm SHA256 "$dataDir\$b").Hash
        if ($actual -ine $expected) { Stop-Run "$b fingerprint $actual does not match the EXP-037 manifest ($expected)" 2 }
    }
    Add-Report "PASS 1e: $($bins.Count) token file(s) match their EXP-037 fingerprints ($($bins -join ', '))"
}

# --- 2) Kaggle access ------------------------------------------------------------------------
if (-not (Test-Path $kaggle)) {
    Add-Report "INFO 2a: installing the Kaggle command-line tool into .venv"
    $r = Invoke-Logged "$python -m pip install --disable-pip-version-check --upgrade kaggle"
    if ($r.Code -ne 0 -or -not (Test-Path $kaggle)) { Stop-Run "could not install the Kaggle tool (see the log lines above)" 2 }
}
$ver = ((Invoke-Logged "$python -m pip show --disable-pip-version-check kaggle").Lines | Where-Object { $_ -like "Version:*" }) -join " "
Add-Report "PASS 2a: Kaggle tool installed ($ver)"

$legacyJson = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not $env:KAGGLE_API_TOKEN) { $env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN", "User") }
if (-not $env:KAGGLE_API_TOKEN -and -not (Test-Path $legacyJson)) {
    Write-Host ""
    Write-Host "FIRST TIME ONLY: paste your Kaggle API token and press Enter (right-click pastes; nothing is shown)."
    $secure = Read-Host "Kaggle API token" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    $plain = $plain.Trim()
    if (-not $plain) { Stop-Run "no token was entered" 2 }
    [Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", $plain, "User")
    $env:KAGGLE_API_TOKEN = $plain
    $plain = $null
    Add-Report "INFO 2b: token saved in your Windows user settings (KAGGLE_API_TOKEN); not in the repository"
}
if (-not $state.username) {
    if (Test-Path $userFile) { $state.username = (Get-Content $userFile -Raw).Trim() }
    if (-not $state.username -and (Test-Path "out\kaggle\EXP-038\state.json")) {
        $state.username = (Get-Content "out\kaggle\EXP-038\state.json" -Raw -Encoding UTF8 | ConvertFrom-Json).username
    }
    if (-not $state.username -and (Test-Path $legacyJson)) { $state.username = (Get-Content $legacyJson -Raw | ConvertFrom-Json).username }
    if (-not $state.username) { $state.username = (Read-Host "FIRST TIME ONLY: your Kaggle username (as in kaggle.com/<username>)").Trim().ToLower() }
    Save-State $state
}
if (-not (Test-Path $userFile)) { $state.username | Set-Content -Path $userFile -Encoding ASCII }
$user = $state.username
$env:KAGGLE_USERNAME = $user
$datasetId = "$user/$datasetSlug"
$kernelId = "$user/$kernelSlug"
$kernelSources = @()
if ($Exp -eq "EXP-043") {
    # two kernels take turns, so that each session reads the OTHER kernel's output (the checkpoint)
    if (-not ($state.PSObject.Properties.Name -contains "kernel_slug")) {
        $state | Add-Member -NotePropertyName kernel_slug -NotePropertyValue ""
    }
    if ($launching) {
        $sessionNo = @(Get-ChildItem "evals\results\EXP-043" -Directory -Filter "session-*" -ErrorAction SilentlyContinue).Count + 1
        $own = "a"; $other = "b"
        if ($sessionNo % 2 -eq 0) { $own = "b"; $other = "a" }
        $state.kernel_slug = "frontier-exp043-$own"
        if ($sessionNo -gt 1) { $kernelSources = @("$user/frontier-exp043-$other") }
        $readsText = "nothing (first session)"
        if ($sessionNo -gt 1) { $readsText = "the output of $user/frontier-exp043-$other" }
        Add-Report "INFO 2c: session $sessionNo runs on kernel $user/$($state.kernel_slug) and reads $readsText"
    }
    if (-not $state.kernel_slug) { Stop-Run "no EXP-043 kernel is remembered; run this line again with -Relaunch" 1 }
    $kernelSlug = $state.kernel_slug
    $kernelId = "$user/$kernelSlug"
}
$r = Invoke-Logged "$kaggle kernels list --mine"
if ($r.Code -ne 0) { Stop-Run "Kaggle refused the login (token or username wrong?). Check them, then run this line again" 2 }
Add-Report "PASS 2b: logged in to Kaggle as $user"

# --- 3) private dataset with hi.bin (uploaded once) ------------------------------------------
if (-not $state.dataset_ready) {
    $r = Invoke-Logged "$kaggle datasets status $datasetId"
    if ($r.Code -ne 0 -or -not (($r.Lines -join " ") -match "ready")) {
        $stage = "$outDir\dataset"
        New-Item -ItemType Directory -Force $stage | Out-Null
        foreach ($f in $dataFiles) {
            $src = Join-Path (Get-Location) "$dataDir\$f"
            $dst = Join-Path (Get-Location) "$stage\$f"
            if ((Test-Path $dst) -and ((Get-Item $dst).Length -eq (Get-Item $src).Length)) { continue }
            # a hard link costs no disk space and no time; copy only if links are impossible here
            try { New-Item -ItemType HardLink -Path $dst -Target $src -ErrorAction Stop | Out-Null }
            catch { Copy-Item $src $dst -Force }
        }
        $meta = [ordered]@{
            title = $datasetSlug
            id = $datasetId
            licenses = @(@{ name = "other" })
            subtitle = $datasetSubtitle
            description = "$datasetText (from AI4Bharat Sangraha Verified, CC BY 4.0), tokenized with frontier-tokenizer-v2. Private working copy for $Exp. Source: https://huggingface.co/datasets/ai4bharat/sangraha"
        }
        Write-Ascii "$stage\dataset-metadata.json" ($meta | ConvertTo-Json -Depth 5)
        Add-Report ""
        Add-Report "=== uploading $($dataFiles.Count) files ($uploadSize) as PRIVATE dataset $datasetId - $(Get-Date -Format 'HH:mm:ss') ==="
        $r = Invoke-Logged "$kaggle datasets create -p $stage"
        if ($r.Code -ne 0) { Stop-Run "the dataset upload failed (see the lines above); run this line again to retry" 1 }
        $ready = $false
        $readyChecks = 60  # every 30 seconds: 30 minutes (EXP-040's 5.6 GB: 90 minutes)
        if ($dataFiles.Count -gt 2) { $readyChecks = 180 }
        for ($i = 0; $i -lt $readyChecks -and -not $ready; $i++) {
            Start-Sleep -Seconds 30
            $r = Invoke-Logged "$kaggle datasets status $datasetId"
            $ready = ($r.Code -eq 0) -and (($r.Lines -join " ") -match "ready")
        }
        if (-not $ready) { Stop-Run "Kaggle has not finished processing the dataset after $($readyChecks / 2) minutes; run this line again later" 1 }
    }
    $state.dataset_ready = $true
    Save-State $state
}
Add-Report "PASS 3: private dataset $datasetId is ready on Kaggle"

# --- 4) launch the kernel (once, or again with -Relaunch) ------------------------------------
if ($launching) {
    $stage = "$outDir\kernel"
    New-Item -ItemType Directory -Force $stage | Out-Null
    $code = (Get-Content $template -Raw -Encoding UTF8).Replace("__PINNED_COMMIT__", $head)
    Write-Ascii "$stage\run.py" $code
    $meta = [ordered]@{
        id = $kernelId
        title = $kernelSlug
        code_file = "run.py"
        language = "python"
        kernel_type = "script"
        is_private = $true
        enable_gpu = $true
        enable_internet = $true
        dataset_sources = @($datasetId)
        competition_sources = @()
        kernel_sources = $kernelSources
        machine_shape = "NvidiaTeslaT4"
    }
    Write-Ascii "$stage\kernel-metadata.json" ($meta | ConvertTo-Json -Depth 5)
    $r = Invoke-Logged "$kaggle kernels push -p $stage"
    if ($r.Code -ne 0 -or (($r.Lines -join " ") -match "error")) { Stop-Run "Kaggle did not accept the kernel (see the lines above)" 1 }
    $state.pinned_commit = $head
    $state.pushed_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    Save-State $state
    Add-Report "PASS 4: GPU kernel $kernelId launched at $($state.pushed_at) with commit $($head.Substring(0, 7))"
    Add-Report "       (you can watch it at https://www.kaggle.com/code/$kernelId - not needed)"
} else {
    Add-Report "PASS 4: GPU kernel $kernelId was launched earlier ($($state.pushed_at), commit $($state.pinned_commit.Substring(0, 7)))"
}

# --- 5) wait for the kernel, fetch and check its results -------------------------------------
try {
    & powercfg /change standby-timeout-ac 0
    & powercfg /change hibernate-timeout-ac 0
} catch { }
Add-Report ""
Add-Report "=== waiting for the GPU run (checks every 10 minutes; $expectedRun). Leave the PC on and online. ==="
$deadline = (Get-Date).AddHours($maxWaitHours)
$status = ""
while ($true) {
    $r = Invoke-Logged "$kaggle kernels status $kernelId"
    $text = ($r.Lines -join " ").ToLower()
    # only a real status line counts; a network hiccup just means "check again later"
    if ($r.Code -eq 0 -and $text -match "has status") {
        if ($text -match "complete") { $status = "complete"; break }
        if ($text -match "error|cancel") { $status = "error"; break }
    }
    if ((Get-Date) -gt $deadline) {
        Stop-Run "the GPU run has not finished after $maxWaitHours hours of waiting; run this line again later" 1
    }
    Write-Host "  $(Get-Date -Format 'HH:mm') still running - next check in 10 minutes"
    Start-Sleep -Seconds $pollSeconds
}
Add-Report "=== kernel finished with status: $status - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$outputDir = "$outDir\output"
# EXP-042 has one launch per session: a fresh download folder per launched commit, so an older
# session's files can never be mistaken for this one's
if ($Exp -eq "EXP-042" -or $Exp -eq "EXP-043") { $outputDir = "$outDir\output-$($state.pinned_commit.Substring(0, 7))" }
New-Item -ItemType Directory -Force $outputDir | Out-Null
if ($Exp -eq "EXP-043") {
    # only the small result files; the 2.3 GB checkpoint stays on Kaggle for the next session
    $r = Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force --file-pattern EXP-043/.*"
    $got = @(Get-ChildItem "$outputDir\EXP-043" -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue)
    if ($r.Code -ne 0 -or $got.Count -eq 0) {
        Add-Report "INFO 5: this Kaggle tool could not download only the result files; downloading the whole output (about 2.3 GB) instead"
        $r = Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force"
    } else {
        Add-Report "PASS 5: downloaded only the small result files (--file-pattern)"
    }
} else {
    $r = Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force"
}
if ($r.Code -ne 0) { Stop-Run "downloading the kernel output failed; run this line again" 1 }
$resultDir = "$outputDir\$Exp"
$publishSrc = $resultDir
if ($Exp -eq "EXP-042" -or $Exp -eq "EXP-043") {
    # the kernel writes <EXP>\session-<n>\; exactly one session folder is expected
    $sessions = @(Get-ChildItem "$outputDir\$Exp" -Directory -Filter "session-*" -ErrorAction SilentlyContinue)
    if ($sessions.Count -gt 1) { Stop-Run "the kernel output has $($sessions.Count) session folders; expected one" 1 }
    if ($sessions.Count -eq 1) { $resultDir = $sessions[0].FullName }
}
$summaryPath = "$resultDir\summary.json"
if (-not (Test-Path $summaryPath)) {
    Add-Report "FAIL 5a: the kernel left no summary.json. Last 60 lines of the kernel log:"
    $klog = Get-ChildItem $outputDir -Filter "*.log" | Select-Object -First 1
    if ($klog) { foreach ($line in (Get-Content $klog.FullName -Tail 60 -Encoding UTF8)) { $report.Add("  $line") } }
    Stop-Run "the GPU run produced no results (after fixing, the chat will tell you to add -Relaunch)" 1
}
$summary = Get-Content $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($summary.smoke) { Stop-Run "summary.json is a smoke test, not a GPU result" 1 }
if ($summary.environment.code_commit -and ($summary.environment.code_commit -ne $state.pinned_commit)) {
    Stop-Run "summary.json was made by commit $($summary.environment.code_commit), expected $($state.pinned_commit)" 1
}
Add-Report "PASS 5a: results downloaded (complete: $($summary.complete))"
if ($Exp -eq "EXP-043" -and $summary.chain_check -and ($summary.chain_check.ok -eq $false)) {
    # not published: the session number stays, so the retry reads the same checkpoint again
    $state.pinned_commit = ""
    Save-State $state
    Add-Report "FAIL 5b: checkpoint chain check: $($summary.chain_check.result)"
    Stop-Run "the checkpoint chain check failed, nothing was trained or published. Do NOT run the line again until the chat says so" 1
}
Add-Report ""
Add-Report "----- SUMMARY.txt -----"
foreach ($line in (Get-Content "$resultDir\SUMMARY.txt" -Encoding UTF8)) { Add-Report $line }
Add-Report "-----------------------"
Add-Report ""

# --- 6) publish, check, commit and push ONLY evals/results/<EXP>/ ----------------------------
$r = Invoke-Logged "$python -u scripts\publish_eval_results.py --exp-id $Exp --src $publishSrc"
if ($r.Code -ne 0) { Stop-Run "publishing failed (exit $($r.Code))" 1 }
Add-Report "PASS 6a: published to $resultsPrefix"

$r = Invoke-Logged "git status --porcelain --untracked-files=all"
$changed = @($r.Lines | Where-Object { $_.Trim() -ne "" })
if ($changed.Count -eq 0) {
    Add-Report "PASS 6b: nothing new to commit (results were already committed earlier)"
} else {
    $bad = @($changed | Where-Object { -not $_.StartsWith("?? $resultsPrefix") })
    if ($bad.Count -gt 0) { Stop-Run "unexpected changes outside ${resultsPrefix}: $($bad -join ' | ')" 1 }
    $r = Invoke-Logged "git add $resultsPrefix"
    if ($r.Code -ne 0) { Stop-Run "git add failed" 1 }
    $r = Invoke-Logged "git diff --cached --name-only"
    $staged = @($r.Lines | Where-Object { $_.Trim() -ne "" })
    $badStaged = @($staged | Where-Object { -not $_.StartsWith($resultsPrefix) })
    if ($r.Code -ne 0 -or $staged.Count -eq 0 -or $badStaged.Count -gt 0) {
        Invoke-Logged "git reset -q" | Out-Null
        Stop-Run "staged files are not exactly $resultsPrefix (unstaged again): $($badStaged -join ' | ')" 1
    }
    Add-Report "PASS 6b: staged $($staged.Count) files, all under $resultsPrefix"
    $msgFile = "$outDir\commit_message.txt"
    "${Exp}: GPU results from one Kaggle T4 (commit $($state.pinned_commit.Substring(0, 7)); checkpoints stay on Kaggle)" |
        Set-Content -Path $msgFile -Encoding ASCII
    $r = Invoke-Logged "git commit -q -F $msgFile"
    if ($r.Code -ne 0) { Stop-Run "git commit failed: $($r.Lines -join ' | ')" 1 }
    $hash = ((Invoke-Logged "git rev-parse --short HEAD").Lines -join "").Trim()
    Add-Report "PASS 6c: committed $hash"
}

$r = Invoke-Logged "git push origin $branch"
if ($r.Code -ne 0) {
    # The branch moved on GitHub while the GPU ran (seen once in EXP-038). Our commit touches only
    # evals/results/<EXP>/, so replaying it on top of the new commits is safe; try that ONCE.
    Add-Report "INFO 6d: push rejected; fetching the newer GitHub commits and replaying the results commit on top"
    $r = Invoke-Logged "git pull --rebase origin $branch"
    if ($r.Code -ne 0) {
        Invoke-Logged "git rebase --abort" | Out-Null
        Add-Report "FAIL 6d: git pull --rebase failed and was undone (the results ARE committed locally; nothing is lost)"
        Add-Report "RESULT: run complete, push FAILED - paste this report into the Arena chat."
        Save-Report
        exit 1
    }
    $r = Invoke-Logged "git push origin $branch"
}
if ($r.Code -ne 0) {
    Add-Report "FAIL 6d: git push failed (the results ARE committed locally; nothing is lost):"
    foreach ($line in $r.Lines) { Add-Report "  $line" }
    Add-Report "RESULT: run complete, push FAILED - paste this report into the Arena chat."
    Save-Report
    exit 1
}
Add-Report "PASS 6d: pushed to origin/$branch"
if ($Exp -eq "EXP-042") {
    # this session is safely on GitHub: forget its commit, so the same line launches the next session
    $state.pinned_commit = ""
    Save-State $state
    Add-Report ""
    if ($summary.complete) {
        Add-Report "LADDER: COMPLETE after session $($summary.session) - no more sessions are needed."
    } else {
        Add-Report "LADDER: NOT finished yet - $(@($summary.remaining).Count) runs are left for the next session."
        Add-Report "NEXT: after pasting this report into the Arena chat, run the same line again (session $($summary.session + 1))."
    }
}
if ($Exp -eq "EXP-043") {
    # this session is safely on GitHub: forget its commit, so the same line launches the next session
    $state.pinned_commit = ""
    Save-State $state
    Add-Report ""
    if ($summary.complete -and $summary.final -and $summary.final.model_final) {
        # the final weights (about 760 MB), once, into out\ (never committed); sha256 checked
        $mf = "$outputDir\exp043_final\model_final.pt"
        if (-not (Test-Path $mf)) { Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force --file-pattern exp043_final/.*" | Out-Null }
        if (-not (Test-Path $mf)) { Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force" | Out-Null }
        if (Test-Path $mf) {
            $h = (Get-FileHash -Algorithm SHA256 $mf).Hash
            if ($h -ieq $summary.final.model_final.sha256) { Add-Report "PASS 7: model_final.pt downloaded to $mf (sha256 matches)" }
            else { Add-Report "FAIL 7: model_final.pt sha256 $h does not match the summary ($($summary.final.model_final.sha256))" }
        } else {
            Add-Report "INFO 7: model_final.pt could not be downloaded now; it stays in the Kaggle output of $kernelId"
        }
    }
    if ($summary.stop_rule) {
        Add-Report "TRAINING: A STOP RULE FIRED ($($summary.stop_rule)). Do NOT run the line again; paste this report into the Arena chat."
    } elseif ($summary.stopped -and -not $summary.complete) {
        Add-Report "TRAINING: this session stopped early ($($summary.stopped)). Do NOT run the line again until the chat says so."
    } elseif ($summary.complete) {
        Add-Report "TRAINING: COMPLETE after session $($summary.session) - the one pass is finished; no more sessions are needed."
    } else {
        if ($summary.main) {
            Add-Report "TRAINING: step $($summary.main.end_step) of $($summary.main.total_steps) done."
        } else {
            Add-Report "TRAINING: learning-rate check: $($summary.lr_choice.reason)"
        }
        Add-Report "NEXT: after pasting this report into the Arena chat, run the same line again (session $($summary.session + 1))."
    }
}
Add-Report ""
Add-Report "RESULT: COMPLETE - finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'). Paste this report into the Arena chat."
Save-Report
exit 0
