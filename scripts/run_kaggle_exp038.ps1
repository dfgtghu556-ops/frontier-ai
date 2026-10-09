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
# EXP-045 with -Final (scripts\run_kaggle_exp045_final.ps1; tests 1-3 and 5 of the approved EXP-045 plan)
# runs ONE GPU session after EXP-043 is complete: its own kernel <username>/frontier-$expSlug-final, its
# own state folder out\kaggle\EXP-045-final, the output of the EXP-043 kernel that ran the LAST session
# (model_final.pt) and the private held-out texts dataset as inputs; it publishes evals/results/EXP-045/final/
# and then downloads the fp16 model copy (about 0.38 GB, never committed), sha256 checked.
# EXP-046 with -Check (scripts\run_kaggle_exp046_check.ps1; approved 2026-10-09) checks the slice-2 backup
# datasets against the build manifests on ONE CPU session: its own kernel and state folder; inputs are the
# founder's private datasets frontier-v2-slice2-a/-b and the build kernels' outputs; it publishes
# evals/results/EXP-046/backup-check/.
# EXP-046 with -Text rewrite|verify (scripts\run_kaggle_exp046_text.ps1; approved 2026-10-09, "DO ALL")
# makes a COMPLETE copy of the slice-2 text on ONE CPU session each: "rewrite" mounts the build kernels'
# outputs and writes the 13 text files as one gzip member each (they stay on Kaggle; the founder makes
# the dataset frontier-v2-slice2-text from that output); "verify" mounts that dataset and checks it.
# Each has its own kernel and state folder; they publish evals/results/EXP-046/text-rewrite|text-verify/.
# If the final push is rejected because the branch moved meanwhile (seen once in EXP-038), it pulls
# with --rebase once and pushes again; the results commit still touches only evals/results/<EXP>/.
#
# It never deletes anything, never edits a tracked file, never force-pushes, and pushes only to
# arena/01a0dc16-frontier-ai. Checkpoints stay on Kaggle's temporary disk and vanish with the
# session; only summary.json, SUMMARY.txt and samples.jsonl come back. No money is spent: Kaggle's
# free GPU quota (about 30 GPU-hours a week) is used, this run at most about 6.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([ValidateSet("EXP-038", "EXP-039", "EXP-040", "EXP-041", "EXP-042", "EXP-043", "EXP-044", "EXP-045", "EXP-046")][string]$Exp = "EXP-038", [switch]$Relaunch, [ValidateSet("A", "B")][string]$Build, [switch]$Final, [switch]$Check, [ValidateSet("rewrite", "verify")][string]$Text)

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
if ($Exp -eq "EXP-040" -or $Exp -eq "EXP-041" -or $Exp -eq "EXP-042" -or $Exp -eq "EXP-043" -or $Exp -eq "EXP-044" -or $Exp -eq "EXP-045" -or $Exp -eq "EXP-046") {
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
if ($Exp -eq "EXP-044") {
    $expectedRun = "EXP-044 is expected to take about 1.5 hours, at most 2"
    $maxWaitHours = 4
}
if ($Exp -eq "EXP-045") {
    # EXP-045 test 4 (contamination check) runs on a Kaggle CPU session: no GPU quota is used
    # (test 4 took 1,871 s of scanning; the coverage re-scan writes EXP-045\coverage\)
    $expectedRun = "EXP-045 coverage re-scan (CPU only, no GPU) should take about 40 minutes; not measured yet"
    $maxWaitHours = 6
}
if ($Exp -eq "EXP-046") {
    # EXP-046 step 1: the 15-minute probe of a Kaggle CPU session (no GPU, processes no data)
    $template = "scripts\kaggle\exp046_probe_kernel.py"
    $kernelSlug = "frontier-exp046-probe"
    $expectedRun = "the EXP-046 probe (CPU only, no GPU) should take about 15 minutes; not measured yet"
    $maxWaitHours = 2
}
if ($Exp -eq "EXP-046" -and $Build) {
    # EXP-046 step 2: build part A or B of v2-slice2 on a Kaggle CPU session (run_kaggle_exp046_build.ps1);
    # its own kernel, state and report folder, so the probe's state is never reused
    $template = "scripts\kaggle\exp046_build_kernel.py"
    $kernelSlug = "frontier-exp046-build-$($Build.ToLower())"
    $outDir = "out\kaggle\EXP-046-build-$Build"
    $statePath = "$outDir\state.json"
    $log = "$outDir\run.log"
    $reportPath = "$outDir\REPORT.txt"
    $expectedRun = "EXP-046 build $Build (CPU only, no GPU) should take about 4 to 6 hours; not measured yet"
    $maxWaitHours = 14
}
if ($Final -and $Exp -ne "EXP-045") { Write-Host "-Final is only for -Exp EXP-045"; exit 2 }
if ($Check -and ($Exp -ne "EXP-046" -or $Build)) { Write-Host "-Check is only for -Exp EXP-046 (without -Build)"; exit 2 }
if ($Text -and ($Exp -ne "EXP-046" -or $Build -or $Check)) { Write-Host "-Text is only for -Exp EXP-046 (without -Build or -Check)"; exit 2 }
if ($Text) {
    # the complete slice-2 text backup (run_kaggle_exp046_text.ps1): CPU sessions; its own kernels,
    # states and report folders (one per step), so no earlier EXP-046 state is ever reused
    $template = "scripts\kaggle\exp046_text_${Text}_kernel.py"
    $kernelSlug = "frontier-exp046-text-$Text"
    $outDir = "out\kaggle\EXP-046-text-$Text"
    $statePath = "$outDir\state.json"
    $log = "$outDir\run.log"
    $reportPath = "$outDir\REPORT.txt"
    $dataFiles = @()  # the 13-language token files are not needed and not mounted
    $expectedRun = "the text $Text step (CPU only, no GPU) should take under 2 hours; not measured yet"
    $maxWaitHours = 5
}
if ($Check) {
    # the slice-2 backup check (run_kaggle_exp046_check.ps1): a CPU session that only reads and hashes;
    # its own kernel, state and report folder, so the probe's and the builds' states are never reused
    $template = "scripts\kaggle\exp046_check_kernel.py"
    $kernelSlug = "frontier-exp046-check"
    $outDir = "out\kaggle\EXP-046-check"
    $statePath = "$outDir\state.json"
    $log = "$outDir\run.log"
    $reportPath = "$outDir\REPORT.txt"
    $dataFiles = @()  # the 13-language token files are not needed and not mounted
    $expectedRun = "the backup check (CPU only, no GPU) should take under 1 hour; not measured yet"
    $maxWaitHours = 3
}
if ($Final) {
    # EXP-045 tests 1-3 and 5: ONE GPU session after EXP-043 is complete (run_kaggle_exp045_final.ps1);
    # its own kernel, state and report folder, so the coverage re-scan's state is never reused
    $template = "scripts\kaggle\exp045_final_kernel.py"
    $kernelSlug = "frontier-$expSlug-final"
    $outDir = "out\kaggle\EXP-045-final"
    $statePath = "$outDir\state.json"
    $log = "$outDir\run.log"
    $reportPath = "$outDir\REPORT.txt"
    $dataFiles = @()  # the 13-language token files are not needed and not mounted
    $expectedRun = "the EXP-045 final evaluation (one GPU) should take under 1 hour; not measured yet"
    $maxWaitHours = 4
}
# CPU-only runs (EXP-045 coverage, EXP-046 probe/builds/check) say so in the report (fix A3, 2026-10-09)
$cpuOnly = (($Exp -eq "EXP-045" -and -not $Final) -or $Exp -eq "EXP-046")
$runKind = "GPU run on a free Kaggle T4 machine"
$runWord = "GPU"
if ($cpuOnly) { $runKind = "CPU run on a free Kaggle CPU session, no GPU quota"; $runWord = "CPU" }
New-Item -ItemType Directory -Force $outDir | Out-Null

$report = New-Object System.Collections.Generic.List[string]
function Add-Report([string]$text) { $report.Add($text); Write-Host $text; $text | Out-File $log -Append -Encoding utf8 }
function Save-Report {
    $report | Set-Content -Path $reportPath -Encoding UTF8
    Write-Host ""
    Write-Host "Report written to $reportPath (opening it in Notepad)."
    try { Start-Process notepad.exe -ArgumentList $reportPath } catch { }
}
# Kaggle's own words go into the report too, not only into run.log (fix A2, 2026-10-09)
function Add-Tail($res, [int]$n = 15) {
    $tail = @($res.Lines | Where-Object { "$_".Trim() -ne "" } | Select-Object -Last $n)
    if ($tail.Count -gt 0) {
        $report.Add("  Last lines from the command:")
        foreach ($line in $tail) { $report.Add("  | $line") }
    }
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

Add-Report "REPORT $Exp ($runKind) - started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
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
if ($launching -and ($Exp -eq "EXP-044" -or $Exp -eq "EXP-045" -or $Exp -eq "EXP-046")) {
    # GPU runs (EXP-044, EXP-045 -Final) start only BETWEEN EXP-043 sessions: never while one is launched
    # and not yet collected. CPU-only runs may start alongside since 2026-10-09 (fix A1): they use no GPU
    # quota and commit only their own results folder. If both runners publish in the same minute, the
    # second one stops safely ("unexpected changes") and the same line is simply run again.
    $s43 = "out\kaggle\EXP-043\state.json"
    $out43 = (Test-Path $s43) -and ((Get-Content $s43 -Raw -Encoding UTF8 | ConvertFrom-Json).pinned_commit)
    if ($out43 -and -not $cpuOnly) {
        Stop-Run "an EXP-043 session is launched and not yet collected. Run scripts\run_kaggle_exp043.ps1 first (it waits for that session); start $Exp after its report" 2
    }
    if ($out43) { Add-Report "INFO 1d0: an EXP-043 session is running; $Exp uses only a CPU session, so it runs alongside" }
    else { Add-Report "PASS 1d0: no EXP-043 session is running" }
}
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
    if ($bins.Count -gt 0) { Add-Report "PASS 1e: $($bins.Count) token file(s) match their EXP-037 fingerprints ($($bins -join ', '))" }
    else { Add-Report "INFO 1e: this run needs no token files from this PC" }
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
if ($Final -and $launching) {
    # the model is EXP-043's model_final.pt, in the output of the kernel that ran the LAST session (odd
    # session numbers ran on -a, even ones on -b); refuse before EXP-043 is complete
    $done = @(Get-ChildItem "evals\results\EXP-043" -Directory -Filter "session-*" -ErrorAction SilentlyContinue | Where-Object {
        (Test-Path "$($_.FullName)\summary.json") -and ((Get-Content "$($_.FullName)\summary.json" -Raw -Encoding UTF8 | ConvertFrom-Json).complete)
    })
    if ($done.Count -ne 1) { Stop-Run "EXP-043 is not complete yet (no published session says complete); the final evaluation runs only after its last session" 2 }
    $finalNo = [int]($done[0].Name.Substring(8))
    $finalKernel = "a"
    if ($finalNo % 2 -eq 0) { $finalKernel = "b" }
    $kernelSources = @("$user/frontier-exp043-$finalKernel")
    Add-Report "INFO 2c: EXP-043 finished in session $finalNo; the evaluation reads the output of $user/frontier-exp043-$finalKernel"
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
        if ($r.Code -ne 0) { Add-Tail $r; Stop-Run "the dataset upload failed; run this line again to retry" 1 }
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
if ($Build -or $Final) {
    # EXP-046 build: the protected held-out texts (about 1 MB) as a second PRIVATE dataset, uploaded once.
    # The build refuses to run without them; they are only used to REMOVE overlapping documents.
    # EXP-045 -Final reads the same dataset (already on Kaggle since the builds) for test 1 only.
    $heldSlug = "frontier-heldout-v1-text"
    $heldId = "$user/$heldSlug"
    $heldFile = "out\eval\heldout-v1.jsonl"
    if (-not ($state.PSObject.Properties.Name -contains "heldout_ready")) {
        $state | Add-Member -NotePropertyName heldout_ready -NotePropertyValue $false
    }
    if (-not $state.heldout_ready) {
        $r = Invoke-Logged "$kaggle datasets status $heldId"
        if ($r.Code -ne 0 -or -not (($r.Lines -join " ") -match "ready")) {
            if (-not (Test-Path $heldFile)) {
                $r = Invoke-Logged "$python scripts\export_heldout_text.py --out $heldFile"
                if ($r.Code -ne 0 -or -not (Test-Path $heldFile)) { Add-Tail $r; Stop-Run "could not export the held-out texts" 2 }
            }
            $stage = "$outDir\heldout"
            New-Item -ItemType Directory -Force $stage | Out-Null
            Copy-Item $heldFile "$stage\heldout-v1.jsonl"
            $meta = [ordered]@{
                title = $heldSlug
                id = $heldId
                licenses = @(@{ name = "other" })
                subtitle = "Protected evaluation texts of the frontier-ai project"
                description = "PRIVATE. The protected held-out suite frontier-heldout-v1 (D-042) as text, checked against SUITE.json. Used only to remove overlapping documents from training data and for evaluation; never for training."
            }
            Write-Ascii "$stage\dataset-metadata.json" ($meta | ConvertTo-Json -Depth 5)
            Add-Report "=== uploading the held-out texts (about 1 MB) as PRIVATE dataset $heldId - $(Get-Date -Format 'HH:mm:ss') ==="
            $r = Invoke-Logged "$kaggle datasets create -p $stage"
            if ($r.Code -ne 0) { Add-Tail $r; Stop-Run "the held-out upload failed; run this line again to retry" 1 }
            $ready = $false
            for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
                Start-Sleep -Seconds 30
                $r = Invoke-Logged "$kaggle datasets status $heldId"
                $ready = ($r.Code -eq 0) -and (($r.Lines -join " ") -match "ready")
            }
            if (-not $ready) { Stop-Run "Kaggle has not finished processing the held-out dataset after 30 minutes; run this line again later" 1 }
        }
        $state.heldout_ready = $true
        Save-State $state
    }
    Add-Report "PASS 3b: private held-out dataset $heldId is ready on Kaggle"
}

# --- 4) launch the kernel (once, or again with -Relaunch) ------------------------------------
if ($launching) {
    $stage = "$outDir\kernel"
    New-Item -ItemType Directory -Force $stage | Out-Null
    $code = (Get-Content $template -Raw -Encoding UTF8).Replace("__PINNED_COMMIT__", $head)
    if ($Build) { $code = $code.Replace("__BUILD_PART__", $Build) }
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
    if (($Exp -eq "EXP-045" -and -not $Final) -or $Exp -eq "EXP-046") {
        # a CPU session (4 cores, 30 GB): no GPU, so no GPU quota
        $meta.enable_gpu = $false
        $meta.Remove("machine_shape")
        Add-Report "INFO 4: $Exp runs on a Kaggle CPU session (no GPU is requested)"
    }
    if ($Build) { $meta.dataset_sources = @($datasetId, $heldId) }  # + the held-out texts
    if ($Final) { $meta.dataset_sources = @($heldId) }  # test 1's texts only; the GPU stays on
    if ($Check) {
        # the backup (founder's private datasets) and the originals (build kernel outputs), read only
        $meta.dataset_sources = @("$user/frontier-v2-slice2-a", "$user/frontier-v2-slice2-b")
        $meta.kernel_sources = @("$user/frontier-exp046-build-a", "$user/frontier-exp046-build-b")
    }
    if ($Text -eq "rewrite") {
        # only the originals (build kernel outputs, read only): the one full copy of the text
        $meta.dataset_sources = @()
        $meta.kernel_sources = @("$user/frontier-exp046-build-a", "$user/frontier-exp046-build-b")
    }
    if ($Text -eq "verify") { $meta.dataset_sources = @("$user/frontier-v2-slice2-text") }  # the new dataset only
    Write-Ascii "$stage\kernel-metadata.json" ($meta | ConvertTo-Json -Depth 5)
    $r = Invoke-Logged "$kaggle kernels push -p $stage"
    if ($r.Code -ne 0 -or (($r.Lines -join " ") -match "error")) { Add-Tail $r; Stop-Run "Kaggle did not accept the kernel (Kaggle's message is above)" 1 }
    $state.pinned_commit = $head
    $state.pushed_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    Save-State $state
    Add-Report "PASS 4: $runWord kernel $kernelId launched at $($state.pushed_at) with commit $($head.Substring(0, 7))"
    Add-Report "       (you can watch it at https://www.kaggle.com/code/$kernelId - not needed)"
} else {
    Add-Report "PASS 4: $runWord kernel $kernelId was launched earlier ($($state.pushed_at), commit $($state.pinned_commit.Substring(0, 7)))"
}

# --- 5) wait for the kernel, fetch and check its results -------------------------------------
try {
    & powercfg /change standby-timeout-ac 0
    & powercfg /change hibernate-timeout-ac 0
} catch { }
Add-Report ""
Add-Report "=== waiting for the $runWord run (checks every 10 minutes; $expectedRun). Leave the PC on and online. ==="
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
        Stop-Run "the $runWord run has not finished after $maxWaitHours hours of waiting; run this line again later" 1
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
} elseif ($Final) {
    # only the small result files; the fp16 copy is fetched after the push (step 7)
    $r = Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force --file-pattern EXP-045/final/.*"
} elseif ($Check) {
    $r = Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force --file-pattern EXP-046/backup-check/.*"
} elseif ($Text) {
    # never the 13 text files (about 5.6 GB that stay on Kaggle): only the small report folder
    $r = Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force --file-pattern EXP-046/text-$Text/.*"
} elseif ($Build) {
    # never the whole output (about 9 GB of corpus that stays on Kaggle): only the small report folder
    $r = Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force --file-pattern EXP-046/build-$Build/.*"
} else {
    $r = Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force"
}
if ($r.Code -ne 0) { Add-Tail $r; Stop-Run "downloading the kernel output failed; run this line again" 1 }
$resultDir = "$outputDir\$Exp"
$publishSrc = $resultDir
if ($Exp -eq "EXP-042" -or $Exp -eq "EXP-043") {
    # the kernel writes <EXP>\session-<n>\; exactly one session folder is expected
    $sessions = @(Get-ChildItem "$outputDir\$Exp" -Directory -Filter "session-*" -ErrorAction SilentlyContinue)
    if ($sessions.Count -gt 1) { Stop-Run "the kernel output has $($sessions.Count) session folders; expected one" 1 }
    if ($sessions.Count -eq 1) { $resultDir = $sessions[0].FullName }
}
# EXP-046: the probe writes EXP-046\probe\ (later build steps get their own folders)
if ($Exp -eq "EXP-046") { $resultDir = "$outputDir\$Exp\probe" }
if ($Build) { $resultDir = "$outputDir\$Exp\build-$Build" }
if ($Check) { $resultDir = "$outputDir\$Exp\backup-check" }
if ($Text) { $resultDir = "$outputDir\$Exp\text-$Text" }
# EXP-045: since test 4 the kernel writes the coverage re-scan to EXP-045\coverage\
if ($Exp -eq "EXP-045") { $resultDir = "$outputDir\$Exp\coverage" }
if ($Final) { $resultDir = "$outputDir\$Exp\final" }
$summaryPath = "$resultDir\summary.json"
if (-not (Test-Path $summaryPath)) {
    Add-Report "FAIL 5a: the kernel left no summary.json. Last 60 lines of the kernel log:"
    $klog = Get-ChildItem $outputDir -Filter "*.log" | Select-Object -First 1
    if ($klog) { foreach ($line in (Get-Content $klog.FullName -Tail 60 -Encoding UTF8)) { $report.Add("  $line") } }
    Stop-Run "the $runWord run produced no results (after fixing, the chat will tell you to add -Relaunch)" 1
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
    $msgText = "${Exp}: GPU results from one Kaggle T4 (commit $($state.pinned_commit.Substring(0, 7)); checkpoints stay on Kaggle)"
    if ($Exp -eq "EXP-045") { $msgText = "${Exp}: coverage re-scan results from one Kaggle CPU session (commit $($state.pinned_commit.Substring(0, 7)))" }
    if ($Exp -eq "EXP-046") { $msgText = "${Exp}: probe results from one Kaggle CPU session (commit $($state.pinned_commit.Substring(0, 7)))" }
    if ($Final) { $msgText = "${Exp}: final evaluation of the EXP-043 model (tests 1-3 and 5) from one Kaggle GPU session (commit $($state.pinned_commit.Substring(0, 7)); the fp16 copy is not committed)" }
    if ($Check) { $msgText = "${Exp}: slice-2 backup check from one Kaggle CPU session (commit $($state.pinned_commit.Substring(0, 7)); only reads and hashes)" }
    if ($Text) { $msgText = "${Exp}: slice-2 text backup, $Text step, from one Kaggle CPU session (commit $($state.pinned_commit.Substring(0, 7)); the text stays on Kaggle)" }
    if ($Build) { $msgText = "${Exp}: build $Build reports (v2-slice2; the corpus stays on Kaggle) from one Kaggle CPU session (commit $($state.pinned_commit.Substring(0, 7)))" }
    $msgText | Set-Content -Path $msgFile -Encoding ASCII
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
if ($Final -and $summary.fp16_copy) {
    # the weights-only fp16 copy (about 0.38 GB), once, into out\ (never committed); sha256 checked
    $fp = "$outputDir\exp045_fp16\$($summary.fp16_copy.file)"
    if (-not (Test-Path $fp)) { Invoke-Logged "$kaggle kernels output $kernelId -p $outputDir --force --file-pattern exp045_fp16/.*" | Out-Null }
    if (Test-Path $fp) {
        $h = (Get-FileHash -Algorithm SHA256 $fp).Hash
        if ($h -ieq $summary.fp16_copy.sha256) { Add-Report "PASS 7: fp16 model copy downloaded to $fp (sha256 matches)" }
        else { Add-Report "FAIL 7: fp16 copy sha256 $h does not match the summary ($($summary.fp16_copy.sha256))" }
    } else {
        Add-Report "INFO 7: the fp16 copy could not be downloaded now; it stays in the Kaggle output of $kernelId"
    }
}
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
if ($Text -eq "rewrite") {
    Add-Report ""
    Add-Report "VERDICT: $($summary.verdict)"
    if (($summary.verdict -as [string]).StartsWith("PASS")) {
        # the founder's one manual step: a private dataset from this kernel's output (Kaggle web page)
        Add-Report "NEXT (on kaggle.com, about 5 minutes of clicks; the copy itself may take longer):"
        Add-Report "  1. Open https://www.kaggle.com/code/$kernelId and click the Output tab."
        Add-Report "  2. Click New Dataset. Title: frontier-v2-slice2-text  (exactly this). Keep it Private. Create."
        Add-Report "  3. When Kaggle shows the dataset as ready, run:"
        Add-Report "     powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp046_text.ps1 -Verify"
    }
}
Add-Report ""
Add-Report "RESULT: COMPLETE - finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'). Paste this report into the Arena chat."
Save-Report
exit 0
