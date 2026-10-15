# scripts/run_hf_demo.ps1
#
# ONE-COMMAND: put the EXP-043 model on a PRIVATE Hugging Face demo page (D-053), so the founder can
# try it in a browser (PC or phone). Only the founder's Hugging Face account can see the page; making
# it public is a separate decision.
#
# What it does:
#   1. installs the Hugging Face tool (huggingface_hub) into .venv if it is missing;
#   2. FIRST TIME ONLY: asks for your Hugging Face token. You paste it into THIS PowerShell window
#      (not into any chat; right-click or Ctrl+V, one * per character); it is stored only on this PC
#      by huggingface_hub;
#   3. finds EXP-043's model_final.pt under out\kaggle\ by its recorded SHA-256; if it is not there,
#      downloads it (about 880 MB, no GPU hours) from the Kaggle kernel that ran EXP-043's last session;
#   4. runs scripts\hf_demo.py: checks the file, makes the fp16 copy (about 380 MB), tests the page's
#      code, and uploads to the private Space <your-user>/frontier-exp043-preview.
# The result is written to out\hf_demo\REPORT.txt - paste it into the Arena chat.
#
# Usage (from the repo root, after "git pull"):
#   powershell -ExecutionPolicy Bypass -File scripts\run_hf_demo.ps1
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$py = ".venv\Scripts\python.exe"
$kaggleExe = ".venv\Scripts\kaggle.exe"
$demoDir = "out\hf_demo"
$reportFile = "$demoDir\REPORT.txt"
New-Item -ItemType Directory -Force $demoDir | Out-Null

function Stop-WithReport([string]$why) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $text = "REPORT hf_demo (EXP-043 private preview) - $stamp`r`n`r`nFAIL: $why`r`nRESULT: FAILED. Paste this report into the Arena chat."
    Set-Content -Path $reportFile -Value $text -Encoding UTF8
    Write-Host ""
    Write-Host $text
    exit 1
}

if (-not (Test-Path $py)) { Stop-WithReport "the .venv folder was not found; run this from the frontier-ai folder" }

# 1. the Hugging Face tool
& $py -c "import huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the Hugging Face tool (huggingface_hub)..."
    & $py -m pip install --quiet "huggingface_hub>=0.30"
    if ($LASTEXITCODE -ne 0) { Stop-WithReport "could not install huggingface_hub (is the internet on?)" }
}

# 2. the Hugging Face login (token stays on this PC)
$hfUser = & $py -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $hfUser) {
    Write-Host ""
    Write-Host "FIRST TIME ONLY: your Hugging Face token is needed (https://huggingface.co/settings/tokens, type Write)."
    Write-Host "  Paste it with a RIGHT-CLICK (or Ctrl+V) and press Enter. You will see one * per character."
    $secureToken = Read-Host -AsSecureString "Hugging Face token"
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    $plainToken = ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)).Trim()
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    if (-not $plainToken.StartsWith("hf_")) {
        Stop-WithReport "the pasted text does not start with hf_ ($($plainToken.Length) characters received); copy the token again and run the line again"
    }
    Write-Host "Received $($plainToken.Length) characters starting with hf_. Checking it with Hugging Face..."
    # handed to Python only through this window's environment (never written to a file, the repo or the chat);
    # huggingface_hub then saves it in its own token file on this PC
    $env:FRONTIER_HF_TOKEN_INPUT = $plainToken
    $plainToken = $null
    & $py -c "import os; from huggingface_hub import login; login(token=os.environ['FRONTIER_HF_TOKEN_INPUT'], add_to_git_credential=False)"
    Remove-Item Env:FRONTIER_HF_TOKEN_INPUT -ErrorAction SilentlyContinue
    $hfUser = & $py -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $hfUser) { Stop-WithReport "Hugging Face did not accept the token; make a new Write token and run the line again" }
}
Write-Host "Logged in to Hugging Face as $hfUser"

# 3. the EXP-043 model file
$expectedSha = (Get-Content "evals\results\EXP-043\session-6\summary.json" -Raw -Encoding UTF8 | ConvertFrom-Json).final.model_final.sha256
function Find-Exp043Model {
    $files = Get-ChildItem -Path "out\kaggle" -Recurse -Filter "model_final.pt" -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        Write-Host "Checking $($f.FullName) ..."
        if ((Get-FileHash -Algorithm SHA256 $f.FullName).Hash -ieq $expectedSha) { return $f.FullName }
    }
    return $null
}
$modelPath = Find-Exp043Model
if (-not $modelPath) {
    if (-not (Test-Path $kaggleExe)) { Stop-WithReport "model_final.pt is not on this PC and the Kaggle tool was not found in .venv" }
    $kaggleUser = ""
    foreach ($s in @("out\kaggle\EXP-043\state.json", "out\kaggle\EXP-038\state.json")) {
        if (-not $kaggleUser -and (Test-Path $s)) { $kaggleUser = (Get-Content $s -Raw -Encoding UTF8 | ConvertFrom-Json).username }
    }
    if (-not $kaggleUser) { Stop-WithReport "model_final.pt is not on this PC and the Kaggle username is unknown" }
    $dl = "out\kaggle\EXP-043\hf-download"
    Write-Host "Downloading model_final.pt (about 880 MB) from $kaggleUser/frontier-exp043-b ..."
    & $kaggleExe kernels output "$kaggleUser/frontier-exp043-b" -p $dl --force --file-pattern "exp043_final/.*"
    $modelPath = Find-Exp043Model
    if (-not $modelPath) { Stop-WithReport "model_final.pt could not be downloaded from Kaggle (or its SHA-256 differs)" }
}
Write-Host "Model file: $modelPath (SHA-256 matches the EXP-043 record)"

# 4. build, test and upload (private)
& $py scripts\hf_demo.py --weights $modelPath
exit $LASTEXITCODE
