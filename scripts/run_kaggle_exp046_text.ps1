# scripts/run_kaggle_exp046_text.ps1
#
# ONE-COMMAND runner for the COMPLETE slice-2 TEXT backup (approved 2026-10-09, founder "DO ALL",
# backup item b). Why: the backup check found that Kaggle's unzip kept only half of each text file in
# the datasets frontier-v2-slice2-a / -b (token files are complete). Two steps, each one Kaggle CPU
# session (no GPU, so no GPU quota; each estimated under 2 hours, not measured yet):
#
#   1. (no switch)  "rewrite": one private kernel <username>/frontier-exp046-text-rewrite mounts the two
#      build outputs (the only full copy) and writes the 13 text files again as ONE gzip member each,
#      checking every file (same content, all documents). The 5.6 GB of text stay on Kaggle. It commits
#      ONLY evals/results/EXP-046/text-rewrite/ and pushes. Then, on kaggle.com, make a private dataset
#      named frontier-v2-slice2-text from that kernel's output (instructions in the report's last lines).
#   2. -Verify      "verify": one private kernel <username>/frontier-exp046-text-verify mounts the new
#      dataset and checks all 13 text files in it. It commits ONLY evals/results/EXP-046/text-verify/.
#
# CPU-only, so it may also run while an EXP-043 session is running (runner fix A1).
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp046_text.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp046_text.ps1 -Verify
# If the laptop was switched off: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Verify, [switch]$Relaunch)

$step = "rewrite"
if ($Verify) { $step = "verify" }
& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-046" -Text $step -Relaunch:$Relaunch
exit $LASTEXITCODE
