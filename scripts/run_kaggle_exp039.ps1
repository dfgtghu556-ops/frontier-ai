# scripts/run_kaggle_exp039.ps1
#
# ONE-COMMAND runner for EXP-039 (step 10 follow-up: does the T4 compute the same function as the
# CPU? float64 comparison, pre-registered pass <= 1e-8 per step, plus the float32 per-step curve).
# It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-039: the same checks, the SAME private
# Kaggle dataset (no new upload), the Kaggle username and token saved during EXP-038, a new private
# one-T4 kernel <username>/frontier-exp039 (hard limit 1 GPU-hour), then it commits ONLY
# evals/results/EXP-039/ and pushes.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp039.ps1
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-039" -Relaunch:$Relaunch
exit $LASTEXITCODE
