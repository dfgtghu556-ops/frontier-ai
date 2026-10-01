# scripts/run_kaggle_exp040.ps1
#
# ONE-COMMAND runner for EXP-040 (step 11, phase 1: the learning rate and RoPE + GQA-2 at GPU scale
# on all 13 languages; 12 training runs of 100 M tokens on one free Kaggle T4, hard limit 9
# GPU-hours). It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-040: the same checks, the Kaggle
# username and token saved during EXP-038, ONE upload of all 13 EXP-037 token files (about 5.6 GB,
# every fingerprint checked first) as a new PRIVATE dataset <username>/frontier-v2-tok2-13lang,
# a new private one-T4 kernel <username>/frontier-exp040, then it commits ONLY
# evals/results/EXP-040/ and pushes.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp040.ps1
# If the laptop was switched off or the upload stopped: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-040" -Relaunch:$Relaunch
exit $LASTEXITCODE
