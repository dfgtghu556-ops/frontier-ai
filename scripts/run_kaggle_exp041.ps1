# scripts/run_kaggle_exp041.ps1
#
# ONE-COMMAND runner for EXP-041 (the EXP-040 follow-up). Part A: a one-step float64 CPU-vs-GPU
# check for both architectures (nothing compounds; the baseline is the control). Part B, only if
# RoPE + GQA-2 passes: its 6 runs that EXP-040 skipped plus one baseline control run, on one free
# Kaggle T4, hard limit 5 GPU-hours. It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-041: the
# same checks, the Kaggle username and token saved earlier, the SAME private 13-language dataset
# uploaded for EXP-040 (fingerprints checked again, no new upload), a new private one-T4 kernel
# <username>/frontier-exp041, then it commits ONLY evals/results/EXP-041/ and pushes.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp041.ps1
# If the laptop was switched off: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-041" -Relaunch:$Relaunch
exit $LASTEXITCODE
