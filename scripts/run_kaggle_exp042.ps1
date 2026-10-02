# scripts/run_kaggle_exp042.ps1
#
# ONE-COMMAND runner for EXP-042 (the IsoFLOP ladder, step 11 phase 2). Each launch is ONE Kaggle
# session of at most 9 GPU-hours on one free T4: the one-step float64 device check (D-048), then the
# next ladder runs that fit. The ladder needs about 3 sessions (about 19-21 GPU-hours, hard cap 25
# over all sessions). It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-042: the same checks, the
# Kaggle username and token saved earlier, the SAME private 13-language dataset uploaded for EXP-040
# (fingerprints checked again, no new upload), a private one-T4 kernel <username>/frontier-exp042,
# then it commits ONLY the new evals/results/EXP-042/session-<n>/ files and pushes. The next launch
# reads those files and continues where this session stopped.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp042.ps1
# If the laptop was switched off: run the same line again - it continues.
# When the report says the ladder is not finished: run the same line again (the next session).
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-042" -Relaunch:$Relaunch
exit $LASTEXITCODE
