# scripts/run_kaggle_exp048.ps1
#
# ONE-COMMAND runner for EXP-048 (approved 2026-10-09, "approve A B C", item C): measure how many NEW,
# CLEAN tokens FineWeb-2 could add for our 12 Indian languages, on one free Kaggle CPU session (no GPU,
# so no GPU quota; about 1 to 3 hours, not measured yet). It adds NOTHING to the corpus.
# It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-048: the same checks, the Kaggle username and
# token saved earlier, the private datasets that are ALREADY on Kaggle (v2-slice1 tokens, v2-slice2 A
# and B, the held-out texts; nothing is uploaded). One private kernel <username>/frontier-exp048.
# Only the small reports come back; then it commits ONLY evals/results/EXP-048/ and pushes.
#
# CPU-only, so it may run while an EXP-043 session is running.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp048.ps1
# If the laptop was switched off: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-048" -Relaunch:$Relaunch
exit $LASTEXITCODE
