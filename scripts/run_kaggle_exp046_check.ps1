# scripts/run_kaggle_exp046_check.ps1
#
# ONE-COMMAND runner for the slice-2 BACKUP CHECK (approved 2026-10-09, "approve A B C", item B).
# One Kaggle CPU session (no GPU, so no GPU quota; estimated under 1 hour, not measured yet). It runs
# scripts\run_kaggle_exp038.ps1 with -Exp EXP-046 -Check: the same checks, the Kaggle username and
# token saved earlier, one private kernel <username>/frontier-exp046-check that mounts your two private
# backup datasets (frontier-v2-slice2-a / -b) and the two original build outputs, and checks every file
# against the committed manifests (it only reads and hashes). It commits ONLY
# evals/results/EXP-046/backup-check/ and pushes.
#
# Start it only BETWEEN two EXP-043 sessions (the runner refuses while one is launched and not yet
# collected).
#
# UPDATE 2026-10-09 (runner fix A1): this is a CPU-only run, so it may now also start while an
# EXP-043 session is running (it uses no GPU quota and commits only its own results folder).
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp046_check.ps1
# If the laptop was switched off: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-046" -Check -Relaunch:$Relaunch
exit $LASTEXITCODE
