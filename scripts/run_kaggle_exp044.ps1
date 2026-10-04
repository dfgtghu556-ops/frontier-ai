# scripts/run_kaggle_exp044.ps1
#
# ONE-COMMAND runner for EXP-044 (step 13: check two-GPU training before using it). One Kaggle
# session of about 1.5 GPU-hours (at most 2) on the free T4 x2 machine: 300 steps of the 190 M model
# on one GPU, the same 300 steps on two GPUs, then the three pre-registered rules (agreement, speed,
# memory). It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-044: the same checks, the Kaggle
# username and token saved earlier, the SAME private 13-language dataset (no new upload), one
# private kernel <username>/frontier-exp044, then it commits ONLY evals/results/EXP-044/ and pushes.
#
# Start it only BETWEEN two EXP-043 sessions: after an EXP-043 report said "run the same line again",
# and before running that line. The runner refuses while an EXP-043 session is launched but not yet
# collected.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp044.ps1
# If the laptop was switched off: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-044" -Relaunch:$Relaunch
exit $LASTEXITCODE
