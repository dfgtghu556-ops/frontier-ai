# scripts/run_kaggle_exp043.ps1
#
# ONE-COMMAND runner for EXP-043 (step 12: train the D-049 model, 14 x 896, about 190 M parameters,
# over one pass of the 2.78 B-token corpus). Each launch is ONE Kaggle session of at most 9 GPU-hours
# on one free T4. Session 1 is the learning-rate check; then the main run continues session after
# session (about 10 sessions, hard cap 100 GPU-hours over all of them). It runs
# scripts\run_kaggle_exp038.ps1 with -Exp EXP-043: the same checks, the Kaggle username and token
# saved earlier, the SAME private 13-language dataset (no new upload), two private one-T4 kernels
# <username>/frontier-exp043-a and -b that take turns (each reads the other's checkpoint), then it
# commits ONLY the new evals/results/EXP-043/session-<n>/ files and pushes. The checkpoint itself
# stays on Kaggle; model_final.pt is downloaded once at the very end into out\kaggle\EXP-043\.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp043.ps1
# If the laptop was switched off: run the same line again - it continues.
# When the report says "NEXT: ... run the same line again": that starts the next session.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-043" -Relaunch:$Relaunch
exit $LASTEXITCODE
