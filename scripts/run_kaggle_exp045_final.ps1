# scripts/run_kaggle_exp045_final.ps1
#
# ONE-COMMAND runner for the EXP-045 final evaluation (tests 1, 2, 3 and 5 of the plan approved
# 2026-10-04) of the finished EXP-043 model. One Kaggle GPU session (estimated under 1 GPU-hour, not
# measured yet). It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-045 -Final: the same checks, the
# Kaggle username and token saved earlier, one private kernel <username>/frontier-exp045-final that
# reads (a) the output of the EXP-043 kernel that ran the LAST session (model_final.pt; the runner
# picks -a or -b itself) and (b) the private held-out texts dataset already on Kaggle. It commits
# ONLY evals/results/EXP-045/final/ and pushes, then downloads the fp16 copy of the model (about
# 0.38 GB) into out\kaggle\EXP-045-final\ (never committed) for "try the model".
#
# Start it only AFTER EXP-043 is complete (an EXP-043 report said "TRAINING: COMPLETE"). It refuses
# before that, and while an EXP-043 session is launched but not yet collected.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp045_final.ps1
# If the laptop was switched off: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-045" -Final -Relaunch:$Relaunch
exit $LASTEXITCODE
