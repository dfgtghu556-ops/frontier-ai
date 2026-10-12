# scripts/run_kaggle_exp047.ps1
#
# ONE-COMMAND runner for EXP-047 (the next base model, D-052: 20 x 1280, about 507 M parameters, over
# one pass of v2-slice1 + v2-slice2, 8.16 B tokens). Each launch is ONE Kaggle session of at most 9
# hours on the free T4 machine (two T4 GPUs). The first sessions are the memory probe and the
# learning-rate check (one GPU; about one learning-rate run per session at this size); then the main
# run continues session after session (about 314 session-hours in total, hard cap 380 over all of
# them, D-052). It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-047: the same checks, the Kaggle
# username and token saved earlier, the SAME private datasets (EXP-037 13-language tokens and the
# two v2-slice2 backups; no new upload), two private kernels <username>/frontier-exp047-a and -b that
# take turns (each reads the other's checkpoint), then it commits ONLY the new
# evals/results/EXP-047/session-<n>/ files and pushes. The checkpoint itself stays on Kaggle;
# model_final.pt is downloaded once at the very end into out\kaggle\EXP-047\.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp047.ps1
# If the laptop was switched off: run the same line again - it continues.
# When the report says "NEXT: ... run the same line again": that starts the next session.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-047" -Relaunch:$Relaunch
exit $LASTEXITCODE
