# scripts/run_kaggle_exp045.ps1
#
# ONE-COMMAND runner for EXP-045 test 4 (the contamination check): are the Belebele test texts
# inside our 2.78 B training tokens? One Kaggle CPU session (no GPU, so no GPU quota; about 1 hour,
# not measured yet). It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-045: the same checks, the
# Kaggle username and token saved earlier, the SAME private 13-language dataset (no new upload), one
# private kernel <username>/frontier-exp045, then it commits ONLY evals/results/EXP-045/ and pushes.
#
# Start it only BETWEEN two EXP-043 sessions (like EXP-044): after an EXP-043 report said "run the
# same line again", and before running that line. The runner refuses while an EXP-043 session is
# launched but not yet collected.
#
# UPDATE 2026-10-09 (runner fix A1): this is a CPU-only run, so it may now also start while an
# EXP-043 session is running (it uses no GPU quota and commits only its own results folder).
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp045.ps1
# If the laptop was switched off: run the same line again - it continues.
#
# Coverage re-scan (approved 2026-10-06): test 4 is done (commit 0332f69) and this PC still remembers
# its kernel, so START the re-scan once with -Relaunch (it pins the current GitHub commit; results go
# to evals/results/EXP-045/coverage/). If the laptop is switched off after that, run the line again
# WITHOUT -Relaunch - it continues.
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp045.ps1 -Relaunch
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-045" -Relaunch:$Relaunch
exit $LASTEXITCODE
