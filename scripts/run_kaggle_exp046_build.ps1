# scripts/run_kaggle_exp046_build.ps1
#
# ONE-COMMAND runner for EXP-046 step 2: build one part of FrontierCorpus v2-slice2 on a Kaggle CPU
# session (no GPU, so no GPU quota; about 4 to 6 hours, not measured yet).
#   Part A = en ur as bn gu hi      Part B = kn ml mr or pa ta te
# It runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-046 -Build <part>: the same checks, the Kaggle
# username and token saved earlier, the SAME private 13-language token dataset (no new upload), plus
# the protected held-out texts (exported on this PC by scripts\export_heldout_text.py, about 1 MB,
# uploaded ONCE as a private dataset). One private kernel <username>/frontier-exp046-build-a (or -b).
# The corpus (about 9 GB per part) stays on Kaggle as that kernel's output; only the small reports
# come back. Then it commits ONLY evals/results/EXP-046/build-<part>/ and pushes.
#
# Start it only BETWEEN two EXP-043 sessions (like EXP-044/045): the runner refuses while an EXP-043
# session is launched but not yet collected. Run part A first, part B in a later gap.
#
# UPDATE 2026-10-09 (runner fix A1): this is a CPU-only run, so it may now also start while an
# EXP-043 session is running (it uses no GPU quota and commits only its own results folder).
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp046_build.ps1 -Part A
# If the laptop was switched off: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([Parameter(Mandatory = $true)][ValidateSet("A", "B")][string]$Part, [switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-046" -Build $Part -Relaunch:$Relaunch
exit $LASTEXITCODE
