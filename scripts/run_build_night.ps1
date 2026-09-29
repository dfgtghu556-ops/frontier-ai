# scripts/run_build_night.ps1
#
# ONE-COMMAND overnight runner for EXP-036 (the FrontierCorpus v2 build on the Sangraha slice
# downloaded in EXP-034, with the 10 rules approved after EXP-035). It runs
# scripts\run_data_night.ps1 with -Exp EXP-036 -Task build: the same checks, the same fingerprint
# check of the files, then scripts\build_sangraha_v2.py, then it commits ONLY
# evals/results/EXP-036/ (summary, manifest, masked samples) and pushes. The corpus itself is
# written to data\frontier_v2\ (git-ignored) and stays on this PC.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_build_night.ps1
#
# Safe to run again after an interruption: files that were already built are checked and reused.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

& "$PSScriptRoot\run_data_night.ps1" -Exp "EXP-036" -Task "build"
exit $LASTEXITCODE
