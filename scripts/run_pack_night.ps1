# scripts/run_pack_night.ps1
#
# ONE-COMMAND overnight runner for EXP-037 (FrontierCorpus v2-slice1, accepted in D-045, packed into
# training token files with Frontier Tokenizer v2). It runs scripts\run_data_night.ps1 with
# -Exp EXP-037 -Task pack: the same checks, no download, then scripts\pack_sangraha_v2.py, then it
# commits ONLY evals/results/EXP-037/ (summary, manifest with counts and fingerprints) and pushes.
# The token files are written to data\frontier_v2\sangraha-slice1-v2-tok2\ (git-ignored) and stay
# on this PC.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_pack_night.ps1
#
# Safe to run again after an interruption: files that were already packed are checked and reused.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

& "$PSScriptRoot\run_data_night.ps1" -Exp "EXP-037" -Task "pack"
exit $LASTEXITCODE
