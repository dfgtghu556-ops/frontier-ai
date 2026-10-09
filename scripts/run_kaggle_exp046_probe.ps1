# scripts/run_kaggle_exp046_probe.ps1
#
# ONE-COMMAND runner for EXP-046 step 1: the 15-minute probe of a Kaggle CPU session (no GPU, so no
# GPU quota). It processes NO data: it measures CPU, RAM and disk, checks that the build tools import,
# and times the download + SHA-256 check of ONE pinned Sangraha slice-2 file (then deletes it). It
# runs scripts\run_kaggle_exp038.ps1 with -Exp EXP-046: the same checks, the Kaggle username and
# token saved earlier, the SAME private 13-language dataset attached (no new upload), one private
# kernel <username>/frontier-exp046-probe, then it commits ONLY evals/results/EXP-046/ and pushes.
#
# Start it only BETWEEN two EXP-043 sessions (like EXP-044/045): after an EXP-043 report said "run
# the same line again", and before running that line. The runner refuses while an EXP-043 session
# is launched but not yet collected.
#
# UPDATE 2026-10-09 (runner fix A1): this is a CPU-only run, so it may now also start while an
# EXP-043 session is running (it uses no GPU quota and commits only its own results folder).
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_kaggle_exp046_probe.ps1
# If the laptop was switched off: run the same line again - it continues.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

param([switch]$Relaunch)

& "$PSScriptRoot\run_kaggle_exp038.ps1" -Exp "EXP-046" -Relaunch:$Relaunch
exit $LASTEXITCODE
