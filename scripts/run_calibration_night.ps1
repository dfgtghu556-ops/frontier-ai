# scripts/run_calibration_night.ps1
#
# ONE-COMMAND overnight runner for EXP-035 (calibration of the v2 build rules on the Sangraha
# slice downloaded in EXP-034). It removes nothing. It runs scripts\run_data_night.ps1 with
# -Exp EXP-035 -Task calibrate: the same checks, the same fingerprint check of the files, then
# scripts\calibrate_sangraha_slice.py, then it commits ONLY evals/results/EXP-035/ and pushes.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_calibration_night.ps1
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page).

& "$PSScriptRoot\run_data_night.ps1" -Exp "EXP-035" -Task "calibrate"
exit $LASTEXITCODE
