"""Fix A (approved 2026-10-09, "approve A B C"): three small fixes to the shared Kaggle runner.

A1. CPU-only runs may start while an EXP-043 session is out; GPU runs still may not.
A2. When Kaggle refuses something, its own lines go into REPORT.txt (not only into run.log).
A3. The report says "CPU run" / "GPU run" correctly, and no misleading "0 token file(s) match" line.
Static checks only: there is no PowerShell in the sandbox.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_kaggle_exp038.ps1"
CPU_EXPR = '(($Exp -eq "EXP-045" -and -not $Final) -or $Exp -eq "EXP-046")'


def _text() -> str:
    return RUNNER.read_text(encoding="ascii")


def test_a1_cpu_runs_may_start_alongside_but_gpu_runs_may_not():
    text = _text()
    # one definition of "CPU only", the same condition that switches the Kaggle GPU off
    assert f"$cpuOnly = {CPU_EXPR}" in text
    j = text.index('        machine_shape = "NvidiaTeslaT4"\n    }\n')
    assert f"if {CPU_EXPR} {{" in text[j : j + 400]
    g = text.index(
        'if ($launching -and ($Exp -eq "EXP-044" -or $Exp -eq "EXP-045" -or $Exp -eq "EXP-046")) {'
    )
    assert text.index("$cpuOnly = ") < g  # defined before the guard uses it
    block = text[g : text.index("\n}\n", g)]
    assert "if ($out43 -and -not $cpuOnly) {" in block and "Stop-Run" in block
    assert "INFO 1d0: an EXP-043 session is running" in block
    # EXP-044 and EXP-045 -Final (GPU) are never CPU-only
    assert '"EXP-044"' not in CPU_EXPR and "-not $Final" in CPU_EXPR
    assert g < text.index("$kaggle kernels push -p $stage")


def test_a2_kaggle_messages_reach_the_report():
    text = _text()
    d = text.index("function Add-Tail(")
    assert d < text.index("function Stop-Run(")
    body = text[d : text.index("\n}\n", d)]
    assert "$report.Add(" in body and "Select-Object -Last $n" in body
    for needle in (
        "the dataset upload failed",
        "could not export the held-out texts",
        "the held-out upload failed",
        "Kaggle did not accept the kernel",
        "downloading the kernel output failed",
    ):
        line = next(x for x in text.splitlines() if needle in x)
        assert "Add-Tail $r; Stop-Run" in line, line
    assert "see the lines above" not in text  # the lines are now in the report itself


def test_a3_report_names_the_run_correctly():
    text = _text()
    assert "GPU run on one free Kaggle T4" not in text
    assert 'Add-Report "REPORT $Exp ($runKind) - started' in text
    assert (
        'if ($cpuOnly) { $runKind = "CPU run on a free Kaggle CPU session, no GPU quota"; $runWord = "CPU" }'
        in text
    )
    assert text.index("$runKind = ") < text.index('Add-Report "REPORT $Exp')
    printed = [x for x in text.splitlines() if "Add-Report" in x or "Stop-Run" in x]
    assert not any(re.search(r"\bGPU kernel\b", x) or "the GPU run" in x for x in printed)
    assert "PASS 4: $runWord kernel $kernelId launched at" in text
    assert "INFO 1e: this run needs no token files from this PC" in text


def test_no_variable_reuses_a_parameter_name():
    """PowerShell names are case-insensitive: `$text = ...` in a script with a `[ValidateSet] $Text`
    parameter re-validates the new value and stops the script (founder's run, 2026-10-09). No script
    may assign to a variable that is also one of its parameters."""
    for ps1 in sorted((ROOT / "scripts").glob("*.ps1")):
        text = ps1.read_text(encoding="ascii")
        m = re.search(r"^param\((.*)\)\s*$", text, flags=re.M)
        if not m:
            continue
        names = re.findall(r"\$(\w+)", m.group(1))
        for line in text.splitlines():
            if line.lstrip().startswith(("function ", "#", "param(")):
                continue
            for name in names:
                pat = rf"(?i)(^|[\s;({{])\${name}\s*=(?!=)|foreach\s*\(\s*\${name}\s+in\b"
                assert not re.search(pat, line), f"{ps1.name}: {line.strip()}"
