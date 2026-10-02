"""EXP-038: static checks of the Kaggle runner (PowerShell) and the kernel template.

The runner cannot execute here (no Windows, no Kaggle account), so these tests pin the promises it
makes to the founder: ASCII-only, the token is never printed or stored in the repository, only
``evals/results/EXP-038/`` is committed, pushes go only to the session branch, the kernel is private
on one T4 GPU, and the kernel runs exactly the pinned commit with arguments the bring-up script accepts.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS1 = ROOT / "scripts" / "run_kaggle_exp038.ps1"
KERNEL = ROOT / "scripts" / "kaggle" / "exp038_kernel.py"
BRANCH = "arena/01a0dc16-frontier-ai"


def _ps1() -> str:
    return PS1.read_text(encoding="ascii")  # raises if any non-ASCII byte is present


def test_runner_is_ascii_and_pushes_only_the_session_branch():
    text = _ps1()
    assert f'$branch = "{BRANCH}"' in text
    pushes = re.findall(r'Invoke-Logged "(git push[^"]*)"', text)
    assert set(pushes) == {"git push origin $branch"} and len(pushes) == 2  # first try + one retry
    assert "--force" not in text.replace("kernels output $kernelId -p $outputDir --force", "")
    assert "Remove-Item" not in text


def test_runner_commits_only_the_exp038_results_folder():
    text = _ps1()
    assert '$resultsPrefix = "evals/results/$Exp/"' in text
    assert (
        'param([ValidateSet("EXP-038", "EXP-039", "EXP-040", "EXP-041", "EXP-042")][string]$Exp = "EXP-038",'
        " [switch]$Relaunch)" in text
    )
    assert "git add $resultsPrefix" in text
    assert re.findall(r'Invoke-Logged "(git add[^"]*)"', text) == ["git add $resultsPrefix"]
    assert "scripts\\publish_eval_results.py --exp-id $Exp --src $publishSrc" in text
    assert (
        "$publishSrc = $resultDir" in text
    )  # EXP-042 keeps $publishSrc = <output>\EXP-042 (session-<n> kept)


def test_token_is_hidden_and_kept_out_of_the_repository():
    text = _ps1()
    assert 'Read-Host "Kaggle API token" -AsSecureString' in text
    assert '[Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", $plain, "User")' in text
    # the token value never reaches a log line, a file or a command line
    for line in text.splitlines():
        if "$plain" in line or "$env:KAGGLE_API_TOKEN" in line:
            assert "Add-Report" not in line and "Write-Host" not in line and "Out-File" not in line
            assert "Invoke-Logged" not in line and "Set-Content" not in line
    # everything the runner writes lives under out\ (git-ignored)
    assert '$outDir = "out\\kaggle\\$Exp"' in text
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^/?out/?$", gitignore, re.MULTILINE)


def test_kernel_is_private_single_t4_with_the_hindi_dataset():
    text = _ps1()
    for needle in (
        "is_private = $true",
        "enable_gpu = $true",
        "enable_internet = $true",
        'machine_shape = "NvidiaTeslaT4"',
        "dataset_sources = @($datasetId)",
        'kernel_type = "script"',
        'code_file = "run.py"',
    ):
        assert needle in text, needle
    assert '$datasetSlug = "frontier-v2-hi-tok2"' in text
    assert (
        'licenses = @(@{ name = "other" })' in text
    )  # source licence (CC BY 4.0) credited in the description
    assert "CC BY 4.0" in text and "ai4bharat/sangraha" in text


def test_runner_checks_the_data_fingerprint_and_the_pushed_commit():
    text = _ps1()
    assert "Get-FileHash -Algorithm SHA256" in text and "evals\\results\\EXP-037\\manifest.json" in text
    assert "git rev-parse FETCH_HEAD" in text and "git fetch -q origin $branch" in text
    assert '.Replace("__PINNED_COMMIT__", $head)' in text
    assert "if ($summary.smoke)" in text and "code_commit" in text


def _kernel_consts() -> dict[str, object]:
    tree = ast.parse(KERNEL.read_text(encoding="ascii"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:  # Path("...")
            value = value.args[0]
        if isinstance(value, ast.Constant):
            out[node.targets[0].id] = value.value
    return out


def test_kernel_template_pins_a_commit_and_stays_within_the_budget():
    c = _kernel_consts()
    assert c["COMMIT"] == "__PINNED_COMMIT__"
    assert c["REPO"] == "https://github.com/dfgtghu556-ops/frontier-ai.git"
    assert 0 < float(c["MAX_HOURS"]) <= 6.0  # pre-registered cap: 6 GPU-hours
    assert str(c["OUT"]).startswith("/kaggle/working/") and str(c["SCRATCH"]).startswith("/tmp/")


def test_kernel_refuses_to_run_with_the_placeholder():
    res = subprocess.run([sys.executable, str(KERNEL)], capture_output=True, text=True, timeout=60)
    assert res.returncode != 0 and "placeholder" in (res.stderr + res.stdout)


def test_kernel_arguments_are_accepted_by_the_bringup_script():
    text = KERNEL.read_text(encoding="ascii")
    flags = set(re.findall(r'"(--[a-z-]+)"', text))
    assert {"--data", "--out", "--scratch", "--exp-id", "--max-hours"} <= flags
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gpu_bringup.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    for flag in {"--data", "--out", "--scratch", "--exp-id", "--max-hours"}:
        assert flag in helptext, flag


def test_push_retry_rebases_once_and_undoes_a_failed_rebase():
    text = _ps1()
    assert text.count('Invoke-Logged "git pull --rebase origin $branch"') == 1
    assert 'Invoke-Logged "git rebase --abort"' in text
    # the retry comes after the results commit and before the final push check
    assert text.index("git commit -q -F $msgFile") < text.index("git pull --rebase origin $branch")


# ------------------------------------------------------------------------------------ EXP-039 --
PS1_039 = ROOT / "scripts" / "run_kaggle_exp039.ps1"
KERNEL_039 = ROOT / "scripts" / "kaggle" / "exp039_kernel.py"


def test_exp039_wrapper_reuses_the_runner_and_the_dataset():
    wrapper = PS1_039.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-039" -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    assert '$template = "scripts\\kaggle\\${expSlug}_kernel.py"' in text
    assert '$kernelSlug = "frontier-$expSlug"' in text
    assert '$resultDir = "$outputDir\\$Exp"' in text
    # the username saved during EXP-038 is reused; the dataset slug is shared (no second upload)
    assert "out\\kaggle\\EXP-038\\state.json" in text and '$userFile = "out\\kaggle\\username.txt"' in text
    # only EXP-040 (all 13 languages) switches to another dataset
    assert text.count("$datasetSlug = ") == 2
    assert (
        'if ($Exp -eq "EXP-040" -or $Exp -eq "EXP-041" -or $Exp -eq "EXP-042") {\n'
        '    $datasetSlug = "frontier-v2-tok2-13lang"' in text
    )
    assert '$dataFiles = @("hi.bin", "hi.meta.json")' in text


def test_exp039_kernel_runs_the_diagnostic_within_one_gpu_hour():
    tree = ast.parse(KERNEL_039.read_text(encoding="ascii"))
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__" and float(consts["MAX_HOURS"]) <= 1.0
    assert consts["OUT"] == "/kaggle/working/EXP-039"
    text = KERNEL_039.read_text(encoding="ascii")
    assert '"--part",\n        "cpu-gpu-diagnostic"' in text and '"EXP-039"' in text
    res = subprocess.run([sys.executable, str(KERNEL_039)], capture_output=True, text=True, timeout=60)
    assert res.returncode != 0 and "placeholder" in (res.stderr + res.stdout)


# ------------------------------------------------------------------------------------ EXP-040 --
PS1_040 = ROOT / "scripts" / "run_kaggle_exp040.ps1"
KERNEL_040 = ROOT / "scripts" / "kaggle" / "exp040_kernel.py"


def test_exp040_wrapper_and_the_13_file_dataset():
    wrapper = PS1_040.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-040" -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    # every bin + meta file listed in the EXP-037 manifest is uploaded, and every .bin is fingerprinted
    assert "foreach ($f in (Get-Content $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json).files)" in text
    assert "$dataFiles += @($f.path, $f.meta)" in text
    assert "foreach ($b in $bins)" in text and 'Get-FileHash -Algorithm SHA256 "$dataDir\\$b"' in text
    # staged with hard links (no 5.6 GB copy), plain copy only as a fallback; nothing deleted
    assert "New-Item -ItemType HardLink -Path $dst -Target $src -ErrorAction Stop" in text
    assert "catch { Copy-Item $src $dst -Force }" in text and "Remove-Item" not in text
    assert "if ($dataFiles.Count -gt 2) { $readyChecks = 180 }" in text  # 90 minutes of dataset processing
    assert "$maxWaitHours = 11" in text  # 9 GPU-hours cap + queue time


def test_exp040_kernel_runs_the_lr_arch_script_within_nine_gpu_hours():
    tree = ast.parse(KERNEL_040.read_text(encoding="ascii"))
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__" and float(consts["MAX_HOURS"]) <= 9.0
    assert consts["OUT"] == "/kaggle/working/EXP-040" and str(consts["SCRATCH"]).startswith("/tmp/")
    assert consts["MANIFEST"] == "evals/results/EXP-037/manifest.json"
    text = KERNEL_040.read_text(encoding="ascii")
    flags = set(re.findall(r'"(--[a-z-]+)"', text)) - {"--quiet", "--no-deps"}  # pip / git flags
    assert flags == {"--data-dir", "--out", "--scratch", "--exp-id", "--max-hours"}
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gpu_lr_arch.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    for flag in flags:
        assert flag in helptext, flag
    res = subprocess.run([sys.executable, str(KERNEL_040)], capture_output=True, text=True, timeout=60)
    assert res.returncode != 0 and "placeholder" in (res.stderr + res.stdout)


# ------------------------------------------------------------------------------------ EXP-041 --
PS1_041 = ROOT / "scripts" / "run_kaggle_exp041.ps1"
KERNEL_041 = ROOT / "scripts" / "kaggle" / "exp041_kernel.py"


def test_exp041_wrapper_reuses_the_13_language_dataset():
    wrapper = PS1_041.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-041" -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    assert 'if ($Exp -eq "EXP-040" -or $Exp -eq "EXP-041" -or $Exp -eq "EXP-042") {' in text  # no new upload
    assert 'if ($Exp -eq "EXP-041") {\n    $expectedRun' in text and "$maxWaitHours = 7" in text


def test_exp041_kernel_runs_the_followup_within_five_gpu_hours():
    tree = ast.parse(KERNEL_041.read_text(encoding="ascii"))
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__" and float(consts["MAX_HOURS"]) <= 5.0
    assert consts["OUT"] == "/kaggle/working/EXP-041" and str(consts["SCRATCH"]).startswith("/tmp/")
    assert consts["PREVIOUS"] == "evals/results/EXP-040/summary.json"
    assert (ROOT / str(consts["PREVIOUS"])).exists()
    text = KERNEL_041.read_text(encoding="ascii")
    assert '"--part",\n        "followup"' in text and '"EXP-041"' in text
    flags = set(re.findall(r'"(--[a-z-]+)"', text)) - {"--quiet", "--no-deps"}
    assert flags == {"--data-dir", "--out", "--scratch", "--exp-id", "--max-hours", "--part", "--prev"}
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gpu_lr_arch.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    for flag in flags:
        assert flag in helptext, flag
    res = subprocess.run([sys.executable, str(KERNEL_041)], capture_output=True, text=True, timeout=60)
    assert res.returncode != 0 and "placeholder" in (res.stderr + res.stdout)


# ------------------------------------------------------------------------------------ EXP-042 --
PS1_042 = ROOT / "scripts" / "run_kaggle_exp042.ps1"
KERNEL_042 = ROOT / "scripts" / "kaggle" / "exp042_kernel.py"


def test_exp042_wrapper_reuses_the_13_language_dataset_and_waits_long_enough():
    wrapper = PS1_042.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-042" -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    assert 'if ($Exp -eq "EXP-040" -or $Exp -eq "EXP-041" -or $Exp -eq "EXP-042") {' in text
    block = text[text.index('if ($Exp -eq "EXP-042") {\n    $expectedRun') :]
    assert "$maxWaitHours = 11" in block.split("}")[0]  # 9-hour kernel + queueing and download


def test_exp042_runner_publishes_one_session_and_then_forgets_the_commit():
    text = _ps1()
    # a fresh download folder per launched commit: an older session can never be picked up
    assert '$outputDir = "$outDir\\output-$($state.pinned_commit.Substring(0, 7))"' in text
    assert 'Get-ChildItem "$outputDir\\$Exp" -Directory -Filter "session-*"' in text
    assert "expected one" in text and "$resultDir = $sessions[0].FullName" in text
    # the session folder keeps its name: <output>\EXP-042\session-<n> -> evals/results/EXP-042/session-<n>
    assert text.index("$publishSrc = $resultDir") < text.index("if ($sessions.Count -eq 1)")
    # the commit is forgotten only AFTER the push succeeded, so a failed push is retried, not relaunched
    push_ok = text.index('Add-Report "PASS 6d: pushed to origin/$branch"')
    forget = text.index('$state.pinned_commit = ""')
    assert push_ok < forget and text.count('$state.pinned_commit = ""') == 1
    assert "run the same line again (session $($summary.session + 1))" in text
    # still only new files under evals/results/EXP-042/ may be committed
    assert '$bad = @($changed | Where-Object { -not $_.StartsWith("?? $resultsPrefix") })' in text


def test_exp042_kernel_runs_one_ladder_session_within_nine_gpu_hours():
    tree = ast.parse(KERNEL_042.read_text(encoding="ascii"))
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__" and float(consts["MAX_HOURS"]) <= 9.0
    assert float(consts["TOTAL_CAP_HOURS"]) == 25.0
    assert consts["OUT"] == "/kaggle/working/EXP-042" and str(consts["SCRATCH"]).startswith("/tmp/")
    assert consts["PREVIOUS_DIR"] == "evals/results/EXP-042"
    text = KERNEL_042.read_text(encoding="ascii")
    assert '"--part",\n        "ladder"' in text and '"EXP-042"' in text
    flags = set(re.findall(r'"(--[a-z-]+)"', text)) - {"--quiet", "--no-deps"}
    assert flags == {
        "--data-dir", "--out", "--scratch", "--exp-id", "--max-hours", "--part", "--prev-dir", "--total-cap-hours",
    }  # fmt: skip
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gpu_lr_arch.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    for flag in flags:
        assert flag in helptext, flag
    res = subprocess.run([sys.executable, str(KERNEL_042)], capture_output=True, text=True, timeout=60)
    assert res.returncode != 0 and "placeholder" in (res.stderr + res.stdout)
