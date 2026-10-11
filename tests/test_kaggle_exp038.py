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
        'param([ValidateSet("EXP-038", "EXP-039", "EXP-040", "EXP-041", "EXP-042", "EXP-043", "EXP-044", "EXP-045", "EXP-046", "EXP-048")]'
        '[string]$Exp = "EXP-038", [switch]$Relaunch, [ValidateSet("A", "B")][string]$Build, [switch]$Final, [switch]$Check, [ValidateSet("rewrite", "verify")][string]$TextStep)' in text
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
        'if ($Exp -eq "EXP-040" -or $Exp -eq "EXP-041" -or $Exp -eq "EXP-042" -or $Exp -eq "EXP-043" -or $Exp -eq "EXP-044" -or $Exp -eq "EXP-045" -or $Exp -eq "EXP-046" -or $Exp -eq "EXP-048") {\n'
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
    assert 'if ($Exp -eq "EXP-040" -or $Exp -eq "EXP-041" -or $Exp -eq "EXP-042" -or $Exp -eq "EXP-043" -or $Exp -eq "EXP-044" -or $Exp -eq "EXP-045" -or $Exp -eq "EXP-046" -or $Exp -eq "EXP-048") {' in text  # no new upload
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
    assert 'if ($Exp -eq "EXP-040" -or $Exp -eq "EXP-041" -or $Exp -eq "EXP-042" -or $Exp -eq "EXP-043" -or $Exp -eq "EXP-044" -or $Exp -eq "EXP-045" -or $Exp -eq "EXP-046" -or $Exp -eq "EXP-048") {' in text
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
    forget = text.index('$state.pinned_commit = ""', text.index('if ($Exp -eq "EXP-042") {\n    # this session'))
    assert push_ok < forget
    # EXP-043 forgets it after the push too, and (unpublished) after a failed checkpoint-chain check
    assert text.count('$state.pinned_commit = ""') == 3
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


# ------------------------------------------------------------------------------------ EXP-043 --
PS1_043 = ROOT / "scripts" / "run_kaggle_exp043.ps1"
KERNEL_043 = ROOT / "scripts" / "kaggle" / "exp043_kernel.py"


def test_exp043_wrapper_reuses_the_13_language_dataset_and_waits_long_enough():
    wrapper = PS1_043.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-043" -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    block = text[text.index('if ($Exp -eq "EXP-043") {\n    $expectedRun') :]
    assert "$maxWaitHours = 11" in block.split("}")[0]


def test_exp043_runner_alternates_two_kernels_that_read_each_other():
    text = _ps1()
    block = text[text.index("$kernelSources = @()") : text.index("# --- 3)")]
    assert '$state.kernel_slug = "frontier-exp043-$own"' in block
    assert 'if ($sessionNo % 2 -eq 0) { $own = "b"; $other = "a" }' in block
    assert '$own = "a"; $other = "b"' in block
    assert 'if ($sessionNo -gt 1) { $kernelSources = @("$user/frontier-exp043-$other") }' in block
    # the session number comes from the committed results; decided only when launching
    assert 'Get-ChildItem "evals\\results\\EXP-043" -Directory -Filter "session-*"' in block
    assert block.index("if ($launching) {") < block.index("$sessionNo =")
    assert "kernel_sources = $kernelSources" in text and "kernel_sources = @()" not in text
    # a kernel never reads itself
    assert "frontier-exp043-$own\") }" not in block


def test_exp043_runner_downloads_only_results_and_the_final_model_once():
    text = _ps1()
    assert "--file-pattern EXP-043/.*" in text and "--file-pattern exp043_final/.*" in text
    i = text.index("--file-pattern EXP-043/.*")
    fallback = text[i : i + 700]
    assert "downloading the whole output (about 2.3 GB) instead" in fallback
    assert '$mf = "$outputDir\\exp043_final\\model_final.pt"' in text
    assert "Get-FileHash -Algorithm SHA256 $mf" in text and "$summary.final.model_final.sha256" in text
    # never committed: out\ is ignored, and only evals/results/<EXP>/ may be staged
    assert "out/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_exp043_runner_does_not_publish_a_failed_chain_check_and_reports_stop_rules():
    text = _ps1()
    i = text.index("$summary.chain_check.ok -eq $false")
    publish = text.index("scripts\\publish_eval_results.py --exp-id")
    assert i < publish and "Stop-Run" in text[i : i + 600]
    assert "A STOP RULE FIRED" in text and "Do NOT run the line again" in text
    assert "run the same line again (session $($summary.session + 1))" in text


def test_exp043_kernel_runs_one_session_within_nine_gpu_hours():
    tree = ast.parse(KERNEL_043.read_text(encoding="ascii"))
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__" and float(consts["MAX_HOURS"]) <= 9.0
    assert consts["OUT"] == "/kaggle/working/EXP-043" and str(consts["SCRATCH"]).startswith("/tmp/")
    assert consts["CHAIN_OUT"] == "/kaggle/working/exp043_chain"
    assert consts["FINAL_DIR"] == "/kaggle/working/exp043_final" and consts["CHAIN_IN"] == "/kaggle/input"
    assert consts["PREVIOUS_DIR"] == "evals/results/EXP-043"
    text = KERNEL_043.read_text(encoding="ascii")
    assert '"--part",\n        "auto"' in text and '"EXP-043"' in text
    flags = set(re.findall(r'"(--[a-z-]+)"', text)) - {"--quiet", "--no-deps"}
    assert flags == {
        "--data-dir", "--part", "--out", "--prev-dir", "--chain-in", "--chain-out", "--final-dir", "--scratch",
        "--exp-id", "--max-hours", "--gpus",
    }  # fmt: skip
    assert '"--gpus",\n        "2",\n    ]' in text  # D-050: the main run on both T4s
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gpu_pretrain.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    for flag in flags:
        assert flag in helptext, flag
    res = subprocess.run([sys.executable, str(KERNEL_043)], capture_output=True, text=True, timeout=60)
    assert res.returncode != 0 and "placeholder" in (res.stderr + res.stdout)


# ------------------------------------------------------------------------------------ EXP-044 --
PS1_044 = ROOT / "scripts" / "run_kaggle_exp044.ps1"
KERNEL_044 = ROOT / "scripts" / "kaggle" / "exp044_kernel.py"


def test_exp044_wrapper_reuses_the_13_language_dataset_and_waits_long_enough():
    wrapper = PS1_044.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-044" -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    block = text[text.index('if ($Exp -eq "EXP-044") {\n    $expectedRun') :]
    assert "$maxWaitHours = 4" in block.split("}")[0]
    # EXP-044 never touches the EXP-043 kernels: no kernel_sources, its own kernel slug
    assert "frontier-exp044" not in text  # the slug comes from $kernelSlug = "frontier-$expSlug"
    i = text.index("$kernelSources = @()")
    assert 'if ($Exp -eq "EXP-043") {' in text[i : i + 60]


def test_exp044_refuses_to_start_while_an_exp043_session_is_out():
    text = _ps1()
    i = text.index('if ($launching -and ($Exp -eq "EXP-044" -or $Exp -eq "EXP-045" -or $Exp -eq "EXP-046" -or $Exp -eq "EXP-048")) {')
    block = text[i : text.index("\n}\n", i)]
    assert '$s43 = "out\\kaggle\\EXP-043\\state.json"' in block
    assert ".pinned_commit" in block and "Stop-Run" in block
    # the guard comes before anything is launched
    assert i < text.index("$kaggle kernels push -p $stage")


def test_exp044_kernel_runs_the_check_and_keeps_no_checkpoint():
    tree = ast.parse(KERNEL_044.read_text(encoding="ascii"))
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__"
    assert consts["OUT"] == "/kaggle/working/EXP-044" and str(consts["SCRATCH"]).startswith("/tmp/")
    text = KERNEL_044.read_text(encoding="ascii")
    assert "gpu_ddp_check.py" in text and "kernel_sources" not in text.split('"""', 2)[2]
    flags = set(re.findall(r'"(--[a-z-]+)"', text)) - {"--quiet", "--no-deps"}
    assert flags == {"--data-dir", "--out", "--scratch"}
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gpu_ddp_check.py"), "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    ).stdout  # fmt: skip
    for f in flags:
        assert f in helptext


# ------------------------------------------------------------------------------------ EXP-045 --
PS1_045 = ROOT / "scripts" / "run_kaggle_exp045.ps1"
KERNEL_045 = ROOT / "scripts" / "kaggle" / "exp045_kernel.py"


def test_exp045_runs_on_a_cpu_session_between_exp043_sessions():
    wrapper = PS1_045.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-045" -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    block = text[text.index('if ($Exp -eq "EXP-045") {\n    # EXP-045 test 4') :]
    assert "$maxWaitHours = 6" in block.split("}")[0]
    # no GPU is requested: enable_gpu is switched off and the T4 machine shape removed, after $meta
    i = text.index("        machine_shape = \"NvidiaTeslaT4\"\n    }\n")
    cpu = text[i : text.index("Write-Ascii \"$stage\\kernel-metadata.json\"", i)]
    assert '$meta.enable_gpu = $false' in cpu and '$meta.Remove("machine_shape")' in cpu
    assert 'if (($Exp -eq "EXP-045" -and -not $Final) -or $Exp -eq "EXP-046" -or $Exp -eq "EXP-048") {' in cpu
    # the same between-sessions guard as EXP-044
    g = text.index('if ($launching -and ($Exp -eq "EXP-044" -or $Exp -eq "EXP-045" -or $Exp -eq "EXP-046" -or $Exp -eq "EXP-048")) {')
    assert g < text.index("$kaggle kernels push -p $stage")
    assert "frontier-exp045" not in text


def test_exp045_kernel_runs_the_scan_and_never_saves_belebele():
    tree = ast.parse(KERNEL_045.read_text(encoding="ascii"))
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__"
    assert consts["OUT"] == "/kaggle/working/EXP-045/coverage"  # test 4's files are never overwritten
    assert str(consts["BELEBELE"]).startswith("/tmp/")  # ShareAlike data stays out of the kernel output
    text = KERNEL_045.read_text(encoding="ascii")
    assert "belebele_contamination.py" in text and "nvidia-smi" not in text
    flags = set(re.findall(r'"(--[a-z-]+)"', text)) - {"--quiet", "--no-deps"}
    assert flags == {"--data-dir", "--out", "--belebele-dir", "--coverage"}
    helptext = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "belebele_contamination.py"), "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    ).stdout  # fmt: skip
    for f in flags:
        assert f in helptext


def test_exp045_results_are_publishable():
    sys.path.insert(0, str(ROOT / "scripts"))
    import publish_eval_results as pub

    assert {"summary.json", "SUMMARY.txt", "contamination.json", "belebele_items.jsonl"} <= set(pub.PUBLISHED)
    assert "coverage.json" in pub.PUBLISHED
    text = _ps1()
    # the coverage re-scan is read from EXP-045\coverage\ but published from EXP-045\ (so it lands in
    # evals/results/EXP-045/coverage/), and only NEW files under evals/results/EXP-045/ are accepted
    assert 'if ($Exp -eq "EXP-045") { $resultDir = "$outputDir\\$Exp\\coverage" }' in text
    i = text.index('if ($Exp -eq "EXP-045") { $resultDir')
    assert text.index("$publishSrc = $resultDir") < i < text.index('$summaryPath = "$resultDir\\summary.json"')
    assert "-Relaunch" in PS1_045.read_text(encoding="ascii")


# ------------------------------------------------------------------------------------ EXP-046 --
PS1_046 = ROOT / "scripts" / "run_kaggle_exp046_probe.ps1"
KERNEL_046 = ROOT / "scripts" / "kaggle" / "exp046_probe_kernel.py"


def test_exp046_probe_runs_on_a_cpu_session_between_exp043_sessions():
    wrapper = PS1_046.read_text(encoding="ascii")
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-046" -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    i = text.index('if ($Exp -eq "EXP-046") {\n    # EXP-046 step 1')
    block = text[i : text.index("\n}\n", i)]
    assert '$template = "scripts\\kaggle\\exp046_probe_kernel.py"' in block
    assert '$kernelSlug = "frontier-exp046-probe"' in block and "$maxWaitHours = 2" in block
    # the override comes after the defaults it replaces and before they are used
    assert text.index('$kernelSlug = "frontier-$expSlug"') < i < text.index('$kernelId = "$user/$kernelSlug"')
    assert i < text.index("foreach ($f in @($python, $template")
    # CPU session: the same switch as EXP-045 (no GPU requested)
    j = text.index('        machine_shape = "NvidiaTeslaT4"\n    }\n')
    cpu = text[j : text.index('Write-Ascii "$stage\\kernel-metadata.json"', j)]
    assert 'if (($Exp -eq "EXP-045" -and -not $Final) -or $Exp -eq "EXP-046" -or $Exp -eq "EXP-048") {' in cpu and "$meta.enable_gpu = $false" in cpu
    # the results folder is EXP-046\probe; publishing keeps the EXP-046 parent (-> evals/results/EXP-046/probe/)
    k = text.index('if ($Exp -eq "EXP-046") { $resultDir = "$outputDir\\$Exp\\probe" }')
    assert text.index("$publishSrc = $resultDir") < k < text.index('$summaryPath = "$resultDir\\summary.json"')
    assert 'if ($Exp -eq "EXP-046") { $msgText = "${Exp}: probe results from one Kaggle CPU session' in text


def test_exp046_probe_kernel_is_ascii_pinned_and_keeps_sangraha_out_of_the_output():
    source = KERNEL_046.read_text(encoding="ascii")
    tree = ast.parse(source)
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__"
    assert consts["OUT"] == "/kaggle/working/EXP-046/probe"
    assert str(consts["SCRATCH"]).startswith("/tmp/")  # downloaded parquet never lands in the output
    assert consts["PINS"] == "corpora/frontier/v2/sangraha_slice2.json"
    assert (ROOT / consts["PINS"]).is_file()
    assert "nvidia-smi\"]" not in source and "kernel_sources" not in source.split('"""', 2)[2]


# ------------------------------------------------------------------------- EXP-046 build A/B --
KERNEL_046B = ROOT / "scripts" / "kaggle" / "exp046_build_kernel.py"


def test_exp046_build_wrapper_and_runner_block():
    wrapper = (ROOT / "scripts" / "run_kaggle_exp046_build.ps1").read_text(encoding="ascii")
    assert 'param([Parameter(Mandatory = $true)][ValidateSet("A", "B")][string]$Part, [switch]$Relaunch)' in wrapper
    assert '& "$PSScriptRoot\\run_kaggle_exp038.ps1" -Exp "EXP-046" -Build $Part -Relaunch:$Relaunch' in wrapper
    text = _ps1()
    i = text.index('if ($Exp -eq "EXP-046" -and $Build) {')
    block = text[i : text.index("\n}\n", i)]
    assert '$template = "scripts\\kaggle\\exp046_build_kernel.py"' in block
    assert '$kernelSlug = "frontier-exp046-build-$($Build.ToLower())"' in block
    # its own state/report folder: the probe's finished state is never reused
    assert '$outDir = "out\\kaggle\\EXP-046-build-$Build"' in block and '$statePath = "$outDir\\state.json"' in block
    assert "$maxWaitHours = 14" in block
    assert text.index('if ($Exp -eq "EXP-046") {\n    # EXP-046 step 1') < i < text.index("New-Item -ItemType Directory -Force $outDir")
    assert i < text.index('$kernelId = "$user/$kernelSlug"') and i < text.index("foreach ($f in @($python, $template")
    # the part is filled in next to the commit
    assert 'if ($Build) { $code = $code.Replace("__BUILD_PART__", $Build) }' in text
    # the held-out texts: exported on the PC, uploaded once as a private dataset, mounted as a 2nd input
    h = text.index('    $heldSlug = "frontier-heldout-v1-text"')
    held = text[h : text.index('Add-Report "PASS 3b', h)]
    assert "scripts\\export_heldout_text.py --out $heldFile" in held and "datasets create -p $stage" in held
    assert "PRIVATE" in held and "$state.heldout_ready = $true" in held
    assert text.index('Add-Report "PASS 3: private dataset') < h < text.index("PASS 4")
    assert "if ($Build) { $meta.dataset_sources = @($datasetId, $heldId) }" in text
    # CPU session like the probe
    j = text.index('        machine_shape = "NvidiaTeslaT4"\n    }\n')
    cpu = text[j : text.index('Write-Ascii "$stage\\kernel-metadata.json"', j)]
    assert 'if (($Exp -eq "EXP-045" -and -not $Final) -or $Exp -eq "EXP-046" -or $Exp -eq "EXP-048") {' in cpu and "dataset_sources = @($datasetId, $heldId)" in cpu
    # only the small report folder is downloaded, never the corpus (about 9 GB)
    d = text.index("} elseif ($Build) {")
    dl = text[d : text.index("} else {", d)]
    assert "--file-pattern EXP-046/build-$Build/.*" in dl
    k = text.index('if ($Build) { $resultDir = "$outputDir\\$Exp\\build-$Build" }')
    assert text.index('if ($Exp -eq "EXP-046") { $resultDir = "$outputDir\\$Exp\\probe" }') < k
    assert k < text.index('$summaryPath = "$resultDir\\summary.json"')
    assert 'if ($Build) { $msgText = "${Exp}: build $Build reports' in text


def test_exp046_build_kernel_is_ascii_and_matches_the_build_script():
    source = KERNEL_046B.read_text(encoding="ascii")
    tree = ast.parse(source)
    consts = {
        n.targets[0].id: (n.value.args[0].value if isinstance(n.value, ast.Call) else n.value.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Call))
    }
    assert consts["COMMIT"] == "__PINNED_COMMIT__" and consts["PART"] == "__BUILD_PART__"
    assert str(consts["SCRATCH"]).startswith("/tmp/") and str(consts["BELEBELE"]).startswith("/tmp/")
    assert consts["HELDOUT_NAME"] == "heldout-v1.jsonl" and consts["WORKERS"] == 4
    assert (ROOT / consts["MANIFEST"]).is_file()
    assert 'WORKING / "EXP-046" / f"build-{PART}"' in source and 'WORKING / f"slice2-{PART}"' in source
    # every flag the kernel passes exists in build_slice2.py
    build = (ROOT / "scripts" / "build_slice2.py").read_text(encoding="utf-8")
    call = source[source.index('str(SRC / "scripts" / "build_slice2.py")') : source.index("subprocess.run(cmd, cwd=SRC)")]
    flags = re.findall(r'"(--[a-z0-9-]+)"', call)
    assert flags == ["--part", "--tokens-dir", "--heldout-jsonl", "--belebele-dir", "--work", "--data-out", "--out", "--workers"]
    assert flags and all(f'"{f}"' in build for f in flags), flags
    assert "--no-verify" not in flags and "--max-docs" not in flags and "--languages" not in flags
    assert "enable_gpu" not in source and "nvidia-smi" not in source
