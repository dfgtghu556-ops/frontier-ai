#!/usr/bin/env python3
"""Build and upload the PRIVATE Hugging Face demo of the EXP-043 model (D-053).

    python scripts/hf_demo.py --weights out/kaggle/EXP-043/.../model_final.pt            # build + upload
    python scripts/hf_demo.py --weights model_final.pt --build-only                      # bundle only

Steps: (1) the float32 ``model_final.pt`` must match the SHA-256 recorded in
``evals/results/EXP-043/session-6/summary.json``; (2) a bundle is assembled in ``--out``: the demo app
(``apps/hf-demo/``), the ``frontier_ai`` package (Python files only), Frontier Tokenizer v2 with its
freeze record, the weights in fp16 and ``MODEL_INFO.json``; (3) the bundle is tested by loading it
the way the Space does and generating a few tokens; (4) unless ``--build-only``, it is uploaded to a
**private** Space ``<your Hugging Face user>/<--space>``. An existing Space that is public is refused:
making the demo public is a separate decision of the founder.

The Hugging Face token is read from this computer's Hugging Face login
(``huggingface_hub.login()``); it is never printed, written to the repository or sent anywhere else.
A one-page report is written to ``out/hf_demo/REPORT.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

APP_DIR = ROOT / "apps" / "hf-demo"
PACKAGE = ROOT / "src" / "frontier_ai"
TOKENIZER = ROOT / "tokenizers" / "frontier-tokenizer-v2"
SUMMARY = ROOT / "evals" / "results" / "EXP-043" / "session-6" / "summary.json"
DEFAULT_OUT = ROOT / "out" / "hf_demo" / "bundle"
REPORT = ROOT / "out" / "hf_demo" / "REPORT.txt"
DEFAULT_SPACE = "frontier-exp043-preview"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recorded_model_final(summary: Path = SUMMARY) -> dict:
    return json.loads(summary.read_text(encoding="utf-8"))["final"]["model_final"]


def _copy_package(dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n == "__pycache__" or (Path(_dir, n).is_file() and not n.endswith(".py"))}

    shutil.copytree(PACKAGE, dst, ignore=ignore)


def load_bundle_app(bundle: Path):
    """The bundle's app.py as a module (gradio is imported only when the page is built)."""
    spec = importlib.util.spec_from_file_location(f"hf_demo_app_{time.monotonic_ns()}", bundle / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SMOKE = """
import json, sys, time
sys.path.insert(0, ".")
import app
import frontier_ai
model, tok, _ = app.load()
t = time.monotonic()
text = app.continue_text(model, tok, "\u092d\u093e\u0930\u0924", max_new_tokens=int(sys.argv[1]), seed=1)
print(json.dumps({"prompt": "\u092d\u093e\u0930\u0924", "new_tokens": int(sys.argv[1]),
                  "seconds": round(time.monotonic() - t, 2), "text": text, "package": frontier_ai.__file__}))
"""


def smoke_test(bundle: Path, new_tokens: int) -> dict:
    """Load the bundle in a fresh Python process (only the bundle's own code), as the Space does."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, "-B", "-c", SMOKE, str(new_tokens)],
        cwd=bundle, env=env, capture_output=True, text=True, encoding="utf-8", timeout=900,
    )  # fmt: skip
    if r.returncode != 0:
        raise SystemExit(f"bundle test failed: {r.stderr.strip()[-800:]}")
    result = json.loads(r.stdout.strip().splitlines()[-1])
    package = Path(result.pop("package")).resolve()
    if Path(bundle).resolve() not in package.parents:  # it must use its own copy, not this repository's
        raise SystemExit(f"bundle test used {package}, not the bundle's frontier_ai package")
    if not isinstance(result.get("text"), str):
        raise SystemExit(f"bundle test produced no text: {result}")
    return result


def build_bundle(weights: Path, out: Path, recorded: dict, smoke_tokens: int = 8) -> dict:
    """Assemble and test the bundle; returns what MODEL_INFO.json records."""
    from frontier_ai.engine.weights import export_fp16, load_weights

    weights = Path(weights)
    digest = sha256_file(weights)
    if digest != recorded["sha256"]:
        raise SystemExit(f"{weights}: sha256 {digest} does not match the recorded {recorded['sha256']}")
    model, _info = load_weights(weights, "cpu")
    cfg = {k: getattr(model.cfg, k) for k in recorded.get("model_config", {})}
    if recorded.get("model_config") and cfg != recorded["model_config"]:
        raise SystemExit(f"model config {cfg} differs from the recorded {recorded['model_config']}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(APP_DIR / name, out / name)
    _copy_package(out / "frontier_ai")
    shutil.copytree(TOKENIZER, out / "tokenizers" / "frontier-tokenizer-v2")
    source = {"file": weights.name, "sha256": digest, "step": recorded.get("step"), "experiment": "EXP-043"}
    fp16 = export_fp16(model, out / "model_fp16.pt", source=source)
    n_params = sum(p.numel() for p in model.parameters())
    del model
    smoke = smoke_test(out, smoke_tokens)
    info = {
        "experiment": "EXP-043",
        "decision": "D-053 (private preview; public release is a separate founder decision)",
        "n_params": n_params,
        "source": source,
        "fp16": fp16,
        "smoke_test": smoke,
    }
    (out / "MODEL_INFO.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return info


def upload(bundle: Path, space: str, api=None) -> str:
    """Upload ``bundle`` to the PRIVATE Space ``<user>/<space>``; returns its URL."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    user = api.whoami()["name"]
    repo_id = space if "/" in space else f"{user}/{space}"
    try:
        existing = api.repo_info(repo_id, repo_type="space")
    except Exception:  # not there yet (huggingface_hub raises RepositoryNotFoundError)
        existing = None
    if existing is not None and not getattr(existing, "private", True):
        raise SystemExit(f"{repo_id} exists and is PUBLIC; this script only uploads to a private Space")
    if existing is None:
        api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=True)
    api.upload_folder(
        folder_path=str(bundle),
        repo_id=repo_id,
        repo_type="space",
        commit_message="EXP-043 research preview (scripts/hf_demo.py)",
    )
    return f"https://huggingface.co/spaces/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", required=True, help="EXP-043 model_final.pt (float32)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--space", default=DEFAULT_SPACE, help="Space name (or user/name)")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--report", default=str(REPORT))
    args = p.parse_args(argv)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"REPORT hf_demo (EXP-043 private preview) - {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    try:
        info = build_bundle(Path(args.weights), Path(args.out), recorded_model_final())
        lines += [
            f"PASS 1: model_final.pt sha256 matches the EXP-043 record ({info['source']['sha256'][:16]}...)",
            f"PASS 2: bundle built ({info['n_params']:,} parameters; "
            f"fp16 file {info['fp16']['bytes']:,} bytes)",
            f"PASS 3: bundle test generated {info['smoke_test']['new_tokens']} tokens in "
            f"{info['smoke_test']['seconds']} s",
        ]
        if args.build_only:
            lines.append(f"INFO 4: --build-only: nothing uploaded (bundle in {args.out})")
        else:
            url = upload(Path(args.out), args.space)
            lines += [
                f"PASS 4: uploaded to the PRIVATE Space {url}",
                "NEXT: open that link while logged in to Hugging Face. The page needs about 5-10 minutes",
                "      to build the first time; then type the beginning of a text and press the button.",
            ]
        lines.append("RESULT: COMPLETE. Paste this report into the Arena chat.")
        code = 0
    except SystemExit as e:
        lines += [f"FAIL: {e}", "RESULT: FAILED. Paste this report into the Arena chat."]
        code = 1
    except Exception as e:  # report every other failure plainly
        lines += [f"FAIL: {type(e).__name__}: {e}", "RESULT: FAILED. Paste this report into the Arena chat."]
        code = 1
    text = "\n".join(lines) + "\n"
    report.write_text(text, encoding="utf-8")
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
