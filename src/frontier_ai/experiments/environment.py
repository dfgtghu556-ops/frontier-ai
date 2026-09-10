"""Environment provenance: enough context to diagnose a result, nothing more.

Policy (D-027): capture a **stable, intentionally selected** set of fields. We do not dump
``pip freeze``, environment variables, hostname or username into every artifact: those
change between machines and runs, leak user information, and make records noisy to diff.
If full environment capture is ever needed, it belongs in a separate opt-in artifact.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

EXCLUDED_BY_POLICY: tuple[str, ...] = (
    "hostname",
    "username",
    "environment variables",
    "full pip freeze output",
)

# Packages that can plausibly change a numerical result in this repository.
TRACKED_PACKAGES: tuple[str, ...] = ("torch", "numpy", "tokenizers")


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception:  # pragma: no cover - optional dependencies
        return "not-installed"
    return str(getattr(module, "__version__", "unknown"))


def _torch_section() -> dict[str, Any]:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency of the project
        return {"available": False}

    section: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "num_threads": int(torch.get_num_threads()),
    }
    if torch.cuda.is_available() and section["device_count"]:  # pragma: no cover - GPU only
        section["device_name"] = torch.cuda.get_device_name(0)
    return section


def capture_environment(extra_packages: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return the environment section of an experiment record."""
    packages = {name: _package_version(name) for name in TRACKED_PACKAGES}
    for name in extra_packages:
        packages.setdefault(name, _package_version(name))

    try:  # installed distribution version, falls back to the source tree version
        from .. import __version__ as project_version
    except Exception:  # pragma: no cover - defensive
        project_version = "unknown"
    packages["frontier-ai"] = str(project_version)

    return {
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "torch": _torch_section(),
        "excluded_by_policy": list(EXCLUDED_BY_POLICY),
    }
