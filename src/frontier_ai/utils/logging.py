"""Tiny structured logging: human-readable console lines + JSONL metrics file."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """Console + JSONL logger.

    The JSONL file is what you plot later; the console line is what you stare at
    while the run is going.
    """

    def __init__(self, out_dir: str | Path, name: str = "train", verbose: bool = True) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{name}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self.verbose = verbose
        self.t0 = time.time()

    def log(self, **fields: Any) -> None:
        record: dict[str, Any] = {"t": round(time.time() - self.t0, 3), **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()
        if self.verbose:
            kv = "  ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
            print(f"[{record['t']:>7.1f}s] {kv}", flush=True)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1e4 or (value and abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def fmt_tokens_per_sec(tps: float) -> str:
    if tps >= 1e6:
        return f"{tps / 1e6:.2f}M tok/s"
    if tps >= 1e3:
        return f"{tps / 1e3:.1f}k tok/s"
    return f"{tps:.0f} tok/s"
