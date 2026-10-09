"""Several corpus slices read as one training corpus (EXP-047 step 1; approved 2026-10-09, "approve D").

EXP-047 trains on v2-slice1 (EXP-037, 13 files) plus v2-slice2 (EXP-046 parts A and B, 13 files). Both
slices have one ``<lang>.bin`` per language, so :class:`~frontier_ai.data.multi.MultiTokenDataset`'s
default names (the file name) would clash. This module adds what EXP-047 §5 asks for, without changing
how EXP-043 reads slice 1:

1. **Pinned manifests:** each slice's manifest in the repository must have the SHA-256 recorded below
   (slice 1: EXP-037, also recorded in the EXP-046 build summaries; slice 2: D-051). An edited or wrong
   manifest stops the run before training.
2. **File checks against the manifest:** every token file exists with the manifest's size, and its meta
   side-car has the manifest's ``n_train`` / ``n_val``; optionally the full SHA-256 too (minutes for
   gigabytes, so off by default; the size check and the meta check catch a wrong or truncated file).
3. **Slice-aware names:** ``<slice>/<lang>`` (``s1/hi``, ``s2a/en``, ``s2b/kn``) in a fixed order
   (slices in the order given, files in manifest order), so natural size-proportional sampling and the
   one-pass order work unchanged over all 26 files.
4. **Validation per language across slices:** :func:`by_language` and :func:`combine_bpb` pool a
   language's validation bits and bytes over the slices (the same pooled formula as EXP-043's
   ``score_all``); slice 1's own per-language numbers stay the main yardstick (EXP-047 §5), so EXP-047 and
   EXP-043 compare on the same text.

Nothing here trains or downloads; it only reads manifests and file headers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .multi import MultiTokenDataset

ROOT = Path(__file__).resolve().parents[3]
BLOCK = 8 << 20


class SliceError(ValueError):
    """A slice's manifest or files do not match what is recorded."""


@dataclass(frozen=True)
class Slice:
    name: str  # the prefix of the part names, e.g. "s1"
    manifest: str  # repository-relative path of the manifest
    manifest_sha256: str  # pinned
    mount_hint: str  # a substring of the Kaggle mount path of its data (to find the folder)
    description: str


V2_SLICE1 = Slice(
    "s1",
    "evals/results/EXP-037/manifest.json",
    "b43c0d91c74efc7cb7a7c2480b190a95a44c34fc35427d4d5c13514f6ff6a11b",
    "frontier-v2-tok2-13lang",
    "v2-slice1 (EXP-037; 2,783,830,088 training tokens)",
)
V2_SLICE2_A = Slice(
    "s2a",
    "evals/results/EXP-046/build-A/manifest.json",
    "3782679fe4e71e45077692b42d63f303ecd0d1914c0d1a10da1f86495f376c42",
    "frontier-v2-slice2-a",
    "v2-slice2 part A (EXP-046 / D-051: en ur as bn gu hi)",
)
V2_SLICE2_B = Slice(
    "s2b",
    "evals/results/EXP-046/build-B/manifest.json",
    "542988db5717357d09a76c7384558b0c27cd2034d07d59e1a16b9092f83f7222",
    "frontier-v2-slice2-b",
    "v2-slice2 part B (EXP-046 / D-051: kn ml mr or pa ta te)",
)
EXP047_SLICES = (V2_SLICE1, V2_SLICE2_A, V2_SLICE2_B)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(s: Slice, root: Path = ROOT) -> dict[str, Any]:
    """The slice's manifest, after checking its SHA-256 against the pinned value."""
    path = root / s.manifest
    got = sha256_file(path)
    if got != s.manifest_sha256:
        raise SliceError(
            f"{s.name}: {s.manifest} has SHA-256 {got}, the recorded value is {s.manifest_sha256}"
        )
    man = json.loads(path.read_text(encoding="utf-8"))
    langs = [f["language"] for f in man["files"]]
    if len(set(langs)) != len(langs):
        raise SliceError(f"{s.name}: a language appears twice in {s.manifest}")
    return man


def check_files(
    s: Slice, manifest: Mapping[str, Any], data_dir: Path, full_sha256: bool = False
) -> list[dict]:
    """Check every token file of the slice in ``data_dir``; raise :class:`SliceError` listing all problems."""
    rows, problems = [], []
    for f in manifest["files"]:
        b = Path(data_dir) / f["path"]
        m = Path(data_dir) / f["meta"]
        row: dict[str, Any] = {"slice": s.name, "language": f["language"], "path": str(b)}
        if not b.exists() or not m.exists():
            problems.append(f"{s.name}/{f['language']}: {b.name} or {m.name} is missing in {data_dir}")
            continue
        row["bytes_ok"] = b.stat().st_size == f["bytes"]
        meta = json.loads(m.read_text(encoding="utf-8"))
        row["meta_ok"] = meta.get("n_train") == f["n_train"] and meta.get("n_val") == f["n_val"]
        if full_sha256:
            row["sha256_ok"] = sha256_file(b) == f["sha256"]
        for key in ("bytes_ok", "meta_ok", "sha256_ok"):
            if row.get(key) is False:
                problems.append(
                    f"{s.name}/{f['language']}: {key.removesuffix('_ok')} differs from {s.manifest}"
                )
        rows.append(row)
    if problems:
        raise SliceError("; ".join(problems))
    return rows


def locate(s: Slice, manifest: Mapping[str, Any], input_root: Path) -> Path:
    """The one folder under ``input_root`` holding all of the slice's token files.

    If several folders hold them, only those whose path contains the slice's ``mount_hint`` count."""
    bins = [f["path"] for f in manifest["files"]]
    holders = sorted(
        {p.parent for p in Path(input_root).rglob(bins[0]) if all((p.parent / x).exists() for x in bins)}
    )
    if len(holders) > 1:
        holders = [d for d in holders if s.mount_hint in str(d)]
    if len(holders) != 1:
        raise SliceError(f"{s.name}: expected one folder with all {len(bins)} token files, found {holders}")
    return holders[0]


def open_slices(
    slices: Sequence[tuple[Slice, Path]], full_sha256: bool = False, root: Path = ROOT
) -> tuple[MultiTokenDataset, dict[str, Any]]:
    """One :class:`MultiTokenDataset` over all slices, names ``<slice>/<lang>``, after every check.

    Returns the dataset and a small record (manifest hashes, file checks, token totals) for summaries."""
    if len({s.name for s, _ in slices}) != len(slices):
        raise SliceError("each slice may be given once")
    paths, names, record = [], [], {"slices": []}
    for s, data_dir in slices:
        man = load_manifest(s, root)
        rows = check_files(s, man, data_dir, full_sha256=full_sha256)
        for f in man["files"]:
            paths.append(Path(data_dir) / f["path"])
            names.append(f"{s.name}/{f['language']}")
        record["slices"].append(
            {
                "name": s.name,
                "description": s.description,
                "manifest": s.manifest,
                "manifest_sha256": s.manifest_sha256,
                "data_dir": str(data_dir),
                "files": len(rows),
                "train_tokens": sum(f["n_train"] for f in man["files"]),
                "val_tokens": sum(f["n_val"] for f in man["files"]),
                "full_sha256_checked": full_sha256,
            }
        )
    ds = MultiTokenDataset(paths, names=names)
    record["train_tokens"] = ds.n_train
    record["val_tokens"] = ds.n_val
    return ds, record


def by_language(names: Sequence[str]) -> dict[str, list[str]]:
    """``{"hi": ["s1/hi", "s2a/hi"], ...}`` in the order the names are given."""
    out: dict[str, list[str]] = {}
    for n in names:
        if "/" not in n:
            raise SliceError(f"{n!r} is not a slice-aware name (<slice>/<lang>)")
        out.setdefault(n.split("/", 1)[1], []).append(n)
    return out


def combine_bpb(ds: MultiTokenDataset, nats: Mapping[str, float]) -> dict[str, Any]:
    """Pool per-part validation losses (nats/token, as ``full_split_loss`` gives) per language.

    bits = loss / ln 2 x validation tokens, summed over a language's parts and divided by their summed
    validation bytes: the pooled formula of EXP-043's ``score_all``. Also returns each slice's own
    per-language bpb (``per_part``), so slice 1 can stay the main yardstick."""
    per_part = {n: nats[n] / math.log(2) * ds.parts[n].tokens_per_byte("val") for n in ds.names}
    per_lang = {}
    for lang, parts in by_language(ds.names).items():
        bits = sum(nats[n] / math.log(2) * ds.parts[n].n_val for n in parts)
        nbytes = sum(ds.parts[n].split_lengths("val")[0] for n in parts)
        per_lang[lang] = bits / nbytes
    return {"per_part": per_part, "per_language": per_lang}
