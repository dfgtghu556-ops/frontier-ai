#!/usr/bin/env python3
"""EXP-037 part 1: freeze Frontier Tokenizer v2 = v1 plus special tokens, nothing else.

    python scripts/freeze_tokenizer_v2.py            # writes tokenizers/frontier-tokenizer-v2/

v2 keeps v1's 32,512 merges, its mark-aware pre-tokenizer and all 32,768 ordinary ids, and
appends 128 special tokens at ids 32768..32895 (``<|endoftext|>``, ``<|pad|>`` and 126 reserved
slots; ``frontier_ai.tokenization.frozen.V2_SPECIAL_TOKENS``). Five gates run before anything is
written:

  A  base      v1 loads through its verifying loader and its fingerprint is the D-041 one;
  B  structure v2 has vocab 32,896, v1's merges (identical list), the same pre-tokenizer and
               exactly the 128 specials in order;
  C  ordinary  on every test text, ``v2.encode_ordinary(text) == v1.encode(text)``: v1's golden
               samples (the ids recorded in v1's FREEZE.json), every excerpt in the committed
               EXP-035 and EXP-036 samples (13 languages) and texts that contain special-token
               strings. No ordinary encoding contains an id >= 32768;
  D  specials  each special string, encoded with ``encode``, is its own single id and decodes back;
               with ``encode_ordinary`` it never produces a special id;
  E  lossless  ``decode(encode_ordinary(text)) == text`` on every test text.

Any gate failure: nothing is written (exit 1). An existing freeze is never overwritten (exit 2).
After writing, the copy is re-verified through ``load_frontier_tokenizer_v2``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_ai.tokenization.frozen import (  # noqa: E402
    ARTIFACT_FILE,
    ARTIFACT_SUBDIR,
    ENDOFTEXT,
    FREEZE_NAME,
    FREEZE_SCHEMA,
    FRONTIER_TOKENIZER_V1,
    FRONTIER_TOKENIZER_V2,
    V2_SPECIAL_TOKENS,
    FrozenTokenizerError,
    artifact_dir_sha256,
    derive_with_special_tokens,
    file_sha256,
    load_frozen_tokenizer,
    read_freeze,
)

V1_DIR_SHA256 = "b39afa08fa86b0d1ccf7b7e1b59df305b5678bdd59061c6fae01245061321144"  # D-041
SAMPLE_FILES = ("evals/results/EXP-035/samples.jsonl", "evals/results/EXP-036/samples.jsonl")
ADVERSARIAL = [
    ("special_in_text", f"A web page may contain {ENDOFTEXT} as plain text."),
    ("special_glued", f"end{ENDOFTEXT}start<|pad|><|reserved_0|>"),
    ("special_near_miss", "<|endoftext| > <|ENDOFTEXT|> <|reserved_126|> <|reserved_-1|>"),
    ("special_indic", f"भारत{ENDOFTEXT}देश — ٹھیک ہے <|pad|> தமிழ்"),
]
CHANGE_POLICY = (
    "Frontier Tokenizer v2 is never edited in place. Any change (retraining, a different vocabulary, "
    "other special tokens, giving a reserved slot a meaning, a different pre-tokenizer or implementation "
    "artifact) requires the founder's approval, is recorded as a new experiment, and produces a new version "
    "with its own freeze record (D-041)."
)
DOWNSTREAM_USE = (
    "Load only via frontier_ai.tokenization.frozen.load_frontier_tokenizer_v2(). Encode training text with "
    "encode_ordinary() (special-token strings inside text stay text) and insert special ids explicitly. "
    "Ordinary text encodes exactly as with frontier-tokenizer-v1, so v1 token counts remain valid. Cite "
    "'frontier-tokenizer-v2' plus artifact.dir_sha256."
)


def test_texts(root: Path, v1_freeze: dict) -> list[tuple[str, str, list[int] | None]]:
    """(id, text, ids recorded by v1 or None). Samples are read from the committed results."""
    texts: list[tuple[str, str, list[int] | None]] = [
        (f"v1_golden:{g['id']}", g["text"], list(g["ids"])) for g in v1_freeze["golden_samples"]
    ]
    for rel in SAMPLE_FILES:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"test texts missing: {path}")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            rec = json.loads(line)
            text = rec.get("excerpt") or rec.get("text")  # EXP-036 uses "excerpt", EXP-035 "text"
            if text:
                texts.append((f"{rel.split('/')[2]}:{n}:{rec.get('language', '?')}", text, None))
    texts.extend((f"adversarial:{sid}", text, None) for sid, text in ADVERSARIAL)
    return texts


def run_gates(root: Path) -> tuple[dict, object, object, list]:
    gates: dict = {}
    v1_dir = root / "tokenizers" / FRONTIER_TOKENIZER_V1
    v1, v1_freeze = load_frozen_tokenizer(v1_dir)
    observed = v1_freeze["artifact"]["dir_sha256"]
    gates["A_base"] = {
        "what": "frontier-tokenizer-v1 verifies through its loader; fingerprint == D-041",
        "expected": V1_DIR_SHA256,
        "observed": observed,
        "pass": observed == V1_DIR_SHA256 and artifact_dir_sha256(v1_dir / ARTIFACT_SUBDIR) == V1_DIR_SHA256,
    }
    v2 = derive_with_special_tokens(v1, V2_SPECIAL_TOKENS)
    problems = []
    if v2.vocab_size != 32768 + len(V2_SPECIAL_TOKENS):
        problems.append(f"vocab_size {v2.vocab_size}")
    if v2.merges != v1.merges:
        problems.append("merges differ from v1")
    if v2._pretoken != v1._pretoken:  # noqa: SLF001
        problems.append("pre-tokenizer differs from v1")
    if list(v2.special_token_ids) != list(V2_SPECIAL_TOKENS):
        problems.append("special tokens differ")
    if [v2.special_token_ids[s] for s in V2_SPECIAL_TOKENS] != list(range(32768, v2.vocab_size)):
        problems.append("special ids are not 32768..")
    gates["B_structure"] = {
        "expected": {"vocab_size": 32896, "merges": 32512, "pretoken": "mark_aware", "specials": 128},
        "problems": problems,
        "pass": not problems,
    }
    texts = test_texts(root, v1_freeze)
    c_fail, e_fail = [], []
    for tid, text, recorded in texts:
        ids = v2.encode_ordinary(text)
        reference = v1.encode(text)
        if ids != reference or (recorded is not None and ids != recorded) or any(i >= 32768 for i in ids):
            c_fail.append(tid)
        if v2.decode(ids) != text:
            e_fail.append(tid)
    gates["C_ordinary_equals_v1"] = {
        "texts": len(texts),
        "v1_golden_samples": len(v1_freeze["golden_samples"]),
        "failing": c_fail[:10],
        "pass": not c_fail,
    }
    d_fail = []
    for s in V2_SPECIAL_TOKENS:
        sid = v2.special_token_ids[s]
        if v2.encode(s) != [sid] or v2.decode([sid]) != s or any(i >= 32768 for i in v2.encode_ordinary(s)):
            d_fail.append(s)
    gates["D_specials"] = {"checked": len(V2_SPECIAL_TOKENS), "failing": d_fail[:10], "pass": not d_fail}
    gates["E_lossless"] = {"texts": len(texts), "failing": e_fail[:10], "pass": not e_fail}
    return gates, v1, v2, texts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT), help="repository root (tests use a copy)")
    p.add_argument("--exp-id", default="EXP-037")
    args = p.parse_args(argv)
    root = Path(args.root)
    dest = root / "tokenizers" / FRONTIER_TOKENIZER_V2
    if (dest / FREEZE_NAME).exists():
        print(
            f"[freeze-v2] {dest / FREEZE_NAME} already exists; a frozen tokenizer is never overwritten",
            file=sys.stderr,
        )
        return 2
    try:
        gates, v1, v2, _texts = run_gates(root)
    except (FrozenTokenizerError, FileNotFoundError) as exc:
        print(f"[freeze-v2] INPUT ERROR: {exc}", file=sys.stderr)
        return 2
    for key, gate in gates.items():
        print(f"[freeze-v2] gate {key}: {'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    if not all(g["pass"] for g in gates.values()):
        print("[freeze-v2] FAILED; nothing was written.", file=sys.stderr)
        return 1

    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    art_dir = tmp / ARTIFACT_SUBDIR
    v2.save(art_dir)
    v1_freeze = read_freeze(root / "tokenizers" / FRONTIER_TOKENIZER_V1)
    golden = [
        {"id": g["id"], "text": g["text"], "ids": v2.encode(g["text"])} for g in v1_freeze["golden_samples"]
    ]
    golden += [{"id": sid, "text": text, "ids": v2.encode(text)} for sid, text in ADVERSARIAL]
    files = sorted(q.name for q in art_dir.iterdir() if q.is_file())
    freeze = {
        "schema": FREEZE_SCHEMA,
        "tokenizer_id": "frontier-tokenizer",
        "tokenizer_version": "v2",
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_by": f"{args.exp_id} (scripts/freeze_tokenizer_v2.py)",
        "decision": 'EXP-037 (founder: "approve EXP-037", 2026-09-30); to be recorded as D-046',
        "artifact": {
            "dir": f"tokenizers/{FRONTIER_TOKENIZER_V2}/{ARTIFACT_SUBDIR}",
            "impl": "bpe_python",
            "impl_version": v2.impl_version(),
            "files": {name: file_sha256(art_dir / name) for name in files},
            "dir_sha256": artifact_dir_sha256(art_dir),
            "size_bytes": sum((art_dir / n).stat().st_size for n in files),
            "vocab_size": v2.vocab_size,
            "merges": len(v2.merges),
            "pretoken": v2._pretoken,  # noqa: SLF001
            "special_tokens": list(v2.special_token_ids),
            "special_token_ids": {
                "first": 32768,
                "last": v2.vocab_size - 1,
                ENDOFTEXT: 32768,
                "<|pad|>": 32769,
            },
            "note": "hashes are over raw bytes; git stores this directory byte-exactly via .gitattributes",
        },
        "lineage": {
            "base": FRONTIER_TOKENIZER_V1,
            "base_dir_sha256": V1_DIR_SHA256,
            "change": "128 special tokens appended at ids 32768..32895; merges, pre-tokenizer and all "
            "ordinary ids unchanged",
            "experiment": args.exp_id,
        },
        "ordinary_text_rule": "encode_ordinary(text) == frontier-tokenizer-v1 encode(text) (gate C)",
        "gates": gates,
        "golden_samples": golden,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "change_policy": CHANGE_POLICY,
        "downstream_use": DOWNSTREAM_USE,
    }
    with open(tmp / FREEZE_NAME, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    tmp.rename(dest)
    try:
        load_frozen_tokenizer(dest)
    except FrozenTokenizerError as exc:
        shutil.rmtree(dest)
        print(f"[freeze-v2] FAILED: the frozen copy did not verify ({exc}); removed.", file=sys.stderr)
        return 1
    assert (dest / ARTIFACT_SUBDIR / ARTIFACT_FILE).is_file()
    print(
        f"[freeze-v2] frozen: {dest} | dir_sha256 {freeze['artifact']['dir_sha256'][:16]} | vocab "
        f"{v2.vocab_size:,} | {gates['C_ordinary_equals_v1']['texts']} texts identical to v1 | "
        f"{len(golden)} golden samples"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
