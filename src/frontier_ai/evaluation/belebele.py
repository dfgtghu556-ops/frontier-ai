"""EXP-045: Belebele reading comprehension, same-text bits, and document scoring (pre-registered).

Everything here was fixed in EXPERIMENTS.md (EXP-045) **before** the EXP-043 model existed:

* **Data**: ``facebook/belebele`` at the pinned revision :data:`REVISION` (CC BY-SA 4.0, not
  gated). One JSON-lines file per language; the file of each of our 13 languages is pinned by its
  git blob hash and size (:data:`FILES`, read from the Hugging Face API on 2026-10-04). The data is
  downloaded at evaluation time and never committed (ShareAlike).
* **Belebele scoring**: zero-shot. The text is the passage, a newline, the question, a newline,
  then one answer option. Each option's score is its summed log-probability divided by its UTF-8
  bytes; the highest score is the model's answer (ties: the lowest option number). If the text
  does not fit the 512-token context, the passage is cut from the left and the item is counted.
* **Report**: accuracy per language with a 95% Wilson interval; chance is 25%; a language is
  "above chance" only if its whole interval is above 25%.
* **Same text in every language**: every Belebele passage is a FLORES-200 passage translated by
  people. We report the total bits needed to encode the same passages in each language.

Implementation details recorded here (they are not choices that could favour a result):

* Training documents end with ``<|endoftext|>`` (EXP-037), so the text after that token is the
  start of a document. Every scored sequence therefore starts with ``<|endoftext|>`` — except a
  Belebele text whose passage had to be cut (it no longer starts at a document start).
* The three pieces (passage, ``"\\n" + question + "\\n"``, option) are tokenized separately with
  ``encode_ordinary`` and concatenated, so an option's tokens are the same in every context.
* Texts longer than the context are scored in overlapping windows (stride = half the context);
  every token is scored exactly once, with at least half a context of preceding text.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = "facebook/belebele"
REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
LICENSE = "CC BY-SA 4.0 (test set; the Belebele training set is CC BY-NC and is NOT used)"
# our language code -> (Belebele dialect = file name, git blob sha1, bytes); HF API, 2026-10-04
FILES: dict[str, tuple[str, str, int]] = {
    "as": ("asm_Beng", "12ff7a2e433e31badb40d2657441177d1c6f93df", 3169616),
    "bn": ("ben_Beng", "72dcfb7bd316e1cb9b5037eff9c6abf5126f31e5", 3235932),
    "en": ("eng_Latn", "a678517e601a401ba28a8c51287b7e186b161ef6", 836071),
    "gu": ("guj_Gujr", "77c173d245611c8aceed4dba8c9f3bd8143aea28", 3051820),
    "hi": ("hin_Deva", "9bb854219e1388acf4f8b293aa775df0c6b1e92e", 3124631),
    "kn": ("kan_Knda", "c5ca4c0b6994ae70e8b023e0fa6843b07a25d03e", 3487845),
    "ml": ("mal_Mlym", "6790d190d2f114d1adcf30bc9d6abaaa0f16ca12", 3830868),
    "mr": ("mar_Deva", "70474042c7cc2f55aed9a0c3e299fca8ac88f83d", 3269717),
    "or": ("ory_Orya", "1135f7099e113c3971fffd65f159cb4b6fdb0e0f", 3365805),
    "pa": ("pan_Guru", "363949d43c3e2ea366a92127c718dd80e0bc6148", 3161744),
    "ta": ("tam_Taml", "ee506b06ce2c9114ccbc667ed3898f97501e005b", 3893367),
    "te": ("tel_Telu", "7d6186bb45a1693a9f2fe9f0377cb3f140301abf", 3331996),
    "ur": ("urd_Arab", "7838c36f052c76ac72d7ab66205a8f567f492e4d", 3075650),
}
LANGUAGES = tuple(FILES)
QUESTIONS_PER_LANGUAGE = 900
CHANCE = 0.25
Z95 = 1.959963984540054
PROTOCOL = (
    "zero-shot; text = passage + '\\n' + question + '\\n' + option (pieces tokenized separately, "
    "Frontier Tokenizer v2 encode_ordinary), preceded by <|endoftext|> unless the passage was cut; "
    "score = summed log-probability of the option tokens / option UTF-8 bytes; answer = highest "
    "score (ties: lowest option number); passage cut from the left if the text exceeds the context"
)
DOC_PROTOCOL = (
    "each text scored as one document: <|endoftext|> + text tokens, every text token scored once "
    "(overlapping windows with stride block/2 if longer than the context); bits / UTF-8 bytes"
)


class BelebeleError(ValueError):
    """The Belebele file is not the pinned one, or its content is not as expected."""


@dataclass(frozen=True)
class Item:
    link: str
    question_number: int
    passage: str
    question: str
    options: tuple[str, str, str, str]
    answer: int  # 0-based index of the correct option

    @property
    def key(self) -> str:
        return f"{self.link}#{self.question_number}"


# ------------------------------------------------------------------------------- data -----------
def git_blob_sha1(data: bytes) -> str:
    """The git object id of ``data`` (what the Hugging Face API reports for non-LFS files)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_url(lang: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{FILES[lang][0]}.jsonl"


def verify_bytes(lang: str, data: bytes) -> None:
    dialect, oid, size = FILES[lang]
    if len(data) != size or git_blob_sha1(data) != oid:
        raise BelebeleError(
            f"{dialect}.jsonl: {len(data)} bytes / git blob {git_blob_sha1(data)[:12]}…, "
            f"pinned {size} bytes / {oid[:12]}…"
        )


def parse_items(data: bytes, lang: str, expected: int | None = QUESTIONS_PER_LANGUAGE) -> list[Item]:
    """The questions of one language file, in file order, with every field checked."""
    dialect = FILES[lang][0]
    items: list[Item] = []
    for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dialect") != dialect:
            raise BelebeleError(f"{dialect} line {n}: dialect {r.get('dialect')!r}")
        options = tuple(str(r[f"mc_answer{i}"]) for i in range(1, 5))
        answer = int(r["correct_answer_num"]) - 1
        if answer not in range(4) or not all(o.strip() for o in options):
            raise BelebeleError(f"{dialect} line {n}: bad options or answer")
        items.append(
            Item(
                str(r["link"]),
                int(r["question_number"]),
                str(r["flores_passage"]),
                str(r["question"]),
                options,
                answer,
            )
        )
    if len({i.key for i in items}) != len(items):
        raise BelebeleError(f"{dialect}: duplicate (link, question_number) keys")
    if expected is not None and len(items) != expected:
        raise BelebeleError(f"{dialect}: {len(items)} questions, expected {expected}")
    return items


def download(lang: str, dest: str | Path, opener: Callable[[str], bytes] | None = None) -> Path:
    """Fetch one pinned language file into ``dest`` (verified; an existing good copy is reused)."""
    path = Path(dest) / f"{FILES[lang][0]}.jsonl"
    if path.is_file():
        try:
            verify_bytes(lang, path.read_bytes())
            return path
        except BelebeleError:
            path.unlink()
    if opener is None:

        def opener(url: str) -> bytes:
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https URL
                return resp.read()

    data = opener(file_url(lang))
    verify_bytes(lang, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def load_language(
    directory: str | Path, lang: str, verify: bool = True, expected: int | None = 900
) -> list[Item]:
    data = (Path(directory) / f"{FILES[lang][0]}.jsonl").read_bytes()
    if verify:
        verify_bytes(lang, data)
    return parse_items(data, lang, expected)


def passages(items: Sequence[Item]) -> dict[str, str]:
    """Distinct passages: key = the sorted question keys that share the passage (same in every language)."""
    groups: dict[tuple[str, str], list[str]] = {}
    for it in items:
        groups.setdefault((it.link, it.passage), []).append(it.key)
    return {"|".join(sorted(keys)): text for (_link, text), keys in groups.items()}


# ----------------------------------------------------------------------------- statistics -------
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def accuracy_report(correct: Sequence[bool]) -> dict[str, Any]:
    n, k = len(correct), int(sum(bool(c) for c in correct))
    lo, hi = wilson(k, n)
    return {
        "questions": n,
        "correct": k,
        "accuracy": (k / n) if n else None,
        "wilson95": [lo, hi],
        "above_chance": bool(n and lo > CHANCE),
    }


# ----------------------------------------------------------------------------- scoring ----------
def segments(seq: Sequence[int], first_target: int, block: int) -> list[tuple[list[int], int]]:
    """Pieces ``(ids, n_target)`` that together score ``seq[first_target:]`` once each.

    A piece feeds ``ids[:-1]`` (at most ``block`` tokens) and scores its last ``n_target``
    tokens. Short sequences are one piece; longer ones use windows with stride ``block // 2``.
    """
    if not 1 <= first_target <= len(seq):
        raise ValueError("first_target must be in [1, len(seq)]")
    seq = list(seq)
    out: list[tuple[list[int], int]] = []
    done = first_target
    stride = max(1, block // 2)
    while done < len(seq):
        end = min(len(seq), max(done + stride, block + 1) if not out else done + stride)
        start = max(0, end - (block + 1))
        out.append((seq[start:end], end - done))
        done = end
    return out


@torch.no_grad()
def score_pieces(
    model: torch.nn.Module,
    pieces: Sequence[tuple[Sequence[int], int]],
    device: torch.device | str,
    batch_size: int = 16,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Summed log-probability (nats, float64) of the last ``n_target`` tokens of each piece."""
    model.eval()
    device = torch.device(device)
    out = np.zeros(len(pieces), dtype=np.float64)
    order = sorted(range(len(pieces)), key=lambda i: len(pieces[i][0]))  # similar lengths per batch
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        width = max(len(pieces[i][0]) for i in idx) - 1
        x = torch.zeros((len(idx), width), dtype=torch.long)
        y = torch.zeros((len(idx), width), dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.bool)
        for row, i in enumerate(idx):
            ids, n_target = pieces[i]
            if not 1 <= n_target <= len(ids) - 1:
                raise ValueError(f"piece {i}: n_target {n_target} for {len(ids)} ids")
            t = torch.as_tensor(list(ids), dtype=torch.long)
            n = len(ids) - 1
            x[row, :n], y[row, :n] = t[:-1], t[1:]
            mask[row, n - n_target : n] = True
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x).logits
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        nll = (nll.reshape(len(idx), width) * mask).double().sum(dim=1).cpu().numpy()
        for row, i in enumerate(idx):
            out[i] = -nll[row]
    return out


def option_pieces(tok: Any, item: Item, block: int, eot: int) -> tuple[list[tuple[list[int], int]], bool]:
    """The four scored sequences of one question, and whether the passage had to be cut."""
    p = tok.encode_ordinary(item.passage)
    q = tok.encode_ordinary("\n" + item.question + "\n")
    pieces, cut = [], False
    for option in item.options:
        o = tok.encode_ordinary(option)
        if not o:
            raise BelebeleError(f"{item.key}: an option encodes to no tokens")
        room = block + 1 - len(q) - len(o)  # a piece may hold block + 1 ids (input = block tokens)
        if 1 + len(p) <= room:
            seq = [eot, *p, *q, *o]
        else:
            cut = True
            if room < 1:  # not even the question fits: keep its end (never seen in practice)
                seq = [*q, *o][-(block + 1) :]
            else:
                seq = [*p[len(p) - room :], *q, *o]
        pieces.append((seq, len(o)))
    return pieces, cut


def belebele_language(
    model: torch.nn.Module,
    tok: Any,
    items: Sequence[Item],
    block: int,
    device: torch.device | str,
    batch_size: int = 16,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Score every question of one language; per-question rows plus the accuracy report."""
    eot = tok.special_token_ids["<|endoftext|>"]
    pieces: list[tuple[list[int], int]] = []
    cuts: list[bool] = []
    for it in items:
        ps, cut = option_pieces(tok, it, block, eot)
        pieces.extend(ps)
        cuts.append(cut)
    logp = score_pieces(model, pieces, device, batch_size, amp_dtype)
    rows = []
    for n, it in enumerate(items):
        scores = [logp[4 * n + j] / len(it.options[j].encode("utf-8")) for j in range(4)]
        pred = int(np.argmax(scores))  # first maximum = lowest option number
        rows.append(
            {
                "key": it.key,
                "pred": pred,
                "answer": it.answer,
                "correct": pred == it.answer,
                "cut": cuts[n],
                "scores": [round(s, 6) for s in scores],
            }
        )
    rep = accuracy_report([r["correct"] for r in rows])
    rep["passages_cut"] = int(sum(cuts))
    return {"report": rep, "rows": rows}


def document_bits(
    model: torch.nn.Module,
    tok: Any,
    texts: Sequence[str],
    block: int,
    device: torch.device | str,
    batch_size: int = 16,
    amp_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, int]:
    """Total bits of each text scored as one document (:data:`DOC_PROTOCOL`), and how many needed windows."""
    eot = tok.special_token_ids["<|endoftext|>"]
    pieces: list[tuple[list[int], int]] = []
    owner: list[int] = []
    windowed = 0
    for n, text in enumerate(texts):
        ids = tok.encode_ordinary(text)
        if not ids:
            raise ValueError(f"text {n} encodes to no tokens")
        segs = segments([eot, *ids], 1, block)
        windowed += len(segs) > 1
        pieces.extend(segs)
        owner.extend([n] * len(segs))
    logp = score_pieces(model, pieces, device, batch_size, amp_dtype)
    bits = np.zeros(len(texts), dtype=np.float64)
    np.add.at(bits, np.asarray(owner, dtype=np.int64), -logp / math.log(2.0))
    return bits, windowed
