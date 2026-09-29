"""Lab OS snapshot: one JSON file that shows the project exactly as the repository records it.

Why this exists
---------------
The founder's Lab OS dashboard (``apps/lab-os``) must never show invented numbers. This module
reads the repository's own records and writes a single deterministic snapshot the dashboard
renders. Nothing here is typed by hand except ``lab/registry.json`` (roadmap step status,
compute, key documents), and that file is cross-checked against the records (see
``validate_registry``).

Sources (all read-only):

* ``EXPERIMENTS.md`` / ``DECISIONS.md`` — every experiment and decision, with its recorded status.
  For experiments the **last** ``**Status:**`` line of a section counts, because the log is
  append-only: a later follow-up note updates an earlier status instead of editing it.
* ``MASTER_CONTEXT.md`` §37 — the 24-step roadmap titles.
* ``tokenizers/frontier-tokenizer-v1/FREEZE.json`` — the frozen tokenizer.
* ``corpora/frontier/v1/manifest.json`` — FrontierCorpus v1: sides, languages, pipeline stages.
* ``corpora/frontier/v2/sangraha_slice1.json`` — the pinned (not yet inspected) v2 slice.
* ``evals/suites/frontier-heldout-v1/SUITE.json`` — the protected evaluation suite.
* ``evals/results/*`` — harness reports (``report.json``) and ablation summaries (``summary.json``).

The snapshot has no timestamps: the same repository content always gives byte-identical output,
so a test can tell whether the committed snapshot is current. Its ``sources`` block lists the
SHA-256 of every input file.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

SCHEMA = "frontier-lab-state-v1"
REGISTRY_SCHEMA = "frontier-lab-registry-v1"
REGISTRY_PATH = "lab/registry.json"
DEFAULT_OUTPUT = "apps/lab-os/src/data/lab_state.json"
REPO_URL = "https://github.com/dfgtghu556-ops/frontier-ai"
BRANCH = "arena/01a0dc16-frontier-ai"

STEP_STATUSES = ("complete", "current", "next", "not_started")
EXP_CATEGORIES = ("complete", "in_progress", "planned", "abandoned", "other")

_EXP_HEAD = re.compile(r"^### (EXP-\d{3}) — (.+?)\s*$")
_DEC_HEAD = re.compile(r"^## (D-\d{3}) — (.+?)\s*$")
_STATUS = re.compile(r"\*\*Status:\*\*\s*(.+)$")
_DATE = re.compile(r"\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})")
_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_STEP = re.compile(r"§37 step (\d+)")
_D_REF = re.compile(r"\bD-\d{3}\b")
_EXP_REF = re.compile(r"\bEXP-\d{3}\b")
_NEW_FIELD = re.compile(r"^\*\*[^*]+:\*\*|^\*\*[^*]+\*\*$")  # "**Label:** ..." or a bold heading line


class LabStateError(RuntimeError):
    """The registry or a source record is inconsistent with the repository."""


# ------------------------------------------------------------------ helpers --
def _clean(text: str) -> str:
    """Markdown -> plain text for short labels (bold, code, links, emoji markers)."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "").replace("✅", "").replace("🔶", "")
    text = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"\1", text)  # *italic*
    return re.sub(r"\s+", " ", text).strip()


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:—-")
    return cut + " …"


def _field_paragraph(lines: list[str], labels: tuple[str, ...]) -> str:
    """Text of the first ``**Label:**`` field (with its continuation lines), or ''."""
    for i, line in enumerate(lines):
        stripped = re.sub(r"^\s*[-*]\s+", "", line).strip()
        for label in labels:
            marker = f"**{label}:**"
            if stripped.startswith(marker):
                parts = [stripped[len(marker) :]]
                for nxt in lines[i + 1 :]:
                    s = nxt.strip()
                    if not s or s.startswith(("- **", "* **", "|", "#", "```")) or _NEW_FIELD.match(s):
                        break
                    parts.append(s)
                return _clean(" ".join(parts))
    return ""


def _sections(text: str, head: re.Pattern[str]) -> list[tuple[str, str, int, list[str]]]:
    """(id, title, 1-based line, body lines) for every heading matching ``head``."""
    lines = text.splitlines()
    starts = [(i, m) for i, line in enumerate(lines) if (m := head.match(line))]
    out = []
    for n, (i, m) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        body = lines[i + 1 : end]
        # a level-2 heading (e.g. "## Open items") also ends a section
        for k, line in enumerate(body):
            if line.startswith("## ") and not head.match(line):
                body = body[:k]
                break
        out.append((m.group(1), m.group(2), i + 1, body))
    return out


def _category(status: str) -> str:
    s = status.lower()
    if s.startswith("complete"):
        return "complete"
    if s.startswith("in progress"):
        return "in_progress"
    if s.startswith(("pre-registered", "approved", "planned", "proposed")):
        return "planned"
    if s.startswith("abandoned"):
        return "abandoned"
    return "other"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _link(path: str, line: int | None = None) -> str:
    url = f"{REPO_URL}/blob/{BRANCH}/{path}"
    return f"{url}?plain=1#L{line}" if line else url


# ---------------------------------------------------------------- parsers --
def parse_experiments(text: str) -> list[dict[str, Any]]:
    out = []
    for exp_id, title, line, body in _sections(text, _EXP_HEAD):
        statuses = [m.group(1) for b in body if (m := _STATUS.search(b))]
        status = _clean(statuses[-1]) if statuses else ""
        status = status.split(" · ")[0]
        date = next((m.group(1) for b in body if (m := _DATE.search(b))), None)
        step = _STEP.search(title)
        clean_title = _clean(title)
        out.append(
            {
                "id": exp_id,
                "title": clean_title,
                "date": date,
                "status": _shorten(status, 160),
                "category": _category(status),
                "step": int(step.group(1)) if step else None,
                "summary": _shorten(
                    _field_paragraph(
                        body, ("Purpose (why this serves the mission)", "Purpose", "Objective", "Question")
                    ),
                    320,
                ),
                "decisions": sorted(set(_D_REF.findall("\n".join(body)))),
                "line": line,
                "url": _link("EXPERIMENTS.md", line),
            }
        )
    return out


def parse_decisions(text: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    decisions = []
    for dec_id, title, line, body in _sections(text, _DEC_HEAD):
        status_line = next((m.group(1) for b in body if (m := _STATUS.search(b))), "")
        status = _clean(status_line)
        date = next((m.group(1) for b in body if (m := _DATE.search(b))), None)
        if date is None:
            found = _ISO_DATE.search(status)
            date = found.group(1) if found else None
        s = status.lower()
        category = next(
            (c for c in ("accepted", "provisional", "superseded", "proposed") if s.startswith(c)), "other"
        )
        decisions.append(
            {
                "id": dec_id,
                "title": _clean(title),
                "date": date,
                "status": _shorten(status, 160),
                "category": category,
                "summary": _shorten(_field_paragraph(body, ("Decision",)), 320),
                "experiments": sorted(set(_EXP_REF.findall("\n".join(body)))),
                "line": line,
                "url": _link("DECISIONS.md", line),
            }
        )
    questions = []
    in_open = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_open = line.startswith("## Open items")
            continue
        if in_open and (m := re.match(r"^\|\s*(Q-\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)):
            questions.append(
                {"id": m.group(1), "question": _clean(m.group(2)), "deferred_to": _clean(m.group(3))}
            )
    return decisions, questions


def parse_roadmap_titles(master_context: str) -> list[str]:
    lines = master_context.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("## 37."))
    except StopIteration as exc:
        raise LabStateError("MASTER_CONTEXT.md has no '## 37.' section") from exc
    titles = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if m := re.match(r"^(\d+)\.\s+(.+?)\s*$", line):
            if int(m.group(1)) != len(titles) + 1:
                raise LabStateError(f"MASTER_CONTEXT §37 numbering breaks at {line!r}")
            titles.append(m.group(2))
    return titles


# --------------------------------------------------------------- registry --
def validate_registry(
    registry: dict[str, Any],
    root: Path,
    experiments: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    titles: list[str],
) -> list[str]:
    """Return problems (empty = the hand-maintained registry agrees with the repository)."""
    problems: list[str] = []
    if registry.get("schema") != REGISTRY_SCHEMA:
        problems.append(f"registry schema must be {REGISTRY_SCHEMA!r}")
    exp = {e["id"]: e for e in experiments}
    dec = {d["id"]: d for d in decisions}
    steps = registry.get("roadmap_steps", [])
    if [s.get("step") for s in steps] != list(range(1, len(titles) + 1)):
        problems.append(f"roadmap_steps must list steps 1..{len(titles)} in order")
    for s in steps:
        n = s.get("step")
        if isinstance(n, int) and 1 <= n <= len(titles) and s.get("title") != titles[n - 1]:
            problems.append(f"step {n}: title {s.get('title')!r} != MASTER_CONTEXT {titles[n - 1]!r}")
        if s.get("status") not in STEP_STATUSES:
            problems.append(f"step {n}: status must be one of {STEP_STATUSES}")
        for ref in s.get("evidence", []):
            if ref.startswith("EXP-"):
                if ref not in exp:
                    problems.append(f"step {n}: evidence {ref} is not in EXPERIMENTS.md")
                elif s.get("status") == "complete" and exp[ref]["category"] != "complete":
                    problems.append(f"step {n} is complete but {ref} is {exp[ref]['category']!r}")
            elif ref.startswith("D-"):
                if ref not in dec:
                    problems.append(f"step {n}: evidence {ref} is not in DECISIONS.md")
                elif s.get("status") == "complete" and dec[ref]["category"] != "accepted":
                    problems.append(f"step {n} is complete but {ref} is {dec[ref]['category']!r}")
            else:
                problems.append(f"step {n}: unknown evidence id {ref!r}")
    statuses = [s.get("status") for s in steps]
    first_open = next((i for i, st in enumerate(statuses) if st != "complete"), len(statuses))
    if "complete" in statuses[first_open:]:
        problems.append("a step is complete after an incomplete step (the §37 order is a dependency order)")
    for group in ("roadmap_steps", "current_work", "compute"):
        for item in registry.get(group, []):
            for f in item.get("evidence_files", []):
                if not (root / f).is_file():
                    problems.append(f"{group}: evidence file {f} does not exist")
            env = item.get("measured_environment_from")
            if env and not (root / env).is_file():
                problems.append(f"{group}: {env} does not exist")
            for ref in item.get("evidence", []) if group == "current_work" else []:
                if ref not in exp and ref not in dec:
                    problems.append(f"current_work: {ref} is not recorded")
    for d in registry.get("documents", []):
        if not (root / d["path"]).is_file():
            problems.append(f"documents: {d['path']} does not exist")
    return problems


# ---------------------------------------------------------------- readers --
def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokenizer(root: Path) -> dict[str, Any]:
    f = _load_json(root / "tokenizers/frontier-tokenizer-v1/FREEZE.json")
    a, lin = f["artifact"], f["lineage"]
    return {
        "id": f"{f['tokenizer_id']}-{f['tokenizer_version']}",
        "status": f["status"],
        "frozen_at": f["frozen_at"],
        "decision": f["decision"],
        "frozen_by": f["frozen_by"],
        "vocab_size": a["vocab_size"],
        "merges": a["merges"],
        "pretoken": a["pretoken"],
        "impl": a["impl"],
        "dir_sha256": a["dir_sha256"],
        "trained_chars": lin["trained_chars"],
        "training_corpus": lin["training_corpus"]["id"],
        "selected_by": lin["selected_by"],
        "url": _link("tokenizers/frontier-tokenizer-v1/FREEZE.json"),
    }


def _corpus_v1(root: Path) -> dict[str, Any]:
    m = _load_json(root / "corpora/frontier/v1/manifest.json")
    reg = m["identity"]["source_registry"]
    per_lang: dict[str, dict[str, int]] = {}
    for side in ("train", "held_out"):
        for lang, v in m["sides"][side]["per_language"].items():
            agg = per_lang.setdefault(
                lang, {"train_documents": 0, "train_chars": 0, "held_out_documents": 0, "held_out_chars": 0}
            )
            agg[f"{side}_documents"] += v["documents"]
            agg[f"{side}_chars"] += v["chars"]
    sources_per_lang: dict[str, int] = {}
    for s in m["identity"]["sources"]:
        sources_per_lang[s["language"]] = sources_per_lang.get(s["language"], 0) + 1
    for lang, n in sources_per_lang.items():
        per_lang.setdefault(lang, {})["sources"] = n
    return {
        "id": m["corpus"]["id"],
        "version": m["corpus"]["version"],
        "created_at": m["created_at"],
        "content_sha256": m["content_sha256"],
        "domains": m["corpus"]["domains"],
        "licenses": reg["licenses"],
        "sources": reg["sources"],
        "normalization": m["identity"]["normalization"],
        "sides": {
            side: {k: m["sides"][side][k] for k in ("documents", "chars", "bytes")}
            for side in ("train", "held_out")
        },
        "per_language": dict(sorted(per_lang.items())),
        "stages": [
            {
                "stage": s["stage"],
                "documents_in": s["stats"].get("documents_in"),
                "kept": s["kept"],
                "removed": s["removed"],
            }
            for s in m["stages"]
        ],
        "url": _link("corpora/frontier/v1/manifest.json"),
    }


def _inspection(root: Path) -> dict[str, Any] | None:
    """EXP-034's measurements per file (``evals/results/EXP-034/summary.json``), if published."""
    path = root / "evals/results/EXP-034/summary.json"
    if not path.is_file():
        return None
    r = _load_json(path)
    per_language = {}
    for f in r["files"]:
        gate, type_chars = f["script_gate"], f["type_chars"]
        per_language[f["language"]] = {
            "documents": f["documents"],
            "chars": f["chars"],
            "script_pass_share": gate["documents"].get("pass", 0) / f["documents"],
            "exact_duplicates": f["exact_duplicates_within_file"]["documents"],
            "suite_hits": f["suite"]["documents_hit"],
            "pdf_char_share": type_chars.get("pdf", 0) / sum(type_chars.values()),
            "estimated_tokens": f["tokens"]["estimated_file_tokens"],
            "token_sample_documents": f["tokens"]["sample_documents"],
            "sha256_verified": f["pinned_sha256_verified"],
        }
    return {
        "experiment": r["exp_id"],
        "complete": bool(r["complete"]) and r["max_docs_per_file"] is None,
        "suite_status": r["suite_status"],
        "finished_at": r["finished_at"],
        "per_language": dict(sorted(per_language.items())),
        "url": _link("evals/results/EXP-034/SUMMARY.txt"),
    }


def _sangraha(root: Path) -> dict[str, Any]:
    """The pinned slice plus EXP-034's measurements. The status is derived, not typed: it says
    whether the published inspection exists and is complete; what it means is in EXPERIMENTS.md."""
    p = _load_json(root / "corpora/frontier/v2/sangraha_slice1.json")
    inspection = _inspection(root)
    if inspection is None:
        status = "pinned (sha256 per file); not yet downloaded or inspected (EXP-034 pending)"
    elif inspection["complete"] and inspection["suite_status"].startswith("CHECKED"):
        status = "downloaded, all files match their pins, fully inspected (EXP-034); nothing filtered yet"
    else:
        status = "EXP-034 inspection published but partial; must be repeated before a build"
    return {
        "slice_id": p["slice_id"],
        "dataset": p["dataset"],
        "subset": p["subset"],
        "revision": p["revision"],
        "license_id": p["license_id"],
        "trust_level": p["trust_level"],
        "attribution": p["attribution"],
        "total_bytes": p["total_bytes"],
        "files": [{"language": f["language"], "size": f["size"], "sha256": f["sha256"]} for f in p["files"]],
        "status": status,
        "inspection": inspection,
        "url": _link("corpora/frontier/v2/sangraha_slice1.json"),
    }


def _suite(root: Path) -> dict[str, Any]:
    s = _load_json(root / "evals/suites/frontier-heldout-v1/SUITE.json")
    return {
        "id": s["suite_id"],
        "status": s["status"],
        "totals": s["totals"],
        "fingerprint": s["fingerprint"],
        "per_language": s["per_language"],
        "url": _link("evals/suites/frontier-heldout-v1/SUITE.json"),
    }


def _reports(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((root / "evals/results").glob("*/*/report.json")):
        r = _load_json(path)
        ov = r["results"]["overall"]
        out.append(
            {
                "experiment": path.parts[-3],
                "label": r["label"],
                "created_at": r["created_at"],
                "n_params": r["checkpoint"]["n_params"],
                "vocab_size": r["checkpoint"]["vocab_size"],
                "step": r["checkpoint"]["step"],
                "tokenizer": r["tokenizer"].get("id") or r["tokenizer"]["spec"],
                "suite": r["suite"]["id"],
                "bits_per_byte": ov["bits_per_byte"],
                "ci95": ov["bits_per_byte_ci95"],
                "documents": ov["documents"],
                "per_language": {
                    k: v["bits_per_byte"] for k, v in sorted(r["results"]["per_language"].items())
                },
                "contamination": r["contamination"]["status"],
                "environment": r["environment"],
                "url": _link(path.relative_to(root).as_posix()),
            }
        )
    return out


def _ablations(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((root / "evals/results").glob("*/summary.json")):
        s = _load_json(path)
        if "rows" not in s:
            continue
        rows = [
            {
                "group": r["group"],
                "arm": r.get("arm", r["group"]),
                "lr": r.get("lr"),
                "seeds": r["seeds"],
                "bpb": r["bpb"],
                "mean": r["mean"],
                "std": r["std"],
                "n_params": r["n_params"],
                "body_params": r["body_params"],
                "kv_cache_values_per_token": r.get("kv_cache_values_per_token"),
            }
            for r in s["rows"]
        ]
        out.append(
            {
                "experiment": s["exp_id"],
                "max_steps": s["max_steps"],
                "rows": rows,
                "summary_url": _link(path.with_name("SUMMARY.txt").relative_to(root).as_posix()),
            }
        )
    return out


def _models(reports: list[dict[str, Any]], ablations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Research models that were actually trained and graded (grouped over seeds)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in reports:
        name = re.sub(r"-seed-\d+$", "", r["label"])
        groups.setdefault((r["experiment"], name), []).append(r)
    models = []
    for (exp, name), rs in sorted(groups.items()):
        vals = [r["bits_per_byte"] for r in rs]
        models.append(
            {
                "id": f"{exp}/{name}",
                "experiment": exp,
                "name": name,
                "seeds": len(rs),
                "n_params": rs[0]["n_params"],
                "body_params": None,
                "vocab_size": rs[0]["vocab_size"],
                "steps": rs[0]["step"],
                "bpb_mean": statistics.fmean(vals),
                "bpb_std": statistics.stdev(vals) if len(vals) > 1 else None,
                "hardware": rs[0]["environment"].get("torch", ""),
            }
        )
    for a in ablations:
        for row in a["rows"]:
            models.append(
                {
                    "id": f"{a['experiment']}/{row['group']}",
                    "experiment": a["experiment"],
                    "name": row["group"],
                    "seeds": len(row["seeds"]),
                    "n_params": row["n_params"],
                    "body_params": row["body_params"],
                    "vocab_size": None,
                    "steps": a["max_steps"],
                    "bpb_mean": row["mean"],
                    "bpb_std": row["std"],
                    "hardware": "CPU",
                }
            )
    return models


# ------------------------------------------------------------------ build --
INPUTS = (
    "EXPERIMENTS.md",
    "DECISIONS.md",
    "MASTER_CONTEXT.md",
    REGISTRY_PATH,
    "tokenizers/frontier-tokenizer-v1/FREEZE.json",
    "corpora/frontier/v1/manifest.json",
    "corpora/frontier/v2/sangraha_slice1.json",
    "evals/suites/frontier-heldout-v1/SUITE.json",
)


def build_state(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    experiments = parse_experiments((root / "EXPERIMENTS.md").read_text(encoding="utf-8"))
    decisions, questions = parse_decisions((root / "DECISIONS.md").read_text(encoding="utf-8"))
    titles = parse_roadmap_titles((root / "MASTER_CONTEXT.md").read_text(encoding="utf-8"))
    registry = _load_json(root / REGISTRY_PATH)
    problems = validate_registry(registry, root, experiments, decisions, titles)
    if problems:
        raise LabStateError("lab/registry.json disagrees with the repository:\n  " + "\n  ".join(problems))

    reports = _reports(root)
    ablations = _ablations(root)
    compute = []
    for c in registry["compute"]:
        item = {k: v for k, v in c.items() if k != "measured_environment_from"}
        src = c.get("measured_environment_from")
        item["measured_environment"] = _load_json(root / src)["environment"] if src else None
        item["measured_environment_source"] = src
        compute.append(item)

    numbers = sorted(int(e["id"][4:]) for e in experiments)
    missing = [n for n in range(1, numbers[-1] + 1) if n not in numbers] if numbers else []

    result_files = sorted(
        p.relative_to(root).as_posix() for p in (root / "evals/results").glob("*/*/report.json")
    )
    result_files += sorted(
        p.relative_to(root).as_posix() for p in (root / "evals/results").glob("*/summary.json")
    )
    sources = {path: _sha256_file(root / path) for path in (*INPUTS, *result_files)}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode("utf-8")).hexdigest()

    models = _models(reports, ablations)
    base = registry.get("baseline_model")
    baseline = None
    if base:
        record = next((m for m in models if m["id"] == base["model"]), None)
        unknown = [ref for ref in base.get("evidence", []) if ref not in {d["id"] for d in decisions}]
        if record is None or unknown:
            raise LabStateError(
                f"lab/registry.json baseline_model: unknown model {base['model']!r} or decisions {unknown}"
            )
        baseline = {**base, "record": record}

    return {
        "schema": SCHEMA,
        "repository": {"url": REPO_URL, "branch": BRANCH},
        "source_digest": digest,
        "sources": sources,
        "roadmap": [
            {
                **s,
                "evidence_links": [
                    next((x["url"] for x in (*experiments, *decisions) if x["id"] == ref), None)
                    for ref in s["evidence"]
                ],
            }
            for s in registry["roadmap_steps"]
        ],
        "current_work": registry["current_work"],
        "experiments": experiments,
        "experiment_numbers_not_in_log": missing,
        "decisions": decisions,
        "open_questions": questions,
        "tokenizer": _tokenizer(root),
        "corpus_v1": _corpus_v1(root),
        "sangraha_slice1": _sangraha(root),
        "suite": _suite(root),
        "evaluation_reports": reports,
        "ablations": ablations,
        "models": models,
        "baseline_model": baseline,
        "compute": compute,
        "documents": [{**d, "url": _link(d["path"])} for d in registry["documents"]],
    }


def render(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
