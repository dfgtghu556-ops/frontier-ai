"""Tests for the fetch retry policy and for scripts/inspect_corpus_sources.py.

The retry exists because a live build (EXP-009) lost Alice to
``IncompleteRead(88816 bytes read, 85495 more expected)``: gutenberg.org dropped the
connection mid-body. A retry must fix that without ever returning a partial body, and
without turning "no route to host" into a long wait.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import re
import socket
import urllib.error
from pathlib import Path

import pytest

from frontier_ai.data import corpora
from frontier_ai.data.corpora import FetchError, fetch_text


class _Response:
    def __init__(self, body: bytes | None = None, error: BaseException | None = None) -> None:
        self._body = body
        self._error = error

    def read(self) -> bytes:
        if self._error is not None:
            raise self._error
        assert self._body is not None
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _scripted(monkeypatch, outcomes: list):
    """urlopen stand-in: each call consumes the next outcome (exception or body)."""
    calls: list[str] = []

    def _urlopen(request, timeout=30.0):
        calls.append(request.full_url)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException) and not isinstance(outcome, http.client.IncompleteRead):
            raise outcome
        if isinstance(outcome, http.client.IncompleteRead):
            return _Response(error=outcome)  # the failure happens while reading the body
        return _Response(body=outcome)

    monkeypatch.setattr(corpora.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(corpora.time, "sleep", lambda _s: None)
    return calls


def test_fetch_retries_a_body_cut_off_mid_transfer(monkeypatch) -> None:
    calls = _scripted(monkeypatch, [http.client.IncompleteRead(b"x" * 10, 20), "पूरा पाठ".encode()])
    assert fetch_text("https://example.invalid/a.txt") == "पूरा पाठ"
    assert len(calls) == 2


def test_fetch_retries_server_busy_but_not_client_errors(monkeypatch) -> None:
    busy = urllib.error.HTTPError("https://example.invalid/b", 503, "busy", {}, None)
    calls = _scripted(monkeypatch, [busy, b"ok"])
    assert fetch_text("https://example.invalid/b") == "ok"
    assert len(calls) == 2

    missing = urllib.error.HTTPError("https://example.invalid/c", 404, "not found", {}, None)
    calls = _scripted(monkeypatch, [missing, b"never"])
    with pytest.raises(FetchError, match="404"):
        fetch_text("https://example.invalid/c")
    assert len(calls) == 1


def test_fetch_does_not_retry_when_there_is_no_route(monkeypatch) -> None:
    """Timeouts, DNS and TLS failures mean no route: retrying would only slow offline runs."""
    for error in (
        urllib.error.URLError(socket.timeout("timed out")),
        urllib.error.URLError(socket.gaierror("Name or service not known")),
        urllib.error.URLError(OSError("TLS/SSL connection has been closed (EOF)")),
        TimeoutError("timed out"),
    ):
        calls = _scripted(monkeypatch, [error, b"never"])
        with pytest.raises(FetchError):
            fetch_text("https://example.invalid/d")
        assert len(calls) == 1, error


def test_fetch_gives_up_after_the_last_attempt_and_never_returns_a_partial_body(monkeypatch) -> None:
    cut = [http.client.IncompleteRead(b"partial", 100) for _ in range(3)]
    calls = _scripted(monkeypatch, cut)
    with pytest.raises(FetchError, match="after 3 attempts: IncompleteRead"):
        fetch_text("https://example.invalid/e")
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# scripts/inspect_corpus_sources.py
# ---------------------------------------------------------------------------
def _load_inspector():
    path = Path(__file__).resolve().parents[1] / "scripts" / "inspect_corpus_sources.py"
    spec = importlib.util.spec_from_file_location("inspect_corpus_sources", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CLEAN_HI = "\n".join(
    ["१", "गाँव में सुबह हुई और लोग खेतों की ओर चल पड़े।", "शाम को सब घर लौटे और खाना खाया।"]
)
DIRTY_HI = "\n".join(
    [
        "पीछे गो-दान (1936) आगे 38655",
        "<span class=\"pagenum\">गाँव</span> में &#160; सुबह\u200b हुई।",
        ".mw-parser-output .wst-center{text-align:center}",
        "{{gap}} आगे का पाठ \ufffd",
    ]
)
ENGLISH = "Alice was beginning to get very tired of sitting by her sister on the bank."


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build(tmp_path: Path) -> Path:
    out = tmp_path / "build"
    (out / "sources").mkdir(parents=True)
    texts = {"hi-clean": CLEAN_HI, "hi-dirty": DIRTY_HI, "en-book": ENGLISH}
    rows = []
    for source_id, text in texts.items():
        (out / "sources" / f"{source_id}.txt").write_text(text, encoding="utf-8")
        rows.append({"source_id": source_id, "language": source_id[:2], "verification_status": "verified",
                     "sha256": _sha(text), "error": ""})
    (out / "sources" / "hi-clean.provenance.json").write_text(json.dumps({"content_check": {
        "page_quality": {"pages": 2, "first_page": "पृष्ठ:W.djvu/११", "last_page": "पृष्ठ:W.djvu/१२",
                         "by_level": {"3": 1, "4": 1}},
        "artifacts_removed": {"U+2060 WORD JOINER": 2, "broken {{gap}} template -> removed": 3,
                              "link to missing template साँचा:GaP -> removed": 1}}}, ensure_ascii=False),
        encoding="utf-8")
    # a file left over from an earlier run for a source whose fetch failed this time
    (out / "sources" / "en-stale.txt").write_text("text from yesterday", encoding="utf-8")
    rows.append({"source_id": "en-stale", "language": "en", "verification_status": "fetch_failed",
                 "sha256": None, "error": "IncompleteRead"})
    (out / "acquisition.json").write_text(json.dumps({"sources": rows}), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps({"languages": [
        {"language": "hi", "examples": 7, "duplicate_documents": 0},
        {"language": "en", "examples": 1, "duplicate_documents": 0},
        {"language": "bn", "examples": 0, "duplicate_documents": 0}]}), encoding="utf-8")
    return out


def test_analyze_text_flags_everything_that_should_not_survive_cleaning() -> None:
    module = _load_inspector()
    dirty = module.analyze_text(DIRTY_HI, "hi")
    flags = " ".join(dirty["flags"])
    for expected in ("html=", "wiki=", "entity=", "css=", "replacement_char=", "invisible=ZWSP:1",
                     "ascii_digit_runs="):
        assert expected in flags, (expected, flags)
    clean = module.analyze_text(CLEAN_HI, "hi")
    assert clean["flags"] == [] and clean["dominant_script"] == "DEVANAGARI"
    assert module.analyze_text(ENGLISH, "en")["flags"] == []
    # the wrong script for the language is a flag (e.g. an English page in a Hindi slot)
    assert "script=LATIN (expected DEVANAGARI)" in module.analyze_text(ENGLISH, "hi")["flags"]
    # ZWJ/ZWNJ are orthographic in Indic scripts and never flagged
    assert module.analyze_text("क्\u200dष और र्\u200cय", "hi")["flags"] == []


def test_analyze_text_shows_where_each_problem_is() -> None:
    module = _load_inspector()
    dirty = module.analyze_text(DIRTY_HI, "hi")
    places = dirty["examples"]
    assert "«{{»gap}} आगे का पाठ" in places["wiki"][0]  # «…» marks the spot, with its context
    assert "सुबह«ZWSP» हुई" in places["invisible"][0]  # an invisible character is named
    assert "«38655»" in places["ascii_digit_runs"][1]
    assert "«�»" in places["replacement_char"][0]
    assert all(len(found) <= module.EXAMPLES for found in places.values())
    # letters of another script are shown (information), counted per script
    assert dirty["other_script_letters"]["LATIN"] > 0 and places["other_script"]
    clean = module.analyze_text(CLEAN_HI, "hi")
    assert clean["examples"] == {} and clean["other_script_letters"] == {}
    # an English word in a Hindi text is shown but is not a flag
    mixed = module.analyze_text("होरी ने कहा, Mr. Khanna आज आएँगे और सब ठीक हो जाएगा।", "hi")
    assert mixed["flags"] == [] and "«Mr»" in mixed["examples"]["other_script"][0]


def test_inspection_report_reads_a_build_and_ignores_stale_files(tmp_path: Path, capsys) -> None:
    module = _load_inspector()
    out = _build(tmp_path)
    code = module.main(["--out", str(out), "--output", str(tmp_path / "report.txt")])
    printed = capsys.readouterr().out
    assert code == 1  # a flagged source makes the exit code non-zero
    assert "flagged sources: hi-dirty" in printed
    assert "where to look" in printed and "«{{»gap}}" in printed
    where = printed.split("where to look", 1)[1].split("repairs by the cleaner", 1)[0]
    assert "hi-dirty" in where and "hi-clean" not in where and "en-book" not in where
    assert "other-script letters (LATIN 61) — information, not a flag — 3 of 13 places shown" in where
    # repairs are listed from the provenance; routine invisible characters are not
    repairs = printed.split("repairs by the cleaner", 1)[1].split("samples (", 1)[0]
    assert ("hi-clean: broken {{gap}} template -> removed (3); "
            "link to missing template साँचा:GaP -> removed (1)") in repairs
    assert "WORD JOINER" not in repairs
    assert "११-१२ q3:1,q4:1" in printed
    assert "गाँव में सुबह हुई" in printed  # the sample lines a human must read
    assert "en-stale" in printed and "no text this run" in printed
    assert "text from yesterday" not in printed  # a stale file is never presented as this run's text
    assert "identical documents per language" in printed and "bn=" not in printed
    assert (tmp_path / "report.txt").read_text(encoding="utf-8").startswith("[inspect]")

    report = module.inspect_build(out)
    stale = next(row for row in report["sources"] if row["source_id"] == "en-stale")
    assert stale["analysis"] is None


def test_inspection_report_shows_every_slots_coverage_and_when_the_build_ran(
    tmp_path: Path, capsys
) -> None:
    module = _load_inspector()
    out = _build(tmp_path)
    (out / "corpus.json").write_text(json.dumps({"built_at": "2026-09-24T19:00:20Z"}), encoding="utf-8")
    acquisition = json.loads((out / "acquisition.json").read_text(encoding="utf-8"))
    acquisition["summary"] = {"pinned": 3}
    (out / "acquisition.json").write_text(json.dumps(acquisition), encoding="utf-8")
    (out / "coverage.json").write_text(json.dumps({
        "targets": {"sentences_per_language": 500, "characters_per_language": 200000},
        "languages": [
            {"language": "hi", "status": "EVALUATED", "reason": "", "actual_sentences": 5329,
             "actual_chars": 763964},
            {"language": "en", "status": "INSUFFICIENT", "actual_sentences": 1, "actual_chars": 76,
             "reason": "verified data is below target: 1 examples / 76 characters vs 500 examples "
                       "/ 200,000 characters"},
            {"language": "ta", "status": "NOT_EVALUATED", "reason": "no source declared",
             "actual_sentences": 0, "actual_chars": 0},
        ]}), encoding="utf-8")
    module.main(["--out", str(out)])
    printed = capsys.readouterr().out
    first = printed.splitlines()[0]
    assert "4 sources, 3 verified, 3 pinned in the manifest, built 2026-09-24T19:00:20Z" in first
    coverage = printed.split("coverage per language slot", 1)[1].split("flagged sources", 1)[0]
    assert coverage.startswith(" (targets: 500 documents and 200,000 characters):")
    assert re.search(r"hi\s+EVALUATED\s+5,329 documents\s+763,964 characters\n", coverage)
    assert re.search(r"en\s+INSUFFICIENT\s+1 documents\s+76 characters\s+— verified data is below", coverage)
    assert re.search(r"ta\s+NOT_EVALUATED\s+0 documents\s+0 characters\s+— no source declared", coverage)
    report = module.inspect_build(out)
    assert [slot["status"] for slot in report["coverage"]["languages"]] == [
        "EVALUATED", "INSUFFICIENT", "NOT_EVALUATED"]
    # an older build without coverage.json / corpus.json still inspects, without the section
    (out / "coverage.json").unlink()
    (out / "corpus.json").unlink()
    module.main(["--out", str(out)])
    printed = capsys.readouterr().out
    assert "coverage per language slot" not in printed and ", built " not in printed


def test_inspection_report_refuses_to_run_without_a_build(tmp_path: Path, capsys) -> None:
    module = _load_inspector()
    assert module.main(["--out", str(tmp_path / "nothing")]) == 2
    assert "run a --fetch build first" in capsys.readouterr().err
