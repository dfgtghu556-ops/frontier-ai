"""Sangraha Verified slice (D-044, EXP-034): pins, verified download, reader, inspection."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from frontier_ai.corpus.decontaminate import SuiteGuard, guard_from_heldout_shards
from frontier_ai.corpus.registry import SCRIPT_PIPELINE_NAMES
from frontier_ai.corpus.sangraha import (
    PinnedFile,
    SliceError,
    check_free_space,
    download_verified,
    load_slice_pins,
)
from frontier_ai.corpus.slice_inspect import inspect_rows, render_text_report
from frontier_ai.evaluation.suite import build_suite, write_suite

REPO = Path(__file__).resolve().parents[1]
PINS = REPO / "corpora" / "frontier" / "v2" / "sangraha_slice1.json"
V1_LANGUAGES = {"as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"}


# ------------------------------------------------------------------- pins --
def test_committed_pins_are_complete_and_pinned():
    header, files = load_slice_pins(PINS)
    assert header["license_id"] == "CC-BY-4.0"
    assert header["trust_level"] == "B"
    assert {f.language for f in files} == V1_LANGUAGES
    assert len(files) == 13
    for f in files:
        assert f.script in set(SCRIPT_PIPELINE_NAMES.values())
        assert f.url == ("https://huggingface.co/datasets/ai4bharat/sangraha/resolve/"
                         f"{header['revision']}/verified/{f.sangraha_code}/data-0.parquet")
    assert header["total_bytes"] == 5_106_130_219


def _write_pins(tmp_path: Path, mutate) -> Path:
    data = json.loads(PINS.read_text(encoding="utf-8"))
    mutate(data)
    p = tmp_path / "pins.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.__setitem__("total_bytes", 1), "total_bytes"),
        (lambda d: d["files"][0].__setitem__("url", d["files"][0]["url"].replace(d["revision"], "main")),
         "not pinned"),
        (lambda d: d["files"][1].__setitem__("source_id", d["files"][0]["source_id"]), "duplicate"),
        (lambda d: d.__setitem__("schema", "other"), "schema"),
    ],
)
def test_tampered_pins_are_refused(tmp_path, mutate, message):
    with pytest.raises(SliceError, match=message):
        load_slice_pins(_write_pins(tmp_path, mutate))


# --------------------------------------------------------------- download --
PAYLOAD = bytes(range(256)) * 4000  # 1,024,000 bytes


class _Server:
    """Local HTTP server with switchable behaviour (Range support, truncation, corruption)."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.honor_range = True
        self.truncate_next = 0  # serve only this many bytes on the next request, then drop
        self.requests: list[str | None] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep test output quiet
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                outer.requests.append(rng)
                body, status = outer.payload, 200
                if rng and outer.honor_range:
                    start = int(rng.split("=")[1].split("-")[0])
                    body, status = outer.payload[start:], 206
                send = body
                if outer.truncate_next:
                    send = body[: outer.truncate_next]
                    outer.truncate_next = 0
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(send)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/verified/hin/data-0.parquet"


def _pin(url: str, payload: bytes = PAYLOAD) -> PinnedFile:
    return PinnedFile(source_id="sangraha-verified-hin-data-0", language="hi", sangraha_code="hin",
                      script="devanagari", path="verified/hin/data-0.parquet", url=url,
                      size=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _quiet(_msg: str) -> None:
    pass


def test_download_verifies_and_then_skips(tmp_path):
    with _Server(PAYLOAD) as srv:
        pf = _pin(srv.url())
        assert download_verified(pf, tmp_path, log=_quiet, backoff_seconds=0) == "downloaded"
        assert pf.local_path(tmp_path).read_bytes() == PAYLOAD
        assert download_verified(pf, tmp_path, log=_quiet, backoff_seconds=0) == "verified"
        assert len(srv.requests) == 1  # second call downloaded nothing


def test_download_resumes_after_a_dropped_connection(tmp_path):
    with _Server(PAYLOAD) as srv:
        srv.truncate_next = 300_000
        pf = _pin(srv.url())
        assert download_verified(pf, tmp_path, log=_quiet, backoff_seconds=0) == "downloaded"
        assert pf.local_path(tmp_path).read_bytes() == PAYLOAD
        assert srv.requests[0] is None
        assert srv.requests[-1] is not None and srv.requests[-1].startswith("bytes=")


def test_download_restarts_when_server_ignores_range(tmp_path):
    with _Server(PAYLOAD) as srv:
        srv.honor_range = False
        pf = _pin(srv.url())
        part = pf.local_path(tmp_path).with_name("data-0.parquet.part")
        part.parent.mkdir(parents=True)
        part.write_bytes(b"\xff" * 1000)  # wrong bytes: must not be spliced in front
        assert download_verified(pf, tmp_path, log=_quiet, backoff_seconds=0) == "downloaded"
        assert pf.local_path(tmp_path).read_bytes() == PAYLOAD


def test_download_with_wrong_hash_is_set_aside_not_used(tmp_path):
    with _Server(PAYLOAD[:-1] + b"\x00") as srv:  # server serves corrupted bytes
        pf = _pin(srv.url())
        with pytest.raises(SliceError, match="does not match its pin"):
            download_verified(pf, tmp_path, log=_quiet, backoff_seconds=0)
        final_dir = pf.local_path(tmp_path).parent
        assert not pf.local_path(tmp_path).exists()
        assert any(".rejected-mismatch-" in p.name for p in final_dir.iterdir())


def test_free_space_check_refuses_when_margin_is_impossible(tmp_path):
    pf = _pin("http://example.invalid/verified/hin/data-0.parquet")
    with pytest.raises(SliceError, match="not enough disk space"):
        check_free_space((pf,), tmp_path, margin_bytes=10**18)
    needed, _free = check_free_space((pf,), tmp_path, margin_bytes=0)
    assert needed == len(PAYLOAD)


# ------------------------------------------------------------- held-out --
class _Doc:
    def __init__(self, doc_id, language, text):
        self.doc_id, self.language, self.source_id, self.text = doc_id, language, "suite-src", text


SUITE_TEXTS = [
    ("s-0", "hi", " ".join(f"शब्द{i}" for i in range(20))),
    ("s-1", "en", "a short protected line"),
]


def _suite():
    return build_suite([_Doc(*t) for t in SUITE_TEXTS], "test-suite", {"corpus": "t"})


def test_guard_from_heldout_shards_accepts_exact_shard_even_with_crlf(tmp_path):
    shard = tmp_path / "heldout-000000.txt"
    shard.write_bytes("\r\n".join(t for _, _, t in reversed(SUITE_TEXTS)).encode("utf-8"))
    guard = guard_from_heldout_shards([shard], _suite())
    assert guard.suite_documents == 2
    assert guard.check(SUITE_TEXTS[1][2]).suite_doc_id == "s-1"


@pytest.mark.parametrize("lines, message", [
    ([SUITE_TEXTS[0][2]], "missing 1"),
    ([SUITE_TEXTS[0][2], SUITE_TEXTS[1][2], "an extra line"], "does not match any suite document"),
])
def test_guard_from_heldout_shards_refuses_wrong_shard(tmp_path, lines, message):
    shard = tmp_path / "heldout-000000.txt"
    shard.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        guard_from_heldout_shards([shard], _suite())


# ------------------------------------------------------------ inspection --
class _CharTokenizer:
    vocab_size = 10

    def encode(self, text):
        return list(range(len(text) // 2))


HINDI = "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं।"
ROWS = [
    ("a1", "web", HINDI),
    ("a2", "web", HINDI),  # exact duplicate
    ("a3", "ocr", "Lorem ipsum dolor sit amet, this is Latin script text."),  # wrong script
    ("a4", "web", "12345 !!!"),  # no letters
    ("a5", "sent", "  "),  # empty after normalization
    ("a6", "web", "पहला वाक्य।\nदूसरा वाक्य, फ़ोन 98765 43210"),
    ("a7", "web", SUITE_TEXTS[0][2] + " और कुछ"),  # touches the protected suite
]


def test_inspect_rows_measures_without_filtering():
    guard = SuiteGuard.from_texts([(i, t) for i, _, t in SUITE_TEXTS])
    r = inspect_rows(ROWS, source_id="src", language="hi", script="devanagari", guard=guard,
                     tokenizer=_CharTokenizer(), token_every=1)
    assert r["documents"] == 7
    assert r["documents_empty_after_normalize"] == 1
    assert r["type_documents"] == {"ocr": 1, "sent": 1, "web": 5}
    assert r["exact_duplicates_within_file"]["documents"] == 1
    assert r["script_gate"]["documents"] == {"no_letters": 1, "pass": 4, "script_mismatch": 1}
    assert r["script_gate"]["mismatch_top_script"] == {"latin": 1}
    assert r["documents_with_newlines"] == 1
    assert r["suite"]["documents_hit"] == 1
    assert r["suite"]["hits_by_reason"] == {"suite_ngram": 1}
    assert r["tokens"]["sample_documents"] == 6
    assert r["tokens"]["estimated_file_tokens"] > 0
    snippet = r["samples"]["pass"][-1]["text"]
    assert "98765" not in snippet and "[masked]" in snippet and " / " in snippet


def test_inspect_rows_respects_max_docs_and_reports_unchecked_suite():
    r = inspect_rows(ROWS, source_id="src", language="hi", script="devanagari", guard=None, max_docs=3)
    assert r["documents"] == 3
    assert r["suite"] == {"checked": False}
    assert r["tokens"] == {"measured": False}


# --------------------------------------------------------- parquet + CLI --
def _parquet(path: Path, rows) -> bytes:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"doc_id": [r[0] for r in rows], "type": [r[1] for r in rows],
                      "text": [r[2] for r in rows]})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path), row_group_size=3)
    return path.read_bytes()


def test_iter_documents_streams_with_provenance_ids(tmp_path):
    from frontier_ai.corpus.sangraha import iter_documents

    data = _parquet(tmp_path / "root" / "verified/hin/data-0.parquet", ROWS)
    pf = _pin("http://x/verified/hin/data-0.parquet", data)
    docs = list(iter_documents(pf, tmp_path / "root", batch_size=2))
    assert [d.doc_id for d in docs][:2] == ["sangraha-verified-hin-data-0-a1", "sangraha-verified-hin-data-0-a2"]
    assert len(docs) == len(ROWS)
    assert all(d.language == "hi" and d.source_id == pf.source_id for d in docs)


def test_inspect_cli_end_to_end(tmp_path):
    import importlib.util

    rev = "0123456789abcdef0123456789abcdef01234567"
    root = tmp_path / "data"
    data = _parquet(root / rev[:12] / "verified/hin/data-0.parquet", ROWS)
    pins = {
        "schema": "frontier-v2-source-pins-v1", "slice_id": "test-slice", "dataset": "t/sangraha",
        "revision": rev, "license_id": "CC-BY-4.0", "attribution": "test attribution",
        "total_bytes": len(data),
        "files": [{"source_id": "sangraha-verified-hin-data-0", "language": "hi", "sangraha_code": "hin",
                   "script": "devanagari", "path": "verified/hin/data-0.parquet",
                   "url": f"https://h/{rev}/verified/hin/data-0.parquet", "size": len(data),
                   "sha256": hashlib.sha256(data).hexdigest()}],
    }
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")
    suite_path = tmp_path / "SUITE.json"
    write_suite(suite_path, _suite())
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    (heldout / "heldout-000000.txt").write_text("\n".join(t for _, _, t in SUITE_TEXTS), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("inspect_cli", REPO / "scripts" / "inspect_sangraha_slice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    code = mod.main(["--pins", str(pins_path), "--root", str(root), "--suite", str(suite_path),
                     "--heldout-dir", str(heldout), "--out", str(out), "--no-tokens"])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert summary["suite_status"].startswith("CHECKED")
    f = summary["files"][0]
    assert f["pinned_sha256_verified"] and f["parquet_rows_declared"] == len(ROWS)
    assert f["suite"]["documents_hit"] == 1
    text = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "test attribution" in text and "hi " in text

    # a missing shard is reported as NOT CHECKED, never as a silent zero
    code = mod.main(["--pins", str(pins_path), "--root", str(root), "--suite", str(suite_path),
                     "--heldout-dir", str(tmp_path / "nowhere"), "--out", str(tmp_path / "out2"), "--no-tokens"])
    assert code == 0
    s2 = json.loads((tmp_path / "out2" / "summary.json").read_text(encoding="utf-8"))
    assert s2["suite_status"].startswith("NOT CHECKED")
    assert s2["files"][0]["suite"] == {"checked": False}
    assert "n/a" in render_text_report(s2)


def test_fast_script_profile_equals_per_character_reference():
    import random
    import unicodedata

    from frontier_ai.corpus.langid import _script_of, script_profile, top_script

    def reference(text):
        profile = {}
        for ch in text:
            if ch.isspace():
                continue
            script = _script_of(ch) if unicodedata.category(ch)[0] == "L" else "common"
            profile[script or "unknown"] = profile.get(script or "unknown", 0) + 1
        return profile

    rng = random.Random(7)
    alphabet = "abcXYZ अआकि্বাংলা ਪੰਜਾਬੀ اردو ಕನ್ನಡ 123 ,.!\n\t\u200b\u00a0Ωж漢"
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        fast, ref = script_profile(text), reference(text)
        assert fast == ref and list(fast) == list(ref)  # same counts AND same key order
        assert top_script(fast) == top_script(ref)


def test_data_night_script_static_safety():
    """The PC runs scripts/run_data_night.ps1 unattended: same static checks as the EXP-033 runner."""
    import re

    raw = (REPO / "scripts" / "run_data_night.ps1").read_bytes()
    text = raw.decode("ascii")  # Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert '$branch = "arena/01a0dc16-frontier-ai"' in code
    assert "--force" not in code and " -f " not in code and "reset --hard" not in code
    assert "Remove-Item" not in code and "git clean" not in code and "git checkout" not in code
    for line in code.splitlines():
        if "Stop-Night" in line or "Add-Report" in line:
            continue
        if re.search(r"\bgit (status|add|commit|push|reset|diff|rev-parse)\b|\$python -", line):
            assert "Invoke-Logged" in line, line
            assert '`"' not in line, line  # no embedded quotes through cmd.exe (PS 5.1 mangles them)
    assert '& cmd.exe /d /c "$cmdline 2>&1"' in code
    assert "git push origin $branch" in code and code.count('Invoke-Logged "git push') == 1
    assert "git commit -q -F $msgFile" in code
    assert 'git add $resultsPrefix' in code and '$resultsPrefix = "evals/results/$Exp/"' in code
    # only the small reports are published: the publisher copies summary.json / SUMMARY.txt
    assert "publish_eval_results.py --exp-id $Exp --src $outDir" in code
    assert '$outDir = "out\\data\\$Exp"' in code
    # EXP-035 reuses the runner: -Task only switches the step-4 script and the texts
    assert '[ValidateSet("inspect", "calibrate")][string]$Task = "inspect"' in code
    assert 'Invoke-Logged "$python -u $stepScript --exp-id $Exp --pins $pins --out $outDir"' in code
    assert '$stepScript = "scripts\\calibrate_sangraha_slice.py"' in code
    # the final report line names the experiment that ran (EXP-035's report said "EXP-034 done")
    result_lines = [ln for ln in code.splitlines() if "RESULT: COMPLETE" in ln]
    assert len(result_lines) == 1 and "Tell the Arena chat: $Exp done" in result_lines[0]
    assert not re.search(r"EXP-0\d\d done", code)
    wrapper = (REPO / "scripts" / "run_calibration_night.ps1").read_bytes().decode("ascii")
    wcode = [ln for ln in wrapper.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert wcode == ['& "$PSScriptRoot\\run_data_night.ps1" -Exp "EXP-035" -Task "calibrate"', "exit $LASTEXITCODE"]
