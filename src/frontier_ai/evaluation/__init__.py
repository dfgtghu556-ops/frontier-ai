"""Evaluation harness v1 (MASTER_CONTEXT §37 step 8, ROADMAP Stage 6 — intrinsic part).

One command (``scripts/eval_report.py``) turns any checkpoint into a versioned report
card over a *protected* evaluation suite: every held-out token scored exactly once,
bits-per-byte / bits-per-character overall and per language / script / source, bootstrap
confidence intervals, a data-identity check, a contamination check and full provenance.
``scripts/eval_compare.py`` compares report groups with paired (same-document) bootstrap
intervals.

Deliberately NOT here yet (ROADMAP Stage 6 downstream / MASTER_CONTEXT Phase C):
downstream benchmarks, generation-quality tasks and human evaluation — meaningless for
the current few-million-parameter research models.
"""

HARNESS_VERSION = "1.0.0"


def make_console_safe() -> None:
    """Never crash on printing: replace characters the console encoding cannot show.

    Windows consoles and pipes default to a legacy code page (e.g. cp1252); printing a
    character outside it raises ``UnicodeEncodeError`` *after* the work is done. Files are
    always written as UTF-8 and are unaffected. (EXP-031: U+2212 crashed eval_compare.)
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
