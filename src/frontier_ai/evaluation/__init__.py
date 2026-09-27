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
