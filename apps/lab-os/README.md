# Lab OS: read-only dashboard of this repository

A small web dashboard that shows where the frontier-ai project stands: the 24-step roadmap,
experiments, decisions, data, the frozen tokenizer, evaluation results, models and compute.

It **only reads the repository**. It has no backend, no database, no login and no write actions,
and it contains no demo or made-up data. Everything it shows comes from one generated file:

    apps/lab-os/src/data/lab_state.json

That file is written by `scripts/export_lab_state.py` (logic in `src/frontier_ai/lab_state.py`) from:

| Source | What the dashboard takes from it |
| --- | --- |
| `EXPERIMENTS.md` | every experiment, its last recorded status, its step and decisions |
| `DECISIONS.md` | every decision, its status, and the open questions table |
| `MASTER_CONTEXT.md` §37 | the 24 roadmap step titles |
| `lab/registry.json` | the only hand-maintained file: step status, current work, baseline, compute, key documents |
| `tokenizers/frontier-tokenizer-v1/FREEZE.json` | the frozen tokenizer |
| `corpora/frontier/v1/manifest.json` | FrontierCorpus v1 sizes, languages and cleaning stages |
| `corpora/frontier/v2/sangraha_slice1.json` | the pinned Sangraha slice (EXP-034) |
| `evals/suites/frontier-heldout-v1/SUITE.json` | the protected evaluation suite |
| `evals/results/*/*/report.json`, `evals/results/*/summary.json` | published results |

The exporter refuses to write a snapshot if `lab/registry.json` disagrees with the records: a step
title that differs from §37, evidence that does not exist, a step marked complete while an
experiment it cites is not complete or a decision it cites is not accepted, or a completed step
after an unfinished one. The snapshot has no timestamps, so the same repository always gives the
same file, and the header shows a digest of all the input files.

## Refresh the data

From the repository root:

    python scripts/export_lab_state.py           # rewrite the snapshot
    python scripts/export_lab_state.py --check   # exit 1 if it is out of date

`tests/test_lab_state.py` fails while the committed snapshot is out of date, so re-run the
exporter and commit the snapshot whenever EXPERIMENTS.md, DECISIONS.md, `lab/registry.json` or
the evaluation results change.

## Run it

Needs Node.js 22 or newer.

    cd apps/lab-os
    npm ci
    npm run dev        # http://localhost:8080 (or the port Vite prints)
    npm run build      # production build into .output/

Checks: `npx tsc --noEmit`, `npm run lint`.

## Origin

The UI shell (layout, theme, table and panel components) comes from the founder's
`frontier-labs-os` prototype (TanStack Start + React + Tailwind, generated with Lovable). Its mock
API and fictional sample data were removed; the pages were rewritten to read the snapshot.
The unused Lovable UI kit in `src/components/ui/` is kept as-is for future pages.
