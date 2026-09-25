# Run reports

The channel between the Arena agent and whoever runs the network steps on the operator's
machine ([docs/tokenizer_corpus_local_runner.md](../../../../docs/tokenizer_corpus_local_runner.md)):

* `JOB.md` — the current job, written by the Arena agent: exact commands, what to expect,
  what to save, and when to stop.
* `EXP-0NN-*.txt` — what a run produced (inspection reports, lock checks), saved by the
  scripts' own `--output` options and pushed as they came out. They are the evidence each
  review and lock decision is based on; they are never edited by hand.
