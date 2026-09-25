### EXP-018 — P004B: fetch all 55 tokenizer corpus sources (indic-tokenizer/v2)

- **Status:** complete
- **Date:** 2026-09-25
- **Objective:** Download and verify all declared sources for indic-tokenizer/v2, generating baseline hashes and coverage report
- **Hypothesis:** All 55 sources will verify successfully (licensed, proofread, accessible), establishing the baseline for locking
- **Baseline:** EXP-017 (Gujarati, Malayalam, Odia, Assamese sources locked)

**Configuration**
- Script: `scripts/build_tokenizer_corpus.py`
- Flags: `--fetch --exp-id EXP-018 --out data/tokenizer/indic-tokenizer-v2`
- Data sources: 55 declared sources from corpora/tokenizer/indic-tokenizer-v2/sources.json
- Hardware: Local machine (operator's computer)
- Seed: Not applicable (data acquisition)
- Deterministic: yes

**Metrics**
- Sources downloaded: 55/55
- Sources verified: 55/55
- Exit code: 0 (no pinned sources changed)
- Documents acquired: 28,451 train / 3,188 held out
- Evaluation status: 11 language slots EVALUATED (≥500 docs, ≥200k chars each)
- Inspection report: `corpora/tokenizer/indic-tokenizer-v2/reports/EXP-018-inspection.txt`

**Results:**
- First fetch of all 55 sources completed successfully
- All sources verified with proper licensing evidence
- Old project copy moved: `frontier-ai/` → `..\frontier-ai-old-copy`
- 4 sources flagged for genuine years in text (ml, kn, ta x2) — these are correct text content, not errors
- Cleaner removed wiki typos: 18 broken {{gap}} templates, missing template links, literal <poem> tag, stray }}
- Coverage shows 11 language slots now EVALUATED: en, hi, bn, gu, ml, or, as, pa, kn, te, ta
- 3 slots remain NOT_EVALUATED: mr, ur, hi-en (awaiting source declaration)

**Fresh-clone re-verification:** Not applicable (data acquisition experiment)

**Next action:** Lock all verified sources with cryptographic hashes (EXP-019), then record experiments and update status documentation.


### EXP-019 — P004B: lock all 55 tokenizer corpus sources (indic-tokenizer/v2)

- **Status:** complete
- **Date:** 2026-09-25
- **Objective:** Write cryptographic hashes for all verified sources to manifest, establishing the locked baseline
- **Hypothesis:** All 55 sources will lock successfully with hashes matching their verified content from EXP-018
- **Baseline:** EXP-018 (fetch of all 55 sources)

**Configuration**
- Script: `scripts/build_tokenizer_corpus.py`
- Flags: `--fetch --pin --exp-id EXP-019 --out data/tokenizer/indic-tokenizer-v2`
- Data sources: 55 declared sources from corpora/tokenizer/indic-tokenizer-v2/sources.json
- Hardware: Local machine (operator's computer)
- Seed: Not applicable (data acquisition)
- Deterministic: yes

**Metrics**
- Sources downloaded: 55/55
- Sources verified: 55/55
- Sources pinned: 55/55
- Exit code: 0 (no pinned sources changed from EXP-018 baseline)
- All hashes written: sha256, verified=true, retrieved_at updated
- Evaluation status unchanged: 11 language slots EVALUATED, 3 NOT_EVALUATED
- Hash prefixes match EXP-018 inspection report output

**Results:**
- All 55 sources locked with cryptographic hashes
- Manifest updated: only sha256, verified, and retrieved_at fields changed
- Each hash begins with prefix reported in EXP-018 inspection
- No text changes detected (exit code 0 confirms no pinned sources altered)
- Licensing compliance maintained: all sources properly attributed and verified
- Text integrity verified: cleaner repairs documented in inspection report

**Fresh-clone re-verification:** Not applicable (data acquisition experiment)

**Next action:** Record experiments in EXPERIMENTS.md, update status documentation (NEW_CHAT_START_HERE.md), then begin research for Marathi and Urdu sources as outlined in handover section 10.