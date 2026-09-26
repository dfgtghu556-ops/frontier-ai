# FRONTIER AI — MASTER PROJECT CONTEXT, MISSION & ROADMAP

> **About this file:**
> * **Source:** the founder's message of 2026-09-26, saved so that no agent depends on a chat
>   conversation (§40).
> * **Changes:** formatting only. Lists and arrow chains are joined onto lines, and one sentence
>   in §10 was lightly reworded; nothing else was reworded.
> * **Authority:** it is the founder's statement of mission, working rules, roadmap and answer
>   format. On the repository's *state*, the repository wins (§2). The verified current state is
>   in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md), section "CURRENT POSITION".

You are joining an ongoing AI research/engineering project called Frontier AI.

Your job is NOT to start from scratch, repeat generic LLM tutorials, or blindly implement
whatever seems interesting.

You must first understand the project's existing architecture, decisions, repository state,
methodology, constraints, and long-term goal. Then help the founder move the project forward
one verified step at a time.

## 1. The mission

Frontier AI is an independent India-based AI/LLM research project.

The long-term goal is to build our own capable foundation-model family that can eventually
compete technically with leading general-purpose AI systems.

The project should eventually have:

- our own foundation-model weights
- our own tokenizer
- our own data pipeline
- our own training pipeline
- our own evaluation infrastructure
- strong Indian-language capabilities
- strong English/global capabilities
- mathematics
- coding
- reasoning
- instruction following
- safety/alignment
- efficient inference
- multimodal capabilities
- speech
- vision
- tool use
- agents
- a model ecosystem/specialist-model layer
- eventually frontier-scale research

We are NOT building a wrapper around OpenAI, Claude, Gemini, or another proprietary
foundation model.

External AI models may be used as:

- research assistants
- coding assistants
- architecture reviewers
- data-analysis helpers
- temporary teacher models where legally and technically appropriate

But the final foundation model must be independently trained with our own model weights.

Open-source infrastructure is acceptable, such as:

- PyTorch
- CUDA
- Linux
- Hugging Face libraries
- standard open-source training/inference libraries

The distinction is: **using open-source infrastructure ≠ using somebody else's
foundation-model weights.**

## 2. Core principle

The project follows this principle:

> Repository truth > current command/test results > git history/experiment records >
> documentation > old conversations > assumptions.

Never claim something is complete merely because an old conversation said it was complete.

Always verify the current repository.

Never fabricate:

- benchmarks
- training results
- GPU performance
- dataset sizes
- licenses
- hashes
- model capabilities
- citations
- experiment results

If something is unknown, explicitly say: **NOT YET VERIFIED**

## 3. How we work

Use this research methodology:

1. Understand the problem.
2. Inspect the current implementation.
3. Decompose the problem.
4. Identify requirements.
5. Identify constraints.
6. Identify known facts.
7. Identify unknowns.
8. Generate alternatives.
9. Research and verify.
10. Attack the assumptions.
11. Compare tradeoffs.
12. Choose the smallest useful experiment.
13. Define success criteria.
14. Implement.
15. Test.
16. Measure.
17. Record results.
18. Update documentation.
19. Decide the next step.

Do NOT jump directly from "I think X is better" to "Let's implement X."

Instead ask:

- What evidence supports X?
- What could prove X wrong?
- What is the cheapest experiment that can distinguish X from the alternatives?
- What would we measure?
- What is the expected failure mode?

## 4. User's working style

The founder is technically a beginner.

Therefore:

- explain complicated engineering in simple language
- do not assume advanced ML knowledge
- give concrete commands
- prefer copy-pasteable instructions
- avoid unnecessary theory unless it helps the immediate decision
- do not dump 30 simultaneous tasks on the founder
- identify the ONE immediate next action clearly
- explain what the command does
- explain what success should look like
- explain what to send back if it fails

The founder wants AI agents to write most of the implementation code.

Therefore the preferred workflow is:

Founder → research/architecture advisor → decision → implementation prompt → coding agent →
tests/results → review → documentation

Do not make the founder manually implement large amounts of code when an AI coding agent can
safely do it.

## 5. Current repository

GitHub repository: <https://github.com/dfgtghu556-ops/frontier-ai>

The repository is the source of truth.

Current public main branch has substantially advanced beyond the earliest project state.

The repository contains:

- configs
- corpora
- docs
- scripts
- src/frontier_ai
- tests

Important project documents include:

- PROJECT_CONTEXT.md
- ROADMAP.md
- DECISIONS.md
- EXPERIMENTS.md
- README.md

There is currently documentation drift.

In particular, PROJECT_CONTEXT.md contains an older repository-state description that no
longer accurately represents the current main branch.

This must be fixed.

Do not blindly trust stale repository-state sections.

## 6. Current project status

### P001 — Base GPT / Training Engine

STATUS: COMPLETE

We built a CPU-first, GPU-ready decoder-only Transformer training stack.

It includes:

- Transformer decoder
- RMSNorm
- SwiGLU
- RoPE
- GQA
- scaled dot-product attention
- tied embeddings
- KV-cache generation
- AdamW
- warmup/cosine scheduling
- gradient accumulation
- gradient clipping
- checkpointing
- checkpoint resume
- mixed precision paths
- device selection
- JSONL metrics
- training/evaluation/generation scripts

A historical CPU smoke model was approximately 139k parameters.

The smoke test successfully demonstrated: train → evaluate → checkpoint → resume → generate

This proves the basic stack works.

It does NOT prove:

- GPU performance
- large-scale training
- good language quality
- good tokenizer quality
- distributed training
- scaling behavior
- production stability

Do not confuse the smoke test with a successful foundation-model run.

## 7. P002 — Tokenizer research

STATUS: Framework COMPLETE. Production tokenizer NOT YET SELECTED.

Repository contains:

- tokenizer interface
- tokenizer registry
- dependency-free byte-level BPE
- mark-aware pretokenization
- optional Hugging Face tokenizer baseline
- adapters
- deterministic Indic probe corpus
- tokenizer evaluator/comparator
- provenance/artifact support

The research specifically considers:

- Indic scripts
- combining marks
- multiple Indian languages
- Hinglish
- Romanized Indian languages
- code-mixed text
- punctuation
- numbers
- URLs
- code

A small earlier probe produced useful measurements, but it was NOT large enough to select the
final tokenizer.

Therefore: NO PRODUCTION TOKENIZER HAS BEEN SELECTED YET.

The next tokenizer decision must use the newly acquired real corpus.

## 8. P003 — Experiment / reproducibility infrastructure

STATUS: COMPLETE

The repository contains experiment infrastructure for:

- configuration
- seeds
- environment capture
- git information
- hashing
- experiment records
- experiment runner
- sweeps
- structured results
- provenance
- deterministic experiments

Experiments should record:

- hypothesis
- dataset
- configuration
- seed
- variables
- controls
- metrics
- results
- interpretation
- limitations
- decision

Never conduct important experiments without recording enough information to reproduce them.

## 9. P004A — Corpus infrastructure

STATUS: COMPLETE

Infrastructure exists for:

- corpus manifests
- provenance
- license allowlists
- source validation
- HTTPS-only acquisition
- attribution
- smoke corpus handling
- local file ingestion
- local directory ingestion
- identical-document split protection
- hash-based identity
- stratified sampling
- truncation metadata
- source verification

Licensing and provenance are first-class requirements.

Never assume: "Publicly accessible = free to train on."

Never assume: "Hash exists = license proven."

License evidence must be tracked separately.

## 10. P004B — Corpus acquisition

STATUS: THE FOUNDER SAYS THIS PART IS NOW COMPLETE.

The tokenizer-research corpus acquisition machinery included:

- preflight
- fetch
- pinning
- SHA-256 verification
- content-shape checks
- index-page refusal
- pinned-hash mismatch refusal
- acquisition reports
- source/license evidence tracking
- local fallback handling

The founder now considers the acquisition stage completed.

However, distinguish tokenizer-corpus acquisition from the full foundation-model pretraining
corpus. The second is NOT automatically complete.

The next project is therefore to transform the acquired/source inventory into a robust,
reproducible Frontier foundation training corpus.

## 11. Current position

The project is approximately here:

- P001 Base training infrastructure → COMPLETE
- P002 Tokenizer framework → COMPLETE, tokenizer decision pending
- P003 Experiment infrastructure → COMPLETE
- P004A Corpus infrastructure → COMPLETE
- P004B Tokenizer-corpus acquisition → COMPLETE according to current project state

NOW: **STAGE 3 — FOUNDATION DATA PIPELINE**

Then: foundation dataset → tokenizer selection → evaluation harness → architecture experiments
→ GPU validation → scaling experiments → first serious base model → distributed training →
large-scale pretraining → post-training → reasoning → safety → inference → multimodal → agents
→ specialist ecosystem → frontier research

## 12. Next major project — FrontierCorpus v1

Do NOT immediately start training a giant model.

First create: **FrontierCorpus v1**

The dataset pipeline should include the following.

### Source registry

Track:

- source
- URL
- language
- domain
- license
- license evidence
- retrieval date
- SHA-256
- document count
- character count
- token count
- provenance
- permitted use
- limitations

### Languages

Prioritize Indian languages including:

- Hindi
- Bengali
- Marathi
- Telugu
- Tamil
- Gujarati
- Kannada
- Malayalam
- Punjabi
- Odia
- Assamese
- Urdu
- Sanskrit
- Nepali

Add other languages where high-quality data is available.

Do not treat Indian languages only as translation targets. They should be represented as
genuine source languages.

Also study:

- Romanized Indian languages
- Hinglish
- code-mixing
- cross-script content

### Domains

Build a diverse mixture:

- literature
- education
- science
- mathematics
- reference material
- government/public-domain material where permitted
- culture
- history
- news where licensing permits
- technical material
- programming/code
- multilingual material
- high-quality English/global knowledge

Do not blindly optimize for raw token count. Quality and diversity matter.

## 13. Data cleaning pipeline

Implement:

raw source → encoding validation → language identification → normalization → HTML/boilerplate
removal → quality filtering → PII filtering → exact deduplication → near-duplicate detection →
domain classification → source weighting → train/validation/test isolation → sharding →
packing → manifest

The pipeline must be reproducible.

## 14. Deduplication

At minimum:

**Exact**

- document SHA-256
- normalized text hash

**Near duplicate.** Investigate:

- n-gram fingerprints
- MinHash
- locality-sensitive hashing
- similarity thresholds

Also detect cross-source duplication.

Do not allow the same document to leak into training and evaluation under slightly different
copies.

## 15. Data quality

Build measurable quality signals:

- language confidence
- document length
- repetition
- boilerplate ratio
- symbol/noise ratio
- duplicate probability
- source quality
- domain
- PII risk
- license confidence

Do not create one mysterious quality score without knowing what contributes to it.

Keep intermediate statistics so we can understand what the pipeline removed.

## 16. Data mixing

Do not arbitrarily declare: "30% Hindi, 20% English, 10% code..."

Instead create multiple reproducible recipes. For example:

- FrontierCorpus-v1-recipe-A
- FrontierCorpus-v1-recipe-B
- FrontierCorpus-v1-recipe-C

Then train controlled small models and measure:

- Indian-language performance
- English performance
- general knowledge
- math
- coding
- reasoning
- validation loss
- token efficiency

Data mixing must become an empirical research problem.

## 17. Storage

Eventually use:

- versioned shards
- deterministic shuffle
- streaming
- resumable loading
- document boundaries
- training manifests
- checksums

A dataset version must identify exactly what training saw.

For example, FrontierCorpus-v1.2 + recipe-B + tokenizer-v1 + seed-42 must be reconstructable.

## 18. Tokenizer v1

After FrontierCorpus is available, test tokenizer candidates.

Potential families:

- BPE
- Unigram

Potential vocabulary sizes should be experimentally evaluated rather than assumed.

Test:

- normalization
- whitespace
- combining marks
- Indic scripts
- Romanized text
- Hinglish
- code
- URLs
- numbers
- emoji
- mixed scripts

Measure:

- tokens/character
- bytes/token
- tokens/word
- per-language fertility
- round-trip accuracy
- tokenizer speed
- training throughput
- inference throughput
- memory impact

But token efficiency alone is NOT enough.

Train the same small model with competing tokenizers. Then compare actual model quality.

Select **Frontier Tokenizer v1**, with a formal decision record.

## 19. Evaluation must be built before large training

Build an evaluation harness. It should eventually cover the following.

**Language modeling**

- validation loss
- bits/token
- bits/byte
- per-language loss
- per-domain loss

**Indian-language capabilities**

- QA
- generation
- summarization
- translation
- reasoning
- transliteration
- code-mixing
- culturally contextual tasks

**General capabilities**

- knowledge
- reading comprehension
- mathematics
- science
- coding
- reasoning
- instruction following
- long context
- factuality

**Human evaluation**, especially native-speaker evaluation for Indian languages.

Create protected evaluation datasets that are NEVER accidentally inserted into pretraining.

## 20. Architecture research

The existing Transformer is the controlled baseline.

Do NOT rewrite everything immediately.

Use the existing architecture to establish a baseline. Then experimentally investigate:

- attention variants
- GQA
- positional methods
- FFN variants
- context length
- dense vs MoE
- multi-token prediction
- inference-efficient architectures

Architecture research must use controlled ablations.

Do not copy another company's architecture simply because it is popular.

## 21. GPU validation

The existing project has GPU-ready paths, but CPU development does not prove GPU correctness.

Therefore perform a real GPU bring-up. Verify:

- CUDA
- BF16
- FP16
- AMP
- gradient checkpointing
- torch.compile
- attention kernels
- memory usage
- throughput
- checkpointing
- restart/resume

Measure:

- tokens/sec
- step time
- peak VRAM
- GPU utilization
- training efficiency

Never invent GPU measurements.

## 22. Scaling experiments

Before deciding on a huge model, train progressively larger controlled models. Record:

- parameters
- training tokens
- validation loss
- Indic-language performance
- English performance
- math
- coding
- reasoning
- training FLOPs
- GPU hours
- inference cost

Use the results to determine the appropriate model/data/compute relationship.

Do NOT start with an arbitrary target such as 7B, 70B, or 100B.

The model size must be justified by:

- available compute
- data quality
- token budget
- scaling experiments
- training economics
- inference economics
- measured performance

## 23. First serious foundation model

Once the data, tokenizer, architecture, GPU stack and evaluation are trustworthy, train
**Frontier-Base**.

This must represent: our initialization + our tokenizer + our data + our architecture + our
training code + our weights

Every serious training run must record:

- code SHA
- config
- dataset version
- dataset hash
- tokenizer version/hash
- seed
- hardware
- optimizer
- learning rate
- batch size
- context length
- checkpoint schedule
- evaluation schedule

## 24. Distributed training

Start with DDP.

Then investigate FSDP / ZeRO-style sharding.

Only later, if necessary:

- tensor parallelism
- pipeline parallelism
- expert parallelism
- hybrid parallelism

Measure:

- scaling efficiency
- communication overhead
- loss equivalence
- checkpoint correctness
- failure recovery

Never assume multi-GPU correctness just because a training process starts.

## 25. Post-training

Once Frontier-Base is strong: Frontier-Base → supervised fine-tuning → Frontier-Instruct

SFT data should include:

- Indian languages
- English
- coding
- mathematics
- reasoning
- useful conversations
- high-quality instruction data
- appropriately licensed synthetic data

Do not allow cheap synthetic English data to overwhelm genuine Indian-language capability.

## 26. Preference optimization

After SFT, investigate:

- preference pairs
- DPO
- related preference-learning methods
- reward modeling if justified

Measure:

- helpfulness
- instruction following
- factuality
- safety
- Indian-language quality
- over-refusal
- under-refusal

Choose the method based on controlled evidence.

## 27. Reasoning

Create a dedicated reasoning track. Start with verifiable tasks:

- mathematics
- programming
- logic
- structured reasoning
- multi-step problems

Potential pipeline: high-quality reasoning data → filtering → rejection sampling →
verifier-based selection → preference optimization → RL-style experiments

Potential methods can include GRPO/PPO-family approaches and newer alternatives.

Do not select an algorithm just because it is currently fashionable.

## 28. Safety

Safety is not only a final release step. Begin at the data layer.

Eventually test:

- jailbreaks
- prompt injection
- PII
- toxicity
- dangerous instructions
- cyber safety
- self-harm safety
- bias
- multilingual safety
- cultural sensitivity
- Indian-language safety bypasses

A model that is safe in English but unsafe in Hindi is not sufficiently evaluated.

## 29. Inference

Optimize the trained models for actual use:

- KV cache
- batching
- streaming
- quantization
- speculative decoding
- continuous batching
- GPU inference
- CPU inference where practical

Eventually build a model family rather than one giant model. Potential families:

- Frontier-Small
- Frontier-Medium
- Frontier-Large
- Frontier-Reasoning

These are placeholders, not predetermined sizes.

## 30. Tools and agents

Do this AFTER reliable instruction following and reasoning. Build:

- function calling
- retrieval
- search
- code execution
- tool use
- planning
- verification
- memory
- multi-step workflows

Do not let flashy agent demos distract from weak underlying model capability.

## 31. Multimodal

These come later.

**Vision**

- image understanding
- OCR
- Indian-script OCR
- documents
- tables
- charts
- handwriting

**Speech**

- speech recognition
- speech generation
- Indian accents
- code-switching
- regional languages
- low-resource languages

Eventually move toward a unified multimodal Frontier model family.

## 32. Specialist Indian model ecosystem

The founder has an important long-term idea: Frontier should be able to work with specialist
Indian AI systems.

Do NOT assume that means merging every model's weights into one checkpoint. Instead
investigate:

- API routing
- local model adapters
- distillation
- specialist experts
- fine-tuning
- model composition
- retrieval
- tool calling

All of this is subject to licenses and technical compatibility.

Architecture: Frontier → capability router → language specialist → speech specialist → vision
specialist → reasoning specialist → external/local models where appropriate

This should become a future capability ecosystem.

## 33. Frontier AI Lab / Lovable

There is also a separate product called **Frontier AI Lab**. It was envisioned/built using
Lovable as a frontend/control center.

It is NOT the foundation model. It is NOT the training engine. It is the future control
plane.

Conceptually: Lovable Frontier AI Lab → Backend/API → training/data/evaluation systems →
Frontier models

Potential dashboard sections:

- Overview
- Research
- Models
- Experiments
- Training
- Datasets
- Evaluation
- Compute
- Infrastructure
- Model Registry
- Documentation
- Security
- Settings

Do NOT connect the Lovable UI directly to shell commands or PyTorch. Eventually create a
proper backend/API.

The API should expose controlled objects such as:

- Dataset
- Experiment
- Training Run
- Checkpoint
- Evaluation
- Model
- GPU Job

Integration should happen after the backend has stable interfaces.

Lovable can be developed in parallel but must NOT block the model roadmap.

## 34. Four parallel tracks

Think of Frontier as four connected tracks:

- **Model track:** Transformer → training → scaling → base model → instruction model →
  reasoning → multimodal
- **Data/research track:** corpus → tokenizer → evaluation → data quality → architecture
  research → efficiency research
- **Product track:** Lovable Lab → backend → APIs → model registry → deployment → user
  products
- **Research ecosystem track:** specialist models → routing → tools → agents →
  external/local capability integrations

The tracks can progress independently.

Do not let the product track force premature model decisions.

## 35. What we should NOT do

Do NOT:

- immediately train a giant model
- choose 70B/100B without evidence
- copy another model architecture blindly
- use undocumented internet scraping
- assume public data is legally usable
- mix evaluation data into training
- optimize only for English benchmarks
- optimize only for token compression
- claim GPU performance without running GPU tests
- claim "best Indian LLM" without measured evidence
- build a huge multi-agent framework before the base model works
- make Lovable the training engine
- replace working infrastructure without evidence
- delete existing experiments
- fabricate missing results

## 36. Definition of "best"

Never treat "best Indian LLM" as a marketing statement. Make it measurable.

Eventually compare Frontier against relevant models on:

- Indian-language understanding
- Indian-language generation
- translation
- transliteration
- code-mixing
- reasoning
- mathematics
- coding
- factuality
- instruction following
- safety
- long context
- inference efficiency
- training efficiency

Report:

- benchmark
- model version
- date
- evaluation population
- prompt protocol
- metrics
- uncertainty/limitations
- contamination considerations

Do not give an overall subjective ranking.

The project should let the evidence demonstrate where Frontier is strong and where it is
weak.

## 37. Immediate roadmap

The immediate sequence is:

1. Synchronize repository documentation with actual GitHub state.
2. Freeze and verify the newly acquired corpus.
3. Build the full FoundationCorpus pipeline.
4. Produce FrontierCorpus v1.
5. Build the production tokenizer sweep.
6. Run tokenizer-vs-tokenizer small-model experiments.
7. Select Frontier Tokenizer v1.
8. Build the evaluation harness.
9. Run architecture ablations.
10. Bring up actual GPU training.
11. Run scaling experiments.
12. Train the first serious Frontier base model.
13. Validate distributed training.
14. Scale pretraining based on evidence.
15. Evaluate extensively.
16. SFT.
17. Preference optimization.
18. Reasoning.
19. Safety.
20. Inference optimization.
21. Multimodal.
22. Tools/agents.
23. Specialist ecosystem.
24. Frontier research and larger-scale models.

## 38. Current immediate action

The next implementation project should be **"FrontierCorpus v1 — Foundation Data Pipeline"**.

Before writing code, inspect the current repository and identify:

1. What P004B actually produced.
2. Which sources are verified.
3. Which manifests exist.
4. Which provenance records exist.
5. Which scripts already exist.
6. What cleaning functionality already exists.
7. What is missing.
8. What tests already exist.
9. What documentation is stale.

Then propose the smallest next implementation step.

DO NOT rewrite existing infrastructure unnecessarily.

DO NOT create duplicate systems.

Extend the existing architecture.

## 39. Required response format when working on this project

When I ask you what to do next, answer in this structure:

- **CURRENT STATE:** what is actually complete, verified from the repository.
- **GAP:** what is missing.
- **DECISION:** what should be done next and why.
- **IMPLEMENTATION:** exactly what should be built.
- **VERIFICATION:** exactly which tests/measurements prove it works.
- **DOCUMENTATION:** which repository files must be updated.
- **NEXT COMMAND:** the smallest useful command or implementation prompt.

Do not give me ten unrelated tasks when one task is sufficient.

## 40. Most important rule

The objective is NOT "Build a Transformer."

The objective is: **build an independent, reproducible, scientifically measured Indian-focused
foundation-model research organization.**

The model is only one component.

The durable advantage should come from: DATA + TOKENIZER + TRAINING + EVALUATION + RESEARCH +
EFFICIENCY + SAFETY + PRODUCT + SPECIALIST ECOSYSTEM

Every major decision must be measurable.

Every major result must be reproducible.

Every important failure must be documented.

Every AI agent joining the project must be able to understand the project from the repository
without depending on a previous ChatGPT conversation.

That is the standard for Frontier AI.
