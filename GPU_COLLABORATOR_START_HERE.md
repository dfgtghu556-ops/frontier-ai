# GPU collaborator — start here

Welcome! This page is for the person who runs frontier-ai's GPU work on their own computer.
You do **not** need to know machine learning to help: the code, the experiment plans and
one-command scripts are prepared for you; you set the machine up once, run the commands,
and the results come back through GitHub. Setup takes about an hour, most of it downloads.

*Last updated 2026-09-28 (branch `arena/01a0dc16-frontier-ai`, after EXP-033).*

## 1. What this project is — honestly

**Goal (the founder's north star):** build, from scratch, a language-model family that can
one day compete with the leading AI systems and become the best model in India, strong in
English and Indian languages. That is a long road; every step is measured and recorded.

**Where it stands today (verified in the repository):**

| Done | Evidence |
|---|---|
| A licensed research corpus of 13 languages, frozen and hash-pinned | D-035, `corpora/` |
| Our own tokenizer, **Frontier Tokenizer v1** (32,768 tokens), frozen | D-040, D-041, `tokenizers/frontier-tokenizer-v1/` |
| An evaluation harness and a protected held-out test suite | D-042, `evals/` |
| Architecture tests on small CPU models (EXP-032, EXP-033) | `EXPERIMENTS.md`, `evals/results/` |

**What does not exist yet:** no model has ever been trained on a GPU, and every model so far
is a tiny CPU research model (~5M parameters). That is why your GPU matters: **step 10 of the
roadmap is "bring up actual GPU training"**, and step 11 is small scaling experiments.
Nothing on your machine starts until the founder approves the step-10 plan.

## 2. How we work (please read — these rules protect the science)

1. **Branch:** all shared work is on **`arena/01a0dc16-frontier-ai`**. Never push to `main`,
   never force-push, never rewrite history.
2. **Experiments are pre-registered.** Each one has a spec (for example
   `configs/ablations/EXP-033.json`) that is committed *before* it runs. Never edit a spec, a
   script or its arguments to get a different result; a changed plan is a new experiment id.
3. **You run, the scripts record.** Each GPU task comes as a task file
   (`PC_TASK_*.md`) with **one command**. The script checks everything, runs, writes a
   report, and commits **only** its small result files under `evals/results/<EXP>/`.
4. **Never commit data or models.** Corpus text, token files and checkpoints stay on your disk
   (`.gitignore` already excludes `data/`, `out/`, `*.bin`, `*.pt`).
5. **The test suite `frontier-heldout-v1` is protected.** Never train on it.
6. **This repository is public.** Never commit passwords, tokens, or personal files.
7. **Your own ideas are welcome** — open a GitHub *Issue*, or a *Pull Request* from your own
   branch; the founder reviews it. Please don't push unreviewed changes to the shared branch.
8. **Stopping is always fine.** `Ctrl+C` stops a run; runs are restartable and skip
   finished work. The GPU will get hot and loud during runs; that is normal.

## 3. One-time setup

### Step 1 — GitHub access
Reading the repository needs no invitation (it is public). To **push results** you need to
be a collaborator:
1. Create a free account at <https://github.com> (if you do not have one) and send your
   **username** to the founder.
2. The founder opens the repository → **Settings → Collaborators → Add people** → your
   username.
3. Accept the invitation from the e-mail (or <https://github.com/notifications>).

### Step 2 — Install three programs (Windows)
1. **NVIDIA driver** — the latest *Game Ready* or *Studio* driver for your card from
   <https://www.nvidia.com/Download/index.aspx> (or the NVIDIA app). Then open **PowerShell**
   and run `nvidia-smi`: it must print your GPU name. Note the **"CUDA Version"** shown at the
   top right: that is the newest CUDA your driver supports.
   *You do NOT need to install the "CUDA Toolkit"; PyTorch brings its own CUDA runtime.*
2. **Git for Windows** — <https://git-scm.com/download/win> (default options are fine).
3. **Python 3.13** (3.12 also works) — <https://www.python.org/downloads/>. On the first
   installer screen tick **"Add python.exe to PATH"**.

Optional but handy: **VS Code** (<https://code.visualstudio.com>) — *Terminal → New Terminal*
gives you a PowerShell in the project folder.

*On Linux:* install the NVIDIA driver from your distribution, `git`, and Python 3.12/3.13
with `venv`; the commands below are the same with `.venv/bin/python` instead of
`.venv\Scripts\python.exe`.

### Step 3 — Get the code
In PowerShell, in a folder with at least 20 GB free (for example `D:\`):

```
git clone https://github.com/dfgtghu556-ops/frontier-ai.git
cd frontier-ai
git checkout arena/01a0dc16-frontier-ai
git config user.name "Your Name"
git config user.email "you@example.com"
```

### Step 4 — Python environment with **CUDA** PyTorch
Order matters: install the CUDA build of PyTorch **first**, then the project (otherwise pip
installs a CPU-only PyTorch).

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Now open <https://pytorch.org/get-started/locally/> and choose **Stable · Windows · Pip ·
Python ·** a **CUDA** version that is **not newer** than the "CUDA Version" from `nvidia-smi`.
The site shows a command like `pip3 install torch torchvision --index-url
https://download.pytorch.org/whl/cuXXX`. Run it through the project's Python, **torch only**:

```
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cuXXX
```

(replace `cuXXX` with the value the site shows; if no listed CUDA version fits, update the
NVIDIA driver first). Then install the project itself:

```
.venv\Scripts\python.exe -m pip install -e ".[dev,tokenizer]"
```

### Step 5 — Readiness report (read-only, about 1 minute)

```
.venv\Scripts\python.exe scripts\gpu_env_report.py
```

It checks the machine, PyTorch and the GPU, and runs a few seconds of tiny GPU tests (random
numbers only; nothing is saved, nothing is trained on real data). It ends with
**`VERDICT: READY for GPU work`** or **`NOT READY`** plus what to fix, and writes
`out\gpu_env\REPORT.txt`. **Send that file to the founder** (WhatsApp/e-mail is fine; it
contains no names, passwords or folder paths).

### Step 6 (optional) — Run the test suite

```
.venv\Scripts\python.exe -m pytest -q -o addopts=""
```

Everything should pass (one test is skipped on purpose); it takes a few minutes.

## 4. What happens after setup

> **Update 2026-09-30 (read this first).** Since this page was written the project moved on: a
> larger Hindi + 12-language training set now exists (EXP-036/037, packed with Frontier
> Tokenizer v2), and the first GPU bring-up (EXP-038) is being run on a free Kaggle T4 GPU.
> Your GPU is **not** part of EXP-038. The data table below describes the old v1 files; what
> your machine would run, and with which data, will be a separate plan written for your GPU
> after your `REPORT.txt` arrives, and nothing runs before the founder approves it. The known
> hardware so far (from the founder, not yet measured): NVIDIA GeForce RTX 3050 with 4 GB
> of GPU memory and a 12th-gen Intel i5. With 4 GB the plan will use small models only.


1. The founder shares your report in the project chat; the step-10 plan (what to run, how
   long, what it measures) is written for **your** GPU and approved by the founder.
2. **Data.** The research data is small (a few tens of MB) but deliberately not in git. The
   founder will share one zip from his computer; you unpack it into the repository folder,
   and a verification command (prepared together with the step-10 plan) checks every file
   against the hashes recorded in the repository:

   | path (git-ignored) | what | recorded size / hash |
   |---|---|---|
   | `out/exp_b/EXP-029/mark_aware-32768.bin` (+ `.meta.json`) | training tokens (Frontier Tokenizer v1) | 3,586,440 bytes, sha256 `fb43d4cd…` |
   | `corpora/frontier/v1/shards/` | FrontierCorpus v1 pilot text (train + held-out) | 5 shards, 10,451,264 bytes, sha256 per shard in `corpora/frontier/v1/manifest.json` |
   | `data/tokenizer/indic-tokenizer-v2/` | frozen tokenizer corpus (needed to rebuild the test suite) | pinned per source in `corpora/tokenizer/indic-tokenizer-v2/sources.json` |

3. **Each GPU task** arrives as a `PC_TASK_*.md` file with one command. You run
   `git pull --ff-only origin arena/01a0dc16-frontier-ai`, then that command, and leave the
   machine alone until it finishes. The script pushes its small results; the founder reviews
   them with the project's AI agent.

**An honest note on scale:** our current training data is about 1.8 million tokens. Your
GPU can train models far bigger than that data can properly feed, so step 10 is about making
GPU training correct, fast and measured — not about a big model yet. Growing the corpus is
the other half of the road to a serious model, and training a model that competes with the
largest systems will still need rented data-centre GPUs later.

## 5. Where to read more

| File | What it is |
|---|---|
| [README.md](README.md) | the code: quickstart, model, training loop |
| [ROADMAP.md](ROADMAP.md) | the 24-step plan and where we are |
| [EXPERIMENTS.md](EXPERIMENTS.md) | every experiment, pre-registration and result (append-only) |
| [DECISIONS.md](DECISIONS.md) | every decision and why (append-only) |
| [NEW_CHAT_START_HERE.md](NEW_CHAT_START_HERE.md) | handover log for AI agents |
| [MASTER_CONTEXT.md](MASTER_CONTEXT.md) | the founder's mission and working rules |

Questions or problems: send the founder the exact message you see (a screenshot is fine).
