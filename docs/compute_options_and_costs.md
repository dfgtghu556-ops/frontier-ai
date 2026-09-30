# Compute options and cost estimates (2026-09-30)

Written for the founder, who asked on 2026-09-30: is there a free way to get a GPU, what would it
cost, and what level of model can we reach compared with ChatGPT? **This is information for
planning, not a plan.** No GPU is used without a step-10 plan the founder approves. The model sizes
below are **examples for estimating cost only**. Model sizes are chosen by scaling experiments
(step 11), not fixed in advance.

Prices and quotas change. Everything below was checked on the web on 2026-09-30. A number that
could not be checked is marked NOT VERIFIED.

## 1. Where we stand today (from the repository)

- **Data:** FrontierCorpus v2-slice1, packed for training (EXP-037, D-046). It has 2,783,830,088
  training tokens in 13 languages. The files (5.6 GB) are only on the founder's laptop.
- **Models so far:** tiny CPU research models of about 5.2 M parameters, with a transformer body
  of about 1 M. They were trained for 150 steps on about 1.8 M tokens (EXP-031 to EXP-033). Their
  job was to check the method. They are not useful assistants.

## 2. Free and cheap GPU options

| Option | What you get | Cost | Catch | Useful for |
|---|---|---|---|---|
| **Kaggle notebooks** | P100 or 2 × T4 (16 GB each), about 30 GPU-hours a week, sessions up to 12 h ([Ultralytics docs](https://docs.ultralytics.com/integrations/kaggle), [AIMultiple](https://aimultiple.com/free-cloud-gpu)) | Free, no card | Phone verification is needed to enable GPUs ([guide](https://aicreditmart.com/ai-credits-providers/kaggle-free-gpu-tpu-30-hours-week-access-guide-2026/)). Sessions can queue at busy times. The two T4s are separate 16 GB GPUs. | Step 10 bring-up and small step-11 scaling runs |
| **Google Colab (free)** | Usually a T4 ([LinkedIn comparison](https://www.linkedin.com/pulse/gpu-acceleration-showdown-kaggle-vs-google-colab-machine-jha-vlvpe)) | Free | No guaranteed GPU. It disconnects when idle. | Quick tests only |
| **Google TPU Research Cloud (TRC)** | Free Cloud TPUs for accepted researchers ([TRC](https://sites.research.google/trc/about/)) | TPUs free. The small VM and storage are paid ([TRC FAQ](https://sites.research.google/trc/faq/)). | You must apply and be accepted, and share the research publicly (our repo is public). Our PyTorch code would need porting to TPU. The grant length is NOT VERIFIED (older posts say 30 days). | Larger runs later, if accepted |
| **The GPU collaborator's machine** | Unknown until his `out\gpu_env\REPORT.txt` arrives | Free | We don't know the GPU model or its memory yet | Step 10 bring-up |
| **IndiaAI compute portal** (subsidised, not free) | H100-class GPUs, reportedly about ₹65–92 per GPU-hour after subsidy ([TechDodo](https://techdodo.in/articles/india-ai-impact-summit-2026-gpu-access-guide), [TechPillow](https://www.techpillow.co/blog/indiaai-mission-38000-gpus-shakti-cloud-compute-2026)) | Cheapest paid option | **Eligibility** ([end-user policy](https://indiaai.s3.ap-south-1.amazonaws.com/docs/end-user-policy-for-indiaai-compute-portal.pdf)): DPIIT-recognised startups that have revenue or funding; researchers (auto-approval at h-index ≥ 5 or 150 citations); students with an APAAR ID and an institute endorsement. An individual without these probably cannot use it today (NOT VERIFIED for our case). | Serious runs once eligible |
| **Rented cloud GPUs** | RTX 4090 about $0.34–0.74 per hour; A100 80 GB about $1.19–1.89; H100 about $1.99–3.99 (RunPod and Vast.ai, 2026) ([Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-runpod-vs-vastai-2026/), [Thunder Compute](https://www.thundercompute.com/blog/runpod-pricing-vs-thunder-compute), [aiofm](https://aiofm.info/en/guides/runpod-vs-vast-ai)) | Pay per hour | The cheapest hosts can be unreliable. Big clouds (AWS, GCP, Azure) charge much more. | Anything, if we pay |

## 3. Cost of one training run (estimate, NOT VERIFIED)

**Method.** Training compute ≈ 6 × parameters × tokens (Kaplan et al. 2020).

**Assumptions:**
- An H100 does 989 TFLOP/s peak (bf16). We assume 35% of that is actually used.
- 2 × T4 do 2 × 65 TFLOP/s peak (fp16). We assume 25% is used.
  **Measured (EXP-038, 2026-09-30, one T4, fp16 + `torch.compile`):** 22% MFU for a 32 M model
  (context 512, 69,127 tokens/s) and 26% for a 139 M model (context 1,024, 17,870 tokens/s).
  So the 25% assumption holds for one T4. Using both T4s at once needs multi-GPU code (step 13),
  which we don't have yet; with one T4, the 139 M model would need about 43.5 hours for 2.8 B
  tokens (2.8 B ÷ 17,870 tokens/s), so the Kaggle column below assumes two GPUs we can't use yet.
- Rent is $2.0–3.5 per H100-hour, at ₹88 per $ (the exchange rate is NOT VERIFIED). IndiaAI rate: ₹92 per hour.

These are final-run costs only. Real projects also spend compute on experiments and failed runs.
DeepSeek's published cost also leaves these out ([DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)).
Our rule of thumb is to multiply by 2–3; that factor is not measured.

| Example (illustration only) | Compute | H100 hours | Rented cost | IndiaAI rate | Kaggle (free, 2 × T4) |
|---|---|---|---|---|---|
| A. Step-10 bring-up: ~50 M params, 0.5 B tokens | 1.5e17 | ~0.1 | < ₹50 | ~₹11 | ~1 h (free) |
| B. Largest step-11 scaling run: ~150 M params, 2.8 B tokens | 2.5e18 | ~2 | ₹360–620 | ~₹190 | ~22 h (under 1 week of quota) |
| C. ~350 M params, one pass over v2-slice1 (2.8 B tokens) | 5.9e18 | ~5 | ₹830–1,450 | ~₹430 | ~50 h (about 2 weeks of quota) |
| D. ~1 B params, 20 B tokens (needs more data than we have) | 1.2e20 | ~96 | ₹17,000–30,000 | ~₹8,900 | ~1,000 h: not practical (and does not fit a 16 GB T4 without sharding) |
| E. ~7 B params, 2 T tokens | 8.4e22 | ~67,000 | ₹1.2–2.1 crore | ~₹62 lakh | impossible |
| F. Frontier class: DeepSeek-V3 (671 B MoE, 14.8 T tokens) | — | 2.788 M H800 hours | $5.576 M for the final run alone ([report](https://arxiv.org/abs/2412.19437)) | — | — |

**Indian reference point.** Sarvam AI trained Sarvam-30B and Sarvam-105B from scratch on IndiaAI
compute ([Fortune India](https://www.fortuneindia.com/business-news/sarvam-ai-launches-30b-and-105b-models-tailored-for-india-focused-deployment/130517)):
- mixture-of-experts models with about 10 B active parameters for the 105B model ([model card](https://huggingface.co/sarvamai/sarvam-105b));
- pre-trained on 12–16 T tokens ([summary](https://lilting.ch/en/articles/sarvam-105b-30b-open-source-llm));
- reported to lead on Indian-language benchmarks while trailing the frontier models on general
  reasoning indexes ([explainx](https://explainx.ai/blog/india-sovereign-ai-status-indiaai-mission-2026)).

To become "the best model in India" we would have to beat models of that scale. Our current data
(2.8 B tokens) is about 0.02% of their pre-training data.

## 4. What this means

- **Free is enough for steps 10–11.** That means proving the training code on a real GPU and
  running small scaling-law experiments that tell us which model size our data supports. Examples
  A–C fit Kaggle's free quota or the collaborator's GPU.
- **Everything from about 1 B parameters upward needs money or a grant.** It also needs much more
  clean data: Sangraha Verified alone is 64.3 B tokens, and we have processed a 2.8 B slice.
- **ChatGPT or Claude class is out of reach for a personal budget.** It needs tens of millions of
  dollars (examples E–F), large teams and a lot of human feedback data. The honest route toward
  the north star goes step by step:
  1. Build a small model that is measurably excellent at Indian languages per unit of compute.
  2. Publish the method and results.
  3. Use that record to qualify for IndiaAI compute, grants, TRC or partners.

  We should not claim anything we have not measured along the way.
- **Update (2026-10-01):** EXP-038 measured one Kaggle T4 (see the "Measured" note in §3): the
  25% MFU assumption is confirmed at 22–26% for fp16 + compile. With one T4 and no multi-GPU code,
  30 free hours a week train about 7.5 B tokens at 32 M parameters or about 1.9 B tokens at
  139 M parameters (measured tokens/s × 108,000 seconds). Larger models were not measured.
