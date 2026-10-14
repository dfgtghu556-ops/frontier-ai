# Free compute credits: step-by-step guide for the founder

Written 2026-10-15 after the founder chose "apply for startup credits" over a second Kaggle account.
Programme details change often; everything below was read from public pages on 2026-10-15 and is
**NOT VERIFIED** until you see it on your own screen. If a screen differs from this guide, follow the
screen and tell the chat what it said.

## Why credits, and not a second Kaggle account

- Kaggle's Terms of Use (version of 22 June 2025, section 06): you "may not have, control, or operate under
  more than one active Kaggle account"; Kaggle "may take action without notice, including banning your
  user account". A ban of `someshwar0112` would lose our private datasets (the 22 GB of slices) and the
  EXP-047 checkpoint chain.
- Google Colab forbids "using multiple accounts to work around access or resource usage restrictions".
- Credits are free, allowed, and much larger than any free tier.

## What the credits could buy (rough, NOT VERIFIED)

EXP-047 needs about 6 × 507.5 M parameters × 8.16 billion tokens ≈ 2.5 × 10^19 operations. On one Kaggle
T4 our code reaches about 14 trillion operations per second (measured: 4,624 tokens/s in session 2).
An A100 or H100 is expected to be about 9× or 25× faster with the same code (assumption, not measured).
So the whole EXP-047 run would be very roughly 55 A100-hours or 20 H100-hours: about $130–$400 of
credits at the public prices seen (A100 ≈ $3.67/h, H100 ≈ $6.98/h on Azure), instead of about 10 weeks of
free Kaggle sessions. Even the first $1,000 would likely be enough; it is valid for **90 days**.

**Nothing moves off Kaggle without a new decision (D-053) that you approve.** Moving the run changes
the hardware in the middle of an experiment; that must be planned, written down and tested first.

## Before you start: what you need

| Item | Microsoft for Startups | NVIDIA Inception |
|---|---|---|
| A **personal** Microsoft account (Outlook/Hotmail) that has **never** had Azure | required | - |
| Your personal LinkedIn profile (identity check) | required | helpful |
| A debit or credit card that works for international online payments (Visa/Mastercard), for the identity check | required | - |
| A **registered company** (for example a Private Limited, LLP or other registration) | needed only for the $5,000 step | they ask for company details |
| A simple company website or public page | helpful | usually asked |

Tell the chat if the company is not registered yet: the first Microsoft step and the NVIDIA application may
still be possible, but the larger credits need registration.

## Safety rules (read before clicking anything)

1. **Never** paste a card number, password, one-time code or API key into the Arena chat or the repo.
2. Credits are placed in an Azure "sponsorship" subscription. **Do not upgrade to "pay-as-you-go"** if
   Azure offers it; that is how a card gets charged.
3. **Do not start any virtual machine or GPU yet.** Credits start counting down only when used, but the
   first $1,000 expires 90 days after it is given. We start only when D-053 is approved and the
   commands are ready in the repo.
4. After the account exists, set a budget alert (the chat will give the exact clicks) so any charge is
   visible immediately.

## Step A: Microsoft for Startups (about 30 minutes)

1. Open https://www.microsoft.com/en-us/startups and click **Get started now**.
2. Sign in with your personal Microsoft account (create one at outlook.com if needed; do not use a work
   or college email).
3. When asked for an **investor referral code**, choose **Continue without a code**.
4. Create the Azure account: profile (choose **India** as country; it cannot be changed later), billing
   address, then the card for identity verification. A small temporary hold on the card is common.
5. Complete the identity check with your LinkedIn if asked.
6. Wait up to 24 hours for the credit balance to appear. Microsoft says reviews take about 3 business
   days.
7. If you have a registered company: in the Azure portal, open the startup credits page and complete
   **business verification** within 90 days to unlock the $5,000 level.

What to send back to the chat (no secrets): "Microsoft: approved / pending / refused, credit shown: $___,
expiry date: ___".

## Step B: NVIDIA Inception (about 20 minutes, free, no deadline)

1. Open https://www.nvidia.com/en-us/startups/ and click **Apply Now**
   (https://programs.nvidia.com/phoenix/application).
2. Fill in the company and product details. You may use the text below.
3. NVIDIA reviews the application and replies by email (check spam). Membership lists "free cloud credits
   from NVIDIA and partners" among its benefits; the amount is not published.

What to send back: "NVIDIA: submitted on ___ / accepted / refused".

## Text you may paste into the applications (honest; edit freely)

**One-line description:**
Frontier AI builds open foundation language models for Indian languages, trained from scratch.

**Product description (short):**
We are training our own decoder-only language models from scratch for 12 Indian languages plus
English, with our own tokenizer and a documented, licence-checked training corpus. Our first research
model (190 million parameters, 13 languages) is trained and evaluated; our next model (507 million
parameters, 8.2 billion training tokens) is in training on free GPUs. Every experiment, decision and
result is recorded in our public repository: https://github.com/dfgtghu556-ops/frontier-ai

**What we will use the credits for:**
Pre-training our 507-million-parameter model, and the larger models that follow it, on A100/H100 GPUs
instead of free T4 notebooks, so a training run takes days instead of months.

**Stage / funding:** pre-seed, self-funded, no outside investment (change this if it is not true).

Do not add claims that are not true (users, revenue, benchmark wins). Our current model cannot yet answer
questions (EXP-045: chance level on Belebele in 12 of 13 languages); say "research model".

## After approval

Paste the short status lines above into the Arena chat. The next steps would then be: a GPU quota request
(new Azure accounts usually start with zero GPU quota), a written plan D-053 for moving EXP-047 (or
starting the next run) on the new hardware, a short test, and only then the real run.
