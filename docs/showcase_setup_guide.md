# Showcase setup guide: GitHub settings + private Hugging Face demo (D-053)

For the founder. Two parts. Do **Part A** first (about 5 minutes, only clicks). Do **Part B** when you have
about 30 minutes and the laptop is plugged in.

Never paste a token or password into any chat (including Arena). Tokens are typed only into your own
PowerShell window.

---

## Part A: three GitHub settings (only you can change these)

The Arena session is not allowed to change repo settings (GitHub answered "403 Resource not accessible by
integration"), so these are clicks for you.

### A1. Make the current work the repo's front page (default branch)

Why: visitors to the repo see the default branch. Right now it is the old `main`, so they see an old README.

1. Open https://github.com/dfgtghu556-ops/frontier-ai and log in as the repo owner.
2. In the row of tabs under the repo name (Code, Issues, Pull requests, ...), click **Settings** (gear icon,
   far right; on a small screen it may be hidden under the **...** menu).
3. You land on **General**. Scroll a little to the box **Default branch**. It shows `main`.
4. Click the **two-arrows icon (⇄)** on the right of that box ("Switch to another branch").
5. A small window opens. Click the dropdown that says `main` and choose **`arena/01a0dc16-frontier-ai`**.
6. Click **Update**.
7. A warning appears ("Updating the default branch..."). Click **I understand, update the default branch.**
8. Check: click the **Code** tab. The branch button on the left shows `arena/01a0dc16-frontier-ai` and the
   README below starts with the new status section.

Nothing is deleted. `main` still exists; it is just no longer the front page.

### A2. Fill in the "About" box

1. Click the **Code** tab (the repo's main page).
2. On the right side there is a box **About**. Click the small **gear icon (⚙)** next to the word "About".
3. In **Description**, paste:
   `Open foundation language models for 12 Indian languages + English, trained from scratch. Every experiment, decision and result recorded.`
4. In **Website**, paste:
   `https://dfgtghu556-ops.github.io/frontier-ai/dashboard/`
   (If you see a tick-box "Use your GitHub Pages website", leave it **unticked**; it would use the address
   without `/dashboard/`. That address also works, it just takes one extra redirect.)
5. In **Topics**, type each word and press **Space** (or Enter) after each, so each becomes a small blue tag:
   `llm` `language-model` `indian-languages` `indic-nlp` `pytorch` `pretraining` `tokenizer` `research`
6. Leave the three tick-boxes at the bottom (Releases, Packages, Deployments) as they are.
7. Click **Save changes**.
8. Check: the About box now shows the description, the link and the blue topic tags.

### A3. Switch on the dashboard website (GitHub Pages)

1. Click **Settings** again.
2. In the **left** menu, under the heading "Code and automation", click **Pages**.
3. Under **Build and deployment**, find **Source**. Choose **Deploy from a branch**.
4. Under **Branch** there are two dropdowns and a Save button:
   - first dropdown (says `None` or `main`): choose **`arena/01a0dc16-frontier-ai`**;
   - second dropdown (says `/ (root)`): choose **`/docs`**;
   - click **Save**.
5. Wait 1–2 minutes, then refresh the page (F5). A box at the top says
   **"Your site is live at https://dfgtghu556-ops.github.io/frontier-ai/"**.
   (While it is still building you may see nothing, or "Your site is ready to be published". Wait and refresh.)
6. Check: open https://dfgtghu556-ops.github.io/frontier-ai/dashboard/ — the Frontier Lab OS dashboard opens.
   Click Experiments, Decisions, Models in its menu; D-053 is on the Decisions page.

If the site shows "404 There isn't a GitHub Pages site here", wait 5 more minutes and refresh. If it still
shows 404 after 15 minutes, take a screenshot of Settings → Pages and tell the chat.

### A4. Tell the chat

Type `done` (or what went wrong) in the Arena chat.

---

## Part B: the private Hugging Face demo page

What you get: a web page, visible only to you when logged in, where you type the beginning of a text
(Hindi, Tamil, Bengali, ... or English) and the EXP-043 model continues it. It runs on Hugging Face's
free computers, so your laptop does not run the model; the laptop only prepares and uploads the files once.

Be honest with yourself about what you will see: EXP-043 is a small (190 M parameter) **base** model. It
continues text; it does not answer questions or chat like ChatGPT. Its sentences will often wander or
repeat. That is expected at this size, and it is why EXP-047 (bigger) is running.

### B1. Make a free Hugging Face account (skip if you have one)

1. Open https://huggingface.co/join
2. Enter your email and a new password → **Next**.
3. Choose a **Username** (this becomes part of the page address, e.g. `huggingface.co/spaces/<username>/...`)
   and your full name. Accept the terms → **Create Account**.
4. Open your email inbox, find the mail from Hugging Face and click the **confirmation link**. (Without the
   confirmed email Hugging Face may refuse to create the page.)

The free account costs nothing. Do not add a card and do not choose "Pro" or any paid hardware.

### B2. Make a "Write" token (a password just for the upload script)

1. Log in at https://huggingface.co
2. Click your **round profile picture** (top right) → **Settings** → in the left menu **Access Tokens**
   (or open https://huggingface.co/settings/tokens directly).
3. Click **+ Create new token**.
4. At the top choose the token type **Write** (not "Read", not "Fine-grained").
5. **Token name:** `frontier-pc`
6. Click **Create token**.
7. The token is shown **once**. It starts with `hf_`. Click the **copy** button next to it. Keep this
   browser tab open until step B5 is finished. Do not paste it anywhere except the PowerShell window in B5.
8. Click **Done** only after you have used it in B5 (if you lose it, just make a new one).

### B3. Open PowerShell in the frontier-ai folder

1. Plug in the laptop. Close other programs (browser can stay open).
2. Open **File Explorer** and go to your `frontier-ai` folder (the same folder you use for the Kaggle
   runs).
3. Click once in the **address bar** at the top (where the folder path is shown), so the path turns
   blue/selected.
4. Type `powershell` and press **Enter**. A blue (or black) PowerShell window opens, already inside the
   frontier-ai folder.

### B4. Get the latest code

Type this line in PowerShell and press Enter:

```
git pull
```

It should end with something like `Fast-forward` or `Already up to date.` If it says "error" or
"conflict", stop and paste its last lines into the chat.

### B5. Run the demo script

Type this line and press Enter:

```
powershell -ExecutionPolicy Bypass -File scripts\run_hf_demo.ps1
```

What happens, step by step:

1. **First run only:** "Installing the Hugging Face tool..." (about 1 minute).
2. **First run only:** it asks for your token: `Enter your token (input will not be visible):`
   - Paste the token: **right-click** inside the PowerShell window (or press Ctrl+V).
   - **Nothing appears on screen. That is normal** (it is hidden like a password).
   - Press **Enter** once.
   - If it asks `Add token as git credential? (Y/n)`, type **n** and press Enter.
   - It prints `Logged in to Hugging Face as <your username>`. The token is now saved only on this laptop;
     later runs will not ask again.
3. `Checking ... model_final.pt ...`: it looks for the EXP-043 model file on your laptop and checks its
   fingerprint (SHA-256) against the record in the repo.
4. If the file is not on the laptop: `Downloading model_final.pt (about 880 MB) ...` from your Kaggle
   account. This uses **no GPU hours**; it is only a download. Time depends on your internet.
5. It prepares the half-size (fp16) copy of the model (about 380 MB), tests the page's code on the laptop
   with a few words, and uploads everything to the **private** page `<username>/frontier-exp043-preview`.
6. At the end it prints the report.

Total time: roughly 10–30 minutes, mostly download and upload (NOT MEASURED on your laptop). Do not use
the laptop meanwhile.

### B6. Read the report and send it

The last lines on screen (also saved in `out\hf_demo\REPORT.txt`) look like this when it worked:

```
PASS 1: model_final.pt sha256 matches the EXP-043 record (5494c0285091917...)
PASS 2: bundle built (190,064,000 parameters; fp16 file ... bytes)
PASS 3: bundle test generated ... tokens in ... s
PASS 4: uploaded to the PRIVATE Space https://huggingface.co/spaces/<username>/frontier-exp043-preview
RESULT: COMPLETE. Paste this report into the Arena chat.
```

Copy those lines into the Arena chat. (To open the file: in File Explorer go to `frontier-ai\out\hf_demo\`
and double-click `REPORT.txt`; select all with Ctrl+A, copy with Ctrl+C.)

If it says `RESULT: FAILED`, paste it anyway; the FAIL line tells the chat what went wrong. Nothing is lost
by running the same line again after a fix.

### B7. Open and try your demo page

1. Open the link from PASS 4 in the browser where you are **logged in** to Hugging Face. (Logged-out
   visitors get "404": that is the private setting working.)
2. The first time, the page shows **Building** for about 5–10 minutes (Hugging Face installs PyTorch on its
   computer). Wait, then refresh.
3. When it shows the page "Frontier AI - research preview (EXP-043, 190 M parameters)":
   - Type the beginning of a text in **"Beginning of a text"**, e.g. `भारत एक विशाल देश है`, or click one of
     the examples below the button.
   - **New tokens:** how much to write (more = slower).
   - **Temperature:** lower = safer and more repetitive, higher = more varied and more mistakes.
   - **Seed:** the same seed with the same text gives the same answer; change it for a different answer.
   - Click **Continue the text** and wait (free computers: several seconds or more).
4. Next to the page name you will see a **Private** label. Keep it that way. Making it public is a separate
   decision (licence for the weights, credit to the data sources, a full model card).
5. After a while without use the page goes to **Sleeping**. Open it and click **Restart this Space**; it
   wakes in a minute or two.

### B8. Keep the token safe

- The token is like a password for your Hugging Face account. Never paste it into a chat, email or file.
- If you think it leaked: https://huggingface.co/settings/tokens → next to `frontier-pc` click the **...** →
  **Delete** (or **Invalidate**), then make a new one; the script will ask again.

---

## Troubleshooting

| You see | What to do |
|---|---|
| `git pull` says "Your local changes would be overwritten" | Stop; paste the message into the chat. |
| "the .venv folder was not found" | PowerShell is not in the frontier-ai folder; redo B3. |
| "the Hugging Face login did not work" | The token was copied wrongly or is type Read. Make a new **Write** token (B2) and run B5 again. |
| "model_final.pt could not be downloaded from Kaggle" | Check the internet; run B5 again. If it repeats, paste the report. |
| "403" or "You don't have the rights" at upload | The email is probably not confirmed (B1 step 4), or the token is not Write. |
| The page stays on "Building" over 20 min or shows "Runtime error" | Click the **Logs** button on the page, copy the last lines, paste into the chat. |
| The model writes nonsense or repeats | Expected for a 190 M base model; try another seed or lower temperature. |
