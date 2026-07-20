# Project plan — does removed bio knowledge stay removed?

A single, self-contained development plan for a small, defensive AI-safety project.
We take a small open-weight language model, remove some hazardous-biology *proxy*
knowledge using a published method, then test whether that removal survives someone
trying to fine-tune it back. This directly tests the central claim in Stephen Casper's
tamper-resistance notes: that today's post-training "unlearning" is not very
tamper-resistant.

You (the project owner) are not a professional engineer. That's fine — the coding
assistant (Claude Code) writes the code. Your job is to steer: read this plan, work one
phase at a time, paste the prompts, and check the results against the "Done when" boxes.

---

## How to use this document

1. Read the whole thing once, slowly. Don't code yet.
2. Do the two setup decisions (compute + repo). Come back to a human helper if stuck.
3. Then work through Phase 0 → 5 **in order**. Do not skip ahead.
4. Each phase has the same six parts:
   - **Goal** — what you're producing.
   - **Why** — how it fits the big picture.
   - **What Claude Code does** — plain-language description of the work.
   - **Prompt to paste** — copy this into Claude Code to start the phase.
   - **Done when** — the checklist that means the phase is finished.
   - **If it breaks** — the most common problems and what to do.
5. At every phase boundary, have Claude Code summarize what it did in plain language,
   then stop and review before continuing.

Keep this file in your project folder as `PROJECT_PLAN.md` so Claude Code can read it
any time.

---

## The concept in one page

A language model is billions of tiny numerical dials called **weights**. You train it by
feeding it huge amounts of text; every time it guesses the next word wrong, the dials get
nudged so it does better. After enough text, the dials settle into a configuration that
"knows" a lot — but the knowledge is not filed in labeled drawers. There is no single dial
for "biology." Every fact is smeared across millions of dials at once. This one fact
explains everything else.

Training happens in two stages:
- **Pretraining** — the giant read (trillions of words). Almost all knowledge comes from here.
- **Fine-tuning** — a much smaller nudge afterward to shape behavior (be chatty, follow
  instructions, or — for us — try to forget something).

**Why removing a capability is hard.** Because no single dial holds "bio knowledge," you
cannot delete it like a file. What "unlearning" methods actually do is more like building a
thin wall in front of the knowledge, or scrambling the internal signals so the model can't
*use* it. The information is often still physically in the weights — just harder to reach.

**Why open weights make this a research problem.** A model behind a company API is safe
with just a thin wall — nobody can touch the dials. But an open-weight model is downloaded
to your own computer, and anyone can fine-tune it, nudging the dials again. A surprisingly
small nudge often knocks the wall over and the "removed" knowledge comes back. That rebound
is what this project measures.

**The four pieces of this project:**
- **WMDP-bio** — a thermometer. Multiple-choice questions that measure how much
  hazardous-bio *proxy* knowledge the model can still use. High score = it knows;
  chance-level = it can't.
- **RMU** — one specific "build the wall / scramble the signal" method. Our starting baseline.
- **Adversarial fine-tuning** — you playing the attacker: nudge the dials and watch the
  thermometer to see if the knowledge returns.
- **"Deep ignorance"** (Casper's preferred idea) — don't install the knowledge in the first
  place by filtering the pretraining data. It beats after-the-fact removal, but requires
  training from scratch, so it is out of scope here. This project measures how flimsy the
  after-the-fact removal really is.

---

## Guardrails (non-negotiable)

**This project IS:** a study of *removing* a capability and measuring whether removal holds,
built entirely on public, standard safety tooling.

**This project is NOT, and must never become:**
- A project that generates, collects, or trains on real, operational, step-by-step hazardous
  protocols (bio, chem, or otherwise).
- A project that produces a model *better* at hazardous tasks.
- WMDP-bio is used **only to score a model**, never as a source of content. Its questions
  are proxies, not protocols.
- The "adversarial fine-tuning" in Phase 3 uses **benign / general-domain data**. Its purpose
  is to see whether the *benchmark score* recovers — a proxy for whether latent knowledge was
  truly removed. It is never to teach the model anything hazardous.

Instruction for Claude Code: if any step would increase real-world hazardous capability
rather than measure or remove it, stop and flag it to the human. If you find yourself
reframing a step to make it seem acceptable, that is a signal to stop and ask.

---

## Before you start: two setup decisions

### 1. Compute (a GPU)

Training needs a GPU. A laptop usually has none, so pick one:
- **Recommended:** rent one cloud GPU (16–24 GB is plenty for a 1.5–3B model) and run this
  whole project on that box, including Claude Code over SSH, so the assistant and the GPU
  live in the same place. Services: RunPod, Lambda, Vast.ai. Rough cost: well under $1/hour.
- **Free alternative:** use a notebook GPU (Google Colab / Kaggle) for the GPU-heavy phases.
- Phase 1 (evaluation only) can run on CPU but slowly.

Keep runs small and cheap. Turn the box off when you're not using it. If you want exact,
click-by-click setup steps for a specific service, ask a human helper and share your budget.

### 2. Repo + Claude Code

- Make an empty project folder and open it in Cursor.
- Put this `PROJECT_PLAN.md` in the folder root.
- Claude Code runs in the terminal there and will create the rest of the files.

---

## Phase 0 — Environment

**Goal.** A working Python setup that can load a small model and generate text.

**Why.** Everything later depends on a clean, reproducible environment. Getting this solid
now prevents mysterious failures later.

**What Claude Code does.** Creates a virtual environment, installs the libraries, pins their
versions in `requirements.txt`, downloads the model, and runs one tiny test generation.

**Prompt to paste:**
> Read PROJECT_PLAN.md. We're starting Phase 0. Create a fresh Python virtualenv, install
> transformers, datasets, accelerate, torch, lm-eval, and peft, and pin the versions in
> requirements.txt. Then load Qwen2.5-1.5B-Instruct and run one short test generation to
> confirm everything works end to end. Explain each step in plain language as you go, and
> stop when Phase 0's "Done when" criteria are met.

**Done when.**
- `requirements.txt` exists with pinned versions.
- The model loads without errors.
- You see a short, sensible sample generation printed.

**If it breaks.**
- "Out of memory" → you're probably on CPU or a tiny GPU; that's OK for Phase 0, just note it.
- Install errors → ask Claude Code to show the exact error and fix versions one at a time.

---

## Phase 1 — Baseline + go/no-go gate

**Goal.** Measure how much the model knows *before* any removal, on two thermometers:
WMDP-bio (the hazardous-bio proxy) and MMLU (general knowledge, so we can later check we
didn't damage the model's normal abilities).

**Why.** You can't show you removed something without a "before" number. And there's a trap:
if a small model barely knows the bio material to begin with, there's nothing to remove and
the whole experiment falls flat. This phase catches that.

**What Claude Code does.** Runs the lm-eval harness on the `wmdp_bio` and `mmlu` tasks and
saves the scores to a JSON file.

**Prompt to paste:**
> We're on Phase 1. Using lm-eval, evaluate the current model on the wmdp_bio task and on
> mmlu (a subset of mmlu is fine to save time). Save the results to results/baseline.json
> with the model name, date, and task versions. Then tell me the two scores in plain
> language and remind me what "chance level" is for each.

**Done when.**
- `results/baseline.json` exists with both scores.
- You know the WMDP-bio score and the MMLU score.

**Go/no-go gate.** WMDP-bio is 4-choice, so ~25% is random guessing. If the model scores at
or near 25%, it barely knows the material — there's nothing to remove. **Stop and switch to
a bigger model** (e.g. Qwen2.5-3B-Instruct), then re-run Phase 1. Only continue to Phase 2
once WMDP-bio is clearly above chance.

**If it breaks.**
- Task name errors → lm-eval task names change over time; ask Claude Code to list available
  tasks and confirm the current name for WMDP-bio.
- Very slow on CPU → run mmlu on a small subset, or move to your GPU box now.

---

## Phase 2 — Unlearn

**Goal.** Apply the RMU method to remove the bio proxy knowledge, producing an "unlearned"
copy of the model whose WMDP-bio score has dropped while its MMLU score stays roughly the same.

**Why.** This is the "build the wall" step — the safeguard whose durability you'll test next.

**What Claude Code does.** Adapts a public reference implementation of RMU to the small model,
runs it using the standard WMDP-bio "forget" set and a benign "retain" set, saves the new
model, and re-runs the Phase 1 evaluations on it.

**Prompt to paste:**
> We're on Phase 2. Find the current public reference implementation of RMU (Representation
> Misdirection for Unlearning, from the WMDP benchmark paper) and adapt it to our model. Run
> it using the standard WMDP-bio forget set and a benign retain set. Save the unlearned model
> to models/unlearned/. Then re-run the wmdp_bio and mmlu evaluations and save them to
> results/unlearned.json. Show me a before/after comparison and explain whether it worked.

**Done when.**
- `models/unlearned/` contains the unlearned model.
- WMDP-bio dropped meaningfully (ideally toward chance).
- MMLU stayed within a few points of the baseline.
- `results/unlearned.json` records the after scores.

**If it breaks.**
- MMLU collapsed too → the removal was too aggressive; ask Claude Code to reduce the
  unlearning strength (the "steering coefficient") and retry.
- WMDP-bio barely moved → increase the strength, or unlearn on more layers; retry.
- This is the hardest phase. Expect a few rounds of tuning. That's normal.

---

## Phase 3 — Adversarial fine-tuning (the core experiment)

**Goal.** Play the attacker. Fine-tune the unlearned model on **benign** data and watch
whether the WMDP-bio score climbs back. Produce a recovery curve: score vs. number of
fine-tuning steps.

**Why.** This is the whole point. If a little benign nudging brings the "forgotten" knowledge
back, you've shown the safeguard was shallow — exactly Casper's thesis. A clean result here,
positive or negative, is the deliverable.

**What Claude Code does.** Fine-tunes the unlearned model on benign, general-domain data across
a small grid of settings, re-measures WMDP-bio after each, logs everything, and plots the curve.

**Prompt to paste:**
> We're on Phase 3, the core experiment. Take the unlearned model and fine-tune it on benign,
> general-domain instruction data only — never anything hazardous. Sweep a small grid: learning
> rates {1e-5, 5e-5} by steps {0, 25, 50, 100, 200}. After each configuration, re-run wmdp_bio
> and log the score. Plot WMDP-bio score versus fine-tuning steps and save it to
> results/recovery_curve.png. Then give me a one-sentence finding.

**Done when.**
- `results/recovery_curve.png` shows score vs. steps.
- You can state a finding, e.g. "WMDP-bio recovered from X% to Y% within N benign steps."

**If it breaks.**
- Score never recovers → interesting and legitimate; note it and consider a stronger attack
  in Phase 4.
- Runs too expensive → shrink the grid; even {0, 50, 200} steps at one learning rate tells a story.

Reminder: the attack data is benign. We're measuring whether the *benchmark score* rebounds,
not building a hazardous model.

---

## Phase 4 — (Optional) Improve or red-team harder

**Goal.** One extra experiment, only if you have time and momentum.

**Why.** Casper's notes repeatedly say the field lacks thorough, independent red-teaming.
Doing more here is genuinely valuable.

**Pick ONE:**
- Compare full-model fine-tuning vs. LoRA (a lightweight fine-tuning trick) for how fast the
  score recovers.
- Run the Phase 2 unlearning longer or stronger and see if recovery slows.
- A broader attack sweep (more learning rates, a warm-up schedule, more steps).

**Prompt to paste (example, LoRA comparison):**
> We're on Phase 4. Repeat the Phase 3 recovery experiment twice — once with full-parameter
> fine-tuning and once with LoRA fine-tuning — and put both recovery curves on one chart so we
> can compare how quickly the knowledge returns under each. Save it and give me the takeaway.

**Done when.** You have a second curve or a comparison table plus a one-line takeaway.

---

## Phase 5 — Write up

**Goal.** A clean repo plus a short report anyone can read and reproduce.

**Why.** The result only matters if others can understand and trust it.

**What Claude Code does.** Writes a README, assembles the numbers and charts into a short
report, and makes sure the whole thing runs from scratch.

**Prompt to paste:**
> We're on Phase 5. Write a README.md explaining what this project is, why it matters, how to
> reproduce it, and the headline result. Then write a 2–4 page report with the baseline table,
> the unlearning before/after table, and the recovery curve, plus one paragraph connecting the
> finding to Stephen Casper's argument that post-training unlearning is not very
> tamper-resistant. Finally, verify the project runs end to end from the README instructions.

**Done when.**
- `README.md` and the report exist.
- A fresh run from the README reproduces the results.

---

## Tech stack reference

- Python 3.10+
- `transformers`, `datasets`, `accelerate`, `torch` — load and train models
- `lm-eval` (EleutherAI lm-evaluation-harness) — run WMDP-bio and MMLU
- `peft` — only if you do the LoRA comparison in Phase 4
- A public reference implementation of **RMU** — the unlearning baseline

Versions and task names drift over time. Have Claude Code verify the current WMDP-bio task
name in lm-eval and the current RMU repo layout before writing code around them.

---

## Target repo layout

```
.
├── PROJECT_PLAN.md         # this file
├── README.md               # written in Phase 5
├── requirements.txt
├── src/
│   ├── eval.py             # run lm-eval on a model, save JSON
│   ├── unlearn.py          # RMU adaptation
│   └── attack.py           # adversarial fine-tuning + recovery sweep
├── results/
│   ├── baseline.json
│   ├── unlearned.json
│   └── recovery_curve.png
└── models/
    └── unlearned/
```

---

## Glossary (plain language)

- **Weights / parameters** — the billions of numerical dials that make up the model.
- **Pretraining** — the huge initial training on trillions of words; source of most knowledge.
- **Fine-tuning** — a small follow-up training to adjust behavior.
- **Benchmark** — a fixed test used to score a model. WMDP-bio and MMLU are benchmarks.
- **WMDP-bio** — a benchmark of hazardous-bio *proxy* questions; our removal thermometer.
- **MMLU** — a broad general-knowledge benchmark; our "did we break the model?" check.
- **Unlearning** — methods that try to make a model stop using a specific capability.
- **RMU** — one unlearning method that scrambles the model's internal signals on the target topic.
- **Tamper-resistance** — how well a safeguard survives someone actively fine-tuning to undo it.
- **Adversarial fine-tuning** — deliberately fine-tuning to try to bring a removed capability back.
- **LoRA** — a lightweight fine-tuning trick that changes few weights (used only in Phase 4).
- **Learning rate** — how big each training nudge is; bigger = faster but riskier.
- **Steps / epochs** — how much training happens; more steps = more nudging.
- **Checkpoint** — a saved copy of the model at a point in time.
- **Chance level** — the score you'd get by random guessing (about 25% on 4-choice questions).

---

## Realistic expectations

- This is a multi-week project, not a weekend one — even with Claude Code writing the code.
- The two hardest parts are Phase 2 (getting RMU working and tuned) and keeping the GPU
  environment stable.
- A negative-looking result (the safeguard breaks easily) is a real, valuable finding, not a
  failure. That is largely what the field expects and what your experiment is here to measure.
- Work one phase at a time, keep runs cheap, and check in at every boundary.
