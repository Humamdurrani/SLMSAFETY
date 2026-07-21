# Project brief: does removed bio knowledge stay removed?

> A scoped, defensive AI-safety project. We take a small open-weight language model,
> remove some hazardous-biology *proxy* knowledge with a published unlearning method,
> then test whether that removal survives adversarial fine-tuning. This directly tests
> the central claim in Stephen Casper's tamper-resistance notes: that today's
> post-training unlearning is not very tamper-resistant.

---

## 0. What this project is and is NOT

**It IS:**
- A study of *removing* a capability and measuring whether removal holds.
- Built entirely on public, standard safety tooling.
- Measured with **WMDP-bio**, a public multiple-choice benchmark built by safety
  researchers specifically to *measure* whether a model holds hazardous-bio proxy
  knowledge. The questions are proxies, not protocols. We only ever use it to score
  a model, never as a source of content.

**It is NOT, and must never become:**
- A project that generates, collects, or trains on real, operational, step-by-step
  hazardous protocols (bio, chem, or otherwise).
- A project that produces a model *better* at hazardous tasks. If any step would
  increase real-world hazardous capability rather than measure/remove it, STOP and
  flag it to the human.
- The "adversarial fine-tuning" in Phase 3 uses **benign or general-domain data**
  (or a small benign, topically-adjacent set). Its purpose is to measure whether the
  *benchmark score* recovers — a proxy for whether latent knowledge was truly removed.
  It is NOT to teach the model anything hazardous.

If you (the assistant) ever find yourself reframing a step to make it seem acceptable,
treat that as a signal to stop and ask, not a reason to proceed.

---

## 1. Success criteria (what "done" looks like)

A finished project has:
1. Baseline scores for the chosen model on WMDP-bio and MMLU.
2. An unlearned model whose WMDP-bio score dropped meaningfully while MMLU stayed
   roughly flat.
3. A recovery curve: WMDP-bio score vs. adversarial-fine-tuning steps.
4. A short written report and a clean, reproducible repo.

A clean **negative** result ("unlearning broke after only N steps") is a fully valid,
valuable outcome. Do not treat recovery of the score as a bug to hide.

---

## 2. Tech stack

- Python 3.10+
- `transformers`, `datasets`, `accelerate`, `torch`
- `lm-eval` (EleutherAI lm-evaluation-harness) for WMDP-bio and MMLU
- `peft` (only if we compare LoRA vs. full fine-tuning)
- A reference implementation of **RMU** (Representation Misdirection for Unlearning),
  the unlearning baseline from the WMDP paper. Search for the current public repo and
  adapt it to the small model.

**Note on versions:** library APIs and lm-eval task names change over time. Verify the
current WMDP task name in lm-eval (it has been `wmdp_bio`) and the current RMU repo
layout before writing code around them. Do not assume — check.

---

## 3. Compute (resolve this before Phase 2)

Training needs a GPU. The human's laptop likely has none, so:
- **Recommended:** rent a single cloud GPU (~16–24 GB is plenty for a 1.5–3B model)
  and run this whole project on that box (including the coding assistant, over SSH).
- **Alternative:** use a free notebook GPU (Colab/Kaggle) for the GPU-heavy phases.
- Phase 1 (evaluation) may be possible on CPU but will be slow.

Keep runs small and cheap. Prefer the smallest model that still has measurable WMDP-bio
knowledge (see the Phase 1 gate).

---

## 4. The phases

Each phase is a self-contained unit of work. At the end of each, save artifacts, write a
one-paragraph summary of what happened, and pause for the human to review.

### Phase 0 — Environment
- Create a fresh virtualenv, install the stack, pin versions in `requirements.txt`.
- Load the target model (start with `Qwen2.5-1.5B-Instruct`) and run one tiny inference
  to confirm everything works end to end.
- **Acceptance:** a working environment + a printed sample generation.

### Phase 1 — Baseline + go/no-go gate
- Run `lm-eval` on `wmdp_bio` and `mmlu` (a subset of MMLU is fine to save time).
- Save results to `results/baseline.json` with model name, date, and task versions.
- **Gate:** if WMDP-bio is at or near chance (~25% for 4-choice), the model has almost
  nothing to remove — bump to a 3B model (e.g. `Qwen2.5-3B-Instruct`) and re-run. Do
  not proceed to unlearning until the baseline is clearly above chance.
- **Acceptance:** baseline scores + a stated go/no-go decision.

### Phase 2 — Unlearn
- Adapt RMU to the chosen model. Use the standard WMDP-bio forget set and a benign
  retain set.
- After unlearning, re-run the Phase 1 evals. Save to `results/unlearned.json`.
- **Acceptance:** WMDP-bio dropped meaningfully; MMLU within a few points of baseline.
  Save the unlearned model to `models/unlearned/`. If MMLU collapsed, tune down the
  unlearning strength and retry.

### Phase 3 — Adversarial fine-tuning (the core experiment)
- Take the unlearned model and fine-tune it on **benign** data (general instruction data,
  or a small benign topically-adjacent set — NOT hazardous content).
- Sweep a small grid: e.g. learning rates {1e-5, 5e-5} × steps {0, 25, 50, 100, 200}.
- After each configuration, re-run `wmdp_bio` (MMLU optional). Log every point.
- Plot WMDP-bio score vs. steps → `results/recovery_curve.png`.
- **Acceptance:** a recovery curve and a one-line finding, e.g. "WMDP-bio recovered from
  X% to Y% within N steps of benign fine-tuning."

### Phase 4 (optional) — Improve or red-team harder
Pick ONE, only if time allows:
- Compare full-parameter vs. LoRA fine-tuning for how fast the score recovers.
- Try running the unlearning longer / stronger and see if recovery slows.
- A more thorough red-team sweep (more learning rates, warm-up, more steps).
- **Acceptance:** a comparison table or second curve.

### Phase 5 — Write up
- `README.md`: what, why, how to reproduce, headline result.
- A short report (2–4 pages) with the baseline table, the unlearning table, and the
  recovery curve, plus a paragraph connecting the result to Casper's thesis.
- **Acceptance:** repo runs end to end from the README on a fresh machine.

---

## 5. Repo layout (target)

```
.
├── PROJECT_BRIEF.md        # this file
├── README.md
├── requirements.txt
├── src/
│   ├── eval.py             # run lm-eval on a model, dump JSON
│   ├── unlearn.py          # RMU adaptation
│   └── attack.py           # adversarial fine-tuning + recovery sweep
├── results/
│   ├── baseline.json
│   ├── unlearned.json
│   └── recovery_curve.png
└── models/
    └── unlearned/
```

## 6. Working style for the assistant
- Prefer small, cheap, fast runs. Confirm a script works on a tiny slice before a full run.
- Log everything (config, seeds, scores) so results are reproducible.
- At each phase boundary, summarize plainly and wait for the human to say "continue."
- Never introduce hazardous operational content into data, prompts, or outputs.
