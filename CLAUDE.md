# CLAUDE.md — bio-unlearning tamper-resistance study

Project context for the coding agent. Keep this current. This is standing guidance loaded
every session; detailed specs live in the linked docs.

## What this project is

A small, defensive AI-safety experiment: take a small open-weight LLM, remove hazardous-bio
*proxy* knowledge (measured by the WMDP-bio benchmark) using the RMU unlearning method, then
test whether that removal survives adversarial fine-tuning on benign data. Goal: measure how
tamper-resistant post-training unlearning actually is.

The owner is not a professional engineer. You write the code; they steer.

## Critical rules (always apply)

- Defensive only: the objective is capability **removal and measurement**, never creation.
- Never generate, collect, store, or train on real operational hazardous procedures
  (bio/chem/nuclear/cyber). No exceptions.
- WMDP-bio is a scoring instrument only. Never mine it for content or surface its items.
- Phase 3 "adversarial fine-tuning" uses **benign, general-domain data only** — it tests
  whether the *benchmark score* rebounds, not whether the model gets more hazardous.
- If any step would plausibly raise real-world hazardous capability, halt and ask the owner.
  Do not reinterpret a step to make it seem acceptable.

## Project docs

- `TECHNICAL_SPEC.md` — authoritative technical detail: data sources, RMU mechanics,
  hyperparameter defaults, JSON schemas, CLI contracts, phase acceptance criteria. Follow it.
- `PROJECT_PLAN.md` — the owner's plain-language roadmap and mental model. Context for how
  they think about the work.
- Any specific default, dataset path, or benchmark task name in these docs was written from
  memory and may be stale. **Verify against the live source before relying on it** (see the
  verification checklist in `TECHNICAL_SPEC.md`).

## Working style

- Small before big: validate every script on a tiny data slice / a few steps before a full
  run. State the plan and expected time/cost first.
- Config-driven and reproducible: read settings from `configs/default.yaml` (+ CLI overrides),
  set and record seeds, embed the resolved config and library versions in every output JSON.
- Explain what you're doing in plain language as you go.
- At each phase boundary, summarize in plain language and **pause for the owner to review**
  before continuing.
- Ask before anything expensive, irreversible, or that installs/downloads something not
  declared in the docs. Fail loudly with actionable messages; never swallow errors silently.

## Environment

- Python ≥ 3.10, single CUDA GPU (16–24 GB is enough for the target model), `bfloat16`.
- Install from `requirements.txt` (pin all versions; commit `pip freeze` at first green run).
- Core libs: `torch`, `transformers`, `datasets`, `accelerate`, `lm-eval`; `peft` only for the
  optional Phase 4 LoRA comparison.
- OOM tactics: bfloat16, gradient checkpointing, smaller batch + gradient accumulation,
  shorter `max_seq_len`.

## How to run (key commands)

```bash
python -m src.eval    --model <id-or-path> --tasks wmdp_bio,mmlu --out results/baseline.json
python -m src.unlearn --base-model Qwen/Qwen2.5-1.5B-Instruct --forget <path> --retain <path> --out models/unlearned
python -m src.attack  --model models/unlearned --data <benign> --lr <lr> --steps <n> --out results/attack/lr<lr>_steps<n>.json
python -m src.sweep   --config configs/default.yaml    # drives the grid, builds recovery_curve.png
```

## Repository map

```
configs/default.yaml   src/{common,eval,unlearn,attack,sweep}.py
results/{baseline.json, unlearned.json, attack/, recovery_curve.png}
models/unlearned/       report/report.md      README.md
```

## Project status (keep updated)

- [x] P0 — environment (model loads, deps pinned)
- [x] P1 — baseline + go/no-go gate  · model: Qwen2.5-1.5B-Instruct  · wmdp_bio: 0.6787 (n=1273)  · mmlu: 0.6215 (n=12173)  · gate: PASS (well above 0.25 chance)
- [x] P2 — RMU unlearn  · FINAL config: layer_id=6, layer_ids=[4,5,6], steering_coeff=4
      (relative multiplier, control_vec_norm=2000), alpha=1200, lr=5e-5, max_steps=150,
      forget=ccdv/pubmed-summarization (open substitute for gated official corpus),
      retain=Salesforce/wikitext. wmdp_bio: 0.6787 -> 0.6080 (-7.1pt, real suppression) ·
      mmlu: 0.6215 -> 0.5821 (-3.9pt, closest to spec's "a few points" tolerance among all
      tuning rounds). Chosen from a 5-point curve over steering_coeff in {2,3,4,5,6.5} (see
      `results/unlearned_round3_sweep.json`); owner-approved 2026-07-20. Model weights not
      persisted from Colab — Phase 3 regenerates this exact deterministic config as its
      first step. `results/unlearned.json` is the canonical record.
- [x] P3 — adversarial fine-tune sweep + recovery curve  · attack data: tatsu-lab/alpaca
      (benign instruction pairs). Grid: lr {1e-5, 5e-5} x steps {0,25,50,100,200}, fine-tuned
      continuously per lr with checkpoint evals. Control (steps=0) reproduced Phase 2's
      exact recorded wmdp_bio (0.6080), confirming deterministic regeneration worked.
      **FINDING: the RMU safeguard was NOT tamper-resistant.** At lr=5e-5, 25 steps (100
      benign docs) recovered ~74% of the suppression gap; 200 steps recovered ~94.5%,
      landing almost exactly back at the pre-unlearning baseline (0.6787). At lr=1e-5,
      slower but still ~73% recovered by 200 steps, neither curve plateaued yet. Full
      numbers in `results/recovery_summary.json`; per-cell results in `results/attack/`;
      chart in `results/recovery_curve.png`.
- [x] P4 — skipped by owner choice (2026-07-20); went straight to write-up.
- [x] P5 — final docs rewritten per owner's style rules (first person, no AI-isms, numbers
      sourced only from results/): `README.md` (GitHub front page), `report/writeup.md`
      (academic-style write-up, supersedes the earlier report/report.md draft, which was
      deleted), `SUMMARY.md` (one-page plain-language version). All referenced files
      verified present. Project complete end-to-end.

**Current focus:** _Project complete (P0-P5 done, P4 skipped by choice)._

## Compute environment (important — read before writing any model-loading code)

- **This machine has no CUDA GPU and only ~3.9 GB RAM** — it cannot load even the 1.5B
  model (confirmed: segfaults on load). Local `.venv` here is CPU-only and good only for
  non-model script/logic work.
- **Actual working environment: Google Colab** (free tier, Tesla T4, 15.6 GB VRAM).
  Confirmed working in Phase 0: model loaded in bf16, `num_hidden_layers=28`,
  `hidden_size=1536`, test generation correct.
- **Workflow:** Claude Code writes a notebook under `notebooks/` for each phase that needs
  the GPU; the owner uploads it to Colab, runs it there, and pastes the output back into
  this session. Claude Code then records results/config files locally from that output.
- `requirements.txt` documents both environments (Colab core deps to `pip install`, plus
  the local CPU venv's full pin list). Full Colab `pip freeze` is archived in
  `requirements-colab-full-freeze.txt`.

## Known gotchas

- Confirmed (P1): lm-eval task names `wmdp_bio` and `mmlu` are current and work as expected
  (lm-eval 0.4.12). `mmlu`'s `--limit` applies per-subtask, not to the group total — a
  "500-sample subset" on the `mmlu` group task actually ran ~12k samples (most of MMLU's 57
  subtasks have <500 examples each). Not wrong, just slower than intended; worth remembering
  if trying to actually cap total mmlu runtime.
- **Corrected (P2): `rmu.layer_id` should be ~20–25% depth, NOT 40-60%.** Verified against the
  live reference implementation (github.com/centerforaisafety/wmdp): Zephyr-7B uses layer
  7/32 (~22%), Yi-34B uses layer 15/60 (~25%). For Qwen2.5-1.5B (28 layers) this project uses
  `layer_id=6`, `layer_ids=[4,5,6]`.
- **Corrected (P2): select target params (`mlp.down_proj.weight`) by NAME, not positional
  index.** The reference implementation's `param_ids=[6]` assumes a bias-less Mistral/Llama
  attention block; Qwen2 has q/k/v biases, which shifts `down_proj` to index 9. `src/unlearn.py`
  selects by name to stay architecture-safe.
- **Forget corpus substitution (P2):** the official `cais/wmdp-bio-forget-corpus` is gated
  (HF access request required). Owner chose to skip that and substitute an open PubMed
  abstract/article corpus (`ccdv/pubmed-summarization`) instead. This is topically similar
  (biomedical literature) but NOT curated for the hazard-adjacent content the official corpus
  targets — expect weaker/less-targeted suppression of `wmdp_bio` than the paper reports, and
  don't compare our results directly to published RMU numbers without noting this. Retain set
  matches the reference exactly: `wikitext-2-raw-v1`, test split.
- **Corrected (P2): bare `wikitext` repo id no longer resolves on the Hub** — confirmed via a
  live `HfUriError`. Current canonical path is `Salesforce/wikitext` (same config/field).
- **Corrected (P2): don't use `streaming=True` for dataset loading in Colab.** First unlearn
  run crashed with a fatal, uncatchable interpreter error (`PyGILState_Release`) after repeated
  network-retry storms on `ccdv/pubmed-summarization`'s streaming reader. `src/unlearn.py` now
  does a bounded, non-streaming slice load (`split[:N]`) instead — slower to start, far more
  reliable. If touching data loading again, keep it non-streaming.
- **Corrected (P2): `steering_coeff` must scale with the model's actual activation
  magnitude, not reuse the reference's raw constant.** First real run (steering_coeff=6.5
  used literally, matching the reference's Zephyr-7B constant) produced a control vector
  ~4000x smaller than Qwen2.5-1.5B's measured activation scale at layer 6 — `forget_loss`
  never moved across 150 steps, and wmdp_bio was unchanged (0.6787 -> 0.6779, noise-level).
  `src/unlearn.py` now measures real activation scale (mean-squared value at the hooked
  layer, on a few forget-set batches) before training and scales the control vector
  relative to it, so `steering_coeff` is a portable multiplier, not an absolute constant.
  Explains why the reference's own constants vary wildly across models (6.5 for Zephyr-7B,
  300 for Mixtral/Yi-34B) — each model has a different natural activation scale.
- If baseline WMDP-bio ≈ chance (~0.25), switch to the fallback model and re-baseline. (P1:
  N/A — baseline was 0.6787, well above chance, gate passed.)

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
