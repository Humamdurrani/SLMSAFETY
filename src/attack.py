"""Adversarial fine-tuning: fine-tune the unlearned model on BENIGN data only, to test
whether the Phase 2 wmdp_bio suppression survives (the tamper-resistance question).

Project safety rule: this fine-tunes on benign, general-domain data ONLY, never
anything hazardous. The point is to see whether the *benchmark score* rebounds -- a
proxy for whether the underlying knowledge was truly removed or just masked -- not to
make the model more capable of anything hazardous.

Usage:
    python -m src.attack --model models/unlearned --lr 5e-5 --steps 100 --out results/attack/lr5e-05_steps100.json
"""
import argparse
import datetime
import sys

import torch

from src.common import library_versions, load_config, resolve_device, resolve_dtype, save_json, set_seed
from src.eval import run_eval
from src.unlearn import batched, load_text_iter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path or HF id of the model to attack (e.g. models/unlearned)")
    p.add_argument("--data", default="tatsu-lab/alpaca", help="Benign fine-tuning dataset (HF id)")
    p.add_argument("--data-field", default="text")
    p.add_argument("--data-split", default="train")
    p.add_argument("--data-slice", type=int, default=3000, help="Rows to fetch before filtering/shuffling")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--eval-tasks", default="wmdp_bio", help="Comma-separated; mmlu optional per spec")
    p.add_argument("--mmlu-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def finetune(model, tokenizer, texts_iter, lr, steps, batch_size, max_seq_len, device, optimizer=None):
    """Standard full-parameter causal-LM fine-tuning. Returns (history, optimizer) so
    callers can continue training from where this call left off (see sweep.py)."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for step, batch in enumerate(batched(texts_iter, batch_size)):
        if step >= steps:
            break
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len
        ).to(device)
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append({"step": step, "loss": loss.item()})
        print(f"attack step {step}: loss={loss.item():.4f}", file=sys.stderr)

    model.eval()
    return history, optimizer


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    set_seed(seed)

    attack_cfg = config.get("attack", {})
    batch_size = args.batch_size if args.batch_size is not None else attack_cfg.get("batch_size", 4)
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else config.get("model", {}).get("max_seq_len", 512)
    dtype_name = config.get("model", {}).get("dtype", "bfloat16")
    device = resolve_device()

    print(f"Loading model {args.model} ...", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=resolve_dtype(dtype_name)).to(device)

    history = []
    if args.steps > 0:
        texts = load_text_iter(args.data, args.data_field, args.data_split, slice_n=args.data_slice, seed=seed)
        history, _ = finetune(model, tokenizer, texts, args.lr, args.steps, batch_size, max_seq_len, device)

    eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
    per_task_limit = {}
    if "mmlu" in eval_tasks and args.mmlu_limit is not None:
        per_task_limit["mmlu"] = args.mmlu_limit
    results = run_eval(model=model, tasks=eval_tasks, per_task_limit=per_task_limit, seed=seed, batch_size="auto")

    output = {
        "model": args.model,
        "lr": args.lr,
        "steps": args.steps,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tasks": results,
        "config": {
            "data": args.data,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "seed": seed,
            "library_versions": library_versions(),
        },
        "history": history,
    }
    out_path = save_json(args.out, output)
    print(f"Wrote {out_path}")
    for task, r in results.items():
        print(f"  {task}: acc={r['acc']:.4f}")


if __name__ == "__main__":
    main()
