"""Drive the (lr, steps) grid for Phase 3's adversarial fine-tuning; build recovery_curve.png.

For efficiency this does NOT reload+retrain the model from scratch for each (lr, steps)
cell -- the spec's steps_grid is cumulative (0,25,50,100,200). Instead, for each lr it loads
the model ONCE and fine-tunes CONTINUOUSLY, evaluating+saving a
results/attack/lr{lr}_steps{n}.json checkpoint at each grid point along the way. This gives
the same result at each checkpoint as an independent from-scratch run would (deterministic
given the fixed seed and data order), but costs 3 model loads total (1 control + one per lr)
instead of 10.

Usage:
    python -m src.sweep --config configs/default.yaml
"""
import argparse
import sys
from pathlib import Path

import torch

from src.attack import finetune
from src.common import load_config, resolve_device, resolve_dtype, save_json, set_seed
from src.eval import run_eval
from src.unlearn import load_text_iter

BASELINE_WMDP_BIO = 0.6787117046347211  # Phase 1 measured baseline (results/baseline.json)
CHANCE_WMDP_BIO = 0.25


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/unlearned")
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default="results/attack")
    p.add_argument("--chart-out", default="results/recovery_curve.png")
    return p.parse_args()


def eval_and_save(model, out_path, lr, steps, seed, eval_tasks):
    results = run_eval(model=model, tasks=eval_tasks, seed=seed, batch_size="auto")
    output = {"model": "models/unlearned", "lr": lr, "steps": steps, "tasks": results}
    save_json(out_path, output)
    label = f"lr={lr}" if lr is not None else "control"
    print(f"[{label} steps={steps}] " + " ".join(f"{k}={v['acc']:.4f}" for k, v in results.items()), file=sys.stderr)
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("seed", 0)
    set_seed(seed)

    attack_cfg = config.get("attack", {})
    learning_rates = attack_cfg.get("learning_rates", [1e-5, 5e-5])
    steps_grid = sorted(attack_cfg.get("steps_grid", [0, 25, 50, 100, 200]))
    batch_size = attack_cfg.get("batch_size", 4)
    max_seq_len = config.get("model", {}).get("max_seq_len", 512)
    data = attack_cfg.get("data") or "tatsu-lab/alpaca"
    dtype_name = config.get("model", {}).get("dtype", "bfloat16")
    device = resolve_device()
    eval_tasks = ["wmdp_bio"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading unlearned model for steps=0 control eval...", file=sys.stderr)
    control_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=resolve_dtype(dtype_name)).to(device)
    control_results = eval_and_save(control_model, out_dir / "lr_control_steps0.json", None, 0, seed, eval_tasks)
    del control_model
    if device == "cuda":
        torch.cuda.empty_cache()

    nonzero_steps = [s for s in steps_grid if s > 0]
    all_results = {}
    for lr in learning_rates:
        print(f"\n{'='*20} lr={lr} {'='*20}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=resolve_dtype(dtype_name)).to(device)

        texts = load_text_iter(data, "text", "train", slice_n=3000, seed=seed)
        optimizer = None
        trained_so_far = 0
        lr_results = {0: control_results}
        for target_steps in nonzero_steps:
            steps_this_leg = target_steps - trained_so_far
            _, optimizer = finetune(
                model, tokenizer, texts, lr, steps_this_leg, batch_size, max_seq_len, device, optimizer=optimizer
            )
            trained_so_far = target_steps
            out_path = out_dir / f"lr{lr}_steps{target_steps}.json"
            lr_results[target_steps] = eval_and_save(model, out_path, lr, target_steps, seed, eval_tasks)
        all_results[lr] = lr_results

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    build_chart(all_results, args.chart_out)


def build_chart(all_results, chart_out):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for lr, by_steps in all_results.items():
        xs = sorted(by_steps.keys())
        ys = [by_steps[s]["wmdp_bio"]["acc"] for s in xs]
        plt.plot(xs, ys, marker="o", label=f"lr={lr}")
    plt.axhline(BASELINE_WMDP_BIO, linestyle="--", color="gray", label="baseline (pre-unlearning)")
    plt.axhline(CHANCE_WMDP_BIO, linestyle=":", color="red", label="chance (0.25)")
    plt.xlabel("Fine-tuning steps (benign data)")
    plt.ylabel("WMDP-bio accuracy")
    plt.title("Recovery curve: does unlearned bio knowledge come back?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_out)
    print(f"Wrote {chart_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
