"""Run lm-eval on a model for the given tasks; save a normalized results JSON.

Usage:
    python -m src.eval --model <hf-id-or-path> --tasks wmdp_bio,mmlu --limit <int|None> --out results/<name>.json
"""
import argparse
import datetime
import sys

from src.common import library_versions, load_config, resolve_device, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--tasks", required=True, help="Comma-separated lm-eval task names")
    p.add_argument("--limit", type=int, default=None, help="Sample limit per task (applies to all tasks given)")
    p.add_argument("--mmlu-limit", type=int, default=None, help="Override limit specifically for the mmlu task")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--config", default=None, help="Path to YAML config (defaults to configs/default.yaml)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", default="auto")
    return p.parse_args()


def extract_metric(task_results, preferred=("acc,none", "acc_norm,none", "acc", "acc_norm")):
    for key in preferred:
        if key in task_results:
            return key, task_results[key]
    # fall back to any key ending in a non-stderr accuracy-like metric
    for key, value in task_results.items():
        if not key.endswith("_stderr,none") and not key.endswith("_stderr") and "stderr" not in key:
            return key, value
    raise KeyError(f"No accuracy-like metric found in task results: {list(task_results.keys())}")


def extract_stderr(task_results, metric_key):
    base = metric_key.split(",")[0]
    suffix = "," + metric_key.split(",", 1)[1] if "," in metric_key else ""
    for candidate in (f"{base}_stderr{suffix}", f"{base}_stderr"):
        if candidate in task_results:
            return task_results[candidate]
    return None


def run_eval(model, tasks, limit=None, per_task_limit=None, seed=0, batch_size="auto"):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    per_task_limit = per_task_limit or {}
    lm = HFLM(pretrained=model, device=resolve_device(), batch_size=batch_size)

    results_out = {}
    for task in tasks:
        task_limit = per_task_limit.get(task, limit)
        raw = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            limit=task_limit,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
        )
        task_results = raw["results"][task]
        metric_key, acc = extract_metric(task_results)
        stderr = extract_stderr(task_results, metric_key)
        n = task_results.get("sample_count")
        if isinstance(n, dict):
            n = n.get(metric_key, next(iter(n.values()), None))
        if n is None:
            n = raw.get("n-samples", {}).get(task, {}).get("effective")
        results_out[task] = {"acc": acc, "acc_stderr": stderr, "n": n, "metric": metric_key}
    return results_out


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    set_seed(seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    per_task_limit = {}
    if args.mmlu_limit is not None and "mmlu" in tasks:
        per_task_limit["mmlu"] = args.mmlu_limit
    elif "mmlu" in tasks and args.limit is None:
        cfg_mmlu_limit = config.get("eval", {}).get("mmlu_limit")
        if cfg_mmlu_limit is not None:
            per_task_limit["mmlu"] = cfg_mmlu_limit

    print(f"Evaluating model={args.model} tasks={tasks} limit={args.limit} "
          f"per_task_limit={per_task_limit} seed={seed}", file=sys.stderr)

    results = run_eval(
        model=args.model,
        tasks=tasks,
        limit=args.limit,
        per_task_limit=per_task_limit,
        seed=seed,
        batch_size=args.batch_size,
    )

    import lm_eval

    output = {
        "model": args.model,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lm_eval_version": lm_eval.__version__,
        "tasks": results,
        "config": {
            "tasks": tasks,
            "limit": args.limit,
            "per_task_limit": per_task_limit,
            "seed": seed,
            "batch_size": args.batch_size,
            "library_versions": library_versions(),
        },
    }
    out_path = save_json(args.out, output)
    print(f"Wrote {out_path}")
    for task, r in results.items():
        print(f"  {task}: acc={r['acc']:.4f} (stderr={r['acc_stderr']}) n={r['n']} metric={r['metric']}")


if __name__ == "__main__":
    main()
