"""RMU (Representation Misdirection for Unlearning) adaptation.

Adapted from the reference implementation (Li et al. 2024, "The WMDP Benchmark"),
github.com/centerforaisafety/wmdp (rmu/unlearn.py, rmu/utils.py), verified against
the live repo source. Key differences from a naive port, both deliberate:
  - Target parameters are selected by NAME (e.g. "mlp.down_proj.weight"), not by a
    hardcoded positional index -- the reference's `param_ids=[6]` assumes a bias-less
    Mistral/Llama-style block; Qwen2 has attention biases, which shifts down_proj to
    index 9. Name-based selection is robust to that.
  - All non-target parameters are explicitly frozen (requires_grad=False), matching
    the project spec's "freeze all other parameters"; the reference relies on the
    optimizer's param group alone and never sets requires_grad=False elsewhere.
  - Forget-set substitute: the official cais/wmdp-bio-forget-corpus is gated
    (requires an HF access request). By owner's choice, this uses an open PubMed
    abstract/article corpus (ccdv/pubmed-summarization) instead -- topically in the
    same domain (biomedical literature) but not curated for hazard-adjacent content
    the way the official corpus is, so suppression may be weaker/less targeted. This
    is a known, documented limitation, not an attempt to reproduce the official corpus.

Usage:
    python -m src.unlearn --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --layer-id 6 --layer-ids 4,5,6 --steering-coeff 6.5 --alpha 1200 \
        --lr 5e-5 --max-steps 150 --batch-size 4 --out models/unlearned
"""
import argparse
import datetime
import sys

import torch

from src.common import library_versions, load_config, resolve_device, resolve_dtype, save_json, set_seed

TARGET_PARAM_NAME = "mlp.down_proj.weight"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=None, help="HF model id or local path (default: config model.base)")
    p.add_argument("--layer-id", type=int, default=None, help="Layer whose activations are hooked for both losses")
    p.add_argument("--layer-ids", default=None, help="Comma-separated layer indices to update (e.g. 4,5,6)")
    p.add_argument("--steering-coeff", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None, help="Truncation length for forget/retain batches")
    p.add_argument("--forget-dataset", default="ccdv/pubmed-summarization")
    p.add_argument("--forget-field", default="article")
    p.add_argument("--forget-split", default="train")
    p.add_argument("--forget-slice", type=int, default=3000, help="Rows to fetch before filtering/shuffling")
    p.add_argument("--retain-dataset", default="Salesforce/wikitext")
    p.add_argument("--retain-config", default="wikitext-2-raw-v1")
    p.add_argument("--retain-field", default="text")
    p.add_argument("--retain-split", default="test")
    p.add_argument("--retain-slice", type=int, default=3000, help="Rows to fetch before filtering/shuffling")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def get_target_params(model, layer_ids, param_name=TARGET_PARAM_NAME):
    """Freeze every parameter, then return (unfrozen) the named param at each layer index."""
    for p in model.parameters():
        p.requires_grad_(False)
    selected = []
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        found = False
        for name, p in layer.named_parameters():
            if name == param_name:
                p.requires_grad_(True)
                selected.append(p)
                found = True
                break
        if not found:
            raise ValueError(f"Parameter '{param_name}' not found in layer {layer_id}")
    return selected


def forward_with_cache(model, inputs, module, no_grad=True):
    cache = {}

    def hook(_mod, _inp, out):
        cache["activation"] = out[0] if isinstance(out, tuple) else out

    handle = module.register_forward_hook(hook)
    try:
        if no_grad:
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)
    finally:
        handle.remove()
    return cache["activation"]


def load_text_iter(dataset_name, field, split, config=None, slice_n=3000, seed=0):
    """Non-streaming, bounded slice load. Deliberately NOT using `streaming=True`:
    it proved unreliable in practice (repeated range-request retries against flaky
    network conditions eventually crashed the Python interpreter itself, not just
    raised a catchable exception). A bounded slice download is slower to start but
    far more robust, and we only ever need a few thousand documents at most."""
    from datasets import load_dataset

    sliced_split = f"{split}[:{slice_n}]"
    ds = load_dataset(dataset_name, config, split=sliced_split) if config else \
        load_dataset(dataset_name, split=sliced_split)
    ds = ds.shuffle(seed=seed)
    for example in ds:
        text = example[field]
        if text and text.strip():
            yield text


def batched(iterator, batch_size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []


def estimate_activation_scale(model, tokenizer, calibration_batches, module, max_seq_len, device):
    """Mean squared per-element activation magnitude at `module`, measured on real data.

    A raw random unit vector (L2 norm 1) scaled by a small constant like the reference
    implementation's steering_coeff=6.5 is utterly negligible next to real hidden-state
    magnitudes for some models -- e.g. on Qwen2.5-1.5B at layer 6, measured mean-squared
    activation was ~162, meaning a norm-6.5 target contributes ~0.03 to the per-element
    MSE (6.5^2 / hidden_size). The forget loss then can't move the model at all: gradient
    pressure toward a target ~4000x smaller than the activations themselves is negligible.
    This measures the real scale so steering_coeff can be a portable multiplier of it,
    instead of an absolute constant tuned for a different model family.
    """
    values = []
    with torch.no_grad():
        for batch in calibration_batches:
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len
            ).to(device)
            act = forward_with_cache(model, inputs, module, no_grad=True)
            values.append(act.pow(2).mean().item())
    return sum(values) / len(values) if values else 1.0


def run_rmu(
    model,
    frozen_model,
    tokenizer,
    forget_texts,
    retain_texts,
    layer_id,
    layer_ids,
    steering_coeff,
    alpha,
    lr,
    max_steps,
    batch_size,
    max_seq_len,
    device,
    n_calibration_batches=3,
):
    hidden_size = model.config.hidden_size
    module = model.model.layers[layer_id]
    frozen_module = frozen_model.model.layers[layer_id]

    forget_batches = batched(forget_texts, batch_size)
    retain_batches = batched(retain_texts, batch_size)

    calibration_batches = []
    for _ in range(n_calibration_batches):
        try:
            calibration_batches.append(next(forget_batches))
        except StopIteration:
            break
    typical_ms = estimate_activation_scale(model, tokenizer, calibration_batches, module, max_seq_len, device)
    typical_rms = typical_ms ** 0.5

    # unit_vector's per-element magnitude is ~1/sqrt(hidden_size); scaling by
    # steering_coeff * typical_rms * sqrt(hidden_size) makes each element's magnitude
    # ~steering_coeff * typical_rms -- i.e. steering_coeff natural-activation-scales.
    unit_vector = torch.rand(1, 1, hidden_size, device=device, dtype=model.dtype)
    unit_vector = unit_vector / unit_vector.norm()
    control_vec = unit_vector * (steering_coeff * typical_rms * (hidden_size ** 0.5))
    print(f"calibration: typical_ms={typical_ms:.4f} typical_rms={typical_rms:.4f} "
          f"control_vec_norm={control_vec.norm().item():.4f}", file=sys.stderr)

    params = get_target_params(model, layer_ids)
    optimizer = torch.optim.AdamW(params, lr=lr)

    history = []
    step = 0
    for forget_batch, retain_batch in zip(forget_batches, retain_batches):
        if step >= max_steps:
            break

        forget_inputs = tokenizer(
            forget_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len
        ).to(device)
        retain_inputs = tokenizer(
            retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len
        ).to(device)

        forget_activations = forward_with_cache(model, forget_inputs, module, no_grad=False)
        forget_loss = torch.nn.functional.mse_loss(forget_activations, control_vec.expand_as(forget_activations))

        retain_activations = forward_with_cache(model, retain_inputs, module, no_grad=False)
        with torch.no_grad():
            frozen_retain_activations = forward_with_cache(frozen_model, retain_inputs, frozen_module, no_grad=True)
        retain_loss = alpha * torch.nn.functional.mse_loss(retain_activations, frozen_retain_activations)

        loss = forget_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append({"step": step, "forget_loss": forget_loss.item(), "retain_loss": retain_loss.item()})
        print(f"step {step}: forget_loss={forget_loss.item():.4f} retain_loss={retain_loss.item():.4f}", file=sys.stderr)
        step += 1

    calibration = {
        "typical_ms": typical_ms,
        "typical_rms": typical_rms,
        "control_vec_norm": control_vec.norm().item(),
    }
    return history, calibration


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    set_seed(seed)

    rmu_cfg = config.get("rmu", {})
    base_model = args.base_model or config.get("model", {}).get("base")
    layer_id = args.layer_id if args.layer_id is not None else rmu_cfg.get("layer_id")
    if layer_id is None:
        raise ValueError("layer_id must be set via --layer-id or config rmu.layer_id")
    if args.layer_ids:
        layer_ids = [int(x) for x in args.layer_ids.split(",")]
    elif rmu_cfg.get("layer_ids"):
        layer_ids = list(rmu_cfg["layer_ids"])
    else:
        layer_ids = [layer_id - 2, layer_id - 1, layer_id]
    steering_coeff = args.steering_coeff if args.steering_coeff is not None else rmu_cfg.get("steering_coeff")
    alpha = args.alpha if args.alpha is not None else rmu_cfg.get("alpha")
    lr = args.lr if args.lr is not None else rmu_cfg.get("lr")
    max_steps = args.max_steps if args.max_steps is not None else rmu_cfg.get("max_steps")
    batch_size = args.batch_size if args.batch_size is not None else rmu_cfg.get("batch_size")
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else config.get("model", {}).get("max_seq_len", 512)
    dtype_name = config.get("model", {}).get("dtype", "bfloat16")
    device = resolve_device()

    print(f"Loading base model {base_model} ...", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=resolve_dtype(dtype_name))
    model = model.to(device)
    frozen_model = AutoModelForCausalLM.from_pretrained(base_model, dtype=resolve_dtype(dtype_name))
    frozen_model = frozen_model.to(device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad_(False)

    print(f"layer_id={layer_id} layer_ids={layer_ids} steering_coeff={steering_coeff} alpha={alpha} "
          f"lr={lr} max_steps={max_steps} batch_size={batch_size}", file=sys.stderr)

    forget_texts = load_text_iter(args.forget_dataset, args.forget_field, args.forget_split,
                                   slice_n=args.forget_slice, seed=seed)
    retain_texts = load_text_iter(args.retain_dataset, args.retain_field, args.retain_split,
                                   config=args.retain_config, slice_n=args.retain_slice, seed=seed)

    history, calibration = run_rmu(
        model=model,
        frozen_model=frozen_model,
        tokenizer=tokenizer,
        forget_texts=forget_texts,
        retain_texts=retain_texts,
        layer_id=layer_id,
        layer_ids=layer_ids,
        steering_coeff=steering_coeff,
        alpha=alpha,
        lr=lr,
        max_steps=max_steps,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        device=device,
    )

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    meta = {
        "base_model": base_model,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "layer_id": layer_id,
            "layer_ids": layer_ids,
            "steering_coeff": steering_coeff,
            "alpha": alpha,
            "lr": lr,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "seed": seed,
            "forget_dataset": args.forget_dataset,
            "forget_field": args.forget_field,
            "retain_dataset": args.retain_dataset,
            "retain_config": args.retain_config,
            "retain_field": args.retain_field,
            "library_versions": library_versions(),
        },
        "calibration": calibration,
        "history": history,
    }
    save_json(f"{args.out}/unlearn_run_meta.json", meta)
    print(f"Saved unlearned model to {args.out}")


if __name__ == "__main__":
    main()
