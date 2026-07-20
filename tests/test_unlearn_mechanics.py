"""Structural smoke test for src/unlearn.py's RMU mechanics on a tiny synthetic
Qwen2 model (random weights, no download). Verifies: target params correctly
selected/frozen, hooks capture the right activations, one training step runs,
and ONLY the intended parameters change after the step.
"""
import copy
import sys
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.unlearn import get_target_params, forward_with_cache, run_rmu, TARGET_PARAM_NAME

torch.manual_seed(0)

config = Qwen2Config(
    vocab_size=200,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
)
model = Qwen2ForCausalLM(config)
frozen_model = copy.deepcopy(model)
for p in frozen_model.parameters():
    p.requires_grad_(False)

layer_id = 3
layer_ids = [1, 2, 3]

# --- test 1: get_target_params freezes everything else, selects the right params ---
before = {name: p.clone() for name, p in model.named_parameters()}
selected = get_target_params(model, layer_ids)
assert len(selected) == len(layer_ids), f"expected {len(layer_ids)} selected params, got {len(selected)}"
n_trainable = sum(p.requires_grad for p in model.parameters())
assert n_trainable == len(layer_ids), f"expected exactly {len(layer_ids)} trainable params, got {n_trainable}"
for layer_id_check in layer_ids:
    p = dict(model.model.layers[layer_id_check].named_parameters())[TARGET_PARAM_NAME]
    assert p.requires_grad, f"layer {layer_id_check} target param not trainable"
print("[OK] get_target_params: correct params selected and frozen elsewhere")

# --- test 2: forward_with_cache captures activations of the right shape ---
tokenizer_vocab_size = config.vocab_size
input_ids = torch.randint(0, tokenizer_vocab_size, (2, 10))
inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
module = model.model.layers[layer_id]
act = forward_with_cache(model, inputs, module, no_grad=True)
assert act.shape == (2, 10, config.hidden_size), f"unexpected activation shape {act.shape}"
print(f"[OK] forward_with_cache: activation shape {tuple(act.shape)} as expected")

# --- test 3: one training step runs and updates ONLY the selected params ---
class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        ids = torch.randint(1, tokenizer_vocab_size, (len(texts), 8))
        return FakeBatch(input_ids=ids, attention_mask=torch.ones_like(ids))

fake_tokenizer = FakeTokenizer()
forget_texts = iter([f"forget doc {i}" for i in range(20)])
retain_texts = iter([f"retain doc {i}" for i in range(20)])

before_state = {name: p.clone() for name, p in model.named_parameters()}
history, calibration = run_rmu(
    model=model,
    frozen_model=frozen_model,
    tokenizer=fake_tokenizer,
    forget_texts=forget_texts,
    retain_texts=retain_texts,
    layer_id=layer_id,
    layer_ids=layer_ids,
    steering_coeff=6.5,
    alpha=1200,
    lr=5e-5,
    max_steps=3,
    batch_size=2,
    max_seq_len=16,
    device="cpu",
)
assert len(history) == 3, f"expected 3 steps, got {len(history)}"
assert all("forget_loss" in h and "retain_loss" in h for h in history)
assert calibration["control_vec_norm"] > 0, "calibrated control vector should have positive norm"
print(f"[OK] calibration: typical_rms={calibration['typical_rms']:.4f} "
      f"control_vec_norm={calibration['control_vec_norm']:.4f}")

changed, unchanged_ok = [], []
for name, p in model.named_parameters():
    same = torch.equal(p, before_state[name])
    is_target = any(name == f"model.layers.{lid}.{TARGET_PARAM_NAME}" for lid in layer_ids)
    if is_target:
        changed.append((name, same))
    elif not same:
        unchanged_ok.append(name)

assert all(not same for _, same in changed), f"target params that DIDN'T change: {[n for n,s in changed if s]}"
assert len(unchanged_ok) == 0, f"non-target params that changed (frozen violation!): {unchanged_ok}"
print(f"[OK] run_rmu: {len(history)} steps ran, only target params changed ({[n for n,_ in changed]})")

print("\nALL MECHANICS TESTS PASSED")
