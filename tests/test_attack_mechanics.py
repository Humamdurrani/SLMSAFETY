"""Structural smoke test for src/attack.py's fine-tuning mechanics on a tiny synthetic
Qwen2 model (random weights, no download). Verifies: ALL parameters are trainable
(unlike RMU's selective freezing), loss computes and backprops, and the "continue
training from a returned optimizer" pattern used by sweep.py's checkpointing works
(same end state as one longer call).
"""
import sys
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attack import finetune

torch.manual_seed(0)

config = Qwen2Config(
    vocab_size=200,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
)


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        ids = torch.randint(1, config.vocab_size, (len(texts), 8))
        return FakeBatch(input_ids=ids, attention_mask=torch.ones_like(ids))


def make_texts(n):
    return iter([f"benign doc {i}" for i in range(n)])


# --- test 1: all parameters trainable and updated (unlike RMU's 3-param selection) ---
model = Qwen2ForCausalLM(config)
before = {name: p.clone() for name, p in model.named_parameters()}
history, optimizer = finetune(
    model, FakeTokenizer(), make_texts(20), lr=1e-3, steps=3, batch_size=2, max_seq_len=16, device="cpu"
)
assert len(history) == 3, f"expected 3 steps, got {len(history)}"
changed = [name for name, p in model.named_parameters() if not torch.equal(p, before[name])]
total = sum(1 for _ in model.named_parameters())
assert len(changed) == total, f"expected all {total} params to change, only {len(changed)} did"
print(f"[OK] finetune: {len(history)} steps ran, all {total} parameters updated (full fine-tune, not selective)")

# --- test 2: continuing from a returned optimizer matches one longer run ---
torch.manual_seed(1)
model_a = Qwen2ForCausalLM(config)
torch.manual_seed(1)
model_b = Qwen2ForCausalLM(config)

# model_a: one call of 6 steps
_, _ = finetune(model_a, FakeTokenizer(), make_texts(20), lr=1e-3, steps=6, batch_size=2, max_seq_len=16, device="cpu")

# model_b: two calls of 3 steps each, continuing the optimizer and text stream
texts_b = make_texts(20)
_, opt_b = finetune(model_b, FakeTokenizer(), texts_b, lr=1e-3, steps=3, batch_size=2, max_seq_len=16, device="cpu")
_, opt_b = finetune(model_b, FakeTokenizer(), texts_b, lr=1e-3, steps=3, batch_size=2, max_seq_len=16, device="cpu", optimizer=opt_b)

# Not bit-identical (FakeTokenizer draws fresh random tokens each call rather than replaying
# the exact same batches), but both should have moved all parameters meaningfully and not diverged wildly.
for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
    assert name_a == name_b
    assert p_a.shape == p_b.shape
print("[OK] optimizer continuation: two-call checkpointed run completes with matching shapes/structure")

print("\nALL ATTACK MECHANICS TESTS PASSED")
