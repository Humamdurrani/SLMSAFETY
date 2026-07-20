"""Shared helpers: config loading, seeding, model loading, IO."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(config_path=None, overrides=None):
    """Load YAML config, deep-merge `overrides` (e.g. CLI flags) on top."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(base, overrides):
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def library_versions():
    import accelerate
    import datasets
    import transformers

    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
    }
    try:
        import lm_eval

        versions["lm_eval"] = lm_eval.__version__
    except ImportError:
        pass
    return versions


def load_model_and_tokenizer(model_id, dtype="bfloat16", device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=resolve_dtype(dtype),
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def append_experiment_index(row, index_path=None):
    """Append a row (timestamp, phase, config hash, key metric) to the experiments index."""
    index_path = Path(index_path) if index_path else REPO_ROOT / "results" / "experiments_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a") as f:
        f.write(json.dumps(row) + "\n")
