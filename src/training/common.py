import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file), config_path


def resolve_from_config(config_path, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def edit_distance(source, target):
    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, 1):
        current = [i]
        for j, target_item in enumerate(target, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (source_item != target_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def cer(predictions, references):
    edits = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        edits += edit_distance(prediction, reference)
        total += max(1, len(reference))
    return edits / total


def wer(predictions, references):
    edits = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        pred_words = prediction.split()
        ref_words = reference.split()
        edits += edit_distance(pred_words, ref_words)
        total += max(1, len(ref_words))
    return edits / total


def chrf_score(predictions, references):
    try:
        import sacrebleu

        return sacrebleu.corpus_chrf(predictions, [references]).score
    except Exception:
        return 0.0


def precision_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def use_amp(name):
    return name in {"bf16", "fp16"} and torch.cuda.is_available()


def total_steps(num_examples, batch_size, grad_accum, epochs):
    steps_per_epoch = math.ceil(num_examples / batch_size / max(1, grad_accum))
    return steps_per_epoch * epochs


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
