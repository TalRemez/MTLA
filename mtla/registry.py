"""Factory: resolve a (model, dataset) pair into validated adapters.

`resolve("internvl", "coco")` returns `(model_adapter, dataset_adapter)` after checking the
dataset's task family is one the model supports — so an unsupported pairing fails fast with a
clear message rather than deep inside a stage. This is the single entry point `run.py` uses to
turn a config's `model:` / `dataset:` keys into the objects that carry every task-specific
function (parse, region_mask, mtla_slot/svar_slot, stage scripts, score).
"""
from __future__ import annotations

from .models import get_model_adapter
from .data import get_dataset_adapter


def resolve(model_key: str, dataset_key: str):
    """Return (model_adapter, dataset_adapter) for a valid pairing, else raise ValueError."""
    model = get_model_adapter(model_key)
    dataset = get_dataset_adapter(dataset_key)
    if model.tasks and dataset.task not in model.tasks:
        raise ValueError(
            f"model '{model_key}' does not support dataset '{dataset_key}': "
            f"dataset task is '{dataset.task}', model supports {list(model.tasks)}")
    return model, dataset
