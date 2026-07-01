"""Adapter registry: resolve a (model, dataset) pair into validated adapters.

Models and datasets self-register with a decorator, so adding one never means editing this file:

    from mtla.registry import register_model
    @register_model("qwen3vl")
    class Qwen3VLAdapter(ModelAdapter):
        ...

`resolve("qwen3vl", "coco")` returns `(model_adapter, dataset_adapter)` after checking the
dataset's task family is one the model supports — an unsupported pairing fails fast with a clear
message instead of deep inside a stage. This is the single entry point the stage scripts use to
turn a config's `model:` / `dataset:` keys into the objects that carry every task-specific piece
(parse, the vLLM request builder, the MTLA extraction callbacks).

Adapters are discovered lazily: the first lookup imports every submodule under `mtla/models/`
and `mtla/data/`, which runs their `@register_*` decorators. Keeping discovery lazy means
`import mtla` (and the CPU-only `score` path) does not eagerly import torch/transformers.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .models.base import ModelAdapter
    from .data.base import DatasetAdapter

_MODELS: dict[str, type] = {}
_DATASETS: dict[str, type] = {}


def register_model(key: str) -> Callable[[type], type]:
    """Class decorator: register a `ModelAdapter` subclass under `key` (e.g. "qwen3vl")."""
    def deco(cls: type) -> type:
        existing = _MODELS.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(f"duplicate model key {key!r}: {existing.__name__} vs {cls.__name__}")
        _MODELS[key] = cls
        return cls
    return deco


def register_dataset(key: str) -> Callable[[type], type]:
    """Class decorator: register a `DatasetAdapter` subclass under `key` (e.g. "coco")."""
    def deco(cls: type) -> type:
        existing = _DATASETS.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(f"duplicate dataset key {key!r}: {existing.__name__} vs {cls.__name__}")
        _DATASETS[key] = cls
        return cls
    return deco


def _discover(package_name: str) -> None:
    """Import every submodule of `package_name` so its `@register_*` decorators run."""
    package = importlib.import_module(package_name)
    for info in pkgutil.iter_modules(package.__path__):
        if info.name == "base" or info.name.startswith("_"):
            continue
        importlib.import_module(f"{package_name}.{info.name}")


def _missing(kind: str, key: str, available: dict, package: str, base: str) -> ValueError:
    have = ", ".join(sorted(available)) or "(none)"
    return ValueError(
        f"unknown {kind} {key!r}. Available {kind}s: {have}.\n"
        f"To add one: create mtla/{package}/<name>.py, subclass {base}, decorate it with "
        f"@register_{kind}('<name>'), and pass <name> in your config. See docs/EXTENDING.md.")


def get_model_adapter(key: str) -> "ModelAdapter":
    """Instantiate the registered model adapter for `key`, else raise a helpful ValueError."""
    _discover("mtla.models")
    if key not in _MODELS:
        raise _missing("model", key, _MODELS, "models", "ModelAdapter")
    return _MODELS[key]()


def get_dataset_adapter(key: str) -> "DatasetAdapter":
    """Instantiate the registered dataset adapter for `key`, else raise a helpful ValueError."""
    _discover("mtla.data")
    if key not in _DATASETS:
        raise _missing("dataset", key, _DATASETS, "data", "DatasetAdapter")
    return _DATASETS[key]()


def available_models() -> list[str]:
    _discover("mtla.models")
    return sorted(_MODELS)


def available_datasets() -> list[str]:
    _discover("mtla.data")
    return sorted(_DATASETS)


def resolve(model_key: str, dataset_key: str) -> tuple["ModelAdapter", "DatasetAdapter"]:
    """Return (model_adapter, dataset_adapter) for a valid pairing, else raise ValueError."""
    model = get_model_adapter(model_key)
    dataset = get_dataset_adapter(dataset_key)
    if model.tasks and dataset.task not in model.tasks:
        raise ValueError(
            f"model {model_key!r} does not support dataset {dataset_key!r}: dataset task is "
            f"{dataset.task!r}, model supports {list(model.tasks)}")
    return model, dataset
