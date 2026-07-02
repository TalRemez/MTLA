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
    from mtla.models.base import ModelAdapter
    from mtla.data.base import DatasetAdapter

_MODELS: dict[str, type] = {}
_DATASETS: dict[str, type] = {}


def register_model(key: str) -> Callable[[type], type]:
    """Register a ``ModelAdapter`` subclass under a config key.

    Returns a class decorator that records the class in the module-level model
    registry, so ``model:`` in a config can later resolve to it. Self-registration
    means adding a model never requires editing this file.

    Args:
        key: the config lookup name for the adapter (e.g. ``"qwen3vl"``).

    Returns:
        A class decorator that registers the decorated class and returns it
        unchanged.

    Raises:
        ValueError: if ``key`` is already bound to a different class (a duplicate
            registration).
    """

    def deco(cls: type) -> type:
        existing = _MODELS.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"duplicate model key {key!r}: {existing.__name__} vs {cls.__name__}"
            )
        _MODELS[key] = cls
        return cls

    return deco


def register_dataset(key: str) -> Callable[[type], type]:
    """Register a ``DatasetAdapter`` subclass under a config key.

    Returns a class decorator that records the class in the module-level dataset
    registry, so ``dataset:`` in a config can later resolve to it. Self-registration
    means adding a dataset never requires editing this file.

    Args:
        key: the config lookup name for the adapter (e.g. ``"coco"``).

    Returns:
        A class decorator that registers the decorated class and returns it
        unchanged.

    Raises:
        ValueError: if ``key`` is already bound to a different class (a duplicate
            registration).
    """

    def deco(cls: type) -> type:
        existing = _DATASETS.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"duplicate dataset key {key!r}: {existing.__name__} vs {cls.__name__}"
            )
        _DATASETS[key] = cls
        return cls

    return deco


def _discover(package_name: str) -> None:
    """Import every submodule of a package so its ``@register_*`` decorators run.

    Walks the package's modules and imports each one, skipping ``base`` and any
    underscore-prefixed module. Importing a module executes its decorators, which is
    what actually populates the registries. Called lazily on the first lookup so that
    ``import mtla`` (and the CPU-only score path) does not eagerly pull in
    torch/transformers.

    Args:
        package_name: the importable package to scan (e.g. ``"mtla.models"`` or
            ``"mtla.data"``).
    """
    package = importlib.import_module(package_name)
    for info in pkgutil.iter_modules(package.__path__):
        if info.name == "base" or info.name.startswith("_"):
            continue
        importlib.import_module(f"{package_name}.{info.name}")


def _missing(
    kind: str, key: str, available: dict, package: str, base: str
) -> ValueError:
    """Build a helpful ``ValueError`` for an unknown model/dataset key.

    Formats an error that lists the registered keys and spells out the steps to add a
    new adapter, so an unsupported config key fails fast with an actionable message.

    Args:
        kind: the adapter kind, ``"model"`` or ``"dataset"`` (used in the message and
            the ``@register_<kind>`` hint).
        key: the requested key that was not found.
        available: the registry mapping to list registered keys from.
        package: the subpackage a new adapter file goes in (``"models"`` or
            ``"data"``).
        base: the base class a new adapter should subclass
            (``"ModelAdapter"`` / ``"DatasetAdapter"``).

    Returns:
        A ``ValueError`` (constructed, not raised) with the formatted message.
    """
    have = ", ".join(sorted(available)) or "(none)"
    return ValueError(
        f"unknown {kind} {key!r}. Available {kind}s: {have}.\n"
        f"To add one: create mtla/{package}/<name>.py, subclass {base}, decorate it with "
        f"@register_{kind}('<name>'), and pass <name> in your config. See docs/EXTENDING.md."
    )


def get_model_adapter(key: str) -> "ModelAdapter":
    """Instantiate the registered model adapter for a key.

    Triggers lazy discovery of ``mtla.models`` (so every adapter registers) and then
    constructs a fresh instance of the class bound to ``key``.

    Args:
        key: the model config key to resolve (e.g. ``"qwen3vl"``).

    Returns:
        A new ``ModelAdapter`` instance for ``key``.

    Raises:
        ValueError: if no model is registered under ``key`` (message lists the
            available keys and how to add one).
    """
    _discover("mtla.models")
    if key not in _MODELS:
        raise _missing("model", key, _MODELS, "models", "ModelAdapter")
    return _MODELS[key]()


def get_dataset_adapter(key: str) -> "DatasetAdapter":
    """Instantiate the registered dataset adapter for a key.

    Triggers lazy discovery of ``mtla.data`` (so every adapter registers) and then
    constructs a fresh instance of the class bound to ``key``.

    Args:
        key: the dataset config key to resolve (e.g. ``"coco"``).

    Returns:
        A new ``DatasetAdapter`` instance for ``key``.

    Raises:
        ValueError: if no dataset is registered under ``key`` (message lists the
            available keys and how to add one).
    """
    _discover("mtla.data")
    if key not in _DATASETS:
        raise _missing("dataset", key, _DATASETS, "data", "DatasetAdapter")
    return _DATASETS[key]()


def available_models() -> list[str]:
    """List every registered model key.

    Runs lazy discovery of ``mtla.models`` first, so the result reflects all adapters
    on disk. Handy for building error messages and ``--help`` output.

    Returns:
        The registered model keys, sorted alphabetically.
    """
    _discover("mtla.models")
    return sorted(_MODELS)


def available_datasets() -> list[str]:
    """List every registered dataset key.

    Runs lazy discovery of ``mtla.data`` first, so the result reflects all adapters on
    disk. Handy for building error messages and ``--help`` output.

    Returns:
        The registered dataset keys, sorted alphabetically.
    """
    _discover("mtla.data")
    return sorted(_DATASETS)


def resolve(
    model_key: str, dataset_key: str
) -> tuple["ModelAdapter", "DatasetAdapter"]:
    """Resolve a config's ``(model, dataset)`` keys into validated adapters.

    The single entry point the stage scripts use: it instantiates both adapters and
    checks the dataset's task family is one the model supports, so an unsupported
    pairing fails fast here rather than deep inside a stage. If the model declares no
    tasks, any dataset is accepted.

    Args:
        model_key: the model config key (e.g. ``"qwen3vl"``).
        dataset_key: the dataset config key (e.g. ``"coco"``).

    Returns:
        A ``(model_adapter, dataset_adapter)`` tuple ready to drive a stage.

    Raises:
        ValueError: if either key is unregistered, or the dataset's task is not in the
            model's supported tasks.
    """
    model = get_model_adapter(model_key)
    dataset = get_dataset_adapter(dataset_key)
    if model.tasks and dataset.task not in model.tasks:
        raise ValueError(
            f"model {model_key!r} does not support dataset {dataset_key!r}: dataset task is "
            f"{dataset.task!r}, model supports {list(model.tasks)}"
        )
    return model, dataset
