"""Benchmark dataset adapters.

Each adapter lives in its own module and self-registers with `@register_dataset("<key>")`; the
registry discovers them automatically (see `mtla.registry`). `get_dataset_adapter(key)` resolves
a config's `dataset:` field.
"""

from mtla.data.base import DatasetAdapter
from mtla.registry import register_dataset, get_dataset_adapter, available_datasets

__all__ = [
    "DatasetAdapter",
    "register_dataset",
    "get_dataset_adapter",
    "available_datasets",
]
