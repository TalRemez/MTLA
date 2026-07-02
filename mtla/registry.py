"""Adapter registry: turn a config's `model:` / `dataset:` keys into adapter instances.

Adapters self-register with `@register_model` / `@register_dataset` and are discovered
lazily on first lookup (importing every submodule under `mtla/models/` and `mtla/data/`
runs their decorators). Lazy so `import mtla` and the CPU-only score path don't pull in
torch/transformers. `resolve(model_key, dataset_key)` is the single entry point.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mtla.models.base import ModelAdapter
    from mtla.data.base import DatasetAdapter


class _Registry:
    """One key -> adapter-class table, self-populated by scanning its package on first use."""

    def __init__(self, kind: str, package: str, base: str):
        # kind: "model" | "dataset" (for messages + the @register_<kind> hint)
        self.kind = kind
        self.package = package  # "mtla.models" | "mtla.data"
        self.base = base  # "ModelAdapter" | "DatasetAdapter"
        self._entries: dict[str, type] = {}
        self._discovered = False

    def register(self, key: str) -> Callable[[type], type]:
        """Class decorator that binds `key` to the decorated adapter (self-registration)."""

        def deco(cls: type) -> type:
            existing = self._entries.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"duplicate {self.kind} key {key!r}: "
                    f"{existing.__name__} vs {cls.__name__}"
                )
            self._entries[key] = cls
            return cls

        return deco

    def _discover(self) -> None:
        # Import every submodule (except base/_*) once, so their @register_* decorators run.
        if self._discovered:
            return
        pkg = importlib.import_module(self.package)
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name == "base" or info.name.startswith("_"):
                continue
            importlib.import_module(f"{self.package}.{info.name}")
        self._discovered = True

    def get(self, key: str) -> object:
        """Instantiate the adapter bound to `key`, or raise a helpful error listing valid keys."""
        self._discover()
        if key not in self._entries:
            have = ", ".join(sorted(self._entries)) or "(none)"
            raise ValueError(
                f"unknown {self.kind} {key!r}. Available {self.kind}s: {have}.\n"
                f"To add one: create mtla/{self.package.split('.')[-1]}/<name>.py, subclass "
                f"{self.base}, decorate it with @register_{self.kind}('<name>'), and pass "
                f"<name> in your config. See docs/EXTENDING.md."
            )
        return self._entries[key]()

    def keys(self) -> list[str]:
        """Every registered key, sorted (for error messages / --help)."""
        self._discover()
        return sorted(self._entries)


_MODELS = _Registry("model", "mtla.models", "ModelAdapter")
_DATASETS = _Registry("dataset", "mtla.data", "DatasetAdapter")

register_model = _MODELS.register
register_dataset = _DATASETS.register
available_models = _MODELS.keys
available_datasets = _DATASETS.keys


def get_model_adapter(key: str) -> "ModelAdapter":
    return _MODELS.get(key)  # type: ignore[return-value]


def get_dataset_adapter(key: str) -> "DatasetAdapter":
    return _DATASETS.get(key)  # type: ignore[return-value]


def resolve(
    model_key: str, dataset_key: str
) -> tuple["ModelAdapter", "DatasetAdapter"]:
    """Resolve a config's (model, dataset) keys into their adapter instances."""
    return get_model_adapter(model_key), get_dataset_adapter(dataset_key)
