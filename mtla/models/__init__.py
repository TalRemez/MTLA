"""Model-family adapters.

Each adapter lives in its own module and self-registers with `@register_model("<key>")`; the
registry discovers them automatically (see `mtla.registry`). `get_model_adapter(key)` resolves a
config's `model:` field.
"""
from .base import ModelAdapter, Prediction
from ..registry import register_model, get_model_adapter, available_models

__all__ = ["ModelAdapter", "Prediction", "register_model", "get_model_adapter",
           "available_models"]
