"""Model-family adapters. `get_model_adapter(key)` resolves a config's `model:` field."""
from .base import ModelAdapter, Prediction


def get_model_adapter(key: str) -> ModelAdapter:
    if key == "internvl":
        from .internvl import InternVLAdapter
        return InternVLAdapter()
    if key == "qwen3vl":
        from .qwen3vl import Qwen3VLAdapter
        return Qwen3VLAdapter()
    raise KeyError(f"unknown model adapter '{key}' (have: internvl, qwen3vl)")


__all__ = ["ModelAdapter", "Prediction", "get_model_adapter"]
