"""Benchmark dataset adapters. `get_dataset_adapter(key)` resolves a config's `dataset:` field."""
from .base import DatasetAdapter


def get_dataset_adapter(key: str) -> DatasetAdapter:
    if key == "coco":
        from .coco import CocoDataset
        return CocoDataset()
    if key == "qvhighlights":
        from .qvhighlights import QVHighlightsDataset
        return QVHighlightsDataset()
    if key == "charades":
        from .charades import CharadesDataset
        return CharadesDataset()
    raise KeyError(f"unknown dataset adapter '{key}' (have: coco, qvhighlights, charades)")


__all__ = ["DatasetAdapter", "get_dataset_adapter"]
