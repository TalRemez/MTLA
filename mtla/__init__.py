"""MTLA: Multi-Token Localized Attention.

A training-free, post-hoc confidence score for grounding predictions from multimodal LLMs.
It asks a simple question of the model's own attention: *when the model drew this box (or
predicted this time span), did it actually look inside the region it claims?* Predictions
whose tokens attend to evidence inside their proposal region are grounded; those that attend
elsewhere are likely hallucinations.

Typical use, starting from the attention records the extract stage writes:

    from mtla import mtla_score, auroc_from_records

    auroc_mtla = auroc_from_records(objects)                           # MTLA (local_attention)
    auroc_fd   = auroc_from_records(objects, signal="first_digit")     # first-digit ablation

Resolve a (model, dataset) pair to its adapters. ``import mtla`` itself stays light (the scoring
helpers above pull in only numpy); the adapter modules are imported lazily on the first
``resolve`` / ``available_*`` call:

    from mtla import resolve
    model, dataset = resolve("qwen3vl", "coco")
"""
from .score import (
    ALL_LAYERS,
    DEFAULT_BAND,
    LAYER_BANDS,
    mtla_score,
    reduce_band,
)
from .mask import (
    bbox_to_internvl_token_indices,
    bbox_to_patch_indices,
    span_to_token_indices,
)
from .voting import iou, nms_fuse, tiou
from .eval import auroc, auroc_from_records, coco_map
from .registry import (
    resolve,
    register_model,
    register_dataset,
    available_models,
    available_datasets,
)

__all__ = [
    "DEFAULT_BAND", "ALL_LAYERS", "LAYER_BANDS",
    "reduce_band", "mtla_score",
    "bbox_to_patch_indices", "bbox_to_internvl_token_indices", "span_to_token_indices",
    "nms_fuse", "iou", "tiou",
    "auroc", "auroc_from_records", "coco_map",
    "resolve", "register_model", "register_dataset",
    "available_models", "available_datasets",
]

__version__ = "0.1.0"
