"""MTLA: Multi-Token Localized Attention.

A training-free, post-hoc confidence score for grounding predictions from multimodal LLMs.
It asks a simple question of the model's own attention: *when the model drew this box (or
predicted this time span), did it actually look inside the region it claims?* Predictions
whose tokens attend to evidence inside their proposal region are grounded; those that attend
elsewhere are likely hallucinations.

The pipeline is three stages — ``generate.py`` → ``extract.py`` → ``score.py``. The library
pieces:

    from mtla import mtla_localized_attention, reduce_band   # the MTLA math (eqs. 2-3, then 4)
    from mtla import auroc, coco_map                          # metrics
    from mtla import resolve                                  # (model, dataset) -> adapters
    model, dataset = resolve("qwen3vl", "coco")

Adapter modules are *discovered* (not eagerly imported) on the first ``resolve`` / ``available_*``
call, so a config's ``model:`` / ``dataset:`` keys resolve to classes without a central list.
"""
from .mtla_attn import mtla_localized_attention
from .score import ALL_LAYERS, DEFAULT_BAND, LAYER_BANDS, mtla_score, reduce_band
from .voting import iou, nms_fuse, tiou
from .metrics import auroc, coco_map, moment_retrieval, recall_at_iou
from .registry import (
    resolve,
    register_model,
    register_dataset,
    available_models,
    available_datasets,
)

__all__ = [
    "mtla_localized_attention",
    "DEFAULT_BAND", "ALL_LAYERS", "LAYER_BANDS", "reduce_band", "mtla_score",
    "nms_fuse", "iou", "tiou",
    "auroc", "coco_map", "moment_retrieval", "recall_at_iou",
    "resolve", "register_model", "register_dataset",
    "available_models", "available_datasets",
]

__version__ = "0.1.0"
