"""Shared type aliases and ``TypedDict`` contracts for the dict shapes passed between stages.

These give the loosely-typed dicts that flow through the pipeline (generation records, feature-shard
records, the per-prediction objects, and the extraction ``ctx`` / ``build_inputs`` payloads) an
explicit, checkable shape. They are documentation-as-types: nothing enforces them at run time, but
``mypy`` uses them to catch a mistyped key or a wrong value type.

Dicts whose exact key set varies by model/task (``Ctx``, ``BuildInputs``, ``Meta``) are declared
``total=False`` so an adapter may include only the keys its task needs.
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict, Union

import numpy as np

# A grounding region: a box ``[x1, y1, x2, y2]`` in [0,1000] (image) or a span ``[t0, t1]`` (video).
Region = list[float]
# An item id: image id (int) or a composite query id (str, e.g. Charades "vid.mp4::caption").
ItemId = Union[int, str]
# Overlap function: iou (boxes) or tiou (spans), ``(region, region) -> float``.
OverlapFn = Callable[[Region, Region], float]


class GTRegion(TypedDict):
    """One ground-truth region + label (label empty for video spans)."""
    region: Region
    label: str


class GenRecord(TypedDict, total=False):
    """Uniform generation record written by ``dataset.gen_record`` (predictions.json entries).

    ``id`` / ``prompt`` / ``response`` / ``gt`` / ``extra`` are always present; ``idx`` is added by
    the generate stage so workers can merge in order.
    """
    id: ItemId
    prompt: str
    response: str
    gt: list[GTRegion]
    extra: dict[str, Any]
    idx: int


class TokenRange(TypedDict):
    """One prediction's response tokens Q_p, located by char offset (from ``query_tokens``)."""
    first_label_tok: int | None
    label_toks: list[int]
    coord_toks: list[int]


class PredObject(TypedDict):
    """Per-prediction saved object in a feature shard (the score stage's atom)."""
    pred_idx: int
    region: Region
    label: str
    is_hallucinated: bool
    extracted: bool
    local_attention: np.ndarray
    first_digit: np.ndarray


class ItemRecord(TypedDict):
    """Top-level feature-shard record: one item, its GT, and every prediction's object."""
    id: ItemId
    gt: list[GTRegion]
    extra: dict[str, Any]
    n_predictions: int
    n_extracted: int
    objects: list[PredObject]


# The extraction context built by ``load_for_extract`` and read by ``compute_mtla`` + callbacks.
# Keys vary by model/task (e.g. video adds preprocess/multi), so it is not total.
class Ctx(TypedDict, total=False):
    model: Any                 # the HF model (typed Any to avoid a hard torch/transformers dep here)
    proc: Any                  # the processor
    tokenizer: Any
    state: Any                 # mtla.mtla_attn.CaptureState
    device: str
    task: str
    n_layers: int
    n_heads: int
    preprocess: dict[str, Any]
    multi: bool
    # model-specific pad-token ids
    pad_id: int
    image_pad_id: int


# The payload ``build_inputs`` returns (or None to skip): the parsed predictions + everything the
# single captured forward and the per-prediction reduction need. Keys vary by model/task.
class BuildInputs(TypedDict, total=False):
    prompt_ids: Any            # 1-D LongTensor
    response: str
    modality_idx_l: list[int]
    predictions: list          # list[Prediction] (avoid importing to keep this leaf module light)
    hallu_flags: list[bool]
    meta: dict[str, Any]
    inputs: Any                # processor BatchFeature (Qwen)
    pixel_values: Any          # tensor (InternVL)
