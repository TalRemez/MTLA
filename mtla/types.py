"""Shared type aliases and ``TypedDict`` contracts for the dict shapes passed between stages.

These give the loosely-typed dicts that flow through the pipeline (generation records, feature-shard
records, the per-prediction objects, and the extraction ``ctx`` / ``build_extraction_inputs`` payloads) an
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
# Voting output: per ``(item id, label)`` group, its kept regions ranked by fused score.
FusedGroups = dict[tuple[ItemId, str], list[tuple[Region, float]]]


class GTRegion(TypedDict):
    """A single ground-truth region paired with its class label.

    The unit of ground truth carried through generation and feature-shard records and
    consumed by the score stage's metrics. ``region`` is a box ``[x1, y1, x2, y2]`` for
    image tasks or a span ``[t0, t1]`` for video; ``label`` is the class name for
    detection and is left empty for video grounding, where a query has no per-region
    label.
    """

    region: Region
    label: str


class GenRecord(TypedDict, total=False):
    """One item's generation result, as written to ``predictions.json``.

    The uniform hand-off from the generate stage to extract: built by
    ``dataset.gen_record`` and later re-read (and parsed into predictions) by
    ``compute_mtla``. Declared ``total=False`` only because ``idx`` is optional, but in
    practice ``id`` / ``prompt`` / ``response`` / ``gt`` / ``extra`` are always present.
    ``gt`` is the item's ground-truth regions, ``extra`` carries task-specific payload
    (e.g. media paths or preprocessing hints), and ``idx`` is stamped by the generate
    stage so parallel workers can merge their outputs back into item order.
    """

    id: ItemId
    prompt: str
    response: str
    gt: list[GTRegion]
    extra: dict[str, Any]
    idx: int


class TokenRange(TypedDict):
    """The response-token positions Q_p that belong to one prediction.

    Returned per prediction by an adapter's ``locate_proposal_tokens`` and consumed by
    ``compute_mtla`` to pull that prediction's rows out of the captured attention. The
    indices are into the tokenized response and are found by char offset. ``label_toks``
    covers the label span and ``coord_toks`` the coordinate/digit span;
    ``first_label_tok`` is the first label token (``None`` when the label has no
    locatable token), used for the label-anchored slot. The first entry of
    ``coord_toks`` is the ``x1`` token used for the ``first_digit`` reduction.
    """

    first_label_tok: int | None
    label_toks: list[int]
    coord_toks: list[int]


class PredObject(TypedDict):
    """One prediction's saved MTLA features inside a feature shard.

    The atom the score stage reduces and votes over: written by ``compute_mtla`` and
    read by ``mtla.evaluate``. ``region`` and ``label`` are the predicted grounding,
    ``is_hallucinated`` is the detection label for AUROC, and ``extracted`` flags
    whether attention was actually captured for this prediction (``False`` predictions
    are skipped when computing AUROC). ``local_attention`` and ``first_digit`` are the
    two ``[L, H]`` reductions (over all Q_p tokens, and over the first coordinate token
    respectively) that the score stage collapses to a scalar over the layer band.
    """

    pred_idx: int
    region: Region
    label: str
    is_hallucinated: bool
    extracted: bool
    local_attention: np.ndarray
    first_digit: np.ndarray


class ItemRecord(TypedDict):
    """The top-level feature-shard record for one item.

    What ``compute_mtla`` returns and the extract stage serializes into a shard: it
    bundles the item id, its ground truth, task-specific ``extra``, and one
    ``PredObject`` per prediction. ``n_predictions`` is how many predictions the
    response parsed to and ``n_extracted`` how many had locatable tokens and were
    actually reduced (so ``n_extracted <= n_predictions``); ``objects`` holds the
    extracted ones.
    """

    id: ItemId
    gt: list[GTRegion]
    extra: dict[str, Any]
    n_predictions: int
    n_extracted: int
    objects: list[PredObject]


class ScoredCand(TypedDict):
    """One flattened, scored prediction — the atom the score stage votes over.

    Produced by ``score.load_candidates`` (one per prediction per rollout) and consumed by
    ``mtla.evaluate.hallucination_auroc`` (reads ``score`` / ``hallu`` / ``extracted`` / ``seed``)
    and by ``mtla.voting.vote`` (reads ``id`` / ``label`` / ``region`` / ``score`` / ``seed``).
    ``score`` is the scalar MTLA value after band reduction, ``hallu`` the detection label,
    ``extracted`` whether attention was captured for this prediction, and ``seed`` the rollout it
    came from.
    """

    id: ItemId
    label: str
    region: Region
    score: float
    hallu: bool
    extracted: bool
    seed: int


# The extraction context built by ``load_for_extract`` and read by ``compute_mtla`` + callbacks.
# Keys vary by model/task (e.g. video adds preprocess/multi), so it is not total.
class Ctx(TypedDict, total=False):
    """The extraction context shared across one extract-stage worker.

    Built once by ``load_for_extract`` (model, processor, tokenizer, attention-capture
    state, device, and geometry) and threaded into ``compute_mtla`` and the adapter
    callbacks for every item. Declared ``total=False`` because the key set varies by
    model and task: video runs add ``preprocess`` and ``multi``, and the pad-token ids
    are model-specific. The heavy objects (``model``, ``proc``, ``state``) are typed
    ``Any`` so this leaf module needs no hard torch/transformers dependency.
    """

    model: Any  # the HF model (typed Any to avoid a hard torch/transformers dep here)
    proc: Any  # the processor
    tokenizer: Any
    state: Any  # mtla.mtla_attn.CaptureState
    device: str
    n_layers: int
    n_heads: int
    preprocess: dict[str, Any]
    multi: bool
    # model-specific pad-token ids
    pad_id: int
    image_pad_id: int


# The payload ``build_extraction_inputs`` returns (or None to skip): the parsed predictions + everything the
# single captured forward and the per-prediction reduction need. Keys vary by model/task.
class BuildInputs(TypedDict, total=False):
    """The per-item payload an adapter's ``build_extraction_inputs`` hands to ``compute_mtla``.

    Everything the single captured forward pass and the per-prediction reduction need
    for one item: the teacher-forcing token ids, the raw ``response`` string, the
    parsed ``predictions`` with their ``hallu_flags`` and ``meta``, and the
    modality-token column indices (``modality_idx_l``) the reduction reads.
    ``build_extraction_inputs`` may instead return ``None`` to skip an item. It is
    ``total=False`` because the model-specific forward payload differs: Qwen carries a
    ``inputs`` BatchFeature while InternVL carries ``pixel_values``. Tensor-typed fields
    are ``Any`` to keep this module free of a hard torch dependency.
    """

    prompt_ids: Any  # 1-D LongTensor
    response: str
    modality_idx_l: list[int]
    predictions: (
        list  # list[Prediction] (avoid importing to keep this leaf module light)
    )
    hallu_flags: list[bool]
    meta: dict[str, Any]
    inputs: Any  # processor BatchFeature (Qwen)
    pixel_values: Any  # tensor (InternVL)
