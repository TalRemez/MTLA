"""Dataset adapter interface + shared machinery.

A dataset adapter is **declarative**: it says what the benchmark *is* (how to load items, the
prompt, the ground truth, the uniform generation record) and *declares* how it should be scored
(which MTLA signal, overlap function, candidate selection, and metric). It does **no** computation
— shard loading, band reduction, voting, NMS, and metric evaluation all live in the ``evaluate.py`` stage,
``mtla.voting``, and ``mtla.metrics``. This keeps every adapter small and identical in shape.

Scoring descriptors (read by the ``evaluate.py`` stage + ``mtla.voting.vote``):
  * ``signal``  — which saved ``[L, H]`` array to reduce: ``local_attention`` (images) or
    ``first_digit`` (video, the validated choice).
  * ``overlap`` — ``iou`` (boxes) or ``tiou`` (temporal spans), used for both voting and metrics.
  * ``select``  — ``fuse`` (keep every NMS-fused cluster; detection, multi-window) or ``argmax``
    (keep the single best; single-span grounding — voted with ``top_k=1``).
  * ``metric``  — names a pure computer in ``mtla.metrics`` (``coco_map`` | ``moment_retrieval`` |
    ``recall_at_iou``).

A ``task`` family (``image_det`` | ``video_span``) tells the model adapter which parser / region
mask to use, so any valid ``(model x dataset)`` pair runs from a config.
"""

from __future__ import annotations

import glob
from typing import TYPE_CHECKING, Any

import torch

from mtla.types import GenRecord, GTRegion, ItemRecord

if TYPE_CHECKING:
    from mtla.config import RunConfig


def load_shards(features_dir: str) -> list["ItemRecord"]:
    """Load and concatenate all ``shard*.pt`` records under one seed's feature dir.

    The extract stage writes features as ``<features>/seedK/shard*.pt``; this reads
    every shard for a single seed and flattens them into one list, preserving shard
    order (shards are sorted lexicographically by filename).

    Args:
        features_dir: Path to a single seed directory, e.g. ``<features>/seed0``,
            containing one or more ``shard*.pt`` files. Each file is a pickled
            ``list[ItemRecord]`` loaded onto CPU.

    Returns:
        The concatenated ``list[ItemRecord]`` across all shards, in shard-sorted
        order. Empty if the directory contains no matching shards.
    """
    recs: list = []
    for sp in sorted(glob.glob(f"{features_dir}/shard*.pt")):
        recs.extend(torch.load(sp, weights_only=False, map_location="cpu"))
    return recs


def print_metrics(name: str, metrics: dict, indent: str = "  ") -> None:
    """Pretty-print a (possibly nested) metrics dict to stdout.

    Renders the metrics mapping the ``evaluate.py`` stage assembles. Top-level float
    values print as ``key = value`` (4 decimals); nested dict values (e.g. a
    ``coco_map`` sub-dict) print on one line as space-joined ``ik=iv`` pairs.

    Args:
        name: Label for the header line, printed as ``[name] results:`` (typically
            the benchmark or run name).
        metrics: The score-stage metrics mapping. Values may be floats, plain
            scalars (int/str), or nested dicts of the same.
        indent: Leading whitespace prepended to each metric line. Defaults to two
            spaces.

    Returns:
        None. Output is written to stdout.
    """
    print(f"[{name}] results:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            inner = "  ".join(
                f"{ik}={iv:.4f}" if isinstance(iv, float) else f"{ik}={iv}"
                for ik, iv in v.items()
            )
            print(f"{indent}{k}: {inner}")
        elif isinstance(v, float):
            print(f"{indent}{k} = {v:.4f}")
        else:
            print(f"{indent}{k} = {v}")


class DatasetAdapter:
    """Declarative interface for one grounding benchmark.

    A subclass describes what a benchmark *is* and *declares* how it should be
    scored; it performs no scoring itself. It implements four small methods
    (:meth:`load_items`, :meth:`prompt`, :meth:`ground_truth`, :meth:`gen_record`;
    video adapters add ``video_path``) and sets a handful of class attributes that
    the ``evaluate.py`` stage reads to route the prediction through the right
    signal, overlap, selection, and metric. This keeps every adapter uniform.

    Class attributes:
        name: Registry key for the benchmark (e.g. ``"coco"``).
        task: Task family. ``"image_det"`` (box detection) or ``"video_span"``
            (temporal grounding). Tells the model adapter which response parser and
            region mask to use, so any valid ``(model x dataset)`` pair runs.
        signal: Which saved ``[L, H]`` (layers x heads) attention array to reduce.
            ``"local_attention"`` for images; ``"first_digit"`` for video (the
            validated choice: attention on the span's first coordinate digit).
        overlap: Overlap function for both voting and metrics. ``"iou"`` for boxes,
            ``"tiou"`` for temporal spans.
        select: How candidates from the N rollouts are combined. ``"fuse"`` pools
            them with NMS (detection and multi-segment retrieval); ``"argmax"``
            keeps the single highest-MTLA candidate (single-span grounding).
        metric: Name of a pure computer in ``mtla.metrics`` used for the benchmark
            score: ``"coco_map"`` | ``"moment_retrieval"`` | ``"recall_at_iou"``.
        greedy_seed0: Whether rollout 0 is a greedy (temperature 0) anchor in the
            N=16 self-consistency recipe (greedy seed 0 + N-1 stochastic). Video
            benchmarks set ``True``; a config ``generate.greedy_seed0`` may override.
        gen_strategy: Execution strategy for ``generate.py``. ``"pooled"`` uses an
            async multi-engine vLLM pool (throughput on many small requests, e.g.
            5k COCO images); ``"sharded"`` runs one engine per GPU (heavy per-item
            work, e.g. video clips).
    """

    name: str = ""
    task: str = ""

    # ---- scoring descriptors (see module docstring; read by the evaluate.py stage) ----
    signal: str = "local_attention"
    overlap: str = "iou"
    select: str = "fuse"
    metric: str = "coco_map"

    # ---- generation behaviour ----
    # Whether rollout 0 is a greedy (T=0) anchor for self-consistency voting. Video benchmarks set
    # True (the N=16 recipe: greedy seed 0 + N-1 stochastic); a config `generate.greedy_seed0`
    # overrides it. See RunConfig.gen_temperature.
    greedy_seed0: bool = False
    # Execution strategy for generate.py: "pooled" = async multi-engine vLLM pool
    # (throughput on many small requests, e.g. 5k COCO images); "sharded" = one engine per GPU
    # (heavy per-item work, e.g. video clips).
    gen_strategy: str = "sharded"

    # ---- per-benchmark: subclasses implement ----
    def load_items(self, cfg: "RunConfig") -> list[dict]:
        """Load the raw work items (images or video queries) to generate on.

        Args:
            cfg: Active run config. Item source paths are resolved via
                ``cfg.path(...)`` (e.g. the annotation file or parquet).

        Returns:
            List of raw item dicts, one per unit of generation work. Their schema
            is benchmark-specific and is consumed only by this adapter's other
            methods.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

    def prompt(self, item: dict) -> str:
        """Build the task prompt string for one item.

        Args:
            item: One raw item from :meth:`load_items`.

        Returns:
            The prompt text sent to the model for this item.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

    def ground_truth(self, item: dict) -> list[GTRegion]:
        """Return the ground-truth regions for one item.

        Args:
            item: One raw item from :meth:`load_items`.

        Returns:
            List of ``{"region", "label"}`` dicts. ``region`` is a box
            ``[x1, y1, x2, y2]`` (image) or a span ``[t0, t1]`` (video); ``label``
            is the class name for detection and empty for video spans.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

    def gen_record(
        self, cfg: "RunConfig", item: dict, response: str, truncated: bool = False
    ) -> GenRecord:
        """Assemble the uniform generation record for one item and response.

        The record is the single shape every downstream stage consumes. The
        response is stored RAW; parsing happens later in the extract/score stages,
        identically for every model. ``extra`` carries whatever the extract stage
        needs to locate the input (e.g. an absolute image or video path resolved
        via ``cfg.path(...)``).

        Args:
            cfg: Active run config, used to resolve input paths for ``extra``.
            item: One raw item from :meth:`load_items`.
            response: The model's raw, unparsed response text.
            truncated: Whether generation hit the token limit before finishing.

        Returns:
            A ``GenRecord``: ``{id, prompt, response (raw), gt, extra}``.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError
