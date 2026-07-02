"""Charades-STA single-span temporal-grounding dataset adapter.

Declarative: loads items, builds the prompt + ground truth, emits the uniform generation record.
Scoring (hallucination AUROC + R@1 @ IoU{0.3,0.5,0.7} + mIoU) is done by ``mtla.evaluate`` /
``mtla.metrics``. Charades emits ONE span per query, so voting is span SELECTION across rollouts
(``select="argmax"``: keep the single highest-MTLA span — the headline rule).

Reproduces: R@1@0.3 76.3, R@1@0.5 55.4, R@1@0.7 29.4, mIoU 0.508 (N=16 self-consistency).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd

from mtla.data.base import DatasetAdapter
from mtla.registry import register_dataset
from mtla.types import GenRecord, GTRegion

if TYPE_CHECKING:
    from mtla.config import RunConfig

PROMPT = (
    "Locate the segment where the following event happens. "
    "Respond with start and end timestamps in seconds. "
    "Event: {query}"
)


@register_dataset("charades")
class CharadesDataset(DatasetAdapter):
    """Charades-STA single-span temporal-grounding adapter (task ``video_span``).

    One item per (video, caption) query; the prompt asks for a single start/end
    span in seconds. Scoring uses the ``first_digit`` signal with temporal IoU and
    keeps the single highest-MTLA span per query (``select="argmax"``, the headline
    rule), reporting R@1 at IoU {0.3, 0.5, 0.7} plus mIoU. Uses the N=16 recipe
    (``greedy_seed0=True``) and the ``"sharded"`` engine-per-GPU strategy.
    """

    name = "charades"
    task = "video_span"
    # scoring: first-digit MTLA; temporal overlap; single-span selection; recall @ IoU + mIoU.
    signal = "first_digit"
    overlap = "tiou"
    select = "argmax"
    metric = "recall_at_iou"
    greedy_seed0 = (
        True  # N=16 recipe: rollout 0 greedy anchor + N-1 stochastic (paper headline)
    )
    gen_strategy = "sharded"

    def load_items(self, cfg: "RunConfig") -> list[dict]:
        """Load the per-query work items from the Charades parquet file.

        Args:
            cfg: Active run config; ``cfg.path("data")`` points at the parquet file
                with one row per (video, caption) query.

        Returns:
            List of item dicts (one per parquet row), each with a ``video`` name, a
            ``caption``/``query``, and a ``timestamp`` ground-truth span.
        """
        return pd.read_parquet(cfg.path("data")).to_dict("records")

    def prompt(self, item: dict) -> str:
        """Build the single-span localization prompt for one query.

        Args:
            item: A query item; the event text is taken from ``caption`` or, if
                absent, ``query`` (a trailing period is stripped).

        Returns:
            The prompt asking for the start and end timestamps of the event.
        """
        query = item.get("caption") or item.get("query") or ""
        return PROMPT.format(query=query.rstrip("."))

    def ground_truth(self, item: dict) -> list[GTRegion]:
        """Return the ground-truth span for one query.

        Args:
            item: A query item with a ``timestamp`` field: a ``[t0, t1]`` pair in
                seconds, or missing/malformed for an unannotated query.

        Returns:
            A single-element list ``[{"region": [t0, t1], "label": ""}]``, or an
            empty list when ``timestamp`` is missing or not a length-2 pair.
        """
        ts = item.get("timestamp")
        if ts is None or len(ts) != 2:
            return []
        return [{"region": [float(ts[0]), float(ts[1])], "label": ""}]

    def video_path(self, cfg: "RunConfig", item: dict) -> str:
        """Resolve the absolute path to one query's video file.

        Args:
            cfg: Active run config; ``cfg.path("video_dir")`` is the video root.
            item: A query item whose ``video`` field is the file name.

        Returns:
            The video directory joined with the item's ``video`` file name.
        """
        return os.path.join(cfg.path("video_dir"), item["video"])

    def gen_record(
        self, cfg: "RunConfig", item: dict, response: str, truncated: bool = False
    ) -> GenRecord:
        """Assemble the uniform generation record for one query.

        The id combines video and caption because one video appears under several
        captions, so the caption is needed to make the query id unique.

        Args:
            cfg: Active run config, used to resolve the video path for ``extra``.
            item: A query item with ``video`` and ``caption``/``query``.
            response: The model's raw, unparsed response text.
            truncated: Whether generation hit the token limit before finishing.

        Returns:
            A ``GenRecord`` with id ``"<video>::<caption>"`` and ``extra`` holding
            ``{"video": <path>}``.
        """
        # A video appears under several captions, so the query id must include the caption.
        caption = item.get("caption") or item.get("query")
        return {
            "id": f"{item['video']}::{caption}",
            "prompt": self.prompt(item),
            "response": response,
            "gt": self.ground_truth(item),
            "extra": {"video": self.video_path(cfg, item)},
        }
