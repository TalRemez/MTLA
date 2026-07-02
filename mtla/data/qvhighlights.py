"""QVHighlights temporal-grounding dataset adapter (multi-segment).

Declarative: loads items, builds the prompt + ground truth, emits the uniform generation record.
Scoring (per-window hallucination AUROC + moment-retrieval mAP / R@1 via the vendored Moment-DETR
evaluator, after NMS pooling across rollouts) is done by ``mtla.evaluate`` / ``mtla.metrics``.

Reproduces: NMS-MTLA mAP 36.6, R@1@0.5 55.1, R@1@0.7 39.5 (N=16 self-consistency).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from mtla.data.base import DatasetAdapter
from mtla.registry import register_dataset
from mtla.types import GenRecord, GTRegion

if TYPE_CHECKING:
    from mtla.config import RunConfig

PROMPT = (
    "Locate every segment in the video where the following event happens. "
    "Respond with a list of [start, end] timestamps in seconds, one pair per segment. "
    "If the event happens multiple times, list all occurrences. "
    "Event: {query}"
)


@register_dataset("qvhighlights")
class QVHighlightsDataset(DatasetAdapter):
    """QVHighlights multi-segment temporal-grounding adapter (task ``video_span``).

    One item per query; the prompt asks for every matching segment in the video.
    Scoring uses the ``first_digit`` signal (the validated video signal) with
    temporal IoU, pools segments across rollouts with NMS (``select="fuse"``), and
    reports moment-retrieval mAP / R@1 via the vendored Moment-DETR evaluator. Uses
    the N=16 recipe (``greedy_seed0=True``) and the ``"sharded"`` engine-per-GPU
    strategy for the heavy per-clip work.
    """

    name = "qvhighlights"
    task = "video_span"
    # scoring: first-digit MTLA (validated video signal); temporal overlap; NMS pool; moment mAP/R@1.
    signal = "first_digit"
    overlap = "tiou"
    select = "fuse"
    metric = "moment_retrieval"
    greedy_seed0 = (
        True  # N=16 recipe: rollout 0 greedy anchor + N-1 stochastic (paper headline)
    )
    gen_strategy = "sharded"  # heavy per-clip work -> one engine per GPU

    def load_items(self, cfg: "RunConfig") -> list[dict]:
        """Load the per-query work items from the QVHighlights JSONL annotations.

        Args:
            cfg: Active run config; ``cfg.path("ann")`` points at the JSONL file
                with one JSON object per line (per query).

        Returns:
            List of query item dicts, each with ``qid``, ``vid``, ``query``, and an
            optional ``relevant_windows`` list of ground-truth spans.
        """
        with open(cfg.path("ann")) as f:
            return [json.loads(ln) for ln in f]

    def prompt(self, item: dict) -> str:
        """Build the multi-segment localization prompt for one query.

        Args:
            item: A query item whose ``query`` field is the event description.

        Returns:
            The prompt asking for every ``[start, end]`` segment of the event.
        """
        return PROMPT.format(query=item["query"])

    def ground_truth(self, item: dict) -> list[GTRegion]:
        """Return the ground-truth segments for one query.

        Args:
            item: A query item with an optional ``relevant_windows`` list of
                ``[t0, t1]`` spans in seconds.

        Returns:
            List of ``{"region", "label"}`` dicts, one per relevant window (empty
            ``label``). Empty when the query has no annotated windows.
        """
        return [
            {"region": list(w), "label": ""} for w in item.get("relevant_windows", [])
        ]

    def video_path(self, cfg: "RunConfig", item: dict) -> str:
        """Resolve the absolute path to one query's video file.

        Args:
            cfg: Active run config; ``cfg.path("video_dir")`` is the video root.
            item: A query item whose ``vid`` field is the video id.

        Returns:
            The video directory joined with ``<vid>.mp4``.
        """
        return os.path.join(cfg.path("video_dir"), f"{item['vid']}.mp4")

    def gen_record(
        self, cfg: "RunConfig", item: dict, response: str, truncated: bool = False
    ) -> GenRecord:
        """Assemble the uniform generation record for one query.

        Args:
            cfg: Active run config, used to resolve the video path for ``extra``.
            item: A query item with ``qid`` and ``vid``.
            response: The model's raw, unparsed response text.
            truncated: Whether generation hit the token limit before finishing.

        Returns:
            A ``GenRecord`` with id ``qid`` and ``extra`` holding
            ``{"video": <path>}``.
        """
        return {
            "id": item["qid"],
            "prompt": self.prompt(item),
            "response": response,
            "gt": self.ground_truth(item),
            "extra": {"video": self.video_path(cfg, item)},
        }
