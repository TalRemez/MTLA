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

from .base import DatasetAdapter
from ..registry import register_dataset
from ..types import GenRecord, GTRegion

if TYPE_CHECKING:
    from ..config import RunConfig

PROMPT = (
    "Locate the segment where the following event happens. "
    "Respond with start and end timestamps in seconds. "
    "Event: {query}"
)


@register_dataset("charades")
class CharadesDataset(DatasetAdapter):
    name = "charades"
    task = "video_span"
    # scoring: first-digit MTLA; temporal overlap; single-span selection; recall @ IoU + mIoU.
    signal = "first_digit"
    overlap = "tiou"
    select = "argmax"
    metric = "recall_at_iou"
    greedy_seed0 = True   # N=16 recipe: rollout 0 greedy anchor + N-1 stochastic (paper headline)
    gen_strategy = "sharded"

    def load_items(self, cfg: "RunConfig") -> list[dict]:
        return pd.read_parquet(cfg.path("data")).to_dict("records")

    def prompt(self, item: dict) -> str:
        query = item.get("caption") or item.get("query") or ""
        return PROMPT.format(query=query.rstrip("."))

    def ground_truth(self, item: dict) -> list[GTRegion]:
        ts = item.get("timestamp")
        if ts is None or len(ts) != 2:
            return []
        return [{"region": [float(ts[0]), float(ts[1])], "label": ""}]

    def video_path(self, cfg: "RunConfig", item: dict) -> str:
        return os.path.join(cfg.path("video_dir"), item["video"])

    def gen_record(self, cfg: "RunConfig", item: dict, response: str,
                   truncated: bool = False) -> GenRecord:
        # A video appears under several captions, so the query id must include the caption.
        caption = item.get("caption") or item.get("query")
        return {"id": f"{item['video']}::{caption}", "prompt": self.prompt(item),
                "response": response, "gt": self.ground_truth(item),
                "extra": {"video": self.video_path(cfg, item)}}
