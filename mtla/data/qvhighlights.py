"""QVHighlights temporal-grounding dataset adapter (multi-segment).

Declarative: loads items, builds the prompt + ground truth, emits the uniform generation record.
Scoring (per-window hallucination AUROC + moment-retrieval mAP / R@1 via the vendored Moment-DETR
evaluator, after NMS pooling across rollouts) is done by ``mtla.evaluate`` / ``mtla.metrics``.

Reproduces: NMS-MTLA mAP 36.6, R@1@0.5 55.1, R@1@0.7 39.5 (N=16 self-consistency).
"""
from __future__ import annotations

import json
import os

from .base import DatasetAdapter
from ..registry import register_dataset

PROMPT = (
    "Locate every segment in the video where the following event happens. "
    "Respond with a list of [start, end] timestamps in seconds, one pair per segment. "
    "If the event happens multiple times, list all occurrences. "
    "Event: {query}"
)


@register_dataset("qvhighlights")
class QVHighlightsDataset(DatasetAdapter):
    name = "qvhighlights"
    task = "video_span"
    # scoring: first-digit MTLA (validated video signal); temporal overlap; NMS pool; moment mAP/R@1.
    signal = "first_digit"
    overlap = "tiou"
    select = "fuse"
    metric = "moment_retrieval"
    greedy_seed0 = True   # N=16 recipe: rollout 0 greedy anchor + N-1 stochastic (paper headline)
    gen_strategy = "sharded"   # heavy per-clip work -> one engine per GPU

    def load_items(self, cfg):
        with open(cfg.path("ann")) as f:
            return [json.loads(ln) for ln in f]

    def prompt(self, item):
        return PROMPT.format(query=item["query"])

    def ground_truth(self, item):
        return [{"region": list(w), "label": ""} for w in item.get("relevant_windows", [])]

    def video_path(self, cfg, item):
        return os.path.join(cfg.path("video_dir"), f"{item['vid']}.mp4")

    def gen_record(self, cfg, item, response, truncated=False):
        return {"id": item["qid"], "prompt": self.prompt(item), "response": response,
                "gt": self.ground_truth(item), "extra": {"video": self.video_path(cfg, item)}}
