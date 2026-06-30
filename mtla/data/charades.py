"""Charades-STA single-span temporal-grounding dataset adapter.

Scoring (CPU): hallucination AUROC + R@1 @ IoU{0.3,0.5,0.7} + mIoU. Charades emits ONE span
per query, so "voting" is span SELECTION across N rollouts: with agg=max we pick the single
highest-MTLA span (the headline rule; clustering does not help single-span tasks).

The MTLA signal is read straight from each window's stored `first_digit` array (the validated
video signal — see configs); `local_attention` (mean over the window's digits) is also stored.

Reproduces: R@1@0.3 76.3, R@1@0.5 55.4, R@1@0.7 29.4, mIoU 0.508.
"""
from __future__ import annotations

from collections import defaultdict

from .base import DatasetAdapter
from ..registry import register_dataset
from ..score import reduce_band
from ..eval import auroc
from ..utils import tiou

# Default MTLA signal for video scoring (the paper/validated choice; images use local_attention).
VIDEO_SIGNAL = "first_digit"

PROMPT = (
    "Locate the segment where the following event happens. "
    "Respond with start and end timestamps in seconds. "
    "Event: {query}"
)


@register_dataset("charades")
class CharadesDataset(DatasetAdapter):
    name = "charades"
    task = "video_span"
    # Video sampling for the Qwen3-VL video ext_* (deterministic, paper-faithful). `multi=False`:
    # Charades emits ONE span per query. These are dataset properties (not user knobs), read by
    # mtla.models.qwen3vl during extraction.
    video = {"fps": 2.0, "min_pixels": 4 * 32 * 32, "max_pixels": 128 * 32 * 32,
             "max_new_tokens": 128, "multi": False}

    def load_items(self, cfg):
        import pandas as pd
        df = pd.read_parquet(cfg.path("data"))
        return df.to_dict("records")

    def prompt(self, item):
        return PROMPT.format(query=(item.get("caption") or item.get("query")).rstrip("."))

    def ground_truth(self, item):
        return [item.get("start"), item.get("end")]

    def video_item(self, p, video_dir):
        """Normalize one Charades prediction record for the video extractor (single span)."""
        import os
        sp = p.get("pred_span")
        return {"video_path": os.path.join(video_dir, p["video"]),
                "query": self.prompt(p),
                "pred_windows": [list(sp)] if sp else [],
                "gt_windows": [list(p["gt_span"])] if p.get("gt_span") else []}

    # ---- GPU stages: config-driven (the model adapter names the video_span script) ----
    def stage_cmd(self, cfg, model, seed, mode):
        args = ["--config", cfg.config_path, "--seed", str(seed)]
        if mode == "generate":
            return model.generate_script(self.task, cfg.generate.engine), args
        return model.extract_script(self.task), args

    def score(self, cfg, model=None) -> dict:
        import numpy as np

        band = cfg.band_indices()
        n = cfg.score.n_rollouts
        signal = VIDEO_SIGNAL

        by_query = defaultdict(list)   # (video,query) -> [{seed, span, mtla}]
        gt_of = {}
        for sd in range(n):
            for r in self.load_shards(cfg.feat_dir(sd)):
                key = (r.get("video"), r.get("query"))
                # one span per Charades query -> one object per record
                obj = r["objects"][0] if r["objects"] else None
                if obj is None:
                    continue
                mtla = reduce_band(obj[signal].astype(np.float32), band)
                by_query[key].append({"seed": sd, "span": list(obj["window"]), "mtla": mtla})
                gt_of[key] = list(r["gt_windows"][0]) if r.get("gt_windows") else None
        queries = sorted(by_query)

        # hallucination AUROC (single rollout, seed 0): grounded = IoU(pred,gt) >= 0.5
        mtla_s, labels = [], []
        for k in queries:
            c0 = next((c for c in by_query[k] if c["seed"] == 0), by_query[k][0])
            hit = tiou(c0["span"], gt_of[k]) >= 0.5 if gt_of[k] else False
            mtla_s.append(c0["mtla"]); labels.append(0 if hit else 1)
        auroc_mtla = auroc(mtla_s, labels) if mtla_s else float("nan")

        # span selection across rollouts. agg=max -> highest-MTLA span (headline).
        def r_at():
            ious = []
            for k in queries:
                span = max(by_query[k], key=lambda c: c["mtla"])["span"]
                ious.append(tiou(span, gt_of[k]) if gt_of[k] else 0.0)
            ious = np.array(ious)
            return (100 * np.mean(ious >= 0.3), 100 * np.mean(ious >= 0.5),
                    100 * np.mean(ious >= 0.7), float(ious.mean()))

        mt = r_at()
        return {"auroc_mtla": auroc_mtla,
                "mtla": {"R@0.3": mt[0], "R@0.5": mt[1], "R@0.7": mt[2], "mIoU": mt[3]}}
