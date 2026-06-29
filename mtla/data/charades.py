"""Charades-STA single-span temporal-grounding dataset adapter.

Scoring (CPU): hallucination AUROC + R@1 @ IoU{0.3,0.5,0.7} + mIoU. Charades emits ONE span
per query, so "voting" is span SELECTION across N rollouts: with agg=max we pick the single
highest-MTLA span (the headline rule; clustering does not help single-span tasks). Lifts the
validated logic from the research charades_voting_with_attn.py.

Reproduces: R@1@0.3 76.3, R@1@0.5 55.4, R@1@0.7 29.4, mIoU 0.508 (SVAR R@1@0.5 43.8).
Requires slot=first_digit.
"""
from __future__ import annotations

import glob
from collections import defaultdict

from .base import DatasetAdapter
from ..score import reduce_band
from ..eval import auroc

SLOT = {"first_digit": 0, "last_digit": 2, "coord_all_mean": 3}

PROMPT = (
    "Locate the segment where the following event happens. "
    "Respond with start and end timestamps in seconds. "
    "Event: {query}"
)


def _inside_sum(frame_sum_LHT, span, duration, n_tokens, band):
    import numpy as np
    if span is None or duration <= 0 or n_tokens <= 0:
        return 0.0
    s, e = span
    fs = max(0, int(np.floor(s * n_tokens / duration)))
    fe = min(n_tokens, int(np.ceil(e * n_tokens / duration)))
    if fe <= fs:
        return 0.0
    return reduce_band(frame_sum_LHT[:, :, fs:fe].sum(axis=2), band)


def _span_iou(a, b):
    if a is None or b is None:
        return 0.0
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 1e-9 else 0.0


class CharadesDataset(DatasetAdapter):
    name = "charades"

    def load_items(self, cfg):
        import pandas as pd
        df = pd.read_parquet(cfg.path("data"))
        return df.to_dict("records")

    def prompt(self, item):
        return PROMPT.format(query=item.get("caption") or item.get("query"))

    def ground_truth(self, item):
        return [item.get("start"), item.get("end")]

    # ---- GPU stage (fused generate+extract in one HF-eager pass) ----
    def generate(self, cfg, model, seed=0):
        self._run(cfg, seed)

    def extract(self, cfg, model, seed=0):
        self._run(cfg, seed)  # single fused pass produces predictions + features

    def _run(self, cfg, seed):
        import os
        from ..stages import run_stage
        run_stage("qwen3vl_charades.py", [
            "--data", cfg.path("data"),
            "--video_dir", cfg.path("video_dir"),
            "--out_dir", os.path.join(cfg.path("features"), f"seed{seed}"),
            "--pred_dir", os.path.join(cfg.path("predictions"), f"seed{seed}"),
            "--gpus", *[str(g) for g in (cfg.extract.gpus or cfg.generate.gpus)],
        ] + (["--seed", str(seed)] if seed else []))

    def score(self, cfg) -> dict:
        import numpy as np
        import torch

        band = cfg.band_indices()
        n = cfg.score.n_rollouts
        slot_idx = SLOT[cfg.score.slot]
        feat_root = cfg.path("features")

        by_query = defaultdict(list)   # (video,caption) -> [{seed, span, mtla, svar}]
        gt_of = {}
        for sd in range(n):
            recs = []
            for sp in sorted(glob.glob(f"{feat_root}/seed{sd}/shard*.pt")):
                recs.extend(torch.load(sp, weights_only=False, map_location="cpu"))
            for r in recs:
                key = (r["video"], r["caption"])
                mtla = _inside_sum(r["attn"]["frame_sum"][slot_idx].astype(np.float32),
                                   r["pred_span"], r["duration_s"], r["T_tokens"], band)
                svar = reduce_band(r["attn"]["video_sum"][slot_idx].astype(np.float32), band)
                span = list(r["pred_span"]) if r["pred_span"] is not None else None
                by_query[key].append({"seed": sd, "span": span, "mtla": mtla, "svar": svar})
                gt_of[key] = list(r["gt_span"])
        queries = sorted(by_query)
        print(f"  queries={len(queries)}")

        # hallucination AUROC (single rollout, seed 0): grounded = IoU(pred,gt) >= 0.5
        mtla_s, svar_s, labels = [], [], []
        for k in queries:
            c0 = next((c for c in by_query[k] if c["seed"] == 0), by_query[k][0])
            hit = _span_iou(c0["span"], gt_of[k]) >= 0.5
            mtla_s.append(c0["mtla"]); svar_s.append(c0["svar"]); labels.append(0 if hit else 1)
        auroc_mtla = auroc(mtla_s, labels) if mtla_s else float("nan")
        auroc_svar = auroc(svar_s, labels) if svar_s else float("nan")
        print(f"\nAUROC (seed 0): MTLA={auroc_mtla:.4f}  SVAR={auroc_svar:.4f}")

        # span selection across rollouts. agg=max -> highest-MTLA span (headline).
        def select(cands, key):
            return max(cands, key=lambda c: c[key])["span"]

        def r_at(select_key):
            ious = []
            for k in queries:
                span = select(by_query[k], select_key)
                ious.append(_span_iou(span, gt_of[k]))
            ious = np.array(ious)
            return (100 * np.mean(ious >= 0.3), 100 * np.mean(ious >= 0.5),
                    100 * np.mean(ious >= 0.7), float(ious.mean()))

        mt = r_at("mtla")
        sv = r_at("svar")
        print(f"\nN={n} selection (agg={cfg.score.agg}, slot={cfg.score.slot})")
        print(f"  max-MTLA : R@.3 {mt[0]:.2f}  R@.5 {mt[1]:.2f}  R@.7 {mt[2]:.2f}  mIoU {mt[3]:.4f}")
        print(f"  max-SVAR : R@.3 {sv[0]:.2f}  R@.5 {sv[1]:.2f}  R@.7 {sv[2]:.2f}  mIoU {sv[3]:.4f}")
        return {"auroc_mtla": auroc_mtla, "auroc_svar": auroc_svar,
                "mtla": {"R@0.3": mt[0], "R@0.5": mt[1], "R@0.7": mt[2], "mIoU": mt[3]},
                "svar": {"R@0.3": sv[0], "R@0.5": sv[1], "R@0.7": sv[2], "mIoU": sv[3]}}
