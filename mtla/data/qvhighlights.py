"""QVHighlights temporal-grounding dataset adapter (multi-segment).

Scoring (CPU): per-window hallucination AUROC + moment-retrieval mAP / R@1 after pooling
windows across rollouts and ranking them with NMS (max fusion = video headline). Uses the
official Moment-DETR evaluator vendored under third_party/moment_detr_eval. Lifts the
validated logic from the original examples/qvhighlights/score.py.

Reproduces: NMS-MTLA mAP 36.6, R@1@0.5 55.1, R@1@0.7 39.5.

The MTLA score per predicted window is read straight from that window's stored `first_digit`
array (the validated video signal); each window was masked to its own inside-span frame tokens
at extract time, so no offline window-slicing is needed.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from .base import DatasetAdapter
from ..registry import register_dataset
from ..score import reduce_band
from ..eval import auroc
from ..voting import nms_fuse, tiou

MAX_PRED_WINDOWS = 10
# Default MTLA signal for video scoring (the paper/validated choice; images use local_attention).
VIDEO_SIGNAL = "first_digit"

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
    # Video sampling for the Qwen3-VL video ext_* (deterministic, paper-faithful). `multi=True`:
    # QVHighlights emits MULTIPLE [start,end] windows per query. Dataset properties (not user
    # knobs), read by mtla.models.qwen3vl during extraction.
    video = {"fps": 1.0, "min_pixels": 4 * 32 * 32, "max_pixels": 64 * 32 * 32,
             "max_new_tokens": 128, "multi": True}

    def load_items(self, cfg):
        items = []
        with open(cfg.path("ann")) as f:
            for ln in f:
                items.append(json.loads(ln))
        return items

    def prompt(self, item):
        return PROMPT.format(query=item["query"])

    def ground_truth(self, item):
        return item.get("relevant_windows", [])

    def video_item(self, p, video_dir):
        """Normalize one QVHighlights prediction record for the video extractor (multi-window)."""
        import os
        return {"video_path": os.path.join(video_dir, f"{p['vid']}.mp4"),
                "query": PROMPT.format(query=p["query"]),
                "pred_windows": [list(w) for w in (p.get("pred_windows") or [])],
                "gt_windows": [list(w) for w in (p.get("gt_windows") or [])]}

    # ---- GPU stages: config-driven (the model adapter names the video_span script) ----
    def stage_cmd(self, cfg, model, seed, mode):
        args = ["--config", cfg.config_path, "--seed", str(seed)]
        if mode == "generate":
            return model.generate_script(self.task, cfg.generate.engine), args
        return model.extract_script(self.task), args

    def score(self, cfg, model=None) -> dict:
        # Each predicted window carries its own reduced MTLA array (`first_digit`); `model` is
        # accepted for interface uniformity but not needed here.
        import numpy as np

        band = cfg.band_indices()
        n = cfg.score.n_rollouts
        agg = cfg.score.agg
        signal = VIDEO_SIGNAL

        # vendored official Moment-DETR eval
        tp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "third_party")
        sys.path.insert(0, tp)
        from moment_detr_eval.eval import compute_mr_ap, compute_mr_r1

        by_query = defaultdict(list)   # qid -> [{seed, segs: [(window, mtla)]}]
        gt_of = {}
        for sd in range(n):
            for r in self.load_shards(cfg.feat_dir(sd)):
                qid = r["qid"]
                segs = [(list(o["window"]), reduce_band(o[signal].astype(np.float32), band))
                        for o in r["objects"]]
                by_query[qid].append({"seed": sd, "segs": segs})
                gt_of[qid] = [list(w) for w in r.get("gt_windows", [])]

        ground_truth = [{"qid": qid, "relevant_windows": gt_of[qid]}
                        for qid in by_query if gt_of.get(qid)]
        gt_qids = {d["qid"] for d in ground_truth}
        queries = sorted(q for q in by_query if q in gt_qids)

        def submission():
            """Pool windows across rollouts, fuse with NMS (agg), -> ranked submission."""
            out = []
            for qid in queries:
                cands = []
                for roll in by_query[qid]:
                    for (w, m) in roll["segs"]:
                        cands.append((w, float(m), roll["seed"]))
                kept = nms_fuse(cands, agg=agg, iou_fn=tiou)
                rows = [[w[0], w[1], float(sc)] for w, sc in kept][:MAX_PRED_WINDOWS]
                out.append({"qid": qid, "pred_relevant_windows": rows or [[0.0, 0.0, 0.0]]})
            return out

        def evaluate(sub):
            apd = compute_mr_ap(sub, ground_truth, num_workers=8)
            r1 = compute_mr_r1(sub, ground_truth)
            return {"mAP": apd["average"], "mAP@0.5": apd["0.5"], "mAP@0.75": apd["0.75"],
                    "R1@0.5": r1["0.5"], "R1@0.7": r1["0.7"]}

        # per-window hallucination AUROC (single rollout, seed 0)
        mtla_s, labels = [], []
        for qid in queries:
            roll0 = next((r for r in by_query[qid] if r["seed"] == 0), by_query[qid][0])
            gts = gt_of[qid]
            for (w, m) in roll0["segs"]:
                hit = any(tiou(w, g) >= 0.5 for g in gts)
                mtla_s.append(m); labels.append(0 if hit else 1)
        auroc_mtla = auroc(mtla_s, labels) if mtla_s else float("nan")

        return {"auroc_mtla": auroc_mtla, "nms_mtla": evaluate(submission())}
