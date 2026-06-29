"""QVHighlights temporal-grounding dataset adapter (multi-segment).

Scoring (CPU): per-window hallucination AUROC + moment-retrieval mAP / R@1 after pooling
windows across rollouts and ranking them with NMS (max fusion = video headline). Uses the
official Moment-DETR evaluator vendored under third_party/moment_detr_eval. Lifts the
validated logic from the original examples/qvhighlights/score.py.

Reproduces: NMS-MTLA mAP 36.6, R@1@0.5 55.1, R@1@0.7 39.5 (SVAR baseline mAP 28.1).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

from .base import DatasetAdapter
from ..score import reduce_band
from ..eval import auroc
from ..voting import nms_fuse, tiou

SLOT = {"first_digit": 0, "first2_mean": 1, "last_digit": 2, "all_mean": 3}
MAX_PRED_WINDOWS = 10

PROMPT = (
    "Locate every segment in the video where the following event happens. "
    "Respond with a list of [start, end] timestamps in seconds, one pair per segment. "
    "If the event happens multiple times, list all occurrences. "
    "Event: {query}"
)


def _seg_mtla(frame_sum_LHT, window, duration, n_tokens, band):
    """Inside-window frame attention -> MTLA scalar (floor/ceil time->token mapping)."""
    import numpy as np
    if window is None or duration <= 0 or n_tokens <= 0:
        return 0.0
    s, e = window
    fs = max(0, int(np.floor(s * n_tokens / duration)))
    fe = min(n_tokens, int(np.ceil(e * n_tokens / duration)))
    if fe <= fs:
        return 0.0
    return reduce_band(frame_sum_LHT[:, :, fs:fe].sum(axis=2), band)


class QVHighlightsDataset(DatasetAdapter):
    name = "qvhighlights"
    task = "video_span"

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

    # ---- GPU stages (decoupled): same stage script, --mode picks generate vs extract ----
    def generate(self, cfg, model, seed=0):
        self._run(cfg, model, seed, "generate")

    def extract(self, cfg, model, seed=0):
        self._run(cfg, model, seed, "extract")

    def _run(self, cfg, model, seed, mode):
        from ..stages import run_stage
        n_items = (cfg.generate if mode == "generate" else cfg.extract).n_items
        args = [
            "--mode", mode,
            "--ann", cfg.path("ann"),
            "--video_dir", cfg.path("video_dir"),
            "--out_dir", os.path.join(cfg.path("features"), f"seed{seed}"),
            "--pred_dir", os.path.join(cfg.path("predictions"), f"seed{seed}"),
            "--gpus", *[str(g) for g in (cfg.extract.gpus or cfg.generate.gpus)],
        ]
        if n_items:
            args += ["--limit", str(n_items)]
        if seed:
            args += ["--seed", str(seed)]
        run_stage("qwen3vl_video.py", args)

    def score(self, cfg, model=None) -> dict:
        # Video scoring uses the record's 4-slot [.,L,H,T] index mechanism directly (single
        # model family); `model` is accepted for interface uniformity but not needed here.
        import numpy as np
        import torch

        band = cfg.band_indices()
        n = cfg.score.n_rollouts
        agg = cfg.score.agg
        slot_idx = SLOT[cfg.score.slot]
        feat_root = cfg.path("features")
        ann_path = cfg.path("ann")

        # vendored official Moment-DETR eval
        tp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "third_party")
        sys.path.insert(0, tp)
        from moment_detr_eval.eval import compute_mr_ap, compute_mr_r1

        by_query = defaultdict(list)
        gt_of = {}
        for sd in range(n):
            recs = []
            for sp in sorted(glob.glob(f"{feat_root}/seed{sd}/shard*.pt")):
                recs.extend(torch.load(sp, weights_only=False, map_location="cpu"))
            for r in recs:
                qid = r["qid"]
                fs = r["attn"]["frame_sum"][slot_idx].astype(np.float32)
                gsvar = reduce_band(r["attn"]["video_sum"][slot_idx].astype(np.float32), band)
                rollout = []
                for ei, w in enumerate(r["pred_windows"] or []):
                    w = list(w)
                    m = _seg_mtla(fs, w, r["duration_s"], r["T_tokens"], band)
                    rollout.append((w, m, gsvar, ei))
                by_query[qid].append({"seed": sd, "segs": rollout})
                gt_of[qid] = [list(w) for w in r["gt_windows"]]

        ground_truth = []
        with open(ann_path) as f:
            for ln in f:
                d = json.loads(ln)
                if d["qid"] in by_query:
                    ground_truth.append({"qid": d["qid"], "relevant_windows": d["relevant_windows"]})
        gt_qids = {d["qid"] for d in ground_truth}
        queries = sorted(q for q in by_query if q in gt_qids)
        print(f"  queries={len(queries)} (GT matched)")

        def submission(key):
            """Pool windows across rollouts, fuse with NMS (agg), -> ranked submission."""
            out = []
            for qid in queries:
                cands = []
                for roll in by_query[qid]:
                    for (w, m, gv, ei) in roll["segs"]:
                        sc = m if key == "mtla" else gv
                        cands.append((w, float(sc), roll["seed"]))
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
        mtla_s, svar_s, labels = [], [], []
        for qid in queries:
            roll0 = next((r for r in by_query[qid] if r["seed"] == 0), by_query[qid][0])
            gts = gt_of[qid]
            for (w, m, gv, ei) in roll0["segs"]:
                hit = any(tiou(w, g) >= 0.5 for g in gts)
                mtla_s.append(m); svar_s.append(gv); labels.append(0 if hit else 1)
        auroc_mtla = auroc(mtla_s, labels) if mtla_s else float("nan")
        auroc_svar = auroc(svar_s, labels) if svar_s else float("nan")
        print(f"\nPer-window AUROC (seed 0): MTLA={auroc_mtla:.4f}  SVAR={auroc_svar:.4f}")

        mtla_res = evaluate(submission("mtla"))
        svar_res = evaluate(submission("svar"))
        print(f"\nN={n} voting (agg={agg}, slot={cfg.score.slot})")
        print(f"  NMS-MTLA : mAP {mtla_res['mAP']:.2f}  R1@.5 {mtla_res['R1@0.5']:.2f}  "
              f"R1@.7 {mtla_res['R1@0.7']:.2f}")
        print(f"  NMS-SVAR : mAP {svar_res['mAP']:.2f}  R1@.5 {svar_res['R1@0.5']:.2f}  "
              f"R1@.7 {svar_res['R1@0.7']:.2f}")
        return {"auroc_mtla": auroc_mtla, "auroc_svar": auroc_svar,
                "mtla": mtla_res, "svar": svar_res}
