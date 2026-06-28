import os
"""Single-rollout (or N-seed) PSDS1 for AF3 AudioSet-Strong predictions,
scored by emission-order / SVAR / MTLA, reusing the verified FlexSED machinery
and the join/fusion helpers from audioset_voting_with_attn.py.

This is the AF3 analog of the md-file table: it answers "does inside-window
attention (MTLA) re-rank AF3's own detections better than global attention
(SVAR) or raw emission order?" under the exact FlexSED PSDS1 protocol.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("CASCADE_ROOT", "."))
sys.path.insert(0, str(ROOT / "artifacts/scripts"))
import score_audioset_strong_psds_flexsed as P
from audioset_voting_with_attn import (reduce_band, build_gt_df, psds1_for_rows,
                                        fuse_nms, fuse_clusters, single_rollout_rows)
import audioset_voting_with_attn as V


def load_af3(pred_root, attn_root, n_seeds, shard):
    cands = defaultdict(list)
    gt_by_vid = {}
    for sd in range(n_seeds):
        pp = Path(pred_root) / f"seed{sd}" / f"predictions_shard{shard:02d}.jsonl"
        preds_by_vid = {}
        with open(pp) as f:
            for line in f:
                r = json.loads(line)
                preds_by_vid[r["video_id"]] = r
        attn_by_vid = {}
        ap = Path(attn_root) / f"seed{sd}" / f"shard{shard:02d}.pt"
        for r in torch.load(ap, map_location="cpu", weights_only=False):
            attn_by_vid[r["video_id"]] = r
        n_join = 0
        for vid, pr in preds_by_vid.items():
            gt_by_vid.setdefault(vid, pr.get("gt_events") or [])
            ar = attn_by_vid.get(vid)
            objs = (ar or {}).get("objects") or []
            for pi, p in enumerate(pr.get("predictions") or []):
                if pi >= len(objs) or objs[pi] is None:
                    continue
                lo, hi = float(p["bbox_2d"][0]), float(p["bbox_2d"][1])
                if hi <= lo:
                    continue
                ao = objs[pi]
                mtla = reduce_band(ao.get("attn_all"), "image_inside_sum")
                svar = reduce_band(ao.get("attn_first"), "image_global_sum")
                cands[vid].append((p["label"], [lo, hi], mtla, svar, sd))
                n_join += 1
        print(f"  seed{sd}: {n_join} (pred,attn) candidates", flush=True)
    return cands, gt_by_vid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", default="/tmp/af3_preds")
    ap.add_argument("--attn_root", default="/tmp/af3_attn")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--layer_band", default="all", help="'all' or 'lo-hi' (e.g. 8-21)")
    ap.add_argument("--n_thresholds", type=int, default=0)
    args = ap.parse_args()

    if args.layer_band == "all":
        V.LF_BAND = None
    else:
        lo, hi = (int(x) for x in args.layer_band.split("-"))
        V.LF_BAND = list(range(lo, hi + 1))
    print(f"layer band = {args.layer_band}", flush=True)

    cands, gt_by_vid = load_af3(args.pred_root, args.attn_root, args.n, args.shard)
    vids = sorted(cands.keys())
    gt_df, meta_df = build_gt_df(gt_by_vid, vids)
    classes_in_gt = sorted(gt_df["event_label"].unique().tolist())
    print(f"clips={len(vids)} GT events={len(gt_df)} classes={len(classes_in_gt)}", flush=True)

    NTH = args.n_thresholds

    def score(name, rows):
        v = psds1_for_rows(rows, gt_df, meta_df, classes_in_gt, n_thresholds=NTH)
        print(f"{name:<26s} {('--' if v is None else f'{v:.4f}'):>9s}", flush=True)

    print("\n" + "=" * 50)
    print(f"AF3 AudioSet-Strong  N={args.n}  PSDS1 (dtc=gtc=0.7, alpha_st=0)")
    print(f"{'variant':<26s} {'PSDS1':>9s}")
    print("-" * 50)

    # single-rollout (mean over seeds): emission / SVAR / MTLA
    for key in ("emit", "svar", "mtla"):
        vals = []
        for sd in range(args.n):
            v = psds1_for_rows(single_rollout_rows(cands, sd, key),
                               gt_df, meta_df, classes_in_gt, n_thresholds=NTH)
            if v is not None:
                vals.append(v)
        print(f"{'single ' + key:<26s} {(f'{np.mean(vals):.4f}' if vals else '--'):>9s}", flush=True)

    if args.n > 1:
        score("NMS-SVAR", fuse_nms(cands, "svar"))
        score("NMS-MTLA", fuse_nms(cands, "mtla"))
        score("NMS-SVAR x support", fuse_nms(cands, "svar", use_support=True))
        score("NMS-MTLA x support", fuse_nms(cands, "mtla", use_support=True))
    print("=" * 50)


if __name__ == "__main__":
    main()
