import os
"""Cascade Stage E: PSDS1 (FlexSED) for the AF3 cascade, scored by emission /
SVAR / MTLA. Reads attn_shard*.pt (Stage D), reuses the verified machinery.
Also prints per-prediction hallucination AUROC (the recall-agnostic ranking
test) for MTLA vs SVAR.
"""
import argparse, glob, json, sys
from collections import defaultdict
import numpy as np, torch
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("CASCADE_ROOT", ".")
sys.path.insert(0, f"{ROOT}/artifacts/scripts")
import score_audioset_strong_psds_flexsed as P
from audioset_voting_with_attn import (build_gt_df, psds1_for_rows, single_rollout_rows,
                                       fuse_nms, fuse_clusters)
import audioset_voting_with_attn as V


def reduce_band(d, key, band):
    t = d[key]
    t = t if isinstance(t, torch.Tensor) else torch.tensor(np.asarray(t))
    bt = t if band is None else t[[l for l in band if l < t.shape[0]], :]
    return float(bt.float().mean(1).sum().item())


def iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 1e-9 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_glob", default="/tmp/af3_cascade/attn_shard*.pt")
    ap.add_argument("--layer_band", default="all")
    ap.add_argument("--n_thresholds", type=int, default=0)
    args = ap.parse_args()
    band = None if args.layer_band == "all" else list(range(*[int(x) for x in args.layer_band.split("-")][:1] + [int(args.layer_band.split("-")[1]) + 1]))
    V.LF_BAND = band

    import time as _t
    _t0 = _t.time()
    def _log(m): print(f"[{_t.time()-_t0:6.1f}s] {m}", flush=True)

    recs = []
    for f in sorted(glob.glob(args.attn_glob)):
        recs += torch.load(f, map_location="cpu", weights_only=False)
    _log(f"loaded {len(recs)} clips")

    # VECTORIZED reduction: stack all objects' [L,H] tensors and reduce in one
    # batched op (mean over heads, sum over band-layers) instead of 378k single
    # python calls. band=None -> all layers.
    gt_by = {}
    meta = []          # (vid, label, lo, hi, seed)
    mt_stack, sv_stack = [], []
    n_seeds = 1
    for r in recs:
        vid = r["video_id"]; gt_by[vid] = r.get("gt_events") or []
        for o in r.get("objects") or []:
            lo, hi = o["bbox_2d"]; sd = o.get("seed", 0); n_seeds = max(n_seeds, sd + 1)
            meta.append((vid, o["label"], lo, hi, sd))
            ta = o["attn_all"]["image_inside_sum"]; tf = o["attn_first"]["image_global_sum"]
            mt_stack.append(ta if isinstance(ta, torch.Tensor) else torch.tensor(np.asarray(ta)))
            sv_stack.append(tf if isinstance(tf, torch.Tensor) else torch.tensor(np.asarray(tf)))
    _log(f"stacking {len(meta)} objects")
    MT = torch.stack(mt_stack).float()   # [N, L, H]
    SV = torch.stack(sv_stack).float()
    if band is not None:
        bidx = [l for l in band if l < MT.shape[1]]
        MT = MT[:, bidx, :]; SV = SV[:, bidx, :]
    mt_all = MT.mean(2).sum(1).numpy()   # mean heads, sum layers -> [N]
    sv_all = SV.mean(2).sum(1).numpy()
    _log("reduced (vectorized)")

    cands = defaultdict(list)
    y, m_s, s_s = [], [], []
    for i, (vid, lab, lo, hi, sd) in enumerate(meta):
        mt = float(mt_all[i]); sv = float(sv_all[i])
        cands[vid].append((lab, [lo, hi], mt, sv, sd))
        hit = any(g.get("event_name") == lab and iou([lo, hi], [g["start"], g["end"]]) >= 0.5
                  for g in gt_by[vid])
        y.append(int(hit)); m_s.append(mt); s_s.append(sv)
    y = np.array(y)
    _log(f"built cands: detections={len(y)} hits(IoU>=0.5,label)={y.sum()} ({100*y.mean():.1f}%) n_seeds={n_seeds}")
    if 0 < y.sum() < len(y):
        _log(f"per-prediction AUROC:  MTLA={roc_auc_score(y,m_s):.4f}  SVAR={roc_auc_score(y,s_s):.4f}")

    vids = sorted(cands.keys())
    gt_df, meta_df = build_gt_df(gt_by, vids)
    classes_in_gt = sorted(gt_df["event_label"].unique().tolist())
    _log(f"GT built: clips={len(vids)} GT events={len(gt_df)} classes={len(classes_in_gt)}")

    print("=" * 46)
    print(f"AF3 CASCADE PSDS1 (dtc=gtc=0.7, alpha_st=0)  band={args.layer_band}")
    print(f"{'variant':<22s}{'PSDS1':>10s}")
    print("-" * 46)
    NTH = args.n_thresholds

    def score(name, rows):
        t = _t.time()
        v = psds1_for_rows(rows, gt_df, meta_df, classes_in_gt, n_thresholds=NTH)
        print(f"{name:<22s}{('--' if v is None else f'{v:.4f}'):>10s}   ({_t.time()-t:.0f}s, {len(rows)} rows)", flush=True)

    # single-rollout reference: seed 0 only (per-seed mean over 16 seeds = 48
    # extra sweeps, not worth it; seed 0 is representative).
    for key in ("emit", "svar", "mtla"):
        score("single " + key, single_rollout_rows(cands, 0, key))

    if n_seeds > 1:
        print("-" * 46)
        score("NMS-SVAR", fuse_nms(cands, "svar"))
        score("NMS-MTLA", fuse_nms(cands, "mtla"))
        score("NMS-SVAR x support", fuse_nms(cands, "svar", use_support=True))
        score("NMS-MTLA x support", fuse_nms(cands, "mtla", use_support=True))
        score("clust vote x MTLA", fuse_clusters(cands, lambda v, m, s: v * m))
        score("clust vote x SVAR", fuse_clusters(cands, lambda v, m, s: v * s))
        score("clust vote-alone", fuse_clusters(cands, lambda v, m, s: v))
    print(f"  (n_seeds detected = {n_seeds})")
    print("=" * 46)


if __name__ == "__main__":
    main()
