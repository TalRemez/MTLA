import os
"""AudioSet-Strong self-consistency voting with MTLA, scored by FlexSED-comparable
PSDS1 (the verified protocol in score_audioset_strong_psds_flexsed.py).

AudioSet-Strong is MULTI-event AND every event has a class label, so this mirrors
the COCO voting recipe (within-label clustering/NMS), not the class-agnostic QVH one:
  1. Pool every predicted (label, [start,end]) across N sampled rollouts per clip.
  2. Normalize labels to the 456 canonical classes (same map as the single-sample
     reparse step) so PSDS1 GT matching works.
  3. Fuse WITHIN (clip, normalized-label): cluster by temporal IoU >= 0.5
     (cluster-then-rank) OR greedy NMS at IoU >= 0.5 (headline). vote = #distinct
     seeds proposing the cluster; score = per-segment MTLA / global SVAR, optionally
     x support (vote).
  4. The fused detection set (one score per kept window) is fed to PSDS1: sweeping
     the score threshold traces the PSD-ROC. FlexSED params: dtc=gtc=0.7,
     alpha_ct=0, alpha_st=0, max_efpr=100 (imported from the verified scorer).

Per-segment MTLA = inside-window audio attention (attn_all.image_inside_sum),
SVAR/GA = global audio attention (attn_first.image_global_sum), both L8-21 mean-H
sum-L -- identical reduction to score_audioset_strong_psds_flexsed.py.

Variants computed (full video parity):
  single rollout (mean over seeds): emission, SVAR, MTLA
  cluster-then-rank: SVAR-alone, MTLA-alone, vote-alone, vote x mean {SVAR,MTLA}
  NMS (headline): NMS-{SVAR,MTLA}, NMS-{SVAR,MTLA} x support

Usage (after rollouts + per-seed attention extraction exist):
  python audioset_voting_with_attn.py --n 5
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
# Reuse the VERIFIED PSDS1 machinery (patches + FlexSED params + run_sweep).
import score_audioset_strong_psds_flexsed as P

PRED_ROOT = ROOT / "artifacts/predictions/audioset_strong_qwen3omni_sample"
ATTN_ROOT = ROOT / "artifacts/features/audioset_strong_attn_sample"
NORM_PATH = ROOT / "data/audioset_strong/label_normalize.json"
CLASSES_PATH = ROOT / "audioset_strong_456_classes.json"
CLIP_DURATION = 10.0
CLUSTER_IOU = 0.5
# Layer band for the mean-H sum-L reduction. None = ALL layers (the AudioSet
# design-doc decision: no justified prior for an audio-side band, and L8-21 was
# tuned for the 36-layer image model — Qwen3-Omni's thinker has 48 layers). Set
# via --layer_band lo-hi (e.g. 8-21 to reproduce the earlier image-band runs).
LF_BAND = None  # set in main() from --layer_band


def reduce_band(arr_dict, key):
    """[L,H] feature -> mean-over-heads sum-over-layers scalar. Uses all layers
    when LF_BAND is None, else the [lo, hi) band."""
    if arr_dict is None or key not in arr_dict:
        return 0.0
    t = arr_dict[key]
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(np.asarray(t))
    if LF_BAND is None:
        band_t = t
    else:
        band = [l for l in LF_BAND if l < t.shape[0]]
        if not band:
            return 0.0
        band_t = t[band, :]
    return float(band_t.float().mean(dim=1).sum().item())


def iou_1d(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 1e-9 else 0.0


def cluster(windows, iou_th=CLUSTER_IOU):
    """Greedy union-find clustering of 1-D windows by temporal IoU."""
    n = len(windows)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if iou_1d(windows[i], windows[j]) >= iou_th:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def load_rollouts(n_seeds):
    """Build cands[vid] = list of (label_norm, [lo,hi], mtla, svar, seed).

    Joins per-seed predictions_normalized with per-seed attention shards by
    (vid, prediction-index). Also returns gt_by_vid for PSDS1 GT construction.
    """
    label_norm = json.loads(NORM_PATH.read_text())
    classes = set(json.loads(CLASSES_PATH.read_text()))

    cands = defaultdict(list)
    gt_by_vid = {}
    seeds = list(range(n_seeds))
    for sd in seeds:
        # predictions for this seed
        preds_by_vid = {}
        pdir = PRED_ROOT / f"seed{sd}"
        for shard in sorted(pdir.glob("predictions_shard*.jsonl")):
            if shard.name.endswith(".bak"):
                continue
            with open(shard) as f:
                for line in f:
                    r = json.loads(line)
                    preds_by_vid[r["video_id"]] = r
        # attention for this seed
        attn_by_vid = {}
        adir = ATTN_ROOT / f"seed{sd}"
        for shard in sorted(adir.glob("shard*.pt")):
            for r in torch.load(shard, map_location="cpu", weights_only=False):
                attn_by_vid[r["video_id"]] = r

        n_join = 0
        for vid, pr in preds_by_vid.items():
            if vid not in gt_by_vid:
                gt_by_vid[vid] = pr.get("gt_events") or []
            attn_r = attn_by_vid.get(vid)
            if attn_r is None:
                continue
            attn_objs = attn_r.get("objects") or []
            # Normalize this rollout's raw predictions (predictions field is raw).
            raw_preds = pr.get("predictions")
            if not isinstance(raw_preds, list):
                continue
            for pi, p in enumerate(raw_preds):
                if pi >= len(attn_objs):
                    break
                raw_label = (p.get("label") or "").strip()
                if not raw_label:
                    continue
                canonical = label_norm.get(raw_label, raw_label)
                if canonical not in classes:
                    continue  # OOV -> not matchable to GT, drop (counts as the model's miss)
                bbox = p.get("bbox_2d") or []
                if len(bbox) != 2:
                    continue
                lo, hi = float(bbox[0]), float(bbox[1])
                if hi <= lo:
                    continue
                ao = attn_objs[pi]
                mtla = reduce_band(ao.get("attn_all"), "image_inside_sum")
                svar = reduce_band(ao.get("attn_first"), "image_global_sum")
                cands[vid].append((canonical, [lo, hi], mtla, svar, sd))
                n_join += 1
        print(f"  seed{sd}: joined {n_join} (pred,attn) candidates", flush=True)
    return cands, gt_by_vid


# ---- fusion -> detection rows (one score per kept window) ------------------
def fuse_clusters(cands, score_fn):
    """cluster-then-rank within (vid,label); representative = mean window;
    emit (vid, label, lo, hi, score)."""
    rows = []
    for vid, lst in cands.items():
        by_label = defaultdict(list)
        for (lab, w, m, s, sd) in lst:
            by_label[lab].append((w, m, s, sd))
        for lab, members in by_label.items():
            windows = [x[0] for x in members]
            for grp in cluster(windows):
                ms = [members[i] for i in grp]
                mean_w = np.mean([x[0] for x in ms], axis=0).tolist()
                vote = len({x[3] for x in ms})
                mean_mtla = float(np.mean([x[1] for x in ms]))
                mean_svar = float(np.mean([x[2] for x in ms]))
                sc = score_fn(vote, mean_mtla, mean_svar)
                rows.append((vid, lab, mean_w[0], mean_w[1], sc))
    return rows


def fuse_nms(cands, key, use_support=False, iou_th=CLUSTER_IOU):
    """Greedy within-(vid,label) NMS over pooled windows. Keeps the actual
    best-scoring window; if use_support, score *= #distinct seeds absorbed."""
    rows = []
    for vid, lst in cands.items():
        by_label = defaultdict(list)
        for (lab, w, m, s, sd) in lst:
            by_label[lab].append((w, m, s, sd))
        for lab, members in by_label.items():
            scored = [(w, (m if key == "mtla" else s), sd) for (w, m, s, sd) in members]
            order = sorted(range(len(scored)), key=lambda i: -scored[i][1])
            taken = [False] * len(scored)
            for i in order:
                if taken[i]:
                    continue
                taken[i] = True
                w_i, sc_i, _ = scored[i]
                seeds = {scored[i][2]}
                for j in order:
                    if taken[j] or j == i:
                        continue
                    if iou_1d(w_i, scored[j][0]) >= iou_th:
                        taken[j] = True
                        seeds.add(scored[j][2])
                final = (len(seeds) * sc_i) if use_support else sc_i
                rows.append((vid, lab, w_i[0], w_i[1], final))
    return rows


def single_rollout_rows(cands, seed, key):
    """One seed's raw predictions, scored by key (mtla/svar/emit)."""
    rows = []
    for vid, lst in cands.items():
        ei = 0
        for (lab, w, m, s, sd) in lst:
            if sd != seed:
                continue
            sc = m if key == "mtla" else (s if key == "svar" else -ei)
            rows.append((vid, lab, w[0], w[1], sc))
            ei += 1
    return rows


# ---- PSDS1 scoring of a detection-row set ----------------------------------
def build_gt_df(gt_by_vid, vids):
    import pandas as pd
    gt_records, meta_records = [], []
    for vid in vids:
        meta_records.append({"filename": vid, "duration": CLIP_DURATION})
        by_class = defaultdict(list)
        for ev in gt_by_vid.get(vid) or []:
            cls = ev.get("event_name")
            if cls is None:
                continue
            by_class[cls].append((float(ev["start"]), float(ev["end"])))
        for cls, intervals in by_class.items():
            intervals.sort()
            merged = []
            for s, e in intervals:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            for s, e in merged:
                gt_records.append({"filename": vid, "onset": s, "offset": e,
                                   "event_label": cls})
    return pd.DataFrame(gt_records), pd.DataFrame(meta_records)


def psds1_for_rows(rows, gt_df, metadata_df, classes_in_gt, n_thresholds=0):
    """Compute FlexSED PSDS1 for a detection-row set via the verified machinery.

    rows: list of (vid, label, lo, hi, score). We adapt them into P.run_sweep's
    'rows' format ({vid,label,lo,hi,<scorekey>}) with a single score column.
    n_thresholds=0 -> exact threshold-independent (all distinct scores, slow);
    >0 -> that many quantile thresholds (fast; converges to within ~0.001 by 100,
    fine for an intermediate directional read)."""
    classes_in_gt_set = set(classes_in_gt)
    sweep_rows = []
    for (vid, lab, lo, hi, sc) in rows:
        if lab not in classes_in_gt_set:
            continue
        sweep_rows.append({"vid": vid, "label": lab, "lo": lo, "hi": hi,
                           "score": float(sc)})
    if not sweep_rows:
        return None
    res = P.run_sweep([("fused", "score")], sweep_rows, gt_df, metadata_df,
                      classes_in_gt, n_thresholds=n_thresholds)
    return res.get("fused")


def main():
    global LF_BAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of seeds/rollouts")
    ap.add_argument("--n_thresholds", type=int, default=0,
                    help="PSDS sweep points per variant. 0=exact (all distinct "
                         "scores, slow); 100=fast directional read (converges to "
                         "~0.001). Use 100 for intermediate checks, 0 for final.")
    ap.add_argument("--layer_band", type=str, default="all",
                    help="'all' (design-doc default, all 48 layers) or 'lo-hi' "
                         "(e.g. 8-21 to reproduce the image-band runs).")
    args = ap.parse_args()
    NTH = args.n_thresholds
    if args.layer_band == "all":
        LF_BAND = None
    else:
        lo, hi = (int(x) for x in args.layer_band.split("-"))
        LF_BAND = list(range(lo, hi + 1))  # inclusive hi, matches L8-21 = 8..21
    print(f"layer band = {args.layer_band} ({'all layers' if LF_BAND is None else LF_BAND})",
          flush=True)

    print(f"loading {args.n} rollouts ...", flush=True)
    cands, gt_by_vid = load_rollouts(args.n)
    vids = sorted(cands.keys())
    print(f"clips with candidates: {len(vids)}", flush=True)

    gt_df, metadata_df = build_gt_df(gt_by_vid, vids)
    classes_in_gt = sorted(gt_df["event_label"].unique().tolist())
    print(f"GT events: {len(gt_df)}; classes: {len(classes_in_gt)}; clips: {len(metadata_df)}",
          flush=True)

    def score(name, rows):
        v = psds1_for_rows(rows, gt_df, metadata_df, classes_in_gt, n_thresholds=NTH)
        print(f"{name:<26s} {('--' if v is None else f'{v:.4f}'):>9s}", flush=True)

    print("\n" + "=" * 50)
    print(f"AudioSet-Strong voting  N={args.n}  PSDS1 (dtc=gtc=0.7, alpha_st=0)"
          f"  [sweep={'exact' if NTH == 0 else NTH}, band={args.layer_band}]")
    print(f"{'variant':<26s} {'PSDS1':>9s}")
    print("-" * 50)

    # single-rollout references (mean over seeds)
    for key in ("emit", "svar", "mtla"):
        vals = []
        for sd in range(args.n):
            v = psds1_for_rows(single_rollout_rows(cands, sd, key),
                               gt_df, metadata_df, classes_in_gt, n_thresholds=NTH)
            if v is not None:
                vals.append(v)
        mv = f"{np.mean(vals):.4f}" if vals else "--"
        print(f"{'single ' + key:<26s} {mv:>9s}", flush=True)

    # cluster-then-rank
    score("clust SVAR-alone", fuse_clusters(cands, lambda v, m, s: s))
    score("clust MTLA-alone", fuse_clusters(cands, lambda v, m, s: m))
    score("clust vote-alone", fuse_clusters(cands, lambda v, m, s: v))
    score("clust vote x SVAR", fuse_clusters(cands, lambda v, m, s: v * s))
    score("clust vote x MTLA", fuse_clusters(cands, lambda v, m, s: v * m))
    # NMS (headline)
    score("NMS-SVAR", fuse_nms(cands, "svar"))
    score("NMS-MTLA", fuse_nms(cands, "mtla"))
    score("NMS-SVAR x support", fuse_nms(cands, "svar", use_support=True))
    score("NMS-MTLA x support", fuse_nms(cands, "mtla", use_support=True))
    print("=" * 50)


if __name__ == "__main__":
    main()
