"""Score COCO detections with MTLA: per-prediction AUROC + self-consistency voting mAP.

Consumes the outputs of ``generate.py`` (predictions JSON per seed) and ``extract.py``
(attention feature shards per seed), and reports:

  * **Hallucination AUROC** (single seed): how well MTLA vs. the SVAR baseline separate
    grounded from hallucinated boxes.
  * **Detection mAP** (N seeds): pool boxes across rollouts, fuse overlaps with NMS, and
    evaluate with official ``pycocotools``. COCO uses **sum**-of-cluster fusion (each image
    yields many boxes, so rewarding boxes that recur across rollouts helps); this is the one
    benchmark where sum beats max.

Example:
    python score.py \
        --features_root  /path/to/features/coco_internvl \
        --predictions_root /path/to/predictions/coco_internvl \
        --coco_gt /path/to/instances_val2017.json \
        --n 16 --agg sum
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from mtla import DEFAULT_BAND, auroc, coco_map, nms_fuse, reduce_band


def load_seed(features_dir, predictions_path):
    """Return (preds_by_id, attn_by_(id,pred_idx)) for one seed.

    attn maps (image_id, pred_idx) -> (mtla, svar, is_hallucinated); the hallucination
    flag is computed during extraction and stored in the feature shards.
    """
    preds = {r["id"]: r for r in json.load(open(predictions_path))}
    attn = {}
    for sp in sorted(glob.glob(f"{features_dir}/shard*.pt")):
        for r in torch.load(sp, weights_only=False, map_location="cpu"):
            for o in r["objects"]:
                cm = o.get("attn_coord_mean", {})
                attn[(r["image_id"], o["pred_idx"])] = (
                    reduce_band(cm.get("image_inside_sum"), DEFAULT_BAND),  # MTLA
                    reduce_band(cm.get("image_sum"), DEFAULT_BAND),         # SVAR
                    bool(o.get("is_hallucinated", False)),
                )
    return preds, attn


def seed_dir(root, seed, sub):
    """Locate a per-seed directory, tolerating a few common layouts."""
    for cand in (os.path.join(root, f"seed{seed}", sub),
                 os.path.join(root, f"seed{seed}"),
                 root if seed == 0 else os.path.join(root, f"seed{seed}")):
        if os.path.exists(cand):
            return cand
    return os.path.join(root, f"seed{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", required=True,
                    help="dir containing seed{K}/shard*.pt attention shards")
    ap.add_argument("--predictions_root", required=True,
                    help="dir containing seed{K}/temp_0/predictions.json")
    ap.add_argument("--coco_gt", required=True, help="instances_val2017.json")
    ap.add_argument("--n", type=int, default=16, help="number of rollouts (seeds) to vote over")
    ap.add_argument("--agg", default="sum", choices=["sum", "max", "support", "mean"],
                    help="cluster fusion rule (COCO headline = sum)")
    args = ap.parse_args()

    seeds = list(range(args.n))
    data = []
    for s in seeds:
        fdir = seed_dir(args.features_root, s, "")
        pj = os.path.join(seed_dir(args.predictions_root, s, "temp_0"), "predictions.json")
        data.append(load_seed(fdir, pj))
        print(f"  loaded seed {s}: {len(data[-1][0])} images")

    # ---- single-seed hallucination AUROC (seed 0) ----
    preds0, attn0 = data[0]
    mtla = [v[0] for v in attn0.values()]
    svar = [v[1] for v in attn0.values()]
    labels = [v[2] for v in attn0.values()]
    if mtla:
        print(f"\nHallucination AUROC (seed 0, {len(mtla)} preds)")
        print(f"   MTLA = {auroc(mtla, labels):.4f}")
        print(f"   SVAR = {auroc(svar, labels):.4f}")

    # ---- N-seed voting mAP ----
    gt = json.load(open(args.coco_gt))
    name2cat = {c["name"].lower(): c["id"] for c in gt["categories"]}
    img_wh = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    image_ids = list(preds0.keys())

    detections = []
    for iid in image_ids:
        # pool candidate boxes from all seeds, per label
        by_label = defaultdict(list)
        for s, (preds, attn) in enumerate(data):
            rec = preds.get(iid)
            if rec is None or rec.get("status") != "success":
                continue
            for pi, pb in enumerate(rec.get("pred_bboxes", [])):
                box, label = pb.get("box"), (pb.get("label") or "").strip().lower()
                if not (label and box and len(box) == 4):
                    continue
                mtla_score = attn.get((iid, pi), (0.0, 0.0, False))[0]
                by_label[label].append((box, mtla_score, s))
        # fuse each label group and emit COCO detections (normalized [0,1000] -> pixels)
        W, H = img_wh.get(iid, (1, 1))
        for label, cands in by_label.items():
            cid = name2cat.get(label)
            if cid is None:
                continue
            for box, score in nms_fuse(cands, agg=args.agg):
                x1, y1, x2, y2 = box
                detections.append({
                    "image_id": iid, "category_id": cid,
                    "bbox": [x1 / 1000 * W, y1 / 1000 * H,
                             (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H],
                    "score": float(score),
                })

    res = coco_map(detections, args.coco_gt)
    print(f"\nDetection mAP (N={args.n}, agg={args.agg}, {len(detections)} dets)")
    for k in ("mAP", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large"):
        print(f"   {k:<10} {res[k]:.2f}")


if __name__ == "__main__":
    main()
