"""COCO detection dataset adapter.

Scoring (CPU): hallucination AUROC (single seed) + detection mAP after self-consistency voting
(sum-of-cluster fusion, the COCO headline).

Reproduces (InternVL3.5-8B, val2017): AUROC 0.873 (MTLA) / 0.803 (SVAR); mAP 41.9 at N=16.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from .base import DatasetAdapter
from ..score import apply_slot
from ..eval import auroc, coco_map
from ..voting import nms_fuse

# 80 COCO class names (the open-vocab detection prompt lists these).
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

PROMPT = (
    "Please detect all instances of {cats} in the image. "
    "Output the bounding boxes in the format <ref>category</ref><box>[[x1, y1, x2, y2], ...]</box> "
    "with coordinates normalized to [0, 1000]."
)


class CocoDataset(DatasetAdapter):
    name = "coco"
    task = "image_det"

    def load_items(self, cfg):
        return json.load(open(cfg.path("data")))

    def prompt(self, item):
        cats = ", ".join(item.get("categories", COCO_CLASSES))
        return PROMPT.format(cats=cats)

    def ground_truth(self, item):
        return json.loads(item["conversations"][1]["value"])

    # ---- GPU stages: which script comes from the MODEL adapter (so COCO is model-agnostic) ----
    def generate(self, cfg, model, seed=0):
        from ..stages import run_stage
        script = model.generate_script(self.task, cfg.generate.engine)
        args = [
            "--model", model.model_id,
            "--dataset", cfg.path("data"),
            "--output_dir", os.path.join(cfg.path("predictions"), f"seed{seed}"),
            "--gpu_ids", *[str(g) for g in cfg.generate.gpus],
            "--temperature", str(cfg.generate.temperature),
            "--seed", str(seed),
        ]
        if cfg.generate.n_items:
            args += ["--limit", str(cfg.generate.n_items)]
        run_stage(script, args)

    def extract(self, cfg, model, seed=0):
        from ..stages import run_stage
        run_stage(model.extract_script(self.task), [
            "--pred_file", os.path.join(cfg.path("predictions"), f"seed{seed}", "predictions.json"),
            "--dataset", cfg.path("data"),
            "--out_dir", os.path.join(cfg.path("features"), f"seed{seed}"),
            "--gpus", *[str(g) for g in cfg.extract.gpus],
            "--n_images", str(cfg.extract.n_items or 5000),
        ])

    # ---- scoring (signal slots come from the model adapter, not hardcoded) ----
    def _load_seed(self, features_dir, predictions_path, band, mtla_spec, svar_spec):
        preds = {r["id"]: r for r in json.load(open(predictions_path))}
        attn = {}
        for sp in sorted(glob.glob(f"{features_dir}/shard*.pt")):
            import torch
            for r in torch.load(sp, weights_only=False, map_location="cpu"):
                for o in r["objects"]:
                    attn[(r["image_id"], o["pred_idx"])] = (
                        apply_slot(o, mtla_spec, band),   # MTLA (inside-region)
                        apply_slot(o, svar_spec, band),   # SVAR (global baseline)
                        bool(o.get("is_hallucinated", False)),
                    )
        return preds, attn

    def score(self, cfg, model) -> dict:
        band = cfg.band_indices()
        n = cfg.score.n_rollouts
        agg = cfg.score.agg
        mtla_spec = model.mtla_slot(self.task, cfg.score.slot)
        svar_spec = model.svar_slot(self.task)
        features_root = cfg.path("features")
        predictions_root = cfg.path("predictions")
        coco_gt = cfg.path("coco_gt")

        data = []
        for s in range(n):
            fdir = os.path.join(features_root, f"seed{s}")
            pj = os.path.join(predictions_root, f"seed{s}", "predictions.json")
            data.append(self._load_seed(fdir, pj, band, mtla_spec, svar_spec))
            print(f"  loaded seed {s}: {len(data[-1][0])} images")

        # single-seed hallucination AUROC
        _, attn0 = data[0]
        mtla = [v[0] for v in attn0.values()]
        svar = [v[1] for v in attn0.values()]
        labels = [v[2] for v in attn0.values()]
        auroc_mtla = auroc(mtla, labels) if mtla else float("nan")
        auroc_svar = auroc(svar, labels) if svar else float("nan")
        print(f"\nHallucination AUROC (seed 0, {len(mtla)} preds)")
        print(f"   MTLA = {auroc_mtla:.4f}")
        print(f"   SVAR = {auroc_svar:.4f}")

        # N-seed voting mAP
        gt = json.load(open(coco_gt))
        name2cat = {c["name"].lower(): c["id"] for c in gt["categories"]}
        img_wh = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
        preds0 = data[0][0]

        detections = []
        for iid in preds0:
            by_label = defaultdict(list)
            for s, (preds, attn) in enumerate(data):
                rec = preds.get(iid)
                if rec is None or rec.get("status") != "success":
                    continue
                for pi, pb in enumerate(rec.get("pred_bboxes", [])):
                    box, label = pb.get("box"), (pb.get("label") or "").strip().lower()
                    if not (label and box and len(box) == 4):
                        continue
                    score = attn.get((iid, pi), (0.0, 0.0, False))[0]
                    by_label[label].append((box, score, s))
            W, H = img_wh.get(iid, (1, 1))
            for label, cands in by_label.items():
                cid = name2cat.get(label)
                if cid is None:
                    continue
                for box, sc in nms_fuse(cands, agg=agg):
                    x1, y1, x2, y2 = box
                    detections.append({
                        "image_id": iid, "category_id": cid,
                        "bbox": [x1 / 1000 * W, y1 / 1000 * H,
                                 (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H],
                        "score": float(sc),
                    })

        res = coco_map(detections, coco_gt)
        print(f"\nDetection mAP (N={n}, agg={agg}, {len(detections)} dets)")
        for k in ("mAP", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large"):
            print(f"   {k:<10} {res[k]:.2f}")
        return {"auroc_mtla": auroc_mtla, "auroc_svar": auroc_svar, **res}
