"""COCO detection dataset adapter.

Scoring (CPU): hallucination AUROC (single seed) + detection mAP after self-consistency voting
(sum-of-cluster fusion, the COCO headline).

Reproduces (InternVL3.5-8B, val2017): AUROC 0.873 (MTLA) / 0.803 (SVAR); mAP 41.9 at N=16.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from .base import DatasetAdapter
from ..registry import register_dataset
from ..score import mtla_score
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


@register_dataset("coco")
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

    # ---- GPU stages: the MODEL adapter names the script (so COCO is model-agnostic) ----
    # Every stage script is config-driven: it reloads the config, resolves (model, dataset) from
    # the registry, and asks this adapter for its items via `load_items(cfg)`. So the command is
    # uniform — just the config path and the rollout seed.
    def stage_cmd(self, cfg, model, seed, mode):
        args = ["--config", cfg.config_path, "--seed", str(seed)]
        if mode == "generate":
            return model.generate_script(self.task, cfg.generate.engine), args
        return model.extract_script(self.task), args

    # ---- scoring (MTLA = reduce_band(local_attention); returns a dict, no prints) ----
    def _load_seed(self, features_dir, predictions_path, band):
        preds = {r["id"]: r for r in json.load(open(predictions_path))}
        attn = {}
        for r in self.load_shards(features_dir):
            for o in r["objects"]:
                attn[(r["image_id"], o["pred_idx"])] = (
                    mtla_score(o, band=band),
                    bool(o.get("is_hallucinated", False)),
                )
        return preds, attn

    def score(self, cfg, model) -> dict:
        band = cfg.band_indices()
        n = cfg.n_rollouts
        agg = cfg.score.agg
        coco_gt = cfg.path("coco_gt")

        data = [self._load_seed(cfg.feat_dir(s),
                                os.path.join(cfg.pred_dir(s), "predictions.json"),
                                band)
                for s in range(n)]

        # single-seed hallucination AUROC
        attn0 = data[0][1]
        auroc_mtla = auroc([v[0] for v in attn0.values()], [v[1] for v in attn0.values()]) if attn0 else float("nan")

        # N-seed voting mAP
        gt = json.load(open(coco_gt))
        name2cat = {c["name"].lower(): c["id"] for c in gt["categories"]}
        img_wh = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}

        detections = []
        for iid in data[0][0]:
            by_label = defaultdict(list)
            for s, (preds, attn) in enumerate(data):
                rec = preds.get(iid)
                if rec is None or rec.get("status") != "success":
                    continue
                for pi, pb in enumerate(rec.get("pred_bboxes", [])):
                    box, label = pb.get("box"), (pb.get("label") or "").strip().lower()
                    if not (label and box and len(box) == 4):
                        continue
                    sc = attn.get((iid, pi), (0.0, False))[0]
                    by_label[label].append((box, sc, s))
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

        return {"auroc_mtla": auroc_mtla,
                "map": coco_map(detections, coco_gt), "n_dets": len(detections),
                "n_rollouts": n, "agg": agg}
