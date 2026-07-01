"""Metrics: pure computers that turn scored predictions into benchmark numbers.

Each function takes already-assembled inputs (scores + labels, or fused detections + ground truth)
and returns numbers. They do **no** shard loading, band reduction, self-consistency voting, or NMS
— that orchestration lives in ``mtla.evaluate`` and runs before these are called. Keeping the
metrics pure makes them reusable across datasets and trivially testable on synthetic inputs.

Protocols used in the paper:
  * ``auroc``            — hallucination detection on any benchmark: how well the MTLA score
                           separates grounded from hallucinated predictions.
  * ``coco_map``         — COCO detection mAP via the official ``pycocotools`` ``COCOeval``.
  * ``moment_retrieval`` — QVHighlights moment-retrieval mAP / R@1 via the official Moment-DETR
                           evaluator vendored under ``third_party/moment_detr_eval``.
  * ``recall_at_iou``    — single-span temporal grounding (Charades): R@1 @ IoU thresholds + mIoU.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Vendored official Moment-DETR evaluator (QVHighlights).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "third_party"))
from moment_detr_eval.eval import compute_mr_ap, compute_mr_r1

from .utils import tiou


def auroc(scores, is_hallucinated) -> float:
    """AUROC of MTLA scores vs. hallucination labels.

    Args:
        scores: per-prediction MTLA scores (higher = more grounded).
        is_hallucinated: per-prediction bool/int, ``True`` for a hallucination.

    Returns:
        AUROC in ``[0, 1]``; the positive class is *grounded* (``not is_hallucinated``).
    """
    y = 1 - np.asarray(is_hallucinated, dtype=np.int32)   # 1 = grounded (positive)
    s = np.asarray(scores, dtype=np.float64)
    return float(roc_auc_score(y, s))


def coco_map(detections, gt_json_path: str) -> dict:
    """COCO bbox mAP for a list of detections.

    Args:
        detections: list of ``{"image_id", "category_id", "bbox": [x,y,w,h], "score"}`` in absolute
            pixel coordinates (COCO format).
        gt_json_path: path to ``instances_val2017.json``.

    Returns:
        dict with ``mAP, mAP50, mAP75, AP_small, AP_medium, AP_large`` (each *100).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json_path)
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP": s[0] * 100, "mAP50": s[1] * 100, "mAP75": s[2] * 100,
            "AP_small": s[3] * 100, "AP_medium": s[4] * 100, "AP_large": s[5] * 100}


def moment_retrieval(submission, ground_truth) -> dict:
    """QVHighlights moment-retrieval metrics via the official Moment-DETR evaluator.

    Args:
        submission: list of ``{"qid", "pred_relevant_windows": [[t0, t1, score], ...]}`` (ranked).
        ground_truth: list of ``{"qid", "relevant_windows": [[t0, t1], ...]}``.

    Returns:
        dict with ``mAP, mAP@0.5, mAP@0.75, R1@0.5, R1@0.7`` (each *100 where the evaluator does so).
    """
    apd = compute_mr_ap(submission, ground_truth, num_workers=8)
    r1 = compute_mr_r1(submission, ground_truth)
    return {"mAP": apd["average"], "mAP@0.5": apd["0.5"], "mAP@0.75": apd["0.75"],
            "R1@0.5": r1["0.5"], "R1@0.7": r1["0.7"]}


def recall_at_iou(pred_spans, gt_spans, thresholds=(0.3, 0.5, 0.7)) -> dict:
    """Single-span temporal grounding metrics (Charades): R@1 at IoU thresholds + mIoU.

    Args:
        pred_spans: one predicted ``[t0, t1]`` per query (the selected span).
        gt_spans: the matching ground-truth ``[t0, t1]`` per query (``None`` counts as a miss).
        thresholds: IoU thresholds for R@1.

    Returns:
        dict with ``R@{thr}`` (percent) for each threshold and ``mIoU``.
    """
    ious = np.array([tiou(p, g) if (p is not None and g is not None) else 0.0
                     for p, g in zip(pred_spans, gt_spans)])
    out = {f"R@{thr}": float(100 * np.mean(ious >= thr)) for thr in thresholds}
    out["mIoU"] = float(ious.mean()) if len(ious) else 0.0
    return out
