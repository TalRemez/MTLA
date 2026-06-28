"""Evaluation helpers: hallucination AUROC and COCO mAP from MTLA scores.

Two protocols are used in the paper:

  * **Hallucination detection** (any benchmark): treat each prediction as positive when it
    matches a ground-truth region (IoU >= 0.5) and negative otherwise, then measure how well
    the MTLA score separates the two with ``roc_auc_score``.
  * **Detection mAP** (COCO): rank fused predictions by their MTLA score and evaluate with the
    official ``pycocotools`` ``COCOeval``.

QVHighlights mAP / R@1 uses the official Moment-DETR ``standalone_eval`` vendored under
``third_party/moment_detr_eval`` (imported by the QVHighlights example, not here).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

from .score import DEFAULT_BAND, reduce_band


def auroc(scores, is_hallucinated) -> float:
    """AUROC of MTLA scores vs. hallucination labels.

    Args:
        scores: per-prediction MTLA scores (higher = more grounded).
        is_hallucinated: per-prediction bool/int, ``True`` for a hallucination.

    Returns:
        AUROC in ``[0, 1]``; the positive class is *grounded* (``not is_hallucinated``).
    """
    from sklearn.metrics import roc_auc_score

    y = 1 - np.asarray(is_hallucinated, dtype=np.int32)  # 1 = grounded (positive)
    s = np.asarray(scores, dtype=np.float64)
    return float(roc_auc_score(y, s))


def auroc_from_records(records, slot: str = "attn_coord_mean",
                       stat: str = "image_inside_sum", band=DEFAULT_BAND) -> float:
    """AUROC straight from a list of extracted prediction records.

    Each record is a dict with ``record[slot][stat]`` a ``[L, H]`` attention aggregate and a
    boolean ``record["is_hallucinated"]``. Use ``stat="image_inside_sum"`` for MTLA (default)
    or ``stat="image_sum"`` for the SVAR baseline.
    """
    scores = [reduce_band(r[slot][stat], band) for r in records]
    labels = [bool(r["is_hallucinated"]) for r in records]
    return auroc(scores, labels)


def coco_map(detections, gt_json_path: str) -> dict:
    """COCO bbox mAP for a list of detections.

    Args:
        detections: list of ``{"image_id", "category_id", "bbox": [x,y,w,h], "score"}`` in
            absolute pixel coordinates (COCO format).
        gt_json_path: path to ``instances_val2017.json``.

    Returns:
        dict with ``mAP, mAP50, mAP75, AP_small, AP_medium, AP_large`` (each *100).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json_path)
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"mAP": s[0] * 100, "mAP50": s[1] * 100, "mAP75": s[2] * 100,
            "AP_small": s[3] * 100, "AP_medium": s[4] * 100, "AP_large": s[5] * 100}
