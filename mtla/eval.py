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


def auroc_from_records(objects, signal: str = "local_attention", band=DEFAULT_BAND) -> float:
    """AUROC straight from a list of extracted prediction objects.

    Each object is a dict with ``object[signal]`` a ``[L, H]`` attention array and a boolean
    ``object["is_hallucinated"]``. ``signal="local_attention"`` (default) is MTLA;
    ``signal="first_digit"`` reads the single first-coordinate-digit token.
    """
    scores = [reduce_band(o[signal], band) for o in objects]
    labels = [bool(o["is_hallucinated"]) for o in objects]
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
