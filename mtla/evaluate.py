"""Score stage: turn voted candidates into benchmark numbers.

This module owns the metric-assembly half of the score stage; the pipeline itself is composed
step-by-step in ``score.py`` (load + score shards → hallucination AUROC → ``mtla.voting.vote`` →
metric). Two things live here:

  * ``hallucination_auroc`` — the single-rollout detection score reported for every benchmark
    (how well the MTLA score separates grounded from hallucinated predictions);
  * ``compute_metric`` + the ``_coco`` / ``_moment_retrieval`` / ``_recall`` handlers — turn the
    voted candidate groups into the benchmark's task metric, dispatching on the dataset's declared
    ``metric``.

This is the only dataset-shaped part of scoring: adding a dataset that reuses an existing metric
needs no change here; a genuinely new metric adds one pure computer in ``mtla.metrics`` and one
handler below. The candidates are the complete input — nothing is re-parsed and no model is
loaded, so this stage is CPU-only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING

from mtla.metrics import auroc, coco_map, moment_retrieval, recall_at_iou
from mtla.types import FusedGroups, ItemId, ScoredCand

if TYPE_CHECKING:
    from mtla.config import RunConfig
    from mtla.data.base import DatasetAdapter


def compute_metric(
    cfg: "RunConfig",
    dataset: "DatasetAdapter",
    voted: FusedGroups,
    gt_by_id: dict[ItemId, list],
) -> dict:
    """Dispatch the voted candidates to the dataset's benchmark metric handler.

    Looks up the handler named by ``dataset.metric`` (see the ``_coco`` / ``_moment_retrieval`` /
    ``_recall`` handlers below) and runs it over the voted groups. This is the only dataset-shaped
    step of the score stage.

    Args:
        cfg: the run config (passed through to the handler, e.g. for the COCO GT path).
        dataset: the dataset adapter, supplying ``metric`` (which handler to run).
        voted: the voted groups from ``mtla.voting.vote``, keyed by ``(id, label)``.
        gt_by_id: per-item ground truth from ``load_candidates``.

    Returns:
        The benchmark-specific metrics dict from the handler.
    """
    return _METRICS[dataset.metric](cfg, voted, gt_by_id)


def hallucination_auroc(cands: list[ScoredCand]) -> float:
    """Single-rollout hallucination AUROC over the extracted candidates.

    Restricts to rollout 0 (the deterministic anchor) and to candidates that actually
    had attention extracted, then measures how well the MTLA score separates grounded
    from hallucinated predictions. This is the detection metric reported per benchmark,
    independent of the voting path.

    Args:
        cands: the flattened candidates from ``load_candidates`` (each carries
            ``score``, ``hallu``, ``seed``, and ``extracted``).

    Returns:
        The AUROC in ``[0, 1]`` (positive class is grounded), or ``nan`` when no
        seed-0 extracted candidate exists.
    """
    s = [c["score"] for c in cands if c["seed"] == 0 and c["extracted"]]
    y = [c["hallu"] for c in cands if c["seed"] == 0 and c["extracted"]]
    return auroc(s, y) if s else float("nan")


# ---------------------------------------------------------------------------
# Metric handlers: voted candidates -> metric dict. One per metric name.
# ---------------------------------------------------------------------------
def _coco(cfg: "RunConfig", fused: FusedGroups, gt_by_id: dict[ItemId, list]) -> dict:
    """Assemble COCO detections from the fused groups and score bbox mAP.

    Maps each fused ``(image_id, label)`` group to COCO detection dicts, rescaling boxes
    from the model's normalized ``[0, 1000]`` frame to the image's absolute pixel size
    and to ``[x, y, w, h]``, dropping labels not in the COCO category set. Delegates the
    actual scoring to ``metrics.coco_map``.

    Args:
        cfg: the run config, supplying the ``coco_gt`` annotations path.
        fused: the voted groups from ``mtla.voting.vote``, keyed by ``(image_id, label)``.
        gt_by_id: per-item ground truth (unused here; COCO scores against the GT JSON).

    Returns:
        A dict ``{"map": <coco_map result>, "n_dets": <count>}`` where ``map`` holds the
        mAP breakdown and ``n_dets`` is the number of assembled detections.
    """
    gt = json.load(open(cfg.path("coco_gt")))
    name2cat = {c["name"].lower(): c["id"] for c in gt["categories"]}
    img_wh = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    detections = []
    for (image_id, label), kept in fused.items():
        cid = name2cat.get((label or "").strip().lower())
        if cid is None:
            continue
        W, H = img_wh.get(image_id, (1, 1))
        for (x1, y1, x2, y2), score in kept:
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [
                        x1 / 1000 * W,
                        y1 / 1000 * H,
                        (x2 - x1) / 1000 * W,
                        (y2 - y1) / 1000 * H,
                    ],
                    "score": float(score),
                }
            )
    return {"map": coco_map(detections, cfg.path("coco_gt")), "n_dets": len(detections)}


def _moment_retrieval(
    cfg: "RunConfig", fused: FusedGroups, gt_by_id: dict[ItemId, list]
) -> dict:
    """Assemble a ranked moment-retrieval submission per query and score mAP / R@1.

    Collapses the fused groups by query id, ranks each query's windows by score, keeps
    the top ``MAX_WINDOWS``, and pairs them with the query's ground-truth windows in the
    format the official Moment-DETR evaluator expects. Delegates scoring to
    ``metrics.moment_retrieval``.

    Args:
        cfg: the run config (unused here; kept for the uniform handler signature).
        fused: the voted groups from ``mtla.voting.vote``, keyed by ``(qid, label)``.
        gt_by_id: per-query ground-truth relevant windows, keyed by qid.

    Returns:
        A dict ``{"nms_mtla": <moment_retrieval result>}`` with the mAP / R@1 metrics.
    """
    MAX_WINDOWS = 10
    by_qid = defaultdict(list)
    for (qid, _label), kept in fused.items():
        by_qid[qid].extend(kept)
    submission, ground_truth = [], []
    for qid, kept in by_qid.items():
        rows = [[w[0], w[1], float(sc)] for w, sc in sorted(kept, key=lambda k: -k[1])][
            :MAX_WINDOWS
        ]
        submission.append(
            {"qid": qid, "pred_relevant_windows": rows or [[0.0, 0.0, 0.0]]}
        )
        ground_truth.append(
            {"qid": qid, "relevant_windows": [g["region"] for g in gt_by_id[qid]]}
        )
    return {"nms_mtla": moment_retrieval(submission, ground_truth)}


def _recall(cfg: "RunConfig", fused: FusedGroups, gt_by_id: dict[ItemId, list]) -> dict:
    """Score single-span temporal grounding: one selected span per query vs its GT.

    Takes the single top span each query kept (``argmax`` selection upstream) and pairs
    it with that query's ground-truth span, then delegates to ``metrics.recall_at_iou``
    for R@1 at IoU thresholds and mIoU. A query with no kept span or no GT contributes a
    miss.

    Args:
        cfg: the run config (unused here; kept for the uniform handler signature).
        fused: the voted groups from ``mtla.voting.vote``, keyed by ``(qid, label)``.
        gt_by_id: per-query ground-truth regions, keyed by qid.

    Returns:
        A dict ``{"mtla": <recall_at_iou result>}`` with the recall / mIoU metrics.
    """
    pred_spans, gt_spans = [], []
    for (qid, _label), kept in fused.items():
        pred_spans.append(kept[0][0] if kept else None)
        gt = gt_by_id.get(qid) or []
        gt_spans.append(gt[0]["region"] if gt else None)
    return {"mtla": recall_at_iou(pred_spans, gt_spans)}


_METRICS = {
    "coco_map": _coco,
    "moment_retrieval": _moment_retrieval,
    "recall_at_iou": _recall,
}
