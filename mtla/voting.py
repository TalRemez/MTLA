"""Self-consistency voting: pool predictions from N rollouts, fuse overlaps, re-score.

Sampling ``N`` stochastic rollouts per input enlarges the candidate pool (recall); we then
merge overlapping predictions with non-maximum suppression and assign each kept prediction
a fused score from its cluster's MTLA scores. The ``agg`` argument selects the fusion rule:

  * ``"max"``     keep the single highest-scoring rollout         (default; video / audio)
  * ``"sum"``     sum the cluster's scores (rewards recurrence)   (COCO detection headline)
  * ``"support"`` ``#distinct seeds * max``                        (ablation)
  * ``"mean"``    average of the cluster's scores                 (ablation)

Works for both spatial boxes (use ``iou``) and temporal spans (use ``tiou``); pass the
matching overlap function via ``iou_fn``.
"""
from __future__ import annotations

from collections import defaultdict

CLUSTER_IOU = 0.5  # overlap threshold for "same prediction" across rollouts


def iou(b1, b2) -> float:
    """Spatial IoU of two ``[x1,y1,x2,y2]`` boxes."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def tiou(a, b) -> float:
    """Temporal IoU of two ``[t_start, t_end]`` spans."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 1e-9 else 0.0


def _fuse(agg: str, rep_score: float, member_scores, n_seeds: int) -> float:
    if agg == "max":
        return rep_score
    if agg == "support":
        return n_seeds * rep_score
    if agg == "sum":
        return sum(member_scores)
    if agg == "mean":
        return sum(member_scores) / len(member_scores)
    raise ValueError(f"unknown agg {agg!r}")


def nms_fuse(candidates, iou_th: float = CLUSTER_IOU, agg: str = "max", iou_fn=iou):
    """Greedy NMS over pooled rollout candidates, with cluster-score fusion.

    Args:
        candidates: list of ``(region, score, seed)``. ``region`` is a box or span, ``score``
            is its MTLA value, ``seed`` identifies the rollout it came from.
        iou_th: overlap at/above which a candidate is absorbed into a kept one.
        agg: cluster fusion rule (see module docstring).
        iou_fn: ``iou`` for boxes (default) or ``tiou`` for temporal spans.

    Returns:
        list of ``(region, fused_score)`` for the kept (non-suppressed) predictions,
        in descending score order before fusion.
    """
    order = sorted(range(len(candidates)), key=lambda i: -candidates[i][1])
    taken = [False] * len(candidates)
    kept = []
    for i in order:
        if taken[i]:
            continue
        taken[i] = True
        region_i, score_i, seed_i = candidates[i]
        seeds = {seed_i}
        member_scores = [score_i]
        for j in order:
            if taken[j] or j == i:
                continue
            if iou_fn(region_i, candidates[j][0]) >= iou_th:
                taken[j] = True
                seeds.add(candidates[j][2])
                member_scores.append(candidates[j][1])
        kept.append((region_i, _fuse(agg, score_i, member_scores, len(seeds))))
    return kept
