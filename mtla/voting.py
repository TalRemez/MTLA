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

from typing import Sequence

# Spatial/temporal IoU live in mtla.utils (the shared primitives home); re-exported here so
# `from mtla.voting import iou, tiou` and `nms_fuse(..., iou_fn=iou)` keep working.
from mtla.utils import iou, tiou
from mtla.types import OverlapFn, Region

CLUSTER_IOU = 0.5  # overlap threshold for "same prediction" across rollouts

# A pooled candidate: (region, MTLA score, seed it came from).
Candidate = tuple[Region, float, int]


def _fuse(
    agg: str, rep_score: float, member_scores: Sequence[float], n_seeds: int
) -> float:
    """Combine one NMS cluster's member scores into a single fused score.

    Args:
        agg: Fusion rule (see module docstring): ``"max"`` returns the cluster representative's
            score, ``"support"`` scales it by the number of distinct seeds, ``"sum"`` adds all
            member scores, ``"mean"`` averages them.
        rep_score: Score of the cluster representative (the highest-scoring member).
        member_scores: Scores of every candidate absorbed into the cluster (includes the rep).
        n_seeds: Number of distinct rollout seeds represented in the cluster.

    Returns:
        The fused cluster score under ``agg``.

    Raises:
        ValueError: If ``agg`` is not one of the four known rules.
    """
    if agg == "max":
        return rep_score
    if agg == "support":
        return n_seeds * rep_score
    if agg == "sum":
        return sum(member_scores)
    if agg == "mean":
        return sum(member_scores) / len(member_scores)
    raise ValueError(f"unknown agg {agg!r}")


def nms_fuse(
    candidates: Sequence[Candidate],
    iou_th: float = CLUSTER_IOU,
    agg: str = "max",
    iou_fn: OverlapFn = iou,
) -> list[tuple[Region, float]]:
    """Merge overlapping rollout candidates by greedy non-maximum suppression, then fuse each cluster.

    Processes candidates from highest to lowest score. Each unclaimed candidate becomes a cluster
    representative and absorbs every remaining candidate that overlaps it by ``>= iou_th``; the
    cluster's members are then collapsed to one score via ``agg`` (see :func:`_fuse`). This both
    de-duplicates predictions the model produced across rollouts and lets recurrence boost a score.

    Args:
        candidates: Pooled candidates as ``(region, score, seed)`` triples, where ``region`` is a
            box (``[x1,y1,x2,y2]``) or span (``[t0,t1]``), ``score`` is its MTLA value, and ``seed``
            identifies the rollout it came from.
        iou_th: Overlap at or above which a candidate is absorbed into the current cluster.
        agg: Cluster fusion rule passed to :func:`_fuse` (``max`` / ``sum`` / ``support`` / ``mean``).
        iou_fn: Overlap function — :func:`mtla.utils.iou` for boxes, :func:`mtla.utils.tiou` for spans.

    Returns:
        One ``(region, fused_score)`` per kept (representative) cluster, in the greedy processing
        order (descending representative score).
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
