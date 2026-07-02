"""Self-consistency voting: pool predictions from N rollouts, fuse overlaps, re-score.

Sampling ``N`` stochastic rollouts per input enlarges the candidate pool (recall). ``vote`` is the
entry point: it groups the pooled candidates by ``(item id, label)``, merges the overlaps in each
group with non-maximum suppression (``nms_fuse``), assigns each kept prediction a fused score from
its cluster's MTLA scores, and keeps the top ``top_k`` per group. The ``agg`` argument selects the
fusion rule:

  * ``"max"``     keep the single highest-scoring rollout         (default; video / audio)
  * ``"sum"``     sum the cluster's scores (rewards recurrence)   (COCO detection headline)
  * ``"support"`` ``#distinct seeds * max``                        (ablation)
  * ``"mean"``    average of the cluster's scores                 (ablation)

Single-span grounding is just ``top_k=1``: NMS already ranks the highest-scoring cluster first, so
its representative *is* the argmax — no separate selection path is needed. Detection and
multi-window retrieval keep every fused cluster (``top_k=None``).

Works for both spatial boxes (use ``iou``) and temporal spans (use ``tiou``); pass the matching
overlap function via ``iou_fn``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

# Spatial/temporal IoU live in mtla.utils (the shared primitives home); re-exported here so
# `from mtla.voting import iou, tiou` and `nms_fuse(..., iou_fn=iou)` keep working.
from mtla.utils import iou, tiou
from mtla.types import FusedGroups, ItemId, OverlapFn, Region, ScoredCand

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


def vote(
    candidates: Sequence[ScoredCand],
    agg: str = "max",
    iou_fn: OverlapFn = iou,
    top_k: int | None = None,
) -> FusedGroups:
    """Group scored candidates by ``(id, label)`` and NMS-fuse each group across rollouts.

    The self-consistency step. Candidates for the same item and label from all rollouts are
    pooled and merged with :func:`nms_fuse`, so a prediction the model repeated across rollouts
    collapses to one fused-score region. Detection and multi-window retrieval keep every fused
    cluster (``top_k=None``); single-span grounding keeps only the top one (``top_k=1``), which is
    the argmax because NMS ranks the highest-scoring cluster first.

    Args:
        candidates: The flat scored candidates from ``score.load_candidates`` (one ``ScoredCand``
            per prediction per rollout); this reads ``id`` / ``label`` / ``region`` / ``score`` /
            ``seed``.
        agg: Cluster fusion rule passed to :func:`nms_fuse` (``max`` / ``sum`` / ``support`` /
            ``mean``).
        iou_fn: Overlap function — :func:`mtla.utils.iou` for boxes, :func:`mtla.utils.tiou` for
            spans.
        top_k: Keep at most this many fused regions per group (already ranked by score); ``None``
            keeps all.

    Returns:
        A mapping ``{(id, label): [(region, score), ...]}`` with each group's regions ranked by
        fused score (descending), truncated to ``top_k``.
    """
    groups: dict[tuple[ItemId, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[(c["id"], c["label"])].append((c["region"], c["score"], c["seed"]))
    out: FusedGroups = {}
    for key, members in groups.items():
        fused = nms_fuse(members, agg=agg, iou_fn=iou_fn)
        out[key] = fused[:top_k] if top_k is not None else fused
    return out
