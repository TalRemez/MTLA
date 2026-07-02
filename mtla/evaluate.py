"""Score stage: turn extracted attention shards into benchmark numbers.

This module owns **all** of the score-stage computation, so the dataset adapters stay purely
declarative (they only say *which* signal / overlap / metric to use — see ``mtla.data.base``).
The pipeline here is:

  1. load every rollout's feature shards (``<features>/seed{K}/shard*.pt``);
  2. reduce each prediction's ``[L, H]`` array to one scalar MTLA score (``mtla.score.reduce_band``,
     paper eq. 4);
  3. single-rollout hallucination AUROC (how well the score separates grounded from hallucinated);
  4. self-consistency voting: pool candidates across rollouts and fuse / select them
     (``mtla.voting.nms_fuse`` for detection & multi-window, ``argmax`` for single-span);
  5. hand the fused candidates to the benchmark's pure metric (``mtla.metrics``).

Steps 4-5 are the only dataset-shaped part; they dispatch on the dataset's declared ``metric``.
Adding a dataset that reuses an existing metric needs no change here; a genuinely new metric adds
one pure function in ``mtla.metrics`` and one handler below.

The shards are the complete input: each saved object carries its region, label, hallucination flag,
and ``[L, H]`` arrays; each record carries the item id + ground truth. Nothing is re-parsed and no
model is loaded — this stage is CPU-only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mtla.data.base import load_shards
from mtla.metrics import auroc, coco_map, moment_retrieval, recall_at_iou
from mtla.score import reduce_band
from mtla.utils import iou, tiou
from mtla.voting import nms_fuse
from mtla.types import ItemId, OverlapFn, Region

if TYPE_CHECKING:
    from mtla.config import RunConfig
    from mtla.data.base import DatasetAdapter

_OVERLAP: dict[str, OverlapFn] = {"iou": iou, "tiou": tiou}

# A flattened scored candidate (one prediction in one rollout) built by ``_candidates``.
Cand = dict[str, Any]
# Fused output: ranked (region, score) lists keyed by (item id, label).
FusedGroups = dict[tuple[ItemId, str], list[tuple[Region, float]]]


def _candidates(
    cfg: "RunConfig", dataset: "DatasetAdapter"
) -> tuple[list[Cand], dict[ItemId, list], list[int]]:
    """Load every rollout's shards and flatten them into scored candidates.

    Walks the discovered feature-shard directories (one per rollout seed), reduces each
    prediction's ``[L, H]`` attention array for the dataset's signal to a scalar MTLA
    score via ``reduce_band`` over the config's layer band (paper eq. 4), and collects
    every prediction as a flat candidate. This is the shared front end for both the
    hallucination-AUROC and the voting/metric paths.

    Args:
        cfg: the run config, supplying the layer band, the seeds on disk, and the
            per-seed feature directories.
        dataset: the dataset adapter, supplying ``signal`` (which stored array to
            reduce).

    Returns:
        A tuple ``(cands, gt_by_id, seeds)`` where ``cands`` is a list of dicts
        ``{id, label, region, score, hallu, extracted, seed}`` (one per prediction per
        rollout), ``gt_by_id`` maps each item id to its ground-truth region list, and
        ``seeds`` is the sorted list of rollout seeds found on disk.
    """
    band = cfg.band_indices()
    signal = dataset.signal
    seeds = cfg.extracted_seeds()  # discovered from <features>/seed*/, not a flag
    cands: list[Cand] = []
    gt_by_id: dict[ItemId, list] = {}
    for seed in seeds:
        for rec in load_shards(cfg.feat_dir(seed)):
            gt_by_id[rec["id"]] = rec["gt"]
            for o in rec["objects"]:
                arr = cast(
                    np.ndarray, dict(o)[signal]
                )  # dynamic signal key (not a TypedDict literal)
                cands.append(
                    {
                        "id": rec["id"],
                        "label": o["label"],
                        "region": o["region"],
                        "score": float(reduce_band(arr.astype(np.float32), band)),
                        "hallu": bool(o["is_hallucinated"]),
                        "extracted": bool(o.get("extracted", True)),
                        "seed": seed,
                    }
                )
    return cands, gt_by_id, seeds


def _hallucination_auroc(cands: list[Cand]) -> float:
    """Single-rollout hallucination AUROC over the extracted candidates.

    Restricts to rollout 0 (the deterministic anchor) and to candidates that actually
    had attention extracted, then measures how well the MTLA score separates grounded
    from hallucinated predictions. This is the detection metric reported per benchmark,
    independent of the voting path.

    Args:
        cands: the flattened candidates from ``_candidates`` (each carries ``score``,
            ``hallu``, ``seed``, and ``extracted``).

    Returns:
        The AUROC in ``[0, 1]`` (positive class is grounded), or ``nan`` when no
        seed-0 extracted candidate exists.
    """
    s = [c["score"] for c in cands if c["seed"] == 0 and c["extracted"]]
    y = [c["hallu"] for c in cands if c["seed"] == 0 and c["extracted"]]
    return auroc(s, y) if s else float("nan")


def _fuse_groups(
    cands: list[Cand], overlap_fn: OverlapFn, agg: str, select: str
) -> FusedGroups:
    """Pool candidates per ``(id, label)`` and fuse or select across rollouts.

    Implements the self-consistency step: candidates for the same item and label from
    all rollouts are grouped, then either merged by NMS with cluster-score fusion
    (``select="fuse"``, for detection and multi-window grounding) or reduced to the
    single highest-scoring candidate (``select="argmax"``, for single-span grounding).

    Args:
        cands: the flattened candidates from ``_candidates``.
        overlap_fn: region overlap function used by NMS (``iou`` for boxes, ``tiou``
            for spans).
        agg: the cluster-score aggregation mode for NMS fusion (``max`` | ``sum`` |
            ``support`` | ``mean``); unused when ``select="argmax"``.
        select: ``"fuse"`` to NMS-merge, or ``"argmax"`` to keep the top candidate.

    Returns:
        A mapping ``{(id, label): [(region, score), ...]}`` with each group's regions
        ranked by score (descending).
    """
    groups: dict[tuple[ItemId, str], list] = defaultdict(list)
    for c in cands:
        groups[(c["id"], c["label"])].append((c["region"], c["score"], c["seed"]))
    out: FusedGroups = {}
    for key, members in groups.items():
        if select == "argmax":
            region, score, _ = max(members, key=lambda m: m[1])
            out[key] = [(region, score)]
        else:
            out[key] = nms_fuse(members, agg=agg, iou_fn=overlap_fn)
    return out


# ---------------------------------------------------------------------------
# Metric handlers: fused candidates -> metric dict. One per metric name.
# ---------------------------------------------------------------------------
def _coco(cfg: "RunConfig", fused: FusedGroups, gt_by_id: dict[ItemId, list]) -> dict:
    """Assemble COCO detections from the fused groups and score bbox mAP.

    Maps each fused ``(image_id, label)`` group to COCO detection dicts, rescaling boxes
    from the model's normalized ``[0, 1000]`` frame to the image's absolute pixel size
    and to ``[x, y, w, h]``, dropping labels not in the COCO category set. Delegates the
    actual scoring to ``metrics.coco_map``.

    Args:
        cfg: the run config, supplying the ``coco_gt`` annotations path.
        fused: the fused groups from ``_fuse_groups``, keyed by ``(image_id, label)``.
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
        fused: the fused groups from ``_fuse_groups``, keyed by ``(qid, label)``.
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
        fused: the fused groups from ``_fuse_groups``, keyed by ``(qid, label)``.
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


def run_score(cfg: "RunConfig", dataset: "DatasetAdapter") -> dict:
    """Run the whole score stage for one run and return its metrics.

    The score-stage entry point: loads and scores every rollout's candidates, computes
    the single-rollout hallucination AUROC, fuses candidates across rollouts (NMS or
    argmax per the dataset), and dispatches the fused groups to the benchmark's metric
    handler. CPU-only; the feature shards are the complete input and no model is loaded.

    Args:
        cfg: the run config, supplying paths, the layer band, and the voting ``agg``.
        dataset: the dataset adapter, supplying ``overlap``, ``signal``, ``select``,
            and ``metric``.

    Returns:
        A metrics dict with ``auroc_mtla``, ``n_rollouts``, ``agg``, and the
        benchmark-specific keys from the metric handler, or ``{"error": ...}`` if no
        candidates were found (the extract stage did not run).
    """
    overlap_fn = _OVERLAP[dataset.overlap]
    cands, gt_by_id, seeds = _candidates(cfg, dataset)
    if not cands:
        return {"error": "no candidates found — did the extract stage run?"}
    print(f"[score] found {len(seeds)} rollout(s) on disk: seeds {seeds}", flush=True)
    fused = _fuse_groups(cands, overlap_fn, cfg.score.agg, dataset.select)
    metrics = {
        "auroc_mtla": _hallucination_auroc(cands),
        "n_rollouts": len(seeds),
        "agg": cfg.score.agg,
    }
    metrics.update(_METRICS[dataset.metric](cfg, fused, gt_by_id))
    return metrics
