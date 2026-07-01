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

import numpy as np

from .data.base import load_shards
from .metrics import auroc, coco_map, moment_retrieval, recall_at_iou
from .score import reduce_band
from .utils import iou, tiou
from .voting import nms_fuse

_OVERLAP = {"iou": iou, "tiou": tiou}


def _candidates(cfg, dataset):
    """Load every rollout and flatten to scored candidates.

    Returns ``(cands, gt_by_id)`` where ``cands`` is a list of dicts
    ``{id, label, region, score, hallu, extracted, seed}`` (one per prediction per rollout) and
    ``gt_by_id`` maps each item id to its ground-truth region list.
    """
    band = cfg.band_indices()
    signal = dataset.signal
    cands, gt_by_id = [], {}
    for seed in range(cfg.n_rollouts):
        for rec in load_shards(cfg.feat_dir(seed)):
            gt_by_id[rec["id"]] = rec["gt"]
            for o in rec["objects"]:
                cands.append({
                    "id": rec["id"], "label": o["label"], "region": o["region"],
                    "score": float(reduce_band(o[signal].astype(np.float32), band)),
                    "hallu": bool(o["is_hallucinated"]), "extracted": bool(o.get("extracted", True)),
                    "seed": seed,
                })
    return cands, gt_by_id


def _hallucination_auroc(cands):
    """Single-rollout (seed 0) hallucination AUROC over the extracted candidates."""
    s = [c["score"] for c in cands if c["seed"] == 0 and c["extracted"]]
    y = [c["hallu"] for c in cands if c["seed"] == 0 and c["extracted"]]
    return auroc(s, y) if s else float("nan")


def _fuse_groups(cands, overlap_fn, agg, select):
    """Group candidates by ``(id, label)`` and fuse/select within each group across rollouts.

    ``select="fuse"`` runs NMS with cluster-score fusion (detection, multi-window grounding);
    ``select="argmax"`` keeps the single highest-scoring candidate (single-span grounding).
    Returns ``{(id, label): [(region, score), ...]}`` ranked by score.
    """
    groups = defaultdict(list)
    for c in cands:
        groups[(c["id"], c["label"])].append((c["region"], c["score"], c["seed"]))
    out = {}
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
def _coco(cfg, fused, gt_by_id) -> dict:
    """Assemble COCO detections from fused (region, score) per (image, label) and score mAP."""
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
            detections.append({
                "image_id": image_id, "category_id": cid,
                "bbox": [x1 / 1000 * W, y1 / 1000 * H, (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H],
                "score": float(score)})
    return {"map": coco_map(detections, cfg.path("coco_gt")), "n_dets": len(detections)}


def _moment_retrieval(cfg, fused, gt_by_id) -> dict:
    """Assemble a ranked moment-retrieval submission (per qid) and score mAP / R@1."""
    MAX_WINDOWS = 10
    by_qid = defaultdict(list)
    for (qid, _label), kept in fused.items():
        by_qid[qid].extend(kept)
    submission, ground_truth = [], []
    for qid, kept in by_qid.items():
        rows = [[w[0], w[1], float(sc)] for w, sc in sorted(kept, key=lambda k: -k[1])][:MAX_WINDOWS]
        submission.append({"qid": qid, "pred_relevant_windows": rows or [[0.0, 0.0, 0.0]]})
        ground_truth.append({"qid": qid, "relevant_windows": [g["region"] for g in gt_by_id[qid]]})
    return {"nms_mtla": moment_retrieval(submission, ground_truth)}


def _recall(cfg, fused, gt_by_id) -> dict:
    """Single-span recall: one selected span per query vs its ground-truth span."""
    pred_spans, gt_spans = [], []
    for (qid, _label), kept in fused.items():
        pred_spans.append(kept[0][0] if kept else None)
        gt = gt_by_id.get(qid) or []
        gt_spans.append(gt[0]["region"] if gt else None)
    return {"mtla": recall_at_iou(pred_spans, gt_spans)}


_METRICS = {"coco_map": _coco, "moment_retrieval": _moment_retrieval, "recall_at_iou": _recall}


def run_score(cfg, dataset) -> dict:
    """Compute the benchmark metrics for a run from its feature shards. Returns a metrics dict."""
    overlap_fn = _OVERLAP[dataset.overlap]
    cands, gt_by_id = _candidates(cfg, dataset)
    if not cands:
        return {"error": "no candidates found — did the extract stage run?"}
    fused = _fuse_groups(cands, overlap_fn, cfg.score.agg, dataset.select)
    metrics = {"auroc_mtla": _hallucination_auroc(cands),
               "n_rollouts": cfg.n_rollouts, "agg": cfg.score.agg}
    metrics.update(_METRICS[dataset.metric](cfg, fused, gt_by_id))
    return metrics
