"""Stage 3 — evaluation. Turn the extracted attention shards into benchmark numbers (CPU only).

Reads every ``<features>/seed{K}/shard*.pt`` the extract stage wrote and computes the hallucination
AUROC + the benchmark's task metric. No GPU and no model weights. The stage is a short, explicit
pipeline, composed in ``main`` so every step is visible:

    candidates = load_candidates(...)        # shards -> flat scored predictions (band reduction)
    auroc      = hallucination_auroc(...)     # single-rollout detection AUROC
    voted      = vote(...)                     # self-consistency NMS voting across rollouts
    metric     = compute_metric(..., voted)    # voted candidates -> benchmark metric

The reusable pieces live in the ``mtla`` package — ``reduce_band`` (band reduction), ``vote``
(voting), and the pure ``metrics`` computers; this script owns the evaluation glue: loading and
scoring the shards, the hallucination AUROC, and reshaping voted candidates into each benchmark's
metric input (the ``_coco`` / ``_moment_retrieval`` / ``_recall`` handlers). The rollout seed set is
discovered from the features dir, so there is no ``--n``: it votes over exactly the rollouts that
were extracted.

    python -m evaluate --config configs/coco_internvl.yaml
    python -m evaluate --config configs/coco_qwen3vl.yaml --agg sum   # COCO N=16 voting headline
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np

from mtla.config import load_config
from mtla.registry import resolve
from mtla.mtla_attn import reduce_band
from mtla.voting import vote
from mtla.utils import overlap_fn
from mtla.metrics import auroc, coco_map, moment_retrieval, recall_at_iou
from mtla.data.base import load_shards, print_metrics
from mtla.types import FusedGroups, ItemId, ScoredCand

if TYPE_CHECKING:
    from mtla.config import RunConfig
    from mtla.data.base import DatasetAdapter


def load_candidates(
    cfg: "RunConfig", dataset: "DatasetAdapter"
) -> tuple[list[ScoredCand], dict[ItemId, list], list[int]]:
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
        A tuple ``(cands, gt_by_id, seeds)`` where ``cands`` is a list of
        ``ScoredCand`` (one per prediction per rollout), ``gt_by_id`` maps each item id
        to its ground-truth region list, and ``seeds`` is the sorted list of rollout
        seeds found on disk.
    """
    band = cfg.band_indices()
    signal = dataset.signal
    seeds = cfg.seeds_on_disk("features")  # what extract wrote, not a flag
    cands: list[ScoredCand] = []
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
# Metric handlers: voted candidates -> metric dict. One per metric name; dispatched by
# `compute_metric` on the dataset's declared `metric`. Each just reshapes the voted groups
# into the benchmark's expected input and calls the matching pure computer in `mtla.metrics`.
# ---------------------------------------------------------------------------
def _coco(cfg: "RunConfig", voted: FusedGroups, gt_by_id: dict[ItemId, list]) -> dict:
    """Assemble COCO detections from the voted groups and score bbox mAP.

    Maps each voted ``(image_id, label)`` group to COCO detection dicts, rescaling boxes
    from the model's normalized ``[0, 1000]`` frame to the image's absolute pixel size
    and to ``[x, y, w, h]``, dropping labels not in the COCO category set. Delegates the
    actual scoring to ``metrics.coco_map``.

    Args:
        cfg: the run config, supplying the ``coco_gt`` annotations path.
        voted: the voted groups from ``mtla.voting.vote``, keyed by ``(image_id, label)``.
        gt_by_id: per-item ground truth (unused here; COCO scores against the GT JSON).

    Returns:
        A dict ``{"map": <coco_map result>, "n_dets": <count>}`` where ``map`` holds the
        mAP breakdown and ``n_dets`` is the number of assembled detections.
    """
    gt = json.load(open(cfg.path("coco_gt")))
    name2cat = {c["name"].lower(): c["id"] for c in gt["categories"]}
    img_wh = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    detections = []
    for (image_id, label), kept in voted.items():
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
    cfg: "RunConfig", voted: FusedGroups, gt_by_id: dict[ItemId, list]
) -> dict:
    """Assemble a ranked moment-retrieval submission per query and score mAP / R@1.

    Collapses the voted groups by query id, ranks each query's windows by score, keeps
    the top ``MAX_WINDOWS``, and pairs them with the query's ground-truth windows in the
    format the official Moment-DETR evaluator expects. Delegates scoring to
    ``metrics.moment_retrieval``.

    Args:
        cfg: the run config (unused here; kept for the uniform handler signature).
        voted: the voted groups from ``mtla.voting.vote``, keyed by ``(qid, label)``.
        gt_by_id: per-query ground-truth relevant windows, keyed by qid.

    Returns:
        A dict ``{"nms_mtla": <moment_retrieval result>}`` with the mAP / R@1 metrics.
    """
    MAX_WINDOWS = 10
    by_qid = defaultdict(list)
    for (qid, _label), kept in voted.items():
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


def _recall(cfg: "RunConfig", voted: FusedGroups, gt_by_id: dict[ItemId, list]) -> dict:
    """Score single-span temporal grounding: one selected span per query vs its GT.

    Takes the single top span each query kept (``top_k=1`` voting upstream) and pairs it
    with that query's ground-truth span, then delegates to ``metrics.recall_at_iou`` for
    R@1 at IoU thresholds and mIoU. A query with no kept span or no GT contributes a miss.

    Args:
        cfg: the run config (unused here; kept for the uniform handler signature).
        voted: the voted groups from ``mtla.voting.vote``, keyed by ``(qid, label)``.
        gt_by_id: per-query ground-truth regions, keyed by qid.

    Returns:
        A dict ``{"mtla": <recall_at_iou result>}`` with the recall / mIoU metrics.
    """
    pred_spans, gt_spans = [], []
    for (qid, _label), kept in voted.items():
        pred_spans.append(kept[0][0] if kept else None)
        gt = gt_by_id.get(qid) or []
        gt_spans.append(gt[0]["region"] if gt else None)
    return {"mtla": recall_at_iou(pred_spans, gt_spans)}


_METRICS = {
    "coco_map": _coco,
    "moment_retrieval": _moment_retrieval,
    "recall_at_iou": _recall,
}


def compute_metric(
    cfg: "RunConfig",
    dataset: "DatasetAdapter",
    voted: FusedGroups,
    gt_by_id: dict[ItemId, list],
) -> dict:
    """Dispatch the voted candidates to the dataset's benchmark metric handler.

    Looks up the handler named by ``dataset.metric`` (one of the ``_coco`` /
    ``_moment_retrieval`` / ``_recall`` handlers above) and runs it over the voted groups. This is
    the only dataset-shaped step of the score stage.

    Args:
        cfg: the run config (passed through to the handler, e.g. for the COCO GT path).
        dataset: the dataset adapter, supplying ``metric`` (which handler to run).
        voted: the voted groups from ``mtla.voting.vote``, keyed by ``(id, label)``.
        gt_by_id: per-item ground truth from ``load_candidates``.

    Returns:
        The benchmark-specific metrics dict from the handler.
    """
    return _METRICS[dataset.metric](cfg, voted, gt_by_id)


def main() -> None:
    """Parse CLI args, run the score-stage pipeline, and print the benchmark metrics.

    Applies the ``--config`` / ``--agg`` overrides, resolves the dataset adapter, then runs the four
    score-stage steps in sequence — load + score the shards, hallucination AUROC, self-consistency
    voting (NMS; ``top_k=1`` for single-span datasets), and the benchmark metric — over the rollouts
    auto-discovered on disk, and pretty-prints the result.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument(
        "--agg", default=None, help="override score.agg (max|sum|support|mean)"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.agg is not None:
        cfg.score.agg = args.agg
    _, dataset = resolve(cfg.model, cfg.dataset)

    cands, gt_by_id, seeds = load_candidates(cfg, dataset)
    if not cands:
        print_metrics(
            f"{cfg.dataset}/{cfg.model}",
            {"error": "no candidates found — did the extract stage run?"},
        )
        return
    print(
        f"[evaluate] found {len(seeds)} rollout(s) on disk: seeds {seeds}", flush=True
    )

    # Single-span grounding keeps one region per query (top_k=1); detection and multi-window keep
    # every fused cluster. NMS already ranks the best cluster first, so top_k=1 IS the argmax.
    top_k = 1 if dataset.select == "argmax" else None
    voted = vote(
        cands, agg=cfg.score.agg, iou_fn=overlap_fn(dataset.overlap), top_k=top_k
    )

    metrics = {
        "auroc_mtla": hallucination_auroc(cands),
        "n_rollouts": len(seeds),
        "agg": cfg.score.agg,
    }
    metrics.update(compute_metric(cfg, dataset, voted, gt_by_id))
    print_metrics(f"{cfg.dataset}/{cfg.model}", metrics)


if __name__ == "__main__":
    main()
