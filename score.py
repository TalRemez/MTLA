"""Stage 3 — scoring. Turn the extracted attention shards into benchmark numbers (CPU only).

Reads every ``<features>/seed{K}/shard*.pt`` the extract stage wrote and computes the hallucination
AUROC + the benchmark's task metric. No GPU and no model weights. The stage is a short, explicit
pipeline, composed in ``main`` so every step is visible:

    candidates = load_candidates(...)        # shards -> flat scored predictions (band reduction)
    auroc      = hallucination_auroc(...)     # single-rollout detection AUROC
    voted      = vote(...)                     # self-consistency NMS voting across rollouts
    metric     = compute_metric(..., voted)    # voted candidates -> benchmark metric

The AUROC / metric assembly live in ``mtla.evaluate`` (+ pure math in ``mtla.metrics``); the voting
step lives in ``mtla.voting``. The rollout seed set is discovered from the features dir, so there is
no ``--n``: score votes over exactly the rollouts that were extracted.

    python -m score --config configs/coco_internvl.yaml
    python -m score --config configs/coco_internvl.yaml --agg sum   # COCO N=16 voting headline
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, cast

import numpy as np

from mtla.config import load_config
from mtla.registry import resolve
from mtla.evaluate import compute_metric, hallucination_auroc
from mtla.score import reduce_band
from mtla.voting import vote
from mtla.utils import overlap_fn
from mtla.data.base import load_shards, print_metrics
from mtla.types import ItemId, ScoredCand

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
    seeds = cfg.extracted_seeds()  # discovered from <features>/seed*/, not a flag
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
    print(f"[score] found {len(seeds)} rollout(s) on disk: seeds {seeds}", flush=True)

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
