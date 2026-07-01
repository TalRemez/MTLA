"""Stage 3 — scoring. Turn the extracted attention shards into benchmark numbers (CPU only).

Reads ``<features>/seed{K}/shard*.pt`` for seeds ``0..n_rollouts-1`` and computes the hallucination
AUROC + the benchmark's task metric. No GPU and no model weights: all the work (band reduction,
self-consistency voting, NMS, metric evaluation) lives in ``mtla.evaluate`` / ``mtla.metrics``.

    python score.py --config configs/coco_internvl.yaml
    python score.py --config configs/coco_internvl_voting.yaml --n 16 --agg sum
"""
import argparse

from mtla.config import load_config
from mtla.registry import resolve
from mtla.evaluate import run_score
from mtla.data.base import print_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--n", type=int, default=None, help="override n_rollouts (how many seeds to vote over)")
    ap.add_argument("--agg", default=None, help="override score.agg (max|sum|support|mean)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n is not None:
        cfg.n_rollouts = args.n
    if args.agg is not None:
        cfg.score.agg = args.agg

    _, dataset = resolve(cfg.model, cfg.dataset)
    metrics = run_score(cfg, dataset)
    print_metrics(f"{cfg.dataset}/{cfg.model}", metrics)


if __name__ == "__main__":
    main()
