"""Unified MTLA pipeline entrypoint.

One command runs any benchmark; the YAML config picks the model + dataset adapters and the
hyperparameters. Three stages:

    python run.py --config configs/coco_internvl.yaml --stage generate   # GPU (vLLM or HF)
    python run.py --config configs/coco_internvl.yaml --stage extract    # GPU (HF eager attn)
    python run.py --config configs/coco_internvl.yaml --stage score      # CPU

`generate` writes predictions, `extract` writes attention feature shards, `score` computes
hallucination AUROC + the benchmark's task metric. The score stage needs no GPU and no model.

CLI flags override the config for quick sweeps:
    --n 16   --agg sum   --seeds 0 1 2 3
"""
import argparse

from mtla.config import load_config
from mtla.registry import resolve
from mtla.data.base import print_metrics


def main():
    ap = argparse.ArgumentParser(description="Unified MTLA generate/extract/score pipeline.")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--stage", required=True, choices=["generate", "extract", "score"])
    ap.add_argument("--n", type=int, default=None, help="override n_rollouts (drives seeds + voting)")
    ap.add_argument("--agg", default=None, help="override score.agg (max|sum|support|mean)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="generate/extract: produce exactly these rollout seeds (e.g. --seeds 0 1 2 3); "
                         "by default the seeds are 0..n_rollouts-1")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n is not None:
        cfg.n_rollouts = args.n
    if args.agg is not None:
        cfg.score.agg = args.agg

    # resolve validates the (model x dataset) task pairing and returns both adapters.
    model, dataset = resolve(cfg.model, cfg.dataset)
    print(f"[MTLA] model={cfg.model}  dataset={cfg.dataset}  stage={args.stage}")

    if args.stage == "score":
        # CPU-only; reads the saved [L,H] attention arrays (no model weights loaded).
        metrics = dataset.score(cfg, model)
        print_metrics(f"{cfg.dataset}/{cfg.model}", metrics)
        return

    # generate / extract are GPU stages: the dataset asks the model adapter which stage script
    # to run (model x task specific). One rollout per seed -> seed{K}/. Seeds default to
    # 0..n_rollouts-1; --seeds produces an explicit subset.
    seeds = args.seeds if args.seeds is not None else cfg.seeds()
    for i, seed in enumerate(seeds):
        print(f"[MTLA] {args.stage} seed {seed}  ({i+1}/{len(seeds)})")
        if args.stage == "generate":
            dataset.generate(cfg, model, seed)
        else:
            dataset.extract(cfg, model, seed)


if __name__ == "__main__":
    main()
