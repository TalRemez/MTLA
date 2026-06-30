"""Unified MTLA pipeline entrypoint.

One command runs any benchmark; the YAML config picks the model + dataset adapters and the
hyperparameters. Three stages:

    python run.py --config configs/coco_internvl.yaml --stage generate   # GPU (vLLM or HF)
    python run.py --config configs/coco_internvl.yaml --stage extract    # GPU (HF eager attn)
    python run.py --config configs/coco_internvl.yaml --stage score      # CPU

`generate` writes predictions, `extract` writes attention feature shards, `score` computes
hallucination AUROC + the benchmark's task metric. The score stage needs no GPU and no model.

CLI flags override the config's score stage for quick sweeps:
    --n 16   --agg sum
"""
import argparse

from mtla.config import load_config
from mtla.registry import resolve
from mtla.data.base import print_metrics


def main():
    ap = argparse.ArgumentParser(description="Unified MTLA generate/extract/score pipeline.")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--stage", required=True, choices=["generate", "extract", "score"])
    ap.add_argument("--n", type=int, default=None, help="override score.n_rollouts")
    ap.add_argument("--agg", default=None, help="override score.agg (max|sum|support|mean)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="generate/extract: rollout seeds to produce (e.g. --seeds 0 1 2 3); "
                         "overrides the stage's `seeds` in the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n is not None:
        cfg.score.n_rollouts = args.n
    if args.agg is not None:
        cfg.score.agg = args.agg
    if args.seeds is not None:
        cfg.generate.seeds = cfg.extract.seeds = args.seeds

    # resolve validates the (model x dataset) task pairing and returns both adapters.
    model, dataset = resolve(cfg.model, cfg.dataset)
    print(f"[MTLA] model={cfg.model}  dataset={cfg.dataset}  stage={args.stage}")

    if args.stage == "score":
        # CPU-only; reads the saved [L,H] attention arrays (no model weights loaded).
        metrics = dataset.score(cfg, model)
        print_metrics(f"{cfg.dataset}/{cfg.model}", metrics)
        return

    # generate / extract are GPU stages: the dataset asks the model adapter which stage script
    # to run (model x task specific) and passes dataset args. One rollout per seed -> seed{K}/.
    seeds = (cfg.generate if args.stage == "generate" else cfg.extract).seeds
    for seed in seeds:
        print(f"[MTLA] {args.stage} seed {seed}  ({seeds.index(seed)+1}/{len(seeds)})")
        if args.stage == "generate":
            dataset.generate(cfg, model, seed)
        else:
            dataset.extract(cfg, model, seed)


if __name__ == "__main__":
    main()
