"""Unified MTLA pipeline entrypoint.

One command runs any benchmark; the YAML config picks the model + dataset adapters and the
hyperparameters. Three stages:

    python run.py --config configs/coco_internvl.yaml --stage generate   # GPU (vLLM or HF)
    python run.py --config configs/coco_internvl.yaml --stage extract    # GPU (HF eager attn)
    python run.py --config configs/coco_internvl.yaml --stage score      # CPU

`generate` writes predictions, `extract` writes attention feature shards, `score` computes
hallucination AUROC + the benchmark's task metric. The score stage needs no GPU and no model.

CLI flags override the config's score stage for quick sweeps:
    --n 16   --agg sum   --slot first_digit
"""
import argparse

from mtla.config import load_config
from mtla.models import get_model_adapter
from mtla.data import get_dataset_adapter


def main():
    ap = argparse.ArgumentParser(description="Unified MTLA generate/extract/score pipeline.")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--stage", required=True, choices=["generate", "extract", "score"])
    ap.add_argument("--n", type=int, default=None, help="override score.n_rollouts")
    ap.add_argument("--agg", default=None, help="override score.agg (max|sum|support|mean)")
    ap.add_argument("--slot", default=None, help="override score.slot")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n is not None:
        cfg.score.n_rollouts = args.n
    if args.agg is not None:
        cfg.score.agg = args.agg
    if args.slot is not None:
        cfg.score.slot = args.slot

    dataset = get_dataset_adapter(cfg.dataset)
    print(f"[MTLA] model={cfg.model}  dataset={cfg.dataset}  stage={args.stage}")

    if args.stage == "score":
        # CPU-only; needs no model.
        dataset.score(cfg)
        return

    # generate / extract are GPU stages: the dataset owns the (model x dataset)-specific stage
    # script and passes the model adapter (for model_id / attn_module_path).
    model = get_model_adapter(cfg.model)
    if args.stage == "generate":
        dataset.generate(cfg, model)
    else:
        dataset.extract(cfg, model)


if __name__ == "__main__":
    main()
