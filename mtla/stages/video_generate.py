"""Shared video_span generation driver (any video grounding model + dataset).

The generate half of the decoupled video pipeline (the extract half is video_extract.py). It loads
the run config, resolves the (model, dataset) adapters, and delegates every specific piece:
  - `dataset.load_items(cfg)`                 -> the raw work items (the adapter owns file I/O).
  - `dataset.prompt(item)` / `video_path(...)`-> the prompt text and the clip path per item.
  - `model.load_for_generate(gpu_id)`         -> ctx (model + processor; stock attention, no hook).
  - `model.generate_video(ctx, path, query, video_cfg, seed)` -> the decoded response.
  - `dataset.make_prediction(item, response, model)` -> the prediction record (parses windows/span,
    flags the GT match) — the same schema the extract stage's `video_item` reads back.

Greedy by default; pass a non-default `--seed` (or config seed != 0) to sample a stochastic rollout
(T=0.7, top-p=0.95), seeded per (seed, rank) for reproducibility.

    python -m mtla.stages.video_generate --config configs/qvhighlights_qwen3vl.yaml --seed 0

Writes `<predictions>/seed{K}/predictions.json` (merged across workers).
"""
import argparse
import json
import os
from multiprocessing import Process, set_start_method

import numpy as np


def worker(rank, gpu_id, items, pred_dir, config_path, video_dir, seed):
    from mtla.config import load_config
    from mtla.registry import resolve
    import torch
    torch.cuda.set_device(gpu_id)
    if seed:
        from transformers import set_seed
        set_seed(seed * 1000 + rank)  # reproducible per (seed, rank) rollout
    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    ctx = model.load_for_generate(gpu_id)
    vcfg = dataset.video
    sample_seed = seed if seed else None
    print(f"[worker {rank}] generate model={cfg.model} dataset={cfg.dataset} gpu={gpu_id} "
          f"n={len(items)} seed={seed}", flush=True)

    preds = []
    n_done = n_skipped = 0
    for cnt, item in enumerate(items):
        video_path = dataset.video_path(item, video_dir)
        if not os.path.exists(video_path):
            n_skipped += 1
            continue
        response = model.generate_video(ctx, video_path, dataset.prompt(item), vcfg,
                                        seed=sample_seed, rank=rank)
        if response is None:
            n_skipped += 1
            continue
        preds.append(dataset.make_prediction(item, response, model))
        n_done += 1
        if n_done % 25 == 0:
            print(f"[worker {rank}] [{cnt+1}/{len(items)}] done={n_done} skip={n_skipped}", flush=True)

    os.makedirs(pred_dir, exist_ok=True)
    with open(f"{pred_dir}/preds_rank{rank}.json", "w") as f:
        json.dump(preds, f)
    print(f"[worker {rank}] saved {len(preds)} preds -> {pred_dir}/preds_rank{rank}.json", flush=True)


def main():
    from mtla.config import load_config
    from mtla.registry import resolve
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--seed", type=int, default=0, help="rollout seed (0 = greedy; >0 = sampled)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, dataset = resolve(cfg.model, cfg.dataset)
    gpus = cfg.generate.gpus or [0]
    video_dir = cfg.path("video_dir")
    pred_dir = cfg.pred_dir(args.seed)

    items = dataset.load_items(cfg)
    if cfg.generate.n_items:
        items = items[:cfg.generate.n_items]

    set_start_method("spawn", force=True)
    chunks = np.array_split(items, len(gpus))
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(gpus, chunks)):
        if len(chunk) == 0:
            continue
        p = Process(target=worker, args=(rank, gpu, list(chunk), pred_dir, cfg.config_path,
                                         video_dir, args.seed))
        p.start(); procs.append(p)
    for p in procs:
        p.join()

    # merge per-rank shards -> predictions.json
    merged = []
    for rank in range(len(gpus)):
        pp = f"{pred_dir}/preds_rank{rank}.json"
        if os.path.exists(pp):
            merged.extend(json.load(open(pp)))
    with open(f"{pred_dir}/predictions.json", "w") as f:
        json.dump(merged, f)
    n_ok = sum(1 for p in merged if p.get("is_correct"))
    print(f"Merged: {len(merged)} preds ({n_ok} correct) -> {pred_dir}/predictions.json")


if __name__ == "__main__":
    main()
