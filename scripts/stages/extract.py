"""Shared MTLA extraction driver (any model, any dataset, image or video).

The single extract stage. It's model- and dataset-agnostic: it loads the run config, resolves the
(model, dataset) adapters from the registry, shards the generation records across GPUs, and for
each record calls ``model.extract_one`` — which runs one HF-eager captured forward and applies the
MTLA math (see ``mtla.mtla_attn``). Every model/task specific lives behind the adapter's ``ext_*``
callbacks; this file is just the multi-GPU harness.

The generation records are self-contained (``{id, prompt, response, gt, extra}``), so the worker
needs no dataset item lookup — it just streams ``predictions.json``.

    python scripts/stages/extract.py --config configs/coco_internvl.yaml --seed 0

Reads ``<predictions>/seed{K}/predictions.json``; writes ``<features>/seed{K}/shard{rank}.pt``.
"""
import argparse
import json
import os
from multiprocessing import Process, set_start_method

import numpy as np
import torch

from mtla.config import load_config
from mtla.registry import resolve


def worker(rank, gpu_id, records, out_dir, config_path):
    torch.cuda.set_device(gpu_id)
    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    ctx = model.load_for_extract(gpu_id, dataset.task)
    if dataset.task == "video_span":
        # Video ext_* reads the same preprocessing the generate stage used, plus multi-span flag.
        ctx["preprocess"] = cfg.preprocess
        ctx["multi"] = (dataset.select == "fuse")
    print(f"[worker {rank}] model={cfg.model} dataset={cfg.dataset} gpu={gpu_id} "
          f"n={len(records)} L={ctx['n_layers']} H={ctx['n_heads']}", flush=True)

    out, n_done = [], 0
    for cnt, rec in enumerate(records):
        r = model.extract_one(rec, ctx, rank=rank)
        if r is None:
            continue
        out.append(r); n_done += 1
        if n_done % 25 == 0:
            print(f"[worker {rank}] [{cnt+1}/{len(records)}] done={n_done}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/shard{rank}.pt"
    torch.save(out, path)
    print(f"[worker {rank}] saved {len(out)} records / "
          f"{sum(len(r['objects']) for r in out)} preds -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--seed", type=int, default=0, help="rollout seed (selects seed{K}/ dirs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gpus = cfg.stage_gpus("extract")
    records = json.load(open(os.path.join(cfg.pred_dir(args.seed), "predictions.json")))
    if cfg.extract.n_items:
        records = records[:cfg.extract.n_items]
    out_dir = cfg.feat_dir(args.seed)

    set_start_method("spawn", force=True)
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(gpus, np.array_split(records, len(gpus)))):
        if len(chunk) == 0:
            continue
        p = Process(target=worker, args=(rank, gpu, list(chunk), out_dir, cfg.config_path))
        p.start(); procs.append((rank, p))
    failed = [(rank, p.exitcode) for rank, p in procs if (p.join() or p.exitcode != 0)]
    if failed:
        # a crashed worker leaves its shard unwritten -> a silently incomplete run. Fail loudly.
        raise SystemExit(f"[extract] {len(failed)} worker(s) failed (rank, exitcode): {failed}; "
                         f"shards under {out_dir} are INCOMPLETE — fix the error and re-run.")
    print(f"all {len(procs)} workers complete")


if __name__ == "__main__":
    main()
