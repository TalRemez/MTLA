"""Shared image-detection MTLA extraction driver (any image_det model).

One HF-eager MTLA stage for any image_det model. The scaffold here is model- and dataset-agnostic:
it loads the run config, resolves the (model, dataset) adapters from the registry, and delegates
every specific piece:
  - `dataset.load_items(cfg)`            -> the dataset items (the adapter owns file I/O).
  - `model.load_for_extract(gpu_id)`     -> ctx (model, tokenizer/processor, n_layers/heads, the
    MTLAState with the MTLA attention forward installed).
  - `model.extract_one(pred, ds_by_id, ctx, svar_shift)` -> the per-image .pt record (or None to
    skip). This delegates to `mtla.mtla_attn.compute_mtla`, the shared per-item MTLA computation.

Invoked as a subprocess by the dataset adapter's `stage_cmd` (see `mtla.data.base`):

    python scripts/stages/image_extract.py --config configs/coco_internvl.yaml --seed 0

Reads `<predictions>/seed{K}/predictions.json` (from the generate stage); writes
`<features>/seed{K}/shard{rank}.pt` feature shards.
"""
import argparse
import json
import os
from multiprocessing import Process, set_start_method

import numpy as np
import torch


def worker(rank, gpu_id, image_indices, out_dir, config_path, pred_file, svar_shift):
    from mtla.config import load_config
    from mtla.registry import resolve
    torch.cuda.set_device(gpu_id)
    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    ctx = model.load_for_extract(gpu_id)
    print(f"[worker {rank}] model={cfg.model} dataset={cfg.dataset} gpu={gpu_id} "
          f"n={len(image_indices)} L={ctx['n_layers']} H={ctx['n_heads']}", flush=True)

    preds_all = json.load(open(pred_file))
    ds_by_id = {d["id"]: d for d in dataset.load_items(cfg)}

    records = []
    n_done = n_skipped = 0
    for cnt, idx in enumerate(image_indices):
        rec = model.extract_one(preds_all[idx], ds_by_id, ctx, svar_shift, rank=rank)
        if rec is None:
            n_skipped += 1
            continue
        records.append(rec)
        n_done += 1
        if n_done % 25 == 0:
            print(f"[worker {rank}] [{cnt+1}/{len(image_indices)}] done={n_done} skip={n_skipped}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard{rank}.pt"
    torch.save(records, out_path)
    print(f"[worker {rank}] saved {len(records)} records / "
          f"{sum(len(r['objects']) for r in records)} preds -> {out_path}", flush=True)


def main():
    from mtla.config import load_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--seed", type=int, default=0, help="rollout seed (selects seed{K}/ dirs)")
    ap.add_argument("--svar_shift", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gpus = cfg.stage_gpus("extract")
    n_images = cfg.extract.n_items or 5000
    pred_file = os.path.join(cfg.pred_dir(args.seed), "predictions.json")
    out_dir = cfg.feat_dir(args.seed)

    set_start_method("spawn", force=True)
    chunks = np.array_split(list(range(n_images)), len(gpus))
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(gpus, chunks)):
        if len(chunk) == 0:
            continue
        p = Process(target=worker, args=(rank, gpu, list(chunk), out_dir, cfg.config_path,
                                         pred_file, args.svar_shift))
        p.start(); procs.append((rank, p))
    failed = []
    for rank, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((rank, p.exitcode))
    if failed:
        # a crashed worker leaves its shard unwritten -> a silently incomplete run. Fail loudly.
        raise SystemExit(f"[image_extract] {len(failed)} worker(s) failed (rank, exitcode): {failed}; "
                         f"shards under {out_dir} are INCOMPLETE — fix the error and re-run.")
    print(f"all {len(procs)} workers complete")


if __name__ == "__main__":
    main()
