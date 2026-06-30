"""Shared image-detection MTLA extraction driver (InternVL + Qwen3-VL).

One HF-eager MTLA stage for any image_det model. The scaffold here is model-agnostic
(arg parse, multi-GPU worker spawn, predictions/dataset load, shard save). All model-specific
work is delegated to the resolved ModelAdapter:
  - `adapter.load_for_extract(gpu_id)` -> ctx (model, tokenizer/processor, n_layers/heads, the
    MTLAState with the MTLA attention forward installed).
  - `adapter.extract_one(pred_record, ds_by_id, ctx, svar_shift)` -> the per-image .pt record
    (or None to skip). This delegates to `mtla.mtla_attn.compute_image_mtla`, the shared per-image
    MTLA computation.

Reads predictions.json (from the generate stage), writes shard{rank}.pt feature shards.
"""
import argparse
import json
import os
from multiprocessing import Process, set_start_method

import numpy as np
import torch


def worker(rank, gpu_id, image_indices, out_dir, model_key, dataset_file, pred_file, svar_shift):
    from mtla.models import get_model_adapter
    torch.cuda.set_device(gpu_id)
    adapter = get_model_adapter(model_key)
    ctx = adapter.load_for_extract(gpu_id)
    print(f"[worker {rank}] model={model_key} gpu={gpu_id} n={len(image_indices)} "
          f"L={ctx['n_layers']} H={ctx['n_heads']}", flush=True)

    preds_all = json.load(open(pred_file))
    ds_by_id = {d["id"]: d for d in json.load(open(dataset_file))}

    records = []
    n_done = n_skipped = 0
    for cnt, idx in enumerate(image_indices):
        rec = adapter.extract_one(preds_all[idx], ds_by_id, ctx, svar_shift, rank=rank)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model adapter key (internvl | qwen3vl)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n_images", type=int, default=5000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pred_file", required=True, help="predictions.json from the generate stage")
    ap.add_argument("--dataset", required=True, help="COCO openvocab dataset json")
    ap.add_argument("--svar_shift", action="store_true")
    args = ap.parse_args()
    set_start_method("spawn", force=True)
    chunks = np.array_split(list(range(args.n_images)), len(args.gpus))
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(args.gpus, chunks)):
        if len(chunk) == 0:
            continue
        p = Process(target=worker, args=(rank, gpu, list(chunk), args.out_dir, args.model,
                                         args.dataset, args.pred_file, args.svar_shift))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    print("all workers complete")


if __name__ == "__main__":
    main()
