"""Shared video_span MTLA extraction driver (any video grounding model + dataset).

The video counterpart of image_extract.py, sharing the same MTLA core (mtla.mtla_attn.compute_mtla):
it loads the run config, resolves the (model, dataset) adapters from the registry, and delegates
every specific piece:
  - the prediction records (from the generate stage) ARE the work items: each carries the response
    + predicted windows/span + GT; the dataset adapter's `video_item(p, video_dir)` normalizes them.
  - `model.load_for_extract(gpu_id, "video_span")` -> ctx (model, processor, MTLAState, frame
    pad-id, ...); the driver adds `dataset` + `video_dir` so the video ext_* can read sampling cfg
    and resolve video paths.
  - `model.extract_one(p, {}, ctx, svar_shift)` -> the per-item .pt record (or None to skip), via
    the shared compute_mtla (windows are masked onto inside-span frame tokens).

Invoked as a subprocess by the dataset adapter's `stage_cmd`:

    python -m mtla.stages.video_extract --config configs/qvhighlights_qwen3vl.yaml --seed 0

Reads `<predictions>/seed{K}/predictions.json`; writes `<features>/seed{K}/shard{rank}.pt`.
"""
import argparse
import json
import os
from multiprocessing import Process, set_start_method

import numpy as np
import torch


def worker(rank, gpu_id, preds, out_dir, config_path, video_dir, svar_shift):
    from mtla.config import load_config
    from mtla.registry import resolve
    torch.cuda.set_device(gpu_id)
    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    ctx = model.load_for_extract(gpu_id, task="video_span")
    ctx["dataset"] = dataset
    ctx["video_dir"] = video_dir
    print(f"[worker {rank}] model={cfg.model} dataset={cfg.dataset} gpu={gpu_id} "
          f"n={len(preds)} L={ctx['n_layers']} H={ctx['n_heads']}", flush=True)

    records = []
    n_done = n_skipped = 0
    for cnt, p in enumerate(preds):
        rec = model.extract_one(p, {}, ctx, svar_shift, rank=rank)
        if rec is None:
            n_skipped += 1
            continue
        records.append(rec)
        n_done += 1
        if n_done % 25 == 0:
            print(f"[worker {rank}] [{cnt+1}/{len(preds)}] done={n_done} skip={n_skipped}", flush=True)

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
    video_dir = cfg.path("video_dir")
    pred_file = os.path.join(cfg.pred_dir(args.seed), "predictions.json")
    out_dir = cfg.feat_dir(args.seed)

    preds = json.load(open(pred_file))
    if cfg.extract.n_items:
        preds = preds[:cfg.extract.n_items]

    set_start_method("spawn", force=True)
    chunks = np.array_split(preds, len(gpus))
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(gpus, chunks)):
        if len(chunk) == 0:
            continue
        p = Process(target=worker, args=(rank, gpu, list(chunk), out_dir, cfg.config_path,
                                         video_dir, args.svar_shift))
        p.start(); procs.append((rank, p))
    failed = []
    for rank, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((rank, p.exitcode))
    if failed:
        # a crashed worker leaves its shard unwritten -> a silently incomplete run. Fail loudly.
        raise SystemExit(f"[video_extract] {len(failed)} worker(s) failed (rank, exitcode): {failed}; "
                         f"shards under {out_dir} are INCOMPLETE — fix the error and re-run.")
    print(f"all {len(procs)} workers complete")


if __name__ == "__main__":
    main()
