"""Stage 2 — MTLA extraction. HF eager-attention forward for any model / dataset / modality.

Model- and dataset-agnostic: it loads the run config, resolves the (model, dataset) adapters,
shards the generation records across GPUs, and for each record calls ``model.extract_one`` — which
runs one HF-eager captured forward and applies the MTLA math (see ``mtla.mtla_attn``). Every
model/task specific lives behind the adapter's extraction callbacks; this file is just the
multi-GPU harness. The generation records are self-contained (``{id, prompt, response, gt, extra}``),
so a worker just streams ``predictions.json`` — no dataset item lookup.

    python extract.py --config configs/coco_internvl.yaml            # seeds 0..n_rollouts-1
    python extract.py --config configs/coco_internvl.yaml --seeds 0 1 2

Reads ``<predictions>/seed{K}/predictions.json``; writes ``<features>/seed{K}/shard{rank}.pt``.
"""
import argparse
import json
import os
from multiprocessing import Process, set_start_method

import numpy as np

from mtla.config import load_config
from mtla.registry import resolve
from mtla.mtla_attn import compute_mtla


def worker(rank, gpu_id, records, out_dir, config_path):
    """Extract one shard of the records on ``gpu_id`` and save it. Runs in a spawned subprocess."""
    # Pin this worker to ONE GPU before any CUDA use, so it sees the target device as cuda:0. Doing
    # it here (not via torch.cuda.set_device on an absolute id) keeps the run correct even when
    # CUDA_VISIBLE_DEVICES is already set, and mirrors the generate stage. spawn re-imports the
    # module in the child, but `import torch` alone does not initialize CUDA, so this is early enough.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    # Load the model + processor, install the attention-capture hook, and get the extraction ctx.
    ctx = model.load_for_extract(0, dataset.task)
    if dataset.task == "video_span":
        # Video extraction reads the same preprocessing the generate stage used, plus the
        # multi-span flag (fuse = multi-window benchmark, argmax = single-span).
        ctx["preprocess"] = cfg.preprocess
        ctx["multi"] = (dataset.select == "fuse")
    print(f"[worker {rank}] model={cfg.model} dataset={cfg.dataset} gpu={gpu_id} "
          f"n={len(records)} L={ctx['n_layers']} H={ctx['n_heads']}", flush=True)

    out, n_done = [], 0
    for cnt, rec in enumerate(records):
        # Run MTLA for this item's predictions (one captured forward -> per-prediction [L,H] arrays);
        # `model` just supplies the callbacks compute_mtla drives. None = nothing extractable, skip.
        result = compute_mtla(model, rec, ctx, rank=rank)
        if result is None:
            continue
        out.append(result); n_done += 1
        if n_done % 25 == 0:
            print(f"[worker {rank}] [{cnt + 1}/{len(records)}] done={n_done}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/shard{rank}.pt"
    torch.save(out, path)
    print(f"[worker {rank}] saved {len(out)} records / "
          f"{sum(len(r['objects']) for r in out)} preds -> {path}", flush=True)


def extract_seed(cfg, seed):
    """Extract MTLA features for one rollout ``seed``, sharded across the extract GPUs."""
    gpus = cfg.stage_gpus("extract")
    records = json.load(open(os.path.join(cfg.pred_dir(seed), "predictions.json")))
    if cfg.extract.n_items:
        records = records[:cfg.extract.n_items]
    out_dir = cfg.feat_dir(seed)

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
    print(f"[extract] all {len(procs)} workers complete (seed {seed})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="rollout seeds to extract (default: 0..n_rollouts-1)")
    ap.add_argument("--n", type=int, default=None, help="override n_rollouts (sets the default seeds)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n is not None:
        cfg.n_rollouts = args.n
    set_start_method("spawn", force=True)
    seeds = args.seeds if args.seeds is not None else cfg.seeds()
    for i, seed in enumerate(seeds):
        print(f"[extract] seed {seed}  ({i + 1}/{len(seeds)})", flush=True)
        extract_seed(cfg, seed)


if __name__ == "__main__":
    main()
