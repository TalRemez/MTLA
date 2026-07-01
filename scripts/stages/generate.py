"""Unified generation driver — any model, any dataset, any modality (vLLM only).

The generate half of the decoupled pipeline (extract is extract.py). One config-driven script for
every benchmark. It owns only the modality-agnostic skeleton:

  1. load the run config + resolve the (model, dataset) adapters from the registry;
  2. load the work items (``dataset.load_items(cfg)``) and slice to ``generate.n_items``;
  3. tag each item with a global ``idx`` (deterministic merge order);
  4. pick the execution STRATEGY the dataset declares (``dataset.gen_strategy``) and run it;
  5. sort by idx, write ``<predictions>/seed{K}/predictions.json``.

Everything model/dataset specific lives behind the adapter contract:
  - the MODEL builds vLLM requests: ``gen_processor``, ``build_request``, ``vllm_engine_args``,
    ``vllm_uses_seed`` (see mtla.models.base);
  - the DATASET owns items + the uniform record + the strategy: ``load_items``, ``gen_record``,
    ``gen_strategy``;
  - the two strategies live in ``_gen_strategies.py``: ``run_pooled`` (async multi-engine pool,
    throughput on many small requests) and ``run_sharded`` (one engine per GPU, heavy per-item work).

    python scripts/stages/generate.py --config configs/coco_internvl.yaml --seed 0

Writes ``<predictions>/seed{K}/predictions.json`` (merged + idx-sorted across workers).
"""
import argparse
import json
import os

import multiprocessing as mp

# Both strategies spawn CUDA subprocesses; the spawn start-method is required.
mp.set_start_method("spawn", force=True)

from _gen_strategies import run_pooled, run_sharded

from mtla.config import load_config
from mtla.registry import resolve


def main():
    ap = argparse.ArgumentParser(description="Unified MTLA vLLM generation (pooled or sharded).")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--seed", type=int, default=0, help="rollout seed (selects seed{K}/ out dir)")
    # Backend tuning knobs (not part of the run config; tune per box). The pool uses all of them;
    # the sharded path uses only the sampling knobs (temperature/top_p/max_new_tokens/seed).
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size per engine (pooled)")
    ap.add_argument("--max_model_len", type=int, default=16384, help="pooled vLLM engine context")
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--concurrency", type=int, default=32, help="pooled: in-flight requests per engine")
    ap.add_argument("--num_cpu_workers", type=int, default=16, help="pooled: request-prep workers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, dataset = resolve(cfg.model, cfg.dataset)

    # One seed per invocation -> one effective temperature (greedy anchor for seed 0 if greedy_seed0).
    temperature = cfg.gen_temperature(args.seed, dataset.greedy_seed0)
    pred_dir = cfg.pred_dir(args.seed)

    items = dataset.load_items(cfg)
    if cfg.generate.n_items:
        items = items[:cfg.generate.n_items]
    for i, it in enumerate(items):
        it.setdefault("idx", i)                        # deterministic merge order

    sargs = {"seed": args.seed, "temperature": temperature, "top_p": cfg.generate.top_p,
             "max_new_tokens": cfg.generate.max_new_tokens,
             "tp": args.tp, "max_model_len": args.max_model_len, "gpu_mem_util": args.gpu_mem_util,
             "concurrency": args.concurrency, "num_cpu_workers": args.num_cpu_workers}

    print(f"[generate] model={cfg.model} dataset={cfg.dataset} task={dataset.task} "
          f"strategy={dataset.gen_strategy} seed={args.seed} T={temperature} n={len(items)}", flush=True)

    if dataset.gen_strategy == "pooled":
        results = run_pooled(cfg, items, sargs)
    else:
        results = run_sharded(cfg, items, sargs, pred_dir)

    results.sort(key=lambda r: r.get("idx", 0))
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(pred_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    n_ok = sum(1 for r in results if r.get("response"))
    print(f"[generate] saved {len(results)} predictions ({n_ok} non-empty) -> "
          f"{pred_dir}/predictions.json", flush=True)


if __name__ == "__main__":
    main()
