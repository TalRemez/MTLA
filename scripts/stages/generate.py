"""Unified generation driver — any model, any dataset, any modality.

The generate half of the decoupled pipeline (extract is {image,video}_extract.py). One config-driven
script for every benchmark. It owns only the modality-agnostic skeleton:

  1. load the run config + resolve the (model, dataset) adapters from the registry;
  2. load the work items (`dataset.load_items(cfg)`) and slice to `generate.n_items`;
  3. tag each item with a global `idx` (for a deterministic merge order);
  4. pick the execution STRATEGY the dataset declares (`dataset.gen_strategy`) and run it;
  5. sort by idx, write `<predictions>/seed{K}/predictions.json`.

Everything model/dataset/modality-specific lives behind the adapter contract:
  - the MODEL builds requests / runs the forward: `gen_processor`, `build_vllm_request`,
    `vllm_engine_args`, `vllm_uses_seed`, `load_hf_gen`, `generate_hf` (see mtla.models.base).
  - the DATASET owns file I/O + the record schema + the strategy + sampling knobs:
    `load_items`, `make_prediction`, `gen_strategy`, `gen_max_new_tokens`, `gen_top_p`.
  - the two strategies live in `_gen_strategies.py`: `run_pooled` (async multi-engine vLLM pool,
    throughput on many small requests) and `run_sharded` (one blocking engine per GPU, heavy
    per-item work; the only path for engine: hf).

    python scripts/stages/generate.py --config configs/coco_internvl.yaml --seed 0
    python scripts/stages/generate.py --config configs/qvhighlights_qwen3vl.yaml --seed 0

Writes `<predictions>/seed{K}/predictions.json` (merged + idx-sorted across workers).
"""
import argparse
import json
import os

import multiprocessing as mp

# Both strategies spawn CUDA subprocesses; spawn start-method is required.
mp.set_start_method("spawn", force=True)

from _gen_strategies import run_pooled, run_sharded


def main():
    from mtla.config import load_config
    from mtla.registry import resolve
    ap = argparse.ArgumentParser(description="Unified MTLA generation (vLLM pooled/sharded or HF).")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--seed", type=int, default=0, help="rollout seed (selects seed{K}/ out dir)")
    # backend tuning knobs (not part of the run config; tune per box). The vLLM pool uses all of
    # these; the sharded path uses top_p/max_new_tokens/temperature/seed.
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size per engine (pooled)")
    ap.add_argument("--max_model_len", type=int, default=16384, help="pooled vLLM engine context")
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--concurrency", type=int, default=32, help="pooled: in-flight requests per engine")
    ap.add_argument("--num_cpu_workers", type=int, default=16, help="pooled: request-prep workers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, dataset = resolve(cfg.model, cfg.dataset)

    engine = cfg.generate.engine
    if engine not in model.gen_engines(dataset.task):
        raise SystemExit(f"model {cfg.model!r} does not support engine {engine!r} for task "
                         f"{dataset.task!r}; supported: {list(model.gen_engines(dataset.task))}")

    # One seed per invocation -> one effective temperature (greedy anchor for seed 0 if greedy_seed0).
    temperature = cfg.gen_temperature(args.seed, dataset.greedy_seed0)
    pred_dir = cfg.pred_dir(args.seed)

    items = dataset.load_items(cfg)
    if cfg.generate.n_items:
        items = items[:cfg.generate.n_items]
    for i, it in enumerate(items):
        it.setdefault("idx", i)                        # deterministic merge order

    # Effective sampling + tuning knobs handed to whichever strategy runs.
    sargs = {"seed": args.seed, "temperature": temperature,
             "top_p": dataset.gen_top_p, "max_new_tokens": dataset.gen_max_new_tokens,
             "tp": args.tp, "max_model_len": args.max_model_len, "gpu_mem_util": args.gpu_mem_util,
             "concurrency": args.concurrency, "num_cpu_workers": args.num_cpu_workers}

    # Force sharded for HF (no async engine); otherwise honor the dataset's declared strategy.
    strategy = "sharded" if engine == "hf" else dataset.gen_strategy
    print(f"[generate] model={cfg.model} dataset={cfg.dataset} task={dataset.task} engine={engine} "
          f"strategy={strategy} seed={args.seed} T={temperature} n={len(items)}", flush=True)

    if strategy == "pooled":
        results = run_pooled(cfg, items, sargs)
    else:
        results = run_sharded(cfg, items, sargs, pred_dir, engine)

    results.sort(key=lambda r: r.get("idx", 0))
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(pred_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    n_ok = sum(1 for r in results if r.get("status") == "success" or r.get("is_correct"))
    print(f"[generate] saved {len(results)} predictions ({n_ok} ok/correct) -> "
          f"{pred_dir}/predictions.json", flush=True)


if __name__ == "__main__":
    main()
