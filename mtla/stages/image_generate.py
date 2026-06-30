"""Shared image_det generation driver (any image model + dataset).

The generate half of the decoupled image pipeline (the extract half is image_extract.py). One
config-driven script for both backends and both models; it owns the orchestration and delegates
every model/dataset specific piece to adapter callbacks (mirrors video_generate.py):

  vLLM (engine: vllm, fast — the default for COCO):
    - `model.make_vllm_prep()`      -> a closure `prep(item, dataset) -> {prompt, multi_modal_data}`
                                       (loads the model's tokenizer/processor once; builds the
                                       model-specific prompt + multimodal payload).
    - `model.vllm_engine_kwargs()`  -> extra AsyncEngineArgs (e.g. trust_remote_code).
    - `model.vllm_uses_seed()`      -> whether to pass the rollout seed to SamplingParams.
  HF (engine: hf, reference, no vLLM install):
    - `model.make_hf_generate(gpu)` -> a closure `gen(item, dataset, seed) -> (response, trunc)`.

  Either backend:
    - `dataset.load_items(cfg)`                         -> the work items (adapter owns file I/O).
    - `dataset.make_prediction(item, response, model, truncated)` -> the prediction record (parses
      via `model.parse`), the same schema the extract stage reads back.

    python -m mtla.stages.image_generate --config configs/coco_internvl.yaml --seed 0

Writes `<predictions>/seed{K}/predictions.json` (merged + idx-sorted across workers).
"""
import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import multiprocessing as mp

mp.set_start_method("spawn", force=True)


# ---------------------------------------------------------------------------
# vLLM backend: CPU prep workers feed an async multi-engine pool (one per GPU group).
# ---------------------------------------------------------------------------
def cpu_worker(raw_queue, ready_queues, config_path, num_engines, worker_id):
    """Build each item's vLLM request via the model adapter's prep closure, route round-robin."""
    try:
        os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
        import torch; torch.set_num_threads(1)
        from mtla.config import load_config
        from mtla.registry import resolve
        cfg = load_config(config_path)
        model, dataset = resolve(cfg.model, cfg.dataset)
        prep = model.make_vllm_prep()
        while True:
            task = raw_queue.get()
            if task is None:
                break
            idx, item = task
            try:
                req = prep(item, dataset)
                ready_queues[idx % num_engines].put({"idx": idx, "item": item, **req})
            except Exception:
                print(f"[CPU {worker_id}] error idx {idx}: {traceback.format_exc()}", flush=True)
                ready_queues[idx % num_engines].put({"idx": idx, "item": item, "error": True})
    except Exception:
        print(f"[CPU {worker_id}] FATAL: {traceback.format_exc()}", flush=True)


def gpu_worker(engine_id, gpu_ids_str, ready_queue, result_queue, config_path, args):
    """One vLLM AsyncLLMEngine on its GPU group; concurrently generates for queued requests."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cache_dir = f"/tmp/vllm_cache_mtla/engine_{engine_id}"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = cache_dir
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        from vllm.utils import random_uuid
        from mtla.config import load_config
        from mtla.registry import resolve
        cfg = load_config(config_path)
        model, dataset = resolve(cfg.model, cfg.dataset)

        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=model.model_id, tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
            max_num_seqs=args.concurrency, max_num_batched_tokens=32768,
            disable_log_stats=True, enforce_eager=False, limit_mm_per_prompt={"image": 1},
            disable_custom_all_reduce=True, **model.vllm_engine_kwargs()))
        print(f"[Engine {engine_id}] ready (GPUs {gpu_ids_str}, TP={args.tp})", flush=True)
        sp_kwargs = dict(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)
        if model.vllm_uses_seed():
            sp_kwargs["seed"] = args.seed
        sampling = SamplingParams(**sp_kwargs)

        async def run():
            local_q = asyncio.Queue(maxsize=args.concurrency * 2)
            active = set(); stats = {"done": 0, "err": 0}; shutdown = False

            def bridge():
                while True:
                    try:
                        it = ready_queue.get(timeout=5)
                        asyncio.run_coroutine_threadsafe(local_q.put(it), loop).result()
                        if it is None:
                            return
                    except queue.Empty:
                        continue
            loop = asyncio.get_event_loop()
            threading.Thread(target=bridge, daemon=True).start()

            async def infer(task):
                try:
                    if task.get("error"):
                        result_queue.put(dataset.make_prediction(task["item"], "", model)); stats["err"] += 1; return
                    final = None
                    async for r in engine.generate(
                        {"prompt": task["prompt"], "multi_modal_data": task["multi_modal_data"]},
                        sampling, random_uuid()):
                        final = r
                    response = final.outputs[0].text if final else ""
                    trunc = len(final.outputs[0].token_ids) >= args.max_new_tokens if final else False
                    rec = dataset.make_prediction(task["item"], response, model, truncated=trunc)
                    rec["idx"] = task["idx"]
                    result_queue.put(rec); stats["done"] += 1
                except Exception:
                    print(f"[Engine {engine_id}] {traceback.format_exc()}", flush=True)
                    result_queue.put(dataset.make_prediction(task["item"], "", model)); stats["err"] += 1
                finally:
                    active.discard(asyncio.current_task())

            while not shutdown:
                while len(active) < args.concurrency:
                    try:
                        task = local_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if task is None:
                        shutdown = True; break
                    active.add(asyncio.create_task(infer(task)))
                if active:
                    _, active = await asyncio.wait(active, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                else:
                    await asyncio.sleep(0.05)
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            print(f"[Engine {engine_id}] done={stats['done']} err={stats['err']}", flush=True)
        asyncio.run(run())
    except Exception:
        print(f"[Engine {engine_id}] FATAL: {traceback.format_exc()}", flush=True)
        sys.exit(1)


def run_vllm(cfg, args, samples):
    gpu_ids = cfg.stage_gpus("generate")
    assert len(gpu_ids) % args.tp == 0, f"#GPUs {len(gpu_ids)} not divisible by TP {args.tp}"
    groups = [gpu_ids[i:i + args.tp] for i in range(0, len(gpu_ids), args.tp)]
    num_engines = len(groups)
    print(f"Samples: {len(samples)}, Engines: {num_engines}, TP: {args.tp}, GPUs: {groups}")

    raw_q = mp.Queue()
    ready_qs = [mp.Queue(maxsize=500) for _ in range(num_engines)]
    result_q = mp.Queue()
    gpu_procs = [mp.Process(target=gpu_worker, args=(i, ",".join(map(str, g)), ready_qs[i],
                                                     result_q, cfg.config_path, args))
                 for i, g in enumerate(groups)]
    for p in gpu_procs:
        p.start()
    cpu_procs = [mp.Process(target=cpu_worker, args=(raw_q, ready_qs, cfg.config_path, num_engines, i))
                 for i in range(args.num_cpu_workers)]
    for p in cpu_procs:
        p.start()
    for idx, item in enumerate(samples):
        raw_q.put((idx, item))
    for _ in range(args.num_cpu_workers):
        raw_q.put(None)

    results = []; start = time.time(); last = start
    while len(results) < len(samples):
        try:
            results.append(result_q.get(timeout=300))
            now = time.time()
            if now - last > 10:
                spd = len(results) / (now - start)
                print(f"  [{len(results)}/{len(samples)}] {spd:.1f}/s ETA {(len(samples)-len(results))/max(spd,1e-9):.0f}s", flush=True)
                last = now
        except queue.Empty:
            if not any(p.is_alive() for p in gpu_procs):
                print("All GPU workers died!", flush=True); break
    for p in cpu_procs:
        p.join(timeout=30)
    for q in ready_qs:
        q.put(None)
    for p in gpu_procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    return results


# ---------------------------------------------------------------------------
# HF backend: one per-GPU worker loops over its chunk (no vLLM install needed).
# ---------------------------------------------------------------------------
def hf_worker(rank, gpu_id, items, pred_dir, config_path, seed):
    import torch
    torch.cuda.set_device(gpu_id)
    from mtla.config import load_config
    from mtla.registry import resolve
    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    gen = model.make_hf_generate(gpu_id)
    print(f"[worker {rank}] hf gpu={gpu_id} n={len(items)} seed={seed}", flush=True)
    results = []
    for cnt, item in enumerate(items):
        try:
            response, trunc = gen(item, dataset, seed)
            rec = dataset.make_prediction(item, response, model, truncated=trunc)
        except Exception as e:
            print(f"[worker {rank}] error {item.get('id')}: {e}", flush=True)
            rec = dataset.make_prediction(item, "", model)
        rec["idx"] = item.get("idx", cnt)
        results.append(rec)
        if (cnt + 1) % 25 == 0:
            print(f"[worker {rank}] [{cnt+1}/{len(items)}]", flush=True)
        torch.cuda.empty_cache()
    os.makedirs(pred_dir, exist_ok=True)
    with open(f"{pred_dir}/preds_rank{rank}.json", "w") as f:
        json.dump(results, f)
    print(f"[worker {rank}] saved {len(results)}", flush=True)


def run_hf(cfg, args, samples):
    import numpy as np
    gpu_ids = cfg.stage_gpus("generate")
    for i, it in enumerate(samples):
        it.setdefault("idx", i)
    chunks = np.array_split(samples, len(gpu_ids))
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(gpu_ids, chunks)):
        if len(chunk) == 0:
            continue
        p = mp.Process(target=hf_worker, args=(rank, gpu, list(chunk), args.pred_dir,
                                               cfg.config_path, args.seed))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    merged = []
    for rank in range(len(gpu_ids)):
        pp = f"{args.pred_dir}/preds_rank{rank}.json"
        if os.path.exists(pp):
            merged.extend(json.load(open(pp)))
    return merged


def main():
    from mtla.config import load_config
    from mtla.registry import resolve
    ap = argparse.ArgumentParser(description="Shared image_det generation (vLLM or HF).")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--seed", type=int, default=0, help="rollout seed (selects seed{K}/ out dir)")
    # backend performance knobs (not part of the run config; tune per box)
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size per engine")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num_cpu_workers", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, dataset = resolve(cfg.model, cfg.dataset)
    args.temperature = cfg.generate.temperature
    args.pred_dir = cfg.pred_dir(args.seed)
    samples = dataset.load_items(cfg)[:(cfg.generate.n_items or len(dataset.load_items(cfg)))]

    if cfg.generate.engine == "vllm":
        results = run_vllm(cfg, args, samples)
    else:
        results = run_hf(cfg, args, samples)

    results.sort(key=lambda r: r.get("idx", 0))
    os.makedirs(args.pred_dir, exist_ok=True)
    with open(os.path.join(args.pred_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    n_ok = sum(1 for r in results if r.get("status") == "success")
    print(f"Saved {len(results)} predictions ({n_ok} ok) to {args.pred_dir}/predictions.json")


if __name__ == "__main__":
    main()
