"""Two reusable generation-execution strategies, shared by scripts/stages/generate.py.

Both drive the SAME model/dataset contract (see mtla.models.base.ModelAdapter generation methods
+ mtla.data.base.DatasetAdapter gen_* attributes); they differ only in HOW work is spread over
GPUs. A dataset picks one via `dataset.gen_strategy`:

  run_pooled   — an async multi-engine vLLM pool: CPU prep workers build requests and feed them
                 round-robin into one persistent AsyncLLMEngine per GPU group. Best for MANY small
                 requests (e.g. 5000 COCO images) where per-request latency hides behind throughput.
  run_sharded  — one blocking engine per GPU (np.array_split of the items). Each worker pins ONE
                 GPU (CUDA_VISIBLE_DEVICES before torch import) and loops its chunk. Best for HEAVY
                 per-item work (video clips) and the only option for engine: hf.

Either modality may use either strategy. Both return the merged list of prediction records
(dataset.make_prediction output); generate.py sorts + writes predictions.json.

The record schema is owned by `dataset.make_prediction(item, response, model, truncated)` — the
strategies never touch it, so image/video/any-future dataset all flow through unchanged.
"""
import asyncio
import json
import os
import queue
import sys
import threading
import time
import traceback

import multiprocessing as mp


# ===========================================================================
# Strategy A — async multi-engine vLLM pool (throughput on many small requests)
# ===========================================================================
def _pooled_cpu_worker(raw_queue, ready_queues, config_path, num_engines, worker_id):
    """Build each item's vLLM request via the model adapter (`build_vllm_request`), route it
    round-robin to an engine's ready-queue. Prep is CPU-only, so many run in parallel."""
    try:
        os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
        import torch; torch.set_num_threads(1)
        from mtla.config import load_config
        from mtla.registry import resolve
        cfg = load_config(config_path)
        model, dataset = resolve(cfg.model, cfg.dataset)
        proc = model.gen_processor()
        while True:
            task = raw_queue.get()
            if task is None:
                break
            idx, item = task
            try:
                req = model.build_vllm_request(proc, item, dataset, cfg)
                if req is None:                       # adapter chose to skip this item
                    ready_queues[idx % num_engines].put({"idx": idx, "item": item, "skip": True})
                else:
                    ready_queues[idx % num_engines].put({"idx": idx, "item": item, **req})
            except Exception:
                print(f"[CPU {worker_id}] error idx {idx}: {traceback.format_exc()}", flush=True)
                ready_queues[idx % num_engines].put({"idx": idx, "item": item, "error": True})
    except Exception:
        print(f"[CPU {worker_id}] FATAL: {traceback.format_exc()}", flush=True)


def _pooled_gpu_worker(engine_id, gpu_ids_str, ready_queue, result_queue, config_path, args):
    """One vLLM AsyncLLMEngine on its GPU group; concurrently generates for queued requests and
    puts dataset.make_prediction records on result_queue."""
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
            model=model.model_id, tensor_parallel_size=args["tp"],
            max_model_len=args["max_model_len"], gpu_memory_utilization=args["gpu_mem_util"],
            max_num_seqs=args["concurrency"], max_num_batched_tokens=32768,
            disable_log_stats=True, enforce_eager=False,
            disable_custom_all_reduce=True, **model.vllm_engine_args(dataset)))
        print(f"[Engine {engine_id}] ready (GPUs {gpu_ids_str}, TP={args['tp']})", flush=True)
        sp_kwargs = dict(temperature=args["temperature"], top_p=args["top_p"],
                         max_tokens=args["max_new_tokens"])
        if model.vllm_uses_seed(dataset.task):
            sp_kwargs["seed"] = args["seed"]
        sampling = SamplingParams(**sp_kwargs)

        async def run():
            local_q = asyncio.Queue(maxsize=args["concurrency"] * 2)
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
                    if task.get("error") or task.get("skip"):
                        result_queue.put(dataset.make_prediction(task["item"], "", model))
                        stats["err"] += 1; return
                    final = None
                    req = {"prompt": task["prompt"], "multi_modal_data": task["multi_modal_data"]}
                    if task.get("mm_processor_kwargs"):
                        req["mm_processor_kwargs"] = task["mm_processor_kwargs"]
                    async for r in engine.generate(req, sampling, random_uuid()):
                        final = r
                    response = final.outputs[0].text if final else ""
                    trunc = len(final.outputs[0].token_ids) >= args["max_new_tokens"] if final else False
                    rec = dataset.make_prediction(task["item"], response, model, truncated=trunc)
                    rec["idx"] = task["idx"]
                    result_queue.put(rec); stats["done"] += 1
                except Exception:
                    print(f"[Engine {engine_id}] {traceback.format_exc()}", flush=True)
                    result_queue.put(dataset.make_prediction(task["item"], "", model)); stats["err"] += 1
                finally:
                    active.discard(asyncio.current_task())

            while not shutdown:
                while len(active) < args["concurrency"]:
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


def run_pooled(cfg, samples, args):
    """Async multi-engine vLLM pool. `args` is a dict of tuning + effective sampling knobs.
    Returns the merged list of prediction records."""
    gpu_ids = cfg.stage_gpus("generate")
    tp = args["tp"]
    assert len(gpu_ids) % tp == 0, f"#GPUs {len(gpu_ids)} not divisible by TP {tp}"
    groups = [gpu_ids[i:i + tp] for i in range(0, len(gpu_ids), tp)]
    num_engines = len(groups)
    print(f"[pooled] samples={len(samples)} engines={num_engines} TP={tp} GPUs={groups}", flush=True)

    raw_q = mp.Queue()
    ready_qs = [mp.Queue(maxsize=500) for _ in range(num_engines)]
    result_q = mp.Queue()
    gpu_procs = [mp.Process(target=_pooled_gpu_worker,
                            args=(i, ",".join(map(str, g)), ready_qs[i], result_q, cfg.config_path, args))
                 for i, g in enumerate(groups)]
    for p in gpu_procs:
        p.start()
    cpu_procs = [mp.Process(target=_pooled_cpu_worker,
                            args=(raw_q, ready_qs, cfg.config_path, num_engines, i))
                 for i in range(args["num_cpu_workers"])]
    for p in cpu_procs:
        p.start()
    for idx, item in enumerate(samples):
        raw_q.put((idx, item))
    for _ in range(args["num_cpu_workers"]):
        raw_q.put(None)

    results = []; start = time.time(); last = start
    while len(results) < len(samples):
        try:
            results.append(result_q.get(timeout=300))
            now = time.time()
            if now - last > 10:
                spd = len(results) / (now - start)
                print(f"  [{len(results)}/{len(samples)}] {spd:.1f}/s "
                      f"ETA {(len(samples)-len(results))/max(spd,1e-9):.0f}s", flush=True)
                last = now
        except queue.Empty:
            if not any(p.is_alive() for p in gpu_procs):
                print("[pooled] all GPU workers died!", flush=True); break
    for p in cpu_procs:
        p.join(timeout=30)
    for q in ready_qs:
        q.put(None)
    for p in gpu_procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    return results


# ===========================================================================
# Strategy B — one blocking engine per GPU (heavy per-item work; the only hf path)
# ===========================================================================
def _sharded_worker(rank, gpu_id, items, out_dir, config_path, engine, args):
    """One GPU worker. Pins its GPU (CUDA_VISIBLE_DEVICES before torch import), builds the engine
    (offline vLLM or HF), loops its chunk, and writes preds_rank{rank}.json."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # BEFORE torch import (vLLM grabs device 0)
    import torch
    from transformers import set_seed
    from mtla.config import load_config
    from mtla.registry import resolve
    set_seed(args["seed"] * 1000 + rank)               # reproducible per (seed, rank) draw
    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    temperature = args["temperature"]
    print(f"[worker {rank}] sharded engine={engine} model={cfg.model} dataset={cfg.dataset} "
          f"gpu={gpu_id} n={len(items)} seed={args['seed']} T={temperature}", flush=True)

    results = []
    if engine == "vllm":
        from vllm import LLM, SamplingParams
        proc = model.gen_processor()
        llm = LLM(model=model.model_id, gpu_memory_utilization=0.90, enforce_eager=False,
                  disable_log_stats=True, **model.vllm_engine_args(dataset))
        sp_kwargs = dict(temperature=temperature, top_p=args["top_p"], max_tokens=args["max_new_tokens"]) \
            if temperature and temperature > 0 else dict(temperature=0.0, max_tokens=args["max_new_tokens"])
        if model.vllm_uses_seed(dataset.task) and temperature and temperature > 0:
            sp_kwargs["seed"] = args["seed"]
        sp = SamplingParams(**sp_kwargs)
        for cnt, item in enumerate(items):
            try:
                req = model.build_vllm_request(proc, item, dataset, cfg)
                if req is None:
                    continue
                gen_req = {"prompt": req["prompt"], "multi_modal_data": req["multi_modal_data"]}
                if req.get("mm_processor_kwargs"):
                    gen_req["mm_processor_kwargs"] = req["mm_processor_kwargs"]
                out = llm.generate([gen_req], sp, use_tqdm=False)
                response = out[0].outputs[0].text.strip() if out and out[0].outputs else ""
                if not response:
                    continue
                rec = dataset.make_prediction(item, response, model)
                rec["idx"] = item.get("idx", cnt)
                results.append(rec)
            except Exception as e:
                print(f"[worker {rank}] skip {cnt}: {e}", flush=True)
            if (cnt + 1) % 25 == 0:
                print(f"[worker {rank}] [{cnt+1}/{len(items)}] done={len(results)}", flush=True)
    else:  # hf
        ctx = model.load_hf_gen(0)                     # one GPU visible -> cuda:0
        for cnt, item in enumerate(items):
            try:
                response, trunc = model.generate_hf(ctx, item, dataset, cfg, args["seed"], temperature)
                if response is None:
                    continue
                rec = dataset.make_prediction(item, response, model, truncated=trunc)
                rec["idx"] = item.get("idx", cnt)
                results.append(rec)
            except Exception as e:
                print(f"[worker {rank}] error {cnt}: {e}", flush=True)
            if (cnt + 1) % 25 == 0:
                print(f"[worker {rank}] [{cnt+1}/{len(items)}] done={len(results)}", flush=True)
            torch.cuda.empty_cache()

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/preds_rank{rank}.json", "w") as f:
        json.dump(results, f)
    print(f"[worker {rank}] saved {len(results)} -> {out_dir}/preds_rank{rank}.json", flush=True)


def run_sharded(cfg, samples, args, out_dir, engine):
    """One blocking engine per GPU. Writes per-rank shards under `out_dir`, then merges + returns."""
    import numpy as np
    gpu_ids = cfg.stage_gpus("generate")
    print(f"[sharded] samples={len(samples)} gpus={gpu_ids} engine={engine}", flush=True)
    chunks = np.array_split(samples, len(gpu_ids))
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(gpu_ids, chunks)):
        if len(chunk) == 0:
            continue
        p = mp.Process(target=_sharded_worker,
                       args=(rank, gpu, list(chunk), out_dir, cfg.config_path, engine, args))
        p.start(); procs.append((rank, p))
    failed = []
    for rank, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((rank, p.exitcode))
    if failed:
        raise SystemExit(f"[sharded] {len(failed)} worker(s) failed (rank, exitcode): {failed}; "
                         f"shards under {out_dir} are INCOMPLETE — fix the error and re-run.")
    merged = []
    for rank in range(len(gpu_ids)):
        pp = f"{out_dir}/preds_rank{rank}.json"
        if os.path.exists(pp):
            merged.extend(json.load(open(pp)))
    return merged
