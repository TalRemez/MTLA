"""Two reusable vLLM generation strategies, shared by generate.py.

Both drive the SAME adapter contract (``model.build_vllm_request`` / ``model.vllm_*`` +
``dataset.gen_record``); they differ only in HOW work is spread over GPUs. A dataset picks one via
``dataset.gen_strategy``:

  run_pooled   — an async multi-engine vLLM pool: CPU prep workers build requests and feed them
                 round-robin into one persistent AsyncLLMEngine per GPU group. Best for MANY small
                 requests (e.g. 5000 COCO images), where per-request latency hides behind throughput.
  run_sharded  — one blocking ``LLM`` engine per GPU (``np.array_split`` of the items). Each worker
                 pins ONE GPU (CUDA_VISIBLE_DEVICES before torch import) and loops its chunk. Best
                 for HEAVY per-item work (video clips).

Both return the merged list of uniform generation records (``dataset.gen_record`` output), one per
item per rollout; the record schema is owned by the dataset, so the strategies never touch it beyond
tagging each with its ``idx`` (item order) and ``rollout`` (which rollout it belongs to), which
generate.py uses to split the flat list into per-rollout ``predictions.json`` files.

All rollouts are produced in one pass: the rollout plan (``args["groups"]``) groups rollouts by
temperature, and each group is one vLLM request with ``SamplingParams(n=<#rollouts>)`` so the prompt
is prefilled once per group and the KV cache is shared across that group's samples.
"""

import asyncio
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback

import multiprocessing as mp

import numpy as np

from mtla.config import load_config
from mtla.registry import resolve

# NOTE: torch / vllm are imported INSIDE the worker functions below, on purpose. Each worker is a
# spawned subprocess that must set CUDA_VISIBLE_DEVICES *before* CUDA initializes; importing torch
# or vllm at module top would grab the wrong device. This is the one deliberate exception to the
# "imports at the top" rule, and it is load-bearing for correct multi-GPU pinning.


# ===========================================================================
# Strategy A — async multi-engine vLLM pool (throughput on many small requests)
# ===========================================================================
def _pooled_cpu_worker(
    raw_queue, ready_queues, config_path: str, num_engines: int, worker_id: int
) -> None:
    """Build vLLM requests on the CPU and route them round-robin to the engines.

    A pooled-strategy prep worker: it pulls raw items off ``raw_queue``, builds each
    item's request via ``model.build_vllm_request``, and pushes the result onto the ready
    queue of engine ``idx % num_engines``. Request prep is CPU-only, so many workers
    run in parallel to keep the GPU engines fed. Runs in a spawned subprocess.

    Args:
        raw_queue: shared queue of ``(idx, item)`` tasks; a ``None`` sentinel signals
            this worker to stop.
        ready_queues: one per engine; the built request is enqueued on the engine
            selected by ``idx % num_engines``.
        config_path: path to the run config, reloaded to resolve the adapters.
        num_engines: number of GPU engines, used for the round-robin routing.
        worker_id: this prep worker's index, used only for log messages.

    Returns:
        None. Side effect: enqueues per-item request dicts (or a ``skip`` marker when the
        adapter returns ``None``, e.g. missing media) onto the ready queues.
    """
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        import torch

        torch.set_num_threads(1)
        cfg = load_config(config_path)
        model, dataset = resolve(cfg.model, cfg.dataset)
        proc = model.gen_processor()
        while True:
            task = raw_queue.get()
            if task is None:
                break
            idx, item = task
            req = model.build_vllm_request(proc, item, dataset, cfg)
            if req is None:  # adapter chose to skip this item
                ready_queues[idx % num_engines].put(
                    {"idx": idx, "item": item, "skip": True}
                )
            else:
                ready_queues[idx % num_engines].put({"idx": idx, "item": item, **req})
    except Exception:
        # A request-prep bug is real: die loudly (nonzero exit) rather than mark items
        # "error" and continue, which would silently hollow out the run.
        print(f"[CPU {worker_id}] FATAL: {traceback.format_exc()}", flush=True)
        sys.exit(1)


def _pooled_gpu_worker(
    engine_id: int,
    gpu_ids_str: str,
    ready_queue,
    result_queue,
    config_path: str,
    args: dict,
) -> None:
    """Run one vLLM AsyncLLMEngine on a GPU group and generate for queued requests.

    A pooled-strategy GPU worker: it pins its GPU group (via ``CUDA_VISIBLE_DEVICES``
    before importing vLLM), builds a persistent ``AsyncLLMEngine``, and concurrently
    decodes requests pulled from ``ready_queue`` (up to ``concurrency`` in flight). For
    each item it issues one request per temperature group with ``SamplingParams(n=...)``
    (prompt prefilled once per group), and emits ONE result per item onto
    ``result_queue``: the list of that item's rollout records. A skipped item emits an
    empty list, so the parent's per-item count stays exact. Runs in a spawned subprocess.

    Args:
        engine_id: this engine's index, used for logging and cache-dir naming.
        gpu_ids_str: comma-joined GPU indices for this engine's tensor-parallel group.
        ready_queue: shared queue of built request dicts (or skip markers); a ``None``
            sentinel signals shutdown.
        result_queue: queue the parent drains; receives one list-of-records per item.
        config_path: path to the run config, reloaded to resolve the adapters.
        args: ``groups`` (rollout plan) + tuning/sampling knobs (``tp``, ``concurrency``,
            ``top_p``, ``max_new_tokens``).

    Returns:
        None. Side effect: puts per-item record lists on ``result_queue``.

    Raises:
        SystemExit: exits the subprocess with code 1 on a fatal engine error.
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Per-engine, user-scoped cache dirs. User-scoped: on shared boxes a dir owned by another
        # user under /tmp is not writable (PermissionError). Per-engine: the engines run concurrently
        # and torch.compile / inductor / triton caches are NOT safe to share across processes — a
        # shared dir races (partial writes -> "UnpicklingError: pickle data was truncated"), so each
        # engine gets its own VLLM / inductor / triton cache dir.
        cache_dir = (
            f"/tmp/vllm_cache_mtla_{os.environ.get('USER', 'u')}/engine_{engine_id}"
        )
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = cache_dir
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_dir}/inductor"
        os.environ["TRITON_CACHE_DIR"] = f"{cache_dir}/triton"
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        from vllm.utils import random_uuid

        cfg = load_config(config_path)
        model, dataset = resolve(cfg.model, cfg.dataset)

        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model.model_id,
                tensor_parallel_size=args["tp"],
                max_model_len=cfg.generate.max_model_len,
                gpu_memory_utilization=cfg.generate.gpu_mem_util,
                max_num_seqs=args["concurrency"],
                max_num_batched_tokens=32768,
                disable_log_stats=True,
                enforce_eager=False,
                disable_custom_all_reduce=True,
            )
        )
        print(
            f"[Engine {engine_id}] ready (GPUs {gpu_ids_str}, TP={args['tp']})",
            flush=True,
        )
        # One SamplingParams per temperature group (n = #rollouts in the group), so vLLM prefills
        # the prompt once per group and shares the KV cache across that group's samples. Force
        # FINAL_ONLY: the default streaming kind yields per-step deltas and the last yielded output
        # can carry fewer than n samples (they finish at different steps) — FINAL_ONLY yields one
        # complete output with all n samples, so no rollout is silently dropped.
        from vllm.sampling_params import RequestOutputKind

        samplings = []
        for g in args["groups"]:
            sp = _build_sampling(g, args, SamplingParams)
            sp.output_kind = RequestOutputKind.FINAL_ONLY
            samplings.append((sp, g["rollouts"]))

        async def run():
            local_q = asyncio.Queue(maxsize=args["concurrency"] * 2)
            active = set()
            stats = {"done": 0, "skip": 0}
            shutdown = False

            def bridge() -> None:
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
                # Emit exactly one queue put PER ITEM (a list of that item's rollout records), so the
                # parent still drains len(samples) puts. A generation error propagates (re-raised
                # where tasks are awaited below), crashing loudly rather than being masked; a skip
                # marker (adapter returned None, e.g. missing media) emits an empty list.
                try:
                    if task.get("skip"):
                        result_queue.put([])
                        stats["skip"] += 1
                        return
                    base = {
                        "prompt": task["prompt"],
                        "multi_modal_data": task["multi_modal_data"],
                    }
                    if task.get("mm_processor_kwargs"):
                        base["mm_processor_kwargs"] = task["mm_processor_kwargs"]
                    recs = []
                    for sp, rollouts in samplings:
                        final = None
                        async for r in engine.generate(base, sp, random_uuid()):
                            final = r
                        samples = final.outputs if final else []
                        # FINAL_ONLY guarantees all n samples; a short count is a real bug, not a
                        # per-item hiccup — fail loud rather than silently drop rollouts.
                        if len(samples) != len(rollouts):
                            raise RuntimeError(
                                f"expected {len(rollouts)} samples (n={sp.n}) but got "
                                f"{len(samples)} for idx {task['idx']}"
                            )
                        # One request -> len(rollouts) samples; sample k is rollout rollouts[k].
                        for rollout, sample in zip(rollouts, samples):
                            trunc = len(sample.token_ids) >= args["max_new_tokens"]
                            rec = dataset.gen_record(
                                cfg, task["item"], sample.text, truncated=trunc
                            )
                            # Store the EXACT prompt the model sent (task + its format suffix), so the
                            # extract stage teacher-forces the identical prompt.
                            rec["prompt"] = model.build_text_prompt(
                                dataset, task["item"]
                            )
                            rec["idx"] = task["idx"]
                            rec["rollout"] = rollout
                            recs.append(rec)
                    result_queue.put(recs)
                    stats["done"] += 1
                finally:
                    active.discard(asyncio.current_task())

            while not shutdown:
                while len(active) < args["concurrency"]:
                    try:
                        task = local_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if task is None:
                        shutdown = True
                        break
                    active.add(asyncio.create_task(infer(task)))
                if active:
                    done, active = await asyncio.wait(
                        active, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        t.result()  # re-raise a failed infer task (do not mask it)
                else:
                    await asyncio.sleep(0.05)
            if active:
                await asyncio.gather(*active)  # re-raise any straggler failure
            print(
                f"[Engine {engine_id}] done={stats['done']} skip={stats['skip']}",
                flush=True,
            )

        asyncio.run(run())
    except Exception:
        print(f"[Engine {engine_id}] FATAL: {traceback.format_exc()}", flush=True)
        sys.exit(1)


def run_pooled(cfg, samples: list, args: dict) -> list:
    """Run generation via an async multi-engine vLLM pool (throughput strategy).

    Spins up one ``AsyncLLMEngine`` per tensor-parallel GPU group plus a set of CPU
    prep workers, streams the samples through the raw queue, and drains one result PER
    ITEM from the result queue — each result being the list of that item's rollout records
    (one per rollout in the plan) — flattening them into the merged output. Best for many
    small requests (e.g. thousands of images).

    Args:
        cfg: the loaded run config (provides the generate-stage GPUs and config path).
        samples: the tagged work items to generate for (each carries an ``idx``).
        args: ``groups`` (rollout plan) + tuning/sampling knobs (``tp``, ``concurrency``,
            ``num_cpu_workers``, ``top_p``, ``max_new_tokens``).

    Returns:
        The merged list of ``dataset.gen_record`` records (one per item per rollout, each
        tagged with ``idx`` and ``rollout``), unordered; the caller splits by ``rollout``.

    Raises:
        AssertionError: if the number of GPUs is not divisible by the tensor-parallel
            size ``tp``.
    """
    gpu_ids = cfg.stage_gpus("generate")
    tp = args["tp"]
    assert len(gpu_ids) % tp == 0, f"#GPUs {len(gpu_ids)} not divisible by TP {tp}"
    groups = [gpu_ids[i : i + tp] for i in range(0, len(gpu_ids), tp)]
    num_engines = len(groups)
    print(
        f"[pooled] samples={len(samples)} engines={num_engines} TP={tp} GPUs={groups}",
        flush=True,
    )

    raw_q = mp.Queue()
    ready_qs = [mp.Queue(maxsize=500) for _ in range(num_engines)]
    result_q = mp.Queue()
    gpu_procs = [
        mp.Process(
            target=_pooled_gpu_worker,
            args=(
                i,
                ",".join(map(str, g)),
                ready_qs[i],
                result_q,
                cfg.config_path,
                args,
            ),
        )
        for i, g in enumerate(groups)
    ]
    for p in gpu_procs:
        p.start()
    cpu_procs = [
        mp.Process(
            target=_pooled_cpu_worker,
            args=(raw_q, ready_qs, cfg.config_path, num_engines, i),
        )
        for i in range(args["num_cpu_workers"])
    ]
    for p in cpu_procs:
        p.start()
    for idx, item in enumerate(samples):
        raw_q.put((idx, item))
    for _ in range(args["num_cpu_workers"]):
        raw_q.put(None)

    results = []
    n_items = (
        0  # results are drained one PER ITEM (a list of that item's rollout records)
    )
    start = time.time()
    last = start
    while n_items < len(samples):
        try:
            results.extend(result_q.get(timeout=300))
            n_items += 1
            now = time.time()
            if now - last > 10:
                spd = n_items / (now - start)
                print(
                    f"  [{n_items}/{len(samples)}] {spd:.1f} items/s "
                    f"ETA {(len(samples)-n_items)/max(spd,1e-9):.0f}s",
                    flush=True,
                )
                last = now
        except queue.Empty:
            if not any(p.is_alive() for p in gpu_procs):
                print("[pooled] all GPU workers died!", flush=True)
                break
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
# Strategy B — one blocking engine per GPU (heavy per-item work)
# ===========================================================================
def _build_sampling(group: dict, args: dict, SamplingParams):
    """Build the ``SamplingParams`` for one temperature group's single ``n=K`` request.

    The group's ``rollouts`` all share one request: ``n`` is their count, so vLLM prefills
    the prompt once and shares the KV cache across the K samples. ``seed`` is the group's
    RNG seed (the greedy and stochastic groups carry different seeds), so a rerun
    reproduces the same K samples.

    Args:
        group: a rollout-plan group ``{"rollouts": [...], "temperature": float, "seed": int}``.
        args: sampling knobs (``top_p``, ``max_new_tokens``).
        SamplingParams: the vLLM class (imported inside the worker).

    Returns:
        A ``SamplingParams`` with ``n=len(group["rollouts"])`` at the group's temperature.
    """
    n = len(group["rollouts"])
    temperature = group["temperature"]
    if temperature and temperature > 0:
        return SamplingParams(
            n=n,
            temperature=temperature,
            top_p=args["top_p"],
            max_tokens=args["max_new_tokens"],
            seed=group["seed"],
        )
    # Greedy (T=0): deterministic, so n is 1 in practice; keep n for uniformity.
    return SamplingParams(
        n=n, temperature=0.0, max_tokens=args["max_new_tokens"], seed=group["seed"]
    )


def _sharded_worker(
    rank: int, gpu_id: int, items: list, out_dir: str, config_path: str, args: dict
) -> None:
    """Generate every rollout for one chunk of items on a single GPU with a blocking engine.

    A sharded-strategy worker: it pins its GPU (via ``CUDA_VISIBLE_DEVICES`` before
    importing vLLM), builds one blocking ``LLM`` engine, and loops its chunk one item at
    a time. For each item it issues one request per temperature group (``args["groups"]``)
    with ``SamplingParams(n=<#rollouts in group>)``, so the prompt is prefilled once per
    group and the KV cache is shared across that group's samples. Each of the K outputs is
    written as its own ``gen_record`` tagged with its rollout number. Runs in a spawned
    subprocess.

    Args:
        rank: this worker's shard index; used to name the output shard file.
        gpu_id: the absolute GPU index this worker pins to (seen as cuda:0 inside).
        items: this worker's slice of work items.
        out_dir: directory to write ``preds_rank{rank}.json`` into (created if missing).
        config_path: path to the run config, reloaded to resolve the adapters.
        args: ``groups`` (the rollout plan) plus sampling knobs (``top_p``,
            ``max_new_tokens``).

    Returns:
        None. Side effect: writes ``<out_dir>/preds_rank{rank}.json`` with this shard's
        generation records, each tagged with ``idx`` and ``rollout``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        gpu_id
    )  # BEFORE torch import (vLLM grabs device 0)
    from vllm import LLM, SamplingParams

    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    groups = args["groups"]
    print(
        f"[worker {rank}] sharded model={cfg.model} dataset={cfg.dataset} gpu={gpu_id} "
        f"n={len(items)} groups={groups}",
        flush=True,
    )

    proc = model.gen_processor()
    llm = LLM(
        model=model.model_id,
        gpu_memory_utilization=cfg.generate.gpu_mem_util,
        enforce_eager=False,
        disable_log_stats=True,
        max_model_len=cfg.generate.max_model_len,
    )
    samplings = [
        (_build_sampling(g, args, SamplingParams), g["rollouts"]) for g in groups
    ]

    results = []
    for cnt, item in enumerate(items):
        req = model.build_vllm_request(proc, item, dataset, cfg)
        if req is None:  # adapter chose to skip this item (e.g. missing clip)
            continue
        gen_req = {
            "prompt": req["prompt"],
            "multi_modal_data": req["multi_modal_data"],
        }
        if req.get("mm_processor_kwargs"):
            gen_req["mm_processor_kwargs"] = req["mm_processor_kwargs"]
        for sp, rollouts in samplings:
            out = llm.generate([gen_req], sp, use_tqdm=False)
            # One request -> len(rollouts) samples; sample k is rollout rollouts[k]. The blocking
            # LLM.generate returns completed outputs (no streaming), so all n should be present; a
            # short count is a real bug, not a per-item hiccup — fail loud rather than drop rollouts.
            samples = out[0].outputs if out else []
            if len(samples) != len(rollouts):
                raise RuntimeError(
                    f"expected {len(rollouts)} samples (n={sp.n}) but got {len(samples)} "
                    f"for idx {item.get('idx', cnt)}"
                )
            for rollout, sample in zip(rollouts, samples):
                response = sample.text.strip()
                if not response:
                    continue
                rec = dataset.gen_record(cfg, item, response)
                # Store the EXACT prompt the model sent (task + its format suffix), so the extract
                # stage teacher-forces the identical prompt.
                rec["prompt"] = model.build_text_prompt(dataset, item)
                rec["idx"] = item.get("idx", cnt)
                rec["rollout"] = rollout
                results.append(rec)
        if (cnt + 1) % 25 == 0:
            print(
                f"[worker {rank}] [{cnt + 1}/{len(items)}] records={len(results)}",
                flush=True,
            )

    # A few skipped items (missing clip, empty response) are tolerated, but a shard that produced
    # NOTHING from a non-trivial number of items is systemic (wrong data dir, every clip missing, a
    # broken path scheme) — fail loud rather than write an empty shard silently.
    if not results and len(items) > 1:
        raise RuntimeError(
            f"[worker {rank}] generated 0 of {len(items)} items; this is a systemic failure "
            f"(check the media paths / data dir), not a few missing examples."
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/preds_rank{rank}.json", "w") as f:
        json.dump(results, f)
    print(
        f"[worker {rank}] saved {len(results)} records -> {out_dir}/preds_rank{rank}.json",
        flush=True,
    )


def run_sharded(cfg, samples: list, args: dict, out_dir: str) -> list:
    """Run generation via one blocking vLLM engine per GPU (heavy-per-item strategy).

    Splits the samples evenly across the generate-stage GPUs, spawns one
    ``_sharded_worker`` per GPU (each producing every rollout for its chunk), joins them,
    then merges their per-rank shard files into one flat, rollout-tagged list. Best for
    heavy per-item work such as video clips. Fails loudly if any worker crashes, since a
    missing shard would silently produce an incomplete run.

    Args:
        cfg: the loaded run config (provides the generate-stage GPUs and config path).
        samples: the tagged work items to generate for.
        args: ``groups`` (rollout plan) + sampling knobs (see ``_sharded_worker``).
        out_dir: base directory; per-rank shards go in an ``_sharded_staging`` subdir that
            is removed after the merge (the caller writes the per-rollout files).

    Returns:
        The merged list of generation records (each tagged with ``idx`` and ``rollout``),
        read back from the per-rank shard files.

    Raises:
        SystemExit: if one or more workers exit with a non-zero code, listing the
            failed ``(rank, exitcode)`` pairs; the shards on disk are incomplete.
    """
    gpu_ids = cfg.stage_gpus("generate")
    staging = os.path.join(out_dir, "_sharded_staging")
    os.makedirs(staging, exist_ok=True)
    print(f"[sharded] samples={len(samples)} gpus={gpu_ids}", flush=True)
    procs = []
    for rank, (gpu, chunk) in enumerate(
        zip(gpu_ids, np.array_split(samples, len(gpu_ids)))
    ):
        if len(chunk) == 0:
            continue
        p = mp.Process(
            target=_sharded_worker,
            args=(rank, gpu, list(chunk), staging, cfg.config_path, args),
        )
        p.start()
        procs.append((rank, p))
    failed = [(rank, p.exitcode) for rank, p in procs if (p.join() or p.exitcode != 0)]
    if failed:
        raise SystemExit(
            f"[sharded] {len(failed)} worker(s) failed (rank, exitcode): {failed}; "
            f"shards under {staging} are INCOMPLETE — fix the error and re-run."
        )
    merged = []
    for rank in range(len(gpu_ids)):
        pp = f"{staging}/preds_rank{rank}.json"
        if os.path.exists(pp):
            merged.extend(json.load(open(pp)))
    shutil.rmtree(staging, ignore_errors=True)
    return merged
