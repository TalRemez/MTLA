"""vLLM multiclass inference for Qwen3-VL — supports TP>1 for large models.

For 8B: 8 engines, TP=1 (one per GPU)
For 32B: 4 engines, TP=2 (each uses 2 GPUs)
"""
import json, re, argparse, time, sys, os, traceback
import multiprocessing as mp
import numpy as np
import queue
import threading
import asyncio
from pathlib import Path
from PIL import Image

mp.set_start_method("spawn", force=True)


def parse_bboxes(response, label_side="right"):
    """Parse `{"bbox_2d": [...], "label": "..."}` predictions from a Qwen3-VL
    response into [{"box", "label", "score"}].

    Qwen emits the label AFTER the box, so `label_side="right"`: the regex
    fallback (used when the response is not valid JSON, e.g. truncated/cascade
    outputs) attributes each box to the FIRST label that follows it, and the
    search is bounded by the next `"bbox_2d"` so it can never bleed into a
    neighbouring object's label. (Set `label_side="left"` for formats where the
    label precedes the box, e.g. InternVL's `<ref>label</ref><box>...</box>`.)
    """
    cleaned = re.sub(r'```json\s*|```\s*', '', response).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [{"box": o["bbox_2d"], "label": o.get("label", "").lower(), "score": 1.0}
                    for o in parsed if "bbox_2d" in o and isinstance(o["bbox_2d"], list) and len(o["bbox_2d"]) == 4]
        if isinstance(parsed, dict):
            results = []
            for cat, items in parsed.items():
                if not isinstance(items, list): continue
                for item in items:
                    if isinstance(item, dict) and "bbox_2d" in item:
                        box = item["bbox_2d"]
                        if isinstance(box, list) and len(box) == 4:
                            results.append({"box": box, "label": item.get("label", cat).lower(), "score": 1.0})
                    elif isinstance(item, list) and len(item) == 4 and all(isinstance(x, (int, float)) for x in item):
                        results.append({"box": [int(x) for x in item], "label": cat.lower(), "score": 1.0})
            if results:
                return results
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    # Regex fallback for bbox_2d format. The label for each box is taken from
    # the window between this box and the adjacent box (the next one for
    # label_side="right", the previous one for "left"), so the search is never
    # contaminated by a neighbouring object's label.
    box_re = re.compile(r'"bbox_2d"\s*:\s*\[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]')
    label_re = re.compile(r'"label"\s*:\s*"([^"]+)"')
    matches = list(box_re.finditer(response))
    results = []
    for k, m in enumerate(matches):
        if label_side == "right":
            # search forward, up to the next box (or end of string)
            lo = m.end()
            hi = matches[k + 1].start() if k + 1 < len(matches) else len(response)
            window = response[lo:hi]
            lm = label_re.search(window)
        else:  # "left": search backward, up to the previous box (or start)
            hi = m.start()
            lo = matches[k - 1].end() if k > 0 else 0
            window = response[lo:hi]
            lms = list(label_re.finditer(window))
            lm = lms[-1] if lms else None
        results.append({"box": [int(m.group(i)) for i in range(1, 5)],
                        "label": lm.group(1).lower() if lm else "",
                        "score": 1.0})
    if results:
        return results
    # Regex fallback for raw array format: "cat": [[x1,y1,x2,y2], ...]
    for m in re.finditer(r'"([^"]+)"\s*:\s*\[\s*\[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]', response):
        results.append({"box": [int(m.group(i)) for i in range(2, 6)], "label": m.group(1).lower(), "score": 1.0})
    return results


def cpu_worker(raw_queue, ready_queues, model_path, worker_id, num_engines):
    try:
        import torch
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(model_path)

        while True:
            task = raw_queue.get()
            if task is None:
                break
            idx, item = task
            try:
                image = Image.open(item["image"]).convert("RGB")
                prompt_text = item["conversations"][0]["value"].replace("<image>\n", "")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

                ready_queues[idx % num_engines].put({
                    "idx": idx, "item": item,
                    "prompt": text,
                    "multi_modal_data": {"image": image_inputs} if image_inputs else {},
                })
            except Exception:
                print(f"[CPU {worker_id}] Error idx {idx}: {traceback.format_exc()}", flush=True)
                ready_queues[idx % num_engines].put({"idx": idx, "item": item, "error": True})
    except Exception:
        print(f"[CPU {worker_id}] FATAL: {traceback.format_exc()}", flush=True)


def gpu_worker(engine_id, gpu_ids_str, ready_queue, result_queue, args):
    """Each engine gets a set of GPUs (e.g. '0,1' for TP=2)."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cache_dir = f"/tmp/vllm_cache/engine_{engine_id}"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = cache_dir

        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        from vllm.utils import random_uuid

        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=args.model,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            max_num_seqs=args.concurrency,
            max_num_batched_tokens=32768,
            disable_log_stats=True,
            enforce_eager=False,
            limit_mm_per_prompt={"image": 1},
            disable_custom_all_reduce=True,
        ))
        print(f"[Engine {engine_id}] Ready (GPUs: {gpu_ids_str}, TP={args.tp})", flush=True)

        sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)

        async def run():
            local_q = asyncio.Queue(maxsize=args.concurrency * 2)
            active = set()
            stats = {"done": 0, "errors": 0}
            shutdown = False

            def bridge():
                while True:
                    try:
                        item = ready_queue.get(timeout=5)
                        if item is None:
                            asyncio.run_coroutine_threadsafe(local_q.put(None), loop).result()
                            return
                        asyncio.run_coroutine_threadsafe(local_q.put(item), loop).result()
                    except queue.Empty:
                        continue

            loop = asyncio.get_event_loop()
            threading.Thread(target=bridge, daemon=True).start()

            async def infer(task):
                try:
                    if task.get("error"):
                        result_queue.put(_error_result(task["idx"], task["item"]))
                        stats["errors"] += 1
                        return
                    final = None
                    async for r in engine.generate(
                        {"prompt": task["prompt"], "multi_modal_data": task["multi_modal_data"]},
                        sampling, random_uuid(),
                    ):
                        final = r

                    response = final.outputs[0].text if final else ""
                    truncated = len(final.outputs[0].token_ids) >= args.max_new_tokens if final else False

                    item = task["item"]
                    result_queue.put({
                        "idx": task["idx"], "status": "success",
                        "id": item["id"], "categories": item["categories"],
                        "gt_response": item["conversations"][1]["value"],
                        "pred_bboxes": parse_bboxes(response),
                        "response": response, "truncated": truncated,
                    })
                    stats["done"] += 1
                except Exception:
                    print(f"[Engine {engine_id}] Error: {traceback.format_exc()}", flush=True)
                    result_queue.put(_error_result(task["idx"], task["item"]))
                    stats["errors"] += 1
                finally:
                    active.discard(asyncio.current_task())

            while not shutdown:
                while len(active) < args.concurrency:
                    try:
                        task = local_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if task is None:
                        shutdown = True
                        break
                    active.add(asyncio.create_task(infer(task)))

                if active:
                    done_tasks, active = await asyncio.wait(active, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                else:
                    await asyncio.sleep(0.05)

            if active:
                await asyncio.gather(*active, return_exceptions=True)
            print(f"[Engine {engine_id}] Done: {stats['done']}, Errors: {stats['errors']}", flush=True)

        asyncio.run(run())
    except Exception:
        print(f"[Engine {engine_id}] FATAL: {traceback.format_exc()}", flush=True)
        sys.exit(1)


def _error_result(idx, item):
    return {
        "idx": idx, "status": "error",
        "id": item.get("id"), "categories": item.get("categories", []),
        "gt_response": item["conversations"][1]["value"],
        "pred_bboxes": [], "response": "", "truncated": False,
    }



def main():
    from mtla.config import load_config
    from mtla.registry import resolve
    parser = argparse.ArgumentParser(description="Qwen3-VL COCO detection generation (vLLM).")
    parser.add_argument("--config", required=True, help="path to a configs/*.yaml")
    parser.add_argument("--seed", type=int, default=0, help="rollout seed (selects seed{K}/ out dir)")
    # vLLM performance knobs (not part of the run config; tune per box)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size per engine")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_mem_util", type=float, default=0.92)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num_cpu_workers", type=int, default=16)
    args = parser.parse_args()

    # Config drives model/dataset/gpus/seed/temperature; the dataset adapter owns item loading.
    cfg = load_config(args.config)
    model, dataset = resolve(cfg.model, cfg.dataset)
    args.model = model.model_id
    args.gpu_ids = cfg.generate.gpus
    args.temperature = cfg.generate.temperature
    args.output_dir = cfg.pred_dir(args.seed)
    args.limit = cfg.generate.n_items or 99999

    samples = dataset.load_items(cfg)[:args.limit]

    gpu_ids = args.gpu_ids or list(range(8))
    # Split GPUs into engine groups based on TP
    assert len(gpu_ids) % args.tp == 0, f"Number of GPUs ({len(gpu_ids)}) must be divisible by TP ({args.tp})"
    engine_gpu_groups = [gpu_ids[i:i+args.tp] for i in range(0, len(gpu_ids), args.tp)]
    num_engines = len(engine_gpu_groups)

    print(f"Samples: {len(samples)}, Engines: {num_engines}, TP: {args.tp}, "
          f"GPU groups: {engine_gpu_groups}, CPU workers: {args.num_cpu_workers}")

    raw_queue = mp.Queue()
    ready_queues = [mp.Queue(maxsize=500) for _ in range(num_engines)]
    result_queue = mp.Queue()

    # Start engine workers
    gpu_procs = []
    for i, gpu_group in enumerate(engine_gpu_groups):
        gpu_ids_str = ",".join(str(g) for g in gpu_group)
        p = mp.Process(target=gpu_worker, args=(i, gpu_ids_str, ready_queues[i], result_queue, args))
        p.start()
        gpu_procs.append(p)

    # Start CPU workers
    cpu_procs = []
    for i in range(args.num_cpu_workers):
        p = mp.Process(target=cpu_worker, args=(raw_queue, ready_queues, args.model, i, num_engines))
        p.start()
        cpu_procs.append(p)

    # Feed samples + poison pills
    for idx, item in enumerate(samples):
        raw_queue.put((idx, item))
    for _ in range(args.num_cpu_workers):
        raw_queue.put(None)

    # Collect results
    results = []
    start = time.time()
    last_print = start
    while len(results) < len(samples):
        try:
            r = result_queue.get(timeout=300)
            results.append(r)
            now = time.time()
            if now - last_print > 10:
                elapsed = now - start
                speed = len(results) / elapsed
                eta = (len(samples) - len(results)) / speed if speed > 0 else 0
                print(f"  [{len(results)}/{len(samples)}] {speed:.1f} samples/s, ETA {eta:.0f}s", flush=True)
                last_print = now
        except queue.Empty:
            if not any(p.is_alive() for p in gpu_procs):
                print("All GPU workers died!", flush=True)
                break

    elapsed = time.time() - start
    print(f"\nInference: {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample, {len(samples)/elapsed:.1f} samples/s)")

    # Shutdown
    for p in cpu_procs:
        p.join(timeout=30)
    for q in ready_queues:
        q.put(None)
    for p in gpu_procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    # Save & evaluate
    results.sort(key=lambda x: x["idx"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
