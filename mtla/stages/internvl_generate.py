"""vLLM inference for InternVL3.5-8B on COCO Det.

Format: native InternVL grounding -- output is `class[[x1,y1,x2,y2],...]class[[...]]...`
with coords in [0, 1000]. We use the "v1_official_listing" prompt that gave the
cleanest output in our format probe.
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


# v1 official-listing prompt (replaces the dataset's existing user_value)
PROMPT_TMPL = (
    "Please detect all instances of {cats} in the image. "
    "Output the bounding boxes in the format <ref>category</ref><box>[[x1, y1, x2, y2], ...]</box> "
    "with coordinates normalized to [0, 1000]."
)


def parse_internvl_bboxes(response):
    """Parse InternVL native grounding output.
    Format A: <ref>label</ref><box>[[x1,y1,x2,y2], [...]]</box>
    Format B: label[[x1,y1,x2,y2], [...]] (no tags, contiguous)
    """
    pairs = []
    # Format A first
    for m in re.finditer(r'<ref>([^<]+)</ref><box>\s*\[(.+?)\]\s*</box>', response, flags=re.DOTALL):
        label = m.group(1).strip().lower()
        for b in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', m.group(2)):
            pairs.append({"box": [int(b.group(i)) for i in range(1, 5)], "label": label, "score": 1.0})
    if pairs:
        return pairs
    # Format B: <label>[[x1,y1,x2,y2], [...]]
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', response)
    pat = re.compile(r'([A-Za-z][A-Za-z _]*?)\s*(\[\[)')
    pos = 0
    while pos < len(cleaned):
        m = pat.search(cleaned, pos)
        if not m: break
        label = m.group(1).strip().lower()
        outer_open = m.start(2)  # index of outer "[" (start of "[[")
        depth = 0; outer_close = -1
        for i in range(outer_open, len(cleaned)):
            if cleaned[i] == '[': depth += 1
            elif cleaned[i] == ']':
                depth -= 1
                if depth == 0:
                    outer_close = i; break
        if outer_close == -1: break
        chunk = cleaned[outer_open:outer_close + 1]
        for b in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', chunk):
            pairs.append({"box": [int(b.group(i)) for i in range(1, 5)], "label": label, "score": 1.0})
        pos = outer_close + 1
    return pairs


# ---------- InternVL image preprocessing (matches their official load_image) ----------
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size):
    best = float('inf'); best_ar = (1, 1)
    area = w * h
    for r in target_ratios:
        rar = r[0] / r[1]
        diff = abs(aspect_ratio - rar)
        if diff < best:
            best = diff; best_ar = r
        elif diff == best:
            if area > 0.5 * image_size * image_size * r[0] * r[1]:
                best_ar = r
    return best_ar


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    ow, oh = image.size; ar = ow / oh
    target_ratios = sorted({(i, j) for n in range(min_num, max_num+1) for i in range(1, n+1) for j in range(1, n+1)
                            if min_num <= i*j <= max_num},
                           key=lambda x: x[0]*x[1])
    target = _find_closest_aspect_ratio(ar, target_ratios, ow, oh, image_size)
    tw, th = image_size*target[0], image_size*target[1]
    blocks = target[0]*target[1]
    img = image.resize((tw, th))
    images = []
    for i in range(blocks):
        box = ((i % target[0])*image_size, (i // target[0])*image_size,
               ((i % target[0])+1)*image_size, ((i // target[0])+1)*image_size)
        images.append(img.crop(box))
    if use_thumbnail and len(images) != 1:
        images.append(image.resize((image_size, image_size)))
    return images


def load_image_internvl(path, input_size=448, max_num=12):
    image = Image.open(path).convert('RGB')
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values, len(images)


def cpu_worker(raw_queue, ready_queues, model_path, worker_id, num_engines):
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        import torch as t_
        t_.set_num_threads(1)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        IMG_CONTEXT = "<IMG_CONTEXT>"
        IMG_START = "<img>"
        IMG_END = "</img>"
        # InternVL replaces <image> with `<img><IMG_CONTEXT>×n_patches</img>`
        # vLLM v0.19's InternVL handler expects `<image>` placeholder in the prompt.

        while True:
            task = raw_queue.get()
            if task is None: break
            idx, item = task
            try:
                pixel_values, num_patches = load_image_internvl(item["image"])
                cats = ", ".join(item["categories"])
                user_text = "<image>\n" + PROMPT_TMPL.format(cats=cats)
                messages = [{"role": "user", "content": user_text}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Convert pixel_values back to PIL by un-normalizing? No — vLLM accepts PIL Image.
                # Easier: pass the original PIL image via multi_modal_data; vLLM will handle InternVL preproc.
                image = Image.open(item["image"]).convert("RGB")
                ready_queues[idx % num_engines].put({
                    "idx": idx, "item": item,
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                })
            except Exception:
                print(f"[CPU {worker_id}] Error idx {idx}: {traceback.format_exc()}", flush=True)
                ready_queues[idx % num_engines].put({"idx": idx, "item": item, "error": True})
    except Exception:
        print(f"[CPU {worker_id}] FATAL: {traceback.format_exc()}", flush=True)


def gpu_worker(engine_id, gpu_ids_str, ready_queue, result_queue, args):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cache_dir = f"/tmp/vllm_cache_internvl/engine_{engine_id}"
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
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
            disable_custom_all_reduce=True,
        ))
        print(f"[Engine {engine_id}] Ready (GPUs: {gpu_ids_str}, TP={args.tp})", flush=True)

        sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens, seed=args.seed)

        async def run():
            local_q = asyncio.Queue(maxsize=args.concurrency * 2)
            active = set(); stats = {"done": 0, "errors": 0}; shutdown = False

            def bridge():
                while True:
                    try:
                        item = ready_queue.get(timeout=5)
                        if item is None:
                            asyncio.run_coroutine_threadsafe(local_q.put(None), loop).result()
                            return
                        asyncio.run_coroutine_threadsafe(local_q.put(item), loop).result()
                    except queue.Empty: continue

            loop = asyncio.get_event_loop()
            threading.Thread(target=bridge, daemon=True).start()

            async def infer(task):
                try:
                    if task.get("error"):
                        result_queue.put(_error_result(task["idx"], task["item"]))
                        stats["errors"] += 1; return
                    final = None
                    async for r in engine.generate(
                        {"prompt": task["prompt"], "multi_modal_data": task["multi_modal_data"]},
                        sampling, random_uuid(),
                    ): final = r
                    response = final.outputs[0].text if final else ""
                    truncated = len(final.outputs[0].token_ids) >= args.max_new_tokens if final else False
                    item = task["item"]
                    result_queue.put({
                        "idx": task["idx"], "status": "success",
                        "id": item["id"], "categories": item["categories"],
                        "gt_response": item["conversations"][1]["value"],
                        "pred_bboxes": parse_internvl_bboxes(response),
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
                    try: task = local_q.get_nowait()
                    except asyncio.QueueEmpty: break
                    if task is None: shutdown = True; break
                    active.add(asyncio.create_task(infer(task)))
                if active:
                    done_tasks, active = await asyncio.wait(active, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                else:
                    await asyncio.sleep(0.05)
            if active: await asyncio.gather(*active, return_exceptions=True)
            print(f"[Engine {engine_id}] Done: {stats['done']}, Errors: {stats['errors']}", flush=True)

        asyncio.run(run())
    except Exception:
        print(f"[Engine {engine_id}] FATAL: {traceback.format_exc()}", flush=True)
        sys.exit(1)


def _error_result(idx, item):
    return {"idx": idx, "status": "error",
            "id": item.get("id"), "categories": item.get("categories", []),
            "gt_response": item["conversations"][1]["value"],
            "pred_bboxes": [], "response": "", "truncated": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--limit", type=int, default=99999)
    parser.add_argument("--output_dir", default="/tmp/internvl_out")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_mem_util", type=float, default=0.92)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num_cpu_workers", type=int, default=16)
    parser.add_argument("--dataset", required=True, help="COCO openvocab dataset json (see docs/DATA.md)")
    args = parser.parse_args()

    data = json.load(open(args.dataset))
    samples = data[:args.limit]
    gpu_ids = args.gpu_ids or list(range(8))
    assert len(gpu_ids) % args.tp == 0
    engine_gpu_groups = [gpu_ids[i:i+args.tp] for i in range(0, len(gpu_ids), args.tp)]
    num_engines = len(engine_gpu_groups)
    print(f"Samples: {len(samples)}, Engines: {num_engines}, TP: {args.tp}, "
          f"GPU groups: {engine_gpu_groups}, CPU workers: {args.num_cpu_workers}")

    raw_queue = mp.Queue()
    ready_queues = [mp.Queue(maxsize=500) for _ in range(num_engines)]
    result_queue = mp.Queue()
    gpu_procs = []
    for i, gpu_group in enumerate(engine_gpu_groups):
        gpu_ids_str = ",".join(str(g) for g in gpu_group)
        p = mp.Process(target=gpu_worker, args=(i, gpu_ids_str, ready_queues[i], result_queue, args))
        p.start(); gpu_procs.append(p)
    cpu_procs = []
    for i in range(args.num_cpu_workers):
        p = mp.Process(target=cpu_worker, args=(raw_queue, ready_queues, args.model, i, num_engines))
        p.start(); cpu_procs.append(p)
    for idx, item in enumerate(samples): raw_queue.put((idx, item))
    for _ in range(args.num_cpu_workers): raw_queue.put(None)

    results = []
    start = time.time(); last_print = start
    while len(results) < len(samples):
        try:
            r = result_queue.get(timeout=600)
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
                print("All GPU workers died!", flush=True); break

    elapsed = time.time() - start
    print(f"\nInference: {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample, {len(samples)/elapsed:.1f}/s)")

    for p in cpu_procs: p.join(timeout=30)
    for q in ready_queues: q.put(None)
    for p in gpu_procs:
        p.join(timeout=60)
        if p.is_alive(): p.terminate()

    results.sort(key=lambda x: x["idx"])
    out_dir = Path(args.output_dir) / "temp_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
