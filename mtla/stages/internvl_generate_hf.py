"""HuggingFace generation for InternVL3.5-8B on COCO detection.

Drop-in alternative to internvl_generate.py (vLLM): same prompt, same parse, and the SAME
predictions.json record schema, so the extract/score stages consume it identically. Use this
when you don't want a vLLM install, or want a single backend across machines. It is slower
than vLLM (no batched paged attention) but needs only `transformers` + `timm`.

Output per image (list written to <output_dir>/predictions.json):
  {idx, status, id, categories, gt_response, pred_bboxes:[{box,label,score}], response, truncated}
"""
import argparse
import json
import os
import re
import time
from multiprocessing import Process, set_start_method

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# transformers v5 compat: InternVL's modeling code expects an attribute v5 doesn't define.
from transformers import AutoModel, AutoTokenizer, modeling_utils
if not hasattr(modeling_utils.PreTrainedModel, '_compat_done'):
    _orig = modeling_utils.PreTrainedModel.__getattr__
    def _patched(self, name):
        if name == 'all_tied_weights_keys':
            try: return _orig(self, name)
            except AttributeError: return {}
        return _orig(self, name)
    modeling_utils.PreTrainedModel.__getattr__ = _patched
    modeling_utils.PreTrainedModel._compat_done = True

MODEL_ID = "OpenGVLab/InternVL3_5-8B"
IMG_CONTEXT_TOK = "<IMG_CONTEXT>"
PATCH_GRID = 16
NUM_IMAGE_TOKEN_PER_TILE = PATCH_GRID * PATCH_GRID  # 256

# Same prompt as the vLLM generator (native InternVL grounding, coords in [0,1000]).
PROMPT_TMPL = (
    "Please detect all instances of {cats} in the image. "
    "Output the bounding boxes in the format <ref>category</ref><box>[[x1, y1, x2, y2], ...]</box> "
    "with coordinates normalized to [0, 1000]."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size):
    best = float('inf'); best_ar = (1, 1); area = w * h
    for r in target_ratios:
        rar = r[0] / r[1]
        diff = abs(aspect_ratio - rar)
        if diff < best:
            best = diff; best_ar = r
        elif diff == best and area > 0.5 * image_size * image_size * r[0] * r[1]:
            best_ar = r
    return best_ar


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    ow, oh = image.size; ar = ow / oh
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                            for i in range(1, n + 1) for j in range(1, n + 1)
                            if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    target = _find_closest_aspect_ratio(ar, target_ratios, ow, oh, image_size)
    n_cols, n_rows = target
    tw, th = image_size * n_cols, image_size * n_rows
    img = image.resize((tw, th))
    tiles = []
    for i in range(n_cols * n_rows):
        c, r = i % n_cols, i // n_cols
        tiles.append(img.crop((c * image_size, r * image_size,
                               (c + 1) * image_size, (r + 1) * image_size)))
    has_thumb = use_thumbnail and len(tiles) != 1
    if has_thumb:
        tiles.append(image.resize((image_size, image_size)))
    return tiles, (n_cols, n_rows), has_thumb


def load_image_internvl(path, input_size=448, max_num=12):
    image = Image.open(path).convert('RGB')
    transform = _build_transform(input_size=input_size)
    tiles, tile_grid, has_thumb = _dynamic_preprocess(image, image_size=input_size,
                                                      use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values, tile_grid, has_thumb


def parse_internvl_bboxes(response):
    """Identical to the vLLM generator's parser (native InternVL grounding)."""
    pairs = []
    for m in re.finditer(r'<ref>([^<]+)</ref><box>\s*\[(.+?)\]\s*</box>', response, flags=re.DOTALL):
        label = m.group(1).strip().lower()
        for b in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', m.group(2)):
            pairs.append({"box": [int(b.group(i)) for i in range(1, 5)], "label": label, "score": 1.0})
    if pairs:
        return pairs
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', response)
    pat = re.compile(r'([A-Za-z][A-Za-z _]*?)\s*(\[\[)')
    pos = 0
    while pos < len(cleaned):
        m = pat.search(cleaned, pos)
        if not m: break
        label = m.group(1).strip().lower()
        outer_open = m.start(2)
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


def worker(rank, gpu_id, items, out_dir, temperature, seed, max_new_tokens):
    print(f"[worker {rank}] gpu={gpu_id} n={len(items)} T={temperature} seed={seed}", flush=True)
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                      trust_remote_code=True).to(device).eval()
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOK)
    do_sample = temperature is not None and temperature > 0
    if do_sample:
        from transformers import set_seed as _hf_set_seed
        _hf_set_seed(seed * 1000 + rank)

    results = []
    t0 = time.time()
    for cnt, item in enumerate(items):
        rec = {"idx": item["idx"], "status": "error", "id": item.get("id"),
               "categories": item.get("categories", []),
               "gt_response": item["conversations"][1]["value"] if item.get("conversations") else "[]",
               "pred_bboxes": [], "response": "", "truncated": False}
        try:
            pixel_values, tile_grid, has_thumb = load_image_internvl(item["image"])
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            n_tiles = tile_grid[0] * tile_grid[1]
            n_image_tokens = n_tiles * NUM_IMAGE_TOKEN_PER_TILE + (NUM_IMAGE_TOKEN_PER_TILE if has_thumb else 0)
            cats = ", ".join(item["categories"])
            user_text = "<image>\n" + PROMPT_TMPL.format(cats=cats)
            chat = tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                                 tokenize=False, add_generation_prompt=True)
            image_token_block = "<img>" + (IMG_CONTEXT_TOK * n_image_tokens) + "</img>"
            prompt_text = chat.replace("<image>", image_token_block, 1)
            input_ids = tokenizer(prompt_text, return_tensors="pt",
                                  add_special_tokens=False).input_ids.to(device)
            attn_mask = torch.ones_like(input_ids)
            gen_kwargs = dict(max_new_tokens=max_new_tokens)
            if do_sample:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                gen_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids,
                                         attention_mask=attn_mask, **gen_kwargs)
            response = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            rec.update(status="success", pred_bboxes=parse_internvl_bboxes(response),
                       response=response,
                       truncated=bool(gen_ids.shape[1] >= max_new_tokens))
        except Exception as e:
            print(f"[worker {rank}] error {item.get('id')}: {e}", flush=True)
        results.append(rec)
        if (cnt + 1) % 5 == 0:
            rate = (cnt + 1) / max(time.time() - t0, 1e-9)
            print(f"[worker {rank}] [{cnt+1}/{len(items)}] {rate:.2f} img/s", flush=True)
        torch.cuda.empty_cache()

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/preds_rank{rank}.json", "w") as f:
        json.dump(results, f)
    print(f"[worker {rank}] saved {len(results)} -> {out_dir}/preds_rank{rank}.json", flush=True)


def main():
    global MODEL_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--dataset", required=True, help="COCO openvocab dataset json")
    ap.add_argument("--output_dir", default="predictions")
    ap.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=99999)
    args = ap.parse_args()
    MODEL_ID = args.model
    set_start_method("spawn", force=True)

    data = json.load(open(args.dataset))[:args.limit]
    for i, d in enumerate(data):
        d.setdefault("idx", i)
    chunks = np.array_split(data, len(args.gpu_ids))
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(args.gpu_ids, chunks)):
        if len(chunk) == 0:
            continue
        p = Process(target=worker, args=(rank, gpu, list(chunk), args.output_dir,
                                         args.temperature, args.seed, args.max_new_tokens))
        p.start(); procs.append(p)
    for p in procs:
        p.join()

    # merge per-rank shards -> predictions.json (same schema/location as the vLLM generator)
    merged = []
    for rank in range(len(args.gpu_ids)):
        pp = f"{args.output_dir}/preds_rank{rank}.json"
        if os.path.exists(pp):
            merged.extend(json.load(open(pp)))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/predictions.json", "w") as f:
        json.dump(merged, f, indent=2)
    n_ok = sum(1 for r in merged if r["status"] == "success")
    print(f"Saved {len(merged)} predictions ({n_ok} ok) to {args.output_dir}/predictions.json")


if __name__ == "__main__":
    main()
