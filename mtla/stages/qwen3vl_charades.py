"""Charades-STA temporal grounding pipeline (v2): frame-aware + per-digit-slot attention.

Adapts QVH v3 machinery to Charades-STA:
  - Reshape video tokens to [T, H*W] using video_grid_thw, so we can compute
    per-frame attention features (frame_sum, frame_max).
  - Save 4 token-position slots (first, first2_mean, last, all_mean) per (L, H).
  - Save event_text_sum (attention to {query} portion of prompt) for mention analysis.
  - Single GT span per query; hallu = IoU(pred, GT) < 0.5.

Output schema, per record's "attn":
  video_max:        [4, L, H]
  video_sum:        [4, L, H]
  frame_sum:        [4, L, H, T]
  frame_max:        [4, L, H, T]
  task_text_sum:    [4, L, H]
  event_text_sum:   [4, L, H]
  system_sum:       [4, L, H]
  specials_sum:     [4, L, H]
  response_sum:     [4, L, H]

Slot order: 0=first_digit, 1=first2_mean, 2=last_digit, 3=all_mean.
"""
import os, json, argparse, time, gc, re
from multiprocessing import Process, set_start_method
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu

# Paths come from the command line (see --help). Defaults are placeholders.
PARQUET  = "charades_test.parquet"   # --data      : Charades-STA test parquet
VIDEO_DIR = "videos"                 # --video_dir : dir of clips
OUT_DIR  = "features"                # --out_dir   : attention shards out
PRED_DIR = "predictions"             # --pred_dir  : predictions out
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

MAX_NEW_TOKENS = 64
SAMPLE_FPS = 2.0
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 128 * 32 * 32

PROMPT_TMPL = (
    "Locate the segment where the following event happens. "
    "Respond with start and end timestamps in seconds. "
    "Event: {query}"
)

_EXTRACT = {
    "active": False, "lang_attn_ids": set(), "lang_attn_order": [],
    "query_positions": None,
    "video_indices": None, "task_text_indices": None, "event_text_indices": None,
    "system_indices": None, "specials_indices": None, "response_indices": None,
    "T": None, "HW": None,
    "video_sum":   None,
    "video_max":   None,
    "frame_sum":   None,
    "frame_max":   None,
    "task_text_sum": None, "event_text_sum": None, "system_sum": None,
    "specials_sum":  None, "response_sum": None,
}


def patched_eager_attention_forward(module, query, key, value, attention_mask,
                                    scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    s = _EXTRACT
    if s["active"] and id(module) in s["lang_attn_ids"]:
        layer_idx = s["lang_attn_order"].index(id(module))
        q_pos = s["query_positions"]
        rows = attn_weights[0].index_select(1, q_pos).transpose(0, 1).float()  # [n_q, heads, k]
        v_idx = s["video_indices"]
        v_rows = rows.index_select(2, v_idx)
        T, HW = s["T"], s["HW"]
        v_grid = v_rows.view(v_rows.shape[0], v_rows.shape[1], T, HW)
        f_sum = v_grid.sum(dim=3)
        f_max = v_grid.max(dim=3).values
        g_sum = v_rows.sum(dim=2)
        g_max = v_rows.max(dim=2).values

        s["video_sum"][:, layer_idx, :] = g_sum.cpu()
        s["video_max"][:, layer_idx, :] = g_max.cpu()
        s["frame_sum"][:, layer_idx, :, :] = f_sum.cpu()
        s["frame_max"][:, layer_idx, :, :] = f_max.cpu()

        for kn in ("system", "task_text", "event_text", "specials", "response"):
            kidx = s[f"{kn}_indices"]
            if kidx is not None and kidx.numel() > 0:
                grp = rows.index_select(2, kidx).sum(dim=2)
                s[f"{kn}_sum"][:, layer_idx, :] += grp.cpu()

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, None


def get_video_duration(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    return len(vr) / fps if fps > 0 else 0


def parse_timestamps(text):
    """Parse various timestamp output formats. Returns (start, end) or None."""
    t = text.lower().replace("seconds", "s")
    patterns = [
        r'start[:\s]+([\d.]+)\s*s?\s*,?\s*end[:\s]+([\d.]+)',
        r'from\s+([\d.]+)\s*s?\s+to\s+([\d.]+)',
        r'happens?\s+in\s+([\d.]+)\s*-\s*([\d.]+)',
        r'between\s+([\d.]+)\s*s?\s+and\s+([\d.]+)',
        r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
        r'\(([\d.]+)\s*,\s*([\d.]+)\)',
        r'([\d.]+)\s*s?\s*-\s*([\d.]+)\s*s',
        r'([\d.]+)\s*-\s*([\d.]+)\s*s',
        r'([\d.]+)\s*s\s+to\s+([\d.]+)',
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:
                continue
    return None


def iou(pred, gt):
    s1, e1 = pred; s2, e2 = gt
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / max(union, 1e-9)


def find_number_token_positions(prompt_len, response_text, tokenizer):
    enc = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    idxs = []
    for ti, (s, e) in enumerate(offsets):
        chunk = response_text[s:e]
        if any(c.isdigit() for c in chunk):
            idxs.append(ti)
    return [prompt_len + ti for ti in idxs]


def slot_aggregate_4d(arr_nq_l_h_t):
    n_q = arr_nq_l_h_t.shape[0]
    if n_q == 1:
        s0 = arr_nq_l_h_t[0]; s1 = arr_nq_l_h_t[0]; s2 = arr_nq_l_h_t[0]; s3 = arr_nq_l_h_t[0]
    else:
        s0 = arr_nq_l_h_t[0]
        s1 = arr_nq_l_h_t[:2].mean(dim=0)
        s2 = arr_nq_l_h_t[-1]
        s3 = arr_nq_l_h_t.mean(dim=0)
    return torch.stack([s0, s1, s2, s3], dim=0)


def slot_max_4d(arr_nq_l_h_t):
    n_q = arr_nq_l_h_t.shape[0]
    if n_q == 1:
        s0 = arr_nq_l_h_t[0]; s1 = arr_nq_l_h_t[0]; s2 = arr_nq_l_h_t[0]; s3 = arr_nq_l_h_t[0]
    else:
        s0 = arr_nq_l_h_t[0]
        s1 = arr_nq_l_h_t[:2].max(dim=0).values
        s2 = arr_nq_l_h_t[-1]
        s3 = arr_nq_l_h_t.max(dim=0).values
    return torch.stack([s0, s1, s2, s3], dim=0)


def worker(rank, gpu_id, items, out_dir, pred_dir, video_dir, svar_shift=False, seed=None,
           mode="extract", responses=None):
    # mode="generate": run model.generate, write predictions only (no attention hook).
    # mode="extract": read each item's saved `response`, run the monkeypatched forward, write
    #   attention .pt only. Vision preprocessing/token-indexing/extraction are identical here.
    print(f"[worker {rank}] mode={mode} gpu={gpu_id} n={len(items)} svar_shift={svar_shift} seed={seed}", flush=True)
    torch.cuda.set_device(gpu_id)
    # Sampled rollout: seed per-worker so each (seed, rank) draw is reproducible.
    # do_sample only flips on when --seed is passed; default stays greedy.
    do_sample = seed is not None
    if do_sample:
        from transformers import set_seed as _hf_set_seed
        _hf_set_seed(seed * 1000 + rank)
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_mod
    if mode == "extract":
        qwen3_mod.eager_attention_forward = patched_eager_attention_forward
    responses = responses or {}

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager",
        device_map=f"cuda:{gpu_id}",
    ).eval()
    decoder_layers = model.model.language_model.layers
    n_layers = len(decoder_layers)
    n_heads  = model.config.text_config.num_attention_heads
    print(f"[worker {rank}] LLM layers={n_layers} heads={n_heads}", flush=True)
    _EXTRACT["lang_attn_ids"]   = {id(L.self_attn) for L in decoder_layers}
    _EXTRACT["lang_attn_order"] = [id(L.self_attn) for L in decoder_layers]

    video_pad_id = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    image_pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    special_token_strings = [
        "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>", "<|video_pad|>",
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|endoftext|>",
    ]
    specials_ids = set()
    for stk in special_token_strings:
        tid = proc.tokenizer.convert_tokens_to_ids(stk)
        if isinstance(tid, int) and tid >= 0: specials_ids.add(tid)
    for tid in (proc.tokenizer.all_special_ids or []):
        if tid != video_pad_id and tid != image_pad_id:
            specials_ids.add(int(tid))

    device = f"cuda:{gpu_id}"
    records = []
    preds_out = []
    t0 = time.time()
    n_done = n_skipped = n_correct = 0

    for cnt, item in enumerate(items):
        video_id = item["video"]
        query    = item["caption"]
        gt_start, gt_end = float(item["timestamp"][0]), float(item["timestamp"][1])
        video_path = f"{video_dir}/{video_id}"
        if not os.path.exists(video_path):
            n_skipped += 1; continue
        try:
            duration = get_video_duration(video_path)
        except Exception as e:
            print(f"[worker {rank}] skip {video_id}: {e}", flush=True)
            n_skipped += 1; continue

        question = PROMPT_TMPL.format(query=query.rstrip("."))
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": f"file://{video_path}",
             "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "fps": SAMPLE_FPS},
            {"type": "text", "text": question},
        ]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        try:
            images, videos, video_kwargs = process_vision_info(
                msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None
            inputs = proc(text=text, images=images, videos=videos, video_metadata=video_metadatas,
                          do_resize=False, return_tensors="pt", **video_kwargs).to(device)
        except Exception as e:
            print(f"[worker {rank}] skip {video_id}: processor {e}", flush=True)
            n_skipped += 1; continue
        prompt_ids = inputs["input_ids"][0]
        prompt_len = prompt_ids.shape[0]

        vgthw = inputs.get("video_grid_thw")
        if vgthw is None or vgthw.shape[0] != 1:
            print(f"[worker {rank}] skip {video_id}: no video_grid_thw", flush=True)
            n_skipped += 1; continue
        T_grid = int(vgthw[0, 0].item())
        H_grid = int(vgthw[0, 1].item())
        W_grid = int(vgthw[0, 2].item())
        spatial_merge_size = getattr(model.config.vision_config, "spatial_merge_size", 2)
        T_tokens = T_grid
        H_tokens = H_grid // spatial_merge_size
        W_tokens = W_grid // spatial_merge_size
        n_video_expected = T_tokens * H_tokens * W_tokens

        if mode == "extract":
            response = responses.get(video_id)
            if response is None:
                n_skipped += 1; continue
        else:
            try:
                with torch.no_grad():
                    if do_sample:
                        gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                                 do_sample=True, temperature=0.7, top_p=0.95)
                    else:
                        gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            except Exception as e:
                print(f"[worker {rank}] skip {video_id}: gen {e}", flush=True)
                n_skipped += 1; torch.cuda.empty_cache(); continue
            new_tokens = gen_ids[0, prompt_len:]
            response = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        parsed = parse_timestamps(response)
        if parsed is None:
            is_hallu = True
            pred_start, pred_end = -1.0, -1.0
            iou_val = 0.0
        else:
            pred_start, pred_end = parsed
            iou_val = iou((pred_start, pred_end), (gt_start, gt_end))
            is_hallu = (iou_val < 0.5)
        if not is_hallu: n_correct += 1

        if mode == "generate":
            preds_out.append({
                "video": video_id, "caption": query,
                "gt_span": [gt_start, gt_end],
                "pred_span": [pred_start, pred_end] if parsed else None,
                "response": response, "iou": float(iou_val),
                "is_correct": (not is_hallu),
                "duration_s": float(duration),
                "T_tokens": int(T_tokens), "H_tokens": int(H_tokens), "W_tokens": int(W_tokens),
                "fps": SAMPLE_FPS, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
            })
            n_done += 1
            continue

        resp_enc = proc.tokenizer(response, add_special_tokens=False)
        resp_ids = torch.tensor(resp_enc["input_ids"], dtype=prompt_ids.dtype, device=device)
        if resp_ids.numel() == 0:
            n_skipped += 1; continue
        full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
        total_len = full_ids.shape[1]
        number_positions = find_number_token_positions(prompt_len, response, proc.tokenizer)
        if not number_positions:
            number_positions = [prompt_len]
        # SVAR-faithful shift: extract attention at the predicting position (target_pos - 1)
        # rather than at the target token itself. For p_target = prompt_len + 0 the predicting
        # position is the last prompt token (prompt_len - 1).
        if svar_shift:
            number_positions = [max(0, p - 1) for p in number_positions]

        im_start_id = proc.tokenizer.convert_tokens_to_ids("<|im_start|>")
        user_id = proc.tokenizer.convert_tokens_to_ids("user")
        assistant_id = proc.tokenizer.convert_tokens_to_ids("assistant")
        prompt_cpu = prompt_ids.cpu().tolist()
        user_turn_start = 0
        for k in range(len(prompt_cpu) - 1):
            if prompt_cpu[k] == im_start_id and prompt_cpu[k + 1] == user_id:
                user_turn_start = k; break
        assistant_start = None
        for k in range(len(prompt_cpu) - 1):
            if prompt_cpu[k] == im_start_id and prompt_cpu[k + 1] == assistant_id:
                assistant_start = k; break
        video_idx_l, system_idx_l, task_idx_l, spc_idx_l = [], [], [], []
        for k, tid in enumerate(prompt_cpu):
            if tid == video_pad_id:
                video_idx_l.append(k)
            elif tid in specials_ids:
                spc_idx_l.append(k)
            elif assistant_start is not None and k >= assistant_start:
                spc_idx_l.append(k)
            elif k >= user_turn_start:
                task_idx_l.append(k)
            else:
                system_idx_l.append(k)
        resp_idx_l = list(range(prompt_len, total_len))
        if not video_idx_l:
            n_skipped += 1; continue

        # event_text_indices: tokens corresponding to {query} (after "Event:" in the prompt).
        event_idx_l = []
        try:
            event_ids = proc.tokenizer(" " + query.rstrip("."), add_special_tokens=False)["input_ids"]
            if event_ids:
                task_set = set(task_idx_l)
                L_e = len(event_ids)
                for start in range(prompt_len - L_e, -1, -1):
                    if (start in task_set
                        and prompt_cpu[start:start + L_e] == event_ids
                        and all(j in task_set for j in range(start, start + L_e))):
                        event_idx_l = list(range(start, start + L_e))
                        break
        except Exception:
            event_idx_l = []

        if len(video_idx_l) != n_video_expected:
            print(f"[worker {rank}] skip {video_id}: video token count mismatch {len(video_idx_l)} vs T*H*W={n_video_expected}", flush=True)
            n_skipped += 1; continue

        T = T_tokens
        HW = H_tokens * W_tokens

        device_idx = lambda L: torch.tensor(L, dtype=torch.long, device=device)
        _EXTRACT["query_positions"]  = device_idx(number_positions)
        _EXTRACT["video_indices"]    = device_idx(video_idx_l)
        _EXTRACT["system_indices"]   = device_idx(system_idx_l)
        _EXTRACT["task_text_indices"]= device_idx(task_idx_l)
        _EXTRACT["event_text_indices"]= device_idx(event_idx_l) if event_idx_l else torch.zeros(0, dtype=torch.long, device=device)
        _EXTRACT["specials_indices"] = device_idx(spc_idx_l)
        _EXTRACT["response_indices"] = device_idx(resp_idx_l)
        _EXTRACT["T"]  = T
        _EXTRACT["HW"] = HW

        n_q = len(number_positions)
        zero3 = lambda: torch.zeros(n_q, n_layers, n_heads, dtype=torch.float32)
        zero4 = lambda: torch.zeros(n_q, n_layers, n_heads, T, dtype=torch.float32)
        _EXTRACT["video_sum"]    = zero3()
        _EXTRACT["video_max"]    = zero3()
        _EXTRACT["frame_sum"]    = zero4()
        _EXTRACT["frame_max"]    = zero4()
        _EXTRACT["task_text_sum"]= zero3()
        _EXTRACT["event_text_sum"]= zero3()
        _EXTRACT["system_sum"]   = zero3()
        _EXTRACT["specials_sum"] = zero3()
        _EXTRACT["response_sum"] = zero3()

        fk = {
            "input_ids": full_ids,
            "attention_mask": torch.ones(1, total_len, device=device, dtype=torch.long),
        }
        for k in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
            if k in inputs: fk[k] = inputs[k]
        if "mm_token_type_ids" in inputs:
            orig = inputs["mm_token_type_ids"]
            extra = total_len - orig.shape[1]
            fk["mm_token_type_ids"] = torch.cat([
                orig, torch.zeros(1, extra, dtype=orig.dtype, device=orig.device)
            ], dim=1) if extra > 0 else orig

        _EXTRACT["active"] = True
        try:
            with torch.no_grad():
                model(**fk)
        except Exception as e:
            _EXTRACT["active"] = False
            print(f"[worker {rank}] skip {video_id}: forward {e}", flush=True)
            n_skipped += 1; torch.cuda.empty_cache(); continue
        _EXTRACT["active"] = False

        def slot3_max(x):
            n = x.shape[0]
            if n == 1: return torch.stack([x[0]]*4, dim=0)
            return torch.stack([x[0], x[:2].max(dim=0).values, x[-1], x.max(dim=0).values], dim=0)

        def slot3_mean(x):
            n = x.shape[0]
            if n == 1: return torch.stack([x[0]]*4, dim=0)
            return torch.stack([x[0], x[:2].mean(dim=0), x[-1], x.mean(dim=0)], dim=0)

        video_max_slots = slot3_max (_EXTRACT["video_max"])
        video_sum_slots = slot3_mean(_EXTRACT["video_sum"])
        frame_sum_slots = slot_aggregate_4d(_EXTRACT["frame_sum"])
        frame_max_slots = slot_max_4d   (_EXTRACT["frame_max"])
        text_sum_slots  = slot3_mean(_EXTRACT["task_text_sum"])
        evt_sum_slots   = slot3_mean(_EXTRACT["event_text_sum"])
        sys_sum_slots   = slot3_mean(_EXTRACT["system_sum"])
        spc_sum_slots   = slot3_mean(_EXTRACT["specials_sum"])
        rsp_sum_slots   = slot3_mean(_EXTRACT["response_sum"])

        records.append({
            "video": video_id,
            "caption": query,
            "gt_span": [gt_start, gt_end],
            "pred_span": [pred_start, pred_end] if parsed else None,
            "response": response,
            "iou": float(iou_val),
            "is_hallucinated": bool(is_hallu),
            "n_video_tokens": len(video_idx_l),
            "n_query_positions": n_q,
            "n_event_text_tokens": int(len(event_idx_l)),
            "duration_s": float(duration),
            "T_tokens": int(T),
            "H_tokens": int(H_tokens),
            "W_tokens": int(W_tokens),
            "attn": {
                "video_sum":     video_sum_slots.to(torch.float16).numpy(),
                "video_max":     video_max_slots.to(torch.float16).numpy(),
                "frame_sum":     frame_sum_slots.to(torch.float16).numpy(),
                "frame_max":     frame_max_slots.to(torch.float16).numpy(),
                "task_text_sum":  text_sum_slots.to(torch.float16).numpy(),
                "event_text_sum": evt_sum_slots .to(torch.float16).numpy(),
                "system_sum":     sys_sum_slots .to(torch.float16).numpy(),
                "specials_sum":  spc_sum_slots  .to(torch.float16).numpy(),
                "response_sum":  rsp_sum_slots  .to(torch.float16).numpy(),
            },
        })
        n_done += 1
        if n_done % 25 == 0:
            rate = n_done / max(time.time() - t0, 1e-9)
            eta = (len(items) - cnt - 1) / max(rate, 1e-9)
            acc = n_correct / max(n_done, 1) * 100
            print(f"[worker {rank}] [{cnt+1}/{len(items)}] done={n_done} skip={n_skipped} "
                  f"R@IoU0.5={acc:.1f}% rate={rate:.2f}qps eta={eta/60:.1f}min", flush=True)
        for kk in list(_EXTRACT.keys()):
            if kk in ("active","lang_attn_ids","lang_attn_order"): continue
            _EXTRACT[kk] = None
        torch.cuda.empty_cache(); gc.collect()

    if mode == "extract":
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/shard{rank}.pt"
        torch.save(records, out_path)
        print(f"[worker {rank}] saved {len(records)} -> {out_path}", flush=True)
    else:
        os.makedirs(pred_dir, exist_ok=True)
        with open(f"{pred_dir}/preds_rank{rank}.json", "w") as f:
            json.dump(preds_out, f)
        print(f"[worker {rank}] saved {len(preds_out)} preds", flush=True)


def main():
    global PARQUET, VIDEO_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--workers_per_gpu", type=int, default=1)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--pred_dir", default=PRED_DIR)
    ap.add_argument("--data", default=PARQUET, help="Charades-STA test parquet")
    ap.add_argument("--video_dir", default=VIDEO_DIR, help="dir of video clips")
    ap.add_argument("--limit", type=int, default=999999)
    ap.add_argument("--svar_shift", action="store_true",
                    help="Extract attention at predicting position (target_pos - 1) "
                         "to match SVAR's convention. Default: extract at target_pos.")
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, sample (T=0.7, top-p=0.95) seeded by this value "
                         "instead of greedy decoding. Used for self-consistency rollouts.")
    ap.add_argument("--mode", choices=["generate", "extract"], default="extract",
                    help="generate -> predictions.json only; extract -> attention .pt only.")
    args = ap.parse_args()
    set_start_method("spawn", force=True)

    PARQUET, VIDEO_DIR = args.data, args.video_dir
    df = pd.read_parquet(PARQUET)
    print(f"Charades-STA test queries: {len(df)}  mode={args.mode}")
    items = df.head(args.limit).to_dict("records")

    # extract mode: load responses produced by the generate stage, keyed by video id.
    responses = {}
    if args.mode == "extract":
        for p in json.load(open(f"{args.pred_dir}/predictions.json")):
            responses[p["video"]] = p["response"]
        items = [it for it in items if it["video"] in responses]

    n_workers = len(args.gpus) * args.workers_per_gpu
    chunks = np.array_split(items, n_workers)
    procs = []
    rank = 0
    for gpu in args.gpus:
        for w in range(args.workers_per_gpu):
            if rank >= len(chunks) or len(chunks[rank]) == 0:
                rank += 1; continue
            p = Process(target=worker, args=(rank, gpu, list(chunks[rank]), args.out_dir,
                                             args.pred_dir, args.video_dir, args.svar_shift,
                                             args.seed, args.mode, responses))
            p.start(); procs.append(p)
            rank += 1
    for p in procs:
        p.join()

    if args.mode == "generate":
        all_preds = []
        for r in range(n_workers):
            pp = f"{args.pred_dir}/preds_rank{r}.json"
            if os.path.exists(pp):
                all_preds.extend(json.load(open(pp)))
        with open(f"{args.pred_dir}/predictions.json", "w") as f:
            json.dump(all_preds, f)
        n_correct = sum(1 for p in all_preds if p["is_correct"])
        print(f"\nMerged: {len(all_preds)} preds, R@IoU0.5={n_correct/max(len(all_preds),1)*100:.2f}%")
    print("all workers complete")


if __name__ == "__main__":
    main()
