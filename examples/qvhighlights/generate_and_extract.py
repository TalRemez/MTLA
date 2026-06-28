"""QVHighlights moment retrieval pipeline (v2): frame-aware + per-digit-slot attention.

Improvements over v1:
  - Reshape video tokens to [T, H*W] using video_grid_thw, so we can compute
    per-frame attention features (frame_sum, frame_max) instead of a single
    sum/max over the entire ~3,750-token soup.
  - Save 4 token-position slots (first, first2_mean, last, all_mean) for
    every per-(L, H) statistic, so we can investigate which response token
    carries the hallucination signal.
  - Save per-frame stats so we can later compute predicted-window-restricted
    features offline (no need to re-extract).

Output schema, per record's "attn":
  video_max:        [4, 36, 32]      max over all video tokens, per slot
  video_sum:        [4, 36, 32]      sum over all video tokens, per slot
  frame_sum:        [4, 36, 32, T]   sum within each frame, per slot
  frame_max:        [4, 36, 32, T]   max within each frame, per slot
  task_text_sum:    [4, 36, 32]
  system_sum:       [4, 36, 32]
  specials_sum:     [4, 36, 32]
  response_sum:     [4, 36, 32]
  T:  scalar int (number of encoded video frames after Qwen3-VL temporal patch merge)

Slot order: 0=first_digit, 1=first2_mean, 2=last_digit, 3=all_mean.

Temporal indexing: encoded frame t corresponds to video time
  t * (duration / T)  seconds (approx; Qwen3-VL applies a temporal patch
  merge so T is roughly  ceil(SAMPLE_FPS * duration / 2) ).
"""
import os, json, argparse, time, gc, re, glob
from multiprocessing import Process, set_start_method
import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu

# Paths come from the command line (see --help). Defaults are placeholders.
ANN_PATH = "highlight_val_release.jsonl"  # --ann       : QVHighlights val annotations (jsonl)
VIDEO_DIR = "videos"                      # --video_dir : dir of {vid}.mp4 clips
OUT_DIR  = "features"                     # --out_dir   : attention shards out
PRED_DIR = "predictions"                  # --pred_dir  : predictions out
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

MAX_NEW_TOKENS = 128
SAMPLE_FPS = 1.0
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 64 * 32 * 32

PROMPT_TMPL = (
    "Locate every segment in the video where the following event happens. "
    "Respond with a list of [start, end] timestamps in seconds, one pair per segment. "
    "If the event happens multiple times, list all occurrences. "
    "Event: {query}"
)

_EXTRACT = {
    "active": False, "lang_attn_ids": set(), "lang_attn_order": [],
    "query_positions": None,
    "video_indices": None, "task_text_indices": None, "event_text_indices": None,
    "system_indices": None, "specials_indices": None, "response_indices": None,
    # video grid: needed to reshape [T*H*W] -> [T, H*W]
    "T": None, "HW": None,
    # accumulators (CPU fp32)
    "video_sum":   None,  # [n_q, n_layers, n_heads]
    "video_max":   None,  # [n_q, n_layers, n_heads]
    "frame_sum":   None,  # [n_q, n_layers, n_heads, T]
    "frame_max":   None,  # [n_q, n_layers, n_heads, T]
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
        v_rows = rows.index_select(2, v_idx)                                    # [n_q, heads, T*H*W]
        T, HW = s["T"], s["HW"]
        # Reshape to per-frame tensor
        v_grid = v_rows.view(v_rows.shape[0], v_rows.shape[1], T, HW)           # [n_q, heads, T, HW]
        # Per-frame stats
        f_sum = v_grid.sum(dim=3)                                                # [n_q, heads, T]
        f_max = v_grid.max(dim=3).values                                         # [n_q, heads, T]
        # Globals (kept for backward compat / sanity check)
        g_sum = v_rows.sum(dim=2)                                                # [n_q, heads]
        g_max = v_rows.max(dim=2).values                                         # [n_q, heads]

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


_TIME = r'(\d{1,3}(?::\d{2})?(?:\.\d+)?)'
def _to_seconds(token):
    if ":" in token:
        m, s = token.split(":")
        return float(m) * 60 + float(s)
    return float(token)


def parse_windows(text):
    t = text.lower().replace("seconds", "s")
    patterns = [
        rf'\[\s*{_TIME}\s*,\s*{_TIME}\s*\]',
        rf'\(\s*{_TIME}\s*,\s*{_TIME}\s*\)',
        rf'from\s+{_TIME}\s*s?\s+to\s+{_TIME}',
        rf'between\s+{_TIME}\s*s?\s+and\s+{_TIME}',
        rf'{_TIME}\s*s?\s*-\s*{_TIME}\s*s',
        rf'{_TIME}\s*s?\s*-\s*{_TIME}',
        rf'{_TIME}\s*s\s+to\s+{_TIME}',
        rf'start[:\s]+{_TIME}\s*s?\s*,?\s*end[:\s]+{_TIME}',
    ]
    seen = set(); windows = []
    for p in patterns:
        for m in re.finditer(p, t):
            try:
                a = _to_seconds(m.group(1)); b = _to_seconds(m.group(2))
            except ValueError:
                continue
            if a > b: a, b = b, a
            if a == b: continue
            key = (round(a, 2), round(b, 2))
            if key in seen: continue
            seen.add(key); windows.append((a, b))
        if windows: break
    return windows


def iou_pair(p, g):
    s1, e1 = p; s2, e2 = g
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / max(union, 1e-9)


def best_match_iou(pred_windows, gt_windows):
    if not pred_windows or not gt_windows:
        return 0.0, 0.0
    used = set(); ious = []
    for g in gt_windows:
        best_i, best_iou = -1, -1.0
        for i, p in enumerate(pred_windows):
            if i in used: continue
            v = iou_pair(p, g)
            if v > best_iou: best_iou, best_i = v, i
        if best_i >= 0:
            used.add(best_i); ious.append(best_iou)
        else:
            ious.append(0.0)
    if not ious: return 0.0, 0.0
    return float(np.mean(ious)), float(min(ious))


def find_number_token_positions(prompt_len, response_text, tokenizer):
    enc = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    idxs = []
    for ti, (s, e) in enumerate(offsets):
        chunk = response_text[s:e]
        if any(c.isdigit() for c in chunk):
            idxs.append(ti)
    return [prompt_len + ti for ti in idxs]


def parse_windows_with_spans(text):
    """Like parse_windows, but returns (windows, char_spans) where char_spans[i]
    is the (start, end) character offset of window i's match in `text` (lowercased
    copy — offsets are valid against the original too since .lower() preserves length).
    Same dedup/ordering as parse_windows so windows[i] aligns with pred_windows[i]."""
    t = text.lower().replace("seconds", "s")  # NOTE: replace changes length; see below
    # To keep char offsets aligned with the ORIGINAL response_text we must not change
    # length. "seconds"->"s" shrinks it, so re-derive spans on a length-preserving lower.
    t = text.lower()
    patterns = [
        rf'\[\s*{_TIME}\s*,\s*{_TIME}\s*\]',
        rf'\(\s*{_TIME}\s*,\s*{_TIME}\s*\)',
        rf'from\s+{_TIME}\s*s?\s+to\s+{_TIME}',
        rf'between\s+{_TIME}\s*s?\s+and\s+{_TIME}',
        rf'{_TIME}\s*s?\s*-\s*{_TIME}\s*s',
        rf'{_TIME}\s*s?\s*-\s*{_TIME}',
        rf'{_TIME}\s*s\s+to\s+{_TIME}',
        rf'start[:\s]+{_TIME}\s*s?\s*,?\s*end[:\s]+{_TIME}',
    ]
    seen = set(); windows = []; spans = []
    for p in patterns:
        for m in re.finditer(p, t):
            try:
                a = _to_seconds(m.group(1)); b = _to_seconds(m.group(2))
            except ValueError:
                continue
            if a > b: a, b = b, a
            if a == b: continue
            key = (round(a, 2), round(b, 2))
            if key in seen: continue
            seen.add(key); windows.append((a, b)); spans.append(m.span())
        if windows: break
    return windows, spans


def find_perwindow_token_positions(prompt_len, response_text, tokenizer, spans):
    """For each window char-span, the response digit-token positions (prompt-relative)
    whose character offsets fall inside that span. Returns list aligned with spans;
    a window with no matched digit tokens gets an empty list."""
    enc = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    per_win = []
    for (cs, ce) in spans:
        toks = []
        for ti, (ts, te) in enumerate(offsets):
            if ts >= cs and te <= ce and any(c.isdigit() for c in response_text[ts:te]):
                toks.append(prompt_len + ti)
        per_win.append(toks)
    return per_win


def find_perwindow_startend_token_positions(prompt_len, response_text, tokenizer, spans):
    """Like find_perwindow_token_positions, but splits each window's number tokens into
    the START number group and the END number group. A window's [start, end] char span
    contains exactly two numbers, each possibly a DECIMAL (e.g. ``27.7``), so a token is
    part of a number if it contains a digit OR is a decimal point flanked by digits
    (so ``27.7`` stays one run). The run breaks only on a true separator (comma, 'to',
    '-', whitespace) between the two numbers. First run -> start, last run -> end.
    Returns (start_list, end_list), each aligned with spans."""
    enc = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    def is_number_tok(ts, te):
        s = response_text[ts:te]
        if any(c.isdigit() for c in s):
            return True
        # an intra-number punctuation token (decimal point ``.`` or time colon ``:``,
        # e.g. ``27.7`` or ``1:21``): part of the number if a digit sits on both sides.
        if s.strip() in (".", ":"):
            left = response_text[ts-1] if ts > 0 else ""
            right = response_text[te] if te < len(response_text) else ""
            return left.isdigit() and right.isdigit()
        return False

    starts, ends = [], []
    for (cs, ce) in spans:
        runs, cur = [], []
        for ti, (ts, te) in enumerate(offsets):
            if not (ts >= cs and te <= ce):
                continue
            if is_number_tok(ts, te):
                cur.append(prompt_len + ti)
            elif cur:                       # a separator token ends the current number
                runs.append(cur); cur = []
        if cur: runs.append(cur)
        starts.append(runs[0] if runs else [])
        ends.append(runs[-1] if runs else [])
    return starts, ends


def slot_aggregate_3d(arr_nq_l_h):
    """arr: [n_q, n_layers, n_heads]  ->  [4, n_layers, n_heads]
    Slots: first, first2_mean, last, all_mean."""
    n_q = arr_nq_l_h.shape[0]
    if n_q == 1:
        s0 = arr_nq_l_h[0]; s1 = arr_nq_l_h[0]; s2 = arr_nq_l_h[0]; s3 = arr_nq_l_h[0]
    elif n_q == 2:
        s0 = arr_nq_l_h[0]
        s1 = arr_nq_l_h[:2].mean(dim=0)
        s2 = arr_nq_l_h[-1]
        s3 = arr_nq_l_h.mean(dim=0)
    else:
        s0 = arr_nq_l_h[0]
        s1 = arr_nq_l_h[:2].mean(dim=0)
        s2 = arr_nq_l_h[-1]
        s3 = arr_nq_l_h.mean(dim=0)
    return torch.stack([s0, s1, s2, s3], dim=0)  # [4, L, H]


def slot_aggregate_4d(arr_nq_l_h_t):
    """arr: [n_q, n_layers, n_heads, T]  ->  [4, n_layers, n_heads, T]"""
    n_q = arr_nq_l_h_t.shape[0]
    if n_q == 1:
        s0 = arr_nq_l_h_t[0]; s1 = arr_nq_l_h_t[0]; s2 = arr_nq_l_h_t[0]; s3 = arr_nq_l_h_t[0]
    else:
        s0 = arr_nq_l_h_t[0]
        s1 = arr_nq_l_h_t[:2].mean(dim=0)
        s2 = arr_nq_l_h_t[-1]
        s3 = arr_nq_l_h_t.mean(dim=0)
    return torch.stack([s0, s1, s2, s3], dim=0)  # [4, L, H, T]


def slot_max_4d(arr_nq_l_h_t):
    """For frame_max we want max-over-q for the all_mean slot (not mean), since
    frame_max is itself a max statistic. We still keep first/first2/last as-is.
    Returning: [4, L, H, T] with slot 1 = max over first 2, slot 3 = max over all."""
    n_q = arr_nq_l_h_t.shape[0]
    if n_q == 1:
        s0 = arr_nq_l_h_t[0]; s1 = arr_nq_l_h_t[0]; s2 = arr_nq_l_h_t[0]; s3 = arr_nq_l_h_t[0]
    else:
        s0 = arr_nq_l_h_t[0]
        s1 = arr_nq_l_h_t[:2].max(dim=0).values
        s2 = arr_nq_l_h_t[-1]
        s3 = arr_nq_l_h_t.max(dim=0).values
    return torch.stack([s0, s1, s2, s3], dim=0)


def worker(rank, gpu_id, items, out_dir, pred_dir, svar_shift=False, seed=None):
    print(f"[worker {rank}] gpu={gpu_id} n={len(items)} svar_shift={svar_shift} seed={seed}", flush=True)
    torch.cuda.set_device(gpu_id)
    # Sampled rollout: seed per-worker so each (seed, rank) draw is reproducible.
    # do_sample only flips on when --seed is passed; default stays greedy.
    do_sample = seed is not None
    if do_sample:
        from transformers import set_seed as _hf_set_seed
        _hf_set_seed(seed * 1000 + rank)
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_mod
    qwen3_mod.eager_attention_forward = patched_eager_attention_forward

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
        vid    = item["vid"]
        query  = item["query"]
        gt_windows = [tuple(w) for w in item["relevant_windows"]]
        video_path = f"{VIDEO_DIR}/{vid}.mp4"
        if not os.path.exists(video_path):
            n_skipped += 1; continue
        try:
            duration = get_video_duration(video_path)
        except Exception as e:
            print(f"[worker {rank}] skip {vid}: {e}", flush=True)
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
            print(f"[worker {rank}] skip {vid}: processor {e}", flush=True)
            n_skipped += 1; continue
        prompt_ids = inputs["input_ids"][0]
        prompt_len = prompt_ids.shape[0]

        # Get video grid (T, H, W) — there should be exactly 1 video
        vgthw = inputs.get("video_grid_thw")
        if vgthw is None or vgthw.shape[0] != 1:
            print(f"[worker {rank}] skip {vid}: no video_grid_thw", flush=True)
            n_skipped += 1; continue
        T_grid = int(vgthw[0, 0].item())
        H_grid = int(vgthw[0, 1].item())
        W_grid = int(vgthw[0, 2].item())
        # Qwen3-VL spatial merge: each token covers a 2x2 patch block
        # video_grid_thw stores the *patch* counts; spatial_merge_size=2 reduces to T x H/2 x W/2 tokens
        spatial_merge_size = getattr(model.config.vision_config, "spatial_merge_size", 2)
        T_tokens = T_grid
        H_tokens = H_grid // spatial_merge_size
        W_tokens = W_grid // spatial_merge_size
        n_video_expected = T_tokens * H_tokens * W_tokens

        try:
            with torch.no_grad():
                if do_sample:
                    gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                             do_sample=True, temperature=0.7, top_p=0.95)
                else:
                    gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        except Exception as e:
            print(f"[worker {rank}] skip {vid}: gen {e}", flush=True)
            n_skipped += 1; torch.cuda.empty_cache(); continue
        new_tokens = gen_ids[0, prompt_len:]
        response = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        pred_windows = parse_windows(response)
        if not pred_windows:
            is_hallu = True; mean_iou = 0.0; min_iou = 0.0
        else:
            mean_iou, min_iou = best_match_iou(pred_windows, gt_windows)
            is_hallu = (mean_iou < 0.5)
        if not is_hallu: n_correct += 1

        resp_enc = proc.tokenizer(response, add_special_tokens=False)
        resp_ids = torch.tensor(resp_enc["input_ids"], dtype=prompt_ids.dtype, device=device)
        if resp_ids.numel() == 0:
            n_skipped += 1; continue
        full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
        total_len = full_ids.shape[1]
        number_positions = find_number_token_positions(prompt_len, response, proc.tokenizer)
        if not number_positions:
            number_positions = [prompt_len]
        # Per-window token attribution: map each window to the indices (into
        # number_positions) of the digit tokens inside ITS [start,end] char span,
        # so multi-segment QVH scores each window by its OWN coords (image-case parity).
        _win, _spans = parse_windows_with_spans(response)
        _perwin_abspos = find_perwindow_token_positions(prompt_len, response, proc.tokenizer, _spans)
        _perwin_start_abspos, _perwin_end_abspos = find_perwindow_startend_token_positions(
            prompt_len, response, proc.tokenizer, _spans)
        pos_to_qidx = {p: i for i, p in enumerate(number_positions)}
        # list (aligned with pred_windows) of entry-indices into the n_q extraction order
        perwin_qidx = [[pos_to_qidx[p] for p in toks if p in pos_to_qidx]
                       for toks in _perwin_abspos]
        perwin_start_qidx = [[pos_to_qidx[p] for p in toks if p in pos_to_qidx]
                             for toks in _perwin_start_abspos]
        perwin_end_qidx   = [[pos_to_qidx[p] for p in toks if p in pos_to_qidx]
                             for toks in _perwin_end_abspos]
        # SVAR-faithful shift: extract attention at the predicting position (target_pos - 1)
        if svar_shift:
            number_positions = [max(0, p - 1) for p in number_positions]
            # shift invalidates the exact pos->qidx map; rebuild perwin against shifted set
            # (shift is monotonic, order preserved, so qidx positions are unchanged)
            pass

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

        # event_text_indices: tokens corresponding to {query} (the part after "Event:" in the prompt).
        # We tokenize the standalone query and search for that token sequence inside prompt_cpu.
        event_idx_l = []
        try:
            # The exact substring placed in the prompt is: "Event: <query>"
            # We want the tokens of "<query>" only (after the colon and space).
            event_substr = f"Event: {query.rstrip('.')}"
            # Find tokens of just the query string with leading space (matches how it would appear after "Event:")
            event_ids = proc.tokenizer(" " + query.rstrip("."), add_special_tokens=False)["input_ids"]
            # Find where this run appears in prompt_cpu, restricted to task_idx_l region
            if event_ids:
                task_set = set(task_idx_l)
                L_e = len(event_ids)
                # Search backwards: the query is near the end of the user turn
                for start in range(prompt_len - L_e, -1, -1):
                    if (start in task_set
                        and prompt_cpu[start:start + L_e] == event_ids
                        and all(j in task_set for j in range(start, start + L_e))):
                        event_idx_l = list(range(start, start + L_e))
                        break
        except Exception as _e:
            event_idx_l = []
        # If we couldn't find them, fall back to empty (event_text features will be zero)

        if len(video_idx_l) != n_video_expected:
            # Shape mismatch — skip this record (safer than misaligning frames)
            print(f"[worker {rank}] skip {vid}: video token count mismatch {len(video_idx_l)} vs T*H*W={n_video_expected}", flush=True)
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
            print(f"[worker {rank}] skip {vid}: forward {e}", flush=True)
            n_skipped += 1; torch.cuda.empty_cache(); continue
        _EXTRACT["active"] = False

        # Slot-aggregate to [4, L, H] and [4, L, H, T]
        video_max_slots   = slot_max_4d(_EXTRACT["video_max"].unsqueeze(-1)).squeeze(-1)  # use max for max; but we have [n_q,L,H], so wrap
        # Simpler: re-implement
        def slot3_max(x):  # x: [n_q, L, H] -> [4, L, H], slot 3 = max over q (since x is itself a max)
            n_q = x.shape[0]
            if n_q == 1: return torch.stack([x[0]]*4, dim=0)
            return torch.stack([x[0], x[:2].max(dim=0).values, x[-1], x.max(dim=0).values], dim=0)

        def slot3_mean(x):  # x: [n_q, L, H] -> [4, L, H], slot 3 = mean over q
            n_q = x.shape[0]
            if n_q == 1: return torch.stack([x[0]]*4, dim=0)
            return torch.stack([x[0], x[:2].mean(dim=0), x[-1], x.mean(dim=0)], dim=0)

        video_max_slots = slot3_max (_EXTRACT["video_max"])      # [4, L, H]
        video_sum_slots = slot3_mean(_EXTRACT["video_sum"])      # [4, L, H]
        frame_sum_slots = slot_aggregate_4d(_EXTRACT["frame_sum"])  # [4, L, H, T]
        frame_max_slots = slot_max_4d   (_EXTRACT["frame_max"])     # [4, L, H, T]

        # Per-window frame_sum: for each predicted window, two variants over ITS OWN
        # coord-digit tokens -> each [n_windows, L, H, T]. Image-case parity for
        # multi-segment QVH (window scored by its own coords, not the response avg).
        #   _mean  = mean over the window's coord-digit tokens   (image coord_mean analog)
        #   _first = the window's FIRST coord-digit token        (first_digit analog)
        fs_full = _EXTRACT["frame_sum"]  # [n_q, L, H, T]
        perwin_mean_rows, perwin_first_rows = [], []
        for qidx_list in perwin_qidx:
            valid = [q for q in qidx_list if 0 <= q < fs_full.shape[0]]
            if valid:
                perwin_mean_rows.append(fs_full[valid].mean(dim=0))
                perwin_first_rows.append(fs_full[valid[0]])
            else:                                  # fallback: response-avg / first if unmatched
                perwin_mean_rows.append(fs_full.mean(dim=0))
                perwin_first_rows.append(fs_full[0])
        frame_sum_perwin = (torch.stack(perwin_mean_rows, dim=0) if perwin_mean_rows
                            else fs_full.mean(dim=0, keepdim=True))   # [n_win, L, H, T]
        frame_sum_perwin_first = (torch.stack(perwin_first_rows, dim=0) if perwin_first_rows
                                  else fs_full[:1])                   # [n_win, L, H, T]
        # per-window START-number and END-number maps (x1 / x2 analogs): mean over
        # each window's start-digit tokens and end-digit tokens respectively.
        def _perwin_rows(qidx_lists):
            rows = []
            for ql in qidx_lists:
                valid = [q for q in ql if 0 <= q < fs_full.shape[0]]
                rows.append(fs_full[valid].mean(dim=0) if valid else fs_full.mean(dim=0))
            return (torch.stack(rows, dim=0) if rows else fs_full.mean(dim=0, keepdim=True))
        frame_sum_perwin_start = _perwin_rows(perwin_start_qidx)   # [n_win, L, H, T]
        frame_sum_perwin_end   = _perwin_rows(perwin_end_qidx)     # [n_win, L, H, T]
        text_sum_slots  = slot3_mean(_EXTRACT["task_text_sum"])
        evt_sum_slots   = slot3_mean(_EXTRACT["event_text_sum"])
        sys_sum_slots   = slot3_mean(_EXTRACT["system_sum"])
        spc_sum_slots   = slot3_mean(_EXTRACT["specials_sum"])
        rsp_sum_slots   = slot3_mean(_EXTRACT["response_sum"])

        records.append({
            "qid": item.get("qid"),
            "vid": vid,
            "query": query,
            "gt_windows": [list(w) for w in gt_windows],
            "pred_windows": [list(w) for w in pred_windows],
            "response": response,
            "mean_iou": float(mean_iou),
            "min_iou": float(min_iou),
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
                # per-window frame_sum: [n_pred_windows, L, H, T], aligned with pred_windows
                "frame_sum_perwin":       frame_sum_perwin.to(torch.float16).numpy(),       # mean over window's coords
                "frame_sum_perwin_first": frame_sum_perwin_first.to(torch.float16).numpy(),  # window's first coord digit
                "frame_sum_perwin_start": frame_sum_perwin_start.to(torch.float16).numpy(),  # window's START number (x1)
                "frame_sum_perwin_end":   frame_sum_perwin_end.to(torch.float16).numpy(),    # window's END number (x2)
                "task_text_sum":  text_sum_slots.to(torch.float16).numpy(),
                "event_text_sum": evt_sum_slots .to(torch.float16).numpy(),
                "system_sum":     sys_sum_slots .to(torch.float16).numpy(),
                "specials_sum":  spc_sum_slots  .to(torch.float16).numpy(),
                "response_sum":  rsp_sum_slots  .to(torch.float16).numpy(),
            },
        })
        preds_out.append({
            "qid": item.get("qid"), "vid": vid, "query": query,
            "gt_windows": [list(w) for w in gt_windows],
            "pred_windows": [list(w) for w in pred_windows],
            "response": response,
            "mean_iou": float(mean_iou), "min_iou": float(min_iou),
            "is_correct": (not is_hallu),
        })
        n_done += 1
        if n_done % 25 == 0:
            rate = n_done / max(time.time() - t0, 1e-9)
            eta = (len(items) - cnt - 1) / max(rate, 1e-9)
            acc = n_correct / max(n_done, 1) * 100
            print(f"[worker {rank}] [{cnt+1}/{len(items)}] done={n_done} skip={n_skipped} "
                  f"R@meanIoU0.5={acc:.1f}% rate={rate:.2f}qps eta={eta/60:.1f}min", flush=True)
        for kk in list(_EXTRACT.keys()):
            if kk in ("active","lang_attn_ids","lang_attn_order"): continue
            _EXTRACT[kk] = None
        torch.cuda.empty_cache(); gc.collect()

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard{rank}.pt"
    torch.save(records, out_path)
    print(f"[worker {rank}] saved {len(records)} -> {out_path}", flush=True)
    os.makedirs(pred_dir, exist_ok=True)
    with open(f"{pred_dir}/preds_rank{rank}.json", "w") as f:
        json.dump(preds_out, f)


def main():
    global ANN_PATH, VIDEO_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--workers_per_gpu", type=int, default=1)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--pred_dir", default=PRED_DIR)
    ap.add_argument("--ann", default=ANN_PATH, help="QVHighlights val annotations jsonl")
    ap.add_argument("--video_dir", default=VIDEO_DIR, help="dir of {vid}.mp4 clips")
    ap.add_argument("--svar_shift", action="store_true",
                    help="Extract attention at predicting position (target_pos - 1) "
                         "to match SVAR's convention. Default: extract at target_pos.")
    ap.add_argument("--limit", type=int, default=999999)
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, sample (T=0.7, top-p=0.95) seeded by this value "
                         "instead of greedy decoding. Used for self-consistency rollouts.")
    args = ap.parse_args()
    ANN_PATH, VIDEO_DIR = args.ann, args.video_dir
    set_start_method("spawn", force=True)

    items = []
    with open(ANN_PATH) as f:
        for ln in f:
            items.append(json.loads(ln))
    items = items[:args.limit]
    print(f"QVHighlights val queries: {len(items)}")

    n_workers = len(args.gpus) * args.workers_per_gpu
    chunks = np.array_split(items, n_workers)
    procs = []
    rank = 0
    for gpu in args.gpus:
        for w in range(args.workers_per_gpu):
            if rank >= len(chunks) or len(chunks[rank]) == 0:
                rank += 1; continue
            p = Process(target=worker, args=(rank, gpu, list(chunks[rank]), args.out_dir, args.pred_dir, args.svar_shift, args.seed))
            p.start(); procs.append(p)
            rank += 1
    for p in procs:
        p.join()

    all_preds = []
    for r in range(n_workers):
        pp = f"{args.pred_dir}/preds_rank{r}.json"
        if os.path.exists(pp):
            all_preds.extend(json.load(open(pp)))
    with open(f"{args.pred_dir}/predictions.json", "w") as f:
        json.dump(all_preds, f)
    n_correct = sum(1 for p in all_preds if p["is_correct"])
    print(f"\nMerged: {len(all_preds)} preds, R@meanIoU0.5={n_correct/max(len(all_preds),1)*100:.2f}%")
    print("all workers complete")


if __name__ == "__main__":
    main()
