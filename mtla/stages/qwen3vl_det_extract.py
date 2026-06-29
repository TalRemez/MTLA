"""Detection attention extraction with bbox-region masking and 3 query variants.

For each predicted (label, bbox), captures (per layer × head) THREE variants of
attention rows — each reduced into the same set of stats:

  variants:
    "first" : attention row at the FIRST label token (legacy)
    "label" : MEAN of attention rows across ALL tokens of the label string
    "coord" : MEAN of attention rows across ALL tokens of the 4 bbox numbers
              (x1, y1, x2, y2)

  per-variant stats (per L,H):
    image_sum, image_max                      : full image keys
    image_inside_sum, image_inside_max        : keys whose patch overlaps bbox
    image_outside_sum, image_outside_max      : keys whose patch does NOT overlap
    task_text_sum, system_sum, specials_sum, response_sum

Patch layout: image_grid_thw = [t, h, w] with merge_size=2 -> merged grid is
(h//2, w//2). Image-pad tokens are laid out row-major in this merged grid.
We map each predicted bbox (in 0-1000 normalized coords) onto the merged grid
using "any overlap" semantics.
"""
import os, json, re, argparse, time, gc
from multiprocessing import Process, set_start_method
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

PRED_FILE     = "predictions.json"   # --pred_file
DATASET_FILE  = "coco_dataset.json"  # --dataset
OUT_DIR       = "features"           # --out_dir
MODEL_ID      = "Qwen/Qwen3-VL-8B-Instruct"

VARIANTS = ("first", "label", "coord")

_EXTRACT = {
    "active": False,
    "lang_attn_ids": set(),
    "lang_attn_order": [],
    # all unique absolute query positions (across all preds and all variants)
    "query_positions": None,         # LongTensor [N_q_unique]
    "image_indices": None,           # absolute key positions of <|image_pad|>
    "system_indices": None,
    "task_text_indices": None,
    "specials_indices": None,
    "response_indices": None,
    # per-pred specs: list of dicts, each with
    #   "first_row"  : int   (index into query_positions)
    #   "label_rows" : LongTensor (indices into query_positions)
    #   "coord_rows" : LongTensor (indices into query_positions)
    #   "inside_idx" : LongTensor (indices into image_indices)
    #   "outside_idx": LongTensor
    "pred_specs": None,
    # buf[variant][stat]: tensor [n_preds, n_layers, n_heads]
    "buf": None,
}


def _new_buf(n_preds, n_layers, n_heads):
    buf = {}
    for v in VARIANTS:
        d = {}
        d["image_sum"]         = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        d["image_max"]         = torch.full((n_preds, n_layers, n_heads), -1.0, dtype=torch.float32)
        d["image_inside_sum"]  = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        d["image_inside_max"]  = torch.full((n_preds, n_layers, n_heads), -1.0, dtype=torch.float32)
        d["image_outside_sum"] = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        d["image_outside_max"] = torch.full((n_preds, n_layers, n_heads), -1.0, dtype=torch.float32)
        d["task_text_sum"]     = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        d["mention_sum"]       = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        d["mention_max"]       = torch.full((n_preds, n_layers, n_heads), -1.0, dtype=torch.float32)
        d["system_sum"]        = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        d["specials_sum"]      = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        d["response_sum"]      = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        buf[v] = d
    # image_argmax only for "first" variant (single-token, meaningful)
    buf["first"]["image_argmax"] = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.int32)
    return buf


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
        # rows: [N_q_unique, heads, k_total]
        rows_at_q = attn_weights[0].index_select(1, q_pos).transpose(0, 1)
        rows = rows_at_q.float()
        img_idx = s["image_indices"]
        # img_rows: [N_q_unique, heads, n_img]
        img_rows = rows.index_select(2, img_idx)

        for pi, spec in enumerate(s["pred_specs"]):
            in_idx, out_idx = spec["inside_idx"], spec["outside_idx"]
            for v in VARIANTS:
                if v == "first":
                    qrow = spec["first_row"]
                    img_row = img_rows[qrow]                         # [H, n_img]
                    full_row = rows[qrow]                            # [H, k_total]
                else:
                    qrows = spec[f"{v}_rows"]
                    if qrows is None or qrows.numel() == 0:
                        continue
                    img_row = img_rows.index_select(0, qrows).mean(dim=0)
                    full_row = rows.index_select(0, qrows).mean(dim=0)

                buf_v = s["buf"][v]
                # global image
                buf_v["image_sum"][pi, layer_idx, :] += img_row.sum(dim=1).cpu()
                buf_v["image_max"][pi, layer_idx, :] = torch.maximum(
                    buf_v["image_max"][pi, layer_idx, :],
                    img_row.max(dim=1).values.cpu())
                if v == "first":
                    buf_v["image_argmax"][pi, layer_idx, :] = (
                        img_row.max(dim=1).indices.to(torch.int32).cpu())
                # bbox-region
                if in_idx.numel() > 0:
                    inside = img_row.index_select(1, in_idx)
                    buf_v["image_inside_sum"][pi, layer_idx, :] += inside.sum(dim=1).cpu()
                    buf_v["image_inside_max"][pi, layer_idx, :] = torch.maximum(
                        buf_v["image_inside_max"][pi, layer_idx, :],
                        inside.max(dim=1).values.cpu())
                if out_idx.numel() > 0:
                    outside = img_row.index_select(1, out_idx)
                    buf_v["image_outside_sum"][pi, layer_idx, :] += outside.sum(dim=1).cpu()
                    buf_v["image_outside_max"][pi, layer_idx, :] = torch.maximum(
                        buf_v["image_outside_max"][pi, layer_idx, :],
                        outside.max(dim=1).values.cpu())
                # other key groups
                for kn in ("system", "task_text", "specials", "response"):
                    kidx = s[f"{kn}_indices"]
                    if kidx is not None and kidx.numel() > 0:
                        buf_v[f"{kn}_sum"][pi, layer_idx, :] += full_row.index_select(1, kidx).sum(dim=1).cpu()
                # per-prediction mention attention (predicted-label tokens in the prompt)
                m_idx = spec.get("mention_idx", None)
                if m_idx is not None and m_idx.numel() > 0:
                    m_row = full_row.index_select(1, m_idx)  # [H, n_mention]
                    buf_v["mention_sum"][pi, layer_idx, :] += m_row.sum(dim=1).cpu()
                    buf_v["mention_max"][pi, layer_idx, :] = torch.maximum(
                        buf_v["mention_max"][pi, layer_idx, :],
                        m_row.max(dim=1).values.cpu())

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, None


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def label_hallu(pred_box, pred_label, gt_objs):
    pl = pred_label.strip().lower()
    for go in gt_objs:
        if str(go.get("label", "")).strip().lower() == pl:
            if iou(pred_box, go["bbox_2d"]) >= 0.5:
                return False
    return True


def find_pred_token_ranges(response_text, pred_bboxes, tokenizer):
    """For each prediction, find token indices for:
      - first_label_tok : first token of the label string (legacy)
      - label_toks      : ALL tokens overlapping the label string
      - coord_toks      : ALL tokens overlapping the 4 bbox numbers
    Returns a list aligned with pred_bboxes; entries are dicts or None.
    """
    enc = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out = []
    search_pos = 0
    full_pat_template = (
        r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,'
        r'\s*"label"\s*:\s*"{label}"'
    )
    for pb in pred_bboxes:
        label = pb["label"]
        full_pat = re.compile(full_pat_template.format(label=re.escape(label)))
        m = full_pat.search(response_text, search_pos)
        if not m:
            m = full_pat.search(response_text)
        coord_ranges = []
        if m:
            coord_ranges = [(m.start(g), m.end(g)) for g in range(1, 5)]
            label_start = m.end() - 1 - len(label)
            label_end   = m.end() - 1
            search_pos  = m.end()
        else:
            # fallback: locate just the label
            label_pat = re.compile(r'"label"\s*:\s*"' + re.escape(label) + r'"')
            ml = label_pat.search(response_text, search_pos)
            if not ml:
                ml = label_pat.search(response_text)
                if not ml:
                    out.append(None); continue
            marker = re.search(r'"label"\s*:\s*"', ml.group(0)).group(0)
            label_start = ml.start() + len(marker)
            label_end   = label_start + len(label)
            search_pos  = ml.end()
        # collect tokens overlapping label range
        label_toks = []
        for ti, (ts, te) in enumerate(offsets):
            if ts < label_end and te > label_start:
                label_toks.append(ti)
        first_label_tok = None
        for ti in label_toks:
            ts, te = offsets[ti]
            if ts >= label_start:
                first_label_tok = ti; break
        if first_label_tok is None and label_toks:
            first_label_tok = label_toks[0]
        # coord tokens
        coord_toks = []
        for (cs, ce) in coord_ranges:
            for ti, (ts, te) in enumerate(offsets):
                if ts < ce and te > cs:
                    coord_toks.append(ti)
        out.append({
            "first_label_tok": first_label_tok,
            "label_toks": label_toks,
            "coord_toks": coord_toks,
            "char_start": label_start,
        })
    return out


def find_token_subseq_positions(needle_ids, haystack_ids):
    n_n = len(needle_ids); n_h = len(haystack_ids)
    out = []; i = 0
    while i + n_n <= n_h:
        if haystack_ids[i:i+n_n] == needle_ids:
            out.append(i); i += n_n
        else:
            i += 1
    return out


def find_label_token_positions(label, tokenizer, prompt_token_ids, user_turn_start, end_pos):
    """Find absolute positions in the prompt covering the predicted label string.
    The detection prompt lists 80 categories comma-separated; each label appears
    typically once. We try several tokenizations to handle BPE leading-space
    quirks. Restricted to [user_turn_start, end_pos)."""
    haystack = list(prompt_token_ids)
    candidates = [
        tokenizer(" " + label, add_special_tokens=False)["input_ids"],
        tokenizer(label,        add_special_tokens=False)["input_ids"],
    ]
    seen = set(); out = []
    for cand in candidates:
        if not cand: continue
        for s in find_token_subseq_positions(cand, haystack):
            if s < user_turn_start or s + len(cand) > end_pos: continue
            for k in range(len(cand)):
                pos = s + k
                if pos not in seen:
                    seen.add(pos); out.append(pos)
    return sorted(out)


def bbox_to_patch_indices(bbox, grid_h, grid_w):
    """Return (inside_idx, outside_idx) — indices INTO the n_img=grid_h*grid_w
    array of image keys, in row-major order. 'any-overlap' semantics: a patch
    counts as inside if it shares any area with the bbox in 0-1000 normalized
    coordinates.

    bbox: [x1, y1, x2, y2] in 0-1000 normalized
    """
    x1, y1, x2, y2 = bbox
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    col_min = int(np.floor(x1 * grid_w / 1000.0))
    col_max = int(np.floor((x2 - 1e-6) * grid_w / 1000.0))
    row_min = int(np.floor(y1 * grid_h / 1000.0))
    row_max = int(np.floor((y2 - 1e-6) * grid_h / 1000.0))
    col_min = max(0, min(grid_w - 1, col_min))
    col_max = max(0, min(grid_w - 1, col_max))
    row_min = max(0, min(grid_h - 1, row_min))
    row_max = max(0, min(grid_h - 1, row_max))
    inside = []
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            inside.append(r * grid_w + c)
    inside_set = set(inside)
    outside = [i for i in range(grid_h * grid_w) if i not in inside_set]
    return inside, outside


def worker(rank, gpu_id, image_indices, out_dir, svar_shift=False, pred_file=PRED_FILE,
           dataset_file=DATASET_FILE):
    print(f"[worker {rank}] gpu={gpu_id} n={len(image_indices)} svar_shift={svar_shift}", flush=True)
    torch.cuda.set_device(gpu_id)

    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_mod
    qwen3_mod.eager_attention_forward = patched_eager_attention_forward

    print(f"[worker {rank}] loading model on cuda:{gpu_id}", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager",
        device_map=f"cuda:{gpu_id}",
    ).eval()
    decoder_layers = model.model.language_model.layers
    n_layers = len(decoder_layers)
    n_heads  = model.config.text_config.num_attention_heads

    _EXTRACT["lang_attn_ids"]   = {id(L.self_attn) for L in decoder_layers}
    _EXTRACT["lang_attn_order"] = [id(L.self_attn) for L in decoder_layers]

    img_pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    special_token_strings = [
        "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|endoftext|>",
    ]
    specials_ids = set()
    for s in special_token_strings:
        tid = proc.tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0: specials_ids.add(tid)
    for tid in (proc.tokenizer.all_special_ids or []):
        if tid != img_pad_id: specials_ids.add(int(tid))

    preds_all = json.load(open(pred_file))
    ds_all    = json.load(open(dataset_file))
    ds_by_id  = {d["id"]: d for d in ds_all}

    device = f"cuda:{gpu_id}"
    records = []
    t0 = time.time()
    n_done = n_skipped = 0

    for cnt, idx in enumerate(image_indices):
        p = preds_all[idx]
        if p.get("status") != "success" or not p.get("pred_bboxes"):
            n_skipped += 1; continue
        ds_item = ds_by_id.get(p["id"])
        if ds_item is None: n_skipped += 1; continue

        gt_objs = json.loads(ds_item["conversations"][1]["value"])
        hallu_flags = [label_hallu(pb["box"], pb["label"], gt_objs) for pb in p["pred_bboxes"]]

        prompt_text = ds_item["conversations"][0]["value"].replace("<image>\n", "")
        try:
            img = Image.open(ds_item["image"]).convert("RGB")
        except Exception as e:
            print(f"[worker {rank}] skip {p['id']}: img {e}", flush=True); n_skipped += 1; continue
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt_text},
        ]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0]
        prompt_len = prompt_ids.shape[0]
        if "image_grid_thw" in inputs:
            t_, h_, w_ = inputs["image_grid_thw"][0].tolist()
            grid_h, grid_w = h_ // 2, w_ // 2
        else:
            n_skipped += 1; continue
        n_img_keys = grid_h * grid_w

        response = p["response"]
        resp_enc = proc.tokenizer(response, add_special_tokens=False)
        resp_ids = torch.tensor(resp_enc["input_ids"], dtype=prompt_ids.dtype, device=device)
        full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
        total_len = full_ids.shape[1]

        token_ranges = find_pred_token_ranges(response, p["pred_bboxes"], proc.tokenizer)
        # SVAR-faithful shift: extract attention at the predicting position (target_pos - 1)
        # rather than at the target token. Applied to all query positions (first / label / coord).
        def _shift(pos):
            return max(0, pos - 1) if svar_shift else pos
        # Build per-pred specs (only valid preds)
        valid_pred_idx = []          # original pred index
        first_q_abs = []             # absolute position of first label tok per pred
        label_q_abs_list = []        # list of LongTensor (absolute positions)
        coord_q_abs_list = []
        char_starts = []
        for i, tr in enumerate(token_ranges):
            if tr is None: continue
            ftt = tr["first_label_tok"]
            if ftt is None: continue
            first_abs = _shift(prompt_len + ftt)
            if first_abs >= total_len: continue
            label_abs = [_shift(prompt_len + t) for t in tr["label_toks"] if (prompt_len + t) < total_len]
            coord_abs = [_shift(prompt_len + t) for t in tr["coord_toks"] if (prompt_len + t) < total_len]
            if not label_abs:
                label_abs = [first_abs]
            valid_pred_idx.append(i)
            first_q_abs.append(first_abs)
            label_q_abs_list.append(label_abs)
            coord_q_abs_list.append(coord_abs)
            char_starts.append(tr["char_start"])
        if not valid_pred_idx:
            n_skipped += 1; continue

        # key-group classification
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
        image_idx_l, system_idx_l, task_idx_l, spc_idx_l = [], [], [], []
        for k, tid in enumerate(prompt_cpu):
            if tid == img_pad_id:                        image_idx_l.append(k)
            elif tid in specials_ids:                    spc_idx_l.append(k)
            elif assistant_start is not None and k >= assistant_start: spc_idx_l.append(k)
            elif k >= user_turn_start:                   task_idx_l.append(k)
            else:                                        system_idx_l.append(k)
        resp_idx_l = list(range(prompt_len, total_len))
        device_idx = lambda L: torch.tensor(L, dtype=torch.long, device=device)
        _EXTRACT["image_indices"]    = device_idx(image_idx_l)
        _EXTRACT["system_indices"]   = device_idx(system_idx_l)
        _EXTRACT["task_text_indices"]= device_idx(task_idx_l)
        _EXTRACT["specials_indices"] = device_idx(spc_idx_l)
        _EXTRACT["response_indices"] = device_idx(resp_idx_l)

        if len(image_idx_l) != n_img_keys:
            print(f"[worker {rank}] WARN {p['id']}: image_idx_l={len(image_idx_l)} != grid_h*grid_w={n_img_keys}",
                  flush=True)
            n_skipped += 1; continue

        # Build unique sorted query positions across all preds and variants
        all_positions = set()
        for fa, la, ca in zip(first_q_abs, label_q_abs_list, coord_q_abs_list):
            all_positions.add(fa)
            for x in la: all_positions.add(x)
            for x in ca: all_positions.add(x)
        sorted_positions = sorted(all_positions)
        pos_to_qrow = {pos: r for r, pos in enumerate(sorted_positions)}
        _EXTRACT["query_positions"] = torch.tensor(sorted_positions, dtype=torch.long, device=device)

        # per-pred specs
        n_preds = len(valid_pred_idx)
        pred_specs = []
        # End of user-turn = assistant_start (or end of prompt if no assistant marker)
        end_pos = assistant_start if assistant_start is not None else len(prompt_cpu)
        # cache mention positions per unique label (avoid redundant search)
        mention_cache = {}
        n_no_mention = 0
        for i, orig_i in enumerate(valid_pred_idx):
            pb = p["pred_bboxes"][orig_i]
            bbox = pb["box"]
            label = pb["label"]
            inside_l, outside_l = bbox_to_patch_indices(bbox, grid_h, grid_w)
            if label not in mention_cache:
                mention_cache[label] = find_label_token_positions(
                    label, proc.tokenizer, prompt_cpu, user_turn_start, end_pos)
            mention_pos = mention_cache[label]
            if not mention_pos:
                n_no_mention += 1
            pred_specs.append({
                "first_row": pos_to_qrow[first_q_abs[i]],
                "label_rows": torch.tensor([pos_to_qrow[x] for x in label_q_abs_list[i]],
                                            dtype=torch.long, device=device),
                "coord_rows": torch.tensor([pos_to_qrow[x] for x in coord_q_abs_list[i]],
                                            dtype=torch.long, device=device),
                "inside_idx":  torch.tensor(inside_l,  dtype=torch.long, device=device),
                "outside_idx": torch.tensor(outside_l, dtype=torch.long, device=device),
                "mention_idx": torch.tensor(mention_pos, dtype=torch.long, device=device),
                "n_mention_positions": len(mention_pos),
                "label": label,
            })
        if rank == 0 and n_done == 0:
            # log first image's mentions for sanity
            for i, spec in enumerate(pred_specs[:3]):
                if spec["mention_idx"].numel() > 0:
                    decoded = proc.tokenizer.decode([prompt_cpu[p2] for p2 in spec["mention_idx"].cpu().tolist()],
                                                    skip_special_tokens=False)
                    print(f"[worker {rank}] sanity: pred {i} label={spec['label']!r}  n_mention={spec['n_mention_positions']}  decoded={decoded!r}", flush=True)
                else:
                    print(f"[worker {rank}] sanity: pred {i} label={spec['label']!r}  NO MENTION", flush=True)
        if n_no_mention > 0 and rank == 0:
            print(f"[worker {rank}] WARN: {n_no_mention}/{n_preds} preds had no mention match for their label", flush=True)
        _EXTRACT["pred_specs"] = pred_specs
        _EXTRACT["buf"] = _new_buf(n_preds, n_layers, n_heads)

        fk = {
            "input_ids": full_ids,
            "attention_mask": torch.ones(1, total_len, device=device, dtype=torch.long),
        }
        for k in ["pixel_values", "image_grid_thw"]:
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
            print(f"[worker {rank}] skip {p['id']}: forward {e}", flush=True)
            n_skipped += 1; torch.cuda.empty_cache(); continue
        _EXTRACT["active"] = False

        out_objs = []
        for i, orig_i in enumerate(valid_pred_idx):
            spec = pred_specs[i]
            attn_blocks = {}
            for v in VARIANTS:
                d = _EXTRACT["buf"][v]
                blk = {
                    "image_sum":         d["image_sum"][i].to(torch.float16).numpy(),
                    "image_max":         d["image_max"][i].to(torch.float16).numpy(),
                    "image_inside_sum":  d["image_inside_sum"][i].to(torch.float16).numpy(),
                    "image_inside_max":  d["image_inside_max"][i].to(torch.float16).numpy(),
                    "image_outside_sum": d["image_outside_sum"][i].to(torch.float16).numpy(),
                    "image_outside_max": d["image_outside_max"][i].to(torch.float16).numpy(),
                    "task_text_sum":     d["task_text_sum"][i].to(torch.float16).numpy(),
                    "mention_sum":       d["mention_sum"][i].to(torch.float16).numpy(),
                    "mention_max":       d["mention_max"][i].to(torch.float16).numpy(),
                    "specials_sum":      d["specials_sum"][i].to(torch.float16).numpy(),
                    "response_sum":      d["response_sum"][i].to(torch.float16).numpy(),
                    "system_sum":        d["system_sum"][i].to(torch.float16).numpy(),
                }
                if v == "first":
                    blk["image_argmax"] = d["image_argmax"][i].numpy()
                attn_blocks[v] = blk
            pb = p["pred_bboxes"][orig_i]
            out_objs.append({
                "pred_idx": orig_i,
                "label": pb["label"],
                "box": pb["box"],
                "is_hallucinated": bool(hallu_flags[orig_i]),
                "abs_query_pos": first_q_abs[i],
                "n_label_toks": int(spec["label_rows"].numel()),
                "n_coord_toks": int(spec["coord_rows"].numel()),
                "char_start": char_starts[i],
                "n_inside_patches": int(spec["inside_idx"].numel()),
                "n_mention_positions": int(spec["mention_idx"].numel()),
                "grid_h": int(grid_h),
                "grid_w": int(grid_w),
                # Legacy single-block name kept as the "first" variant for back-compat
                "attn":            attn_blocks["first"],
                "attn_label_mean": attn_blocks["label"],
                "attn_coord_mean": attn_blocks["coord"],
            })
        records.append({
            "image_id": p["id"],
            "n_pred_bboxes": len(p["pred_bboxes"]),
            "n_extracted": len(out_objs),
            "grid_hw": (int(grid_h), int(grid_w)),
            "objects": out_objs,
        })
        n_done += 1
        if n_done % 5 == 0:
            rate = n_done / max(time.time() - t0, 1e-9)
            eta = (len(image_indices) - cnt - 1) / max(rate, 1e-9)
            print(f"[worker {rank}] [{cnt+1}/{len(image_indices)}] done={n_done} skip={n_skipped} "
                  f"rate={rate:.2f}img/s eta={eta/60:.1f}min", flush=True)
        for kk in list(_EXTRACT.keys()):
            if kk in ("active","lang_attn_ids","lang_attn_order"): continue
            _EXTRACT[kk] = None
        torch.cuda.empty_cache(); gc.collect()

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard{rank}.pt"
    torch.save(records, out_path)
    print(f"[worker {rank}] saved {len(records)} records / {sum(len(r['objects']) for r in records)} preds -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n_images", type=int, default=5000)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--pred_file", default=PRED_FILE)
    ap.add_argument("--dataset", default=DATASET_FILE, help="COCO openvocab dataset json")
    ap.add_argument("--svar_shift", action="store_true",
                    help="Extract attention at predicting position (target_pos - 1) "
                         "to match SVAR's convention. Default: extract at target_pos.")
    args = ap.parse_args()
    set_start_method("spawn", force=True)
    image_indices = list(range(args.n_images))
    chunks = np.array_split(image_indices, len(args.gpus))
    procs = []
    for rank, (gpu, chunk) in enumerate(zip(args.gpus, chunks)):
        if len(chunk) == 0: continue
        p = Process(target=worker, args=(rank, gpu, list(chunk), args.out_dir, args.svar_shift,
                                          args.pred_file, args.dataset))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    print("all workers complete")


if __name__ == "__main__":
    main()
