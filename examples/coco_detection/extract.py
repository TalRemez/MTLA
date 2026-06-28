"""InternVL3.5-8B Detection attention extraction (COCO val5k T=0.7).

Adapted from extract_attn_det_bbox_t07_v2.py but for InternVL3.5-8B.

Key differences:
  - Model: InternVLChatModel (Qwen3-8B LLM + InternViT-300M vision)
  - Image preprocessing: dynamic tiling (variable n_tiles + optional thumbnail)
  - Image token: <IMG_CONTEXT> (id 151671)
  - Per-tile grid: 16x16 = 256 patches per tile (after pixel_shuffle 0.5)
  - LLM attention layers: transformers.models.qwen3.modeling_qwen3
  - Total image tokens = n_tiles * 256 + (256 if thumbnail else 0)

bbox-to-patch mapping for InternVL:
  We map a bbox in [0, 1000] normalized coords to:
    - For each tile: which 16x16 patches inside this tile overlap the bbox?
    - For the thumbnail: which 16x16 patches overlap the bbox?
  The image-token sequence has tile1[0..255], tile2[0..255], ..., thumbnail[0..255].
  We compute global indices into this concatenated sequence.

Variants per prediction (slot in {first, label, coord, first_digit}):
  image_sum, image_max, image_inside_sum, image_inside_max, image_outside_sum,
  image_outside_max, task_text_sum, mention_sum, mention_max, system_sum,
  specials_sum, response_sum.

VARIANTS = ("first", "label", "coord", "first_digit")
"""
import os, json, re, argparse, time, gc, sys
from multiprocessing import Process, set_start_method
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# transformers v5 compat: InternVL's modeling code expects attribute that v5 doesn't define
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

from transformers.models.qwen3.modeling_qwen3 import repeat_kv

# Paths are provided on the command line (see --help). Defaults are placeholders.
PRED_FILE     = "predictions.json"   # --pred_file : output of generate.py
DATASET_FILE  = "coco_dataset.json"  # --dataset   : COCO openvocab dataset json (see docs/DATA.md)
OUT_DIR       = "features"           # --out_dir   : where attention shards are written
MODEL_ID      = "OpenGVLab/InternVL3_5-8B"
IMG_CONTEXT_TOK = "<IMG_CONTEXT>"

VARIANTS = ("first", "label", "coord", "first_digit")
PATCH_GRID = 16  # per-tile patch grid after pixel_shuffle 0.5
NUM_IMAGE_TOKEN_PER_TILE = PATCH_GRID * PATCH_GRID  # 256


# ---------- InternVL image preprocessing ----------
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
    """Returns: list of PIL tile images (in row-major tile order), tile_grid (n_cols, n_rows),
    use_thumbnail flag."""
    ow, oh = image.size; ar = ow / oh
    target_ratios = sorted({(i, j) for n in range(min_num, max_num+1) for i in range(1, n+1) for j in range(1, n+1)
                            if min_num <= i*j <= max_num},
                           key=lambda x: x[0]*x[1])
    target = _find_closest_aspect_ratio(ar, target_ratios, ow, oh, image_size)
    n_cols, n_rows = target  # tile grid
    tw, th = image_size * n_cols, image_size * n_rows
    blocks = n_cols * n_rows
    img = image.resize((tw, th))
    images = []
    for i in range(blocks):
        c = i % n_cols; r = i // n_cols
        box = (c * image_size, r * image_size, (c + 1) * image_size, (r + 1) * image_size)
        images.append(img.crop(box))
    has_thumb = use_thumbnail and len(images) != 1
    if has_thumb:
        images.append(image.resize((image_size, image_size)))
    return images, (n_cols, n_rows), has_thumb


def load_image_internvl(path, input_size=448, max_num=12):
    image = Image.open(path).convert('RGB')
    transform = _build_transform(input_size=input_size)
    tiles, tile_grid, has_thumb = _dynamic_preprocess(image, image_size=input_size,
                                                     use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in tiles])
    return pixel_values, tile_grid, has_thumb


def bbox_to_internvl_token_indices(bbox, tile_grid, has_thumb):
    """Map bbox in [0, 1000] normalized coords to image-token indices.

    Image-token sequence layout (when n_tiles = n_cols*n_rows >= 1):
      tile_0_tokens[0..255], tile_1_tokens[0..255], ..., tile_{N-1}_tokens[0..255]
      [thumbnail_tokens[0..255]] (if has_thumb and n_tiles != 1)

    Each tile contributes 16x16 = 256 patches in row-major order.
    Tiles tile the image in row-major order: tile (col, row) covers the rectangle
    [col/n_cols, row/n_rows, (col+1)/n_cols, (row+1)/n_rows] of the image.
    """
    x1, y1, x2, y2 = bbox
    n_cols, n_rows = tile_grid
    n_tiles = n_cols * n_rows
    total_tokens = n_tiles * NUM_IMAGE_TOKEN_PER_TILE + (NUM_IMAGE_TOKEN_PER_TILE if has_thumb else 0)
    if x2 <= x1 or y2 <= y1:
        return [], list(range(total_tokens))
    # Normalized [0, 1] bbox
    bx1, by1 = x1 / 1000.0, y1 / 1000.0
    bx2, by2 = x2 / 1000.0, y2 / 1000.0
    inside = []
    # For each tile (col, row): patches inside are those whose 16x16 cell overlaps the bbox.
    # Tile (col, row) covers image x ∈ [col/n_cols, (col+1)/n_cols], y ∈ [row/n_rows, (row+1)/n_rows].
    # Within the tile, patch (pc, pr) covers tile-local x ∈ [pc/16, (pc+1)/16], etc.
    # Convert: the patch in image coords covers
    #   x ∈ [(col + pc/16)/n_cols, (col + (pc+1)/16)/n_cols]
    for tile_idx in range(n_tiles):
        col = tile_idx % n_cols
        row = tile_idx // n_cols
        # Tile bounds in image coords
        tile_x0 = col / n_cols
        tile_x1 = (col + 1) / n_cols
        tile_y0 = row / n_rows
        tile_y1 = (row + 1) / n_rows
        # Tile must overlap bbox at all
        if tile_x1 <= bx1 or tile_x0 >= bx2 or tile_y1 <= by1 or tile_y0 >= by2:
            continue
        # Within tile, find patches overlapping bbox
        # Map bbox to tile-local coords [0, 1]
        local_x0 = max(0.0, (bx1 - tile_x0) / (tile_x1 - tile_x0))
        local_x1 = min(1.0, (bx2 - tile_x0) / (tile_x1 - tile_x0))
        local_y0 = max(0.0, (by1 - tile_y0) / (tile_y1 - tile_y0))
        local_y1 = min(1.0, (by2 - tile_y0) / (tile_y1 - tile_y0))
        # Patch grid is 16x16
        col_min = int(np.floor(local_x0 * PATCH_GRID))
        col_max = int(np.floor((local_x1 - 1e-6) * PATCH_GRID))
        row_min = int(np.floor(local_y0 * PATCH_GRID))
        row_max = int(np.floor((local_y1 - 1e-6) * PATCH_GRID))
        col_min = max(0, min(PATCH_GRID - 1, col_min))
        col_max = max(0, min(PATCH_GRID - 1, col_max))
        row_min = max(0, min(PATCH_GRID - 1, row_min))
        row_max = max(0, min(PATCH_GRID - 1, row_max))
        tile_token_offset = tile_idx * NUM_IMAGE_TOKEN_PER_TILE
        for pr in range(row_min, row_max + 1):
            for pc in range(col_min, col_max + 1):
                inside.append(tile_token_offset + pr * PATCH_GRID + pc)
    # Thumbnail: full image at 16x16 resolution
    if has_thumb:
        col_min = int(np.floor(bx1 * PATCH_GRID))
        col_max = int(np.floor((bx2 - 1e-6) * PATCH_GRID))
        row_min = int(np.floor(by1 * PATCH_GRID))
        row_max = int(np.floor((by2 - 1e-6) * PATCH_GRID))
        col_min = max(0, min(PATCH_GRID - 1, col_min))
        col_max = max(0, min(PATCH_GRID - 1, col_max))
        row_min = max(0, min(PATCH_GRID - 1, row_min))
        row_max = max(0, min(PATCH_GRID - 1, row_max))
        thumb_offset = n_tiles * NUM_IMAGE_TOKEN_PER_TILE
        for pr in range(row_min, row_max + 1):
            for pc in range(col_min, col_max + 1):
                inside.append(thumb_offset + pr * PATCH_GRID + pc)
    inside_set = set(inside)
    outside = [i for i in range(total_tokens) if i not in inside_set]
    return inside, outside


# ---------- IoU & hallucination flag (same as v2) ----------
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
            if iou(pred_box, go["bbox_2d"]) >= 0.5: return False
    return True


# ---------- _EXTRACT state and patched attention hook ----------
_EXTRACT = {
    "active": False,
    "lang_attn_ids": set(),
    "lang_attn_order": [],
    "query_positions": None,
    "image_indices": None,
    "system_indices": None,
    "task_text_indices": None,
    "specials_indices": None,
    "response_indices": None,
    "pred_specs": None,
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
        rows_at_q = attn_weights[0].index_select(1, q_pos).transpose(0, 1)
        rows = rows_at_q.float()
        img_idx = s["image_indices"]
        img_rows = rows.index_select(2, img_idx)

        for pi, spec in enumerate(s["pred_specs"]):
            in_idx, out_idx = spec["inside_idx"], spec["outside_idx"]
            for v in VARIANTS:
                if v == "first":
                    qrow = spec["first_row"]
                    img_row = img_rows[qrow]; full_row = rows[qrow]
                elif v == "first_digit":
                    qrow = spec.get("first_digit_row")
                    if qrow is None: continue
                    img_row = img_rows[qrow]; full_row = rows[qrow]
                else:
                    qrows = spec[f"{v}_rows"]
                    if qrows is None or qrows.numel() == 0: continue
                    img_row = img_rows.index_select(0, qrows).mean(dim=0)
                    full_row = rows.index_select(0, qrows).mean(dim=0)

                buf_v = s["buf"][v]
                buf_v["image_sum"][pi, layer_idx, :] += img_row.sum(dim=1).cpu()
                buf_v["image_max"][pi, layer_idx, :] = torch.maximum(
                    buf_v["image_max"][pi, layer_idx, :], img_row.max(dim=1).values.cpu())
                if v == "first":
                    buf_v["image_argmax"][pi, layer_idx, :] = (
                        img_row.max(dim=1).indices.to(torch.int32).cpu())
                if in_idx.numel() > 0:
                    inside = img_row.index_select(1, in_idx)
                    buf_v["image_inside_sum"][pi, layer_idx, :] += inside.sum(dim=1).cpu()
                    buf_v["image_inside_max"][pi, layer_idx, :] = torch.maximum(
                        buf_v["image_inside_max"][pi, layer_idx, :], inside.max(dim=1).values.cpu())
                if out_idx.numel() > 0:
                    outside = img_row.index_select(1, out_idx)
                    buf_v["image_outside_sum"][pi, layer_idx, :] += outside.sum(dim=1).cpu()
                    buf_v["image_outside_max"][pi, layer_idx, :] = torch.maximum(
                        buf_v["image_outside_max"][pi, layer_idx, :], outside.max(dim=1).values.cpu())
                for kn in ("system", "task_text", "specials", "response"):
                    kidx = s[f"{kn}_indices"]
                    if kidx is not None and kidx.numel() > 0:
                        buf_v[f"{kn}_sum"][pi, layer_idx, :] += full_row.index_select(1, kidx).sum(dim=1).cpu()
                m_idx = spec.get("mention_idx", None)
                if m_idx is not None and m_idx.numel() > 0:
                    m_row = full_row.index_select(1, m_idx)
                    buf_v["mention_sum"][pi, layer_idx, :] += m_row.sum(dim=1).cpu()
                    buf_v["mention_max"][pi, layer_idx, :] = torch.maximum(
                        buf_v["mention_max"][pi, layer_idx, :], m_row.max(dim=1).values.cpu())

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, None


# ---------- find_pred_token_ranges (InternVL output format: native grounding) ----------
def find_pred_token_ranges(response_text, pred_bboxes, tokenizer):
    """Find token positions of label and coord tokens in InternVL native grounding output.

    Format: '<label>[[x1,y1,x2,y2], [x3,y3,x4,y4]]<next_label>[[...]]...'
    For each pred bbox we find:
      - first_label_tok: first token of the label string
      - label_toks: all tokens of the label string
      - coord_toks: tokens of the 4 numbers (x1,y1,x2,y2)
    """
    enc = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    out = []
    for pb in pred_bboxes:
        label = pb["label"]
        box = pb["box"]
        # Build a precise pattern: the 4-coord bbox preceded by anything within label's block
        # Strategy: locate the label's block (label[[ ... ]]), then within the block find this specific bbox.
        # Multiple bboxes per label: scan all '[a, b, c, d]' inside the label block.
        # Easier: regex for `\[<x1>,<y1>,<x2>,<y2>\]` AND find which label block it falls in.
        bbox_pat = re.compile(rf'\[\s*{box[0]}\s*,\s*{box[1]}\s*,\s*{box[2]}\s*,\s*{box[3]}\s*\]')
        m = bbox_pat.search(response_text)
        if not m:
            out.append(None); continue
        # Coord token indices
        # Find the 4 numbers within the match span
        coord_toks = []
        # The bbox group: 4 consecutive numbers
        for num_m in re.finditer(r'\d+', m.group(0)):
            num_start = m.start() + num_m.start()
            num_end = m.start() + num_m.end()
            for ti, (ts, te) in enumerate(offsets):
                if ts < num_end and te > num_start:
                    coord_toks.append(ti)
        # Now find the label block: walk back from m.start() to find the most recent `]<word>[[` or just `<word>[[`.
        # Strategy: search backward for the most recent occurrence of `<word>[[` where the [[ is followed by digits.
        # Simpler: find all label[[ patterns in the response, and pick the one whose [[ ... ]] span contains m.
        label_block_pat = re.compile(
            rf'(?:^|<ref>|[\s\]\)\.])({re.escape(label)})(?:</ref>\s*<box>)?\s*\[\[',
            flags=re.IGNORECASE,
        )
        label_pos = None
        # iterate over candidates, pick the latest one whose start is before m.start()
        for lm in label_block_pat.finditer(response_text):
            if lm.start() <= m.start():
                label_pos = lm
            else:
                break
        if label_pos is None:
            # Fallback: any occurrence of the bare label string before m.start()
            simple_pat = re.compile(rf'\b{re.escape(label)}\b', flags=re.IGNORECASE)
            cand = None
            for lm in simple_pat.finditer(response_text):
                if lm.start() <= m.start(): cand = lm
                else: break
            label_pos = cand
        if label_pos is None:
            out.append(None); continue
        # Locate label tokens
        # Find the actual span of the label name within label_pos's match
        label_match = re.search(re.escape(label), label_pos.group(0), flags=re.IGNORECASE)
        if label_match is None:
            out.append(None); continue
        label_start = label_pos.start() + label_match.start()
        label_end = label_pos.start() + label_match.end()
        label_toks = []
        for ti, (ts, te) in enumerate(offsets):
            if ts < label_end and te > label_start:
                label_toks.append(ti)
        if not label_toks:
            out.append(None); continue
        first_label_tok = label_toks[0]
        out.append({
            "first_label_tok": first_label_tok,
            "label_toks": label_toks,
            "coord_toks": coord_toks,
        })
    return out


# ---------- find_label_token_positions (mention sum) ----------
def find_label_token_positions(label, tokenizer, prompt_token_ids, user_turn_start, end_pos):
    """Find positions of `label` substring in the prompt's user turn."""
    # Decode the user turn back to text
    user_token_ids = prompt_token_ids[user_turn_start:end_pos]
    user_text = tokenizer.decode(user_token_ids, skip_special_tokens=False)
    # Find all occurrences of the label (case-insensitive)
    pat = re.compile(re.escape(label), flags=re.IGNORECASE)
    matches = list(pat.finditer(user_text))
    if not matches:
        return []
    # Tokenize again with offsets to map char positions to token positions
    enc = tokenizer(user_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    positions = []
    for m in matches:
        for ti, (ts, te) in enumerate(offsets):
            if ts < m.end() and te > m.start():
                positions.append(user_turn_start + ti)
    return sorted(set(positions))


# ---------- Worker ----------
def worker(rank, gpu_id, image_indices, out_dir, svar_shift=False, pred_file=PRED_FILE,
           dataset_file=DATASET_FILE):
    print(f"[worker {rank}] gpu={gpu_id} n={len(image_indices)} svar_shift={svar_shift}", flush=True)
    torch.cuda.set_device(gpu_id)

    import transformers.models.qwen3.modeling_qwen3 as q3_mod
    q3_mod.eager_attention_forward = patched_eager_attention_forward

    print(f"[worker {rank}] loading model on cuda:{gpu_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(f"cuda:{gpu_id}").eval()
    # Set img_context_token_id (required by InternVL.forward to find image-token positions)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOK)
    # Force eager attention on the LLM (InternVL doesn't expose attn_implementation arg)
    # The patched eager_attention_forward will fire only if LLM uses eager attention.
    # InternVL uses sdpa by default; force eager:
    for layer in model.language_model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    decoder_layers = model.language_model.model.layers
    n_layers = len(decoder_layers)
    n_heads  = model.language_model.config.num_attention_heads

    _EXTRACT["lang_attn_ids"]   = {id(L.self_attn) for L in decoder_layers}
    _EXTRACT["lang_attn_order"] = [id(L.self_attn) for L in decoder_layers]

    img_pad_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOK)
    img_start_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_id = tokenizer.convert_tokens_to_ids("</img>")
    special_token_strings = ["<|im_start|>", "<|im_end|>", "<img>", "</img>", "<IMG_CONTEXT>",
                              "<quad>", "</quad>", "<ref>", "</ref>", "<box>", "</box>", "<|endoftext|>"]
    specials_ids = set()
    for s in special_token_strings:
        tid = tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0: specials_ids.add(tid)
    for tid in (tokenizer.all_special_ids or []):
        if tid != img_pad_id: specials_ids.add(int(tid))

    preds_all = json.load(open(pred_file))
    ds_all = json.load(open(dataset_file))
    ds_by_id = {d["id"]: d for d in ds_all}

    PROMPT_TMPL = (
        "Please detect all instances of {cats} in the image. "
        "Output the bounding boxes in the format <ref>category</ref><box>[[x1, y1, x2, y2], ...]</box> "
        "with coordinates normalized to [0, 1000]."
    )

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

        try:
            pixel_values, tile_grid, has_thumb = load_image_internvl(ds_item["image"])
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"[worker {rank}] skip {p['id']}: image load {e}", flush=True)
            n_skipped += 1; continue

        n_tiles = tile_grid[0] * tile_grid[1]
        n_image_tokens = n_tiles * NUM_IMAGE_TOKEN_PER_TILE + (NUM_IMAGE_TOKEN_PER_TILE if has_thumb else 0)

        cats = ", ".join(p["categories"])
        user_text = "<image>\n" + PROMPT_TMPL.format(cats=cats)
        # InternVL prompt template (Qwen-style chat)
        chat = tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                             tokenize=False, add_generation_prompt=True)
        # InternVL replaces <image> with <img><IMG_CONTEXT>×n_tokens</img>
        image_token_block = "<img>" + (IMG_CONTEXT_TOK * n_image_tokens) + "</img>"
        prompt_text = chat.replace("<image>", image_token_block, 1)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
        prompt_len = prompt_ids.shape[0]

        # RAW-RESPONSE FIX: vLLM stripped <ref>...</ref><box>...</box> from the
        # saved response. Reconstruct it so re-tokenization matches what the
        # model actually emitted (otherwise predicting positions are misaligned).
        response = re.sub(
            r'([A-Za-z][A-Za-z _]*?)(\[\[.+?\]\])',
            lambda m: f'<ref>{m.group(1)}</ref><box>{m.group(2)}</box>',
            p["response"], flags=re.DOTALL,
        )
        resp_enc = tokenizer(response, add_special_tokens=False)
        resp_ids = torch.tensor(resp_enc["input_ids"], dtype=prompt_ids.dtype, device=device)
        full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
        total_len = full_ids.shape[1]

        # Verify image tokens are present
        prompt_cpu = prompt_ids.cpu().tolist()
        image_idx_l = [i for i, t in enumerate(prompt_cpu) if t == img_pad_id]
        if len(image_idx_l) != n_image_tokens:
            if rank == 0 and n_done < 3:
                print(f"[worker {rank}] WARN {p['id']}: image_idx={len(image_idx_l)} != expected={n_image_tokens}", flush=True)
            n_skipped += 1; continue

        token_ranges = find_pred_token_ranges(response, p["pred_bboxes"], tokenizer)

        def _shift(pos):
            return max(0, pos - 1) if svar_shift else pos

        valid_pred_idx = []
        first_q_abs = []; label_q_abs_list = []; coord_q_abs_list = []
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
        if not valid_pred_idx:
            n_skipped += 1; continue

        # Find user-turn boundaries for mention/text-side classification
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        user_id = tokenizer.convert_tokens_to_ids("user")
        assistant_id = tokenizer.convert_tokens_to_ids("assistant")
        user_turn_start = 0
        assistant_start = None
        for k in range(len(prompt_cpu) - 1):
            if prompt_cpu[k] == im_start_id and prompt_cpu[k + 1] == user_id:
                user_turn_start = k + 2  # past <|im_start|>user
            if prompt_cpu[k] == im_start_id and prompt_cpu[k + 1] == assistant_id:
                assistant_start = k

        # Classify keys
        system_idx_l = []; task_idx_l = []; spc_idx_l = []; resp_idx_l = []
        for i, tid in enumerate(prompt_cpu):
            if i in image_idx_l: continue
            if tid in specials_ids:
                spc_idx_l.append(i); continue
            if i < user_turn_start:
                system_idx_l.append(i)
            elif assistant_start is not None and i >= assistant_start:
                resp_idx_l.append(i)
            else:
                task_idx_l.append(i)
        # Response tokens (post-prompt)
        for i in range(prompt_len, total_len):
            resp_idx_l.append(i)

        def device_idx(lst):
            return torch.tensor(lst, dtype=torch.long, device=device) if lst else torch.zeros(0, dtype=torch.long, device=device)

        _EXTRACT["image_indices"]    = device_idx(image_idx_l)
        _EXTRACT["system_indices"]   = device_idx(system_idx_l)
        _EXTRACT["task_text_indices"]= device_idx(task_idx_l)
        _EXTRACT["specials_indices"] = device_idx(spc_idx_l)
        _EXTRACT["response_indices"] = device_idx(resp_idx_l)

        # Unique sorted query positions
        all_positions = set()
        for fa, la, ca in zip(first_q_abs, label_q_abs_list, coord_q_abs_list):
            all_positions.add(fa)
            for x in la: all_positions.add(x)
            for x in ca: all_positions.add(x)
        sorted_positions = sorted(all_positions)
        pos_to_qrow = {pos: r for r, pos in enumerate(sorted_positions)}
        _EXTRACT["query_positions"] = torch.tensor(sorted_positions, dtype=torch.long, device=device)

        # Build per-pred specs
        n_preds = len(valid_pred_idx)
        pred_specs = []
        end_pos = assistant_start if assistant_start is not None else len(prompt_cpu)
        mention_cache = {}
        n_no_mention = 0
        for i, orig_i in enumerate(valid_pred_idx):
            pb = p["pred_bboxes"][orig_i]
            label = pb["label"]
            inside_l, outside_l = bbox_to_internvl_token_indices(pb["box"], tile_grid, has_thumb)
            if label not in mention_cache:
                mention_cache[label] = find_label_token_positions(label, tokenizer, prompt_cpu, user_turn_start, end_pos)
            mention_pos = mention_cache[label]
            if not mention_pos: n_no_mention += 1
            fd_qrow = pos_to_qrow[coord_q_abs_list[i][0]] if len(coord_q_abs_list[i]) > 0 else None
            pred_specs.append({
                "first_row": pos_to_qrow[first_q_abs[i]],
                "label_rows": torch.tensor([pos_to_qrow[x] for x in label_q_abs_list[i]],
                                            dtype=torch.long, device=device),
                "coord_rows": torch.tensor([pos_to_qrow[x] for x in coord_q_abs_list[i]],
                                            dtype=torch.long, device=device),
                "first_digit_row": fd_qrow,
                "inside_idx":  torch.tensor(inside_l,  dtype=torch.long, device=device),
                "outside_idx": torch.tensor(outside_l, dtype=torch.long, device=device),
                "mention_idx": torch.tensor(mention_pos, dtype=torch.long, device=device),
                "n_mention_positions": len(mention_pos),
                "label": label,
            })
        if n_no_mention > 0 and rank == 0 and n_done < 3:
            print(f"[worker {rank}] WARN: {n_no_mention}/{n_preds} preds had no mention match", flush=True)
        _EXTRACT["pred_specs"] = pred_specs
        _EXTRACT["buf"] = _new_buf(n_preds, n_layers, n_heads)

        # Forward via model.language_model (InternVL processes vision tokens internally during forward)
        # Actually we need to do a full model forward so the vision encoder produces the visual tokens
        # that get inserted at <IMG_CONTEXT> positions. InternVL uses model.forward(input_ids, pixel_values, ...).
        try:
            _EXTRACT["active"] = True
            n_pv = pixel_values.shape[0]
            image_flags = torch.ones(n_pv, dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(
                    input_ids=full_ids,
                    pixel_values=pixel_values,
                    image_flags=image_flags,
                    attention_mask=torch.ones(1, total_len, device=device, dtype=torch.long),
                )
            _EXTRACT["active"] = False
        except Exception as e:
            _EXTRACT["active"] = False
            print(f"[worker {rank}] skip {p['id']}: forward {e}", flush=True)
            n_skipped += 1; torch.cuda.empty_cache(); continue

        # Build per-pred records
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
                "label": pb["label"], "box": pb["box"],
                "is_hallucinated": bool(hallu_flags[orig_i]),
                "abs_query_pos": first_q_abs[i],
                "n_label_toks": int(spec["label_rows"].numel()),
                "n_coord_toks": int(spec["coord_rows"].numel()),
                "n_inside_patches": int(spec["inside_idx"].numel()),
                "n_image_tokens": n_image_tokens,
                "n_mention_positions": int(spec["mention_idx"].numel()),
                "tile_grid": tile_grid,
                "has_thumb": has_thumb,
                "attn":              attn_blocks["first"],
                "attn_label_mean":   attn_blocks["label"],
                "attn_coord_mean":   attn_blocks["coord"],
                "attn_first_digit":  attn_blocks["first_digit"],
            })
        records.append({
            "image_id": p["id"],
            "n_pred_bboxes": len(p["pred_bboxes"]),
            "n_extracted": len(out_objs),
            "tile_grid": tile_grid, "has_thumb": has_thumb,
            "n_image_tokens": n_image_tokens,
            "objects": out_objs,
        })
        n_done += 1
        if n_done % 5 == 0:
            rate = n_done / max(time.time() - t0, 1e-9)
            eta = (len(image_indices) - cnt - 1) / max(rate, 1e-9)
            print(f"[worker {rank}] [{cnt+1}/{len(image_indices)}] done={n_done} skip={n_skipped} "
                  f"rate={rate:.2f}img/s eta={eta/60:.1f}min", flush=True)
        for kk in list(_EXTRACT.keys()):
            if kk in ("active", "lang_attn_ids", "lang_attn_order"): continue
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
    ap.add_argument("--pred_file", default=PRED_FILE, help="predictions.json from generate.py")
    ap.add_argument("--dataset", default=DATASET_FILE, help="COCO openvocab dataset json")
    ap.add_argument("--svar_shift", action="store_true")
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
    for p in procs: p.join()
    print("all workers complete")


if __name__ == "__main__":
    main()
