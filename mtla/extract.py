"""The MTLA computation (image + video).

This module *is* MTLA. The paper's score (appendix pseudo-code) is

    x = attn[..., mask_idx]   # keep modality tokens inside the proposal region M(R_p)
    x = x.sum(-1)             # sum over region        -> Localized Attention   (eq. 2)
    x = x.mean(1)             # mean over Q_p           -> Multi-Token LA        (eq. 3)
    x = x.mean(-1)            # mean over heads
    return x[band].mean()     # mean over middle layers -> s(p)                  (eq. 4)

It reads from a `[n_layer, n_query, n_heads, n_modality_tok]` attention tensor. At detection
scale that tensor is terabytes and is never materialized: fused kernels don't even expose the
weights. So the *modality-token* reductions (the first three lines: restrict to M(R_p), sum the
region = LA, mean over Q_p) run **streaming, one decoder layer at a time, inside the forward
pass** via a monkeypatched eager-attention function. The remaining head/layer reductions (eq. 4)
run later, on CPU, in `mtla.score` over the small `[L, H]` per-prediction arrays saved here.

This module owns the model-agnostic MTLA kernel the per-model stage drivers share:
  * `MTLAState` — the per-forward state (query positions, modality indices, per-prediction
    region masks, output buffers).
  * `new_mtla_buffer(...)` — the multi-variant `[n_preds, L, H]` accumulators (first / label /
    coord / first_digit) holding LA summed over the region and meaned over each Q_p group.
  * `make_mtla_attention_forward(state, repeat_kv)` — a drop-in `eager_attention_forward` that,
    when `state.active`, computes MTLA's localized-attention sums; numerically identical to the
    stock eager forward otherwise.
  * `compute_image_mtla(...)` — the per-image MTLA computation: build inputs, run the patched
    forward, return the saved record.
  * `install(module_path, hook)` — monkeypatch a model family's attention function.

The per-model pieces (prompt build, token-position finding, region mask, forward kwargs, record
assembly) live in the model adapters; this module is the common MTLA math.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Token-aggregation "variants" = which response tokens Q_p we read attention from. first = first
# response token; label = mean over label tokens; coord = mean over the bbox-digit tokens;
# first_digit = the first coordinate digit (x1). New code emits all four for every COCO model
# (uniform); a model that doesn't use first_digit for scoring leaves its first_digit_row unset
# and the block stays zero.
MTLA_VARIANTS = ("first", "label", "coord", "first_digit")

# per-variant [n_preds, L, H] stats accumulated by the MTLA attention forward
_SUM_STATS = ("image_sum", "image_inside_sum", "image_outside_sum",
              "task_text_sum", "mention_sum", "system_sum", "specials_sum", "response_sum")
_MAX_STATS = ("image_max", "image_inside_max", "image_outside_max", "mention_max")


@dataclass
class MTLAState:
    """Mutable state shared between the MTLA attention forward and the per-image driver.

    Index tensors live on the model device; buffers accumulate on CPU in fp32. The driver sets
    the indices + pred_specs + buf before each forward and reads buf after it.
    """
    active: bool = False
    lang_attn_ids: set = field(default_factory=set)
    lang_attn_order: list = field(default_factory=list)
    query_positions: torch.Tensor | None = None
    image_indices: torch.Tensor | None = None
    system_indices: torch.Tensor | None = None
    task_text_indices: torch.Tensor | None = None
    specials_indices: torch.Tensor | None = None
    response_indices: torch.Tensor | None = None
    pred_specs: list | None = None
    buf: dict | None = None


def new_mtla_buffer(variants, n_preds, n_layers, n_heads):
    """Allocate the per-variant [n_preds, L, H] MTLA accumulators for one forward pass."""
    buf = {}
    for v in variants:
        d = {}
        for st in _SUM_STATS:
            d[st] = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32)
        for st in _MAX_STATS:
            d[st] = torch.full((n_preds, n_layers, n_heads), -1.0, dtype=torch.float32)
        buf[v] = d
    buf["first"]["image_argmax"] = torch.zeros(n_preds, n_layers, n_heads, dtype=torch.int32)
    return buf


def make_mtla_attention_forward(state: MTLAState, repeat_kv, variants=MTLA_VARIANTS):
    """Build the MTLA `eager_attention_forward` (the streaming, per-layer MTLA kernel).

    Numerically identical to the stock eager forward; additionally, when `state.active`, it
    computes each prediction's localized attention (LA, paper eq. 2): the per-Q_p-group mean of
    the attention summed over image tokens inside the proposal region (and, for analysis, all /
    outside-region and text-side token groups). `repeat_kv` is the model family's grouped-query
    helper. The remaining head/layer reductions (eq. 4) happen later in `mtla.score`.
    """
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        s = state
        if s.active and id(module) in s.lang_attn_ids:
            layer_idx = s.lang_attn_order.index(id(module))
            q_pos = s.query_positions
            rows = attn_weights[0].index_select(1, q_pos).transpose(0, 1).float()  # [Nq, H, K]
            img_rows = rows.index_select(2, s.image_indices)                        # [Nq, H, n_img]

            for pi, spec in enumerate(s.pred_specs):
                in_idx, out_idx = spec["inside_idx"], spec["outside_idx"]
                for v in variants:
                    if v == "first":
                        qrow = spec["first_row"]
                        img_row = img_rows[qrow]; full_row = rows[qrow]
                    elif v == "first_digit":
                        qrow = spec.get("first_digit_row")
                        if qrow is None:
                            continue
                        img_row = img_rows[qrow]; full_row = rows[qrow]
                    else:
                        qrows = spec[f"{v}_rows"]
                        if qrows is None or qrows.numel() == 0:
                            continue
                        img_row = img_rows.index_select(0, qrows).mean(dim=0)
                        full_row = rows.index_select(0, qrows).mean(dim=0)

                    buf_v = s.buf[v]
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
                        kidx = getattr(s, f"{kn}_indices")
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

    return patched


def install(module_path: str, hook):
    """Monkeypatch `eager_attention_forward` in `module_path`; return the module's repeat_kv."""
    mod = importlib.import_module(module_path)
    mod.eager_attention_forward = hook
    return mod


# ---------------------------------------------------------------------------
# Per-image MTLA computation
# ---------------------------------------------------------------------------
# The per-image flow is identical across image_det models: skip-check the
# prediction, flag hallucinations, build the full token sequence, filter to
# valid predictions, dedup query positions, build per-prediction specs, run the
# patched MTLA forward, then read the multi-variant buffer into the saved record.
# Everything that genuinely differs between models is delegated to a small set
# of adapter callbacks (prompt build, token-range finding, key-group
# classification, region mask, mention finder, forward kwargs, record extras).
# This is the actual dedup: one MTLA driver, byte-identical to the original
# per-model worker bodies (verified by exact-.pt parity).


def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]); a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def label_hallu(pred_box, pred_label, gt_objs):
    """A predicted (box,label) is hallucinated if no same-label GT box has IoU>=0.5."""
    pl = pred_label.strip().lower()
    for go in gt_objs:
        if str(go.get("label", "")).strip().lower() == pl:
            if _iou(pred_box, go["bbox_2d"]) >= 0.5:
                return False
    return True


def _device_idx(lst, device):
    if lst:
        return torch.tensor(lst, dtype=torch.long, device=device)
    return torch.zeros(0, dtype=torch.long, device=device)


def compute_image_mtla(adapter, p, ds_by_id, ctx, svar_shift, variants, rank=0):
    """Compute MTLA for one image's predictions via a single eager-attention forward.

    `adapter` supplies the model-specific callbacks (see InternVLAdapter /
    Qwen3VLAdapter `ext_*` methods). Returns the saved .pt record or None to skip.
    """
    import gc
    s = ctx["state"]; device = ctx["device"]; tokenizer = ctx["tokenizer"]
    model = ctx["model"]; n_layers, n_heads = ctx["n_layers"], ctx["n_heads"]
    specials_ids = ctx["specials_ids"]

    if p.get("status") != "success" or not p.get("pred_bboxes"):
        return None
    ds_item = ds_by_id.get(p["id"])
    if ds_item is None:
        return None
    gt_objs = json.loads(ds_item["conversations"][1]["value"])
    hallu_flags = [label_hallu(pb["box"], pb["label"], gt_objs) for pb in p["pred_bboxes"]]

    # Model-specific: load image, build prompt, tokenize, locate image tokens, geometry meta.
    inp = adapter.ext_build_inputs(p, ds_item, ctx, rank)
    if inp is None:
        return None
    prompt_ids = inp["prompt_ids"]; response = inp["response"]
    image_idx_l = inp["image_idx_l"]; meta = inp["meta"]
    prompt_len = prompt_ids.shape[0]
    prompt_cpu = prompt_ids.cpu().tolist()

    resp_ids = torch.tensor(tokenizer(response, add_special_tokens=False)["input_ids"],
                            dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
    total_len = full_ids.shape[1]

    token_ranges = adapter.ext_token_ranges(response, p["pred_bboxes"], tokenizer)

    def _shift(pos):
        return max(0, pos - 1) if svar_shift else pos

    valid_pred_idx = []; first_q_abs = []; label_q_abs_list = []; coord_q_abs_list = []; valid_tr = []
    for i, tr in enumerate(token_ranges):
        if tr is None:
            continue
        ftt = tr["first_label_tok"]
        if ftt is None:
            continue
        first_abs = _shift(prompt_len + ftt)
        if first_abs >= total_len:
            continue
        label_abs = [_shift(prompt_len + t) for t in tr["label_toks"] if (prompt_len + t) < total_len]
        coord_abs = [_shift(prompt_len + t) for t in tr["coord_toks"] if (prompt_len + t) < total_len]
        if not label_abs:
            label_abs = [first_abs]
        valid_pred_idx.append(i); first_q_abs.append(first_abs)
        label_q_abs_list.append(label_abs); coord_q_abs_list.append(coord_abs); valid_tr.append(tr)
    if not valid_pred_idx:
        return None

    # Model-specific: classify prompt tokens into system / task-text / specials / response groups.
    groups = adapter.ext_classify_keys(prompt_cpu, image_idx_l, specials_ids, tokenizer,
                                       prompt_len, total_len)
    s.image_indices = _device_idx(image_idx_l, device)
    s.system_indices = _device_idx(groups["system"], device)
    s.task_text_indices = _device_idx(groups["task"], device)
    s.specials_indices = _device_idx(groups["specials"], device)
    s.response_indices = _device_idx(groups["response"], device)
    user_turn_start = groups["user_turn_start"]; assistant_start = groups["assistant_start"]

    all_positions = set()
    for fa, la, ca in zip(first_q_abs, label_q_abs_list, coord_q_abs_list):
        all_positions.add(fa)
        for x in la:
            all_positions.add(x)
        for x in ca:
            all_positions.add(x)
    sorted_positions = sorted(all_positions)
    pos_to_qrow = {pos: r for r, pos in enumerate(sorted_positions)}
    s.query_positions = torch.tensor(sorted_positions, dtype=torch.long, device=device)

    n_preds = len(valid_pred_idx)
    pred_specs = []
    end_pos = assistant_start if assistant_start is not None else len(prompt_cpu)
    mention_cache = {}
    for i, orig_i in enumerate(valid_pred_idx):
        pb = p["pred_bboxes"][orig_i]; label = pb["label"]
        inside_l, outside_l = adapter.ext_region_mask(pb["box"], meta)
        if label not in mention_cache:
            mention_cache[label] = adapter.ext_mentions(label, tokenizer, prompt_cpu,
                                                        user_turn_start, end_pos)
        mention_pos = mention_cache[label]
        fd_qrow = pos_to_qrow[coord_q_abs_list[i][0]] if len(coord_q_abs_list[i]) > 0 else None
        pred_specs.append({
            "first_row": pos_to_qrow[first_q_abs[i]],
            "label_rows": torch.tensor([pos_to_qrow[x] for x in label_q_abs_list[i]], dtype=torch.long, device=device),
            "coord_rows": torch.tensor([pos_to_qrow[x] for x in coord_q_abs_list[i]], dtype=torch.long, device=device),
            "first_digit_row": fd_qrow,
            "inside_idx": torch.tensor(inside_l, dtype=torch.long, device=device),
            "outside_idx": torch.tensor(outside_l, dtype=torch.long, device=device),
            "mention_idx": torch.tensor(mention_pos, dtype=torch.long, device=device),
            "label": label,
        })
    s.pred_specs = pred_specs
    s.buf = new_mtla_buffer(variants, n_preds, n_layers, n_heads)

    fk = adapter.ext_forward_kwargs(full_ids, total_len, device, inp)
    try:
        s.active = True
        with torch.no_grad():
            model(**fk)
        s.active = False
    except Exception as e:
        s.active = False
        print(f"[worker {rank}] skip {p['id']}: forward {e}", flush=True)
        return None

    out_objs = []
    for i, orig_i in enumerate(valid_pred_idx):
        spec = pred_specs[i]
        attn_blocks = {}
        for v in variants:
            d = s.buf[v]
            blk = {st: d[st][i].to(torch.float16).numpy() for st in d if st != "image_argmax"}
            if v == "first":
                blk["image_argmax"] = d["image_argmax"][i].numpy()
            attn_blocks[v] = blk
        pb = p["pred_bboxes"][orig_i]
        obj = {
            "pred_idx": orig_i, "label": pb["label"], "box": pb["box"],
            "is_hallucinated": bool(hallu_flags[orig_i]),
            "abs_query_pos": first_q_abs[i],
            "n_label_toks": int(spec["label_rows"].numel()),
            "n_coord_toks": int(spec["coord_rows"].numel()),
            "n_inside_patches": int(spec["inside_idx"].numel()),
            "n_mention_positions": int(spec["mention_idx"].numel()),
            "attn": attn_blocks["first"], "attn_label_mean": attn_blocks["label"],
            "attn_coord_mean": attn_blocks["coord"], "attn_first_digit": attn_blocks["first_digit"],
        }
        obj.update(adapter.ext_obj_extras(meta, valid_tr[i]))
        out_objs.append(obj)
    rec = {"image_id": p["id"], "n_pred_bboxes": len(p["pred_bboxes"]),
           "n_extracted": len(out_objs), "objects": out_objs}
    rec.update(adapter.ext_rec_extras(meta))
    for kk in ("query_positions", "image_indices", "system_indices", "task_text_indices",
               "specials_indices", "response_indices", "pred_specs", "buf"):
        setattr(s, kk, None)
    torch.cuda.empty_cache(); gc.collect()
    return rec
