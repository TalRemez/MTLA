"""The MTLA computation (image; video shares the same idea).

This module *is* MTLA. The paper's score (appendix pseudo-code) is

    x = attn[..., mask_idx]   # keep modality tokens inside the proposal region M(R_p)
    x = x.sum(-1)             # sum over region        -> Localized Attention   (eq. 2)
    x = x.mean(1)             # mean over Q_p           -> Multi-Token LA        (eq. 3)
    x = x.mean(-1)            # mean over heads
    return x[band].mean()     # mean over middle layers -> s(p)                  (eq. 4)

It reads from a `[n_layer, n_query, n_heads, n_modality_tok]` attention tensor. At detection
scale that tensor is terabytes and is never materialized: fused kernels don't even expose the
weights. So the *modality-token* reductions (restrict to M(R_p), sum the region = LA, mean over
Q_p) run **streaming, one decoder layer at a time, inside the forward pass** via a monkeypatched
eager-attention function. The remaining head/layer reductions (eq. 4) run later, on CPU, in
`mtla.score` over the small `[L, H]` per-prediction array saved here.

Each prediction stores two `[L, H]` arrays of localized attention (eqs. 2-3):
  * `local_attention` — meaned over *all* the prediction's tokens Q_p (the MTLA score).
  * `first_digit`     — read at the single first coordinate digit (x1), the autoregressive
    decision point analyzed in the paper.

Public API:
  * `MTLAState` — per-forward state (query positions, image-token indices, per-prediction region
    masks, output buffer).
  * `new_mtla_buffer(...)` — the per-prediction `[n_preds, L, H]` accumulators.
  * `make_mtla_attention_forward(state, repeat_kv)` — the streaming, per-layer MTLA kernel; a
    drop-in `eager_attention_forward`, numerically identical to the stock one when not active.
  * `compute_image_mtla(...)` — the per-image MTLA computation: build inputs, run the patched
    forward, return the saved record.
  * `install(module_path, hook)` — monkeypatch a model family's attention function.

The model-specific pieces (prompt build, prediction-token finding, region mask, forward kwargs,
record geometry) live in the model adapters' `ext_*` methods; this module is the common math.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class MTLAState:
    """Mutable state shared between the MTLA attention forward and the per-image driver.

    Index tensors live on the model device; the buffer accumulates on CPU in fp32. The driver
    sets `query_positions` / `image_indices` / `pred_specs` / `buf` before each forward and reads
    `buf` after it.
    """
    active: bool = False
    lang_attn_ids: set = field(default_factory=set)
    lang_attn_order: list = field(default_factory=list)
    query_positions: torch.Tensor | None = None   # unique response-token positions Q_p (sorted)
    image_indices: torch.Tensor | None = None      # absolute positions of the modality tokens
    pred_specs: list | None = None                 # per-prediction {qp_rows, first_digit_row, inside_idx}
    buf: dict | None = None                         # {"local_attention": [P,L,H], "first_digit": [P,L,H]}


def new_mtla_buffer(n_preds, n_layers, n_heads):
    """Allocate the per-prediction `[n_preds, L, H]` MTLA accumulators for one forward pass."""
    return {
        "local_attention": torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32),
        "first_digit": torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32),
    }


def make_mtla_attention_forward(state: MTLAState, repeat_kv):
    """Build the MTLA `eager_attention_forward` (the streaming, per-layer MTLA kernel).

    Numerically identical to the stock eager forward; additionally, when `state.active`, it
    accumulates each prediction's localized attention (eqs. 2-3): for the `local_attention`
    signal, the mean over the prediction's tokens Q_p of the attention summed over the image
    tokens inside the proposal region; for `first_digit`, the same but read at the single x1
    token. `repeat_kv` is the model family's grouped-query helper. The head/layer reductions
    (eq. 4) happen later in `mtla.score`.
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
            rows = attn_weights[0].index_select(1, s.query_positions).transpose(0, 1).float()  # [Nq, H, K]
            img_rows = rows.index_select(2, s.image_indices)                                    # [Nq, H, n_img]
            for pi, spec in enumerate(s.pred_specs):
                in_idx = spec["inside_idx"]
                # MTLA: mean over Q_p (eq. 3), then sum over the proposal region (eq. 2 = LA).
                la = img_rows.index_select(0, spec["qp_rows"]).mean(dim=0)        # [H, n_img]
                s.buf["local_attention"][pi, layer_idx, :] += la.index_select(1, in_idx).sum(dim=1).cpu()
                fd = spec["first_digit_row"]
                if fd is not None:
                    la_fd = img_rows[fd]                                          # [H, n_img]
                    s.buf["first_digit"][pi, layer_idx, :] += la_fd.index_select(1, in_idx).sum(dim=1).cpu()

        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return attn_output, None

    return patched


def install(module_path: str, hook):
    """Monkeypatch `eager_attention_forward` in `module_path`; return the imported module."""
    mod = importlib.import_module(module_path)
    mod.eager_attention_forward = hook
    return mod


# ---------------------------------------------------------------------------
# Per-image MTLA computation
# ---------------------------------------------------------------------------
# The per-image flow is identical across image_det models: skip-check the prediction, flag
# hallucinations, build the full token sequence, find each prediction's tokens Q_p, run one
# patched forward, then read the per-prediction buffer into the saved record. The few pieces that
# differ between models are delegated to the adapter's `ext_*` callbacks (prompt build,
# prediction-token finding, region mask, forward kwargs, record geometry).


def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]); a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def label_hallu(pred_box, pred_label, gt_objs):
    """A predicted (box, label) is hallucinated if no same-label GT box has IoU >= 0.5."""
    pl = pred_label.strip().lower()
    for go in gt_objs:
        if str(go.get("label", "")).strip().lower() == pl and _iou(pred_box, go["bbox_2d"]) >= 0.5:
            return False
    return True


def compute_image_mtla(adapter, p, ds_by_id, ctx, svar_shift, rank=0):
    """Compute MTLA for one image's predictions via a single eager-attention forward.

    `adapter` supplies the model-specific callbacks (see InternVLAdapter / Qwen3VLAdapter
    `ext_*` methods). Returns the saved .pt record or None to skip.
    """
    import gc
    s = ctx["state"]; device = ctx["device"]; tokenizer = ctx["tokenizer"]
    model = ctx["model"]; n_layers, n_heads = ctx["n_layers"], ctx["n_heads"]

    if p.get("status") != "success" or not p.get("pred_bboxes"):
        return None
    ds_item = ds_by_id.get(p["id"])
    if ds_item is None:
        return None
    gt_objs = json.loads(ds_item["conversations"][1]["value"])
    hallu_flags = [label_hallu(pb["box"], pb["label"], gt_objs) for pb in p["pred_bboxes"]]

    # Model-specific: load image, build prompt, tokenize, locate the modality tokens, geometry.
    inp = adapter.ext_build_inputs(p, ds_item, ctx, rank)
    if inp is None:
        return None
    prompt_ids = inp["prompt_ids"]; response = inp["response"]
    image_idx_l = inp["image_idx_l"]; meta = inp["meta"]
    prompt_len = prompt_ids.shape[0]

    resp_ids = torch.tensor(tokenizer(response, add_special_tokens=False)["input_ids"],
                            dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
    total_len = full_ids.shape[1]

    # Model-specific: per prediction, the response-token spans for its label and coordinates.
    token_ranges = adapter.ext_token_ranges(response, p["pred_bboxes"], tokenizer)

    def _shift(pos):
        return max(0, pos - 1) if svar_shift else pos

    # Q_p = label tokens + coordinate tokens; x1 = first coordinate digit.
    valid_pred_idx = []; qp_abs_list = []; x1_abs_list = []
    for i, tr in enumerate(token_ranges):
        if tr is None or tr["first_label_tok"] is None:
            continue
        label_abs = [_shift(prompt_len + t) for t in tr["label_toks"] if prompt_len + t < total_len]
        coord_abs = [_shift(prompt_len + t) for t in tr["coord_toks"] if prompt_len + t < total_len]
        if not label_abs:
            label_abs = [_shift(prompt_len + tr["first_label_tok"])]
        qp_abs = sorted(set(label_abs + coord_abs))
        if not qp_abs:
            continue
        valid_pred_idx.append(i); qp_abs_list.append(qp_abs)
        x1_abs_list.append(coord_abs[0] if coord_abs else None)
    if not valid_pred_idx:
        return None

    sorted_positions = sorted({x for qp in qp_abs_list for x in qp})
    pos_to_qrow = {pos: r for r, pos in enumerate(sorted_positions)}
    s.query_positions = torch.tensor(sorted_positions, dtype=torch.long, device=device)
    s.image_indices = torch.tensor(image_idx_l, dtype=torch.long, device=device)

    n_preds = len(valid_pred_idx)
    pred_specs = []
    for i, orig_i in enumerate(valid_pred_idx):
        inside_idx = adapter.ext_region_mask(p["pred_bboxes"][orig_i]["box"], meta)
        x1 = x1_abs_list[i]
        pred_specs.append({
            "qp_rows": torch.tensor([pos_to_qrow[x] for x in qp_abs_list[i]], dtype=torch.long, device=device),
            "first_digit_row": pos_to_qrow[x1] if x1 is not None else None,
            "inside_idx": torch.tensor(inside_idx, dtype=torch.long, device=device),
        })
    s.pred_specs = pred_specs
    s.buf = new_mtla_buffer(n_preds, n_layers, n_heads)

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
        pb = p["pred_bboxes"][orig_i]
        obj = {
            "pred_idx": orig_i, "label": pb["label"], "box": pb["box"],
            "is_hallucinated": bool(hallu_flags[orig_i]),
            "n_qp_tokens": int(pred_specs[i]["qp_rows"].numel()),
            "local_attention": s.buf["local_attention"][i].to(torch.float16).numpy(),
            "first_digit": s.buf["first_digit"][i].to(torch.float16).numpy(),
        }
        obj.update(adapter.ext_obj_extras(meta))
        out_objs.append(obj)
    rec = {"image_id": p["id"], "n_pred_bboxes": len(p["pred_bboxes"]),
           "n_extracted": len(out_objs), "objects": out_objs}
    rec.update(adapter.ext_rec_extras(meta))
    s.query_positions = s.image_indices = s.pred_specs = s.buf = None
    torch.cuda.empty_cache(); gc.collect()
    return rec
