"""The MTLA computation (shared by image detection and video grounding).

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
  * `MTLAState` — per-forward state (query positions, modality-token indices, per-prediction
    region masks, output buffer).
  * `new_mtla_buffer(...)` — the per-prediction `[n_preds, L, H]` accumulators.
  * `make_mtla_attention_forward(state, repeat_kv)` — the streaming, per-layer MTLA kernel; a
    drop-in `eager_attention_forward`, numerically identical to the stock one when not active.
  * `compute_mtla(...)` — the per-item MTLA computation (image boxes or video windows): build
    inputs, run the patched forward, return the saved record.
  * `install(module_path, hook)` — monkeypatch a model family's attention function.

The model/task-specific pieces (prompt build, prediction enumeration + hallucination flags,
prediction-token finding, region mask, forward kwargs, record fields) live in the model adapters'
`ext_*` methods; this module is the common math.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class MTLAState:
    """Mutable state shared between the MTLA attention forward and the per-item driver.

    Index tensors live on the model device; the buffer accumulates on CPU in fp32. The driver
    sets `query_positions` / `modality_indices` / `pred_specs` / `buf` before each forward and
    reads `buf` after it.
    """
    active: bool = False
    lang_attn_ids: set = field(default_factory=set)
    lang_attn_order: list = field(default_factory=list)
    query_positions: torch.Tensor | None = None   # unique response-token positions Q_p (sorted)
    modality_indices: torch.Tensor | None = None   # absolute positions of the modality tokens
                                                   # (image patches, or video frame tokens)
    pred_specs: list | None = None                 # per-prediction {qp_rows, first_digit_row, inside_idx}
    buf: dict | None = None                         # {"local_attention": [P,L,H], "first_digit": [P,L,H]}


def new_mtla_buffer(n_preds, n_layers, n_heads):
    """Allocate the per-prediction `[n_preds, L, H]` MTLA accumulators for one forward pass."""
    return {
        "local_attention": torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32),
        "first_digit": torch.zeros(n_preds, n_layers, n_heads, dtype=torch.float32),
    }


def make_mtla_attention_forward(state: MTLAState, repeat_kv=None):
    """Build the MTLA `eager_attention_forward` (the streaming, per-layer MTLA kernel).

    Numerically identical to the stock eager forward; additionally, when `state.active`, it
    accumulates each prediction's localized attention (eqs. 2-3): for the `local_attention`
    signal, the mean over the prediction's tokens Q_p of the attention summed over the modality
    tokens inside the proposal region; for `first_digit`, the same but read at the single x1
    token. `repeat_kv` is the grouped-query KV-repeat helper; it defaults to `mtla.utils.repeat_kv`
    (identical to HF's Qwen3/Qwen3-VL impl) so model families don't need to pass it. The
    head/layer reductions (eq. 4) happen later in `mtla.score`.
    """
    if repeat_kv is None:
        from .utils import repeat_kv
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
            mod_rows = rows.index_select(2, s.modality_indices)                                 # [Nq, H, n_mod]
            for pi, spec in enumerate(s.pred_specs):
                in_idx = spec["inside_idx"]
                # MTLA: mean over Q_p (eq. 3), then sum over the proposal region (eq. 2 = LA).
                la = mod_rows.index_select(0, spec["qp_rows"]).mean(dim=0)        # [H, n_mod]
                s.buf["local_attention"][pi, layer_idx, :] += la.index_select(1, in_idx).sum(dim=1).cpu()
                fd = spec["first_digit_row"]
                if fd is not None:
                    la_fd = mod_rows[fd]                                          # [H, n_mod]
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
# Per-item MTLA computation (shared by image_det AND video_span)
# ---------------------------------------------------------------------------
# The per-item flow is identical across modalities: build the full token sequence, find each
# prediction's tokens Q_p, run one patched forward, then read the per-prediction buffer into the
# saved record. A "prediction" is a bbox (image) or a time-span window (video); its proposal
# region M(R_p) is the modality tokens inside it (image patches, or the frames inside the span).
# Everything model/task-specific is delegated to the adapter's `ext_*` callbacks:
#   ext_build_inputs   -> preprocess input, build prompt, locate modality tokens, list the
#                         predictions + their hallucination flags (None to skip the item)
#   ext_token_ranges   -> per prediction, the response tokens Q_p (label/coord or window digits)
#   ext_region_mask    -> the modality-token indices inside one prediction's region M(R_p)
#   ext_forward_kwargs -> kwargs for the single patched forward
#   ext_obj_record     -> the per-prediction record fields (pred_idx/label/box or window/span)
#   ext_record         -> the top-level record wrapper (id keys, counts, objects)


def compute_mtla(adapter, p, ds_by_id, ctx, svar_shift, rank=0):
    """Compute MTLA for one item's predictions via a single eager-attention forward.

    Works for both image detection (predictions = boxes) and video grounding (predictions =
    time-span windows): the math is identical, only the `ext_*` callbacks differ. `adapter`
    supplies them (see InternVLAdapter / Qwen3VLAdapter). Returns the saved .pt record or None
    to skip the item.
    """
    import gc
    s = ctx["state"]; device = ctx["device"]; tokenizer = ctx["tokenizer"]
    model = ctx["model"]; n_layers, n_heads = ctx["n_layers"], ctx["n_heads"]

    # Model/task-specific: preprocess input, build prompt, locate modality tokens, and enumerate
    # the predictions (boxes or windows) with their hallucination flags. None => skip the item.
    inp = adapter.ext_build_inputs(p, ds_by_id, ctx, rank)
    if inp is None:
        return None
    prompt_ids = inp["prompt_ids"]; response = inp["response"]
    modality_idx_l = inp["modality_idx_l"]; meta = inp["meta"]
    predictions = inp["predictions"]; hallu_flags = inp["hallu_flags"]
    prompt_len = prompt_ids.shape[0]

    resp_ids = torch.tensor(tokenizer(response, add_special_tokens=False)["input_ids"],
                            dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
    total_len = full_ids.shape[1]

    # Model-specific: per prediction, the response-token spans for its Q_p. For images that is the
    # label + coordinate tokens; for video, the window's digit tokens (returned as `coord_toks`
    # with `first_label_tok` set to the first digit, so the assembly below is identical).
    # Contract: token_ranges is index-aligned with `predictions` (one entry each), so `valid_pred_idx`
    # below can index `predictions` directly.
    token_ranges = adapter.ext_token_ranges(response, predictions, tokenizer)
    assert len(token_ranges) == len(predictions), (
        f"ext_token_ranges must align with predictions: got {len(token_ranges)} vs {len(predictions)}")

    def _shift(pos):
        return max(0, pos - 1) if svar_shift else pos

    # Q_p = label tokens + coordinate tokens; x1 = first coordinate/digit token.
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
    s.modality_indices = torch.tensor(modality_idx_l, dtype=torch.long, device=device)

    n_preds = len(valid_pred_idx)
    pred_specs = []
    for i, orig_i in enumerate(valid_pred_idx):
        inside_idx = adapter.ext_region_mask(predictions[orig_i], meta)
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
        print(f"[worker {rank}] skip item: forward {e}", flush=True)
        return None

    out_objs = []
    for i, orig_i in enumerate(valid_pred_idx):
        obj = adapter.ext_obj_record(predictions[orig_i], orig_i, meta)
        obj["is_hallucinated"] = bool(hallu_flags[orig_i])
        obj["n_qp_tokens"] = int(pred_specs[i]["qp_rows"].numel())
        obj["local_attention"] = s.buf["local_attention"][i].to(torch.float16).numpy()
        obj["first_digit"] = s.buf["first_digit"][i].to(torch.float16).numpy()
        out_objs.append(obj)
    rec = adapter.ext_record(p, meta, out_objs, n_predictions=len(predictions))
    s.query_positions = s.modality_indices = s.pred_specs = s.buf = None
    torch.cuda.empty_cache(); gc.collect()
    return rec
