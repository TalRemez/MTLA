"""The MTLA computation (shared by image detection and video grounding).

This module *is* MTLA. The paper's score (appendix pseudo-code) reads a prediction's own
attention from its response tokens Q_p to the input modality tokens, restricted to the tokens
inside its proposal region M(R_p)::

    x = attn[..., region_idx]   # keep modality tokens inside the region M(R_p)
    x = x.sum(-1)               # sum over the region     -> Localized Attention   (eq. 2)
    x = x.mean(-1)              # mean over Q_p            -> Multi-Token LA         (eq. 3)
    # (head/layer reduction, eq. 4, is done later on CPU in mtla.score.reduce_band)

`mtla_localized_attention` below is exactly that — an independent, model-agnostic function that
takes a prediction's `[L, H, Q_p, n_mod]` attention slice and its region mask and returns the
`[L, H]` localized-attention array. Everything else in this module just *gets the model's real
attention weights to that function*.

Getting the weights is the only hard part. A fused attention kernel never exposes the weights, and
`output_attentions=True` materializes every layer's `[H, Q, Q]` map at once — hundreds of GB for a
video clip. So we install a thin **capture** wrapper around the model's own eager-attention
forward: it runs the stock forward unchanged, then slices the returned weights down to just the
rows/cols MTLA reads (the query positions Q_p and the modality tokens) and offloads that small
`[H, N_q, n_mod]` tensor to CPU, freeing the GPU copy. GPU peak stays at one layer; the reduction
runs afterward from the CPU list. Because the wrapper reuses the model's own forward verbatim, the
captured weights are bit-for-bit what the model computed, for any model / transformers version.

Public API:
  * `mtla_localized_attention(attn, region_idx)` — the pure MTLA math (eqs. 2-3).
  * `CaptureState` / `install_capture(module_path, state)` — install the capture wrapper.
  * `compute_mtla(adapter, record, ds_by_id, ctx, rank)` — per-item driver (image boxes or video
    windows); builds inputs, runs one captured forward, returns the saved record.

The model/task-specific pieces (prompt build, prediction parsing + hallucination flags,
Q_p token finding, region mask, forward kwargs, record fields) live in the model adapters' `ext_*`
methods; this module is the common math and plumbing.
"""
from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass, field

import numpy as np
import torch


# ---------------------------------------------------------------------------
# The MTLA math (paper eqs. 2-3) — pure, independent, model-agnostic.
# ---------------------------------------------------------------------------
def mtla_localized_attention(attn: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
    """Localized attention for one prediction (paper eqs. 2-3).

    Args:
        attn:       ``[L, H, Q_p, n_mod]`` attention from the prediction's response tokens Q_p to
                    the input modality tokens, per layer ``L`` and head ``H``.
        region_idx: indices into the ``n_mod`` axis that fall inside the proposal region M(R_p).

    Returns:
        ``[L, H]`` localized attention: summed over the region (eq. 2), meaned over Q_p (eq. 3).
    """
    x = attn.index_select(-1, region_idx)   # keep modality tokens inside M(R_p)  [L,H,Q_p,|R|]
    x = x.sum(dim=-1)                        # sum over the region   -> LA   (eq.2) [L,H,Q_p]
    x = x.mean(dim=-1)                       # mean over Q_p         -> MTLA (eq.3) [L,H]
    return x


# ---------------------------------------------------------------------------
# Attention capture: get the model's real weights to the function above.
# ---------------------------------------------------------------------------
@dataclass
class CaptureState:
    """State shared between the capture wrapper and the per-item driver.

    ``layer_ids`` maps each LM decoder self-attention module id to its layer index (so the wrapper
    ignores vision-tower attention that routes through the same function). Before each forward the
    driver sets ``qpos`` / ``modidx`` (the rows/cols to keep) and allocates ``captured``; after the
    forward it reads ``captured`` (a CPU list of ``[H, N_q, n_mod]`` tensors, one per layer)."""
    layer_ids: dict = field(default_factory=dict)
    active: bool = False
    qpos: torch.Tensor | None = None       # query positions to keep (rows), on the model device
    modidx: torch.Tensor | None = None     # modality-token positions to keep (cols), on device
    captured: list | None = None           # per-layer [H, N_q, n_mod], fp32, on CPU


def make_capture_forward(state: CaptureState, orig_forward):
    """Wrap the model's own ``eager_attention_forward``: run it verbatim, then capture the weights.

    The stock eager forward already returns ``(attn_output, attn_weights)``; this wrapper calls it
    unchanged (so no attention math is reimplemented and it can't drift from the model), then, when
    active and this is an LM decoder layer, slices the weights to the query rows ``qpos`` and
    modality cols ``modidx`` ON GPU and offloads that small tensor to CPU — freeing the GPU copy so
    peak memory stays at one layer's map.
    """
    def wrapper(module, *args, **kwargs):
        attn_output, attn_weights = orig_forward(module, *args, **kwargs)
        if state.active and attn_weights is not None and id(module) in state.layer_ids:
            w = attn_weights[0]                          # [H, Q, K]
            if state.qpos is not None:
                w = w.index_select(1, state.qpos)        # [H, N_q, K]
            if state.modidx is not None:
                w = w.index_select(2, state.modidx)      # [H, N_q, n_mod]
            # Keep the tiny slice on the compute device: the full [H,Q,K] map (the memory wall) is
            # freed when the stock forward's output goes out of scope after this returns; only the
            # pre-sliced [H,N_q,n_mod] is retained, so holding all L layers is cheap (~1 GB).
            state.captured[state.layer_ids[id(module)]] = w.detach()
        return attn_output, attn_weights
    return wrapper


def install_capture(module_path: str, state: CaptureState):
    """Install the capture wrapper over ``module_path``'s ``eager_attention_forward``.

    Returns the imported module. The model must use ``attn_implementation="eager"`` so the weights
    flow through this function. Records the original forward and wraps it in place.
    """
    mod = importlib.import_module(module_path)
    mod.eager_attention_forward = make_capture_forward(state, mod.eager_attention_forward)
    return mod


# ---------------------------------------------------------------------------
# Per-item driver (shared by image_det AND video_span).
# ---------------------------------------------------------------------------
# The flow is identical across modalities: build the full token sequence, find each prediction's
# response tokens Q_p, run one captured forward, then apply the MTLA math per prediction. A
# "prediction" is a bbox (image) or a [t0,t1] window (video); its region M(R_p) is the modality
# tokens inside it. All model/task specifics come from the adapter's ext_* callbacks:
#   ext_build_inputs   -> preprocess input, build prompt, PARSE the response into predictions +
#                         hallucination flags, locate modality tokens  (None to skip the item)
#   ext_token_ranges   -> per prediction, its response tokens Q_p (label/coord or window digits)
#   ext_region_mask    -> the modality-token indices inside one prediction's region M(R_p)
#   ext_forward_kwargs -> kwargs for the single captured forward
#   ext_obj_record     -> per-prediction record fields (box/window + geometry)
#   ext_record         -> the top-level record wrapper (id keys, counts, objects)


def compute_mtla(adapter, record, ctx, rank=0):
    """Compute MTLA for one item's predictions via a single captured forward.

    Works for both image detection (predictions = boxes) and video grounding (predictions =
    time-span windows): the math is identical, only the ``ext_*`` callbacks differ. The ``record``
    (a generation record: ``{id, prompt, response, gt, extra}``) is self-contained — everything the
    extraction needs is on it. Returns the saved .pt record or None to skip the item.
    """
    state = ctx["state"]; device = ctx["device"]; tokenizer = ctx["tokenizer"]
    model = ctx["model"]; n_layers, n_heads = ctx["n_layers"], ctx["n_heads"]

    inp = adapter.ext_build_inputs(record, ctx, rank)
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

    # Per-prediction Q_p tokens (index-aligned with `predictions`): label + coordinate tokens for
    # images, the window's digit tokens for video.
    token_ranges = adapter.ext_token_ranges(response, predictions, tokenizer)
    assert len(token_ranges) == len(predictions), (
        f"ext_token_ranges must align with predictions: {len(token_ranges)} vs {len(predictions)}")

    # Absolute Q_p positions per prediction; x1 = the first coordinate/digit token.
    valid, qp_abs_list, x1_abs_list = [], [], []
    for i, tr in enumerate(token_ranges):
        if tr is None or tr["first_label_tok"] is None:
            continue
        label_abs = [prompt_len + t for t in tr["label_toks"] if prompt_len + t < total_len]
        coord_abs = [prompt_len + t for t in tr["coord_toks"] if prompt_len + t < total_len]
        if not label_abs:
            label_abs = [prompt_len + tr["first_label_tok"]]
        qp_abs = sorted(set(label_abs + coord_abs))
        if not qp_abs:
            continue
        valid.append(i); qp_abs_list.append(qp_abs)
        x1_abs_list.append(coord_abs[0] if coord_abs else None)
    if not valid:
        return None

    # Capture only the rows (union of all Q_p) and cols (modality tokens) MTLA reads.
    query_positions = sorted({x for qp in qp_abs_list for x in qp})
    row_of = {pos: r for r, pos in enumerate(query_positions)}
    state.qpos = torch.tensor(query_positions, dtype=torch.long, device=device)
    state.modidx = torch.tensor(modality_idx_l, dtype=torch.long, device=device)
    state.captured = [None] * n_layers

    fk = adapter.ext_forward_kwargs(full_ids, total_len, device, inp)
    try:
        state.active = True
        with torch.no_grad():
            model(**fk)
        state.active = False
    except Exception as e:
        state.active = False
        print(f"[worker {rank}] skip item: forward {e}", flush=True)
        return None

    # Stack the captured per-layer maps into one [L, H, N_q, n_mod] tensor and reduce on the same
    # device the forward ran on (the slice is tiny, so this stays on-GPU and fast). Upcast to fp32
    # once so the MTLA reduction matches the paper's math; only the [L,H] results move to CPU.
    attn = torch.stack(state.captured, dim=0).float()  # [L, H, N_q, n_mod]

    # Localized attention for the predictions whose Q_p we located, keyed by prediction index.
    la_by_i, fd_by_i = {}, {}
    for i, orig_i in enumerate(valid):
        region_idx = torch.tensor(adapter.ext_region_mask(predictions[orig_i], meta),
                                  dtype=torch.long, device=device)
        qp_rows = torch.tensor([row_of[x] for x in qp_abs_list[i]], dtype=torch.long, device=device)
        la_by_i[orig_i] = mtla_localized_attention(attn.index_select(2, qp_rows), region_idx).cpu()
        x1 = x1_abs_list[i]
        if x1 is not None:
            fd_rows = torch.tensor([row_of[x1]], dtype=torch.long, device=device)
            fd_by_i[orig_i] = mtla_localized_attention(attn.index_select(2, fd_rows), region_idx).cpu()

    # Emit an object for EVERY prediction so the shards are the complete candidate set for scoring
    # (a prediction whose Q_p couldn't be located gets a zero array + extracted=False, so it still
    # appears as a candidate but carries no grounding signal). `zeros` is [L, H].
    zeros = torch.zeros(n_layers, n_heads)
    out_objs = []
    for i, pred in enumerate(predictions):
        la = la_by_i.get(i, zeros); fd = fd_by_i.get(i, la_by_i.get(i, zeros))
        obj = adapter.ext_obj_record(pred, i, meta)
        obj["is_hallucinated"] = bool(hallu_flags[i])
        obj["extracted"] = i in la_by_i
        obj["local_attention"] = la.to(torch.float16).numpy()
        obj["first_digit"] = fd.to(torch.float16).numpy()
        out_objs.append(obj)

    rec = adapter.ext_record(record, meta, out_objs, n_predictions=len(predictions))
    del attn
    state.qpos = state.modidx = state.captured = None
    torch.cuda.empty_cache(); gc.collect()
    return rec
