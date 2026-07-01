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
  * `compute_mtla(adapter, record, ctx, rank)` — per-item driver (image boxes or video windows);
    builds inputs, runs one captured forward, reduces per prediction, returns the saved record.

The model/task-specific pieces (prompt build, prediction parsing + hallucination flags,
Q_p token finding, region mask, forward kwargs, record fields) live in the model adapters'
callbacks (build_inputs / query_tokens / region_mask / forward_kwargs / prediction_record /
item_record); this module is the common math and plumbing.
"""
from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, cast

import torch

from .types import BuildInputs, Ctx, ItemRecord, PredObject, TokenRange

if TYPE_CHECKING:
    from .models.base import ModelAdapter
    from .types import GenRecord


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


def make_capture_forward(state: CaptureState, orig_forward: Callable) -> Callable:
    """Wrap the model's own ``eager_attention_forward``: run it verbatim, then capture the weights.

    The stock eager forward already returns ``(attn_output, attn_weights)``; this wrapper calls it
    unchanged (so no attention math is reimplemented and it can't drift from the model), then, when
    active and this is an LM decoder layer, slices the weights to the query rows ``qpos`` and
    modality cols ``modidx`` ON GPU and offloads that small tensor to CPU — freeing the GPU copy so
    peak memory stays at one layer's map.
    """
    def wrapper(module: Any, *args: Any, **kwargs: Any) -> tuple:
        attn_output, attn_weights = orig_forward(module, *args, **kwargs)
        if state.active and attn_weights is not None and id(module) in state.layer_ids:
            assert state.captured is not None            # set by the driver before each forward
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


def install_capture(module_path: str, state: CaptureState) -> Any:
    """Install the capture wrapper over ``module_path``'s ``eager_attention_forward``.

    Returns the imported module. The model must use ``attn_implementation="eager"`` so the weights
    flow through this function. Records the original forward and wraps it in place.
    """
    mod = importlib.import_module(module_path)
    mod.eager_attention_forward = make_capture_forward(  # type: ignore[attr-defined]
        state, mod.eager_attention_forward)
    return mod


# ---------------------------------------------------------------------------
# Per-item driver (shared by image_det AND video_span).
# ---------------------------------------------------------------------------
# All model/task specifics come from the adapter callbacks the helpers below call:
#   build_inputs      -> preprocess input, build prompt, PARSE the response into predictions +
#                        hallucination flags, locate modality tokens  (None to skip the item)
#   query_tokens      -> per prediction, its response tokens Q_p (label/coord or window digits)
#   region_mask       -> the modality-token indices inside one prediction's region M(R_p)
#   forward_kwargs    -> kwargs for the single captured forward
#   prediction_record -> per-prediction record fields (box/window + geometry)
#   item_record       -> the top-level record wrapper (id keys, counts, objects)


def _resolve_qp_positions(token_ranges: list[TokenRange | None], prompt_len: int, total_len: int
                          ) -> tuple[list[int], list[list[int]], list[int | None]]:
    """Map each prediction's response tokens Q_p to absolute sequence positions.

    ``token_ranges[i]`` (from ``query_tokens``) gives prediction i's label + coordinate token
    offsets within the response; here they are shifted past the prompt and clamped to the sequence.
    Returns ``(kept, qp_positions, x1_positions)`` for the predictions with locatable tokens:
    ``kept`` are their indices into ``predictions``, ``qp_positions[j]`` the sorted absolute Q_p
    positions of ``kept[j]``, and ``x1_positions[j]`` its first coordinate/digit token (or None).
    """
    kept: list[int] = []
    qp_positions: list[list[int]] = []
    x1_positions: list[int | None] = []
    for i, tr in enumerate(token_ranges):
        if tr is None or tr["first_label_tok"] is None:
            continue
        label_abs = [prompt_len + t for t in tr["label_toks"] if prompt_len + t < total_len]
        coord_abs = [prompt_len + t for t in tr["coord_toks"] if prompt_len + t < total_len]
        if not label_abs:
            label_abs = [prompt_len + tr["first_label_tok"]]
        qp = sorted(set(label_abs + coord_abs))
        if not qp:
            continue
        kept.append(i)
        qp_positions.append(qp)
        x1_positions.append(coord_abs[0] if coord_abs else None)
    return kept, qp_positions, x1_positions


def _run_captured_forward(model: Any, state: CaptureState, fk: dict, query_positions: list[int],
                          modality_idx: list[int], n_layers: int, device: str, rank: int
                          ) -> "torch.Tensor | None":
    """Run one teacher-forced forward with attention capture active, then return the captured maps
    stacked as ``[L, H, N_q, n_mod]`` (fp32, on ``device``), or None if the forward failed.

    ``query_positions`` (the union of all Q_p) and ``modality_idx`` are the rows/cols the capture
    wrapper keeps; the forward is what materializes the attention the wrapper slices per layer.
    """
    state.qpos = torch.tensor(query_positions, dtype=torch.long, device=device)
    state.modidx = torch.tensor(modality_idx, dtype=torch.long, device=device)
    state.captured = [None] * n_layers
    try:
        state.active = True
        with torch.no_grad():
            model(**fk)
    except Exception as e:
        print(f"[worker {rank}] skip item: forward {e}", flush=True)
        return None
    finally:
        state.active = False
    # The per-layer slices are tiny, so stacking + reducing stays on the compute device; upcast to
    # fp32 once here so the MTLA reduction matches the paper's math.
    return torch.stack(state.captured, dim=0).float()


def compute_mtla(adapter: "ModelAdapter", record: "GenRecord", ctx: Ctx, rank: int = 0
                 ) -> "ItemRecord | None":
    """Compute MTLA for every prediction in one item and return its feature-shard record.

    Steps: build the input and parse the response into predictions (``build_inputs``); teacher-force
    the prompt+response through one attention-capturing forward; then, per prediction, take its Q_p
    rows out of the captured attention and reduce them with ``mtla_localized_attention`` to an
    ``[L, H]`` array. Two arrays are stored per prediction: ``local_attention`` (over all Q_p tokens)
    and ``first_digit`` (the first coordinate/digit token only).

    Works for both image detection (predictions = boxes) and video grounding (predictions = time
    spans) — only the adapter callbacks differ. ``record`` is a self-contained generation record
    ``{id, prompt, response, gt, extra}``. Returns the record to save, or None to skip the item
    (nothing to build, or no prediction had locatable tokens).
    """
    state = ctx["state"]; device = ctx["device"]; tokenizer = ctx["tokenizer"]
    model = ctx["model"]; n_layers, n_heads = ctx["n_layers"], ctx["n_heads"]
    # Let a multi-task model's callbacks dispatch on the task (single-task models ignore it).
    adapter._task = ctx.get("task", adapter.tasks[0] if adapter.tasks else None)

    inp = adapter.build_inputs(record, ctx, rank)
    if inp is None:
        return None
    response = inp["response"]; predictions = inp["predictions"]
    hallu_flags = inp["hallu_flags"]; meta = inp["meta"]
    prompt_ids = inp["prompt_ids"]; prompt_len = prompt_ids.shape[0]

    # Teacher-force prompt + response through the model.
    resp_ids = torch.tensor(tokenizer(response, add_special_tokens=False)["input_ids"],
                            dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
    total_len = full_ids.shape[1]

    # Locate each prediction's response tokens Q_p (index-aligned with `predictions`).
    token_ranges = adapter.query_tokens(response, predictions, tokenizer)
    assert len(token_ranges) == len(predictions), (
        f"query_tokens must align with predictions: {len(token_ranges)} vs {len(predictions)}")
    kept, qp_positions, x1_positions = _resolve_qp_positions(token_ranges, prompt_len, total_len)
    if not kept:
        return None

    # Capture only the rows (union of all Q_p) and cols (modality tokens) the reduction reads.
    query_positions = sorted({p for qp in qp_positions for p in qp})
    row_of = {pos: r for r, pos in enumerate(query_positions)}
    fk = adapter.forward_kwargs(full_ids, total_len, device, inp)
    attn = _run_captured_forward(model, state, fk, query_positions, inp["modality_idx_l"],
                                 n_layers, device, rank)                 # [L, H, N_q, n_mod]
    if attn is None:
        return None

    # Reduce per prediction: local_attention over all Q_p tokens, first_digit over the x1 token.
    la_by_i, fd_by_i = {}, {}
    for j, i in enumerate(kept):
        region_idx = torch.tensor(adapter.region_mask(predictions[i], meta),
                                  dtype=torch.long, device=device)
        qp_rows = torch.tensor([row_of[p] for p in qp_positions[j]], dtype=torch.long, device=device)
        la_by_i[i] = mtla_localized_attention(attn.index_select(2, qp_rows), region_idx).cpu()
        x1_pos = x1_positions[j]
        if x1_pos is not None:
            x1_row = torch.tensor([row_of[x1_pos]], dtype=torch.long, device=device)
            fd_by_i[i] = mtla_localized_attention(attn.index_select(2, x1_row), region_idx).cpu()

    # Emit an object for EVERY prediction so the shards are the complete candidate set at score time;
    # a prediction whose Q_p couldn't be located gets a zero array and `extracted=False`.
    zeros = torch.zeros(n_layers, n_heads)
    out_objs: list[PredObject] = []
    for i, pred in enumerate(predictions):
        la = la_by_i.get(i, zeros)
        obj = adapter.prediction_record(pred, i, meta)
        obj["is_hallucinated"] = bool(hallu_flags[i])
        obj["extracted"] = i in la_by_i
        obj["local_attention"] = la.to(torch.float16).numpy()
        obj["first_digit"] = fd_by_i.get(i, la).to(torch.float16).numpy()
        out_objs.append(cast(PredObject, obj))
    rec = adapter.item_record(record, meta, out_objs, n_predictions=len(predictions))

    del attn
    state.qpos = state.modidx = state.captured = None
    torch.cuda.empty_cache(); gc.collect()
    return rec
