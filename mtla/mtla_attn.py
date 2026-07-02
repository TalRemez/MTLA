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

Public API (in file order):
  * `mtla_localized_attention(attn, region_idx)` — the pure MTLA math (eqs. 2-3).
  * `compute_mtla(adapter, record, ctx, rank)` — per-item driver (image boxes or video windows);
    builds inputs, runs one captured forward, reduces per prediction, returns the saved record.
  * `CaptureState` / `install_capture(module_path, state)` — the attention-capture machinery
    `compute_mtla` relies on (below it, since it is plumbing rather than the interface).

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

import numpy as np
import torch

from mtla.types import BuildInputs, Ctx, ItemRecord, PredObject, TokenRange

if TYPE_CHECKING:
    from mtla.models.base import ModelAdapter
    from mtla.types import GenRecord


# ---------------------------------------------------------------------------
# The MTLA math (paper eqs. 2-3) — pure, independent, model-agnostic.
# ---------------------------------------------------------------------------
def mtla_localized_attention(
    attn: torch.Tensor, region_idx: torch.Tensor
) -> torch.Tensor:
    """Localized attention for one prediction (paper eqs. 2-3).

    Args:
        attn:       ``[L, H, Q_p, n_mod]`` attention from the prediction's response tokens Q_p to
                    the input modality tokens, per layer ``L`` and head ``H``.
        region_idx: indices into the ``n_mod`` axis that fall inside the proposal region M(R_p).

    Returns:
        ``[L, H]`` localized attention: summed over the region (eq. 2), meaned over Q_p (eq. 3).
    """
    x = attn.index_select(
        -1, region_idx
    )  # keep modality tokens inside M(R_p)  [L,H,Q_p,|R|]
    x = x.sum(dim=-1)  # sum over the region   -> LA   (eq.2) [L,H,Q_p]
    x = x.mean(dim=-1)  # mean over Q_p         -> MTLA (eq.3) [L,H]
    return x


# ---------------------------------------------------------------------------
# Per-item driver: one item's response -> its feature-shard record.
# ---------------------------------------------------------------------------
# `compute_mtla` is the whole extraction for one item; it is identical for image boxes and video
# spans — only the adapter callbacks it calls differ:
#   build_inputs      preprocess + build the prompt, PARSE the response into predictions +
#                     hallucination flags, and locate the modality tokens  (None to skip the item)
#   query_tokens      per prediction, its response tokens Q_p (label/coord, or window digits)
#   region_mask       the modality-token indices inside one prediction's region M(R_p)
#   forward_kwargs    kwargs for the single captured forward
#   prediction_record / item_record   how to shape the saved per-prediction / top-level record


@dataclass
class _Pred:
    """One prediction as it flows through the driver, so no parallel arrays are kept aligned by hand.

    ``qp`` holds its absolute Q_p positions (empty when its response tokens couldn't be located — it
    then stays a zero-filled, ``extracted=False`` candidate); ``x1`` is its first coordinate/digit
    position; ``la`` / ``fd`` are the ``[L, H]`` reductions, filled once the attention is captured.
    """

    idx: int  # index into the parsed `predictions` list
    pred: Any  # the Prediction (region + label)
    qp: list[int]  # absolute Q_p positions ([] if not locatable)
    x1: int | None = None  # first coordinate/digit position, or None
    la: "np.ndarray | None" = None  # local_attention  (over all Q_p tokens)
    fd: "np.ndarray | None" = None  # first_digit      (over the x1 token only)


def _slots(
    predictions: list,
    token_ranges: list[TokenRange | None],
    prompt_len: int,
    total_len: int,
) -> list[_Pred]:
    """One ``_Pred`` per prediction, carrying it plus its absolute Q_p positions.

    ``query_tokens`` reports each prediction's Q_p as offsets within the response; the teacher-forced
    sequence is ``[prompt, response]``, so offsets are shifted past the prompt and dropped past the
    sequence end. A prediction whose tokens can't be located keeps ``qp=[]`` (rather than being
    dropped) so the shard still emits a candidate for it.

    Args:
        predictions: the item's parsed predictions, index-aligned with ``token_ranges``.
        token_ranges: Per-prediction ``TokenRange`` (or ``None``) from the adapter's ``query_tokens``.
        prompt_len: Number of prompt tokens preceding the response in the teacher-forced sequence.
        total_len: Length of the full ``[prompt, response]`` sequence (clamp bound).

    Returns:
        One ``_Pred`` per input prediction, in the same order.
    """
    slots: list[_Pred] = []
    for i, tr in enumerate(token_ranges):
        qp: list[int] = []
        x1: int | None = None
        if tr is not None and tr["first_label_tok"] is not None:
            label = [
                prompt_len + t for t in tr["label_toks"] if prompt_len + t < total_len
            ]
            coord = [
                prompt_len + t for t in tr["coord_toks"] if prompt_len + t < total_len
            ]
            qp = sorted(set((label or [prompt_len + tr["first_label_tok"]]) + coord))
            x1 = coord[0] if coord else None
        slots.append(_Pred(idx=i, pred=predictions[i], qp=qp, x1=x1))
    return slots


def compute_mtla(
    adapter: "ModelAdapter", record: "GenRecord", ctx: Ctx, rank: int = 0
) -> "ItemRecord | None":
    """Compute MTLA for every prediction in one item and return its feature-shard record.

    Build the input and parse the response into predictions (``build_inputs``), teacher-force the
    prompt+response through one attention-capturing forward, then per prediction take its Q_p rows
    out of the captured attention and reduce them with ``mtla_localized_attention`` to an ``[L, H]``
    array. Two arrays are stored per prediction: ``local_attention`` (over all Q_p tokens) and
    ``first_digit`` (the first coordinate/digit token only). This is the per-item core of the
    extract stage, and it is identical for image detection (boxes) and video grounding (spans) —
    only the adapter callbacks differ.

    Args:
        adapter: the resolved model adapter supplying the task-specific callbacks
            (``build_inputs``, ``query_tokens``, ``region_mask``, ``forward_kwargs``).
        record: a self-contained generation record ``{id, prompt, response, gt, extra}`` for one item.
        ctx: the extraction context (model, tokenizer, capture state, device, layer/head counts,
            task); see ``mtla.types.Ctx``.
        rank: worker rank for multi-GPU logging; does not affect the result.

    Returns:
        The ``ItemRecord`` to save (one ``PredObject`` per prediction with its ``[L, H]`` arrays), or
        ``None`` when there is nothing to build or no prediction had locatable response tokens.
    """
    device, tokenizer, model = ctx["device"], ctx["tokenizer"], ctx["model"]
    n_layers, n_heads = ctx["n_layers"], ctx["n_heads"]
    # Let a multi-task model's callbacks dispatch on the task (single-task models ignore it).
    adapter._task = ctx.get("task", adapter.tasks[0] if adapter.tasks else None)

    inp: BuildInputs | None = adapter.build_inputs(record, ctx, rank)
    if inp is None:
        return None
    response, predictions, meta = inp["response"], inp["predictions"], inp["meta"]
    prompt_ids = inp["prompt_ids"]

    # Teacher-force [prompt, response], then locate each prediction's Q_p (aligned with predictions).
    resp_ids = torch.tensor(
        tokenizer(response, add_special_tokens=False)["input_ids"],
        dtype=prompt_ids.dtype,
        device=device,
    )
    full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
    total_len = full_ids.shape[1]
    token_ranges = adapter.query_tokens(response, predictions, tokenizer)
    assert len(token_ranges) == len(
        predictions
    ), f"query_tokens must align with predictions: {len(token_ranges)} vs {len(predictions)}"
    slots = _slots(predictions, token_ranges, prompt_ids.shape[0], total_len)
    if not any(s.qp for s in slots):
        return None

    # Capture only the rows (union of all Q_p) and cols (modality tokens) the reduction reads.
    query_positions = sorted({p for s in slots for p in s.qp})
    row_of = {pos: r for r, pos in enumerate(query_positions)}
    attn = _run_captured_forward(
        model,
        ctx["state"],
        adapter.forward_kwargs(full_ids, total_len, device, inp),
        query_positions,
        inp["modality_idx_l"],
        n_layers,
        device,
        rank,
    )  # [L, H, N_q, n_mod]
    if attn is None:
        return None

    def reduce_to_region(positions: list[int], region: "torch.Tensor") -> "np.ndarray":
        """Reduce this prediction's captured attention onto its region M(R_p) -> [L,H] (eqs. 2-3);
        ``positions`` are absolute Q_p positions, mapped back to the compressed capture rows.
        """
        rows = torch.tensor(
            [row_of[p] for p in positions], dtype=torch.long, device=device
        )
        out = mtla_localized_attention(attn.index_select(2, rows), region)
        return out.to(torch.float16).cpu().numpy()

    # Reduce per prediction: local_attention over all Q_p tokens, first_digit over the x1 token only.
    for s in slots:
        if not s.qp:
            continue
        region = torch.tensor(
            adapter.region_mask(s.pred, meta), dtype=torch.long, device=device
        )
        s.la = reduce_to_region(s.qp, region)
        s.fd = reduce_to_region([s.x1], region) if s.x1 is not None else s.la

    # Emit an object for EVERY prediction so the shards are the complete candidate set at score time;
    # a prediction whose Q_p couldn't be located gets a zero array and `extracted=False`.
    zeros = np.zeros((n_layers, n_heads), dtype=np.float16)
    out_objs: list[PredObject] = []
    for s in slots:
        obj = adapter.prediction_record(s.pred, s.idx, meta)
        obj["is_hallucinated"] = bool(inp["hallu_flags"][s.idx])
        obj["extracted"] = s.la is not None
        obj["local_attention"] = s.la if s.la is not None else zeros
        obj["first_digit"] = s.fd if s.fd is not None else zeros
        out_objs.append(cast(PredObject, obj))
    rec = adapter.item_record(record, meta, out_objs, n_predictions=len(predictions))

    del attn
    ctx["state"].qpos = ctx["state"].modidx = ctx["state"].captured = None
    torch.cuda.empty_cache()
    gc.collect()
    return rec


# ---------------------------------------------------------------------------
# Attention capture: get the model's real weights to the function above.
# ---------------------------------------------------------------------------
@dataclass
class CaptureState:
    """State shared between the capture wrapper and the per-item driver.

    ``layer_ids`` maps each LM decoder self-attention module id to its layer index (so the wrapper
    ignores vision-tower attention that routes through the same function). Before each forward the
    driver sets ``qpos`` / ``modidx`` (the rows/cols to keep) and allocates ``captured``; after the
    forward it reads ``captured`` (a CPU list of ``[H, N_q, n_mod]`` tensors, one per layer).
    """

    layer_ids: dict = field(default_factory=dict)
    active: bool = False
    qpos: torch.Tensor | None = (
        None  # query positions to keep (rows), on the model device
    )
    modidx: torch.Tensor | None = (
        None  # modality-token positions to keep (cols), on device
    )
    captured: list | None = None  # per-layer [H, N_q, n_mod], fp32, on CPU


def make_capture_forward(state: CaptureState, orig_forward: Callable) -> Callable:
    """Build a drop-in replacement for a model's ``eager_attention_forward`` that captures weights.

    The returned wrapper calls the original forward verbatim (so the attention math is never
    reimplemented and cannot drift from the model), then — only while ``state.active`` and only for
    the registered LM decoder layers — slices the returned weights down to the query rows
    (``state.qpos``) and modality columns (``state.modidx``) on-GPU and stashes that small tensor in
    ``state.captured``. Keeping only the slice means peak memory stays at one layer's full map.

    Args:
        state: Shared capture state; the wrapper reads ``active``/``layer_ids``/``qpos``/``modidx``
            and writes into ``captured``.
        orig_forward: The model's stock ``eager_attention_forward`` to delegate to.

    Returns:
        A wrapper with the same ``(module, *args, **kwargs) -> (attn_output, attn_weights)`` contract
        as the original, safe to monkeypatch in its place.
    """

    def wrapper(module: Any, *args: Any, **kwargs: Any) -> tuple:
        attn_output, attn_weights = orig_forward(module, *args, **kwargs)
        if state.active and attn_weights is not None and id(module) in state.layer_ids:
            assert state.captured is not None  # set by the driver before each forward
            w = attn_weights[0]  # [H, Q, K]
            if state.qpos is not None:
                w = w.index_select(1, state.qpos)  # [H, N_q, K]
            if state.modidx is not None:
                w = w.index_select(2, state.modidx)  # [H, N_q, n_mod]
            # Keep the tiny slice on the compute device: the full [H,Q,K] map (the memory wall) is
            # freed when the stock forward's output goes out of scope after this returns; only the
            # pre-sliced [H,N_q,n_mod] is retained, so holding all L layers is cheap (~1 GB).
            state.captured[state.layer_ids[id(module)]] = w.detach()
        return attn_output, attn_weights

    return wrapper


def install_capture(module_path: str, state: CaptureState) -> Any:
    """Monkeypatch a modeling module's ``eager_attention_forward`` with the capture wrapper.

    The target model must be loaded with ``attn_implementation="eager"`` so its attention actually
    routes through this function. The patch is process-global (each extract worker is its own
    process), so only one model can be captured per process.

    Args:
        module_path: Dotted path of the HF modeling module to patch, e.g.
            ``"transformers.models.qwen3_vl.modeling_qwen3_vl"``.
        state: The capture state the installed wrapper will read from and write into.

    Returns:
        The imported (and now patched) module object.
    """
    mod = importlib.import_module(module_path)
    mod.eager_attention_forward = make_capture_forward(  # type: ignore[attr-defined]
        state, mod.eager_attention_forward
    )
    return mod


def _run_captured_forward(
    model: Any,
    state: CaptureState,
    fk: dict,
    query_positions: list[int],
    modality_idx: list[int],
    n_layers: int,
    device: str,
    rank: int,
) -> "torch.Tensor | None":
    """Run one attention-capturing forward and return the stacked per-layer attention slice.

    Sets the rows/cols the capture wrapper keeps, runs a single ``torch.no_grad`` forward (which is
    what materializes the attention the wrapper slices per layer), then stacks the per-layer results.
    A failed forward is caught and reported, not raised, so one bad item cannot kill the shard.

    Args:
        model: The loaded HF model (its attention already patched by :func:`install_capture`).
        state: Capture state to configure (``qpos``/``modidx``/``captured``/``active``).
        fk: Forward kwargs from the adapter's ``forward_kwargs`` (input ids, pixel values, mask, ...).
        query_positions: Absolute positions of the union of all Q_p tokens (the rows to keep).
        modality_idx: Positions of the input-modality tokens (the columns to keep).
        n_layers: Number of decoder layers (size of the per-layer capture buffer).
        device: Device to place the row/col index tensors on.
        rank: Worker rank, used only for the skip log line.

    Returns:
        The captured attention stacked as ``[L, H, N_q, n_mod]`` (fp32, on ``device``), or ``None``
        if the forward raised.
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
