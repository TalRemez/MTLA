"""The MTLA computation (shared by image detection and video grounding).

This module *is* MTLA. The paper's score (appendix pseudo-code) reads a prediction's own
attention from its response tokens Q_p to the input modality tokens, restricted to the tokens
inside its proposal region M(R_p)::

    x = attn[..., region_idx]   # keep modality tokens inside the region M(R_p)
    x = x.sum(-1)               # sum over the region     -> Localized Attention   (eq. 2)
    x = x.mean(-1)              # mean over Q_p            -> Multi-Token LA         (eq. 3)
    # (`reduce_band` finishes the score, eq. 4, at score time: mean over heads + a layer band)

`mtla_localized_attention` below is exactly that — an independent, model-agnostic function that
takes a prediction's `[L, H, Q_p, n_mod]` attention slice and its region mask and returns the
`[L, H]` localized-attention array. `reduce_band` collapses that `[L, H]` array to the final scalar
score (eq. 4); it runs later, on CPU, in the evaluation stage. Everything else in this module just
*gets the model's real attention weights to those two functions*.

Getting the weights is the only hard part. A fused attention kernel never exposes the weights, and
`output_attentions=True` materializes every layer's `[H, Q, Q]` map at once — hundreds of GB for a
video clip. So we install a thin **capture** wrapper around the model's own eager-attention
forward: it runs the stock forward unchanged, then slices the returned weights down to just the
rows/cols MTLA reads (the query positions Q_p and the modality tokens) and keeps only that small
`[H, N_q, n_mod]` slice. The full `[H, Q, K]` map is freed as soon as the stock forward's output
goes out of scope, so GPU peak stays at one layer even though all L slices are retained (~1 GB).
The reduction then runs on those retained slices. Because the wrapper reuses the model's own
forward verbatim, the captured weights are bit-for-bit what the model computed, for any model /
transformers version.

Public API (in file order):
  * `mtla_localized_attention(attn, region_idx)` — the pure MTLA math (eqs. 2-3).
  * `reduce_band(attn, band)` / `DEFAULT_BAND` — the score-time head+layer reduction (eq. 4).
  * `compute_mtla(adapter, record, ctx, rank)` — per-item driver (image boxes or video windows);
    builds inputs, runs one captured forward, reduces per prediction, returns the saved record.
  * `CaptureState` / `install_capture(module_path, state)` — the attention-capture machinery
    `compute_mtla` relies on (below it, since it is plumbing rather than the interface).

The model/task-specific pieces (prompt build, prediction parsing + hallucination flags,
Q_p token finding, region mask, forward kwargs, record fields) live in the model adapters'
callbacks (build_extraction_inputs / locate_proposal_tokens / proposal_region_attn_mask /
hf_extraction_kwargs / prediction_record / item_record); this module is the common math and
plumbing.
"""

from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

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
        ``[L, H]`` fp32 localized attention: summed over the region (eq. 2), meaned over Q_p (eq. 3).
        The region sum and Q_p mean accumulate in fp32 (``dtype=torch.float32``) even when ``attn`` is
        bf16, so summing many region tokens keeps precision without upcasting the whole tensor — the
        input can stay bf16 on GPU; only this reduced result is fp32.
    """
    x = attn.index_select(
        -1, region_idx
    )  # keep modality tokens inside M(R_p)  [L,H,Q_p,|R|]
    x = x.sum(
        dim=-1, dtype=torch.float32
    )  # sum over region -> LA (eq.2) [L,H,Q_p], fp32 accum
    x = x.mean(
        dim=-1
    )  # mean over Q_p -> MTLA (eq.3) [L,H]; x is already fp32 so this means in fp32
    return x


# Default middle-layer band. L8-21 (14 layers) is the paper default, used for both models here
# (Qwen3-VL and InternVL3.5-8B, 36 layers each). Pass ``band=None`` to reduce over all layers.
DEFAULT_BAND: list[int] = list(range(8, 22))


def reduce_band(
    attn: np.ndarray | Sequence | None,
    band: Sequence[int] | None = DEFAULT_BAND,
) -> float | np.ndarray:
    """Collapse a per-prediction ``[L, H]`` localized-attention array to one scalar score (eq. 4).

    The score-time tail of MTLA: ``mtla_localized_attention`` produced the ``[L, H]`` array during
    extraction (eqs. 2-3); this keeps only the band's layers, then means over heads and over those
    layers — ``attn[band].mean(over heads).mean(over layers)`` (mean vs sum over a fixed band is a
    constant factor, so it changes no ranking / AUROC / mAP). Higher means more grounded. Pure NumPy
    and CPU-side, so the evaluation stage calls it without loading a model.

    Args:
        attn: Attention aggregate of shape ``[L, H]`` for a single prediction, or ``[N, L, H]`` for
            a batch (``L`` layers, ``H`` heads). ``None`` is treated as an absent prediction.
        band: Layer indices to keep before reducing; ``None`` uses every layer. Out-of-range
            indices are dropped so one band works across model depths (36-layer vs 42-layer, ...).

    Returns:
        ``0.0`` if ``attn`` is ``None``; a Python ``float`` for a single ``[L, H]`` input; or a
        ``[N]`` array of floats for a batched ``[N, L, H]`` input.

    Raises:
        ValueError: If ``attn`` is not 2-D or 3-D, or if ``band`` selects no valid layer for the
            given tensor depth.
    """
    if attn is None:
        return 0.0
    a = np.asarray(attn, dtype=np.float32)
    single = a.ndim == 2
    if single:
        a = a[None, ...]  # -> [1, L, H]
    if a.ndim != 3:
        raise ValueError(f"expected [L,H] or [N,L,H], got shape {a.shape}")

    n_layers = a.shape[1]
    if band is None:
        layers = list(range(n_layers))
    else:
        layers = [l for l in band if 0 <= l < n_layers]
        if not layers:
            raise ValueError(
                f"band {list(band)} has no valid layer for {n_layers}-layer tensor"
            )

    scores = (
        a[:, layers, :].mean(axis=2).mean(axis=1)
    )  # mean over heads, mean over band
    return float(scores[0]) if single else scores


# ---------------------------------------------------------------------------
# Per-item driver: one item's response -> its feature-shard record.
# ---------------------------------------------------------------------------
# `compute_mtla` is the whole extraction for one item; it is identical for image boxes and video
# spans — only the adapter callbacks it calls differ:
#   build_extraction_inputs    preprocess + build the prompt, PARSE the response into
#                              predictions + hallucination flags, locate the modality tokens
#                              (None to skip the item)
#   locate_proposal_tokens     per prediction, its response tokens Q_p (label/coord, or digits)
#   proposal_region_attn_mask  the modality-token indices inside one prediction's region M(R_p)
#   hf_extraction_kwargs       kwargs for the single captured forward
#   prediction_record / item_record   how to shape the saved per-prediction / top-level record


@dataclass
class _Pred:
    """One prediction as it flows through the driver, so no parallel arrays are kept aligned by hand.

    Carries the prediction's response-token positions split into the paper's Q_p **slots** (absolute
    positions in the teacher-forced sequence):
      * ``qp``    — all of the prediction's tokens (label + coords); the ``all`` slot. Empty when the
                    tokens couldn't be located, in which case it stays a zero-filled,
                    ``extracted=False`` candidate.
      * ``coord`` — the coordinate/digit tokens (the ``digits`` slot).
      * ``label`` — the label tokens (the ``label`` slot; empty for video spans, which have no label).
      * ``x1``    — the first coordinate/digit token (the ``first`` slot), or ``None``.
    ``feats`` holds the extracted ``{"<slot>_<signal>": [L, H]}`` aggregates (slot x local/global),
    filled once the attention is captured.
    """

    idx: int  # index into the parsed `predictions` list
    pred: Any  # the Prediction (region + label)
    qp: list[int]  # all Q_p positions ([] if not locatable) -> "all" slot
    coord: list[int] = field(
        default_factory=list
    )  # coord/digit positions -> "digits" slot
    label: list[int] = field(default_factory=list)  # label positions -> "label" slot
    x1: int | None = None  # first coordinate/digit position -> "first" slot
    feats: dict = field(
        default_factory=dict
    )  # {"<slot>_<signal>": [L, H]}, filled after capture


def _slots(
    predictions: list,
    token_ranges: list[TokenRange | None],
    prompt_len: int,
    total_len: int,
) -> list[_Pred]:
    """One ``_Pred`` per prediction, carrying it plus its absolute Q_p positions.

    ``locate_proposal_tokens`` reports each prediction's Q_p as offsets within the response;
    the teacher-forced sequence is ``[prompt, response]``, so offsets are shifted past the
    prompt and dropped past the sequence end. A prediction whose tokens can't be located keeps
    ``qp=[]`` (rather than being dropped) so the shard still emits a candidate for it.

    Args:
        predictions: the item's parsed predictions, index-aligned with ``token_ranges``.
        token_ranges: Per-prediction ``TokenRange`` (or ``None``) from the adapter's
            ``locate_proposal_tokens``.
        prompt_len: Number of prompt tokens preceding the response in the teacher-forced sequence.
        total_len: Length of the full ``[prompt, response]`` sequence (clamp bound).

    Returns:
        One ``_Pred`` per input prediction, in the same order.
    """
    slots: list[_Pred] = []
    for i, tr in enumerate(token_ranges):
        qp: list[int] = []
        coord: list[int] = []
        label: list[int] = []
        x1: int | None = None
        if tr is not None and tr["first_label_tok"] is not None:
            label = [
                prompt_len + t for t in tr["label_toks"] if prompt_len + t < total_len
            ]
            coord = [
                prompt_len + t for t in tr["coord_toks"] if prompt_len + t < total_len
            ]
            # "all" = label + coords (fall back to the first label token if no label toks resolved).
            qp = sorted(set((label or [prompt_len + tr["first_label_tok"]]) + coord))
            x1 = coord[0] if coord else None
        slots.append(
            _Pred(idx=i, pred=predictions[i], qp=qp, coord=coord, label=label, x1=x1)
        )
    return slots


def compute_mtla(
    adapter: "ModelAdapter", record: "GenRecord", ctx: Ctx, rank: int = 0
) -> "ItemRecord | None":
    """Compute MTLA for every prediction in one item and return its feature-shard record.

    Build the input and parse the response into predictions (``build_extraction_inputs``),
    teacher-force the prompt+response through one attention-capturing forward, then per prediction
    reduce the captured attention with ``mtla_localized_attention`` to ``[L, H]`` arrays — one per
    **slot x signal**: each Q_p slot (``all`` / ``digits`` / ``label`` / ``first``) reduced against
    both the local region (the MTLA score) and all modality tokens (the SVAR/global baseline), stored
    under ``<slot>_<signal>`` keys (plus the ``local_attention`` / ``first_digit`` back-compat
    aliases). This is the per-item core of the extract stage, and it is identical for image detection
    (boxes) and video grounding (spans) — only the adapter callbacks differ.

    Args:
        adapter: the resolved model adapter supplying the task-specific callbacks
            (``build_extraction_inputs``, ``locate_proposal_tokens``,
            ``proposal_region_attn_mask``, ``hf_extraction_kwargs``).
        record: a self-contained generation record ``{id, prompt, response, gt, extra}`` for one item.
        ctx: the extraction context (model, tokenizer, capture state, device, layer/head
            counts); see ``mtla.types.Ctx``.
        rank: worker rank for multi-GPU logging; does not affect the result.

    Returns:
        The ``ItemRecord`` to save (one ``PredObject`` per prediction with its ``[L, H]`` arrays), or
        ``None`` when there is nothing to build or no prediction had locatable response tokens.
    """
    device, tokenizer, model = ctx["device"], ctx["tokenizer"], ctx["model"]
    n_layers, n_heads = ctx["n_layers"], ctx["n_heads"]
    inp: BuildInputs | None = adapter.build_extraction_inputs(record, ctx, rank)
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
    token_ranges = adapter.locate_proposal_tokens(response, predictions, tokenizer)
    assert len(token_ranges) == len(
        predictions
    ), f"locate_proposal_tokens must align with predictions: {len(token_ranges)} vs {len(predictions)}"
    slots = _slots(predictions, token_ranges, prompt_ids.shape[0], total_len)
    if not any(s.qp for s in slots):
        return None

    # Capture only the rows (union of all Q_p) and cols (modality tokens) the reduction reads.
    query_positions = sorted({p for s in slots for p in s.qp})
    row_of = {pos: r for r, pos in enumerate(query_positions)}
    attn = _run_captured_forward(
        model,
        ctx["state"],
        adapter.hf_extraction_kwargs(full_ids, total_len, device, inp),
        query_positions,
        inp["modality_idx_l"],
        n_layers,
        device,
        rank,
    )  # [L, H, N_q, n_mod]
    if attn is None:  # this item ran out of GPU memory (state.n_oom bumped); skip it
        return None

    all_mod = torch.arange(
        attn.shape[-1], device=device
    )  # global (SVAR): all modality tokens

    def reduce(positions: list[int], cols: "torch.Tensor") -> "np.ndarray":
        """Reduce a Q_p slot onto a modality-token set -> [L, H] (eqs. 2-3).

        ``positions`` are absolute Q_p positions (mapped back to the compressed capture rows via
        ``row_of``); ``cols`` are the modality columns to sum over — the proposal region for the
        local signal, or all tokens for the global/SVAR signal. The slice stays bf16 on GPU; the
        reduction accumulates in fp32 (see ``mtla_localized_attention``) and only the small [L,H]
        result is upcast, so nothing large is upcast on-device. Heads survive to the [L,H] result.
        """
        rows = torch.tensor(
            [row_of[p] for p in positions], dtype=torch.long, device=device
        )
        sl = attn.index_select(2, rows)  # [L,H,|slot|,n_mod], bf16 on GPU
        return mtla_localized_attention(sl, cols).cpu().numpy()  # [L, H] fp32

    # For every prediction compute each Q_p SLOT (all / digits / label / first) against both SIGNALS:
    # local (attention inside the proposal region M(R_p), the MTLA score) and global (attention over
    # all modality tokens, the SVAR baseline). All come from the one captured [L,H,N_q,n_mod] tensor.
    for s in slots:
        if not s.qp:
            continue
        region = torch.tensor(
            adapter.proposal_region_attn_mask(s.pred, meta),
            dtype=torch.long,
            device=device,
        )
        slot_positions = {
            "all": s.qp,
            "digits": s.coord,
            "label": s.label,
            "first": [s.x1] if s.x1 is not None else [],
        }
        for slot, pos in slot_positions.items():
            if not pos:  # skip empty slots (e.g. label for video spans)
                continue
            s.feats[f"{slot}_local"] = reduce(pos, region)
            s.feats[f"{slot}_global"] = reduce(pos, all_mod)

    # Emit an object for EVERY prediction so the shards are the complete candidate set at score time;
    # a prediction whose Q_p couldn't be located gets zero arrays and `extracted=False`. Back-compat
    # aliases: `local_attention` = all_local (the MTLA score COCO reads), `first_digit` = first_local
    # (the video signal); the full slot x signal set is stored alongside for the paper's tables.
    zeros = np.zeros((n_layers, n_heads), dtype=np.float32)
    out_objs: list[PredObject] = []
    for s in slots:
        obj = adapter.prediction_record(s.pred, s.idx, meta)
        obj["is_hallucinated"] = bool(inp["hallu_flags"][s.idx])
        obj["extracted"] = bool(s.feats)
        for name, arr in s.feats.items():
            obj[name] = arr
        obj["local_attention"] = s.feats.get("all_local", zeros)
        obj["first_digit"] = s.feats.get("first_local", s.feats.get("all_local", zeros))
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
    forward it reads ``captured`` (a list of ``[H, N_q, n_mod]`` tensors, one per layer, kept on the
    compute device in the model's dtype).
    """

    layer_ids: dict = field(default_factory=dict)
    active: bool = False
    qpos: torch.Tensor | None = (
        None  # query positions to keep (rows), on the model device
    )
    modidx: torch.Tensor | None = (
        None  # modality-token positions to keep (cols), on device
    )
    captured: list | None = (
        None  # per-layer [H, N_q, n_mod], model dtype, on the compute device
    )
    n_oom: int = 0  # count of items skipped for CUDA OOM (the extract worker caps this)


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
    rank: int = 0,
) -> "torch.Tensor | None":
    """Run one attention-capturing forward and return the stacked per-layer attention slice.

    Sets the rows/cols the capture wrapper keeps, runs a single ``torch.no_grad`` forward (which is
    what materializes the attention the wrapper slices per layer), then stacks the per-layer results.

    The one tolerated failure is CUDA out-of-memory: a pathologically long response materializes a
    full ``[H, Q, K]`` map for one layer that does not fit, which is a property of that single item,
    not a bug. It is caught, logged, and reported as ``None`` (the driver skips the item and the
    extract worker caps how many such skips it tolerates before failing the run). Every OTHER
    exception propagates — it is a real model/input bug, not a per-item hiccup to swallow.

    Args:
        model: The loaded HF model (its attention already patched by :func:`install_capture`).
        state: Capture state to configure (``qpos``/``modidx``/``captured``/``active``).
        fk: Forward kwargs from the adapter's ``hf_extraction_kwargs`` (input ids, pixel
            values, mask, ...).
        query_positions: Absolute positions of the union of all Q_p tokens (the rows to keep).
        modality_idx: Positions of the input-modality tokens (the columns to keep).
        n_layers: Number of decoder layers (size of the per-layer capture buffer).
        device: Device to place the row/col index tensors on.
        rank: Worker rank, used only for the OOM-skip log line.

    Returns:
        The captured attention stacked as ``[L, H, N_q, n_mod]`` in the model dtype, on ``device``
        (the fp32 upcast is deferred to the small per-prediction reduction), or ``None`` if the item
        ran out of GPU memory.
    """
    state.qpos = torch.tensor(query_positions, dtype=torch.long, device=device)
    state.modidx = torch.tensor(modality_idx, dtype=torch.long, device=device)
    state.captured = [None] * n_layers
    try:
        state.active = True
        with torch.no_grad():
            model(**fk)
        # Stack INSIDE the OOM guard: for a response with many predictions the union of Q_p rows
        # (N_q) is large, and even the [L, H, N_q, n_mod] stack can exhaust memory after the forward
        # succeeds. That is still a "this one item is too big" event, so it is skipped like a forward
        # OOM rather than crashing the shard. Keep the model dtype here (do NOT .float() the whole
        # [L,H,N_q,n_mod] tensor — that doubled peak memory and caused ~30 GiB OOMs on 200-box COCO
        # images); the fp32 upcast happens later on the tiny per-prediction slice in the reduction.
        attn = torch.stack(state.captured, dim=0)
    except torch.cuda.OutOfMemoryError as e:
        state.n_oom += 1
        n_qp = len(query_positions)
        print(
            f"[worker {rank}] OOM on one item (seq_len={fk['input_ids'].shape[-1]}, "
            f"n_qp={n_qp}, n_mod={len(modality_idx)}); skipping it: {e}",
            flush=True,
        )
        state.captured = None
        torch.cuda.empty_cache()
        return None
    finally:
        state.active = False  # must reset even if the forward raises
    # The per-layer slices are tiny, so stacking + reducing stays on the compute device; upcast to
    # fp32 once above so the MTLA reduction matches the paper's math.
    return attn
