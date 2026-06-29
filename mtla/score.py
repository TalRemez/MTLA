"""Core MTLA scoring: reduce a per-prediction attention tensor to one scalar.

The whole method, once attention has been extracted, is a deterministic, parameter-free
reduction. For a prediction ``p`` the extractor produces, for every transformer layer
``l`` and head ``h``, the attention that ``p``'s output tokens pay to the input-modality
tokens that fall *inside* its proposal region. We average over heads and sum over a fixed
band of middle layers:

    s(p) = sum_{l in band}  mean_h  A[l, h]

where ``A`` is one of the extracted ``[L, H]`` aggregates:

  * ``image_inside_sum``  -> MTLA  (Multi-Token Localized Attention, ours)
  * ``image_sum``         -> GA / SVAR baseline (attention over *all* modality tokens)

A higher score means the prediction looks more grounded; a lower score flags a likely
hallucination. See ``docs/METHOD.md`` for the full derivation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Default middle-layer band. L8-21 (14 layers) is the paper default and is used for
# every image/video model we tested (Qwen3-VL, InternVL: 36 layers; Gemma-4: 42).
# Audio (Audio Flamingo 3, 28 layers) uses all layers; pass ``band=ALL_LAYERS``.
DEFAULT_BAND: list[int] = list(range(8, 22))
ALL_LAYERS = None  # sentinel: use every layer present in the tensor

# Convenience presets keyed by model family (number of decoder layers in parentheses).
LAYER_BANDS = {
    "qwen3-vl-8b": list(range(8, 22)),      # 36 layers
    "internvl3.5-8b": list(range(8, 22)),   # 36 layers
    "gemma-4": list(range(8, 22)),          # 42 layers
    "audio-flamingo-3": None,               # 28 layers, all-layers
}


def reduce_band(
    attn: np.ndarray | Sequence,
    band: Sequence[int] | None = DEFAULT_BAND,
) -> float | np.ndarray:
    """Reduce an attention aggregate to a scalar MTLA/SVAR score.

    Args:
        attn: array of shape ``[L, H]`` for a single prediction, or ``[N, L, H]`` for a
            batch. ``L`` = number of layers, ``H`` = number of heads.
        band: layer indices to keep before summing. ``None`` uses every layer.
            Out-of-range indices are dropped (so the same band works across model sizes).

    Returns:
        A Python ``float`` for a single ``[L, H]`` input, or a ``[N]`` array for a batch.

    The reduction is exactly ``attn[band].mean(axis=heads).sum(axis=layers)``.
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
            raise ValueError(f"band {list(band)} has no valid layer for {n_layers}-layer tensor")

    scores = a[:, layers, :].mean(axis=2).sum(axis=1)  # mean over heads, sum over band
    return float(scores[0]) if single else scores


def apply_slot(record: dict, spec, band=DEFAULT_BAND) -> float:
    """Reduce one signal from an extracted prediction `record` per a model's `SlotSpec`.

    Lets dataset adapters read MTLA/SVAR without naming model-specific record keys: the model
    adapter supplies the `SlotSpec` (block + stat, or the count-weighted "all" recipe).

    spec.combine == "all": count-weighted mean over spec.parts = [(block, count_field), ...] of
    each block's `spec.stat` array, then reduce_band. Otherwise: reduce_band of
    record[spec.block][spec.stat]. Missing blocks fall back gracefully to 0.
    """
    if spec is None:
        return 0.0
    if spec.combine == "all":
        num = None
        denom = 0
        for block, count_field in spec.parts:
            blk = record.get(block)
            if blk is None or spec.stat not in blk:
                continue
            cnt = record.get(count_field, 0) or 0
            arr = np.asarray(blk[spec.stat], dtype=np.float32) * cnt
            num = arr if num is None else num + arr
            denom += cnt
        if num is None:
            return 0.0
        return reduce_band(num / max(denom, 1), band)
    blk = record.get(spec.block, {})
    return reduce_band(blk.get(spec.stat), band)


def mtla_score(record: dict, slot: str = "attn_coord_mean", band=DEFAULT_BAND) -> float:
    """MTLA score for one extracted prediction record (inside-region attention)."""
    return reduce_band(record[slot]["image_inside_sum"], band)


def svar_score(record: dict, slot: str = "attn_coord_mean", band=DEFAULT_BAND) -> float:
    """SVAR / Global-Attention baseline for one record (attention over all tokens)."""
    return reduce_band(record[slot]["image_sum"], band)
