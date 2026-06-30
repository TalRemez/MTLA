"""Core MTLA scoring: reduce a per-prediction attention array to one scalar.

The MTLA computation (`mtla.mtla_attn`) already did the modality-token reductions during
extraction, saving each prediction's ``local_attention``: a ``[L, H]`` array of localized
attention (the attention its tokens Q_p pay to the modality tokens inside its proposal region,
summed over the region and meaned over Q_p — paper eqs. 2-3). This module finishes the score
(paper eq. 4) by averaging over heads and over a fixed band of middle layers:

    s(p) = mean_{l in band}  mean_h  local_attention[l, h]

A higher score means the prediction looks more grounded; a lower score flags a likely
hallucination. See the Method section of the README for the full derivation.
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

    The reduction is exactly ``attn[band].mean(axis=heads).mean(axis=layers)``.
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

    scores = a[:, layers, :].mean(axis=2).mean(axis=1)  # mean over heads, mean over band
    return float(scores[0]) if single else scores


def mtla_score(obj: dict, signal: str = "local_attention", band=DEFAULT_BAND) -> float:
    """MTLA score for one extracted prediction object.

    `signal` selects which saved ``[L, H]`` array to reduce:
      * ``local_attention`` (default) — localized attention meaned over all the prediction's
        tokens Q_p; this is MTLA.
      * ``first_digit`` — the same, read at only the first coordinate digit (x1).
    """
    return reduce_band(obj[signal], band)
