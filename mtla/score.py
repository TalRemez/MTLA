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

# Default middle-layer band. L8-21 (14 layers) is the paper default, used for both models here
# (Qwen3-VL and InternVL3.5-8B, 36 layers each). Pass ``band=None`` to reduce over all layers.
DEFAULT_BAND: list[int] = list(range(8, 22))


def reduce_band(
    attn: np.ndarray | Sequence | None,
    band: Sequence[int] | None = DEFAULT_BAND,
) -> float | np.ndarray:
    """Collapse a per-prediction ``[L, H]`` attention aggregate to one scalar score (paper eq. 4).

    Keeps only the band's layers, then means over heads and over those layers. This is the final
    MTLA (or SVAR) score: ``attn[band].mean(over heads).mean(over layers)``.

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
