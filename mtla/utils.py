"""Small generic helpers shared across the package.

Home for the parameter-free primitives that several modules need and that don't belong to any
one of them: spatial/temporal IoU, the grouped-query-attention KV repeat, and the
offset-mapping tokenization helpers used by every model adapter's response-token finder. Topical
modules import from here rather than re-defining these.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------
def iou(b1, b2) -> float:
    """Spatial IoU of two ``[x1, y1, x2, y2]`` boxes."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def tiou(a, b) -> float:
    """Temporal IoU of two ``[t_start, t_end]`` spans."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Grouped-query attention
# ---------------------------------------------------------------------------
def repeat_kv(hidden_states, n_rep: int):
    """Expand grouped KV heads to full attention heads (HF's ``repeat_kv``).

    ``[batch, n_kv_heads, seq, head_dim]`` -> ``[batch, n_kv_heads * n_rep, seq, head_dim]``.
    Identical to the per-model-family ``repeat_kv`` in HF's modeling files (Qwen3, Qwen3-VL),
    so the MTLA kernel can use one copy instead of taking it as a parameter.
    """
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Offset-mapping tokenization helpers
# ---------------------------------------------------------------------------
# Model adapters locate a prediction's response tokens Q_p by char span. A fast tokenizer's
# ``return_offsets_mapping=True`` gives per-token ``(char_start, char_end)``; these helpers turn
# char spans into token indices so each adapter's `find_*_token_*` stops re-rolling the loop.
def tokens_overlapping_char_span(offsets, lo: int, hi: int) -> list:
    """Token indices whose char span ``[ts, te)`` overlaps the char range ``[lo, hi)``."""
    return [ti for ti, (ts, te) in enumerate(offsets) if ts < hi and te > lo]


def digit_token_positions(offsets, text: str) -> list:
    """Token indices whose text contains a digit (the timestamp/coordinate digit tokens)."""
    return [ti for ti, (ts, te) in enumerate(offsets) if any(c.isdigit() for c in text[ts:te])]
