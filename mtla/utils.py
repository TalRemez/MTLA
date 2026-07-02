"""Small generic helpers shared across the package.

Home for the parameter-free primitives that several modules need and that don't belong to any
one of them: spatial/temporal IoU, the grouped-query-attention KV repeat, and the
offset-mapping tokenization helpers used by every model adapter's response-token finder. Topical
modules import from here rather than re-defining these.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------
def iou(b1: list[float], b2: list[float]) -> float:
    """Intersection-over-union of two axis-aligned boxes.

    Args:
        b1: First box as ``[x1, y1, x2, y2]`` (top-left, bottom-right).
        b2: Second box, same format.

    Returns:
        IoU in ``[0, 1]`` (intersection area / union area); ``0.0`` for non-overlapping or
        degenerate boxes. A tiny epsilon guards against zero division.
    """
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def tiou(a: list[float], b: list[float]) -> float:
    """Temporal intersection-over-union of two time spans.

    Args:
        a: First span as ``[t_start, t_end]`` in seconds.
        b: Second span, same format.

    Returns:
        Temporal IoU in ``[0, 1]`` (overlap duration / union duration); ``0.0`` when the spans
        do not overlap or the union is empty.
    """
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Grouped-query attention
# ---------------------------------------------------------------------------
def repeat_kv(hidden_states: "torch.Tensor", n_rep: int) -> "torch.Tensor":
    """Expand grouped-query-attention KV heads to the full number of attention heads.

    Repeats each key/value head ``n_rep`` times so a grouped-query attention tensor lines up with
    the query heads. Byte-for-byte identical to the per-family ``repeat_kv`` in HF's modeling files
    (Qwen3, Qwen3-VL), kept here so the MTLA capture path has one shared copy.

    Args:
        hidden_states: KV tensor of shape ``[batch, n_kv_heads, seq, head_dim]``.
        n_rep: How many times to repeat each KV head (``n_attention_heads // n_kv_heads``).

    Returns:
        Tensor of shape ``[batch, n_kv_heads * n_rep, seq, head_dim]``; the input itself when
        ``n_rep == 1``.
    """
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, n_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Offset-mapping tokenization helpers
# ---------------------------------------------------------------------------
# Model adapters locate a prediction's response tokens Q_p by char span. A fast tokenizer's
# ``return_offsets_mapping=True`` gives per-token ``(char_start, char_end)``; these helpers turn
# char spans into token indices so each adapter's `find_*_token_*` stops re-rolling the loop.
def tokens_overlapping_char_span(
    offsets: list[tuple[int, int]], lo: int, hi: int
) -> list[int]:
    """Find the tokens whose characters fall inside a character range.

    Uses a fast tokenizer's offset mapping to turn a character span (e.g. a matched
    ``[start, end]`` timestamp) into the response-token indices covering it.

    Args:
        offsets: Per-token ``(char_start, char_end)`` pairs from ``return_offsets_mapping=True``,
            indexed by token position.
        lo: Inclusive start of the character range.
        hi: Exclusive end of the character range.

    Returns:
        Token indices whose ``[char_start, char_end)`` overlaps ``[lo, hi)``, in token order.
    """
    return [ti for ti, (ts, te) in enumerate(offsets) if ts < hi and te > lo]
