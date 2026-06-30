"""Qwen3-VL video_span extraction helpers (used by Qwen3VLAdapter's video ext_* callbacks).

Mirrors the image-detection helpers in qwen3vl.py, for temporal grounding: build the video input,
re-derive the frame-token grid, map each predicted [start,end] window to the response digit tokens
inside it (its Q_p, image-case parity) and to the modality (frame) tokens inside its time span
(its region M(R_p)). The shared per-item driver in mtla.mtla_attn.compute_mtla does the rest.
"""
from __future__ import annotations

import re

from ..utils import tiou, tokens_overlapping_char_span, digit_token_positions

_TIME = r'(\d{1,3}(?::\d{2})?(?:\.\d+)?)'
_PATTERNS = [
    rf'\[\s*{_TIME}\s*,\s*{_TIME}\s*\]',
    rf'\(\s*{_TIME}\s*,\s*{_TIME}\s*\)',
    rf'from\s+{_TIME}\s*s?\s+to\s+{_TIME}',
    rf'between\s+{_TIME}\s*s?\s+and\s+{_TIME}',
    rf'{_TIME}\s*s?\s*-\s*{_TIME}\s*s',
    rf'{_TIME}\s*s?\s*-\s*{_TIME}',
    rf'{_TIME}\s*s\s+to\s+{_TIME}',
    rf'start[:\s]+{_TIME}\s*s?\s*,?\s*end[:\s]+{_TIME}',
]


def _to_seconds(tok: str) -> float:
    if ":" in tok:
        m, s = tok.split(":")
        return float(m) * 60 + float(s)
    return float(tok)


def parse_windows_with_spans(text: str):
    """Parse [start,end] windows AND their char offsets in `text`, same dedup/ordering as the
    validated parser. Offsets are taken against a length-preserving lowercase copy so they align
    with the original response. Returns (windows, char_spans) aligned by index."""
    t = text.lower()  # length-preserving (do NOT replace "seconds"->"s" here; would shift offsets)
    seen = set(); rows = []  # (window, char_span)
    for p in _PATTERNS:
        for m in re.finditer(p, t):
            try:
                a, b = _to_seconds(m.group(1)), _to_seconds(m.group(2))
            except ValueError:
                continue
            if a > b:
                a, b = b, a
            if a == b:
                continue
            key = (round(a, 2), round(b, 2))
            if key in seen:
                continue
            seen.add(key); rows.append(([a, b], m.span()))
    rows.sort(key=lambda r: r[0][0])  # order by start time — must match parse_spans so the
    windows = [w for w, _ in rows]    # extract-time Q_p attribution aligns with pred_windows[i]
    spans = [s for _, s in rows]
    return windows, spans


def perwindow_qp_tokens(response: str, windows, spans, tokenizer):
    """For each predicted window, the response digit-token indices whose char offsets fall inside
    that window's [start,end] match (its Q_p). Returns a list aligned with `windows`, each a dict
    {first_label_tok, label_toks=[], coord_toks=[...]} — the same shape the image adapter returns,
    so the shared driver's Q_p assembly is identical. A window with no matched digits -> None."""
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out = []
    for (cs, ce) in spans:
        # digit tokens that lie within this window's char span
        toks = [ti for ti in tokens_overlapping_char_span(offsets, cs, ce)
                if any(c.isdigit() for c in response[offsets[ti][0]:offsets[ti][1]])]
        if not toks:
            out.append(None); continue
        out.append({"first_label_tok": toks[0], "label_toks": [], "coord_toks": toks})
    # windows with no char span (shouldn't happen — spans aligns with windows) -> None
    out += [None] * (len(windows) - len(out))
    return out


def span_to_frame_token_indices(span, duration_s, T_tokens, H_tokens, W_tokens):
    """Modality-token indices inside a time `span` M(R_p): the frames whose timestamps fall in the
    span, expanded across all H*W spatial tokens of each such frame. The video token block is laid
    out frame-major: frame t occupies tokens [t*HW : (t+1)*HW). Token t covers time
    t*duration/T_tokens. Returns a flat index list into the n_video=T*H*W modality tokens."""
    import numpy as np
    HW = H_tokens * W_tokens
    if span is None or duration_s <= 0 or T_tokens <= 0:
        return []
    s, e = span
    fs = max(0, int(np.floor(s * T_tokens / duration_s)))
    fe = min(T_tokens, int(np.ceil(e * T_tokens / duration_s)))
    if fe <= fs:
        return []
    return [f * HW + k for f in range(fs, fe) for k in range(HW)]


def hallu_flags_windows(pred_windows, gt_windows, multi):
    """Per-prediction hallucination flags for video. A predicted window is grounded if it overlaps
    some GT window with tIoU >= 0.5 (multi, QVH) or the single GT span (single, Charades)."""
    flags = []
    for w in pred_windows:
        hit = any(tiou(w, g) >= 0.5 for g in gt_windows)
        flags.append(not hit)
    return flags
