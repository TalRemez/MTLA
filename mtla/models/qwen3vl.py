"""Qwen3-VL-8B model adapter (video temporal grounding).

Shared by QVHighlights (multi-segment) and Charades-STA (single-span): same model, same
eager-attention monkeypatch, same frame-grid extraction. The benchmark differences (prompt,
single-vs-multi span parsing, metric) live in the dataset adapters; the per-benchmark video
sampling (fps, max-pixels) comes from the config's generate.extra.

`parse` extracts [start, end] spans from the response (single for Charades, list for QVH).
`region_mask` maps a temporal span to frame-token indices via `mtla.mask.span_to_token_indices`.

This adapter holds the small, pure, CPU-testable pieces. The heavy GPU generate+extract driver
lives with the dataset adapter (QVHighlights and Charades use different fused stage scripts
under `mtla/stages/`), which `run.py` invokes.
"""
from __future__ import annotations

import re

from .base import ModelAdapter, Prediction
from ..mask import span_to_token_indices

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

_TIME = r'(\d{1,3}(?::\d{2})?(?:\.\d+)?)'


def _to_seconds(s: str) -> float:
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)


# The canonical, validated multi-window parser lives in the video stage script
# (mtla/stages/qwen3vl_video.py). We reuse it verbatim here so the generate stage and any
# offline parsing share one source of truth and cannot drift.
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


def parse_spans(response: str, multi: bool = True) -> list:
    """Parse temporal spans from a Qwen3-VL response.

    multi=True (QVHighlights): all [start,end] windows; multi=False (Charades): the first.
    Mirrors the validated parser in mtla/stages/qwen3vl_video.py: try each pattern family in
    order, keep the first that matches, dedup, order low->high.
    """
    t = response.lower().replace("seconds", "s")
    seen = set(); spans = []
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
            seen.add(key); spans.append(Prediction([a, b], ""))
        if spans:
            break
    return spans if multi else spans[:1]


class Qwen3VLAdapter(ModelAdapter):
    model_id = MODEL_ID
    attn_module_path = "transformers.models.qwen3_vl.modeling_qwen3_vl"

    def parse(self, response: str, multi: bool = True, **kw) -> list:
        return parse_spans(response, multi=multi)

    def region_mask(self, region, meta: dict):
        """region = [t_start, t_end] seconds; meta has duration_s + n_tokens (frames)."""
        return span_to_token_indices(region, meta["duration_s"], meta["n_tokens"])
