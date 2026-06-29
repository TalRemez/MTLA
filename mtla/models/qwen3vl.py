"""Qwen3-VL-8B model adapter.

Supports two task families:
  - "video_span": temporal grounding (QVHighlights multi-segment, Charades single-span). `parse`
    extracts [start,end] spans; `region_mask` -> frame tokens via span_to_token_indices.
  - "image_det": COCO detection. `parse` reads JSON {"bbox_2d","label"}; `region_mask` -> image
    patches via bbox_to_patch_indices (fixed grid). Reproduces the paper's COCO AUROC 0.902.

This adapter holds the small, pure pieces (parse, region_mask, signal slots, attn module path,
stage-script names). The heavy GPU work lives in mtla/stages/; video sampling (fps, pixels) is
fixed in the video stage scripts. The pipeline is decoupled: generate then a separate HF-eager
extract.
"""
from __future__ import annotations

import json
import re

from .base import ModelAdapter, Prediction, SlotSpec
from ..mask import span_to_token_indices, bbox_to_patch_indices

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


def parse_bboxes(response: str) -> list:
    """Parse Qwen3-VL detection JSON `[{"bbox_2d":[x1,y1,x2,y2],"label":...}, ...]` into
    [Prediction(region=[x1,y1,x2,y2], label)]. JSON first, then a regex fallback (label is the
    first one AFTER each box, since Qwen emits the label to the right of the box)."""
    cleaned = re.sub(r'```json\s*|```\s*', '', response).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            out = []
            for o in parsed:
                if isinstance(o, dict) and isinstance(o.get("bbox_2d"), list) and len(o["bbox_2d"]) == 4:
                    out.append(Prediction([int(x) for x in o["bbox_2d"]], o.get("label", "").lower()))
            if out:
                return out
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    box_re = re.compile(r'"bbox_2d"\s*:\s*\[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]')
    label_re = re.compile(r'"label"\s*:\s*"([^"]+)"')
    matches = list(box_re.finditer(response))
    out = []
    for k, m in enumerate(matches):
        hi = matches[k + 1].start() if k + 1 < len(matches) else len(response)
        lm = label_re.search(response[m.end():hi])
        out.append(Prediction([int(m.group(i)) for i in range(1, 5)],
                              lm.group(1).lower() if lm else ""))
    return out


class Qwen3VLAdapter(ModelAdapter):
    model_id = MODEL_ID
    attn_module_path = "transformers.models.qwen3_vl.modeling_qwen3_vl"
    tasks = ("video_span", "image_det")

    def parse(self, response: str, task: str = "video_span", multi: bool = True, **kw) -> list:
        if task == "image_det":
            return parse_bboxes(response)
        return parse_spans(response, multi=multi)

    def region_mask(self, region, meta: dict):
        """video_span: [t_start,t_end]+duration_s/n_tokens -> frames.
        image_det: bbox [x1,y1,x2,y2]+grid_h/grid_w -> patches (fixed grid)."""
        if "grid_h" in meta:
            return bbox_to_patch_indices(region, meta["grid_h"], meta["grid_w"])
        return span_to_token_indices(region, meta["duration_s"], meta["n_tokens"])

    # ---- image_det signal slots (COCO with Qwen3-VL; paper AUROC 0.902) ----
    # Qwen emits {bbox,label} per box, so the first response token (block "attn") is a fair
    # per-box SVAR token (unlike InternVL where the label is shared and we use first_digit).
    def mtla_slot(self, task: str, slot: str = "all") -> SlotSpec:
        if slot in ("all", "attn_all"):
            return SlotSpec(stat="image_inside_sum", combine="all",
                            parts=[("attn_label_mean", "n_label_toks"),
                                   ("attn_coord_mean", "n_coord_toks")])
        block = {"coord": "attn_coord_mean", "label": "attn_label_mean", "first": "attn"}.get(slot, slot)
        return SlotSpec(stat="image_inside_sum", block=block)

    def svar_slot(self, task: str) -> SlotSpec:
        return SlotSpec(stat="image_sum", block="attn")  # first token, global

    def generate_script(self, task: str, engine: str) -> str:
        if task == "image_det":
            if engine != "vllm":
                raise NotImplementedError("Qwen3-VL COCO generation is vLLM-only (engine: vllm)")
            return "qwen3vl_det_generate.py"
        raise NotImplementedError("video_span generation is dataset-driven (qwen3vl_video/charades)")

    def extract_script(self, task: str) -> str:
        if task == "image_det":
            return "qwen3vl_det_extract.py"
        raise NotImplementedError("video_span extraction is dataset-driven")
