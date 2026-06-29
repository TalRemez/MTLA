"""InternVL3.5-8B model adapter (image detection).

InternVL emits native grounding: ``<ref>label</ref><box>[[x1,y1,x2,y2], ...]</box>`` with
coordinates in [0,1000]. Images are encoded with dynamic tiling (variable tiles + optional
thumbnail), so the region mask uses `mtla.mask.bbox_to_internvl_token_indices`.

This adapter holds the small, pure, CPU-testable pieces: the output `parse` and the
`region_mask`, plus the `attn_module_path` to monkeypatch during extraction. The heavy GPU
generate/extract drivers live with the dataset adapter (they are model x dataset specific),
which calls the validated stage scripts under `mtla/stages/`.
"""
from __future__ import annotations

import re

from .base import ModelAdapter, Prediction, SlotSpec
from ..mask import bbox_to_internvl_token_indices

MODEL_ID = "OpenGVLab/InternVL3_5-8B"


def parse_internvl(response: str) -> list:
    """Parse InternVL native grounding output into [Prediction(region=[x1,y1,x2,y2], label)].

    Handles both `<ref>label</ref><box>[[...]]</box>` and the bare `label[[...]]` form.
    """
    preds = []
    for m in re.finditer(r'<ref>([^<]+)</ref><box>\s*\[(.+?)\]\s*</box>', response, flags=re.DOTALL):
        label = m.group(1).strip().lower()
        for b in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', m.group(2)):
            preds.append(Prediction([int(b.group(i)) for i in range(1, 5)], label))
    if preds:
        return preds
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', response)
    pat = re.compile(r'([A-Za-z][A-Za-z _]*?)\s*(\[\[)')
    pos = 0
    while pos < len(cleaned):
        m = pat.search(cleaned, pos)
        if not m:
            break
        label = m.group(1).strip().lower()
        outer_open = m.start(2)
        depth = 0; outer_close = -1
        for i in range(outer_open, len(cleaned)):
            if cleaned[i] == '[':
                depth += 1
            elif cleaned[i] == ']':
                depth -= 1
                if depth == 0:
                    outer_close = i; break
        if outer_close == -1:
            break
        chunk = cleaned[outer_open:outer_close + 1]
        for b in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', chunk):
            preds.append(Prediction([int(b.group(i)) for i in range(1, 5)], label))
        pos = outer_close + 1
    return preds


class InternVLAdapter(ModelAdapter):
    model_id = MODEL_ID
    # InternVL's LLM backbone is Qwen3; that is the module we monkeypatch for attention.
    attn_module_path = "transformers.models.qwen3.modeling_qwen3"
    tasks = ("image_det",)

    def parse(self, response: str, task: str = None, **kw) -> list:
        return parse_internvl(response)

    def region_mask(self, region, meta: dict):
        """region = bbox [x1,y1,x2,y2] in [0,1000]; meta has tile_grid + has_thumb."""
        return bbox_to_internvl_token_indices(region, meta["tile_grid"], meta["has_thumb"])

    # ---- signal slots in the saved record (image_det) ----
    # MTLA = inside-region attention; "all" = count-weighted mean over label+coord tokens
    # (the canonical recipe). SVAR = global attention read at the first coord digit (x1):
    # InternVL emits `label[[box1],[box2],...]` so the label token is shared across a category's
    # boxes; the fair per-box SVAR token is x1 (attn_first_digit).
    def mtla_slot(self, task: str, slot: str = "all") -> SlotSpec:
        if slot in ("all", "attn_all"):
            return SlotSpec(stat="image_inside_sum", combine="all",
                            parts=[("attn_label_mean", "n_label_toks"),
                                   ("attn_coord_mean", "n_coord_toks")])
        block = {"coord": "attn_coord_mean", "label": "attn_label_mean",
                 "first": "attn", "first_digit": "attn_first_digit"}.get(slot, slot)
        return SlotSpec(stat="image_inside_sum", block=block)

    def svar_slot(self, task: str) -> SlotSpec:
        return SlotSpec(stat="image_sum", block="attn_first_digit")

    # ---- stage scripts ----
    def generate_script(self, task: str, engine: str) -> str:
        return {"vllm": "internvl_generate.py", "hf": "internvl_generate_hf.py"}[engine]

    def extract_script(self, task: str) -> str:
        return "internvl_extract.py"
