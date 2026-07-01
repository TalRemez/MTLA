"""InternVL3.5-8B model adapter (image detection).

Uses the native HF checkpoint ``OpenGVLab/InternVL3_5-8B-HF`` — ``InternVLForConditionalGeneration``
with a standard ``AutoProcessor`` (no ``trust_remote_code``). The LLM backbone is Qwen3, so the
attention capture hooks ``transformers.models.qwen3.modeling_qwen3``.

InternVL emits native grounding — ``<ref>label</ref><box>[[x1,y1,x2,y2], ...]</box>`` in [0,1000].
Images are encoded with dynamic tiling (a ``n_cols x n_rows`` grid of 448px tiles + a thumbnail),
so the region mask (bbox → image-token indices) depends on that tile grid. The processor doesn't
return the grid, so we recompute it with the processor's own ``get_optimal_tiled_canvas`` helper —
reusing HF's tiling decision rather than reimplementing it.
"""
from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

from .base import ModelAdapter, Prediction, hallucinated
from ..registry import register_model
from ..mtla_attn import CaptureState, install_capture
from ..utils import iou
from ..types import BuildInputs, Ctx, GenRecord, TokenRange

MODEL_ID = "OpenGVLab/InternVL3_5-8B-HF"
IMAGE_TOKEN = "<IMG_CONTEXT>"
TILE_SIZE = 448
TOKENS_PER_TILE = 256          # 16x16 patches per tile after pixel-shuffle 0.5
PATCH_GRID = 16                # per-tile patch grid side
MIN_TILES, MAX_TILES = 1, 12


def parse_internvl(response: str) -> list:
    """Parse InternVL native grounding output into ``[Prediction([x1,y1,x2,y2], label)]``.

    Handles both ``<ref>label</ref><box>[[...]]</box>`` and the bare ``label[[...]]`` form.
    """
    preds: list[Prediction] = []
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
        pm = pat.search(cleaned, pos)
        if not pm:
            break
        label = pm.group(1).strip().lower()
        depth = 0; outer_close = -1
        for i in range(pm.start(2), len(cleaned)):
            if cleaned[i] == '[':
                depth += 1
            elif cleaned[i] == ']':
                depth -= 1
                if depth == 0:
                    outer_close = i; break
        if outer_close == -1:
            break
        chunk = cleaned[pm.start(2):outer_close + 1]
        for b in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', chunk):
            preds.append(Prediction([int(b.group(i)) for i in range(1, 5)], label))
        pos = outer_close + 1
    return preds


def tile_grid_for(width: int, height: int) -> tuple[int, int, bool]:
    """The InternVL tile grid ``(n_cols, n_rows)`` for an image, via HF's own tiling decision.
    Returns ``(n_cols, n_rows, has_thumb)``; a thumbnail tile is appended whenever >1 tile."""
    n_cols, n_rows = get_optimal_tiled_canvas((height, width), (TILE_SIZE, TILE_SIZE),
                                              MIN_TILES, MAX_TILES)
    return n_cols, n_rows, (n_cols * n_rows) != 1


def bbox_region_mask(bbox: list[float], n_cols: int, n_rows: int, has_thumb: bool) -> list[int]:
    """Image-token indices overlapping ``bbox`` M(R_p) under InternVL's dynamic tiling.

    The token sequence is ``tile_0[0..255], ..., tile_{N-1}[0..255], [thumbnail[0..255]]``, each
    tile a row-major ``PATCH_GRID x PATCH_GRID`` grid. ``bbox`` is ``[x1,y1,x2,y2]`` in [0,1000].
    """
    x1, y1, x2, y2 = bbox
    n_tiles = n_cols * n_rows
    per_tile = PATCH_GRID * PATCH_GRID
    total = n_tiles * per_tile + (per_tile if has_thumb else 0)
    if x2 <= x1 or y2 <= y1:
        return []

    def clamp(v):
        return max(0, min(PATCH_GRID - 1, v))

    bx1, by1, bx2, by2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
    inside = []
    for tile_idx in range(n_tiles):
        col, row = tile_idx % n_cols, tile_idx // n_cols
        tx0, tx1 = col / n_cols, (col + 1) / n_cols
        ty0, ty1 = row / n_rows, (row + 1) / n_rows
        if tx1 <= bx1 or tx0 >= bx2 or ty1 <= by1 or ty0 >= by2:
            continue  # tile does not overlap bbox
        lx0 = max(0.0, (bx1 - tx0) / (tx1 - tx0)); lx1 = min(1.0, (bx2 - tx0) / (tx1 - tx0))
        ly0 = max(0.0, (by1 - ty0) / (ty1 - ty0)); ly1 = min(1.0, (by2 - ty0) / (ty1 - ty0))
        col_min, col_max = clamp(int(np.floor(lx0 * PATCH_GRID))), clamp(int(np.floor((lx1 - 1e-6) * PATCH_GRID)))
        row_min, row_max = clamp(int(np.floor(ly0 * PATCH_GRID))), clamp(int(np.floor((ly1 - 1e-6) * PATCH_GRID)))
        off = tile_idx * per_tile
        for pr in range(row_min, row_max + 1):
            for pc in range(col_min, col_max + 1):
                inside.append(off + pr * PATCH_GRID + pc)
    if has_thumb:
        col_min, col_max = clamp(int(np.floor(bx1 * PATCH_GRID))), clamp(int(np.floor((bx2 - 1e-6) * PATCH_GRID)))
        row_min, row_max = clamp(int(np.floor(by1 * PATCH_GRID))), clamp(int(np.floor((by2 - 1e-6) * PATCH_GRID)))
        off = n_tiles * per_tile
        for pr in range(row_min, row_max + 1):
            for pc in range(col_min, col_max + 1):
                inside.append(off + pr * PATCH_GRID + pc)
    return [i for i in inside if i < total]


def find_pred_token_ranges(response: str, predictions: list["Prediction"],
                           tokenizer: Any) -> list[TokenRange | None]:
    """Per predicted box, the response tokens Q_p (its label + coordinate tokens), via offsets."""
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out: list[TokenRange | None] = []
    for pred in predictions:
        label, box = pred.label, pred.region
        m = re.compile(rf'\[\s*{box[0]}\s*,\s*{box[1]}\s*,\s*{box[2]}\s*,\s*{box[3]}\s*\]').search(response)
        if not m:
            out.append(None); continue
        coord_toks = []
        for num_m in re.finditer(r'\d+', m.group(0)):
            ns, ne = m.start() + num_m.start(), m.start() + num_m.end()
            coord_toks += [ti for ti, (ts, te) in enumerate(offsets) if ts < ne and te > ns]
        label_block = re.compile(
            rf'(?:^|<ref>|[\s\]\)\.])({re.escape(label)})(?:</ref>\s*<box>)?\s*\[\[', flags=re.IGNORECASE)
        label_pos = None
        for lm in label_block.finditer(response):
            if lm.start() <= m.start():
                label_pos = lm
            else:
                break
        if label_pos is None:
            for lm in re.compile(rf'\b{re.escape(label)}\b', flags=re.IGNORECASE).finditer(response):
                if lm.start() <= m.start():
                    label_pos = lm
                else:
                    break
        if label_pos is None:
            out.append(None); continue
        lmatch = re.search(re.escape(label), label_pos.group(0), flags=re.IGNORECASE)
        if lmatch is None:
            out.append(None); continue
        ls, le = label_pos.start() + lmatch.start(), label_pos.start() + lmatch.end()
        label_toks = [ti for ti, (ts, te) in enumerate(offsets) if ts < le and te > ls]
        if not label_toks:
            out.append(None); continue
        out.append({"first_label_tok": label_toks[0], "label_toks": label_toks, "coord_toks": coord_toks})
    return out


@register_model("internvl")
class InternVLAdapter(ModelAdapter):
    model_id = MODEL_ID
    attn_module_path = "transformers.models.qwen3.modeling_qwen3"   # InternVL-HF LLM backbone
    tasks = ("image_det",)

    def parse(self, response: str, task: str | None = None) -> list["Prediction"]:
        self._check_task(task)
        return parse_internvl(response)

    # ---- vLLM generation ----
    def vllm_engine_args(self, dataset: Any) -> dict:
        return {"limit_mm_per_prompt": {"image": 1}}

    def vllm_uses_seed(self, task: str) -> bool:
        return True

    def build_request(self, proc: Any, item: dict, dataset: Any, cfg: Any) -> dict | None:
        # `item` is a raw dataset item (from load_items), not a generation record.
        image = Image.open(item["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": dataset.prompt(item)}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": image}}

    # ---- MTLA extraction ----
    def load_for_extract(self, gpu_id: int, task: str = "image_det") -> Ctx:
        device = f"cuda:{gpu_id}"
        state = CaptureState()
        install_capture(self.attn_module_path, state)
        proc = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()  # type: ignore[arg-type]
        layers = model.model.language_model.layers
        state.layer_ids = {id(L.self_attn): i for i, L in enumerate(layers)}
        return cast(Ctx, {"model": model, "proc": proc, "tokenizer": proc.tokenizer, "state": state,
                "device": device, "task": task, "n_layers": len(layers),
                "n_heads": model.config.text_config.num_attention_heads,
                "image_pad_id": proc.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)})

    def build_inputs(self, record: GenRecord, ctx: Ctx, rank: int) -> BuildInputs | None:
        proc = ctx["proc"]; device = ctx["device"]; image_pad_id = ctx["image_pad_id"]
        response = record.get("response")
        preds = self.parse(response, "image_det") if response else []
        if not preds:
            return None
        try:
            img = Image.open(record["extra"]["image"]).convert("RGB")
        except Exception as e:
            print(f"[worker {rank}] skip {record['id']}: image {e}", flush=True)
            return None
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": record["prompt"]}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0]
        image_idx = [k for k, t in enumerate(prompt_ids.tolist()) if t == image_pad_id]
        n_cols, n_rows, has_thumb = tile_grid_for(*img.size)
        n_expected = (n_cols * n_rows + (1 if has_thumb else 0)) * TOKENS_PER_TILE
        if len(image_idx) != n_expected:
            print(f"[worker {rank}] skip {record['id']}: image tokens {len(image_idx)} != {n_expected}",
                  flush=True)
            return None
        gt = record.get("gt", [])
        hallu = [hallucinated(p.region, p.label, gt, iou) for p in preds]
        return cast(BuildInputs, {
                "prompt_ids": prompt_ids, "response": response, "modality_idx_l": image_idx,
                "predictions": preds, "hallu_flags": hallu, "pixel_values": inputs["pixel_values"],
                "meta": {"tile": (n_cols, n_rows, has_thumb)}})

    def query_tokens(self, response: str, predictions: list["Prediction"],
                     tokenizer: Any) -> list[TokenRange | None]:
        return find_pred_token_ranges(response, predictions, tokenizer)

    def region_mask(self, prediction: "Prediction", meta: dict) -> list[int]:
        return bbox_region_mask(prediction.region, *meta["tile"])

    def forward_kwargs(self, full_ids: torch.Tensor, total_len: int, device: str,
                       inp: BuildInputs) -> dict:
        return {"input_ids": full_ids, "pixel_values": inp["pixel_values"],
                "attention_mask": torch.ones(1, total_len, device=device, dtype=torch.long)}
