"""InternVL3.5-8B model adapter for image object detection.

Uses the native HF checkpoint ``OpenGVLab/InternVL3_5-8B-HF``
(``InternVLForConditionalGeneration`` with a standard ``AutoProcessor``, no
``trust_remote_code``). The LLM backbone is Qwen3, so attention capture hooks
``transformers.models.qwen3.modeling_qwen3``.

InternVL emits native grounding: ``<ref>label</ref><box>[[x1,y1,x2,y2], ...]</box>``
in ``[0,1000]`` coordinates. Images are encoded with dynamic tiling (an
``n_cols x n_rows`` grid of 448px tiles plus a thumbnail), so the region mask
(bbox to image-token indices) depends on that tile grid. The processor does not return
the grid, so it is recomputed with the processor's own ``get_optimal_tiled_canvas``
helper, reusing HF's tiling decision rather than reimplementing it.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers.models.got_ocr2.image_processing_got_ocr2 import (
    get_optimal_tiled_canvas,
)

from mtla.models.base import ModelAdapter, Prediction
from mtla.registry import register_model
from mtla.utils import iou
from mtla.types import BuildInputs, Ctx, GenRecord, TokenRange

MODEL_ID = "OpenGVLab/InternVL3_5-8B-HF"
IMAGE_TOKEN = "<IMG_CONTEXT>"
TILE_SIZE = 448
PATCH_GRID = 16  # per-tile patch grid side
TOKENS_PER_TILE = (
    PATCH_GRID * PATCH_GRID
)  # 16x16 patches per tile after pixel-shuffle 0.5
MIN_TILES, MAX_TILES = 1, 12

# Module-level compiled regexes for fully static patterns used in parse_response.
_RE_REF_BOX = re.compile(
    r"<ref>([^<]+)</ref><box>\s*\[(.+?)\]\s*</box>", flags=re.DOTALL
)
_RE_BOX4 = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*|\s*```")
_RE_LABEL_BBOX = re.compile(r"([A-Za-z][A-Za-z _]*?)\s*(\[\[)")


@register_model("internvl_image")
class InternVLImageAdapter(ModelAdapter):
    """Adapter for InternVL3.5-8B image object detection (key ``internvl_image``).

    Parses native ``<ref>/<box>`` grounding output, seeds vLLM per request, and maps
    boxes to image tokens through InternVL's dynamic tile grid.
    """

    model_id = MODEL_ID
    # InternVL-HF's LLM backbone is Qwen3, so the attention capture hooks that module.
    attn_module_path = "transformers.models.qwen3.modeling_qwen3"
    modality_pad_token = IMAGE_TOKEN
    overlap = staticmethod(iou)  # boxes use spatial IoU for the hallucination test
    # InternVL's native grounding syntax; this suffix elicits it (parse_response reads it). Matches
    # the paper's COCO prompt for InternVL.
    detection_prompt_suffix = (
        " and output the bounding boxes in the format "
        "<ref>category</ref><box>[[x1, y1, x2, y2], ...]</box> "
        "with coordinates normalized to [0, 1000]."
    )

    def parse_response(self, response: str) -> list["Prediction"]:
        """Parse InternVL native grounding output into detection predictions.

        Handles both the ``<ref>label</ref><box>[[...]]</box>`` form and the bare
        ``label[[...]]`` fallback form (with brace-depth matching for nested box lists).

        Args:
            response: The raw generated text.

        Returns:
            A list of ``Prediction([x1,y1,x2,y2], label)`` in ``[0,1000]`` coordinates
            with lowercased labels, in response order (empty if none match).
        """
        preds: list[Prediction] = []
        for m in _RE_REF_BOX.finditer(response):
            label = m.group(1).strip().lower()
            for b in _RE_BOX4.finditer(m.group(2)):
                preds.append(Prediction([int(b.group(i)) for i in range(1, 5)], label))
        if preds:
            return preds
        cleaned = _RE_JSON_FENCE.sub("", response)
        pos = 0
        while pos < len(cleaned):
            pm = _RE_LABEL_BBOX.search(cleaned, pos)
            if not pm:
                break
            label = pm.group(1).strip().lower()
            depth = 0
            outer_close = -1
            for i in range(pm.start(2), len(cleaned)):
                if cleaned[i] == "[":
                    depth += 1
                elif cleaned[i] == "]":
                    depth -= 1
                    if depth == 0:
                        outer_close = i
                        break
            if outer_close == -1:
                break
            chunk = cleaned[pm.start(2) : outer_close + 1]
            for b in _RE_BOX4.finditer(chunk):
                preds.append(Prediction([int(b.group(i)) for i in range(1, 5)], label))
            pos = outer_close + 1
        return preds

    def build_vllm_request(
        self, proc: Any, item: dict, dataset: Any, cfg: Any
    ) -> dict | None:
        """Build a vLLM request for one image-detection item.

        Args:
            proc: The processor from ``gen_processor``.
            item: A raw dataset item (from ``load_items``); ``item["image"]`` is the
                image path.
            dataset: The dataset object; supplies the prompt via ``dataset.prompt``.
            cfg: Run config (unused for images).

        Returns:
            A dict ``{prompt, multi_modal_data}`` with the chat-templated prompt and the
            RGB image.
        """
        # `item` is a raw dataset item (from load_items), not a generation record.
        image = Image.open(item["image"]).convert("RGB")
        msgs = self._image_message(image, self.build_text_prompt(dataset, item))
        prompt = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "multi_modal_data": {"image": image}}

    def _load_model(self, device: str) -> Any:
        """Load InternVL3.5-8B for extraction.

        Args:
            device: CUDA device string to place the model on.

        Returns:
            The bf16 ``AutoModelForImageTextToText`` in eval mode with eager attention.
        """
        return (
            AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=torch.bfloat16, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )  # type: ignore[arg-type]

    def _encode_attn_extraction_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> tuple["torch.Tensor", dict, dict, int] | None:
        """Encode the image + prompt and compute the dynamic-tiling geometry.

        Opens the image and chat-templates the prompt in one processor call, then derives
        the expected image-token count from the tile grid.

        Args:
            record: The generation record (prompt, ``extra["image"]``).
            ctx: The extraction context (``proc`` / ``device``).
            rank: Worker rank, used only for the missing-image skip log.

        Returns:
            ``(prompt_ids, {"pixel_values": ...}, {"tile": (...)} , n_expected)``, or
            ``None`` to skip a missing image file (tolerated: the driver caps the total
            skip rate).

        Raises:
            OSError: If the image file exists but cannot be decoded (a corrupt file is a
                real error, not missing data, so it propagates).
        """
        proc, device = ctx["proc"], ctx["device"]
        image_path = record["extra"]["image"]
        if not os.path.exists(image_path):
            print(
                f"[worker {rank}] skip {record['id']}: image not found {image_path}",
                flush=True,
            )
            return None
        img = Image.open(image_path).convert("RGB")
        msgs = self._image_message(img, record["prompt"])
        inputs = proc.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        prompt_ids = inputs["input_ids"][0]
        n_cols, n_rows, has_thumb = self._compute_tile_grid(*img.size)
        n_expected = (n_cols * n_rows + (1 if has_thumb else 0)) * TOKENS_PER_TILE
        return (
            prompt_ids,
            {"pixel_values": inputs["pixel_values"]},
            {"tile": (n_cols, n_rows, has_thumb)},
            n_expected,
        )

    def locate_proposal_tokens(
        self, response: str, predictions: list["Prediction"], tokenizer: Any
    ) -> list[TokenRange | None]:
        """Locate each predicted box's response tokens Q_p via tokenizer char offsets.

        For each prediction, finds the coordinate-digit tokens of its ``[x1,y1,x2,y2]``
        match and the tokens of the nearest preceding label mention.

        Args:
            response: The raw generated text.
            predictions: The parsed boxes to locate, in order.
            tokenizer: The tokenizer providing an offset mapping over ``response``.

        Returns:
            One entry per prediction, index-aligned with ``predictions``: a ``TokenRange``
            dict ``{first_label_tok, label_toks, coord_toks}``, or ``None`` if the box or
            its label could not be located.
        """
        enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        out: list[TokenRange | None] = []
        for pred in predictions:
            label, box = pred.label, pred.region
            m = re.compile(
                rf"\[\s*{box[0]}\s*,\s*{box[1]}\s*,\s*{box[2]}\s*,\s*{box[3]}\s*\]"
            ).search(response)
            if not m:
                out.append(None)
                continue
            coord_toks = []
            for num_m in re.finditer(r"\d+", m.group(0)):
                ns, ne = m.start() + num_m.start(), m.start() + num_m.end()
                coord_toks += [
                    ti for ti, (ts, te) in enumerate(offsets) if ts < ne and te > ns
                ]
            label_block = re.compile(
                rf"(?:^|<ref>|[\s\]\)\.])({re.escape(label)})(?:</ref>\s*<box>)?\s*\[\[",
                flags=re.IGNORECASE,
            )
            label_pos = None
            for lm in label_block.finditer(response):
                if lm.start() <= m.start():
                    label_pos = lm
                else:
                    break
            if label_pos is None:
                for lm in re.compile(
                    rf"\b{re.escape(label)}\b", flags=re.IGNORECASE
                ).finditer(response):
                    if lm.start() <= m.start():
                        label_pos = lm
                    else:
                        break
            if label_pos is None:
                out.append(None)
                continue
            lmatch = re.search(
                re.escape(label), label_pos.group(0), flags=re.IGNORECASE
            )
            if lmatch is None:
                out.append(None)
                continue
            ls, le = (
                label_pos.start() + lmatch.start(),
                label_pos.start() + lmatch.end(),
            )
            label_toks = [
                ti for ti, (ts, te) in enumerate(offsets) if ts < le and te > ls
            ]
            if not label_toks:
                out.append(None)
                continue
            out.append(
                {
                    "first_label_tok": label_toks[0],
                    "label_toks": label_toks,
                    "coord_toks": coord_toks,
                }
            )
        return out

    def proposal_region_attn_mask(
        self, prediction: "Prediction", meta: dict
    ) -> list[int]:
        """Map a box to image-token indices M(R_p) under InternVL dynamic tiling.

        The token sequence is
        ``tile_0[0..255], ..., tile_{N-1}[0..255], [thumbnail[0..255]]``, each tile a
        row-major ``PATCH_GRID x PATCH_GRID`` grid (256 tokens per tile after pixel-shuffle
        0.5). A tile that does not overlap the box contributes no tokens.

        Args:
            prediction: The prediction whose box ``[x1,y1,x2,y2]`` (in ``[0,1000]``)
                defines the region.
            meta: Must contain ``meta["tile"] = (n_cols, n_rows, has_thumb)``: the tile-grid
                dimensions and whether a trailing thumbnail tile is present.

        Returns:
            The image-token indices (into the flattened tile sequence) that overlap the
            box; empty if the box is degenerate (``x2 <= x1`` or ``y2 <= y1``).
        """
        x1, y1, x2, y2 = prediction.region
        n_cols, n_rows, has_thumb = meta["tile"]
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
            lx0 = max(0.0, (bx1 - tx0) / (tx1 - tx0))
            lx1 = min(1.0, (bx2 - tx0) / (tx1 - tx0))
            ly0 = max(0.0, (by1 - ty0) / (ty1 - ty0))
            ly1 = min(1.0, (by2 - ty0) / (ty1 - ty0))
            col_min, col_max = clamp(int(np.floor(lx0 * PATCH_GRID))), clamp(
                int(np.floor((lx1 - 1e-6) * PATCH_GRID))
            )
            row_min, row_max = clamp(int(np.floor(ly0 * PATCH_GRID))), clamp(
                int(np.floor((ly1 - 1e-6) * PATCH_GRID))
            )
            off = tile_idx * per_tile
            for pr in range(row_min, row_max + 1):
                for pc in range(col_min, col_max + 1):
                    inside.append(off + pr * PATCH_GRID + pc)
        if has_thumb:
            col_min, col_max = clamp(int(np.floor(bx1 * PATCH_GRID))), clamp(
                int(np.floor((bx2 - 1e-6) * PATCH_GRID))
            )
            row_min, row_max = clamp(int(np.floor(by1 * PATCH_GRID))), clamp(
                int(np.floor((by2 - 1e-6) * PATCH_GRID))
            )
            off = n_tiles * per_tile
            for pr in range(row_min, row_max + 1):
                for pc in range(col_min, col_max + 1):
                    inside.append(off + pr * PATCH_GRID + pc)
        return [i for i in inside if i < total]

    def hf_extraction_kwargs(
        self, full_ids: torch.Tensor, total_len: int, device: str, inp: BuildInputs
    ) -> dict:
        """Build kwargs for the captured InternVL forward.

        No ``attention_mask`` is passed: for a single unpadded sequence HF builds the
        all-ones causal mask itself.

        Args:
            full_ids: The full input-id tensor (prompt + response).
            total_len: Total sequence length (unused; kept for the callback signature).
            device: CUDA device string (unused; kept for the callback signature).
            inp: The ``BuildInputs`` dict; supplies ``pixel_values``.

        Returns:
            ``{input_ids, pixel_values}`` for ``model(**fk)``.
        """
        return {
            "input_ids": full_ids,
            "pixel_values": inp["pixel_values"],
        }

    def _compute_tile_grid(self, width: int, height: int) -> tuple[int, int, bool]:
        """Compute the InternVL tile grid for an image via HF's own tiling decision.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            ``(n_cols, n_rows, has_thumb)``: the tile-grid dimensions and whether a
            thumbnail tile is appended (true whenever the grid has more than one tile).
        """
        n_cols, n_rows = get_optimal_tiled_canvas(
            (height, width), (TILE_SIZE, TILE_SIZE), MIN_TILES, MAX_TILES
        )
        return n_cols, n_rows, (n_cols * n_rows) != 1
