"""Qwen3-VL-8B model adapters for image detection and video temporal grounding.

A shared ``Qwen3VLBase`` holds the family wiring (checkpoint, attention module to hook,
model loading, and the common capture-forward kwargs); two registered single-task
adapters build on it:

  * ``Qwen3VLImageAdapter`` (key ``qwen3vl_image``): COCO detection, the headline COCO
    detector. ``parse_response`` reads JSON ``{"bbox_2d","label"}``; the region masks onto
    the merged patch grid. Prompted with all 80 COCO classes (open-vocab): hallucination
    AUROC 0.890, and mAP 36.9 with N=16 self-consistency voting (``--agg sum``).
  * ``Qwen3VLVideoAdapter`` (key ``qwen3vl_video``): temporal grounding (QVHighlights
    multi-segment, Charades single-span). ``parse_response`` extracts ``[start, end]``
    spans; the region masks onto the frame tokens inside a span.

Video preprocessing (fps / pixel budget) comes from ``cfg.preprocess`` and feeds both
the generate and extract stages, so the two stages see the same frames. The LLM
backbone attention is captured at ``transformers.models.qwen3_vl.modeling_qwen3_vl``.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from typing import Any

from mtla.models.base import ModelAdapter, Prediction
from mtla.registry import register_model
from mtla.utils import (
    iou,
    tiou,
    tokens_overlapping_char_span,
    video_duration,
    parse_timestamp,
)
from mtla.types import BuildInputs, Ctx, GenRecord, TokenRange

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# A single timestamp: seconds (``123``, ``12.5``) or ``MM:SS`` (``1:30``, ``1:30.5``).
_TIMESTAMP = r"(\d{1,3}(?::\d{2})?(?:\.\d+)?)"
# The timestamp-span formats the model may emit for a ``[start, end]`` window; every
# variant captures exactly two ``_TIMESTAMP`` groups.
_SPAN_FORMATS = [
    rf"\[\s*{_TIMESTAMP}\s*,\s*{_TIMESTAMP}\s*\]",
    rf"\(\s*{_TIMESTAMP}\s*,\s*{_TIMESTAMP}\s*\)",
    rf"from\s+{_TIMESTAMP}\s*s?\s+to\s+{_TIMESTAMP}",
    rf"between\s+{_TIMESTAMP}\s*s?\s+and\s+{_TIMESTAMP}",
    rf"{_TIMESTAMP}\s*s?\s*-\s*{_TIMESTAMP}\s*s",
    rf"{_TIMESTAMP}\s*s?\s*-\s*{_TIMESTAMP}",
    rf"{_TIMESTAMP}\s*s\s+to\s+{_TIMESTAMP}",
    rf"start[:\s]+{_TIMESTAMP}\s*s?\s*,?\s*end[:\s]+{_TIMESTAMP}",
]

# Fully-static (non-interpolated) compiled patterns.
_RE_JSON_FENCE = re.compile(r"```json\s*|```\s*")
_RE_BBOX_2D = re.compile(
    r'"bbox_2d"\s*:\s*\[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]'
)
_RE_LABEL = re.compile(r'"label"\s*:\s*"([^"]+)"')
_RE_LABEL_MARKER = re.compile(r'"label"\s*:\s*"')
# Template (per-prediction: filled with the escaped label, then compiled) that matches a full
# ``"bbox_2d":[x1,y1,x2,y2],"label":"<label>"`` object, capturing the four coordinate groups.
_BBOX_LABEL_TEMPLATE = (
    r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,'
    r'\s*"label"\s*:\s*"{label}"'
)


class Qwen3VLBase(ModelAdapter):
    """Shared Qwen3-VL-8B wiring for the task-specific image and video adapters.

    Holds only what is identical across tasks: the checkpoint id, the attention module
    to hook, model loading, and the capture forward kwargs. It is intentionally *not*
    registered; the concrete single-task subclasses below carry the ``@register_model``
    decorator and the per-task callbacks.
    """

    model_id = MODEL_ID
    attn_module_path = "transformers.models.qwen3_vl.modeling_qwen3_vl"

    def _load_model(self, device: str) -> Any:
        """Load Qwen3-VL-8B for extraction.

        Args:
            device: CUDA device string; passed as ``device_map`` to place the model.

        Returns:
            The bf16 ``Qwen3VLForConditionalGeneration`` in eval mode with eager attention.
        """
        return Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=device,
        ).eval()

    # Modality tensor keys to copy from the processor output into the capture forward;
    # each subclass sets its own (image vs video).
    mm_keys: tuple = ()

    def hf_extraction_kwargs(
        self, full_ids: torch.Tensor, total_len: int, device: str, inp: BuildInputs
    ) -> dict:
        """Assemble the captured-forward kwargs (shared by both Qwen3-VL tasks).

        Copies this task's ``mm_keys`` multimodal tensors from the processor output and
        pads ``mm_token_type_ids`` out to the teacher-forced length (the processor
        produces it for the prompt only, so it must be zero-extended over the appended
        response, or the model errors on a shape mismatch). No ``attention_mask`` is
        passed: for a single unpadded sequence HF builds the all-ones causal mask itself.

        Args:
            full_ids: The full input-id tensor (prompt + response).
            total_len: Total sequence length (used to pad ``mm_token_type_ids``).
            device: CUDA device string (unused; kept for the callback signature).
            inp: The ``BuildInputs`` dict; supplies the processor ``inputs``.

        Returns:
            A kwargs dict for ``model(**fk)``: ``input_ids``, the present ``mm_keys``
            tensors, and (if present) a length-padded ``mm_token_type_ids``.
        """
        inputs = inp["inputs"]
        fk = {"input_ids": full_ids}
        for k in self.mm_keys:
            if k in inputs:
                fk[k] = inputs[k]
        if "mm_token_type_ids" in inputs:
            orig = inputs["mm_token_type_ids"]
            extra = total_len - orig.shape[1]
            fk["mm_token_type_ids"] = (
                torch.cat(
                    [orig, torch.zeros(1, extra, dtype=orig.dtype, device=orig.device)],
                    dim=1,
                )
                if extra > 0
                else orig
            )
        return fk


@register_model("qwen3vl_image")
class Qwen3VLImageAdapter(Qwen3VLBase):
    """Qwen3-VL-8B adapter for image object detection (key ``qwen3vl_image``).

    Parses JSON ``bbox_2d`` boxes and maps them to the merged patch grid. Used by image
    datasets (e.g. COCO).
    """

    modality_pad_token = "<|image_pad|>"
    mm_keys = ("pixel_values", "image_grid_thw")
    overlap = staticmethod(iou)  # boxes use spatial IoU for the hallucination test
    # Qwen3-VL's natural detection syntax is a JSON list of {"bbox_2d","label"}; this suffix elicits
    # it (parse_response reads that JSON). Matches the paper's COCO prompt for Qwen.
    detection_prompt_suffix = " and output the bbox coordinates in JSON format."

    def parse_response(self, response: str) -> list["Prediction"]:
        """Parse Qwen detection JSON into predictions.

        Tries strict JSON first (``[{"bbox_2d":[x1,y1,x2,y2],"label":...}, ...]``), then
        falls back to a regex that pairs each box with the ``"label"`` that follows it
        (the order Qwen emits).

        Args:
            response: The raw generated text (optionally fenced in a ```json block).

        Returns:
            A list of ``Prediction([x1,y1,x2,y2], label)`` with integer coords and
            lowercased labels, in response order (empty if none parse).
        """
        cleaned = _RE_JSON_FENCE.sub("", response).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                out = []
                for o in parsed:
                    if (
                        isinstance(o, dict)
                        and isinstance(o.get("bbox_2d"), list)
                        and len(o["bbox_2d"]) == 4
                    ):
                        out.append(
                            Prediction(
                                [int(x) for x in o["bbox_2d"]],
                                o.get("label", "").lower(),
                            )
                        )
                if out:
                    return out
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        matches = list(_RE_BBOX_2D.finditer(response))
        out = []
        for k, m in enumerate(matches):
            hi = matches[k + 1].start() if k + 1 < len(matches) else len(response)
            lm = _RE_LABEL.search(response[m.end() : hi])
            out.append(
                Prediction(
                    [int(m.group(i)) for i in range(1, 5)],
                    lm.group(1).lower() if lm else "",
                )
            )
        return out

    def build_vllm_request(
        self, proc: Any, item: dict, dataset: Any, cfg: Any
    ) -> dict | None:
        """Build a vLLM request for one image-detection item.

        Args:
            proc: The processor from ``gen_processor``.
            item: A raw dataset item (from ``load_items``); ``item["image"]`` is the path.
            dataset: The dataset object; supplies the prompt via ``dataset.prompt``.
            cfg: Run config (unused for images).

        Returns:
            A dict ``{prompt, multi_modal_data}`` with the chat-templated prompt and the
            processed image inputs (empty ``multi_modal_data`` if none were produced).
        """
        image = Image.open(item["image"]).convert("RGB")
        msgs = self._image_message(image, self.build_text_prompt(dataset, item))
        text = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
        return {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
        }

    # ---- MTLA extraction ----
    def _encode_attn_extraction_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> tuple["torch.Tensor", dict, dict, int] | None:
        """Encode the image + prompt and compute the merged-patch-grid geometry.

        Args:
            record: The generation record (prompt, ``extra["image"]``).
            ctx: The extraction context (``proc`` / ``device``).
            rank: Worker rank (unused; kept for the hook signature).

        Returns:
            ``(prompt_ids, {"inputs": ...}, {"task","grid_h","grid_w"}, grid_h*grid_w)``.

        Raises:
            ValueError: If the processor returns no ``image_grid_thw`` (a preprocessing
                bug: the region mask cannot be built without the patch grid).
        """
        proc, device = ctx["proc"], ctx["device"]
        img = Image.open(record["extra"]["image"]).convert("RGB")
        msgs = self._image_message(img, record["prompt"])
        text = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        if "image_grid_thw" not in inputs:
            raise ValueError(
                f"{record['id']}: processor returned no 'image_grid_thw' for an image "
                f"input; cannot map boxes to patch tokens"
            )
        _, h_, w_ = inputs["image_grid_thw"][0].tolist()
        grid_h, grid_w = h_ // 2, w_ // 2  # //2 = Qwen's 2x2 spatial-merge factor
        return (
            inputs["input_ids"][0],
            {"inputs": inputs},
            {"task": "image_det", "grid_h": grid_h, "grid_w": grid_w},
            grid_h * grid_w,
        )

    def locate_proposal_tokens(
        self, response: str, predictions: list["Prediction"], tokenizer: Any
    ) -> list[TokenRange | None]:
        """Locate each detection box's response tokens Q_p via tokenizer char offsets.

        Prefers the full ``"bbox_2d":[...],"label":"..."`` template (recovering both
        coordinate and label tokens); if only the label is found, coordinate tokens are
        left empty.

        Args:
            response: The raw generated text.
            predictions: The parsed boxes to locate, in order.
            tokenizer: The tokenizer providing an offset mapping over ``response``.

        Returns:
            One entry per prediction, index-aligned: a ``TokenRange`` dict
            ``{first_label_tok, label_toks, coord_toks}``, or ``None`` if the label could
            not be located.
        """
        enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        out: list[TokenRange | None] = []
        search_pos = 0
        for pred in predictions:
            label = pred.label
            full_re = re.compile(_BBOX_LABEL_TEMPLATE.format(label=re.escape(label)))
            m = full_re.search(response, search_pos) or full_re.search(response)
            if m:
                coord_ranges = [(m.start(g), m.end(g)) for g in range(1, 5)]
                label_start, label_end = m.end() - 1 - len(label), m.end() - 1
                search_pos = m.end()
            else:
                lp = re.compile(r'"label"\s*:\s*"' + re.escape(label) + r'"')
                ml = lp.search(response, search_pos) or lp.search(response)
                if not ml:
                    out.append(None)
                    continue
                mk = _RE_LABEL_MARKER.search(ml.group(0))
                assert mk is not None  # the pattern that produced ml contains it
                marker = mk.group(0)
                label_start = ml.start() + len(marker)
                label_end = label_start + len(label)
                coord_ranges = []
                search_pos = ml.end()
            label_toks = [
                ti
                for ti, (ts, te) in enumerate(offsets)
                if ts < label_end and te > label_start
            ]
            first = next(
                (ti for ti in label_toks if offsets[ti][0] >= label_start),
                label_toks[0] if label_toks else None,
            )
            coord_toks = [
                ti
                for (cs, ce) in coord_ranges
                for ti, (ts, te) in enumerate(offsets)
                if ts < ce and te > cs
            ]
            out.append(
                {
                    "first_label_tok": first,
                    "label_toks": label_toks,
                    "coord_toks": coord_toks,
                }
            )
        return out

    def proposal_region_attn_mask(
        self, prediction: "Prediction", meta: dict
    ) -> list[int]:
        """Map a box to merged-patch-grid token indices M(R_p).

        Returns the row-major token indices (``r * grid_w + c``) of the merged patches
        overlapping the box; the box is auto-ordered if its corners are reversed.

        Args:
            prediction: The prediction whose box ``[x1,y1,x2,y2]`` (in ``[0,1000]``)
                defines the region.
            meta: Must contain ``meta["grid_h"]`` and ``meta["grid_w"]`` (the merged
                patch grid dimensions).

        Returns:
            The image-token indices inside the box.
        """
        x1, y1, x2, y2 = prediction.region
        grid_h, grid_w = meta["grid_h"], meta["grid_w"]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        col_min = max(0, min(grid_w - 1, int(np.floor(x1 * grid_w / 1000.0))))
        col_max = max(0, min(grid_w - 1, int(np.floor((x2 - 1e-6) * grid_w / 1000.0))))
        row_min = max(0, min(grid_h - 1, int(np.floor(y1 * grid_h / 1000.0))))
        row_max = max(0, min(grid_h - 1, int(np.floor((y2 - 1e-6) * grid_h / 1000.0))))
        return [
            r * grid_w + c
            for r in range(row_min, row_max + 1)
            for c in range(col_min, col_max + 1)
        ]


@register_model("qwen3vl_video")
class Qwen3VLVideoAdapter(Qwen3VLBase):
    """Qwen3-VL-8B adapter for video temporal grounding (key ``qwen3vl_video``).

    Parses ``[start, end]`` spans and maps them to frame tokens. Used by video datasets
    (QVHighlights, Charades).
    """

    modality_pad_token = "<|video_pad|>"
    mm_keys = ("pixel_values_videos", "video_grid_thw")
    overlap = staticmethod(tiou)  # spans use temporal IoU for the hallucination test

    def parse_response(self, response: str) -> list["Prediction"]:
        """Parse a response into temporal spans (all occurrences).

        Args:
            response: The raw generated text.

        Returns:
            One ``Prediction([start, end], "")`` per span found (empty labels; spans in
            seconds), in start order.
        """
        spans, _ = self._parse_spans_with_offsets(response)
        return [Prediction(w, "") for w in spans]

    def build_vllm_request(
        self, proc: Any, item: dict, dataset: Any, cfg: Any
    ) -> dict | None:
        """Build a vLLM request for one video-grounding item.

        Args:
            proc: The processor from ``gen_processor``.
            item: A raw dataset item; the dataset resolves its clip via
                ``dataset.video_path``.
            dataset: The dataset object; supplies the prompt and video path.
            cfg: Run config; ``cfg.preprocess`` carries the video fps/pixel budget.

        Returns:
            A dict ``{prompt, multi_modal_data, mm_processor_kwargs}`` for vLLM, or
            ``None`` to skip (missing clip file, or no video decoded).
        """
        video_path = dataset.video_path(cfg, item)
        if not os.path.exists(video_path):
            print(f"[generate] skip: video not found {video_path}", flush=True)
            return None
        msgs = self._video_message(
            video_path, self.build_text_prompt(dataset, item), cfg.preprocess
        )
        text = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        _, videos, video_kwargs = process_vision_info(
            msgs,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if not videos:
            print(f"[generate] skip: no video decoded from {video_path}", flush=True)
            return None
        return {
            "prompt": text,
            "multi_modal_data": {"video": videos[0]},
            "mm_processor_kwargs": video_kwargs,
        }

    def _select_predictions(
        self, predictions: list["Prediction"], ctx: Ctx
    ) -> list["Prediction"]:
        """Keep all spans for multi-segment benchmarks, or just the first for single-span.

        Args:
            predictions: The parsed spans, in start order.
            ctx: The extraction context; ``ctx["multi"]`` is the multi-segment flag.

        Returns:
            All spans when ``ctx["multi"]``; otherwise only the first (Charades).
        """
        return predictions if ctx["multi"] else predictions[:1]

    def _encode_attn_extraction_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> tuple["torch.Tensor", dict, dict, int] | None:
        """Decode the clip + prompt and compute the frame-token geometry.

        Uses the same video preprocessing (fps / pixel budget) the generate stage used,
        then derives ``T*H*W`` frame tokens after the spatial merge and the clip duration.

        Args:
            record: The generation record (prompt, ``extra["video"]``).
            ctx: The extraction context (``proc`` / ``device`` / ``preprocess`` / ``model``).
            rank: Worker rank (unused; kept for the callback signature).

        Returns:
            ``(prompt_ids, {"inputs": ...}, {"task","duration_s","T","H","W"}, T*H*W)``,
            or ``None`` to skip a missing clip (tolerated: a few absent files should not
            kill a long run; the driver caps the total skip rate).

        Raises:
            ValueError: If the processor returns no valid ``video_grid_thw`` (a
                preprocessing bug: spans cannot be mapped to frame tokens). A present clip
                that fails to decode also propagates, since that is a real error, not
                missing data.
        """
        proc, device = ctx["proc"], ctx["device"]
        video_path = record["extra"]["video"]
        if not os.path.exists(video_path):
            print(
                f"[worker {rank}] skip {record['id']}: video not found {video_path}",
                flush=True,
            )
            return None
        msgs = self._video_message(video_path, record["prompt"], ctx["preprocess"])
        text = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            msgs,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        vids: Any = videos
        metas: list | None = None
        if videos:
            vids_t, metas_t = zip(*videos)
            vids, metas = list(vids_t), list(metas_t)
        inputs = proc(
            text=text,
            images=images,
            videos=vids,
            video_metadata=metas,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        ).to(device)
        vgthw = inputs.get("video_grid_thw")
        if vgthw is None or vgthw.shape[0] != 1:
            raise ValueError(
                f"{record['id']}: processor returned no valid 'video_grid_thw' for a "
                f"video input; cannot map spans to frame tokens"
            )
        T_grid, H_grid, W_grid = (int(vgthw[0, i].item()) for i in range(3))
        sms = getattr(ctx["model"].config.vision_config, "spatial_merge_size", 2)
        T, H, W = T_grid, H_grid // sms, W_grid // sms
        duration_s = float(
            record["extra"].get("duration_s") or video_duration(video_path)
        )
        return (
            inputs["input_ids"][0],
            {"inputs": inputs},
            {"task": "video_span", "duration_s": duration_s, "T": T, "H": H, "W": W},
            T * H * W,
        )

    def locate_proposal_tokens(
        self, response: str, predictions: list["Prediction"], tokenizer: Any
    ) -> list[TokenRange | None]:
        """Locate each predicted span's Q_p: the digit tokens of its timestamp match.

        Re-parses the response to recover per-span char offsets, keyed by 2-decimal-rounded
        ``(start, end)``, then keeps only the digit-bearing tokens overlapping that match.

        Args:
            response: The raw generated text.
            predictions: The parsed spans to locate (each with a ``[start,end]`` region),
                in order.
            tokenizer: The tokenizer providing an offset mapping over ``response``.

        Returns:
            One entry per prediction, index-aligned: a ``TokenRange`` dict
            ``{first_label_tok, label_toks: [], coord_toks}`` (label tokens are empty for
            spans), or ``None`` if the span's offsets or digit tokens could not be found.
        """
        enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        spans, offs = self._parse_spans_with_offsets(response)
        off_by_key = {
            (round(w[0], 2), round(w[1], 2)): sp for w, sp in zip(spans, offs)
        }
        out: list[TokenRange | None] = []
        for pred in predictions:
            w = pred.region
            sp = off_by_key.get((round(w[0], 2), round(w[1], 2)))
            if sp is None:
                out.append(None)
                continue
            cs, ce = sp
            toks = [
                ti
                for ti in tokens_overlapping_char_span(offsets, cs, ce)
                if any(c.isdigit() for c in response[offsets[ti][0] : offsets[ti][1]])
            ]
            out.append(
                {"first_label_tok": toks[0], "label_toks": [], "coord_toks": toks}
                if toks
                else None
            )
        return out

    def proposal_region_attn_mask(
        self, prediction: "Prediction", meta: dict
    ) -> list[int]:
        """Map a time span to frame-token indices M(R_p).

        Selects the frames whose timestamps fall in the span and expands them across all
        ``H*W`` spatial tokens. The video block is frame-major: frame ``t`` holds tokens
        ``[t*HW : (t+1)*HW)`` and frame ``t`` covers time ``t*duration/T``.

        Args:
            prediction: The prediction whose span ``[start, end]`` (seconds) defines the
                region.
            meta: Must contain ``meta["duration_s"]`` (clip duration) and the video token
                grid ``meta["T"]/["H"]/["W"]`` (temporal / spatial token counts).

        Returns:
            The frame-token indices inside the span; empty if the span is ``None``, the
            duration or token count is non-positive, or the span maps to no frames.
        """
        span = prediction.region
        duration_s, T_tokens = meta["duration_s"], meta["T"]
        HW = meta["H"] * meta["W"]
        if span is None or duration_s <= 0 or T_tokens <= 0:
            return []
        s, e = span
        fs = max(0, int(np.floor(s * T_tokens / duration_s)))
        fe = min(T_tokens, int(np.ceil(e * T_tokens / duration_s)))
        if fe <= fs:
            return []
        return [f * HW + k for f in range(fs, fe) for k in range(HW)]

    def _parse_spans_with_offsets(
        self,
        response: str,
    ) -> tuple[list[list[float]], list[tuple[int, int]]]:
        """Extract every ``[start, end]`` span in a response with its char offset.

        Matches all timestamp families (``[a,b]``, ``(a,b)``, ``from a to b``, etc.)
        against a length-preserving lowercase copy so offsets align with the original, so a
        response mixing formats keeps every window. Spans are order-normalized
        (``start <= end``), zero-length spans dropped, and duplicates removed by
        2-decimal-rounded key.

        Args:
            response: The raw generated text.

        Returns:
            A tuple ``(spans, offsets)`` aligned by index and sorted by start time: each
            span is ``[start, end]`` in seconds and each offset is the ``(char_start,
            char_end)`` of its match in ``response``.
        """
        t = (
            response.lower()
        )  # length-preserving (do NOT replace "seconds"->"s": would shift offsets)
        seen, rows = set(), []
        for pat in _SPAN_FORMATS:
            for m in re.finditer(pat, t):
                try:
                    a, b = parse_timestamp(m.group(1)), parse_timestamp(m.group(2))
                except ValueError:
                    continue
                if a > b:
                    a, b = b, a
                if a == b:
                    continue
                key = (round(a, 2), round(b, 2))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(([a, b], m.span()))
        rows.sort(key=lambda r: r[0][0])  # order by start time
        return [w for w, _ in rows], [sp for _, sp in rows]

    def _video_message(self, video_path: str, prompt: str, pre: dict) -> list[dict]:
        """Build chat messages for one video clip with the config's preprocessing.

        Args:
            video_path: Filesystem path to the video. Absolutized before wrapping as a
                ``file://`` URI: a ``file://`` authority must be an absolute path, so a
                relative ``video_path`` (e.g. ``data/.../clip.mp4``) would parse its first
                segment as a hostname and fail to open even though the file exists.
            prompt: The text prompt to append after the video.
            pre: Preprocessing dict with ``min_pixels``, ``max_pixels`` and ``fps``.

        Returns:
            A single-turn chat-message list ``[{"role": "user", "content": [...]}]``.
        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{os.path.abspath(video_path)}",
                        "min_pixels": pre["min_pixels"],
                        "max_pixels": pre["max_pixels"],
                        "fps": pre["fps"],
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
