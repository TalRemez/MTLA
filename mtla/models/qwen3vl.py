"""Qwen3-VL-8B model adapter for image detection and temporal grounding.

One ``AutoProcessor`` serves both tasks:
  * ``image_det``  : COCO detection. ``parse`` reads JSON ``{"bbox_2d","label"}``; the
    region masks onto the fixed merged patch grid. Reproduces the paper's COCO AUROC
    0.902.
  * ``video_span`` : temporal grounding (QVHighlights multi-segment, Charades
    single-span). ``parse`` extracts ``[start, end]`` spans; the region masks onto the
    frame tokens inside a span.

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
from decord import VideoReader, cpu
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from typing import Any, cast

from mtla.models.base import ModelAdapter, Prediction, hallucinated
from mtla.registry import register_model
from mtla.utils import iou, tiou, tokens_overlapping_char_span
from mtla.types import BuildInputs, Ctx, GenRecord, TokenRange

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_TIME = r"(\d{1,3}(?::\d{2})?(?:\.\d+)?)"
_PATTERNS = [
    rf"\[\s*{_TIME}\s*,\s*{_TIME}\s*\]",
    rf"\(\s*{_TIME}\s*,\s*{_TIME}\s*\)",
    rf"from\s+{_TIME}\s*s?\s+to\s+{_TIME}",
    rf"between\s+{_TIME}\s*s?\s+and\s+{_TIME}",
    rf"{_TIME}\s*s?\s*-\s*{_TIME}\s*s",
    rf"{_TIME}\s*s?\s*-\s*{_TIME}",
    rf"{_TIME}\s*s\s+to\s+{_TIME}",
    rf"start[:\s]+{_TIME}\s*s?\s*,?\s*end[:\s]+{_TIME}",
]


def _to_seconds(s: str) -> float:
    """Convert a matched timestamp string to seconds.

    Args:
        s: A timestamp, either ``"MM:SS(.fff)"`` or a plain ``"SSS(.fff)"`` seconds
            value.

    Returns:
        The timestamp in seconds as a float.
    """
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)


def parse_spans_with_offsets(
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
    for pat in _PATTERNS:
        for m in re.finditer(pat, t):
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
            seen.add(key)
            rows.append(([a, b], m.span()))
    rows.sort(key=lambda r: r[0][0])  # order by start time
    return [w for w, _ in rows], [sp for _, sp in rows]


def parse_spans(response: str, multi: bool = True) -> list["Prediction"]:
    """Parse temporal spans from a response into predictions.

    Args:
        response: The raw generated text.
        multi: If ``True`` keep all spans; if ``False`` keep only the first
            (start-ordered).

    Returns:
        A list of ``Prediction([start, end], "")`` (empty labels; spans in seconds).
    """
    spans, _ = parse_spans_with_offsets(response)
    preds = [Prediction(w, "") for w in spans]
    return preds if multi else preds[:1]


def parse_bboxes(response: str) -> list["Prediction"]:
    """Parse Qwen detection JSON into predictions.

    Tries strict JSON first (``[{"bbox_2d":[x1,y1,x2,y2],"label":...}, ...]``), then
    falls back to a regex that pairs each box with the ``"label"`` that follows it (the
    order Qwen emits).

    Args:
        response: The raw generated text (optionally fenced in a ```json block).

    Returns:
        A list of ``Prediction([x1,y1,x2,y2], label)`` with integer coords and
        lowercased labels, in response order (empty if none parse).
    """
    cleaned = re.sub(r"```json\s*|```\s*", "", response).strip()
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
                            [int(x) for x in o["bbox_2d"]], o.get("label", "").lower()
                        )
                    )
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
        lm = label_re.search(response[m.end() : hi])
        out.append(
            Prediction(
                [int(m.group(i)) for i in range(1, 5)],
                lm.group(1).lower() if lm else "",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Region masks + Q_p token finders
# ---------------------------------------------------------------------------
def bbox_region_mask(bbox: list[float], grid_h: int, grid_w: int) -> list[int]:
    """Image-token indices inside a bbox on Qwen's merged patch grid M(R_p).

    Args:
        bbox: The box ``[x1,y1,x2,y2]`` in ``[0,1000]`` coordinates (auto-ordered if
            reversed).
        grid_h: Number of rows in the merged patch grid.
        grid_w: Number of columns in the merged patch grid.

    Returns:
        The row-major token indices (``r * grid_w + c``) of patches overlapping the box.
    """
    x1, y1, x2, y2 = bbox
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


def span_region_mask(
    span: list[float] | None,
    duration_s: float,
    T_tokens: int,
    H_tokens: int,
    W_tokens: int,
) -> list[int]:
    """Frame-token indices inside a time span M(R_p).

    Selects the frames whose timestamps fall in the span and expands them across all
    ``H*W`` spatial tokens. The video block is frame-major: frame ``t`` holds tokens
    ``[t*HW : (t+1)*HW)`` and frame ``t`` covers time ``t*duration/T_tokens``.

    Args:
        span: The time span ``[start, end]`` in seconds, or ``None``.
        duration_s: Clip duration in seconds (maps time to frame index).
        T_tokens: Number of temporal (frame) tokens.
        H_tokens: Spatial-token grid height per frame.
        W_tokens: Spatial-token grid width per frame.

    Returns:
        The frame-token indices inside the span; empty if ``span`` is ``None``, the
        duration or token count is non-positive, or the span maps to no frames.
    """
    HW = H_tokens * W_tokens
    if span is None or duration_s <= 0 or T_tokens <= 0:
        return []
    s, e = span
    fs = max(0, int(np.floor(s * T_tokens / duration_s)))
    fe = min(T_tokens, int(np.ceil(e * T_tokens / duration_s)))
    if fe <= fs:
        return []
    return [f * HW + k for f in range(fs, fe) for k in range(HW)]


def find_bbox_token_ranges(
    response: str, predictions: list["Prediction"], tokenizer: Any
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
    full_tmpl = (
        r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,'
        r'\s*"label"\s*:\s*"{label}"'
    )
    for pred in predictions:
        label = pred.label
        m = re.compile(full_tmpl.format(label=re.escape(label))).search(
            response, search_pos
        ) or re.compile(full_tmpl.format(label=re.escape(label))).search(response)
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
            mk = re.search(r'"label"\s*:\s*"', ml.group(0))
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


def find_span_token_ranges(
    response: str, predictions: list["Prediction"], tokenizer: Any
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
    spans, offs = parse_spans_with_offsets(response)
    off_by_key = {(round(w[0], 2), round(w[1], 2)): sp for w, sp in zip(spans, offs)}
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


# ---------------------------------------------------------------------------
# Vision-input builders (shared by generate + extract)
# ---------------------------------------------------------------------------
def _video_messages(video_path: str, prompt: str, pre: dict) -> list[dict]:
    """Build chat messages for one video clip with the config's preprocessing.

    Args:
        video_path: Filesystem path to the video (wrapped as a ``file://`` URI).
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
                    "video": f"file://{video_path}",
                    "min_pixels": pre["min_pixels"],
                    "max_pixels": pre["max_pixels"],
                    "fps": pre["fps"],
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _process_video(proc: Any, msgs: list[dict], device: str) -> Any:
    """Run the processor on video messages into model-ready inputs.

    Chat-templates the messages, extracts vision info (frames + metadata) via
    ``process_vision_info``, and runs the processor with resizing disabled so the frames
    match those used at generation.

    Args:
        proc: The Qwen3-VL processor.
        msgs: Chat messages from ``_video_messages``.
        device: CUDA device string to move the tensors to.

    Returns:
        The processor's batch-encoding ``inputs`` moved to ``device`` (includes
        ``input_ids``, ``video_grid_thw``, ``pixel_values_videos``, etc.).
    """
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
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
    return inputs


@register_model("qwen3vl")
class Qwen3VLAdapter(ModelAdapter):
    """Adapter for Qwen3-VL-8B image detection and video temporal grounding.

    Supports ``image_det`` (JSON ``bbox_2d`` boxes on the merged patch grid) and
    ``video_span`` (``[start,end]`` spans on frame tokens). The active task is chosen
    per item via ``ctx["task"]`` / ``self._task``.
    """

    model_id = MODEL_ID
    attn_module_path = "transformers.models.qwen3_vl.modeling_qwen3_vl"
    tasks = ("image_det", "video_span")

    def parse(
        self, response: str, task: str | None = "video_span"
    ) -> list["Prediction"]:
        """Parse a response into detections or spans depending on ``task``.

        Args:
            response: The raw generated text.
            task: ``image_det`` for boxes or ``video_span`` (default) for spans;
                validated against ``tasks``.

        Returns:
            Detection boxes (``image_det``) or all temporal spans (``video_span``).

        Raises:
            ValueError: If ``task`` is unsupported.
        """
        self._check_task(task)
        if task == "image_det":
            return parse_bboxes(response)
        return parse_spans(response, multi=True)

    # ---- vLLM generation ----
    def vllm_engine_args(self, dataset: Any) -> dict:
        """vLLM engine kwargs, per the dataset's modality.

        Args:
            dataset: The dataset being generated for; ``dataset.task`` picks the limit.

        Returns:
            For video, ``{"limit_mm_per_prompt": {"video": 1}, "max_model_len":
            32768}``; otherwise ``{"limit_mm_per_prompt": {"image": 1}}``.
        """
        if dataset.task == "video_span":
            return {"limit_mm_per_prompt": {"video": 1}, "max_model_len": 32768}
        return {"limit_mm_per_prompt": {"image": 1}}

    def vllm_uses_seed(self, task: str) -> bool:
        """Whether to seed vLLM SamplingParams.

        COCO does not seed vLLM (matches the validated paper preds); video does (each
        rollout is a fresh draw).

        Args:
            task: The task family for the run.

        Returns:
            ``True`` for ``video_span``, ``False`` for ``image_det``.
        """
        # COCO did NOT seed vLLM (matches the validated paper preds); video DOES (per-rollout draw).
        return task == "video_span"

    def build_request(
        self, proc: Any, item: dict, dataset: Any, cfg: Any
    ) -> dict | None:
        """Build a vLLM request for one image or video item.

        Args:
            proc: The processor from ``gen_processor``.
            item: A raw dataset item (from ``load_items``); the dataset owns its prompt
                and media path.
            dataset: The dataset object; supplies ``dataset.task``, ``dataset.prompt``
                and ``dataset.video_path``.
            cfg: Run config; ``cfg.preprocess`` carries the video fps/pixel budget.

        Returns:
            For video, ``{prompt, multi_modal_data: {video}, mm_processor_kwargs}``; for
            images, ``{prompt, multi_modal_data}``. ``None`` if the media is missing or
            no frames are produced.
        """
        # `item` is a raw dataset item (from load_items); the dataset owns its prompt + media path.
        if dataset.task == "video_span":
            video_path = dataset.video_path(cfg, item)
            if not os.path.exists(video_path):
                return None
            msgs = _video_messages(video_path, dataset.prompt(item), cfg.preprocess)
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
                return None
            return {
                "prompt": text,
                "multi_modal_data": {"video": videos[0]},
                "mm_processor_kwargs": video_kwargs,
            }
        image = Image.open(item["image"]).convert("RGB")
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": dataset.prompt(item)},
                ],
            }
        ]
        text = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
        return {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs} if image_inputs else {},
        }

    # ---- MTLA extraction ----
    def _load_model(self, device: str) -> Any:
        """Load Qwen3-VL-8B for extraction.

        Args:
            device: CUDA device string; passed as ``device_map``.

        Returns:
            The bf16 ``Qwen3VLForConditionalGeneration`` in eval mode with eager
            attention.
        """
        return Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=device,
        ).eval()

    def _pad_token(self, task: str) -> str:
        """The modality pad token for ``task``.

        Args:
            task: The task family for the run.

        Returns:
            ``<|video_pad|>`` for ``video_span``, else ``<|image_pad|>``.
        """
        return "<|video_pad|>" if task == "video_span" else "<|image_pad|>"

    def build_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> BuildInputs | None:
        """Assemble MTLA driver inputs, dispatching on the context's task.

        Args:
            record: The generation record for one item.
            ctx: The extraction context; ``ctx["task"]`` selects image vs. video.
            rank: Worker rank, used only for logging skips.

        Returns:
            The ``BuildInputs`` dict from the image or video builder, or ``None`` to
            skip.
        """
        if ctx["task"] == "video_span":
            return self._video_inputs(record, ctx, rank)
        return self._image_inputs(record, ctx, rank)

    def query_tokens(
        self, response: str, predictions: list["Prediction"], tokenizer: Any
    ) -> list[TokenRange | None]:
        """Locate each prediction's Q_p, dispatching on the active task.

        Args:
            response: The raw generated text.
            predictions: The parsed predictions to locate, in order.
            tokenizer: The tokenizer providing an offset mapping.

        Returns:
            One ``TokenRange`` or ``None`` per prediction, index-aligned: span digit
            tokens for ``video_span``, box label+coord tokens otherwise.
        """
        if self._task == "video_span":
            return find_span_token_ranges(response, predictions, tokenizer)
        return find_bbox_token_ranges(response, predictions, tokenizer)

    def region_mask(self, prediction: "Prediction", meta: dict) -> list[int]:
        """Map a prediction to modality-token indices M(R_p), dispatching on task.

        Args:
            prediction: The prediction whose region defines M(R_p).
            meta: Geometry from ``build_inputs``. For video: ``task``, ``duration_s``,
                ``T``, ``H``, ``W``; for images: ``task``, ``grid_h``, ``grid_w``.

        Returns:
            Frame-token indices for ``video_span`` (from ``span_region_mask``) or
            image-token indices otherwise (from ``bbox_region_mask``).
        """
        if meta["task"] == "video_span":
            return span_region_mask(
                prediction.region, meta["duration_s"], meta["T"], meta["H"], meta["W"]
            )
        return bbox_region_mask(prediction.region, meta["grid_h"], meta["grid_w"])

    def forward_kwargs(
        self, full_ids: torch.Tensor, total_len: int, device: str, inp: BuildInputs
    ) -> dict:
        """Build kwargs for the captured Qwen3-VL forward.

        Forwards whichever of the image/video pixel and grid tensors are present, and
        right-pads ``mm_token_type_ids`` to ``total_len`` when needed.

        Args:
            full_ids: The full input-id tensor (prompt + response).
            total_len: Total sequence length, used to size the mask and pad token types.
            device: CUDA device string for the attention mask.
            inp: The ``BuildInputs`` dict; ``inp["inputs"]`` holds the processor
                outputs.

        Returns:
            A kwargs dict for ``model(**fk)`` with ``input_ids``, ``attention_mask``,
            any present pixel/grid tensors, and a padded ``mm_token_type_ids``.
        """
        inputs = inp["inputs"]
        fk = {
            "input_ids": full_ids,
            "attention_mask": torch.ones(1, total_len, device=device, dtype=torch.long),
        }
        for k in [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ]:
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

    # ---- per-task input builders ----
    def _image_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> BuildInputs | None:
        """Assemble MTLA inputs for one COCO detection item.

        Opens the image, chat-templates the prompt, derives the merged patch grid from
        ``image_grid_thw`` (halved per spatial-merge), verifies the image-token count,
        and computes per-box hallucination flags with IoU.

        Args:
            record: The generation record (prompt, response, gt, ``extra["image"]``).
            ctx: The extraction context.
            rank: Worker rank, used only for logging skips.

        Returns:
            A ``BuildInputs`` dict (with ``inputs`` and ``meta`` grid_h/grid_w), or
            ``None`` to skip (no boxes, unreadable image, missing grid, or token-count
            mismatch).
        """
        proc = ctx["proc"]
        device = ctx["device"]
        pad_id = ctx["pad_id"]
        response = record.get("response")
        preds = parse_bboxes(response) if response else []
        if not preds:
            return None
        try:
            img = Image.open(record["extra"]["image"]).convert("RGB")
        except Exception as e:
            print(f"[worker {rank}] skip {record['id']}: img {e}", flush=True)
            return None
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": record["prompt"]},
                ],
            }
        ]
        text = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0]
        if "image_grid_thw" not in inputs:
            return None
        _, h_, w_ = inputs["image_grid_thw"][0].tolist()
        grid_h, grid_w = h_ // 2, w_ // 2
        image_idx = [k for k, t in enumerate(prompt_ids.tolist()) if t == pad_id]
        if len(image_idx) != grid_h * grid_w:
            print(
                f"[worker {rank}] skip {record['id']}: img tokens {len(image_idx)} != {grid_h*grid_w}",
                flush=True,
            )
            return None
        hallu = [
            hallucinated(p.region, p.label, record.get("gt", []), iou) for p in preds
        ]
        return cast(
            BuildInputs,
            {
                "prompt_ids": prompt_ids,
                "response": response,
                "modality_idx_l": image_idx,
                "predictions": preds,
                "hallu_flags": hallu,
                "inputs": inputs,
                "meta": {"task": "image_det", "grid_h": grid_h, "grid_w": grid_w},
            },
        )

    def _video_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> BuildInputs | None:
        """Assemble MTLA inputs for one temporal-grounding item.

        Processes the video with the config preprocessing, derives ``(T, H, W)`` from
        ``video_grid_thw`` (spatial dims halved by ``spatial_merge_size``), verifies the
        video-token count, resolves the clip duration, and computes per-span
        hallucination flags with tIoU (labels are empty, so the test reduces to overlap,
        the same rule as the image path).

        Args:
            record: The generation record (prompt, response, gt, ``extra["video"]``,
                optional ``extra["duration_s"]``).
            ctx: The extraction context; provides ``preprocess`` and ``multi``.
            rank: Worker rank, used only for logging skips.

        Returns:
            A ``BuildInputs`` dict (with ``inputs`` and ``meta`` duration/T/H/W), or
            ``None`` to skip (no spans, missing video, processor error, missing grid, or
            token-count mismatch).
        """
        proc = ctx["proc"]
        device = ctx["device"]
        pad_id = ctx["pad_id"]
        pre = ctx["preprocess"]
        multi = ctx["multi"]
        response = record.get("response")
        preds = parse_spans(response, multi=multi) if response else []
        if not preds:
            return None
        video_path = record["extra"]["video"]
        if not os.path.exists(video_path):
            return None
        try:
            inputs = _process_video(
                proc, _video_messages(video_path, record["prompt"], pre), device
            )
        except Exception as e:
            print(f"[worker {rank}] skip {video_path}: processor {e}", flush=True)
            return None
        prompt_ids = inputs["input_ids"][0]
        vgthw = inputs.get("video_grid_thw")
        if vgthw is None or vgthw.shape[0] != 1:
            return None
        T_grid, H_grid, W_grid = (int(vgthw[0, i].item()) for i in range(3))
        sms = getattr(ctx["model"].config.vision_config, "spatial_merge_size", 2)
        T, H, W = T_grid, H_grid // sms, W_grid // sms
        video_idx = [k for k, t in enumerate(prompt_ids.tolist()) if t == pad_id]
        if not video_idx or len(video_idx) != T * H * W:
            print(
                f"[worker {rank}] skip {video_path}: video tokens {len(video_idx)} != {T*H*W}",
                flush=True,
            )
            return None
        duration_s = float(
            record["extra"].get("duration_s") or video_duration(video_path)
        )
        # A span is grounded iff it overlaps some GT window by tIoU >= 0.5 (labels are empty for
        # spans, so `hallucinated` reduces to the overlap test — same rule as the image path).
        hallu = [
            hallucinated(p.region, p.label, record.get("gt", []), tiou) for p in preds
        ]
        return cast(
            BuildInputs,
            {
                "prompt_ids": prompt_ids,
                "response": response,
                "modality_idx_l": video_idx,
                "predictions": preds,
                "hallu_flags": hallu,
                "inputs": inputs,
                "meta": {
                    "task": "video_span",
                    "duration_s": duration_s,
                    "T": T,
                    "H": H,
                    "W": W,
                },
            },
        )


def video_duration(video_path: str) -> float:
    """Read a video's duration in seconds via decord.

    Used as a fallback when the record does not carry ``duration_s``.

    Args:
        video_path: Filesystem path to the video.

    Returns:
        Duration in seconds (frame count / average fps), or ``0.0`` if fps is
        non-positive.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    return len(vr) / fps if fps > 0 else 0.0
