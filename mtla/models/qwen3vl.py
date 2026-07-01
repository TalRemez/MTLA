"""Qwen3-VL-8B model adapter (image detection + temporal grounding).

One ``AutoProcessor`` serves both tasks:
  - ``image_det``  : COCO detection. ``parse`` reads JSON ``{"bbox_2d","label"}``; the region masks
    onto the fixed merged patch grid. Reproduces the paper's COCO AUROC 0.902.
  - ``video_span`` : temporal grounding (QVHighlights multi-segment, Charades single-span).
    ``parse`` extracts ``[start, end]`` spans; the region masks onto the frame tokens inside a span.

Video preprocessing (fps / pixel budget) comes from ``cfg.preprocess`` and feeds both the generate
and extract stages, so the two stages see the same frames. The LLM backbone attention is captured
at ``transformers.models.qwen3_vl.modeling_qwen3_vl``.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from .base import ModelAdapter, Prediction, hallucinated
from ..registry import register_model
from ..mtla_attn import CaptureState, install_capture
from ..utils import iou, tiou, tokens_overlapping_char_span

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
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


def _to_seconds(s: str) -> float:
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)


def parse_spans_with_offsets(response: str):
    """Every ``[start, end]`` span in ``response`` with its char offset, deduped and start-ordered.

    Collects matches across all timestamp families against a length-preserving lowercase copy (so
    offsets align with the original), so a response mixing formats keeps every window. Returns
    ``(spans, offsets)`` aligned by index.
    """
    t = response.lower()   # length-preserving (do NOT replace "seconds"->"s": would shift offsets)
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
            seen.add(key); rows.append(([a, b], m.span()))
    rows.sort(key=lambda r: r[0][0])   # order by start time
    return [w for w, _ in rows], [sp for _, sp in rows]


def parse_spans(response: str, multi: bool = True) -> list:
    """Temporal spans as ``[Prediction([start, end], "")]``. ``multi=False`` keeps only the first."""
    spans, _ = parse_spans_with_offsets(response)
    preds = [Prediction(w, "") for w in spans]
    return preds if multi else preds[:1]


def parse_bboxes(response: str) -> list:
    """Parse Qwen detection JSON ``[{"bbox_2d":[x1,y1,x2,y2],"label":...}, ...]`` into Predictions.
    JSON first, then a regex fallback (label is the one AFTER each box, as Qwen emits it)."""
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
        out.append(Prediction([int(m.group(i)) for i in range(1, 5)], lm.group(1).lower() if lm else ""))
    return out


# ---------------------------------------------------------------------------
# Region masks + Q_p token finders
# ---------------------------------------------------------------------------
def bbox_region_mask(bbox, grid_h, grid_w) -> list:
    """Modality-token indices inside ``bbox`` on Qwen's fixed ``grid_h x grid_w`` merged patch grid."""
    x1, y1, x2, y2 = bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    col_min = max(0, min(grid_w - 1, int(np.floor(x1 * grid_w / 1000.0))))
    col_max = max(0, min(grid_w - 1, int(np.floor((x2 - 1e-6) * grid_w / 1000.0))))
    row_min = max(0, min(grid_h - 1, int(np.floor(y1 * grid_h / 1000.0))))
    row_max = max(0, min(grid_h - 1, int(np.floor((y2 - 1e-6) * grid_h / 1000.0))))
    return [r * grid_w + c for r in range(row_min, row_max + 1) for c in range(col_min, col_max + 1)]


def span_region_mask(span, duration_s, T_tokens, H_tokens, W_tokens) -> list:
    """Modality-token indices inside a time ``span`` M(R_p): the frames whose timestamps fall in the
    span, expanded across all H*W spatial tokens. The video block is frame-major: frame ``t`` holds
    tokens ``[t*HW : (t+1)*HW)``; token ``t`` covers time ``t*duration/T_tokens``."""
    HW = H_tokens * W_tokens
    if span is None or duration_s <= 0 or T_tokens <= 0:
        return []
    s, e = span
    fs = max(0, int(np.floor(s * T_tokens / duration_s)))
    fe = min(T_tokens, int(np.ceil(e * T_tokens / duration_s)))
    if fe <= fs:
        return []
    return [f * HW + k for f in range(fs, fe) for k in range(HW)]


def find_bbox_token_ranges(response, predictions, tokenizer):
    """Per predicted box, its response tokens Q_p (label + coordinate tokens), via char offsets."""
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out, search_pos = [], 0
    full_tmpl = (r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,'
                 r'\s*"label"\s*:\s*"{label}"')
    for pred in predictions:
        label = pred.label
        m = re.compile(full_tmpl.format(label=re.escape(label))).search(response, search_pos) \
            or re.compile(full_tmpl.format(label=re.escape(label))).search(response)
        if m:
            coord_ranges = [(m.start(g), m.end(g)) for g in range(1, 5)]
            label_start, label_end = m.end() - 1 - len(label), m.end() - 1
            search_pos = m.end()
        else:
            lp = re.compile(r'"label"\s*:\s*"' + re.escape(label) + r'"')
            ml = lp.search(response, search_pos) or lp.search(response)
            if not ml:
                out.append(None); continue
            marker = re.search(r'"label"\s*:\s*"', ml.group(0)).group(0)
            label_start = ml.start() + len(marker); label_end = label_start + len(label)
            coord_ranges = []; search_pos = ml.end()
        label_toks = [ti for ti, (ts, te) in enumerate(offsets) if ts < label_end and te > label_start]
        first = next((ti for ti in label_toks if offsets[ti][0] >= label_start),
                     label_toks[0] if label_toks else None)
        coord_toks = [ti for (cs, ce) in coord_ranges
                      for ti, (ts, te) in enumerate(offsets) if ts < ce and te > cs]
        out.append({"first_label_tok": first, "label_toks": label_toks, "coord_toks": coord_toks})
    return out


def find_span_token_ranges(response, predictions, tokenizer):
    """Per predicted span (``Prediction`` with a ``[start,end]`` region), the response digit tokens
    inside its timestamp match — its Q_p. Aligned index-for-index with ``predictions``."""
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    spans, offs = parse_spans_with_offsets(response)
    off_by_key = {(round(w[0], 2), round(w[1], 2)): sp for w, sp in zip(spans, offs)}
    out = []
    for pred in predictions:
        w = pred.region
        sp = off_by_key.get((round(w[0], 2), round(w[1], 2)))
        if sp is None:
            out.append(None); continue
        cs, ce = sp
        toks = [ti for ti in tokens_overlapping_char_span(offsets, cs, ce)
                if any(c.isdigit() for c in response[offsets[ti][0]:offsets[ti][1]])]
        out.append({"first_label_tok": toks[0], "label_toks": [], "coord_toks": toks} if toks else None)
    return out


# ---------------------------------------------------------------------------
# Vision-input builders (shared by generate + extract)
# ---------------------------------------------------------------------------
def _video_messages(video_path, prompt, pre):
    """Chat messages for one video clip with the config's preprocessing (fps / pixel budget)."""
    return [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video_path}",
         "min_pixels": pre["min_pixels"], "max_pixels": pre["max_pixels"], "fps": pre["fps"]},
        {"type": "text", "text": prompt}]}]


def _process_video(proc, msgs, device):
    """Run the processor on video messages -> (inputs on device, text). None on failure."""
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    if videos:
        videos, metas = zip(*videos)
        videos, metas = list(videos), list(metas)
    else:
        metas = None
    inputs = proc(text=text, images=images, videos=videos, video_metadata=metas,
                  do_resize=False, return_tensors="pt", **video_kwargs).to(device)
    return inputs


@register_model("qwen3vl")
class Qwen3VLAdapter(ModelAdapter):
    model_id = MODEL_ID
    attn_module_path = "transformers.models.qwen3_vl.modeling_qwen3_vl"
    tasks = ("image_det", "video_span")

    def parse(self, response, task="video_span"):
        self._check_task(task)
        if task == "image_det":
            return parse_bboxes(response)
        return parse_spans(response, multi=True)

    # ---- vLLM generation ----
    def vllm_engine_args(self, dataset):
        if dataset.task == "video_span":
            return {"limit_mm_per_prompt": {"video": 1}, "max_model_len": 32768}
        return {"limit_mm_per_prompt": {"image": 1}}

    def vllm_uses_seed(self, task):
        # COCO did NOT seed vLLM (matches the validated paper preds); video DOES (per-rollout draw).
        return task == "video_span"

    def build_request(self, proc, item, dataset, cfg):
        # `item` is a raw dataset item (from load_items); the dataset owns its prompt + media path.
        if dataset.task == "video_span":
            video_path = dataset.video_path(cfg, item)
            if not os.path.exists(video_path):
                return None
            msgs = _video_messages(video_path, dataset.prompt(item), cfg.preprocess)
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            _, videos, video_kwargs = process_vision_info(
                msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
            if not videos:
                return None
            return {"prompt": text, "multi_modal_data": {"video": videos[0]},
                    "mm_processor_kwargs": video_kwargs}
        image = Image.open(item["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": dataset.prompt(item)}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
        return {"prompt": text, "multi_modal_data": {"image": image_inputs} if image_inputs else {}}

    # ---- MTLA extraction ----
    def load_for_extract(self, gpu_id, task="image_det"):
        device = f"cuda:{gpu_id}"
        state = CaptureState()
        install_capture(self.attn_module_path, state)
        proc = AutoProcessor.from_pretrained(self.model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
        layers = model.model.language_model.layers
        state.layer_ids = {id(L.self_attn): i for i, L in enumerate(layers)}
        pad = "<|video_pad|>" if task == "video_span" else "<|image_pad|>"
        return {"model": model, "proc": proc, "tokenizer": proc.tokenizer, "state": state,
                "device": device, "task": task, "n_layers": len(layers),
                "n_heads": model.config.text_config.num_attention_heads,
                "pad_id": proc.tokenizer.convert_tokens_to_ids(pad)}

    def build_inputs(self, record, ctx, rank):
        if ctx["task"] == "video_span":
            return self._video_inputs(record, ctx, rank)
        return self._image_inputs(record, ctx, rank)

    def query_tokens(self, response, predictions, tokenizer):
        if self._task == "video_span":
            return find_span_token_ranges(response, predictions, tokenizer)
        return find_bbox_token_ranges(response, predictions, tokenizer)

    def region_mask(self, prediction, meta):
        if meta["task"] == "video_span":
            return span_region_mask(prediction.region, meta["duration_s"], meta["T"], meta["H"], meta["W"])
        return bbox_region_mask(prediction.region, meta["grid_h"], meta["grid_w"])

    def forward_kwargs(self, full_ids, total_len, device, inp):
        inputs = inp["inputs"]
        fk = {"input_ids": full_ids,
              "attention_mask": torch.ones(1, total_len, device=device, dtype=torch.long)}
        for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if k in inputs:
                fk[k] = inputs[k]
        if "mm_token_type_ids" in inputs:
            orig = inputs["mm_token_type_ids"]
            extra = total_len - orig.shape[1]
            fk["mm_token_type_ids"] = (torch.cat(
                [orig, torch.zeros(1, extra, dtype=orig.dtype, device=orig.device)], dim=1)
                if extra > 0 else orig)
        return fk

    # ---- per-task input builders ----
    def _image_inputs(self, record, ctx, rank):
        proc = ctx["proc"]; device = ctx["device"]; pad_id = ctx["pad_id"]
        response = record.get("response")
        preds = parse_bboxes(response) if response else []
        if not preds:
            return None
        try:
            img = Image.open(record["extra"]["image"]).convert("RGB")
        except Exception as e:
            print(f"[worker {rank}] skip {record['id']}: img {e}", flush=True)
            return None
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": record["prompt"]}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0]
        if "image_grid_thw" not in inputs:
            return None
        _, h_, w_ = inputs["image_grid_thw"][0].tolist()
        grid_h, grid_w = h_ // 2, w_ // 2
        image_idx = [k for k, t in enumerate(prompt_ids.tolist()) if t == pad_id]
        if len(image_idx) != grid_h * grid_w:
            print(f"[worker {rank}] skip {record['id']}: img tokens {len(image_idx)} != {grid_h*grid_w}",
                  flush=True)
            return None
        hallu = [hallucinated(p.region, p.label, record.get("gt", []), iou) for p in preds]
        return {"prompt_ids": prompt_ids, "response": response, "modality_idx_l": image_idx,
                "predictions": preds, "hallu_flags": hallu, "inputs": inputs,
                "meta": {"task": "image_det", "grid_h": grid_h, "grid_w": grid_w}}

    def _video_inputs(self, record, ctx, rank):
        proc = ctx["proc"]; device = ctx["device"]; pad_id = ctx["pad_id"]
        pre = ctx["preprocess"]; multi = ctx["multi"]
        response = record.get("response")
        preds = parse_spans(response, multi=multi) if response else []
        if not preds:
            return None
        video_path = record["extra"]["video"]
        if not os.path.exists(video_path):
            return None
        try:
            inputs = _process_video(proc, _video_messages(video_path, record["prompt"], pre), device)
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
            print(f"[worker {rank}] skip {video_path}: video tokens {len(video_idx)} != {T*H*W}",
                  flush=True)
            return None
        duration_s = float(record["extra"].get("duration_s") or video_duration(video_path))
        # A span is grounded iff it overlaps some GT window by tIoU >= 0.5 (labels are empty for
        # spans, so `hallucinated` reduces to the overlap test — same rule as the image path).
        hallu = [hallucinated(p.region, p.label, record.get("gt", []), tiou) for p in preds]
        return {"prompt_ids": prompt_ids, "response": response, "modality_idx_l": video_idx,
                "predictions": preds, "hallu_flags": hallu, "inputs": inputs,
                "meta": {"task": "video_span", "duration_s": duration_s, "T": T, "H": H, "W": W}}


def video_duration(video_path) -> float:
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    return len(vr) / fps if fps > 0 else 0.0
