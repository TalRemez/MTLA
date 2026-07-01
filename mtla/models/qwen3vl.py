"""Qwen3-VL-8B model adapter.

Supports two task families:
  - "video_span": temporal grounding (QVHighlights multi-segment, Charades single-span). `parse`
    extracts [start,end] spans.
  - "image_det": COCO detection. `parse` reads JSON {"bbox_2d","label"}; the proposal region
    masks onto the fixed merged patch grid. Reproduces the paper's COCO AUROC 0.902.

This adapter holds the small pure pieces (parse, attn module path, stage-script names) and the
image_det `ext_*` callbacks that feed the shared MTLA driver. The heavy GPU work lives in
mtla/mtla_attn.py; the pipeline is decoupled: generate then a separate HF-eager extract.
"""
from __future__ import annotations

import json
import re

from .base import ModelAdapter, Prediction
from ..registry import register_model

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

_TIME = r'(\d{1,3}(?::\d{2})?(?:\.\d+)?)'


def _to_seconds(s: str) -> float:
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)


# The canonical multi-window timestamp parser. `parse_windows_with_spans` in
# `mtla/models/_qwen3vl_video.py` (extract-time Q_p attribution) shares these exact patterns +
# dedup/ordering, so generation and extraction cannot drift.
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

    multi=True (QVHighlights): all [start,end] windows; multi=False (Charades): the earliest.
    Collects matches across ALL timestamp pattern families, dedups, and orders by start time, so
    a response that mixes formats (e.g. "from 10s to 19s and again [30,42]") keeps every window.
    (Verified to leave the validated QVH/Charades outputs unchanged — the models emit one format
    per response, and overlapping families dedup to the same spans.)
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
            seen.add(key); spans.append((a, b))
    spans.sort(key=lambda ab: ab[0])  # order by start time
    preds = [Prediction([a, b], "") for a, b in spans]
    return preds if multi else preds[:1]


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


@register_model("qwen3vl")
class Qwen3VLAdapter(ModelAdapter):
    model_id = MODEL_ID
    attn_module_path = "transformers.models.qwen3_vl.modeling_qwen3_vl"
    tasks = ("video_span", "image_det")

    def parse(self, response: str, task: str = "video_span", multi: bool = True, **kw) -> list:
        self._check_task(task)
        if task == "image_det":
            return parse_bboxes(response)
        return parse_spans(response, multi=multi)

    def generate_script(self, task: str, engine: str) -> str:
        if task == "image_det" and engine != "vllm":
            raise NotImplementedError("Qwen3-VL COCO generation is vLLM-only (engine: vllm)")
        return "generate.py"

    def extract_script(self, task: str) -> str:
        if task == "image_det":
            return "image_extract.py"
        return "video_extract.py"

    # ---- generation contract (scripts/stages/generate.py); image_det + video_span ----
    def gen_engines(self, task):
        # COCO detection is vLLM-only (that is how the paper preds were produced); video grounding
        # runs under vLLM (fast) or HF (reference).
        return ("vllm",) if task == "image_det" else ("vllm", "hf")

    def vllm_engine_args(self, dataset):
        if dataset.task == "video_span":
            return {"limit_mm_per_prompt": {"video": 1}, "max_model_len": 32768}
        return {"limit_mm_per_prompt": {"image": 1}}

    def vllm_uses_seed(self, task):
        # COCO did NOT seed vLLM (matches the validated paper preds); video DOES (per-rollout draw).
        return task == "video_span"

    def gen_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id)

    def build_vllm_request(self, proc, item, dataset, cfg):
        """(item, dataset) -> {prompt, multi_modal_data, mm_processor_kwargs?} for vLLM. Dispatches
        on the dataset's task: an image payload for detection, a (frames, metadata) video payload
        for grounding — both via `process_vision_info` so the prompt matches the HF/extract path."""
        from qwen_vl_utils import process_vision_info
        if dataset.task == "video_span":
            video_path = dataset.video_path(item, cfg.path("video_dir"))
            import os
            if not os.path.exists(video_path):
                return None
            vcfg = dataset.video
            msgs = [{"role": "user", "content": [
                {"type": "video", "video": f"file://{video_path}",
                 "min_pixels": vcfg["min_pixels"], "max_pixels": vcfg["max_pixels"], "fps": vcfg["fps"]},
                {"type": "text", "text": dataset.prompt(item)}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            _, videos, video_kwargs = process_vision_info(
                msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
            if not videos:
                return None
            # vLLM's Qwen3-VL wants the video as a (frames, metadata) tuple in multi_modal_data.
            return {"prompt": text, "multi_modal_data": {"video": videos[0]},
                    "mm_processor_kwargs": video_kwargs}
        # image_det
        from PIL import Image as _PILImage
        image = _PILImage.open(item["image"]).convert("RGB")
        prompt_text = item["conversations"][0]["value"].replace("<image>\n", "")
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt_text}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
        return {"prompt": text, "multi_modal_data": {"image": image_inputs} if image_inputs else {}}

    def load_hf_gen(self, gpu_id):
        """HF video generation (image_det is vLLM-only). Stock attention, no MTLA hook."""
        return _load_qwen_for_generate(gpu_id)

    def generate_hf(self, ctx, item, dataset, cfg, seed, temperature):
        if dataset.task != "video_span":
            raise NotImplementedError("Qwen3-VL image_det generation is vLLM-only (engine: vllm)")
        video_path = dataset.video_path(item, cfg.path("video_dir"))
        import os
        if not os.path.exists(video_path):
            return None, False
        resp = _generate_video(ctx, video_path, dataset.prompt(item), dataset.video,
                               seed=seed, temperature=temperature,
                               max_new_tokens=dataset.gen_max_new_tokens, top_p=dataset.gen_top_p)
        return (resp, False) if resp is not None else (None, False)

    # ---- HF-eager extraction ----
    # The per-item flow (extract_one) is shared in the base class; this adapter supplies the
    # Qwen-specific pieces via the `ext_*` callbacks. The same flow serves image detection (boxes)
    # and video grounding (time-span windows); each `ext_*` dispatches on `self._task`, which the
    # base `extract_one` records from `ctx["task"]` (set by `load_for_extract`).
    def load_for_extract(self, gpu_id, task="image_det"):
        self._task = task
        ctx = _load_qwen_for_extract(gpu_id, task)
        ctx["task"] = task
        return ctx

    # ---- ext_* dispatch (image_det <-> video_span); extract_one is inherited from the base ----
    def ext_build_inputs(self, p, ds_by_id, ctx, rank):
        if ctx.get("task") == "video_span":
            return self._vid_build_inputs(p, ds_by_id, ctx, rank)
        return self._img_build_inputs(p, ds_by_id, ctx, rank)

    def ext_token_ranges(self, response, predictions, tokenizer):
        if getattr(self, "_task", "image_det") == "video_span":
            return self._vid_token_ranges(response, predictions, tokenizer)
        return find_pred_token_ranges(response, predictions, tokenizer)

    def ext_region_mask(self, prediction, meta):
        if meta.get("task") == "video_span":
            return self._vid_region_mask(prediction, meta)
        return _bbox_to_patch_indices(prediction["box"], meta["grid_h"], meta["grid_w"])[0]

    def ext_forward_kwargs(self, full_ids, total_len, device, inp):
        # Identical for image and video: forward the cached vision inputs + a full attention mask,
        # padding mm_token_type_ids to the response length. (video adds pixel_values_videos /
        # video_grid_thw, which are simply whatever keys `inp["inputs"]` carries.)
        import torch
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

    def ext_obj_record(self, prediction, pred_idx, meta):
        if meta.get("task") == "video_span":
            return {"pred_idx": pred_idx, "window": list(prediction)}
        return {"pred_idx": pred_idx, "label": prediction["label"], "box": prediction["box"],
                "grid_h": int(meta["grid_h"]), "grid_w": int(meta["grid_w"])}

    def ext_record(self, p, meta, objects, n_predictions):
        if meta.get("task") == "video_span":
            return {"qid": p.get("qid"), "video": p.get("video") or p.get("vid"),
                    "query": p.get("query") or p.get("caption"),
                    "gt_windows": meta["gt_windows"],
                    "duration_s": meta["duration_s"], "T_tokens": meta["T_tokens"],
                    "n_pred_windows": n_predictions, "n_extracted": len(objects), "objects": objects}
        return {"image_id": p["id"], "n_pred_bboxes": n_predictions, "n_extracted": len(objects),
                "objects": objects, "grid_hw": (int(meta["grid_h"]), int(meta["grid_w"]))}

    # ---- video_span ext_* (Qwen3-VL temporal grounding) ----
    def _vid_build_inputs(self, p, ds_by_id, ctx, rank):
        return _vid_build_inputs(p, ds_by_id, ctx, rank)

    def _vid_token_ranges(self, response, predictions, tokenizer):
        # Q_p per stored predicted window, aligned index-for-index with `predictions` (keys off the
        # stored windows, not an independent re-parse, so token_ranges[i] <-> predictions[i]).
        from ._qwen3vl_video import perwindow_qp_tokens
        return perwindow_qp_tokens(response, predictions, tokenizer)

    def _vid_region_mask(self, prediction, meta):
        from ._qwen3vl_video import span_to_frame_token_indices
        return span_to_frame_token_indices(prediction, meta["duration_s"], meta["T_tokens"],
                                           meta["H_tokens"], meta["W_tokens"])

    def _img_build_inputs(self, p, ds_by_id, ctx, rank):
        """Build the Qwen detection prompt via the processor, derive the merged patch grid from
        image_grid_thw, locate the <|image_pad|> tokens, and flag hallucinations against the GT.
        Returns None to skip."""
        import json
        from .base import label_hallu_bbox
        proc = ctx["proc"]; device = ctx["device"]; img_pad_id = ctx["img_pad_id"]
        if p.get("status") != "success" or not p.get("pred_bboxes"):
            return None
        ds_item = ds_by_id.get(p["id"])
        if ds_item is None:
            return None
        prompt_text = ds_item["conversations"][0]["value"].replace("<image>\n", "")
        try:
            img = _Image.open(ds_item["image"]).convert("RGB")
        except Exception as e:
            print(f"[worker {rank}] skip {p['id']}: img {e}", flush=True)
            return None
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt_text}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0]
        if "image_grid_thw" not in inputs:
            return None
        t_, h_, w_ = inputs["image_grid_thw"][0].tolist()
        grid_h, grid_w = h_ // 2, w_ // 2
        prompt_cpu = prompt_ids.cpu().tolist()
        image_idx_l = [k for k, t in enumerate(prompt_cpu) if t == img_pad_id]
        if len(image_idx_l) != grid_h * grid_w:
            print(f"[worker {rank}] WARN {p['id']}: image_idx_l={len(image_idx_l)} != "
                  f"grid_h*grid_w={grid_h * grid_w}", flush=True)
            return None
        gt_objs = json.loads(ds_item["conversations"][1]["value"])
        hallu_flags = [label_hallu_bbox(pb["box"], pb["label"], gt_objs) for pb in p["pred_bboxes"]]
        return {"prompt_ids": prompt_ids, "response": p["response"], "modality_idx_l": image_idx_l,
                "predictions": p["pred_bboxes"], "hallu_flags": hallu_flags,
                "inputs": inputs, "meta": {"task": "image_det", "grid_h": grid_h, "grid_w": grid_w}}


# ============================================================================
# HF-eager MTLA extraction implementation for Qwen3-VL.
# Supplies the Qwen-specific pieces of mtla.mtla_attn.compute_mtla.
# ============================================================================
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image as _Image  # noqa: E402
from ..mtla_attn import MTLAState, make_mtla_attention_forward, install  # noqa: E402


def _bbox_to_patch_indices(bbox, grid_h, grid_w):
    """Modality-token indices inside `bbox` on Qwen's fixed grid_h x grid_w merged patch grid.
    Returns (inside, outside); the caller uses only `inside`."""
    x1, y1, x2, y2 = bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    col_min = int(np.floor(x1 * grid_w / 1000.0))
    col_max = int(np.floor((x2 - 1e-6) * grid_w / 1000.0))
    row_min = int(np.floor(y1 * grid_h / 1000.0))
    row_max = int(np.floor((y2 - 1e-6) * grid_h / 1000.0))
    col_min = max(0, min(grid_w - 1, col_min)); col_max = max(0, min(grid_w - 1, col_max))
    row_min = max(0, min(grid_h - 1, row_min)); row_max = max(0, min(grid_h - 1, row_max))
    inside = []
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            inside.append(r * grid_w + c)
    inside_set = set(inside)
    outside = [i for i in range(grid_h * grid_w) if i not in inside_set]
    return inside, outside


def find_pred_token_ranges(response_text, pred_bboxes, tokenizer):
    enc = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out = []
    search_pos = 0
    full_pat_template = (
        r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,'
        r'\s*"label"\s*:\s*"{label}"'
    )
    for pb in pred_bboxes:
        label = pb["label"]
        full_pat = re.compile(full_pat_template.format(label=re.escape(label)))
        m = full_pat.search(response_text, search_pos)
        if not m:
            m = full_pat.search(response_text)
        coord_ranges = []
        if m:
            coord_ranges = [(m.start(g), m.end(g)) for g in range(1, 5)]
            label_start = m.end() - 1 - len(label)
            label_end = m.end() - 1
            search_pos = m.end()
        else:
            label_pat = re.compile(r'"label"\s*:\s*"' + re.escape(label) + r'"')
            ml = label_pat.search(response_text, search_pos)
            if not ml:
                ml = label_pat.search(response_text)
                if not ml:
                    out.append(None); continue
            marker = re.search(r'"label"\s*:\s*"', ml.group(0)).group(0)
            label_start = ml.start() + len(marker)
            label_end = label_start + len(label)
            search_pos = ml.end()
        label_toks = []
        for ti, (ts, te) in enumerate(offsets):
            if ts < label_end and te > label_start:
                label_toks.append(ti)
        first_label_tok = None
        for ti in label_toks:
            ts, te = offsets[ti]
            if ts >= label_start:
                first_label_tok = ti; break
        if first_label_tok is None and label_toks:
            first_label_tok = label_toks[0]
        coord_toks = []
        for (cs, ce) in coord_ranges:
            for ti, (ts, te) in enumerate(offsets):
                if ts < ce and te > cs:
                    coord_toks.append(ti)
        out.append({"first_label_tok": first_label_tok, "label_toks": label_toks,
                    "coord_toks": coord_toks})
    return out


def _load_qwen_for_extract(gpu_id, task="image_det"):
    """Load Qwen3-VL + processor on `gpu_id`, install the MTLA attention forward, return the ctx.
    Same model/processor for both tasks; `task` only selects which modality pad-id the driver
    locates (`<|image_pad|>` for image, `<|video_pad|>` for video)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    state = MTLAState()
    install("transformers.models.qwen3_vl.modeling_qwen3_vl",
            make_mtla_attention_forward(state))

    device = f"cuda:{gpu_id}"
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager",
        device_map=device).eval()
    decoder_layers = model.model.language_model.layers
    state.lang_attn_ids = {id(L.self_attn) for L in decoder_layers}
    state.lang_attn_order = [id(L.self_attn) for L in decoder_layers]

    pad_tok = "<|video_pad|>" if task == "video_span" else "<|image_pad|>"
    return {"model": model, "proc": proc, "tokenizer": proc.tokenizer, "state": state,
            "device": device, "n_layers": len(decoder_layers),
            "n_heads": model.config.text_config.num_attention_heads,
            "img_pad_id": proc.tokenizer.convert_tokens_to_ids(pad_tok),
            "video_pad_id": proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")}


def _load_qwen_for_generate(gpu_id):
    """Load Qwen3-VL + processor for generation. Stock attention (no MTLA hook) — generation
    needs no attention readout, so this stays fast and faithful to the model's default forward."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    device = f"cuda:{gpu_id}"
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    return {"model": model, "proc": proc, "tokenizer": proc.tokenizer, "device": device}


def _generate_video(ctx, video_path, query, vcfg, seed=0, temperature=0.7,
                    max_new_tokens=128, top_p=0.95):
    """Run video preprocessing + model.generate for one clip; return the decoded response (or None
    to skip). Deterministic preprocessing (fps/pixels from the dataset's `video` cfg). Samples at
    `temperature` (the sharded worker pre-seeds the RNG per (seed, rank)); temperature==0 -> greedy.
    The vLLM counterpart of this path is `Qwen3VLAdapter.build_vllm_request` + the shared strategy."""
    from qwen_vl_utils import process_vision_info
    proc = ctx["proc"]; model = ctx["model"]; device = ctx["device"]
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video_path}",
         "min_pixels": vcfg["min_pixels"], "max_pixels": vcfg["max_pixels"], "fps": vcfg["fps"]},
        {"type": "text", "text": query},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        images, videos, video_kwargs = process_vision_info(
            msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = proc(text=text, images=images, videos=videos, video_metadata=video_metadatas,
                      do_resize=False, return_tensors="pt", **video_kwargs).to(device)
    except Exception as e:
        print(f"[generate] skip {video_path}: processor {e}", flush=True)
        return None
    prompt_len = inputs["input_ids"].shape[1]
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    try:
        with torch.no_grad():
            gen_ids = model.generate(**inputs, **gen_kwargs)
    except Exception as e:
        print(f"[generate] skip {video_path}: gen {e}", flush=True)
        torch.cuda.empty_cache()
        return None
    return proc.tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True).strip()


def _vid_build_inputs(p, ds_by_id, ctx, rank):
    """Build the Qwen video grounding input for one prediction record, locate the <|video_pad|>
    frame tokens, derive the frame-token grid (T,H,W), and flag each predicted window against the
    GT. Mirrors _img_build_inputs but for time-span windows. Returns None to skip.

    `p` is a prediction record from the generate stage. The dataset adapter normalizes it (it owns
    its own record schema) into {video_path, query, pred_windows, gt_windows} via `video_item`;
    `ds_by_id` is unused for video (predictions are self-contained)."""
    import os
    from qwen_vl_utils import process_vision_info
    from ._qwen3vl_video import hallu_flags_windows
    proc = ctx["proc"]; device = ctx["device"]; video_pad_id = ctx["video_pad_id"]
    dataset = ctx["dataset"]; vcfg = dataset.video

    response = p.get("response")
    if not response:
        return None
    vi = dataset.video_item(p, ctx["video_dir"])
    pred_windows = [list(w) for w in vi["pred_windows"]]
    if not pred_windows:
        return None
    gt_windows = [list(w) for w in vi["gt_windows"]]
    video_path = vi["video_path"]
    if not os.path.exists(video_path):
        return None
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video_path}",
         "min_pixels": vcfg["min_pixels"], "max_pixels": vcfg["max_pixels"], "fps": vcfg["fps"]},
        {"type": "text", "text": vi["query"]},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        images, videos, video_kwargs = process_vision_info(
            msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = proc(text=text, images=images, videos=videos, video_metadata=video_metadatas,
                      do_resize=False, return_tensors="pt", **video_kwargs).to(device)
    except Exception as e:
        print(f"[worker {rank}] skip {vi['video_path']}: processor {e}", flush=True)
        return None
    prompt_ids = inputs["input_ids"][0]
    vgthw = inputs.get("video_grid_thw")
    if vgthw is None or vgthw.shape[0] != 1:
        return None
    T_grid, H_grid, W_grid = (int(vgthw[0, i].item()) for i in range(3))
    sms = getattr(ctx["model"].config.vision_config, "spatial_merge_size", 2)
    T_tokens, H_tokens, W_tokens = T_grid, H_grid // sms, W_grid // sms
    n_video_expected = T_tokens * H_tokens * W_tokens

    prompt_cpu = prompt_ids.cpu().tolist()
    video_idx_l = [k for k, t in enumerate(prompt_cpu) if t == video_pad_id]
    if not video_idx_l or len(video_idx_l) != n_video_expected:
        print(f"[worker {rank}] skip {vi['video_path']}: video tokens {len(video_idx_l)} != "
              f"T*H*W={n_video_expected}", flush=True)
        return None

    duration_s = float(p.get("duration_s") or get_video_duration(video_path))
    hallu_flags = hallu_flags_windows(pred_windows, gt_windows, vcfg["multi"])
    return {"prompt_ids": prompt_ids, "response": response, "modality_idx_l": video_idx_l,
            "predictions": pred_windows, "hallu_flags": hallu_flags, "inputs": inputs,
            "meta": {"task": "video_span", "duration_s": duration_s, "T_tokens": T_tokens,
                     "H_tokens": H_tokens, "W_tokens": W_tokens, "gt_windows": gt_windows}}


def get_video_duration(video_path):
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    return len(vr) / fps if fps > 0 else 0.0
