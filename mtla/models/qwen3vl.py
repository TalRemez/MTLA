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
            return "image_extract.py"
        raise NotImplementedError("video_span extraction is dataset-driven")

    # ---- HF-eager extraction (image_det) ----
    # Shares the per-image flow with InternVL via mtla.extract.compute_image_mtla; this adapter
    # supplies only the Qwen-specific pieces via the `ext_*` callbacks below.
    def load_for_extract(self, gpu_id):
        return _load_qwen_det_for_extract(gpu_id)

    def extract_one(self, p, ds_by_id, ctx, svar_shift, rank=0):
        from ..extract import compute_image_mtla
        return compute_image_mtla(self, p, ds_by_id, ctx, svar_shift, VARIANTS_DET, rank)

    def ext_build_inputs(self, p, ds_item, ctx, rank):
        """Build the Qwen detection prompt via the processor, derive the merged patch grid from
        image_grid_thw, and locate the <|image_pad|> tokens. Returns None to skip."""
        proc = ctx["proc"]; device = ctx["device"]; img_pad_id = ctx["img_pad_id"]
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
        return {"prompt_ids": prompt_ids, "response": p["response"], "image_idx_l": image_idx_l,
                "inputs": inputs, "meta": {"grid_h": grid_h, "grid_w": grid_w}}

    def ext_token_ranges(self, response, pred_bboxes, tokenizer):
        return find_pred_token_ranges(response, pred_bboxes, tokenizer)

    def ext_classify_keys(self, prompt_cpu, image_idx_l, specials_ids, tokenizer, prompt_len, total_len):
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        user_id = tokenizer.convert_tokens_to_ids("user")
        assistant_id = tokenizer.convert_tokens_to_ids("assistant")
        user_turn_start = 0
        for k in range(len(prompt_cpu) - 1):
            if prompt_cpu[k] == im_start_id and prompt_cpu[k + 1] == user_id:
                user_turn_start = k; break
        assistant_start = None
        for k in range(len(prompt_cpu) - 1):
            if prompt_cpu[k] == im_start_id and prompt_cpu[k + 1] == assistant_id:
                assistant_start = k; break
        img_pad_id = image_idx_l[0] if image_idx_l else None
        image_set = set(image_idx_l)
        system = []; task = []; spc = []
        for k, tid in enumerate(prompt_cpu):
            if k in image_set:
                continue
            elif tid in specials_ids:
                spc.append(k)
            elif assistant_start is not None and k >= assistant_start:
                spc.append(k)
            elif k >= user_turn_start:
                task.append(k)
            else:
                system.append(k)
        resp = list(range(prompt_len, total_len))
        return {"system": system, "task": task, "specials": spc, "response": resp,
                "user_turn_start": user_turn_start, "assistant_start": assistant_start}

    def ext_region_mask(self, box, meta):
        return _bbox_to_patch_indices(box, meta["grid_h"], meta["grid_w"])

    def ext_mentions(self, label, tokenizer, prompt_cpu, user_turn_start, end_pos):
        return find_label_token_positions(label, tokenizer, prompt_cpu, user_turn_start, end_pos)

    def ext_forward_kwargs(self, full_ids, total_len, device, inp):
        import torch
        inputs = inp["inputs"]
        fk = {"input_ids": full_ids,
              "attention_mask": torch.ones(1, total_len, device=device, dtype=torch.long)}
        for k in ["pixel_values", "image_grid_thw"]:
            if k in inputs:
                fk[k] = inputs[k]
        if "mm_token_type_ids" in inputs:
            orig = inputs["mm_token_type_ids"]
            extra = total_len - orig.shape[1]
            fk["mm_token_type_ids"] = (torch.cat(
                [orig, torch.zeros(1, extra, dtype=orig.dtype, device=orig.device)], dim=1)
                if extra > 0 else orig)
        return fk

    def ext_obj_extras(self, meta, tr):
        return {"char_start": tr["char_start"], "grid_h": int(meta["grid_h"]),
                "grid_w": int(meta["grid_w"])}

    def ext_rec_extras(self, meta):
        return {"grid_hw": (int(meta["grid_h"]), int(meta["grid_w"]))}


# ============================================================================
# HF-eager extraction implementation (image_det), lifted verbatim from the
# validated qwen3vl_det_extract.py and wired to the shared hook in mtla.extract.
# New code emits 4 variants (adds first_digit, unused by Qwen scoring) so the
# image_extract driver is uniform across models; the first/label/coord blocks
# are byte-identical to the 3-variant baseline.
# ============================================================================
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image as _Image  # noqa: E402
from ..extract import MTLAState, make_mtla_attention_forward, install  # noqa: E402

VARIANTS_DET = ("first", "label", "coord", "first_digit")


def _bbox_to_patch_indices(bbox, grid_h, grid_w):
    """Verbatim from qwen3vl_det_extract.py (swaps reversed coords; differs from
    mtla.mask.bbox_to_patch_indices on degenerate boxes — kept for byte-parity)."""
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
                    "coord_toks": coord_toks, "char_start": label_start})
    return out


def find_token_subseq_positions(needle_ids, haystack_ids):
    n_n = len(needle_ids); n_h = len(haystack_ids)
    out = []; i = 0
    while i + n_n <= n_h:
        if haystack_ids[i:i + n_n] == needle_ids:
            out.append(i); i += n_n
        else:
            i += 1
    return out


def find_label_token_positions(label, tokenizer, prompt_token_ids, user_turn_start, end_pos):
    haystack = list(prompt_token_ids)
    candidates = [
        tokenizer(" " + label, add_special_tokens=False)["input_ids"],
        tokenizer(label, add_special_tokens=False)["input_ids"],
    ]
    seen = set(); out = []
    for cand in candidates:
        if not cand:
            continue
        for s in find_token_subseq_positions(cand, haystack):
            if s < user_turn_start or s + len(cand) > end_pos:
                continue
            for k in range(len(cand)):
                pos = s + k
                if pos not in seen:
                    seen.add(pos); out.append(pos)
    return sorted(out)


def _load_qwen_det_for_extract(gpu_id):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_mod

    state = MTLAState()
    install("transformers.models.qwen3_vl.modeling_qwen3_vl",
            make_mtla_attention_forward(state, qwen3_mod.repeat_kv, VARIANTS_DET))

    device = f"cuda:{gpu_id}"
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager",
        device_map=device).eval()
    decoder_layers = model.model.language_model.layers
    state.lang_attn_ids = {id(L.self_attn) for L in decoder_layers}
    state.lang_attn_order = [id(L.self_attn) for L in decoder_layers]

    img_pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    special_token_strings = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                             "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
                             "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|endoftext|>"]
    specials_ids = set()
    for s in special_token_strings:
        tid = proc.tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0:
            specials_ids.add(tid)
    for tid in (proc.tokenizer.all_special_ids or []):
        if tid != img_pad_id:
            specials_ids.add(int(tid))

    return {"model": model, "proc": proc, "tokenizer": proc.tokenizer, "state": state,
            "device": device, "n_layers": len(decoder_layers),
            "n_heads": model.config.text_config.num_attention_heads,
            "img_pad_id": img_pad_id, "specials_ids": specials_ids}
