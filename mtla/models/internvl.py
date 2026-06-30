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

from .base import ModelAdapter, Prediction
from ..mask import bbox_to_internvl_token_indices

MODEL_ID = "OpenGVLab/InternVL3_5-8B"
IMG_CONTEXT_TOK = "<IMG_CONTEXT>"
NUM_IMAGE_TOKEN_PER_TILE = 16 * 16  # 256, InternVL per-tile patches after pixel-shuffle 0.5

PROMPT_TMPL = (
    "Please detect all instances of {cats} in the image. "
    "Output the bounding boxes in the format <ref>category</ref><box>[[x1, y1, x2, y2], ...]</box> "
    "with coordinates normalized to [0, 1000]."
)


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

    # ---- stage scripts ----
    def generate_script(self, task: str, engine: str) -> str:
        return {"vllm": "internvl_generate.py", "hf": "internvl_generate_hf.py"}[engine]

    def extract_script(self, task: str) -> str:
        return "image_extract.py"

    # ---- HF-eager extraction (image_det) ----
    # The per-image flow is shared with Qwen in mtla.mtla_attn.compute_image_mtla; this adapter
    # supplies only the InternVL-specific pieces via the `ext_*` callbacks below.
    def load_for_extract(self, gpu_id):
        return _load_internvl_for_extract(gpu_id)

    def extract_one(self, p, ds_by_id, ctx, svar_shift, rank=0):
        from ..mtla_attn import compute_image_mtla
        return compute_image_mtla(self, p, ds_by_id, ctx, svar_shift, rank)

    def ext_build_inputs(self, p, ds_item, ctx, rank):
        """Tile the image, build the InternVL grounding prompt, re-insert the <ref>/<box>
        tags vLLM stripped, and locate the <IMG_CONTEXT> tokens. Returns None to skip."""
        import torch
        tokenizer = ctx["tokenizer"]; device = ctx["device"]; img_pad_id = ctx["img_pad_id"]
        try:
            pixel_values, tile_grid, has_thumb = load_image_internvl(ds_item["image"])
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"[worker {rank}] skip {p['id']}: image load {e}", flush=True)
            return None
        n_tiles = tile_grid[0] * tile_grid[1]
        n_image_tokens = n_tiles * NUM_IMAGE_TOKEN_PER_TILE + (NUM_IMAGE_TOKEN_PER_TILE if has_thumb else 0)
        cats = ", ".join(p["categories"])
        user_text = "<image>\n" + PROMPT_TMPL.format(cats=cats)
        chat = tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                             tokenize=False, add_generation_prompt=True)
        image_token_block = "<img>" + (IMG_CONTEXT_TOK * n_image_tokens) + "</img>"
        prompt_text = chat.replace("<image>", image_token_block, 1)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
        # RAW-RESPONSE FIX: vLLM stripped the <ref>/<box> tags; reconstruct so re-tokenization aligns.
        response = re.sub(r'([A-Za-z][A-Za-z _]*?)(\[\[.+?\]\])',
                          lambda m: f'<ref>{m.group(1)}</ref><box>{m.group(2)}</box>',
                          p["response"], flags=re.DOTALL)
        prompt_cpu = prompt_ids.cpu().tolist()
        image_idx_l = [i for i, t in enumerate(prompt_cpu) if t == img_pad_id]
        if len(image_idx_l) != n_image_tokens:
            return None
        return {"prompt_ids": prompt_ids, "response": response, "image_idx_l": image_idx_l,
                "pixel_values": pixel_values,
                "meta": {"tile_grid": tile_grid, "has_thumb": has_thumb, "n_image_tokens": n_image_tokens}}

    def ext_token_ranges(self, response, pred_bboxes, tokenizer):
        return find_pred_token_ranges(response, pred_bboxes, tokenizer)

    def ext_region_mask(self, box, meta):
        """Modality-token indices inside the proposal box M(R_p) (InternVL dynamic tiling)."""
        return bbox_to_internvl_token_indices(box, meta["tile_grid"], meta["has_thumb"])[0]

    def ext_forward_kwargs(self, full_ids, total_len, device, inp):
        import torch
        pv = inp["pixel_values"]
        return {"input_ids": full_ids, "pixel_values": pv,
                "image_flags": torch.ones(pv.shape[0], dtype=torch.long, device=device),
                "attention_mask": torch.ones(1, total_len, device=device, dtype=torch.long)}

    def ext_obj_extras(self, meta):
        return {"tile_grid": meta["tile_grid"], "has_thumb": meta["has_thumb"]}

    def ext_rec_extras(self, meta):
        return {"tile_grid": meta["tile_grid"], "has_thumb": meta["has_thumb"]}


# ============================================================================
# HF-eager MTLA extraction implementation (image_det) for InternVL3.5.
# Supplies the InternVL-specific pieces of mtla.mtla_attn.compute_image_mtla.
# ============================================================================
import torch  # noqa: E402
import torchvision.transforms as _T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode as _Interp  # noqa: E402
from PIL import Image as _Image  # noqa: E402
from ..mtla_attn import MTLAState, make_mtla_attention_forward, install  # noqa: E402

_IMAGENET_MEAN = (0.485, 0.456, 0.406); _IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size=448):
    return _T.Compose([
        _T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        _T.Resize((input_size, input_size), interpolation=_Interp.BICUBIC),
        _T.ToTensor(), _T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size):
    best = float('inf'); best_ar = (1, 1); area = w * h
    for r in target_ratios:
        rar = r[0] / r[1]
        diff = abs(aspect_ratio - rar)
        if diff < best:
            best = diff; best_ar = r
        elif diff == best and area > 0.5 * image_size * image_size * r[0] * r[1]:
            best_ar = r
    return best_ar


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    ow, oh = image.size; ar = ow / oh
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                            for j in range(1, n + 1) if min_num <= i * j <= max_num},
                           key=lambda x: x[0] * x[1])
    target = _find_closest_aspect_ratio(ar, target_ratios, ow, oh, image_size)
    n_cols, n_rows = target
    tw, th = image_size * n_cols, image_size * n_rows
    blocks = n_cols * n_rows
    img = image.resize((tw, th))
    images = []
    for i in range(blocks):
        c = i % n_cols; r = i // n_cols
        images.append(img.crop((c * image_size, r * image_size,
                                (c + 1) * image_size, (r + 1) * image_size)))
    has_thumb = use_thumbnail and len(images) != 1
    if has_thumb:
        images.append(image.resize((image_size, image_size)))
    return images, (n_cols, n_rows), has_thumb


def load_image_internvl(path, input_size=448, max_num=12):
    image = _Image.open(path).convert('RGB')
    transform = _build_transform(input_size=input_size)
    tiles, tile_grid, has_thumb = _dynamic_preprocess(image, image_size=input_size,
                                                      use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in tiles])
    return pixel_values, tile_grid, has_thumb


def find_pred_token_ranges(response_text, pred_bboxes, tokenizer):
    enc = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out = []
    for pb in pred_bboxes:
        label = pb["label"]; box = pb["box"]
        bbox_pat = re.compile(rf'\[\s*{box[0]}\s*,\s*{box[1]}\s*,\s*{box[2]}\s*,\s*{box[3]}\s*\]')
        m = bbox_pat.search(response_text)
        if not m:
            out.append(None); continue
        coord_toks = []
        for num_m in re.finditer(r'\d+', m.group(0)):
            num_start = m.start() + num_m.start(); num_end = m.start() + num_m.end()
            for ti, (ts, te) in enumerate(offsets):
                if ts < num_end and te > num_start:
                    coord_toks.append(ti)
        label_block_pat = re.compile(
            rf'(?:^|<ref>|[\s\]\)\.])({re.escape(label)})(?:</ref>\s*<box>)?\s*\[\[',
            flags=re.IGNORECASE)
        label_pos = None
        for lm in label_block_pat.finditer(response_text):
            if lm.start() <= m.start():
                label_pos = lm
            else:
                break
        if label_pos is None:
            simple_pat = re.compile(rf'\b{re.escape(label)}\b', flags=re.IGNORECASE)
            cand = None
            for lm in simple_pat.finditer(response_text):
                if lm.start() <= m.start():
                    cand = lm
                else:
                    break
            label_pos = cand
        if label_pos is None:
            out.append(None); continue
        label_match = re.search(re.escape(label), label_pos.group(0), flags=re.IGNORECASE)
        if label_match is None:
            out.append(None); continue
        label_start = label_pos.start() + label_match.start()
        label_end = label_pos.start() + label_match.end()
        label_toks = [ti for ti, (ts, te) in enumerate(offsets) if ts < label_end and te > label_start]
        if not label_toks:
            out.append(None); continue
        out.append({"first_label_tok": label_toks[0], "label_toks": label_toks, "coord_toks": coord_toks})
    return out


def _load_internvl_for_extract(gpu_id):
    from transformers import AutoModel, AutoTokenizer, modeling_utils
    import transformers.models.qwen3.modeling_qwen3 as q3_mod
    # transformers v5 compat shim (InternVL remote code expects an attr v5 dropped)
    if not hasattr(modeling_utils.PreTrainedModel, '_compat_done'):
        _orig = modeling_utils.PreTrainedModel.__getattr__
        def _patched(self, name):
            if name == 'all_tied_weights_keys':
                try:
                    return _orig(self, name)
                except AttributeError:
                    return {}
            return _orig(self, name)
        modeling_utils.PreTrainedModel.__getattr__ = _patched
        modeling_utils.PreTrainedModel._compat_done = True

    state = MTLAState()
    install("transformers.models.qwen3.modeling_qwen3",
            make_mtla_attention_forward(state, q3_mod.repeat_kv))

    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                      trust_remote_code=True).to(device).eval()
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOK)
    for layer in model.language_model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    decoder_layers = model.language_model.model.layers
    state.lang_attn_ids = {id(L.self_attn) for L in decoder_layers}
    state.lang_attn_order = [id(L.self_attn) for L in decoder_layers]

    return {"model": model, "tokenizer": tokenizer, "state": state, "device": device,
            "n_layers": len(decoder_layers),
            "n_heads": model.language_model.config.num_attention_heads,
            "img_pad_id": tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOK)}


