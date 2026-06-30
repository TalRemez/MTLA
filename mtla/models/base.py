"""Model adapter interface.

A model adapter holds everything that depends on the *model family* (Qwen3-VL, InternVL, ...):
the output parser, which transformer module to monkeypatch for attention, which GPU stage scripts
implement generate/extract, and the few model-specific pieces of the MTLA computation.

Everything model-*agnostic* (the MTLA kernel, layer-band reduction, voting, AUROC) stays in the
core `mtla` package; everything *dataset*-specific (prompt text, class list, metric) lives in
`mtla.data`.

A `task` is a coarse modality family, not a benchmark:
  - "image_det"  : image object detection (COCO; InternVL or Qwen3-VL)
  - "video_span" : temporal grounding (QVHighlights, Charades; Qwen3-VL)
Using a family (not a benchmark) is what lets one dataset run on multiple models.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prediction:
    """One parsed grounding prediction.

    region: bounding box [x1,y1,x2,y2] in [0,1000] (image) or [t_start,t_end] in seconds
            (video/audio); label: the predicted class string.
    """
    region: list
    label: str


class ModelAdapter:
    """Base class for model-family adapters."""

    model_id: str = ""
    # dotted path to the module whose `eager_attention_forward` is monkeypatched, e.g.
    # "transformers.models.qwen3.modeling_qwen3" (InternVL LLM) or "...qwen3_vl.modeling_qwen3_vl".
    attn_module_path: str = ""
    # task families this model supports, e.g. ("image_det",) or ("video_span",).
    tasks: tuple = ()

    # ---- pure, CPU-testable ----
    def parse(self, response: str, task: str = None, **kw) -> list:
        """Parse a raw model response into a list of `Prediction`."""
        raise NotImplementedError

    # ---- which GPU stage scripts implement this model's stages ----
    def generate_script(self, task: str, engine: str) -> str:
        """Filename in mtla/stages/ for the generate stage (depends on engine: hf|vllm)."""
        raise NotImplementedError(f"{type(self).__name__} has no generate for task {task}")

    def extract_script(self, task: str) -> str:
        """Filename in mtla/stages/ for the (always HF-eager) extract stage."""
        raise NotImplementedError(f"{type(self).__name__} has no extract for task {task}")

    # ---- HF-eager MTLA extraction (driven by mtla/stages/image_extract.py) ----
    def load_for_extract(self, gpu_id: int) -> dict:
        """Load model + processor on `gpu_id`, install the MTLA attention forward, and return a
        ctx dict (model, tokenizer, MTLAState, device, n_layers, n_heads, img_pad_id) the shared
        driver passes back to `extract_one`."""
        raise NotImplementedError(f"{type(self).__name__} has no load_for_extract")

    def extract_one(self, pred_record: dict, ds_by_id: dict, ctx: dict, svar_shift: bool, rank: int = 0):
        """Compute MTLA for one image's predictions and return its saved .pt record (or None to
        skip). Image-det adapters delegate to `mtla.mtla_attn.compute_image_mtla`, which drives the
        shared per-image MTLA computation and calls back the `ext_*` methods below."""
        raise NotImplementedError(f"{type(self).__name__} has no extract_one")

    # ---- image_det model-specific callbacks invoked by mtla.mtla_attn.compute_image_mtla ----
    # Each returns plain data; the shared driver owns the common flow (valid-pred filtering,
    # query-position dedup, pred_specs, forward, buffer->record). Override these to add a model.
    def ext_build_inputs(self, p, ds_item, ctx, rank):
        """Preprocess image + build prompt. Return dict with keys: prompt_ids (1-D tensor),
        response (str), image_idx_l (modality-token positions), meta (geometry), plus any keys
        ext_forward_kwargs needs (e.g. pixel_values / inputs). Return None to skip."""
        raise NotImplementedError

    def ext_token_ranges(self, response, pred_bboxes, tokenizer):
        """Per-prediction {first_label_tok, label_toks, coord_toks} (the tokens Q_p) or None."""
        raise NotImplementedError

    def ext_region_mask(self, box, meta):
        """Modality-token indices inside the proposal box M(R_p)."""
        raise NotImplementedError

    def ext_forward_kwargs(self, full_ids, total_len, device, inp):
        """kwargs dict for the patched `model(**fk)` forward."""
        raise NotImplementedError

    def ext_obj_extras(self, meta) -> dict:
        """Per-object record fields beyond the shared ones (e.g. tile_grid / grid_h)."""
        return {}

    def ext_rec_extras(self, meta) -> dict:
        """Top-level record fields beyond the shared ones."""
        return {}
