"""Model adapter interface.

A model adapter holds everything that depends on the *model family* (Qwen3-VL, InternVL, ...):
the output parser, the region mask, which transformer module to monkeypatch for attention, which
GPU stage scripts implement generate/extract, and — crucially for keeping dataset adapters
model-agnostic — how the model's saved attention record encodes the MTLA and SVAR signals.

Everything model-*agnostic* (layer-band reduction, voting, AUROC) stays in the core `mtla`
package; everything *dataset*-specific (prompt text, class list, metric) lives in `mtla.data`.

A `task` is a coarse modality family, not a benchmark:
  - "image_det"  : image object detection (COCO; InternVL or Qwen3-VL)
  - "video_span" : temporal grounding (QVHighlights, Charades; Qwen3-VL)
Using a family (not a benchmark) is what lets one dataset run on multiple models.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Prediction:
    """One parsed grounding prediction.

    region: bounding box [x1,y1,x2,y2] in [0,1000] (image) or [t_start,t_end] in seconds
            (video/audio); label: the predicted class string.
    """
    region: list
    label: str


@dataclass
class SlotSpec:
    """How to read one signal (MTLA or SVAR) out of a saved attention record.

    A record stores, per token-aggregation "block", a dict of [L,H] arrays keyed by `stat`
    (e.g. "image_inside_sum", "image_sum"). To compute a signal we reduce one such [L,H] array.

    block: the record key for the token aggregation, e.g. "attn_coord_mean".
    stat:  the array key within that block, e.g. "image_inside_sum" (MTLA) / "image_sum" (SVAR).
    combine: if set, names a recipe that combines multiple blocks instead of reading one:
             "all" = count-weighted mean of `parts` blocks (label+coord) — the canonical MTLA.
    parts: for combine="all", list of (block, count_field) pairs to weight, e.g.
           [("attn_label_mean","n_label_toks"), ("attn_coord_mean","n_coord_toks")].
    """
    stat: str
    block: str | None = None
    combine: str | None = None
    parts: list = field(default_factory=list)


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

    def region_mask(self, region, meta: dict):
        """Return (inside_idx, outside_idx) into the modality tokens for `region` (via mtla.mask)."""
        raise NotImplementedError

    # ---- how this model's records encode the signals (read by dataset score) ----
    def mtla_slot(self, task: str, slot: str = "all") -> SlotSpec:
        """SlotSpec for the MTLA (inside-region) signal at the requested token-aggregation."""
        raise NotImplementedError

    def svar_slot(self, task: str) -> SlotSpec:
        """SlotSpec for the SVAR (global) baseline signal."""
        raise NotImplementedError

    # ---- which GPU stage scripts implement this model's stages ----
    def generate_script(self, task: str, engine: str) -> str:
        """Filename in mtla/stages/ for the generate stage (depends on engine: hf|vllm)."""
        raise NotImplementedError(f"{type(self).__name__} has no generate for task {task}")

    def extract_script(self, task: str) -> str:
        """Filename in mtla/stages/ for the (always HF-eager) extract stage."""
        raise NotImplementedError(f"{type(self).__name__} has no extract for task {task}")
