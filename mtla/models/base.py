"""Model adapter interface.

A model adapter holds everything that depends on the *model family* (Qwen3-VL, InternVL, ...):
how to build a prompt's inputs, how to parse the model's grounding output, which transformer
module to monkeypatch for attention extraction, where a prediction's tokens (`Q_p`) live, and
how to turn a predicted region into the set of modality-token indices inside it (`M(R_p)`).

Everything model-*agnostic* (the layer-band reduction, voting, AUROC) stays in the core
`mtla` package; everything *dataset*-specific (prompt text, metric) lives in `mtla.data`.
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
    """Base class for model-family adapters. Subclasses set the class attributes and
    implement the methods used by the generate/extract stages."""

    model_id: str = ""
    # dotted path to the module whose `eager_attention_forward` is monkeypatched, e.g.
    # "transformers.models.qwen3.modeling_qwen3" (InternVL LLM) or ".qwen3_vl.modeling_qwen3_vl"
    attn_module_path: str = ""

    def parse(self, response: str, **kw) -> list:
        """Parse a raw model response into a list of `Prediction`."""
        raise NotImplementedError

    def region_mask(self, region, meta: dict):
        """Return (inside_idx, outside_idx) into the modality tokens for `region`.

        `meta` carries the per-item geometry the mask needs (e.g. image patch grid / tiling
        for images, or duration_s + n_tokens for video/audio). Implementations delegate to
        `mtla.mask`.
        """
        raise NotImplementedError

    # generate / extract stages call into model-specific machinery (heavy, GPU). Adapters that
    # support a stage implement these; `run.py` calls them. Kept loose (dict in/out) so each
    # family can carry its own preprocessing without forcing a lowest-common-denominator API.
    def generate(self, cfg, dataset):
        raise NotImplementedError(f"{type(self).__name__} has no generate stage")

    def extract(self, cfg, dataset):
        raise NotImplementedError(f"{type(self).__name__} has no extract stage")
