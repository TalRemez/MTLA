"""Model adapter interface.

A model adapter holds everything that depends on the *model family* (Qwen3-VL, InternVL, ...):
the output parser, which transformer module to hook for attention, and the model-specific pieces
of the two GPU stages — vLLM request building (generate) and the MTLA extraction callbacks
(extract). Everything model-*agnostic* (the MTLA kernel, band reduction, voting, metrics) lives in
the core ``mtla`` package; everything *dataset*-specific (items, prompt, ground truth, metric)
lives in ``mtla.data``.

A ``task`` is a coarse modality family, not a benchmark:
  - ``image_det``  : image object detection (COCO; InternVL or Qwen3-VL)
  - ``video_span`` : temporal grounding (QVHighlights, Charades; Qwen3-VL)
Using a family (not a benchmark) is what lets one dataset run on multiple models.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prediction:
    """One parsed grounding prediction.

    region: bounding box ``[x1,y1,x2,y2]`` in ``[0,1000]`` (image) or ``[t_start,t_end]`` in
            seconds (video); label: the predicted class string (empty for video spans).
    """
    region: list
    label: str


def hallucinated(region, label, gt, overlap) -> bool:
    """Is a predicted ``(region, label)`` a hallucination? True when no ground-truth region of the
    same label overlaps it by >= 0.5 (``overlap`` is ``iou`` for boxes or ``tiou`` for spans).
    ``gt`` is a list of ``{"region", "label"}`` dicts; video labels are empty so only overlap counts.
    """
    pl = (label or "").strip().lower()
    for g in gt:
        if (g.get("label", "") or "").strip().lower() == pl and overlap(region, g["region"]) >= 0.5:
            return False
    return True


class ModelAdapter:
    """Base class for model-family adapters."""

    model_id: str = ""
    # dotted path to the module whose `eager_attention_forward` is hooked for capture, e.g.
    # "transformers.models.qwen3.modeling_qwen3" (Qwen3 LM backbone, shared by InternVL-HF).
    attn_module_path: str = ""
    # task families this model supports, e.g. ("image_det",) or ("image_det", "video_span").
    tasks: tuple = ()

    def _check_task(self, task):
        """Raise if ``task`` is a family this model doesn't support (``None`` = the model default)."""
        if task is not None and task not in self.tasks:
            raise ValueError(
                f"{type(self).__name__} does not support task {task!r}; supported: {list(self.tasks)}")

    # ---- pure, CPU-testable ----
    def parse(self, response: str, task: str = None) -> list:
        """Parse a raw model response into a list of ``Prediction``. Called by BOTH the extract
        stage (to build Q_p) and the score stage (to recover the full candidate set)."""
        raise NotImplementedError

    # ---- generation (vLLM only; driven by generate.py) ----
    def gen_processor(self):
        """Load + return the processor/tokenizer used to build vLLM requests (once per worker)."""
        raise NotImplementedError(f"{type(self).__name__} has no gen_processor")

    def vllm_engine_args(self, dataset) -> dict:
        """Model/modality-specific vLLM engine kwargs (e.g. ``limit_mm_per_prompt``,
        ``max_model_len``). May read ``dataset.task`` to pick the modality limit."""
        return {}

    def vllm_uses_seed(self, task: str) -> bool:
        """Whether to pass the rollout seed into vLLM SamplingParams (per-request reproducibility).
        Task-aware: e.g. Qwen COCO does NOT seed (matches the paper preds) but Qwen video does."""
        return False

    def build_request(self, proc, item, dataset, cfg):
        """``(proc, item, dataset, cfg)`` -> ``{prompt, multi_modal_data, mm_processor_kwargs?}`` or
        None to skip. Builds the model-specific prompt + multimodal payload; ``cfg`` carries paths
        and ``cfg.preprocess`` (video fps/pixels)."""
        raise NotImplementedError(f"{type(self).__name__} has no build_request")

    # ---- MTLA extraction (HF eager attention; driven by extract.py) ----
    # extract.py calls mtla.mtla_attn.compute_mtla(model, record, ctx) directly — the model does
    # not mediate the computation, it only loads the model and supplies the callbacks below.
    def load_for_extract(self, gpu_id: int, task: str) -> dict:
        """Load model + processor on ``gpu_id``, install the capture wrapper, and return a ctx dict
        (model, tokenizer, state, device, n_layers, n_heads, task, ...) that ``compute_mtla`` and
        the callbacks below read."""
        raise NotImplementedError(f"{type(self).__name__} has no load_for_extract")

    # ---- model/task-specific callbacks invoked by compute_mtla ----
    # Each returns plain data; the shared driver owns the common flow (Q_p assembly, the single
    # captured forward, the MTLA math, buffer->record). Override these to add a model/task.
    def build_inputs(self, record, ctx, rank):
        """Preprocess the input, build the prompt, PARSE the response into predictions + their
        hallucination flags, and locate the modality tokens. Return a dict (or None to skip) with:
          prompt_ids (1-D tensor), response (str), modality_idx_l (modality-token positions),
          predictions (boxes/windows), hallu_flags (list[bool], aligned), meta (geometry), plus any
          keys forward_kwargs needs (e.g. pixel_values / inputs)."""
        raise NotImplementedError

    def query_tokens(self, response, predictions, tokenizer):
        """Per prediction, its response tokens Q_p as ``{first_label_tok, label_toks, coord_toks}``
        (or None), index-aligned with ``predictions``."""
        raise NotImplementedError

    def region_mask(self, prediction, meta):
        """Modality-token indices inside one prediction's region M(R_p): image patches inside a
        bbox, or frame tokens inside a time span. ``prediction`` is a ``Prediction`` (use
        ``prediction.region``); the mask depends on the model's token layout, so it lives here."""
        raise NotImplementedError

    def forward_kwargs(self, full_ids, total_len, device, inp):
        """kwargs dict for the single captured ``model(**fk)`` forward."""
        raise NotImplementedError

    # ---- generic record shape (model-agnostic; the shards are the score stage's whole input) ----
    def prediction_record(self, prediction, pred_idx, meta) -> dict:
        """Per-prediction record fields. Generic: region + label. The driver adds is_hallucinated /
        extracted / local_attention / first_digit."""
        return {"pred_idx": pred_idx, "region": list(prediction.region), "label": prediction.label}

    def item_record(self, record, meta, objects, n_predictions) -> dict:
        """Top-level saved record. Generic: the item id + its ground truth + dataset extras +
        objects, so the score stage reads everything it needs straight from the shards."""
        return {"id": record["id"], "gt": record.get("gt", []), "extra": record.get("extra", {}),
                "n_predictions": n_predictions, "n_extracted": sum(o["extracted"] for o in objects),
                "objects": objects}
