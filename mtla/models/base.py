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


def label_hallu_bbox(pred_box, pred_label, gt_objs) -> bool:
    """Image-det hallucination label: a predicted (box, label) is hallucinated if no same-label
    GT box has IoU >= 0.5. Shared by the image_det adapters (InternVL, Qwen3-VL)."""
    from ..utils import iou
    pl = pred_label.strip().lower()
    for go in gt_objs:
        if str(go.get("label", "")).strip().lower() == pl and iou(pred_box, go["bbox_2d"]) >= 0.5:
            return False
    return True


class ModelAdapter:
    """Base class for model-family adapters."""

    model_id: str = ""
    # dotted path to the module whose `eager_attention_forward` is monkeypatched, e.g.
    # "transformers.models.qwen3.modeling_qwen3" (InternVL LLM) or "...qwen3_vl.modeling_qwen3_vl".
    attn_module_path: str = ""
    # task families this model supports, e.g. ("image_det",) or ("video_span",).
    tasks: tuple = ()

    def _check_task(self, task):
        """Raise if `task` is a family this model doesn't support. `None` means the model's own
        default (kept for single-task models whose `parse` ignores the argument)."""
        if task is not None and task not in self.tasks:
            raise ValueError(
                f"{type(self).__name__} does not support task {task!r}; "
                f"supported: {list(self.tasks)}")

    # ---- pure, CPU-testable ----
    def parse(self, response: str, task: str = None, **kw) -> list:
        """Parse a raw model response into a list of `Prediction`."""
        raise NotImplementedError

    # ---- which GPU stage scripts implement this model's stages ----
    def generate_script(self, task: str, engine: str) -> str:
        """Filename in scripts/stages/ for the generate stage (depends on engine: hf|vllm)."""
        raise NotImplementedError(f"{type(self).__name__} has no generate for task {task}")

    def extract_script(self, task: str) -> str:
        """Filename in scripts/stages/ for the (always HF-eager) extract stage."""
        raise NotImplementedError(f"{type(self).__name__} has no extract for task {task}")

    # ---- generation contract (driven by scripts/stages/generate.py) ----
    # One uniform surface for BOTH modalities and BOTH engines. The driver owns the orchestration
    # (which GPUs, which strategy, shard-merge); the model owns only "how do I turn an item into a
    # request / a response". Two backends:
    #   vLLM (fast): build a request dict, the strategy runs the engine + SamplingParams + decode.
    #   HF (reference): the model owns the whole forward and returns the decoded text.
    # Both `run_pooled` (async multi-engine, best for many small image requests) and `run_sharded`
    # (one blocking engine per GPU, best for heavy video clips) drive these SAME methods.
    def gen_engines(self, task: str) -> tuple:
        """Engines this model supports for `task`, e.g. ("vllm",) or ("vllm", "hf")."""
        return ("vllm",)

    def gen_processor(self):
        """Load + return the processor/tokenizer used to build vLLM requests and chat prompts
        (called once per worker). vLLM path only."""
        raise NotImplementedError(f"{type(self).__name__} has no gen_processor")

    def vllm_engine_args(self, dataset) -> dict:
        """Model/modality-specific vLLM engine kwargs beyond the strategy's tuning knobs — e.g.
        `limit_mm_per_prompt`, `max_model_len`, `trust_remote_code`. May read `dataset.task` to pick
        the modality limit ({"image":1} vs {"video":1})."""
        return {}

    def vllm_uses_seed(self, task: str) -> bool:
        """Whether to pass the rollout seed into vLLM SamplingParams (per-request reproducibility).
        Task-aware: e.g. Qwen COCO does NOT seed (matches the paper preds) but Qwen video does."""
        return False

    def build_vllm_request(self, proc, item, dataset, cfg):
        """(proc, item, dataset, cfg) -> {prompt, multi_modal_data, mm_processor_kwargs?} or None to
        skip. Builds the model-specific prompt + multimodal payload; `cfg` resolves paths (e.g. the
        video dir)."""
        raise NotImplementedError(f"{type(self).__name__} has no build_vllm_request")

    def load_hf_gen(self, gpu_id: int) -> dict:
        """Load model + processor for HF generation on `gpu_id` and return a ctx dict. The sharded
        worker pins one GPU via CUDA_VISIBLE_DEVICES before importing torch, so `gpu_id` is 0."""
        raise NotImplementedError(f"{type(self).__name__} has no load_hf_gen")

    def generate_hf(self, ctx, item, dataset, cfg, seed, temperature):
        """(ctx, item, dataset, cfg, seed, temperature) -> (response, truncated) or (None, False) to
        skip. The worker pre-seeds the RNG per (seed, rank); temperature==0 means greedy."""
        raise NotImplementedError(f"{type(self).__name__} has no generate_hf")

    # ---- HF-eager MTLA extraction (driven by scripts/stages/{image,video}_extract.py) ----
    def load_for_extract(self, gpu_id: int) -> dict:
        """Load model + processor on `gpu_id`, install the MTLA attention forward, and return a
        ctx dict (model, tokenizer, MTLAState, device, n_layers, n_heads, ...) the shared driver
        passes back to `extract_one`."""
        raise NotImplementedError(f"{type(self).__name__} has no load_for_extract")

    def extract_one(self, p: dict, ds_by_id: dict, ctx: dict, svar_shift: bool, rank: int = 0):
        """Compute MTLA for one item's predictions and return its saved .pt record (or None to
        skip). This is shared: it delegates to `mtla.mtla_attn.compute_mtla`, which drives the
        per-item MTLA computation (identical for image boxes and video windows) and calls back the
        `ext_*` methods below. `self._task` is recorded from the ctx so multi-task adapters'
        `ext_*` can dispatch on it (single-task adapters simply ignore it)."""
        from ..mtla_attn import compute_mtla
        self._task = ctx.get("task", self.tasks[0] if self.tasks else None)
        return compute_mtla(self, p, ds_by_id, ctx, svar_shift, rank)

    # ---- model/task-specific callbacks invoked by mtla.mtla_attn.compute_mtla ----
    # Each returns plain data; the shared driver owns the common flow (Q_p assembly, query-position
    # dedup, pred_specs, the single forward, buffer->record). Override these to add a model/task.
    # A "prediction" is a bbox (image_det) or a [t0,t1] window (video_span).
    def ext_build_inputs(self, p, ds_by_id, ctx, rank):
        """Preprocess the input (image/video), build the prompt, locate the modality tokens, and
        enumerate the predictions. Return a dict (or None to skip the item) with keys:
          prompt_ids (1-D tensor), response (str), modality_idx_l (modality-token positions),
          predictions (list of bbox/window), hallu_flags (list[bool], aligned with predictions),
          meta (geometry), plus any keys ext_forward_kwargs needs (e.g. pixel_values / inputs)."""
        raise NotImplementedError

    def ext_token_ranges(self, response, predictions, tokenizer):
        """Per-prediction {first_label_tok, label_toks, coord_toks} (the tokens Q_p) or None.
        Images: label + coordinate tokens. Video: the window's digit tokens (as `coord_toks`,
        with `first_label_tok` = the first digit), so the driver's Q_p assembly is identical."""
        raise NotImplementedError

    def ext_region_mask(self, prediction, meta):
        """Modality-token indices inside one prediction's proposal region M(R_p): image patches
        inside a bbox, or the frame tokens inside a time span."""
        raise NotImplementedError

    def ext_forward_kwargs(self, full_ids, total_len, device, inp):
        """kwargs dict for the patched `model(**fk)` forward."""
        raise NotImplementedError

    def ext_obj_record(self, prediction, pred_idx, meta) -> dict:
        """Per-prediction record fields (the driver adds is_hallucinated / n_qp_tokens /
        local_attention / first_digit). Images: {pred_idx, label, box, ...geometry}; video:
        {pred_idx, window, ...geometry}."""
        raise NotImplementedError

    def ext_record(self, p, meta, objects, n_predictions) -> dict:
        """Wrap the per-prediction objects into the top-level saved record (id keys + counts)."""
        raise NotImplementedError
