"""Model-family adapter interface for MTLA generation and extraction.

A model adapter holds everything that depends on the model family (Qwen3-VL,
InternVL, ...): the output parser, which transformer module to hook for attention,
and the model-specific pieces of the two GPU stages, namely vLLM request building
(generate) and the MTLA extraction callbacks (extract). Everything model-agnostic
(the MTLA kernel, band reduction, voting, metrics) lives in the core ``mtla``
package; everything dataset-specific (items, prompt, ground truth, metric) lives in
``mtla.data``. Splitting the family-specific and dataset-specific parts is what lets
one dataset run on multiple models.

A ``task`` is a coarse modality family, not a benchmark:
  * ``image_det``  : image object detection (COCO; InternVL or Qwen3-VL).
  * ``video_span`` : temporal grounding (QVHighlights, Charades; Qwen3-VL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoProcessor

from mtla.mtla_attn import CaptureState, install_capture
from mtla.types import (
    BuildInputs,
    Ctx,
    GenRecord,
    GTRegion,
    ItemRecord,
    OverlapFn,
    PredObject,
    Region,
    TokenRange,
)

if TYPE_CHECKING:
    import torch


@dataclass
class Prediction:
    """One parsed grounding prediction (a candidate region plus its label).

    Produced by ``ModelAdapter.parse`` and consumed by both GPU stages: extraction
    (to build the query tokens Q_p) and scoring (to recover the full candidate set).

    Attributes:
        region: The predicted region. For images a bounding box ``[x1,y1,x2,y2]`` in
            ``[0,1000]`` normalized coordinates; for video a ``[t_start,t_end]`` time
            span in seconds.
        label: Predicted class string (lowercased); empty for video spans, which
            carry no class.
    """

    region: Region
    label: str


def hallucinated(
    region: Region, label: str, gt: list[GTRegion], overlap: OverlapFn
) -> bool:
    """Decide whether a predicted region is a hallucination.

    A prediction is grounded (not a hallucination) iff some ground-truth region of the
    same label overlaps it by at least 0.5. For video spans labels are empty, so the
    test reduces to overlap alone.

    Args:
        region: The predicted region: a bbox ``[x1,y1,x2,y2]`` in ``[0,1000]`` or a
            ``[t_start,t_end]`` span in seconds.
        label: The predicted class string (compared case-insensitively; empty for
            video spans).
        gt: Ground-truth regions as a list of ``{"region", "label"}`` dicts, matching
            ``region``/``label`` in format.
        overlap: Overlap function, ``iou`` for boxes or ``tiou`` for spans, taking two
            regions and returning a value in ``[0, 1]``.

    Returns:
        ``True`` if no matching-label ground-truth region overlaps by >= 0.5 (i.e. the
        prediction is a hallucination), else ``False``.
    """
    pl = (label or "").strip().lower()
    for g in gt:
        if (g.get("label", "") or "").strip().lower() == pl and overlap(
            region, g["region"]
        ) >= 0.5:
            return False
    return True


class ModelAdapter:
    """Base class for model-family adapters.

    Subclasses fill in the family-specific pieces of two GPU stages. Generation
    (vLLM): ``gen_processor``, ``vllm_engine_args``, ``vllm_uses_seed``,
    ``build_request``. Extraction (HF eager attention): ``_load_model`` and
    ``_pad_token`` (used by the shared ``load_for_extract``), ``build_inputs``,
    ``query_tokens``, ``region_mask``, ``forward_kwargs``, ``prediction_record``,
    ``item_record``. The shared driver ``mtla.mtla_attn.compute_mtla`` owns the common
    flow and only calls these callbacks.

    Attributes:
        model_id: HF checkpoint id for the family.
        attn_module_path: Dotted path to the module whose ``eager_attention_forward``
            is hooked for capture, e.g. ``transformers.models.qwen3.modeling_qwen3``
            (the Qwen3 LM backbone, shared by InternVL-HF).
        tasks: Task families this model supports, e.g. ``("image_det",)`` or
            ``("image_det", "video_span")``.
        _task: The task the current extraction is running (set per-item by
            ``compute_mtla``; ``None`` outside extraction).
    """

    model_id: str = ""
    # dotted path to the module whose `eager_attention_forward` is hooked for capture, e.g.
    # "transformers.models.qwen3.modeling_qwen3" (Qwen3 LM backbone, shared by InternVL-HF).
    attn_module_path: str = ""
    # task families this model supports, e.g. ("image_det",) or ("image_det", "video_span").
    tasks: tuple = ()
    # the task the current extraction is running (set per-item by compute_mtla; None outside extract).
    _task: str | None = None

    def _check_task(self, task: str | None) -> None:
        """Validate that this model supports ``task``.

        Args:
            task: A task family name, or ``None`` to accept the model default.

        Raises:
            ValueError: If ``task`` is not ``None`` and not in ``self.tasks``.
        """
        if task is not None and task not in self.tasks:
            raise ValueError(
                f"{type(self).__name__} does not support task {task!r}; supported: {list(self.tasks)}"
            )

    # ---- pure, CPU-testable ----
    def parse(self, response: str, task: str | None = None) -> list["Prediction"]:
        """Parse a raw model response into grounding predictions.

        Called by both the extract stage (to build Q_p) and the score stage (to
        recover the full candidate set), so it must be pure and CPU-testable.

        Args:
            response: The raw generated text from the model.
            task: Task family controlling the parse (bbox vs. span), or ``None`` for
                the model default.

        Returns:
            A list of ``Prediction`` in response order.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    # ---- generation (vLLM only; driven by generate.py) ----
    def gen_processor(self) -> Any:
        """Load the processor/tokenizer used to build vLLM requests (once per worker).

        Override only if a family needs non-default processor kwargs.

        Returns:
            An ``AutoProcessor`` for ``self.model_id`` (its ``.tokenizer`` is also used
            during extraction).
        """
        return AutoProcessor.from_pretrained(self.model_id)

    def vllm_engine_args(self, dataset: Any) -> dict:
        """Model- and modality-specific vLLM engine kwargs.

        Args:
            dataset: The dataset being generated for; may be read (e.g.
                ``dataset.task``) to pick the modality limit.

        Returns:
            A kwargs dict merged into the vLLM engine args, e.g. ``limit_mm_per_prompt``
            or ``max_model_len``. Empty by default.
        """
        return {}

    def vllm_uses_seed(self, task: str) -> bool:
        """Whether to seed vLLM SamplingParams per request for reproducibility.

        Task-aware: e.g. Qwen COCO does not seed (matches the validated paper preds)
        but Qwen video does.

        Args:
            task: The task family for the current run.

        Returns:
            ``True`` to pass the rollout seed into SamplingParams; ``False`` by default.
        """
        return False

    def build_request(
        self, proc: Any, item: dict, dataset: Any, cfg: Any
    ) -> dict | None:
        """Build a single vLLM request for one dataset item.

        Assembles the model-specific prompt and multimodal payload.

        Args:
            proc: The processor from ``gen_processor``.
            item: A raw dataset item (from ``load_items``), not a generation record.
            dataset: The dataset object; owns its prompt and media path.
            cfg: Run config carrying paths and ``cfg.preprocess`` (video fps/pixels).

        Returns:
            A dict ``{prompt, multi_modal_data, mm_processor_kwargs?}`` for vLLM, or
            ``None`` to skip this item.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} has no build_request")

    # ---- MTLA extraction (HF eager attention; driven by extract.py) ----
    # extract.py calls mtla.mtla_attn.compute_mtla(model, record, ctx) directly — the model does
    # not mediate the computation, it only loads the model and supplies the callbacks below.
    def _load_model(self, device: str) -> Any:
        """Load the HF model for extraction (family-specific hook).

        Loads with eager attention and ``.eval()``; the checkpoint class and device
        placement are family-specific, while the common wiring lives in
        ``load_for_extract``.

        Args:
            device: The CUDA device string, e.g. ``"cuda:0"``.

        Returns:
            The loaded HF model in eval mode on ``device``.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} has no _load_model")

    def _pad_token(self, task: str) -> str:
        """The modality pad token for ``task`` (family-specific hook).

        Its positions in the prompt mark the input-modality tokens (image patches or
        frame tokens) that MTLA scores against.

        Args:
            task: The task family for the current run.

        Returns:
            The pad-token string, e.g. ``<|image_pad|>``, ``<|video_pad|>``, or
            ``<IMG_CONTEXT>``.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} has no _pad_token")

    def load_for_extract(self, gpu_id: int, task: str) -> Ctx:
        """Load the model and processor and assemble the extraction context.

        Installs the attention-capture wrapper on ``attn_module_path`` and builds the
        per-layer id map. Family-specific bits are the ``_load_model`` and
        ``_pad_token`` hooks; the wiring (device, capture install, layer map, ctx
        assembly) is shared here.

        Args:
            gpu_id: CUDA device ordinal to place the model on.
            task: The task family for this extraction run.

        Returns:
            A ``Ctx`` dict read by ``compute_mtla`` and the callbacks, with keys
            ``model``, ``proc``, ``tokenizer``, ``state``, ``device``, ``task``,
            ``n_layers``, ``n_heads`` and ``pad_id``.
        """
        device = f"cuda:{gpu_id}"
        state = CaptureState()
        install_capture(self.attn_module_path, state)
        proc = self.gen_processor()
        model = self._load_model(device)
        layers = model.model.language_model.layers
        state.layer_ids = {id(L.self_attn): i for i, L in enumerate(layers)}
        return cast(
            Ctx,
            {
                "model": model,
                "proc": proc,
                "tokenizer": proc.tokenizer,
                "state": state,
                "device": device,
                "task": task,
                "n_layers": len(layers),
                "n_heads": model.config.text_config.num_attention_heads,
                "pad_id": proc.tokenizer.convert_tokens_to_ids(self._pad_token(task)),
            },
        )

    # ---- model/task-specific callbacks invoked by compute_mtla ----
    # Each returns plain data; the shared driver owns the common flow (Q_p assembly, the single
    # captured forward, the MTLA math, buffer->record). Override these to add a model/task.
    def build_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> BuildInputs | None:
        """Preprocess one item and assemble everything the MTLA driver needs.

        Preprocesses the input, builds the prompt, parses the response into predictions
        and their hallucination flags, and locates the modality tokens in the prompt.

        Args:
            record: The generation record for one item (id, prompt, response, gt,
                extra).
            ctx: The extraction context from ``load_for_extract``.
            rank: Worker rank, used only for logging skipped items.

        Returns:
            A ``BuildInputs`` dict, or ``None`` to skip the item. Keys: ``prompt_ids``
            (1-D tensor), ``response`` (str), ``modality_idx_l`` (modality-token
            positions), ``predictions`` (parsed boxes/windows), ``hallu_flags``
            (index-aligned ``list[bool]``), ``meta`` (region geometry), plus any keys
            ``forward_kwargs`` needs (e.g. ``pixel_values`` / ``inputs``).

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def query_tokens(
        self, response: str, predictions: list["Prediction"], tokenizer: Any
    ) -> list[TokenRange | None]:
        """Locate each prediction's response tokens Q_p via the tokenizer offsets.

        Args:
            response: The raw generated text (same string that was parsed).
            predictions: The parsed predictions to locate, in order.
            tokenizer: The tokenizer providing an offset mapping over ``response``.

        Returns:
            One entry per prediction, index-aligned with ``predictions``: a
            ``TokenRange`` dict ``{first_label_tok, label_toks, coord_toks}`` (the
            coordinate-digit and label tokens), or ``None`` if Q_p could not be located.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def region_mask(self, prediction: "Prediction", meta: dict) -> list[int]:
        """Modality-token indices inside one prediction's region M(R_p).

        Returns the image patches inside a bbox, or the frame tokens inside a time
        span. The mapping depends on the model's token layout, so it lives on the
        adapter.

        Args:
            prediction: The prediction whose ``region`` defines M(R_p).
            meta: Geometry from ``build_inputs`` (e.g. tile grid, patch grid, or
                video T/H/W and duration) needed to map the region to token indices.

        Returns:
            The modality-token indices (positions within ``modality_idx_l``) that fall
            inside the region.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def forward_kwargs(
        self, full_ids: "torch.Tensor", total_len: int, device: str, inp: BuildInputs
    ) -> dict:
        """Build the kwargs for the single captured ``model(**fk)`` forward.

        Args:
            full_ids: The full input-id tensor (prompt + response) for the pass.
            total_len: Total sequence length, used to size the attention mask.
            device: CUDA device string for newly allocated tensors.
            inp: The ``BuildInputs`` dict from ``build_inputs`` (source of
                ``pixel_values`` / ``inputs`` and any grid tensors).

        Returns:
            A kwargs dict passed straight to ``model(**fk)`` for the capture forward.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    # ---- generic record shape (model-agnostic; the shards are the score stage's whole input) ----
    def prediction_record(
        self, prediction: "Prediction", pred_idx: int, meta: dict
    ) -> dict:
        """Build the saved per-prediction record fields.

        Generic base fields only; the driver later adds ``is_hallucinated``,
        ``extracted``, ``local_attention`` and ``first_digit``.

        Args:
            prediction: The prediction being recorded.
            pred_idx: Its index within the item's prediction list.
            meta: Region geometry from ``build_inputs`` (unused here; available to
                overrides).

        Returns:
            A dict with ``pred_idx``, ``region`` (list form) and ``label``.
        """
        return {
            "pred_idx": pred_idx,
            "region": list(prediction.region),
            "label": prediction.label,
        }

    def item_record(
        self,
        record: GenRecord,
        meta: dict,
        objects: list[PredObject],
        n_predictions: int,
    ) -> ItemRecord:
        """Build the top-level saved record for one item.

        Generic across models so the score stage reads everything it needs straight
        from the shards.

        Args:
            record: The generation record for the item (source of ``id``, ``gt``,
                ``extra``).
            meta: Region geometry from ``build_inputs`` (unused here; available to
                overrides).
            objects: The per-prediction records (each from ``prediction_record`` plus
                driver-added fields).
            n_predictions: Number of parsed predictions for the item.

        Returns:
            An ``ItemRecord`` dict with ``id``, ``gt``, ``extra``, ``n_predictions``,
            ``n_extracted`` (count of extracted objects) and ``objects``.
        """
        return {
            "id": record["id"],
            "gt": record.get("gt", []),
            "extra": record.get("extra", {}),
            "n_predictions": n_predictions,
            "n_extracted": sum(o["extracted"] for o in objects),
            "objects": objects,
        }
