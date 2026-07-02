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
from mtla.utils import is_hallucinated
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


class ModelAdapter:
    """Base class for model-family adapters.

    Subclasses fill in the family-specific pieces of two GPU stages. Generation
    (vLLM): ``gen_processor`` and ``build_vllm_request``. Extraction
    (HF eager attention): the ``_load_model`` hook and the ``modality_pad_token`` attribute (used
    by the shared ``load_for_extract``), ``build_extraction_inputs``, ``locate_proposal_tokens``,
    ``proposal_region_attn_mask``, ``hf_extraction_kwargs``, ``prediction_record``, and
    ``item_record``. The shared driver ``mtla.mtla_attn.compute_mtla`` owns the common flow and
    only calls these callbacks.

    Attributes:
        model_id: HF checkpoint id for the family.
        attn_module_path: Dotted path to the module whose ``eager_attention_forward``
            is hooked for capture, e.g. ``transformers.models.qwen3.modeling_qwen3``
            (the Qwen3 LM backbone, shared by InternVL-HF).
        modality_pad_token: The modality pad token whose positions in the prompt mark the
            input-modality tokens MTLA scores against, e.g. ``<|image_pad|>``,
            ``<|video_pad|>``, or ``<IMG_CONTEXT>``.
    """

    model_id: str = ""
    # dotted path to the module whose `eager_attention_forward` is hooked for capture, e.g.
    # "transformers.models.qwen3.modeling_qwen3" (Qwen3 LM backbone, shared by InternVL-HF).
    attn_module_path: str = ""
    # modality pad token; its prompt positions are the input-modality tokens MTLA scores against.
    modality_pad_token: str = ""

    # ---- pure, CPU-testable ----
    def parse_response(self, response: str) -> list["Prediction"]:
        """Parse a raw model response into grounding predictions.

        Called by both the extract stage (to build Q_p) and the score stage (to
        recover the full candidate set), so it must be pure and CPU-testable. Each
        adapter is single-task, so the parse mode is fixed by the class.

        Args:
            response: The raw generated text from the model.

        Returns:
            A list of ``Prediction`` in response order.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    @staticmethod
    def _image_message(image: Any, prompt: str) -> list[dict]:
        """Build the one-turn chat message for a single image + text prompt.

        Args:
            image: The image (a PIL image) to place before the text.
            prompt: The task prompt to append after the image.

        Returns:
            A one-element chat list ``[{"role": "user", "content": [image, text]}]``
            (wrapped in a list because that is what the processors accept).
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # ---- generation (vLLM only; driven by generate.py) ----
    def gen_processor(self) -> Any:
        """Load the processor/tokenizer used to build vLLM requests (once per worker).

        Override only if a family needs non-default processor kwargs.

        Returns:
            An ``AutoProcessor`` for ``self.model_id`` (its ``.tokenizer`` is also used
            during extraction).
        """
        return AutoProcessor.from_pretrained(self.model_id)

    def build_vllm_request(
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
        raise NotImplementedError(f"{type(self).__name__} has no build_vllm_request")

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

    def load_for_extract(self, gpu_id: int) -> Ctx:
        """Load the model and processor and assemble the extraction context.

        Installs the attention-capture wrapper on ``attn_module_path`` and builds the
        per-layer id map. Family-specific bits are the ``_load_model`` hook and the
        ``modality_pad_token`` attribute; the wiring (device, capture install, layer map, ctx
        assembly) is shared here.

        Args:
            gpu_id: CUDA device ordinal to place the model on.

        Returns:
            A ``Ctx`` dict read by ``compute_mtla`` and the callbacks, with keys
            ``model``, ``proc``, ``tokenizer``, ``state``, ``device``, ``n_layers``,
            ``n_heads`` and ``pad_id``.
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
                "n_layers": len(layers),
                "n_heads": model.config.text_config.num_attention_heads,
                "pad_id": proc.tokenizer.convert_tokens_to_ids(self.modality_pad_token),
            },
        )

    # ---- shared helpers for build_extraction_inputs (below) ----
    @staticmethod
    def _locate_modality_tokens(
        prompt_ids: "torch.Tensor",
        pad_id: int,
        n_expected: int,
        record_id: Any,
        rank: int,
    ) -> list[int] | None:
        """Positions of the modality pad token in the prompt, or ``None`` if the count is off.

        A count that does not match ``n_expected`` means the item was preprocessed
        differently than the region geometry assumes, so the region-to-token mapping
        would be wrong; the item is logged and skipped rather than scored on a
        misaligned grid.

        Args:
            prompt_ids: The prompt's 1-D input-id tensor.
            pad_id: The modality pad-token id to find (image or video).
            n_expected: The number of modality tokens the computed grid implies.
            record_id: The item id, for the skip log line.
            rank: Worker rank, for the skip log line.

        Returns:
            The modality-token positions (a list of indices into ``prompt_ids``), or
            ``None`` when their count differs from ``n_expected``.
        """
        idx = [k for k, t in enumerate(prompt_ids.tolist()) if t == pad_id]
        if len(idx) != n_expected:
            print(
                f"[worker {rank}] skip {record_id}: modality tokens {len(idx)} != {n_expected}",
                flush=True,
            )
            return None
        return idx

    def _overlap_fn(self) -> OverlapFn:
        """Return this adapter's overlap metric, or fail loud if it was never set.

        ``overlap`` has no base default (it is ``None``); each concrete adapter must
        declare ``iou`` (boxes) or ``tiou`` (spans). Guarding here turns a forgotten
        declaration into a clear error instead of a cryptic ``None`` call deep in
        :func:`mtla.utils.is_hallucinated`.

        Returns:
            The adapter's ``overlap`` function.

        Raises:
            NotImplementedError: If the adapter left ``overlap`` unset.
        """
        if self.overlap is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set the `overlap` class attribute "
                "(iou for boxes, tiou for spans)"
            )
        return self.overlap

    @staticmethod
    def _hallu_flags(
        predictions: list["Prediction"], gt: list[GTRegion], overlap: OverlapFn
    ) -> list[bool]:
        """Per-prediction hallucination flags, index-aligned with ``predictions``.

        Args:
            predictions: The parsed predictions to label.
            gt: The item's ground-truth regions.
            overlap: The overlap function (``iou`` for boxes, ``tiou`` for spans).

        Returns:
            One bool per prediction: ``True`` if it is a hallucination (see
            :func:`mtla.utils.is_hallucinated`).
        """
        return [is_hallucinated(p.region, p.label, gt, overlap) for p in predictions]

    # ---- model/task-specific callbacks invoked by compute_mtla ----
    # Each returns plain data; the shared driver owns the common flow (Q_p assembly, the single
    # captured forward, the MTLA math, buffer->record). Override these to add a model/task.

    # Overlap function for the hallucination test: ``iou`` (boxes) or ``tiou`` (spans).
    # No default: each concrete adapter must declare the metric its task uses.
    overlap: OverlapFn | None = None

    def _select_predictions(
        self, predictions: list["Prediction"], ctx: Ctx
    ) -> list["Prediction"]:
        """Optionally narrow the parsed predictions before extraction.

        Default: keep all. Single-span video benchmarks override this to keep only the
        first span.

        Args:
            predictions: The parsed predictions, in response order.
            ctx: The extraction context (e.g. ``ctx["multi"]`` for video).

        Returns:
            The predictions to actually extract and score.
        """
        return predictions

    def _encode_attn_extraction_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> tuple["torch.Tensor", dict, dict, int] | None:
        """Re-encode one item's prompt + media for the capture forward (family hook).

        Builds the model inputs (prompt tokens + image/video tensors) and the region
        geometry, and reports how many modality tokens the geometry implies.

        Args:
            record: The generation record for one item.
            ctx: The extraction context from ``load_for_extract``.
            rank: Worker rank, used only for logging skips.

        Returns:
            ``(prompt_ids, inputs, meta, n_expected)`` — the prompt id tensor, the extra
            forward tensors (stored under ``inputs``/``pixel_values`` as the family's
            ``hf_extraction_kwargs`` expects), the ``meta`` region geometry, and the expected
            modality-token count — or ``None`` to skip the item.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def build_extraction_inputs(
        self, record: GenRecord, ctx: Ctx, rank: int
    ) -> BuildInputs | None:
        """Preprocess one item and assemble everything the MTLA driver needs.

        Shared template: parse the response into predictions, re-encode the prompt +
        media (``_encode_attn_extraction_inputs``), locate the modality tokens
        (``_locate_modality_tokens``), and label each prediction's hallucination flag
        (``_hallu_flags`` with ``overlap``). Families customize only
        ``_encode_attn_extraction_inputs``, ``modality_pad_token``, ``overlap``, and (for
        single-span video) ``_select_predictions``.

        Args:
            record: The generation record for one item (id, prompt, response, gt,
                extra).
            ctx: The extraction context from ``load_for_extract``.
            rank: Worker rank, used only for logging skipped items.

        Returns:
            A ``BuildInputs`` dict, or ``None`` to skip the item (no predictions,
            un-encodable input, or modality-token-count mismatch).
        """
        response = record.get("response")
        preds = self.parse_response(response) if response else []
        preds = self._select_predictions(preds, ctx)
        if not preds:
            return None
        encoded = self._encode_attn_extraction_inputs(record, ctx, rank)
        if encoded is None:
            return None
        prompt_ids, extra_inputs, meta, n_expected = encoded
        modality_idx = self._locate_modality_tokens(
            prompt_ids, ctx["pad_id"], n_expected, record["id"], rank
        )
        if modality_idx is None:
            return None
        return cast(
            BuildInputs,
            {
                "prompt_ids": prompt_ids,
                "response": response,
                "modality_idx_l": modality_idx,
                "predictions": preds,
                "hallu_flags": self._hallu_flags(
                    preds, record.get("gt", []), self._overlap_fn()
                ),
                "meta": meta,
                **extra_inputs,
            },
        )

    def locate_proposal_tokens(
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

    def proposal_region_attn_mask(
        self, prediction: "Prediction", meta: dict
    ) -> list[int]:
        """Modality-token indices inside one prediction's region M(R_p).

        Returns the image patches inside a bbox, or the frame tokens inside a time
        span. The mapping depends on the model's token layout, so it lives on the
        adapter.

        Args:
            prediction: The prediction whose ``region`` defines M(R_p).
            meta: Geometry from ``build_extraction_inputs`` (e.g. tile grid, patch grid, or
                video T/H/W and duration) needed to map the region to token indices.

        Returns:
            The modality-token indices (positions within ``modality_idx_l``) that fall
            inside the region.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def hf_extraction_kwargs(
        self, full_ids: "torch.Tensor", total_len: int, device: str, inp: BuildInputs
    ) -> dict:
        """Build the kwargs for the single captured ``model(**fk)`` forward.

        Args:
            full_ids: The full input-id tensor (prompt + response) for the pass.
            total_len: Total sequence length, used to size the attention mask.
            device: CUDA device string for newly allocated tensors.
            inp: The ``BuildInputs`` dict from ``build_extraction_inputs`` (source of
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
            meta: Region geometry from ``build_extraction_inputs`` (unused here; available to
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
            meta: Region geometry from ``build_extraction_inputs`` (unused here; available to
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
