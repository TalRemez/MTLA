"""Attention extraction core: a monkeypatch that captures per-token attention to modality tokens.

MTLA needs the raw softmax attention weights from a prediction's output tokens to the input
(image / video / audio) tokens. Most efficient attention kernels (SDPA, FlashAttention) never
materialize that matrix, so we swap in an *eager* attention forward that computes softmax
explicitly and, for a chosen set of query positions, records their attention to the modality
tokens before continuing the forward pass unchanged.

This module provides the model-agnostic machinery:

  * ``ExtractionState`` - holds the query positions, modality-token indices, per-prediction
    region masks, and the output buffers for one forward pass.
  * ``make_eager_attention_forward(state)`` - returns a drop-in replacement for a HF
    ``eager_attention_forward`` that fills ``state`` while running.
  * ``install(module_path, state)`` - patch a model's attention function by dotted module path.

The per-model glue (building the prompt, locating each prediction's coordinate/label token
positions, and mapping its box/span to modality indices via :mod:`mtla.mask`) lives in the
runnable examples, which faithfully reproduce the paper numbers. See
``examples/coco_detection/extract.py`` and ``examples/qvhighlights/extract.py``.

Output per prediction ``p``, per layer ``l`` and head ``h`` (shape ``[L, H]`` each):

  * ``image_sum``        - attention to *all* modality tokens (the SVAR/GA baseline)
  * ``image_inside_sum`` - attention to tokens inside ``p``'s proposal region (MTLA)
  * ``image_outside_sum``- attention to tokens outside it (diagnostic)
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ExtractionState:
    """Mutable state shared between the patched attention function and the driver loop.

    Set the index tensors before a forward pass and read the accumulated ``buf`` after it.
    All index tensors live on the model's device; buffers accumulate on CPU in fp32.
    """
    active: bool = False
    lang_attn_ids: set = field(default_factory=set)   # id() of each decoder self-attn module
    lang_attn_order: list = field(default_factory=list)  # same ids, in layer order
    query_positions: torch.Tensor | None = None        # [Nq] absolute positions to record
    image_indices: torch.Tensor | None = None          # [n_img] modality-token positions
    pred_specs: list | None = None                      # per-prediction dict (see below)
    buf: dict | None = None                             # {pred_idx: {stat: [L,H] tensor}}
    n_layers: int = 0
    n_heads: int = 0

    def new_buffer(self, n_preds: int):
        """Allocate zeroed ``[n_preds, L, H]`` accumulators for one forward pass."""
        z = lambda: torch.zeros(n_preds, self.n_layers, self.n_heads, dtype=torch.float32)
        self.buf = {
            "image_sum": z(),
            "image_inside_sum": z(),
            "image_outside_sum": z(),
        }
        return self.buf


def make_eager_attention_forward(state: ExtractionState, repeat_kv):
    """Build a patched ``eager_attention_forward`` that records attention into ``state``.

    ``repeat_kv`` is the model family's grouped-query-attention helper (e.g.
    ``transformers.models.qwen3.modeling_qwen3.repeat_kv``). The returned function is numerically
    identical to the stock eager forward; it only *additionally* reads off attention rows for
    the recorded query positions when ``state.active`` is true.
    """
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        if state.active and id(module) in state.lang_attn_ids:
            layer_idx = state.lang_attn_order.index(id(module))
            q_pos = state.query_positions
            # rows: [Nq, H, K] attention from each recorded query position to every key
            rows = attn_weights[0].index_select(1, q_pos).transpose(0, 1).float()
            img_rows = rows.index_select(2, state.image_indices)  # [Nq, H, n_img]
            for pi, spec in enumerate(state.pred_specs):
                # average the prediction's token rows (multi-token aggregation, eq. MTLA)
                qrows = spec["qrows"]
                img_row = img_rows.index_select(0, qrows).mean(dim=0)  # [H, n_img]
                state.buf["image_sum"][pi, layer_idx, :] += img_row.sum(dim=1).cpu()
                in_idx = spec["inside_idx"]
                if in_idx.numel() > 0:
                    state.buf["image_inside_sum"][pi, layer_idx, :] += (
                        img_row.index_select(1, in_idx).sum(dim=1).cpu())
                out_idx = spec["outside_idx"]
                if out_idx.numel() > 0:
                    state.buf["image_outside_sum"][pi, layer_idx, :] += (
                        img_row.index_select(1, out_idx).sum(dim=1).cpu())

        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return attn_output, None

    return patched


def install(module_path: str, state: ExtractionState):
    """Monkeypatch ``eager_attention_forward`` in ``module_path`` (e.g.
    ``"transformers.models.qwen3.modeling_qwen3"``). Returns the patched function so the caller
    can keep a reference. The module must expose ``repeat_kv``.
    """
    mod = importlib.import_module(module_path)
    patched = make_eager_attention_forward(state, mod.repeat_kv)
    mod.eager_attention_forward = patched
    return patched
