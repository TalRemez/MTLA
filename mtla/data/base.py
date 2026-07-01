"""Dataset adapter interface + shared machinery.

A dataset adapter is **declarative**: it says what the benchmark *is* (how to load items, the
prompt, the ground truth, the uniform generation record) and *declares* how it should be scored
(which MTLA signal, overlap function, candidate selection, and metric). It does **no** computation
— shard loading, band reduction, voting, NMS, and metric evaluation all live in ``mtla.evaluate``
and ``mtla.metrics``. This keeps every adapter small and identical in shape.

Scoring descriptors (read by ``mtla.evaluate.run_score``):
  * ``signal``  — which saved ``[L, H]`` array to reduce: ``local_attention`` (images) or
    ``first_digit`` (video, the validated choice).
  * ``overlap`` — ``iou`` (boxes) or ``tiou`` (temporal spans), used for both voting and metrics.
  * ``select``  — ``fuse`` (NMS pool across rollouts; detection, multi-window) or ``argmax``
    (keep the single best candidate; single-span grounding).
  * ``metric``  — names a pure computer in ``mtla.metrics`` (``coco_map`` | ``moment_retrieval`` |
    ``recall_at_iou``).

A ``task`` family (``image_det`` | ``video_span``) tells the model adapter which parser / region
mask to use, so any valid ``(model x dataset)`` pair runs from a config.
"""
from __future__ import annotations

import glob

import torch


def load_shards(features_dir: str) -> list:
    """Load + concatenate all ``shard*.pt`` records under a seed's feature dir."""
    recs = []
    for sp in sorted(glob.glob(f"{features_dir}/shard*.pt")):
        recs.extend(torch.load(sp, weights_only=False, map_location="cpu"))
    return recs


def print_metrics(name: str, metrics: dict, indent: str = "  ") -> None:
    """Pretty-print a (possibly nested) metrics dict returned by ``mtla.evaluate.run_score``."""
    print(f"[{name}] results:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            inner = "  ".join(f"{ik}={iv:.4f}" if isinstance(iv, float) else f"{ik}={iv}"
                              for ik, iv in v.items())
            print(f"{indent}{k}: {inner}")
        elif isinstance(v, float):
            print(f"{indent}{k} = {v:.4f}")
        else:
            print(f"{indent}{k} = {v}")


class DatasetAdapter:
    """Base class for benchmark adapters."""

    name: str = ""
    task: str = ""

    # ---- scoring descriptors (see module docstring; read by mtla.evaluate) ----
    signal: str = "local_attention"
    overlap: str = "iou"
    select: str = "fuse"
    metric: str = "coco_map"

    # ---- generation behaviour ----
    # Whether rollout 0 is a greedy (T=0) anchor for self-consistency voting. Video benchmarks set
    # True (the N=16 recipe: greedy seed 0 + N-1 stochastic); a config `generate.greedy_seed0`
    # overrides it. See RunConfig.gen_temperature.
    greedy_seed0: bool = False
    # Execution strategy for generate.py: "pooled" = async multi-engine vLLM pool
    # (throughput on many small requests, e.g. 5k COCO images); "sharded" = one engine per GPU
    # (heavy per-item work, e.g. video clips).
    gen_strategy: str = "sharded"

    # ---- per-benchmark: subclasses implement ----
    def load_items(self, cfg) -> list:
        """Load the work items (images or video queries) for generation."""
        raise NotImplementedError

    def prompt(self, item) -> str:
        """The task prompt for one item."""
        raise NotImplementedError

    def ground_truth(self, item) -> list:
        """Ground truth as a list of ``{"region", "label"}`` dicts (label empty for video spans)."""
        raise NotImplementedError

    def gen_record(self, cfg, item, response: str, truncated: bool = False) -> dict:
        """Uniform generation record: ``{id, prompt, response(raw), gt, extra}``. The response is
        stored RAW — parsing happens in the extract/score stages, identically for every model.
        ``extra`` carries anything the extract stage needs to locate the input (e.g. an absolute
        video/image path, resolved via ``cfg.path(...)``)."""
        raise NotImplementedError
