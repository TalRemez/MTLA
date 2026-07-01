"""Dataset adapter interface + shared machinery.

A dataset adapter holds what depends on the *benchmark*: how to load items, the prompt, the
ground truth, and the task metric. Everything mechanical that the benchmarks share — iterating
seeds for the GPU stages, loading feature shards, computing hallucination AUROC, printing a
metrics dict — lives here in the base so the per-benchmark adapters stay small.

`score(cfg, model)` returns a metrics dict (it does not print); `run.py` prints it via
`print_metrics`. A `task` family ("image_det" | "video_span") tells the model adapter which
parser/mask/slots/stage-scripts to use, so any valid (model x dataset) pair runs from a config.
"""
from __future__ import annotations

import glob


class DatasetAdapter:
    """Base class for benchmark adapters."""

    name: str = ""
    task: str = ""
    # Default for self-consistency voting: whether rollout 0 is a greedy (T=0) anchor. Video
    # benchmarks set True (the paper's N=16 recipe: greedy seed 0 + N-1 stochastic); a config
    # `generate.greedy_seed0` overrides it. See RunConfig.gen_temperature.
    greedy_seed0: bool = False

    # ---- generation-stage properties (read by scripts/stages/generate.py) ----
    # Execution strategy: "pooled" = async multi-engine vLLM pool (throughput on many small
    # requests, e.g. 5k COCO images); "sharded" = one blocking engine per GPU (heavy per-item
    # work, e.g. video clips). Either modality may use either; forced to "sharded" for engine: hf.
    gen_strategy: str = "sharded"
    # Sampling knobs the generation driver passes to whichever engine runs (both are here so the
    # model stays engine-agnostic). `gen_max_new_tokens` caps the decode; `gen_top_p` is the
    # nucleus for stochastic rollouts (temperature comes from cfg.gen_temperature).
    gen_max_new_tokens: int = 4096
    gen_top_p: float = 1.0

    # ---- per-benchmark: subclasses implement ----
    def load_items(self, cfg) -> list:
        """Load the work items (images or video queries) for generation/extraction."""
        raise NotImplementedError

    def prompt(self, item) -> str:
        raise NotImplementedError

    def ground_truth(self, item):
        raise NotImplementedError

    def stage_cmd(self, cfg, model, seed: int, mode: str):
        """Return (script_name, arg_list) for one GPU stage. `mode` is "generate" | "extract".
        The base generate/extract loop over seeds calls this; subclasses build the dataset's args."""
        raise NotImplementedError

    def score(self, cfg, model) -> dict:
        """Compute metrics from the run's feature shards and RETURN a dict (no printing)."""
        raise NotImplementedError

    # ---- shared machinery (base) ----
    def generate(self, cfg, model, seed=0):
        from ..pipeline import run_stage
        run_stage(*self.stage_cmd(cfg, model, seed, "generate"))

    def extract(self, cfg, model, seed=0):
        from ..pipeline import run_stage
        run_stage(*self.stage_cmd(cfg, model, seed, "extract"))

    @staticmethod
    def load_shards(features_dir: str) -> list:
        """Load + concatenate all shard*.pt records under a seed's feature dir."""
        import torch
        recs = []
        for sp in sorted(glob.glob(f"{features_dir}/shard*.pt")):
            recs.extend(torch.load(sp, weights_only=False, map_location="cpu"))
        return recs


def print_metrics(name: str, metrics: dict, indent: str = "  ") -> None:
    """Pretty-print a (possibly nested) metrics dict returned by `DatasetAdapter.score`."""
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
