"""Run configuration: a YAML file fully specifies one model x dataset run.

A single ``run.py --config <file>.yaml --stage <generate|extract|score>`` invocation reads a
``RunConfig`` from YAML and resolves it to one model adapter (`mtla.models`) and one dataset
adapter (`mtla.data`). The same three stages then apply to every benchmark.

Example (configs/coco_internvl.yaml):

    model: internvl          # -> mtla.models.internvl.InternVLAdapter
    dataset: coco            # -> mtla.data.coco.CocoDataset
    paths:
      data: data/coco/coco_val_openvocab_80.json     # repo-relative; see scripts/prepare_coco.py
      coco_gt: data/coco/instances_val2017.json
      predictions: runs/coco/predictions
      features: runs/coco/features
    n_rollouts: 1            # one number drives everything: generate/extract produce seeds
                             # 0..n_rollouts-1, score votes over them
    generate: {engine: vllm, gpus: null}             # gpus: null = all visible GPUs
    extract:  {gpus: null, n_items: 5000}
    score:    {agg: sum}
    band: [8, 21]            # inclusive layer band; null = all layers
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def all_visible_gpus() -> list:
    """All visible CUDA device indices (respects CUDA_VISIBLE_DEVICES); [0] if CUDA is absent."""
    try:
        import torch
        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else [0]
    except Exception:
        return [0]


@dataclass
class StageCfg:
    engine: str = "hf"                 # generate engine: "hf" or "vllm"
    gpus: list | None = None           # None = all visible GPUs (see RunConfig.stage_gpus)
    n_items: int = 0                   # 0 = all
    agg: str = "max"                   # voting fusion: max | sum | support | mean
    temperature: float = 0.7           # sampling temperature for stochastic rollouts
    greedy_seed0: bool | None = None   # generate: rollout 0 is a greedy (T=0) anchor, 1..N-1
                                       # sample at `temperature`. None = use the dataset default
                                       # (video benchmarks default True — the paper's N=16 recipe).


@dataclass
class RunConfig:
    model: str                          # adapter key in mtla.models
    dataset: str                        # adapter key in mtla.data
    paths: dict = field(default_factory=dict)
    n_rollouts: int = 1                 # the single rollout knob: generate/extract produce seeds
                                        # 0..n_rollouts-1; score votes over the same range.
    generate: StageCfg = field(default_factory=StageCfg)
    extract: StageCfg = field(default_factory=StageCfg)
    score: StageCfg = field(default_factory=StageCfg)
    band: list | None = None            # [lo, hi] inclusive, or None for all layers
    config_path: str = ""               # absolute path this config was loaded from (set by
                                        # load_config); lets a GPU stage subprocess rebuild the
                                        # config and ask the dataset adapter for its items.

    def band_indices(self):
        """Return the layer-index list for `mtla.reduce_band` (None means all layers)."""
        if self.band is None:
            return None
        lo, hi = self.band
        return list(range(lo, hi + 1))

    def seeds(self) -> list:
        """Rollout seeds to produce / score: 0 .. n_rollouts-1 (one knob drives all stages)."""
        return list(range(max(1, self.n_rollouts)))

    def stage_gpus(self, stage: str) -> list:
        """GPU list for a stage; resolves `gpus: null` to all visible GPUs at run time."""
        g = getattr(self, stage).gpus
        return list(g) if g else all_visible_gpus()

    def gen_temperature(self, seed: int, dataset_greedy_seed0: bool = False) -> float:
        """Effective generation temperature for a rollout `seed`. If greedy_seed0 is on (config
        overrides the dataset default), rollout 0 decodes greedily (T=0) as a deterministic anchor
        and rollouts 1..N-1 sample at `temperature` — the paper's N=16 recipe. Video benchmarks
        default to greedy_seed0=True; images to False."""
        greedy0 = self.generate.greedy_seed0
        if greedy0 is None:
            greedy0 = dataset_greedy_seed0
        return 0.0 if (greedy0 and seed == 0) else self.generate.temperature

    def path(self, key: str) -> str:
        if key not in self.paths:
            raise KeyError(f"config paths is missing '{key}'")
        return os.path.expanduser(self.paths[key])

    def pred_dir(self, seed: int) -> str:
        """Per-rollout predictions dir `<predictions>/seed{K}/` (holds predictions.json)."""
        return os.path.join(self.path("predictions"), f"seed{seed}")

    def feat_dir(self, seed: int) -> str:
        """Per-rollout feature-shard dir `<features>/seed{K}/` (holds shard*.pt)."""
        return os.path.join(self.path("features"), f"seed{seed}")


def _stage(d: dict | None) -> StageCfg:
    d = dict(d or {})
    known = {f for f in StageCfg.__dataclass_fields__}
    # Keep only recognized keys; ignore unknown ones (forgiving to stray/legacy YAML keys).
    base = {k: v for k, v in d.items() if k in known}
    return StageCfg(**base)


def load_config(path: str) -> RunConfig:
    """Load a RunConfig from a YAML file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    # n_rollouts is top-level now; accept a legacy score.n_rollouts as a fallback.
    n_rollouts = raw.get("n_rollouts", (raw.get("score") or {}).get("n_rollouts", 1))
    return RunConfig(
        model=raw["model"],
        dataset=raw["dataset"],
        paths=raw.get("paths", {}),
        n_rollouts=n_rollouts,
        generate=_stage(raw.get("generate")),
        extract=_stage(raw.get("extract")),
        score=_stage(raw.get("score")),
        band=raw.get("band", [8, 21]),
        config_path=os.path.abspath(path),
    )
