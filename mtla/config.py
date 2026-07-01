"""Run configuration: one YAML file fully specifies a model x dataset run.

A single ``{generate,extract,score}.py --config <file>.yaml`` invocation reads a ``RunConfig`` and
resolves it to one model adapter (``mtla.models``) and one dataset adapter (``mtla.data``). The
same three stages apply to every benchmark.

The config owns everything that parameterizes a run — including the vision **preprocessing**
knobs (video fps / pixel budget), which used to live on the dataset. Keeping them here makes a
run self-documenting and lets the identical settings feed both the generate and the extract stage.

Example (configs/coco_internvl.yaml)::

    model: internvl            # -> mtla.models.internvl.InternVLAdapter
    dataset: coco              # -> mtla.data.coco.CocoDataset
    paths:
      data: data/coco/coco_val_openvocab_80.json
      coco_gt: data/coco/annotations/instances_val2017.json
      predictions: runs/coco/predictions
      features: runs/coco/features
    n_rollouts: 1              # one knob: generate/extract produce seeds 0..n-1, score votes
    generate: {temperature: 0.7, top_p: 1.0, max_new_tokens: 4096, gpus: null}
    extract:  {gpus: null, n_items: 5000}
    score:    {agg: sum}
    band: [8, 21]              # inclusive middle-layer band; null = all layers
    preprocess: {}             # video runs set {fps, min_pixels, max_pixels}
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import yaml


def all_visible_gpus() -> list:
    """All visible CUDA device indices (respects CUDA_VISIBLE_DEVICES); [0] if CUDA is absent."""
    try:
        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else [0]
    except Exception:
        return [0]


@dataclass
class StageCfg:
    """Per-stage knobs. Not every field applies to every stage; unknown YAML keys are ignored."""
    gpus: list | None = None           # None = all visible GPUs (see RunConfig.stage_gpus)
    n_items: int = 0                   # extract/generate: 0 = all items
    agg: str = "max"                   # score: voting fusion (max | sum | support | mean)
    temperature: float = 0.7           # generate: sampling temperature for stochastic rollouts
    top_p: float = 1.0                 # generate: nucleus sampling top-p
    max_new_tokens: int = 4096         # generate: decode cap
    greedy_seed0: bool | None = None   # generate: rollout 0 is a greedy (T=0) anchor. None = use
                                       # the dataset default (video defaults True — the N=16 recipe).


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
    preprocess: dict = field(default_factory=dict)  # vision preprocessing (video: fps/pixels)
    config_path: str = ""               # absolute path this config was loaded from (set by
                                        # load_config); lets a GPU stage subprocess rebuild the run.

    def band_indices(self):
        """Layer-index list for ``mtla.reduce_band`` (None means all layers)."""
        if self.band is None:
            return None
        lo, hi = self.band
        return list(range(lo, hi + 1))

    def seeds(self) -> list:
        """Rollout seeds to produce / score: 0 .. n_rollouts-1 (one knob drives all stages)."""
        return list(range(max(1, self.n_rollouts)))

    def stage_gpus(self, stage: str) -> list:
        """GPU list for a stage; resolves ``gpus: null`` to all visible GPUs at run time."""
        g = getattr(self, stage).gpus
        return list(g) if g else all_visible_gpus()

    def gen_temperature(self, seed: int, dataset_greedy_seed0: bool = False) -> float:
        """Effective generation temperature for rollout ``seed``. When greedy_seed0 is on, rollout
        0 decodes greedily (T=0) as a deterministic anchor and 1..N-1 sample at ``temperature`` —
        the paper's N=16 recipe. The config value overrides the dataset default."""
        greedy0 = self.generate.greedy_seed0
        if greedy0 is None:
            greedy0 = dataset_greedy_seed0
        return 0.0 if (greedy0 and seed == 0) else self.generate.temperature

    def path(self, key: str) -> str:
        if key not in self.paths:
            raise KeyError(f"config paths is missing '{key}'")
        return os.path.expanduser(self.paths[key])

    def pred_dir(self, seed: int) -> str:
        """Per-rollout predictions dir ``<predictions>/seed{K}/`` (holds predictions.json)."""
        return os.path.join(self.path("predictions"), f"seed{seed}")

    def feat_dir(self, seed: int) -> str:
        """Per-rollout feature-shard dir ``<features>/seed{K}/`` (holds shard*.pt)."""
        return os.path.join(self.path("features"), f"seed{seed}")


def _stage(d: dict | None) -> StageCfg:
    d = dict(d or {})
    known = set(StageCfg.__dataclass_fields__)
    return StageCfg(**{k: v for k, v in d.items() if k in known})


def load_config(path: str) -> RunConfig:
    """Load a ``RunConfig`` from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RunConfig(
        model=raw["model"],
        dataset=raw["dataset"],
        paths=raw.get("paths", {}),
        n_rollouts=raw.get("n_rollouts", 1),
        generate=_stage(raw.get("generate")),
        extract=_stage(raw.get("extract")),
        score=_stage(raw.get("score")),
        band=raw.get("band", [8, 21]),
        preprocess=raw.get("preprocess", {}) or {},
        config_path=os.path.abspath(path),
    )
