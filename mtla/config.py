"""Run configuration: a YAML file fully specifies one model x dataset run.

A single ``run.py --config <file>.yaml --stage <generate|extract|score>`` invocation reads a
``RunConfig`` from YAML and resolves it to one model adapter (`mtla.models`) and one dataset
adapter (`mtla.data`). The same three stages then apply to every benchmark.

Example (configs/coco_internvl.yaml):

    model: internvl          # -> mtla.models.internvl.InternVLAdapter
    dataset: coco            # -> mtla.data.coco.CocoDataset
    paths:
      data: /data/coco_val_openvocab_80.json
      coco_gt: /data/instances_val2017.json
      predictions: runs/coco/predictions
      features: runs/coco/features
    generate: {engine: vllm, gpus: [0,1,2,3,4,5,6,7]}
    extract:  {gpus: [0,1,2,3,4,5,6,7], n_items: 5000}
    score:    {n_rollouts: 16, agg: sum, slot: attn_coord_mean}
    band: [8, 21]            # inclusive layer band; null = all layers
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class StageCfg:
    engine: str = "hf"                 # generate engine: "hf" or "vllm"
    gpus: list = field(default_factory=lambda: [0])
    n_items: int = 0                   # 0 = all
    n_rollouts: int = 1
    agg: str = "max"                   # voting fusion: max | sum | support | mean
    slot: str = "first_digit"          # attention slot (model-specific meaning)
    temperature: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    model: str                          # adapter key in mtla.models
    dataset: str                        # adapter key in mtla.data
    paths: dict = field(default_factory=dict)
    generate: StageCfg = field(default_factory=StageCfg)
    extract: StageCfg = field(default_factory=StageCfg)
    score: StageCfg = field(default_factory=StageCfg)
    band: list | None = None            # [lo, hi] inclusive, or None for all layers

    def band_indices(self):
        """Return the layer-index list for `mtla.reduce_band` (None means all layers)."""
        if self.band is None:
            return None
        lo, hi = self.band
        return list(range(lo, hi + 1))

    def path(self, key: str) -> str:
        if key not in self.paths:
            raise KeyError(f"config paths is missing '{key}'")
        return os.path.expanduser(self.paths[key])


def _stage(d: dict | None) -> StageCfg:
    d = dict(d or {})
    known = {f for f in StageCfg.__dataclass_fields__}
    extra = {k: v for k, v in d.items() if k not in known}
    base = {k: v for k, v in d.items() if k in known}
    return StageCfg(extra=extra, **base)


def load_config(path: str) -> RunConfig:
    """Load a RunConfig from a YAML file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return RunConfig(
        model=raw["model"],
        dataset=raw["dataset"],
        paths=raw.get("paths", {}),
        generate=_stage(raw.get("generate")),
        extract=_stage(raw.get("extract")),
        score=_stage(raw.get("score")),
        band=raw.get("band", [8, 21]),
    )
