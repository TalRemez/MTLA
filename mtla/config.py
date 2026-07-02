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
    generate: {temperature: 0.7, top_p: 1.0, max_new_tokens: 4096}
    score:    {agg: sum}
    band: [8, 21]              # inclusive middle-layer band; null = all layers
    preprocess: {}             # video runs set {fps, min_pixels, max_pixels}

The rollout count, GPUs, and item limit are set at launch, not in the config: ``--n``
(n_rollouts), ``--gpus``, and ``--limit`` (first N items; default: all) on generate.py /
extract.py (defaults: 1 rollout, all visible GPUs, all items).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import yaml


def all_visible_gpus() -> list:
    """List the CUDA device indices this process can use.

    Reads the count from torch (which already honors ``CUDA_VISIBLE_DEVICES``) and
    falls back to a single logical device when CUDA is unavailable, so the CPU-only
    score path still gets a usable list. Used to resolve ``gpus: null`` at run time.

    Returns:
        Device indices ``[0, ..., N-1]`` when CUDA reports N > 0 devices, otherwise
        ``[0]`` (also the fallback if the torch query raises).
    """
    try:
        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else [0]
    except Exception:
        return [0]


@dataclass
class StageCfg:
    """Per-stage knobs for one of the generate / extract / score stages.

    Holds the settings that can differ between stages within a single run. Not every
    field applies to every stage: the ``temperature`` / ``top_p`` / ``max_new_tokens``
    / ``greedy_seed0`` group is decode-time and only used by generate, while ``agg`` is
    the voting-fusion mode read by score. Unknown YAML keys are dropped when the config
    loads (see ``_stage``), so a stage block may set only the fields it cares about.

    Non-obvious fields: ``gpus=None`` means "all visible GPUs" (resolved lazily by
    ``RunConfig.stage_gpus``); ``n_items=0`` means "all items" and is normally set at
    launch via ``--limit`` rather than in the YAML; ``greedy_seed0=None`` defers to the
    dataset default (video defaults to True, making rollout 0 a greedy T=0 anchor in
    the N=16 recipe).
    """

    gpus: list | None = None  # None = all visible GPUs (see RunConfig.stage_gpus)
    n_items: int = 0  # 0 = all items; set at launch via --limit, not in configs
    agg: str = "max"  # score: voting fusion (max | sum | support | mean)
    temperature: float = 0.7  # generate: sampling temperature for stochastic rollouts
    top_p: float = 1.0  # generate: nucleus sampling top-p
    max_new_tokens: int = 4096  # generate: decode cap
    greedy_seed0: bool | None = (
        None  # generate: rollout 0 is a greedy (T=0) anchor. None = use
    )
    # the dataset default (video defaults True — the N=16 recipe).


@dataclass
class RunConfig:
    """The full specification of one model x dataset run, loaded from a YAML file.

    One ``RunConfig`` resolves (via ``mtla.registry.resolve``) to a single model
    adapter and a single dataset adapter, and carries every parameter the three stages
    need: the ``paths`` map, the three ``StageCfg`` blocks, the vision ``preprocess``
    knobs (video fps / pixel budget), the layer ``band``, and the rollout count. Owning
    the preprocessing knobs here (rather than on the dataset) makes a run
    self-documenting and feeds identical settings to both generate and extract.

    Non-obvious fields: ``band`` is an inclusive ``[lo, hi]`` layer range (default the
    L8-21 middle band) or ``None`` for all layers; ``n_rollouts`` is the single knob for
    self-consistency (stages produce/consume seeds ``0 .. n_rollouts-1``);
    ``config_path`` is the absolute path the config was loaded from, stored so a GPU
    stage subprocess can rebuild the same run.
    """

    model: str  # adapter key in mtla.models
    dataset: str  # adapter key in mtla.data
    paths: dict = field(default_factory=dict)
    n_rollouts: int = 1  # the single rollout knob: generate/extract produce seeds
    # 0..n_rollouts-1; score votes over the same range.
    generate: StageCfg = field(default_factory=StageCfg)
    extract: StageCfg = field(default_factory=StageCfg)
    score: StageCfg = field(default_factory=StageCfg)
    band: list | None = None  # [lo, hi] inclusive, or None for all layers
    preprocess: dict = field(
        default_factory=dict
    )  # vision preprocessing (video: fps/pixels)
    config_path: str = ""  # absolute path this config was loaded from (set by
    # load_config); lets a GPU stage subprocess rebuild the run.

    def band_indices(self) -> list[int] | None:
        """Expand the ``band`` range into the explicit layer indices to reduce over.

        Turns the stored inclusive ``[lo, hi]`` band (the L8-21 middle band by default)
        into the list of layer indices ``mtla.score.reduce_band`` selects when reducing
        an ``[L, H]`` attention array to a scalar.

        Returns:
            The inclusive index list ``[lo, ..., hi]``, or ``None`` when ``band`` is
            ``None`` (meaning reduce over all layers).
        """
        if self.band is None:
            return None
        lo, hi = self.band
        return list(range(lo, hi + 1))

    def seeds(self) -> list:
        """Rollout seeds this run should produce.

        The generate/extract stages iterate over these seeds to write one rollout
        directory each; ``--n`` on generate sets ``n_rollouts``, which drives this.

        Returns:
            ``[0, ..., n_rollouts-1]`` (always at least ``[0]``, since a run has one
            rollout minimum).
        """
        return list(range(max(1, self.n_rollouts)))

    def _seeds_on_disk(self, root_key: str) -> list:
        """Discover the rollout seeds a previous stage actually wrote to disk.

        Scans ``<root_key>/seed{K}/`` directories and parses their integer seeds, so
        extract and score infer their input seed set from what exists rather than from
        an ``--n`` flag.

        Args:
            root_key: the ``paths`` key of the directory to scan (``"predictions"`` or
                ``"features"``).

        Returns:
            The discovered seed integers, sorted ascending (empty if none exist).
        """
        import glob
        import re

        root = self.path(root_key)
        found = []
        for d in glob.glob(os.path.join(root, "seed*")):
            m = re.fullmatch(r"seed(\d+)", os.path.basename(d))
            if m and os.path.isdir(d):
                found.append(int(m.group(1)))
        return sorted(found)

    def predicted_seeds(self) -> list:
        """Rollout seeds that have a predictions directory on disk.

        This is the input set the extract stage consumes, discovered from what generate
        wrote under ``<predictions>/seed{K}/``.

        Returns:
            The sorted seed integers with a predictions directory (empty if none).
        """
        return self._seeds_on_disk("predictions")

    def extracted_seeds(self) -> list:
        """Rollout seeds that have a features directory on disk.

        This is the input set the score stage consumes, discovered from what extract
        wrote under ``<features>/seed{K}/``.

        Returns:
            The sorted seed integers with a features directory (empty if none).
        """
        return self._seeds_on_disk("features")

    def stage_gpus(self, stage: str) -> list:
        """Resolve the GPU list a stage should run on.

        Reads the stage's ``gpus`` field and expands a ``None``/empty value to all
        visible GPUs at run time, so a config can leave GPU selection to the launch
        environment.

        Args:
            stage: which stage to look up (``"generate"``, ``"extract"``, or
                ``"score"``).

        Returns:
            The explicit device-index list from the stage config, or all visible GPUs
            when it is unset.
        """
        g = getattr(self, stage).gpus
        return list(g) if g else all_visible_gpus()

    def gen_temperature(self, seed: int, dataset_greedy_seed0: bool = False) -> float:
        """Effective decode temperature for one rollout.

        Implements the paper's N=16 recipe: when greedy-seed0 is on, rollout 0 decodes
        greedily (T=0) as a deterministic anchor and rollouts 1..N-1 sample at the
        configured ``temperature``. The config's ``greedy_seed0`` overrides the dataset
        default when set.

        Args:
            seed: the rollout index being generated.
            dataset_greedy_seed0: the dataset's default for greedy-seed0, used only when
                the config leaves ``generate.greedy_seed0`` as ``None``.

        Returns:
            ``0.0`` when greedy-seed0 is in effect and ``seed == 0``, otherwise the
            configured sampling temperature.
        """
        greedy0 = self.generate.greedy_seed0
        if greedy0 is None:
            greedy0 = dataset_greedy_seed0
        return 0.0 if (greedy0 and seed == 0) else self.generate.temperature

    def path(self, key: str) -> str:
        """Look up a configured path by key, with ``~`` expanded.

        Central accessor for the ``paths`` map so every stage reads paths the same way
        and a missing entry fails loudly instead of silently defaulting.

        Args:
            key: the ``paths`` entry to fetch (e.g. ``"data"``, ``"predictions"``,
                ``"features"``, ``"coco_gt"``).

        Returns:
            The configured path with the user home (``~``) expanded.

        Raises:
            KeyError: if ``key`` is not present in ``paths``.
        """
        if key not in self.paths:
            raise KeyError(f"config paths is missing '{key}'")
        return os.path.expanduser(self.paths[key])

    def pred_dir(self, seed: int) -> str:
        """Directory holding one rollout's predictions.

        Args:
            seed: the rollout index.

        Returns:
            The path ``<predictions>/seed{seed}/``, which holds that rollout's
            ``predictions.json``.
        """
        return os.path.join(self.path("predictions"), f"seed{seed}")

    def feat_dir(self, seed: int) -> str:
        """Directory holding one rollout's feature shards.

        Args:
            seed: the rollout index.

        Returns:
            The path ``<features>/seed{seed}/``, which holds that rollout's
            ``shard*.pt`` files.
        """
        return os.path.join(self.path("features"), f"seed{seed}")


def _stage(d: dict | None) -> StageCfg:
    """Build a ``StageCfg`` from a raw YAML stage block, dropping unknown keys.

    Filters the mapping to the dataclass's declared fields before constructing it, so a
    stray or misspelled YAML key is ignored rather than raising ``TypeError``.

    Args:
        d: the raw stage mapping from the YAML (``None`` is treated as empty).

    Returns:
        A ``StageCfg`` populated from the recognized keys, defaults elsewhere.
    """
    d = dict(d or {})
    known = set(StageCfg.__dataclass_fields__)
    return StageCfg(**{k: v for k, v in d.items() if k in known})


def load_config(path: str) -> RunConfig:
    """Parse a YAML file into a fully-populated ``RunConfig``.

    Reads the YAML, coerces each of the three stage blocks through ``_stage``, applies
    defaults (notably the L8-21 ``band`` and one rollout), and records the file's
    absolute path in ``config_path`` so a GPU stage subprocess can rebuild the run.

    Args:
        path: filesystem path to the run's YAML config.

    Returns:
        The assembled ``RunConfig``.

    Raises:
        KeyError: if the YAML is missing a required top-level key (``model`` or
            ``dataset``).
    """
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
