"""Stage 1 — generation. vLLM decoding for any model / dataset / modality.

The first stage of the pipeline (then ``extract.py``, then ``evaluate.py``). One config-driven script
for every benchmark; it owns only the modality-agnostic skeleton:

  1. load the run config + resolve the (model, dataset) adapters from the registry;
  2. load the work items (``dataset.load_items(cfg)``) and slice to ``generate.n_items``;
  3. tag each item with a global ``idx`` (deterministic merge order);
  4. build the rollout plan (temperature groups) and run the dataset's STRATEGY once;
  5. write ``<predictions>/rollout{K}/predictions.json`` (idx-sorted) for each rollout K.

All ``n_rollouts`` rollouts are produced in a single pass: the engine is loaded once and each
prompt is prefilled once per temperature (vLLM ``SamplingParams(n=...)`` shares the KV cache across
a temperature group's samples), rather than reloading the engine and re-prefilling per rollout.

Everything model/dataset specific lives behind the adapter contract:
  - the MODEL builds vLLM requests: ``gen_processor`` and ``build_vllm_request``
    (see mtla.models.base);
  - the DATASET owns items + the uniform record + the strategy: ``load_items``, ``gen_record``,
    ``gen_strategy``;
  - the two strategies live in ``gen_strategies.py``: ``run_pooled`` (async multi-engine pool,
    throughput on many small requests) and ``run_sharded`` (one engine per GPU, heavy per-item work).

    python -m generate --config configs/coco_internvl.yaml            # 1 rollout
    python -m generate --config configs/coco_internvl.yaml --n 16      # 16 rollouts (rollout0..15)

Writes ``<predictions>/rollout{K}/predictions.json`` (merged + idx-sorted across workers).
"""

import argparse
import json
import os

import multiprocessing as mp

# Both strategies spawn CUDA subprocesses; the spawn start-method is required.
mp.set_start_method("spawn", force=True)

from gen_strategies import run_pooled, run_sharded

from mtla.config import load_config
from mtla.registry import resolve


def rollout_plan(cfg, dataset) -> list[dict]:
    """Group the ``n_rollouts`` rollouts into vLLM sampling requests by temperature.

    The N=16 recipe is one greedy anchor (rollout 0 at T=0) plus N-1 stochastic rollouts
    at the config temperature. Rollouts that share a temperature are served by ONE vLLM
    request with ``n=<count>``, so the prompt (a whole video for grounding) is prefilled
    once and the KV cache is shared across that request's samples. Each group carries a
    distinct ``seed`` (the greedy and stochastic requests never share a base seed), so a
    rerun reproduces the same batch.

    Args:
        cfg: the run config (``n_rollouts``, ``generate.temperature``/``greedy_seed0``).
        dataset: the dataset adapter (its ``greedy_seed0`` default when the config leaves
            ``generate.greedy_seed0`` unset).

    Returns:
        A list of group dicts ``{"rollouts": [rollout, ...], "temperature": float,
        "seed": int}``, one per distinct temperature. The ``seed`` is the vLLM RNG seed
        (greedy and stochastic groups use different seeds). Each group maps to one vLLM
        request whose ``n`` output samples are written, in order, to those rollout dirs.
    """
    n = max(1, cfg.n_rollouts)
    greedy0 = cfg.generate.greedy_seed0
    if greedy0 is None:
        greedy0 = dataset.greedy_seed0

    groups: list[dict] = []
    if greedy0:
        # Rollout 0 is the deterministic T=0 anchor: a single sample (n=1 for T=0).
        groups.append({"rollouts": [0], "temperature": 0.0, "seed": 0})
        stochastic = list(range(1, n))
    else:
        stochastic = list(range(n))
    if stochastic:
        # All stochastic rollouts share ONE request (n=len) at the config temperature, with a
        # seed distinct from the greedy group's.
        groups.append(
            {
                "rollouts": stochastic,
                "temperature": cfg.generate.temperature,
                "seed": 1,
            }
        )
    return groups


def write_predictions(cfg, records: list) -> None:
    """Split the flat, rollout-tagged records into per-rollout ``predictions.json`` files.

    Each record carries a ``rollout`` (which rollout dir it belongs to) and an ``idx``
    (item order). Records are grouped by rollout and, within each, sorted by ``idx`` so
    every ``<predictions>/rollout{K}/predictions.json`` is written in a deterministic order.

    Args:
        cfg: the run config (resolves ``pred_dir(rollout)``).
        records: the merged strategy output; each record has ``rollout`` and ``idx``.

    Returns:
        None. Side effect: writes one ``predictions.json`` per rollout.
    """
    by_rollout: dict[int, list] = {}
    for r in records:
        by_rollout.setdefault(r.pop("rollout"), []).append(r)
    for rollout, recs in sorted(by_rollout.items()):
        recs.sort(key=lambda r: r.get("idx", 0))
        pred_dir = cfg.pred_dir(rollout)
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, "predictions.json"), "w") as f:
            json.dump(recs, f, indent=2)
        n_ok = sum(1 for r in recs if r.get("response"))
        print(
            f"[generate] rollout {rollout}: saved {len(recs)} predictions ({n_ok} non-empty) "
            f"-> {pred_dir}/predictions.json",
            flush=True,
        )


def generate_all(cfg, model, dataset, tuning: dict) -> None:
    """Generate every rollout in one pass and write the per-rollout predictions.

    Loads the work items once, tags each with a deterministic ``idx``, builds the rollout
    plan (temperature groups), dispatches to the dataset's execution strategy a single
    time (the engine is loaded once and each prompt is prefilled once per temperature),
    then writes one ``predictions.json`` per rollout.

    Args:
        cfg: the loaded run config (model/dataset names, paths, sampling knobs).
        model: the resolved model adapter (builds vLLM requests).
        dataset: the resolved dataset adapter (owns items, records, and the strategy).
        tuning: backend tuning knobs (``tp``, ``concurrency``, ``num_cpu_workers``); the
            pooled path uses all of them, the sharded path none.

    Returns:
        None. Side effect: writes ``<predictions>/rollout{K}/predictions.json`` for each K.
    """
    items = dataset.load_items(cfg)  # the dataset owns its file I/O
    if cfg.generate.n_items:
        items = items[: cfg.generate.n_items]
    for i, it in enumerate(items):
        it.setdefault("idx", i)  # tag each item so workers can merge in order

    groups = rollout_plan(cfg, dataset)
    sargs = {
        "groups": groups,
        "top_p": cfg.generate.top_p,
        "max_new_tokens": cfg.generate.max_new_tokens,
        **tuning,
    }
    print(
        f"[generate] model={cfg.model} dataset={cfg.dataset} task={dataset.task} "
        f"strategy={dataset.gen_strategy} n_items={len(items)} groups={groups}",
        flush=True,
    )

    # The dataset declares HOW work is spread over GPUs: an async engine pool for many small
    # requests (images), or one blocking engine per GPU for heavy per-item work (video clips).
    if dataset.gen_strategy == "pooled":
        records = run_pooled(cfg, items, sargs)
    else:
        records = run_sharded(cfg, items, sargs, cfg.path("predictions"))

    write_predictions(cfg, records)


def main() -> None:
    """Parse CLI args and run the generation stage for all rollouts in one pass.

    Applies the ``--config``/``--n``/``--gpus``/``--limit`` overrides, resolves the
    adapters, fetches the model snapshot once in the parent and goes offline (so the
    spawned workers load from the local cache and avoid Hub rate-limiting), then calls
    ``generate_all`` once (which loads the engine once and produces every rollout).
    """
    ap = argparse.ArgumentParser(description="MTLA generation stage (vLLM).")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="number of rollouts to produce (rollout 0..n-1; default 1)",
    )
    ap.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU indices to run on (default: all visible GPUs)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run on only the first N items (e.g. 100 for a quick test); "
        "default: the full set",
    )
    # Backend tuning knobs (not part of the run config; tune per box). Only the pooled path uses
    # these; the sampling knobs (temperature/top_p/max_new_tokens/seed) come from the rollout plan.
    ap.add_argument(
        "--tp",
        type=int,
        default=1,
        help="vLLM tensor-parallel size per engine (pooled)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="pooled: in-flight requests per engine",
    )
    ap.add_argument(
        "--num_cpu_workers", type=int, default=16, help="pooled: request-prep workers"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n is not None:
        cfg.n_rollouts = args.n
    if args.gpus is not None:
        cfg.generate.gpus = args.gpus
    if args.limit is not None:
        cfg.generate.n_items = args.limit
    model, dataset = resolve(cfg.model, cfg.dataset)
    tuning = {
        "tp": args.tp,
        "concurrency": args.concurrency,
        "num_cpu_workers": args.num_cpu_workers,
    }

    # Fetch the model ONCE in the parent (weights + config + processor), then go offline so the
    # spawned workers load purely from the local snapshot. Without this each worker fetches
    # independently: transformers pings the Hub even for a cached model, so 8+ concurrent workers get
    # the IP rate-limited (HTTP 429). snapshot_download (not just the processor) is required because
    # vLLM, once offline, resolves model_id to the local snapshot dir and needs config.json + weights.
    from huggingface_hub import snapshot_download

    snapshot_download(model.model_id)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    generate_all(cfg, model, dataset, tuning)


if __name__ == "__main__":
    main()
