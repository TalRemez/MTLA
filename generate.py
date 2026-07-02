"""Stage 1 — generation. vLLM decoding for any model / dataset / modality.

The first stage of the pipeline (then ``extract.py``, then ``evaluate.py``). One config-driven script
for every benchmark; it owns only the modality-agnostic skeleton:

  1. load the run config + resolve the (model, dataset) adapters from the registry;
  2. load the work items (``dataset.load_items(cfg)``) and slice to ``generate.n_items``;
  3. tag each item with a global ``idx`` (deterministic merge order);
  4. pick the execution STRATEGY the dataset declares (``dataset.gen_strategy``) and run it;
  5. sort by idx, write ``<predictions>/seed{K}/predictions.json``.

Everything model/dataset specific lives behind the adapter contract:
  - the MODEL builds vLLM requests: ``gen_processor`` and ``build_vllm_request``
    (see mtla.models.base);
  - the DATASET owns items + the uniform record + the strategy: ``load_items``, ``gen_record``,
    ``gen_strategy``;
  - the two strategies live in ``gen_strategies.py``: ``run_pooled`` (async multi-engine pool,
    throughput on many small requests) and ``run_sharded`` (one engine per GPU, heavy per-item work).

    python -m generate --config configs/coco_internvl.yaml            # 1 rollout
    python -m generate --config configs/coco_internvl.yaml --n 16      # 16 rollouts (seeds 0..15)

Writes ``<predictions>/seed{K}/predictions.json`` (merged + idx-sorted across workers).
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


def generate_seed(cfg, model, dataset, seed: int, tuning: dict) -> None:
    """Generate one rollout for a single ``seed`` and write its predictions.json.

    Loads the dataset work items, tags each with a deterministic ``idx``, resolves
    the sampling temperature for this seed (rollout 0 may be a greedy anchor), then
    dispatches to the execution strategy the dataset declares and writes the merged,
    idx-sorted results to ``<predictions>/seed{seed}/predictions.json``.

    Args:
        cfg: the loaded run config (model/dataset names, paths, sampling knobs).
        model: the resolved model adapter (builds vLLM requests).
        dataset: the resolved dataset adapter (owns items, records, and the
            generation strategy name).
        seed: the rollout index; also seeds vLLM sampling when the model uses it.
        tuning: backend tuning knobs (``tp``, ``max_model_len``, ``gpu_mem_util``,
            ``concurrency``, ``num_cpu_workers``); the pooled path uses all of them,
            the sharded path uses only the sampling knobs.

    Returns:
        None. Side effect: writes ``predictions.json`` for this seed.
    """
    # rollout 0 may be a greedy anchor (T=0) if the dataset/config asks for it; else config temp.
    temperature = cfg.gen_temperature(seed, dataset.greedy_seed0)
    pred_dir = cfg.pred_dir(seed)

    items = dataset.load_items(cfg)  # the dataset owns its file I/O
    if cfg.generate.n_items:
        items = items[: cfg.generate.n_items]
    for i, it in enumerate(items):
        it.setdefault("idx", i)  # tag each item so workers can merge in order

    # Effective sampling knobs (from the config) + backend tuning knobs, handed to the strategy.
    sargs = {
        "seed": seed,
        "temperature": temperature,
        "top_p": cfg.generate.top_p,
        "max_new_tokens": cfg.generate.max_new_tokens,
        **tuning,
    }
    print(
        f"[generate] model={cfg.model} dataset={cfg.dataset} task={dataset.task} "
        f"strategy={dataset.gen_strategy} seed={seed} T={temperature} n={len(items)}",
        flush=True,
    )

    # The dataset declares HOW work is spread over GPUs: an async engine pool for many small
    # requests (images), or one blocking engine per GPU for heavy per-item work (video clips).
    if dataset.gen_strategy == "pooled":
        results = run_pooled(cfg, items, sargs)
    else:
        results = run_sharded(cfg, items, sargs, pred_dir)

    results.sort(
        key=lambda r: r.get("idx", 0)
    )  # restore item order (workers finish out of order)
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(pred_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    n_ok = sum(1 for r in results if r.get("response"))
    print(
        f"[generate] saved {len(results)} predictions ({n_ok} non-empty) -> "
        f"{pred_dir}/predictions.json",
        flush=True,
    )


def main() -> None:
    """Parse CLI args and run the generation stage over every requested seed.

    Applies the ``--config``/``--n``/``--gpus``/``--limit`` overrides, resolves the
    adapters, fetches the model snapshot once in the parent and goes offline (so the
    spawned workers load from the local cache and avoid Hub rate-limiting), then calls
    ``generate_seed`` for each seed in ``0..n-1``.
    """
    ap = argparse.ArgumentParser(description="MTLA generation stage (vLLM).")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="number of rollouts to produce (seeds 0..n-1; default 1)",
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
    # Backend tuning knobs (not part of the run config; tune per box). The pool uses all of them;
    # the sharded path uses only the sampling knobs (temperature/top_p/max_new_tokens/seed).
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

    seeds = cfg.seeds()
    for i, seed in enumerate(seeds):
        print(f"[generate] seed {seed}  ({i + 1}/{len(seeds)})", flush=True)
        generate_seed(cfg, model, dataset, seed, tuning)


if __name__ == "__main__":
    main()
