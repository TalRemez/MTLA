"""Stage 2 — MTLA extraction. HF eager-attention forward for any model / dataset / modality.

Model- and dataset-agnostic: it loads the run config, resolves the (model, dataset) adapters,
shards the generation records across GPUs, and for each record calls ``model.extract_one`` — which
runs one HF-eager captured forward and applies the MTLA math (see ``mtla.mtla_attn``). Every
model/task specific lives behind the adapter's extraction callbacks; this file is just the
multi-GPU harness. The generation records are self-contained (``{id, prompt, response, gt, extra}``),
so a worker just streams ``predictions.json`` — no dataset item lookup.

    python -m extract --config configs/coco_internvl.yaml   # extracts every rollout generate wrote

Reads ``<predictions>/seed{K}/predictions.json``; writes ``<features>/seed{K}/shard{rank}.pt``.
The seed set is discovered from the predictions dir, so there is no ``--n``: extract processes
exactly the rollouts the generate stage produced.
"""

import argparse
import json
import os
from multiprocessing import Process, set_start_method

import numpy as np

from mtla.config import load_config
from mtla.registry import resolve
from mtla.mtla_attn import compute_mtla


def worker(
    rank: int, gpu_id: int, records: list, out_dir: str, config_path: str
) -> None:
    """Extract MTLA features for one shard of records on a single GPU.

    Runs in a spawned subprocess: it pins itself to ``gpu_id`` (via
    ``CUDA_VISIBLE_DEVICES`` before importing torch), loads the model and installs
    the attention-capture hook, then runs one captured HF-eager forward per record
    and applies the MTLA math, saving the shard's results to disk.

    Args:
        rank: the worker's shard index; used to name the output shard file.
        gpu_id: the absolute GPU index this worker pins to (seen as cuda:0 inside).
        records: this shard's slice of generation records (self-contained
            ``{id, prompt, response, gt, extra}`` dicts streamed from predictions.json).
        out_dir: directory to write ``shard{rank}.pt`` into (created if missing).
        config_path: path to the run config, reloaded inside the child to resolve
            the model and dataset adapters.

    Returns:
        None. Side effect: writes ``<out_dir>/shard{rank}.pt`` with the list of
        per-record extraction results (records with no extractable prediction are
        skipped).
    """
    # Pin this worker to ONE GPU before any CUDA use, so it sees the target device as cuda:0. Doing
    # it here (not via torch.cuda.set_device on an absolute id) keeps the run correct even when
    # CUDA_VISIBLE_DEVICES is already set, and mirrors the generate stage. spawn re-imports the
    # module in the child, but `import torch` alone does not initialize CUDA, so this is early enough.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch

    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    # Load the model + processor, install the attention-capture hook, and get the extraction ctx.
    ctx = model.load_for_extract(0, dataset.task)
    if dataset.task == "video_span":
        # Video extraction reads the same preprocessing the generate stage used, plus the
        # multi-span flag (fuse = multi-window benchmark, argmax = single-span).
        ctx["preprocess"] = cfg.preprocess
        ctx["multi"] = dataset.select == "fuse"
    print(
        f"[worker {rank}] model={cfg.model} dataset={cfg.dataset} gpu={gpu_id} "
        f"n={len(records)} L={ctx['n_layers']} H={ctx['n_heads']}",
        flush=True,
    )

    out, n_done = [], 0
    for cnt, rec in enumerate(records):
        # Run MTLA for this item's predictions (one captured forward -> per-prediction [L,H] arrays);
        # `model` just supplies the callbacks compute_mtla drives. None = nothing extractable, skip.
        result = compute_mtla(model, rec, ctx, rank=rank)
        if result is None:
            continue
        out.append(result)
        n_done += 1
        if n_done % 25 == 0:
            print(
                f"[worker {rank}] [{cnt + 1}/{len(records)}] done={n_done}", flush=True
            )

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/shard{rank}.pt"
    torch.save(out, path)
    print(
        f"[worker {rank}] saved {len(out)} records / "
        f"{sum(len(r['objects']) for r in out)} preds -> {path}",
        flush=True,
    )


def extract_seed(cfg, seed: int) -> None:
    """Extract MTLA features for one rollout, sharded across the extract GPUs.

    Reads the seed's ``predictions.json``, splits the records evenly across the
    configured GPUs, spawns one ``worker`` per GPU, and joins them. Fails loudly if
    any worker crashes, since a missing shard would silently produce an incomplete run.

    Args:
        cfg: the loaded run config (paths, extract GPUs, optional item limit).
        seed: the rollout index whose predictions to extract.

    Returns:
        None. Side effect: writes one ``shard{rank}.pt`` per worker under the seed's
        features directory.

    Raises:
        SystemExit: if one or more workers exit with a non-zero code, listing the
            failed ``(rank, exitcode)`` pairs; the shards on disk are incomplete.
    """
    gpus = cfg.stage_gpus("extract")
    records = json.load(open(os.path.join(cfg.pred_dir(seed), "predictions.json")))
    if cfg.extract.n_items:
        records = records[: cfg.extract.n_items]
    out_dir = cfg.feat_dir(seed)

    procs = []
    for rank, (gpu, chunk) in enumerate(zip(gpus, np.array_split(records, len(gpus)))):
        if len(chunk) == 0:
            continue
        p = Process(
            target=worker, args=(rank, gpu, list(chunk), out_dir, cfg.config_path)
        )
        p.start()
        procs.append((rank, p))
    failed = [(rank, p.exitcode) for rank, p in procs if (p.join() or p.exitcode != 0)]
    if failed:
        # a crashed worker leaves its shard unwritten -> a silently incomplete run. Fail loudly.
        raise SystemExit(
            f"[extract] {len(failed)} worker(s) failed (rank, exitcode): {failed}; "
            f"shards under {out_dir} are INCOMPLETE — fix the error and re-run."
        )
    print(f"[extract] all {len(procs)} workers complete (seed {seed})")


def main() -> None:
    """Parse CLI args and run the extraction stage over every rollout on disk.

    Applies the ``--config``/``--gpus``/``--limit`` overrides, fetches the model
    snapshot once in the parent and goes offline (so per-GPU workers load locally and
    avoid Hub rate-limiting), auto-discovers the seed set from the predictions dir
    (there is no ``--n``), then calls ``extract_seed`` for each discovered seed.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.gpus is not None:
        cfg.extract.gpus = args.gpus
    if args.limit is not None:
        cfg.extract.n_items = args.limit

    # Fetch the model ONCE in the parent (weights + config + processor), then go offline so the
    # per-GPU workers load purely from the local snapshot. Otherwise each worker fetches concurrently
    # and transformers pings the Hub even for a cached model -> HTTP 429 rate-limit. snapshot_download
    # (not just the processor) so the offline workers find config.json + weights on disk.
    from huggingface_hub import snapshot_download

    model, _ = resolve(cfg.model, cfg.dataset)
    snapshot_download(model.model_id)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    set_start_method("spawn", force=True)
    seeds = cfg.seeds_on_disk("predictions")  # what generate wrote, not a flag
    if not seeds:
        raise SystemExit(
            f"[extract] no rollouts found under {cfg.path('predictions')}/seed*/ — "
            f"run the generate stage first."
        )
    print(f"[extract] found {len(seeds)} rollout(s) on disk: seeds {seeds}", flush=True)
    for i, seed in enumerate(seeds):
        print(f"[extract] seed {seed}  ({i + 1}/{len(seeds)})", flush=True)
        extract_seed(cfg, seed)
    print(
        f"[extract] done: extracted {len(seeds)} rollout(s) -> {cfg.path('features')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
