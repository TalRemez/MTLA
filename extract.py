"""Stage 2 — MTLA extraction. HF eager-attention forward for any model / dataset / modality.

Model- and dataset-agnostic: it loads the run config, resolves the (model, dataset) adapters,
shards the generation records across GPUs, and for each record calls ``model.extract_one`` — which
runs one HF-eager captured forward and applies the MTLA math (see ``mtla.mtla_attn``). Every
model/task specific lives behind the adapter's extraction callbacks; this file is just the
multi-GPU harness. The generation records are self-contained (``{id, prompt, response, gt, extra}``),
so a worker just streams ``predictions.json`` — no dataset item lookup.

    python -m extract --config configs/coco_internvl.yaml   # extracts every rollout generate wrote

Reads ``<predictions>/rollout{K}/predictions.json``; writes
``<features>/rollout{K}/shard{rank}.pt``. The rollout set is discovered from the predictions dir,
so there is no ``--n``: extract processes exactly the rollouts the generate stage produced.
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
    rank: int, gpu_id: int, n_workers: int, rollouts: list, config_path: str
) -> None:
    """Extract MTLA features for one GPU's shard across every rollout.

    Runs in a spawned subprocess: it pins itself to ``gpu_id`` (via
    ``CUDA_VISIBLE_DEVICES`` before importing torch), loads the model and installs the
    attention-capture hook **once**, then loops over all ``rollouts`` — reading each
    rollout's predictions, taking its ``rank``-th shard of records, running one captured
    HF-eager forward per record, and saving that rollout's shard to disk. Loading the
    model once and iterating rollouts (rather than reloading per rollout) is the whole
    point of this shape.

    Args:
        rank: the worker's shard index; selects its slice of each rollout's records and
            names the output shard file.
        gpu_id: the absolute GPU index this worker pins to (seen as cuda:0 inside).
        n_workers: total number of workers, so the per-rollout record split
            (``np.array_split(records, n_workers)[rank]``) matches across workers.
        rollouts: every rollout number to extract, in order.
        config_path: path to the run config, reloaded inside the child to resolve the
            model and dataset adapters and the per-rollout paths.

    Returns:
        None. Side effect: writes ``<features>/rollout{K}/shard{rank}.pt`` for each
        rollout K whose shard for this rank is non-empty.

    Raises:
        RuntimeError: if a rollout's shard extracts nothing from a non-trivial number of
            records (a systemic failure, e.g. a wrong data dir or broken path scheme).
    """
    # Pin this worker to ONE GPU before any CUDA use, so it sees the target device as cuda:0. Doing
    # it here (not via torch.cuda.set_device on an absolute id) keeps the run correct even when
    # CUDA_VISIBLE_DEVICES is already set, and mirrors the generate stage. spawn re-imports the
    # module in the child, but `import torch` alone does not initialize CUDA, so this is early enough.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch

    cfg = load_config(config_path)
    model, dataset = resolve(cfg.model, cfg.dataset)
    # Load the model + processor and install the attention-capture hook ONCE, reuse across rollouts.
    ctx = model.load_for_extract(0)
    if dataset.task == "video_span":
        # Video extraction reads the same preprocessing the generate stage used, plus the
        # multi-span flag (fuse = multi-window benchmark, argmax = single-span).
        ctx["preprocess"] = cfg.preprocess
        ctx["multi"] = dataset.select == "fuse"
    print(
        f"[worker {rank}] model={cfg.model} dataset={cfg.dataset} gpu={gpu_id} "
        f"rollouts={rollouts} L={ctx['n_layers']} H={ctx['n_heads']}",
        flush=True,
    )

    for rollout in rollouts:
        records = json.load(
            open(os.path.join(cfg.pred_dir(rollout), "predictions.json"))
        )
        chunk = list(np.array_split(records, n_workers)[rank])
        if (
            not chunk
        ):  # more workers than records: nothing for this rank in this rollout
            continue

        oom_before = ctx["state"].n_oom
        out, n_done = [], 0
        for cnt, rec in enumerate(chunk):
            # Run MTLA for this item's predictions (one captured forward -> per-prediction [L,H]
            # arrays). None = nothing extractable OR this item OOM'd (state.n_oom bumped); skip.
            result = compute_mtla(model, rec, ctx, rank=rank)
            if result is None:
                continue
            out.append(result)
            n_done += 1
            if n_done % 25 == 0:
                print(
                    f"[worker {rank}] rollout {rollout} [{cnt + 1}/{len(chunk)}] done={n_done}",
                    flush=True,
                )

        # An OOM on one huge-attention item is tolerated (it is skipped), but many OOMs mean the run
        # is mis-sized (max_model_len too high, batch too big) rather than a few outliers — fail loud
        # so the user fixes it instead of getting a silently gutted shard.
        n_oom = ctx["state"].n_oom - oom_before
        oom_cap = max(3, len(chunk) // 10)
        if n_oom > oom_cap:
            raise RuntimeError(
                f"[worker {rank}] rollout {rollout}: {n_oom} of {len(chunk)} items OOM'd "
                f"(cap {oom_cap}); the run is mis-sized (lower max_model_len / batch), not a few "
                f"outliers."
            )

        # A few skipped items (missing file, unparseable response, occasional OOM) are tolerated, but
        # a shard that extracted NOTHING from a non-trivial number of records is systemic (wrong data
        # dir, every clip missing, a broken path scheme) — fail loud rather than write empty silently.
        if not out and len(chunk) > 1:
            raise RuntimeError(
                f"[worker {rank}] rollout {rollout}: extracted 0 of {len(chunk)} records; this is "
                f"a systemic failure (check the media paths / data dir), not a few missing examples."
            )

        out_dir = cfg.feat_dir(rollout)
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/shard{rank}.pt"
        torch.save(out, path)
        print(
            f"[worker {rank}] rollout {rollout}: saved {len(out)} records / "
            f"{sum(len(r['objects']) for r in out)} preds -> {path}",
            flush=True,
        )


def main() -> None:
    """Parse CLI args and run the extraction stage over every rollout on disk.

    Applies the ``--config``/``--gpus`` overrides, fetches the model snapshot once in
    the parent and goes offline (so per-GPU workers load locally and avoid Hub
    rate-limiting), auto-discovers the rollout set from the predictions dir (there is no
    ``--n`` or ``--limit``: extract processes exactly what generate wrote), then spawns one
    worker per GPU that loads the model once and extracts its shard across every rollout.
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.gpus is not None:
        cfg.extract.gpus = args.gpus

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
    rollouts = cfg.rollouts_on_disk("predictions")  # what generate wrote, not a flag
    if not rollouts:
        raise SystemExit(
            f"[extract] no rollouts found under {cfg.path('predictions')}/rollout*/ — "
            f"run the generate stage first."
        )
    print(f"[extract] found {len(rollouts)} rollout(s) on disk: {rollouts}", flush=True)

    # One worker per GPU, spawned once: each loads the model a single time and extracts its shard
    # across every rollout (vs. respawning per rollout, which reloaded the model N times per GPU).
    gpus = cfg.stage_gpus("extract")
    n_workers = len(gpus)
    procs = []
    for rank, gpu in enumerate(gpus):
        p = Process(
            target=worker, args=(rank, gpu, n_workers, rollouts, cfg.config_path)
        )
        p.start()
        procs.append((rank, p))
    failed = [(rank, p.exitcode) for rank, p in procs if (p.join() or p.exitcode != 0)]
    if failed:
        # a crashed worker leaves its shards unwritten -> a silently incomplete run. Fail loudly.
        raise SystemExit(
            f"[extract] {len(failed)} worker(s) failed (rank, exitcode): {failed}; "
            f"shards under {cfg.path('features')} are INCOMPLETE — fix the error and re-run."
        )
    print(
        f"[extract] done: extracted {len(rollouts)} rollout(s) -> {cfg.path('features')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
