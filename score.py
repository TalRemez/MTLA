"""Stage 3 — scoring. Turn the extracted attention shards into benchmark numbers (CPU only).

Reads every ``<features>/seed{K}/shard*.pt`` the extract stage wrote and computes the hallucination
AUROC + the benchmark's task metric. No GPU and no model weights: all the work (band reduction,
self-consistency voting, NMS, metric evaluation) lives in ``mtla.evaluate`` / ``mtla.metrics``. The
rollout seed set is discovered from the features dir, so there is no ``--n``: score votes over
exactly the rollouts that were extracted.

    python -m score --config configs/coco_internvl.yaml
    python -m score --config configs/coco_internvl.yaml --agg sum   # COCO N=16 voting headline
"""

import argparse

from mtla.config import load_config
from mtla.registry import resolve
from mtla.evaluate import run_score
from mtla.data.base import print_metrics


def main() -> None:
    """Parse CLI args and run the scoring stage, printing the benchmark metrics.

    Applies the ``--config``/``--agg`` overrides, resolves the dataset adapter, and
    calls ``mtla.evaluate.run_score``, which loads the extracted shards, reduces and
    votes over the auto-discovered rollouts (no ``--n``), and computes the AUROC plus
    the task metric; the result is then pretty-printed.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument(
        "--agg", default=None, help="override score.agg (max|sum|support|mean)"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.agg is not None:
        cfg.score.agg = args.agg

    _, dataset = resolve(cfg.model, cfg.dataset)
    metrics = run_score(cfg, dataset)
    print_metrics(f"{cfg.dataset}/{cfg.model}", metrics)


if __name__ == "__main__":
    main()
