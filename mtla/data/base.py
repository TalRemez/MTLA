"""Dataset adapter interface.

A dataset adapter holds everything that depends on the *benchmark*: how to load its items,
the prompt to ask, the ground truth, and how to turn extracted attention into the benchmark's
metrics (hallucination AUROC plus the task metric: COCO mAP, QVHighlights mAP/R@1, Charades
R@1@IoU). It owns the voting/fusion choice for its task (sum for COCO, max for video).

Scoring is CPU-only and reads the feature shards written by the extract stage; it does not
need the model. `run.py --stage score` calls `score()`.
"""
from __future__ import annotations


class DatasetAdapter:
    """Base class for benchmark adapters."""

    name: str = ""

    def load_items(self, cfg) -> list:
        """Load the list of work items (images or video queries) for generation/extraction."""
        raise NotImplementedError

    def prompt(self, item) -> str:
        """The task prompt for one item."""
        raise NotImplementedError

    def ground_truth(self, item):
        """Ground-truth regions/labels for one item (used to label hallucinations + metric)."""
        raise NotImplementedError

    # task family ("image_det" | "video_span"); used to ask the model adapter for the right
    # stage scripts and signal slots. Subclasses set this.
    task: str = ""

    def generate(self, cfg, model, seed=0):
        """GPU stage: run the model to produce predictions for one rollout `seed`,
        writing seed{seed}/. Dataset-specific stage script."""
        raise NotImplementedError(f"{type(self).__name__} has no generate stage")

    def extract(self, cfg, model, seed=0):
        """GPU stage: HF-eager pass capturing attention into seed{seed}/ feature shards."""
        raise NotImplementedError(f"{type(self).__name__} has no extract stage")

    def score(self, cfg, model) -> dict:
        """Compute hallucination AUROC and the task metric from the run's feature shards.

        Reads `cfg.path('features')` (and predictions / GT as needed), applies the layer-band
        reduction (`cfg.band_indices()`), the dataset's fusion (`cfg.score.agg`), and the model
        adapter's signal slots. Returns a dict of metrics; prints a summary as a side effect.
        """
        raise NotImplementedError
