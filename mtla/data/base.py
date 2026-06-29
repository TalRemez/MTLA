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

    def generate(self, cfg, model):
        """GPU stage: run the model to produce predictions. Dataset-specific stage script."""
        raise NotImplementedError(f"{type(self).__name__} has no generate stage")

    def extract(self, cfg, model):
        """GPU stage: HF-eager pass capturing attention into feature shards."""
        raise NotImplementedError(f"{type(self).__name__} has no extract stage")

    def score(self, cfg) -> dict:
        """Compute hallucination AUROC and the task metric from the run's feature shards.

        Reads `cfg.path('features')` (and predictions / GT as needed), applies the layer-band
        reduction (`cfg.band_indices()`) and the dataset's fusion (`cfg.score.agg`), and
        returns a dict of metrics. Prints a human-readable summary as a side effect.
        """
        raise NotImplementedError
