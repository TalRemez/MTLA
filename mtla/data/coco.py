"""COCO detection dataset adapter (open-vocabulary, val2017).

Declarative: it loads items, builds the prompt + ground truth, and emits the uniform generation
record. All scoring (band reduction, NMS voting, mAP) is done by ``mtla.evaluate`` /
``mtla.metrics`` per the descriptors below.

Reproduces (InternVL3.5-8B): hallucination AUROC 0.873 (MTLA); detection mAP 41.9 at N=16.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .base import DatasetAdapter
from ..registry import register_dataset
from ..types import GenRecord, GTRegion

if TYPE_CHECKING:
    from ..config import RunConfig

# 80 COCO class names (the open-vocab detection prompt lists these).
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

PROMPT = (
    "Please detect all instances of {cats} in the image. "
    "Output the bounding boxes in the format <ref>category</ref><box>[[x1, y1, x2, y2], ...]</box> "
    "with coordinates normalized to [0, 1000]."
)


@register_dataset("coco")
class CocoDataset(DatasetAdapter):
    name = "coco"
    task = "image_det"
    # scoring: MTLA over all Q_p tokens; box overlap; NMS pool across rollouts; COCO mAP.
    signal = "local_attention"
    overlap = "iou"
    select = "fuse"
    metric = "coco_map"
    # 5000 small image requests -> async multi-engine vLLM pool (throughput).
    gen_strategy = "pooled"

    def load_items(self, cfg: "RunConfig") -> list[dict]:
        return json.load(open(cfg.path("data")))

    def prompt(self, item: dict) -> str:
        return PROMPT.format(cats=", ".join(item.get("categories", COCO_CLASSES)))

    def ground_truth(self, item: dict) -> list[GTRegion]:
        return [{"region": o["bbox_2d"], "label": o["label"]} for o in item["gt"]]

    def gen_record(self, cfg: "RunConfig", item: dict, response: str,
                   truncated: bool = False) -> GenRecord:
        return {"id": item["id"], "prompt": self.prompt(item), "response": response,
                "gt": self.ground_truth(item), "extra": {"image": item["image"]}}
