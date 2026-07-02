"""COCO detection dataset adapter (open-vocabulary, val2017).

Declarative: it loads items, builds the prompt + ground truth, and emits the uniform generation
record. All scoring (band reduction, NMS voting, mAP) is done by the ``evaluate.py`` stage
(+ ``mtla.voting`` / ``mtla.metrics``) per the descriptors below.

Reproduces (InternVL3.5-8B): hallucination AUROC 0.873 (MTLA); detection mAP 41.9 at N=16.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mtla.data.base import DatasetAdapter
from mtla.registry import register_dataset
from mtla.types import GenRecord, GTRegion

if TYPE_CHECKING:
    from mtla.config import RunConfig

# 80 COCO class names (the open-vocab detection prompt lists these).
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

PROMPT = (
    "Please detect all instances of {cats} in the image. "
    "Output the bounding boxes in the format <ref>category</ref><box>[[x1, y1, x2, y2], ...]</box> "
    "with coordinates normalized to [0, 1000]."
)


@register_dataset("coco")
class CocoDataset(DatasetAdapter):
    """COCO open-vocabulary detection adapter (val2017, task ``image_det``).

    One item per image; the prompt asks for boxes over the 80 COCO classes (or an
    item-supplied subset). Scoring uses the ``local_attention`` signal with box
    IoU, pools candidates across rollouts with NMS (``select="fuse"``), and reports
    COCO mAP; generation runs on the ``"pooled"`` vLLM strategy for throughput over
    the ~5000 small image requests.
    """

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
        """Load the per-image work items from the COCO JSON file.

        Args:
            cfg: Active run config; ``cfg.path("data")`` points at the JSON file
                (a list of item dicts, each with ``id``, ``image``, ``gt``, and an
                optional ``categories`` list).

        Returns:
            The list of image item dicts parsed from the JSON.
        """
        return json.load(open(cfg.path("data")))

    @property
    def prompt(self) -> str:
        """Build the open-vocabulary detection prompt for one image.

        Returns:
            The prompt asking for normalized boxes over the item's class list.
        """
        return PROMPT.format(cats=", ".join(COCO_CLASSES))

    def ground_truth(self, item: dict) -> list[GTRegion]:
        """Return the ground-truth boxes for one image.

        Args:
            item: An image item with a ``gt`` list of ``{"bbox_2d", "label"}``
                objects (boxes in absolute pixel ``[x1, y1, x2, y2]``).

        Returns:
            List of ``{"region", "label"}`` dicts, one per ground-truth box.
        """
        return [{"region": o["bbox_2d"], "label": o["label"]} for o in item["gt"]]

    def gen_record(
        self, cfg: "RunConfig", item: dict, response: str, truncated: bool = False
    ) -> GenRecord:
        """Assemble the uniform generation record for one image.

        Args:
            cfg: Active run config (unused here; images are already absolute paths
                on the item).
            item: An image item with ``id`` and ``image`` (the image path).
            response: The model's raw, unparsed response text.
            truncated: Whether generation hit the token limit before finishing.

        Returns:
            A ``GenRecord`` whose ``extra`` holds ``{"image": <path>}``.
        """
        return {
            "id": item["id"],
            "prompt": self.prompt(item),
            "response": response,
            "gt": self.ground_truth(item),
            "extra": {"image": item["image"]},
        }
