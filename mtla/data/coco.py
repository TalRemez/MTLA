"""COCO detection dataset adapter (open-vocabulary, val2017).

Declarative: it loads items, builds the prompt + ground truth, and emits the uniform generation
record. All scoring (band reduction, NMS voting, mAP) is done by the ``evaluate.py`` stage
(+ ``mtla.voting`` / ``mtla.metrics``) /
``mtla.metrics`` per the descriptors below.

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

# Task only, no output-FORMAT instruction: the format is model-specific and lives on the model
# adapter (ModelAdapter.detection_prompt_suffix), so the dataset states only the task. Each model
# appends the phrasing that elicits its natural grounding syntax (Qwen: "... in JSON format." ->
# {"bbox_2d","label"}; InternVL: the <ref>label</ref><box>[[...]]</box> template), which its own
# parse_response reads. build_text_prompt(dataset, item) composes the two.
PROMPT = "Locate all instances of {cats} in this image"


@register_dataset("coco")
class CocoDataset(DatasetAdapter):
    """COCO open-vocabulary detection adapter (val2017, task ``image_det``).

    One item per image; the prompt asks for boxes over the 80 COCO classes (or an
    item-supplied subset). Scoring uses the ``digits`` slot inside the region
    (``digits_local``) with box IoU, pools candidates across rollouts with NMS
    (``select="fuse"``), and reports COCO mAP; generation runs on the ``"pooled"``
    vLLM strategy for throughput over the ~5000 small image requests.
    """

    name = "coco"
    task = "image_det"
    # scoring: MTLA over the COORDINATE tokens (paper's coord_mean slot), inside the region; box
    # overlap; NMS pool across rollouts scored by support x score; COCO mAP. Matches the paper's
    # headline COCO recipe (coco_voting_nms_internvl.py: digits_local + support fusion). The best slot
    # is model-specific (InternVL: digits; Qwen3-VL: all), overridden per model config via score.slot.
    slot = "digits"
    attn_scope = "local"
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

    def prompt(self, item: dict) -> str:
        """Build the open-vocabulary detection prompt for one image.

        The COCO prompt lists the full fixed 80-class vocabulary, so it does not
        depend on ``item``; the argument is kept to match the ``DatasetAdapter``
        contract (video datasets build the prompt from the item's query text).

        Args:
            item: The image item (unused; the class list is fixed).

        Returns:
            The task prompt asking to detect the 80 COCO classes (no output-format
            instruction; the model answers in its own natural grounding format).
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
