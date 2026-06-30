"""Prepare COCO detection data for the image_det benchmark.

Downloads COCO val2017 images + annotations, then builds the open-vocabulary dataset JSON the
`coco` adapter loads (`CocoDataset.load_items`). Produces, under `--out` (default `data/coco/`):
  - val2017/                       the images
  - instances_val2017.json         COCO GT for mAP            (config `paths.coco_gt`)
  - coco_val_openvocab_80.json     the open-vocab dataset     (config `paths.data`)

The open-vocab JSON has one entry per image:
  {id, image, categories (sorted class names present in the GT), num_objects,
   conversations: [{from:human, value:<prompt>}, {from:gpt, value:<GT as bbox_2d JSON>}]}
Boxes are scaled to [0,1000] (Qwen/InternVL grounding convention); the GT list keeps COCO's
annotation order. The prompt lists all 80 COCO classes (the model adapters either use it verbatim
or build their own from `categories`).

    python scripts/prepare_coco.py            # -> data/coco/
    python scripts/prepare_coco.py --out /data/coco
"""
import argparse
import json
import os

from _prep_utils import download, unzip, out_dir, done_banner

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

COCO_80 = [
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

PROMPT = ("<image>\nLocate all instances of " + ", ".join(COCO_80) +
          " in this image and output the bbox coordinates in JSON format.")


def build_openvocab(instances_json: str, images_dir: str, out_json: str) -> int:
    """Build coco_val_openvocab_80.json from instances_val2017.json (matches the paper's file)."""
    import contextlib
    import io
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(instances_json)
    cat_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    records = []
    for iid in coco.getImgIds():
        img = coco.loadImgs(iid)[0]
        W, H = img["width"], img["height"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid))  # COCO annotation order
        gt = []
        for a in anns:
            if a.get("iscrowd", 0):
                continue  # skip crowd regions (large blanket boxes); matches the paper's dataset
            x, y, w, h = a["bbox"]
            gt.append({
                "bbox_2d": [round(x / W * 1000), round(y / H * 1000),
                            round((x + w) / W * 1000), round((y + h) / H * 1000)],
                "label": cat_name[a["category_id"]],
            })
        categories = sorted({g["label"] for g in gt})
        records.append({
            "id": iid,
            "image": os.path.join(images_dir, img["file_name"]),
            "categories": categories,
            "num_objects": len(gt),
            "conversations": [
                {"from": "human", "value": PROMPT},
                {"from": "gpt", "value": json.dumps(gt)},
            ],
        })
    with open(out_json, "w") as f:
        json.dump(records, f)
    return len(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default: <repo>/data/coco)")
    ap.add_argument("--skip-images", action="store_true",
                    help="skip the ~1GB val2017 image download (annotations + JSON only)")
    args = ap.parse_args()
    root = out_dir(args.out, "coco")

    # 1) annotations -> instances_val2017.json
    ann_zip = download(ANNOS_URL, os.path.join(root, "annotations_trainval2017.zip"))
    instances = os.path.join(root, "annotations", "instances_val2017.json")
    unzip(ann_zip, root, skip_marker=instances)

    # 2) images
    images_dir = os.path.join(root, "val2017")
    if not args.skip_images:
        img_zip = download(IMAGES_URL, os.path.join(root, "val2017.zip"))
        unzip(img_zip, root, skip_marker=os.path.join(images_dir, "000000397133.jpg"))
    else:
        print("  [skip] images (--skip-images); set the image paths yourself if needed")

    # 3) build the open-vocab dataset JSON
    out_json = os.path.join(root, "coco_val_openvocab_80.json")
    n = build_openvocab(instances, images_dir, out_json)
    print(f"  built {out_json}  ({n} images)")

    done_banner("COCO", [f"paths.data:    {out_json}",
                         f"paths.coco_gt: {instances}"])


if __name__ == "__main__":
    main()
