# Datasets

No data ships with this repo — download each dataset from its source and point the example
scripts at it with the `--dataset` / `--ann` / `--video_dir` / `--coco_gt` flags.

## COCO detection (image example)

- **Images + annotations:** https://cocodataset.org/#download — `val2017.zip` and
  `annotations_trainval2017.zip`. The scorer needs `instances_val2017.json` (`--coco_gt`).
- **Open-vocabulary dataset JSON** (`--dataset`): a JSON list, one entry per image, giving
  the image path, the class-name list to detect, and the ground-truth boxes. Each entry:

  ```json
  {
    "id": 397133,
    "image": "/path/to/val2017/000000397133.jpg",
    "categories": ["person", "bicycle", "..."],
    "conversations": [
      {"from": "human", "value": "<prompt>"},
      {"from": "gpt",   "value": "[{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"person\"}, ...]"}
    ]
  }
  ```

  `categories` is the prompt's class list; `conversations[1].value` is the GT used to label
  hallucinations (boxes in `[0,1000]`). The 80 COCO class names are the standard set.

## QVHighlights (video example)

- **Annotations:** https://github.com/jayleicn/moment_detr — `highlight_val_release.jsonl`
  (`--ann`). Each line has `qid`, `query`, `duration`, and `relevant_windows`.
- **Videos:** the QVHighlights clips as `{video_id}.mp4` in one directory (`--video_dir`),
  obtained via the moment_detr data instructions.

## Other benchmarks in the paper

- **Charades-STA** (single-span video grounding): https://github.com/jiyanggao/TALL — same
  Qwen3-VL pipeline as QVHighlights, scored by R@1@IoU{0.3,0.5,0.7}. Use `--slot first_digit`
  and `--agg max`.
- **AudioSet-Strong** (sound-event detection): audio grounding with Audio Flamingo 3, scored
  by PSDS1 (DCASE Task 4). All-layers reduction (28-layer model). Headline numbers in
  [`RESULTS.md`](RESULTS.md); the audio extraction maps time→token at a fixed 25 Hz.
