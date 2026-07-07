# Datasets

No data ships with this repo. Each dataset has a one-command preparation script under `scripts/`
that downloads the source and writes the files the adapters load; you then point the config's
`paths:` at them (the GPU stages are config-driven — no per-run path flags).

```bash
python -m scripts.prepare_coco          # -> data/coco/        (images + instances + openvocab JSON)
python -m scripts.prepare_qvhighlights  # -> data/qvhighlights/(val annotations; videos separate)
python -m scripts.prepare_charades      # -> data/charades/    (downloads the open test annotations)
```

Each script prints the exact `paths:` values to copy into your `configs/*.yaml`, and downloads
everything it needs — including the QVHighlights / Charades video clips (large: ~134GB and ~15GB
respectively). Pass `--skip-videos` to fetch annotations only and provide the clips yourself.
Details + sizes per dataset below.

## COCO detection (image example)

`prepare_coco.py` downloads `val2017.zip` + `annotations_trainval2017.zip` from
https://cocodataset.org/#download and builds the open-vocabulary dataset JSON from
`instances_val2017.json` (which is also the scorer's `paths.coco_gt`).

- **Open-vocabulary dataset JSON** (`paths.data`): a JSON list, one entry per image, giving
  the image path, the class-name list to detect, and the ground-truth boxes. Each entry:

  ```json
  {
    "id": 397133,
    "image": "/path/to/val2017/000000397133.jpg",
    "categories": ["person", "bicycle", "..."],
    "gt": [{"bbox_2d": [x1, y1, x2, y2], "label": "person"}, "..."]
  }
  ```

  `gt` is the ground-truth boxes used to label hallucinations (scaled to `[0,1000]`, in COCO
  annotation order, `iscrowd` regions excluded). No prompt is stored — the prompt lives in the
  `coco` adapter (`mtla.data.coco.PROMPT`) and lists the full 80-class COCO vocabulary on every
  image (open-vocabulary detection). `categories` records the per-image GT class set for reference
  and is not used to build the prompt.

## QVHighlights (video example)

`prepare_qvhighlights.py` fetches `highlight_val_release.jsonl` (the official Moment-DETR val
annotations, https://github.com/jayleicn/moment_detr) to `paths.ann`. Each line has `qid`, `vid`,
`query`, `duration`, `relevant_windows` — loaded as-is.

- **Videos** (`paths.video_dir`): the QVHighlights clips as `{vid}.mp4`. Downloaded + extracted by
  default from the QVHighlights authors' raw-video tarball (~134GB, all splits); `--skip-videos` to
  skip and place the clips there yourself.

## Charades-STA (video example)

`prepare_charades.py` writes the test parquet (`paths.data`) with columns `video` (`{id}.mp4`),
`caption`, `timestamp = [start, end]`. By default (no arguments) it downloads the original
Charades-STA test annotations (Gao et al., TALL; `VIDID start end##caption` lines — open, no auth)
and builds the parquet. Two overrides:
- `--from-annotations PATH` — convert a local copy of `charades_sta_test.txt` instead of downloading.
- `--hf` — the HuggingFace `lmms-lab/Charades-STA` parquet (gated: needs `huggingface_hub` + an
  accepted-terms HF login).

- **Videos** (`paths.video_dir`): `Charades_v1_480` clips as `{id}.mp4`. Downloaded + extracted by
  default from the Charades authors' open ~15GB zip; `--skip-videos` to skip and place them yourself.
  Scored by R@1@IoU{0.3,0.5,0.7}; use `--agg max`.

## Other benchmarks in the paper
- **AudioSet-Strong** (sound-event detection): audio grounding with Audio Flamingo 3, scored
  by PSDS1 (DCASE Task 4). All-layers reduction (28-layer model). Headline numbers in the
  [Results](../README.md#results) section of the README; the audio extraction maps time→token at a fixed 25 Hz.
