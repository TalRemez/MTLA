# Datasets

No data ships with this repo. Each dataset has a one-command preparation script under `scripts/`
that downloads the source and writes the files the adapters load; you then point the config's
`paths:` at them (the GPU stages are config-driven — no per-run path flags).

```bash
python scripts/prepare_coco.py          # -> data/coco/        (images + instances + openvocab JSON)
python scripts/prepare_qvhighlights.py  # -> data/qvhighlights/(val annotations; videos separate)
python scripts/prepare_charades.py --from-annotations charades_sta_test.txt   # -> data/charades/
```

Each script prints the exact `paths:` values to copy into your `configs/*.yaml`. Video clips for
QVHighlights / Charades are large and distributed separately — the scripts say where to get them
and where to place them. Details + the manual route per dataset below.

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
    "conversations": [
      {"from": "human", "value": "<prompt>"},
      {"from": "gpt",   "value": "[{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"person\"}, ...]"}
    ]
  }
  ```

  `categories` is the per-image set of class names present in the GT; `conversations[1].value`
  is the GT used to label hallucinations (boxes scaled to `[0,1000]`, in COCO annotation order,
  `iscrowd` regions excluded). The prompt lists all 80 COCO classes.

## QVHighlights (video example)

`prepare_qvhighlights.py` fetches `highlight_val_release.jsonl` (the official Moment-DETR val
annotations, https://github.com/jayleicn/moment_detr) to `paths.ann`. Each line has `qid`, `vid`,
`query`, `duration`, `relevant_windows` — loaded as-is.

- **Videos** (`paths.video_dir`): the QVHighlights clips as `{vid}.mp4` in one directory,
  obtained via the Moment-DETR data instructions (large; Google Drive).

## Charades-STA (video example)

`prepare_charades.py` writes the test parquet (`paths.data`) with columns `video` (`{id}.mp4`),
`caption`, `timestamp = [start, end]`. Get the annotations either way:
- `--from-annotations charades_sta_test.txt` — convert the original Charades-STA annotations
  (Gao et al., TALL; `VIDID start end##caption` lines). Open, no auth.
- `--hf` — the HuggingFace `lmms-lab/Charades-STA` parquet (gated: needs `huggingface_hub` + an
  accepted-terms HF login).

- **Videos** (`paths.video_dir`): `Charades_v1_480` (~16GB) from
  https://prior.allenai.org/projects/charades, clips as `{video_id}.mp4`. Scored by
  R@1@IoU{0.3,0.5,0.7}; use `--agg max`.

## Other benchmarks in the paper
- **AudioSet-Strong** (sound-event detection): audio grounding with Audio Flamingo 3, scored
  by PSDS1 (DCASE Task 4). All-layers reduction (28-layer model). Headline numbers in the
  [Results](../README.md#results) section of the README; the audio extraction maps time→token at a fixed 25 Hz.
