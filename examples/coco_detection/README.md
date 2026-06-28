# COCO detection (InternVL3.5-8B)

Full reproduction of the image-detection results: generate detections, extract MTLA
attention, and score. Three stages, three scripts.

```
generate.py   ->  predictions/seed{K}/temp_0/predictions.json   (vLLM, GPU)
extract.py    ->  features/seed{K}/shard*.pt                      (HF eager attention, GPU)
score.py      ->  hallucination AUROC + COCO mAP                  (CPU)
```

## Expected numbers (COCO val2017, 5k images)

| Metric | MTLA (ours) | SVAR (baseline) |
|---|---|---|
| Hallucination AUROC (single seed) | **0.873** | 0.779 |
| Detection mAP, N=16, sum-fusion | **41.9** | — |

(N=5 reaches 40.8 mAP. The single-seed AUROC and `agg=sum` voting are reproduced by
`score.py`; see the repo `docs/RESULTS.md` for the full table.)

## Environments

Two GPU environments (an 8×H100 box was used for the paper):

- **Generation** (vLLM): `pip install vllm>=0.19 transformers>=4.57`
- **Extraction** (HF eager attention): `pip install transformers>=4.57 torch>=2.1 torchvision pillow`
- **Scoring** (CPU): `pip install -e ".[coco]"` from the repo root.

## Data

See [`../../docs/DATA.md`](../../docs/DATA.md). You need COCO val2017 images, the
`instances_val2017.json` annotations, and an open-vocabulary dataset JSON listing the 80
class names per image (the `--dataset` file).

## 1. Generate (per seed)

```bash
python generate.py \
    --model OpenGVLab/InternVL3_5-8B \
    --dataset /path/to/coco_val_openvocab_80.json \
    --output_dir predictions/seed0/temp_0 \
    --gpu_ids 0 1 2 3 4 5 6 7 \
    --temperature 0.0 --seed 0          # seed 0 = greedy; use 1..15 with T>0 for voting
```

For self-consistency voting, run seeds `1..15` with `--temperature 0.7`. The model emits
native grounding (`<ref>label</ref><box>[[x1,y1,x2,y2]]</box>`) in `[0,1000]` coordinates.

## 2. Extract attention (per seed)

```bash
python extract.py \
    --pred_file predictions/seed0/temp_0/predictions.json \
    --dataset   /path/to/coco_val_openvocab_80.json \
    --out_dir   features/seed0 \
    --gpus 0 1 2 3 4 5 6 7 --n_images 5000
```

This runs a post-hoc forward pass with **eager attention** (it monkeypatches
`eager_attention_forward`; see `mtla/extract.py` for the mechanism). For each predicted box
it sums attention from the box's coordinate+label tokens to the image tokens **inside** the
box (`image_inside_sum` = MTLA) and to **all** image tokens (`image_sum` = SVAR), per layer
and head. InternVL's dynamic tiling is handled by `mtla.bbox_to_internvl_token_indices`.

## 3. Score (CPU)

```bash
python score.py \
    --features_root    features \
    --predictions_root predictions \
    --coco_gt /path/to/instances_val2017.json \
    --n 16 --agg sum
```

`--agg sum` is the COCO headline: each kept box is scored by the **sum** of its cluster's
MTLA scores across rollouts (rewarding boxes that recur). COCO is the only benchmark where
sum beats `max`; video/audio use `--agg max`.

> **Layout note:** `score.py` expects `features_root/seed{K}/shard*.pt` and
> `predictions_root/seed{K}/temp_0/predictions.json`. Put each seed's outputs under its own
> `seed{K}` directory (symlinks are fine).
