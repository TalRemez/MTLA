# QVHighlights temporal grounding (Qwen3-VL-8B)

Full reproduction of the video moment-retrieval results. Generation and attention extraction
happen in one pass (the model is run with eager attention), then scoring uses the **official
Moment-DETR** evaluator vendored under `third_party/moment_detr_eval`.

```
generate_and_extract.py  ->  features/seed{K}/shard*.pt + predictions/   (Qwen3-VL, GPU)
score.py                 ->  mAP, R@1, hallucination AUROC               (CPU)
```

## Expected numbers (QVHighlights val)

| Metric | MTLA (ours) | SVAR (baseline) |
|---|---|---|
| mAP (avg IoU 0.5:0.95), N=16 | **36.6** | 28.1 |
| R@1 @ IoU 0.5 | **55.1** | — |
| R@1 @ IoU 0.7 | **39.5** | — |
| Hallucination AUROC (single seed) | **0.80** | 0.42 |

## Two gotchas that change the numbers

1. **`--slot first_digit`** — score the attention of the *first digit token* of each window's
   timestamps. The script's historical default (`all_mean`) gives different, worse numbers.
   The paper uses `first_digit`; always pass it.
2. **`--agg max`** (the default here) — video uses **max**-of-cluster fusion, not the
   sum-fusion COCO uses. Summing collapses the MTLA/SVAR gap on single-/few-segment tasks.

## Environment

- **Generation + extraction** (GPU): `pip install transformers>=4.57 torch>=2.1 qwen-vl-utils decord`
- **Scoring** (CPU): `pip install -e .` from the repo root (numpy + scikit-learn only; the
  Moment-DETR evaluator is vendored, no extra install).

## Data

See [`../../docs/DATA.md`](../../docs/DATA.md): QVHighlights val annotations
(`highlight_val_release.jsonl`) and the `{video_id}.mp4` clips.

## 1. Generate + extract (per seed)

```bash
python generate_and_extract.py \
    --ann       /path/to/highlight_val_release.jsonl \
    --video_dir /path/to/qvhighlights/videos \
    --out_dir   features/seed0 \
    --pred_dir  predictions/seed0 \
    --gpus 0 1 2 3 4 5 6 7        # seed 0 = greedy
```

For voting, add `--seed K` (K = 1..15) to sample with T=0.7. Videos are sampled at 1 fps;
for each predicted `[start, end]` window the script records, per layer/head, the attention
from the window's timestamp tokens to the video-frame tokens **inside** the window
(`frame_sum` restricted to the span = MTLA) and over **all** frames (`video_sum` = SVAR).
Four token slots are saved; `score.py --slot first_digit` selects the one used in the paper.

## 2. Score (CPU)

```bash
python score.py \
    --feat_root features \
    --ann       /path/to/highlight_val_release.jsonl \
    --slot first_digit --n 16
```

Reports single-rollout numbers (per seed, averaged) and N-rollout voting for several fusion
variants. The headline row is **NMS-MTLA [max]**; **NMS-SVAR [max]** is the baseline.
