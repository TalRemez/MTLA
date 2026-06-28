# Method: Multi-Token Localized Attention (MTLA)

MTLA is a **training-free, post-hoc** confidence score for grounding predictions from
multimodal LLMs. It needs no extra parameters, no fine-tuning, and no auxiliary model: it
reads the model's own attention from the same forward pass that produced the prediction.

## Problem setup

A grounding MLLM autoregressively emits one or more localized predictions. Each prediction
`p` has a **proposal region** `R_p` and a label. Depending on the modality, `R_p` is

- a bounding box `[x1, y1, x2, y2]` in an image (coordinates in `[0, 1000]`),
- a temporal interval `[t_start, t_end]` in video or audio.

A prediction is **hallucinated** if its region matches no ground-truth region under the
task's criterion (IoU ≥ 0.5 throughout). The goal is a scalar score `s(p)`, computed from
the model's attention, that is high for grounded predictions and low for hallucinations.

## The signal

Let the transformer have `L` layers and `H` heads. Let `X = {k_1, ..., k_N}` be the
input-modality tokens (image patches, video frames, or audio frames). For a response token
at position `q`, the model produces attention weights `a[l,h](q -> k)` over `k ∈ X`.

**Global Attention (GA / SVAR baseline)** sums attention over *all* modality tokens:

```
GA[l,h](q) = sum over k in X of  a[l,h](q -> k)
```

**Localized Attention (LA)** — the key idea — restricts the sum to the tokens *inside* the
proposal region, `M(R_p)` (image patches overlapping the box; frames whose timestamps fall
in the span):

```
LA[l,h](q) = sum over k in M(R_p) of  a[l,h](q -> k)
```

A grounded prediction attends strongly to evidence inside its own region; a hallucination
relies on context scattered elsewhere, so its LA is low even when its GA is comparable.

## Multi-token aggregation

A prediction is several tokens (the digits of each coordinate, plus the label). Any single
token's attention is noisy; averaging across the prediction's tokens `Q_p` is far more
robust. **Multi-Token Localized Attention**:

```
MTLA[l,h](p) = (1/|Q_p|) * sum over q in Q_p of  LA[l,h](q)
```

## Layer and head reduction

Average over heads and sum over a fixed band of middle layers `L` to get one scalar:

```
s(p) = sum over l in band of  ( (1/H) * sum over h of  MTLA[l,h](p) )
```

The default band is **layers 8–21** (`mtla.DEFAULT_BAND`), used for every image and video
model tested (Qwen3-VL, InternVL: 36 layers; Gemma-4: 42). Audio (Audio Flamingo 3, 28
layers) uses **all** layers. MTLA is not very sensitive to the exact band — see the
layer-band ablation in [`RESULTS.md`](RESULTS.md). This whole reduction is
`mtla.reduce_band`.

## Self-consistency voting

Sampling `N` stochastic rollouts per input enlarges the candidate pool (better recall). We
pool predictions across rollouts, merge overlaps with non-maximum suppression, and score
each kept prediction from its cluster's MTLA values (`mtla.nms_fuse`):

- **max** (default): keep the single highest-scoring rollout. Used for video and audio.
- **sum**: sum the cluster's scores, rewarding regions that recur across rollouts. Used for
  **COCO detection only**, where each image yields many predictions; there it beats max.

## Code map

| Paper concept | Code |
|---|---|
| `M(R_p)` mask (image / tiling / temporal) | `mtla/mask.py` |
| LA / MTLA extraction (eager-attention hook) | `mtla/extract.py` (+ per-model glue in `examples/`) |
| Layer-band + head reduction `s(p)` | `mtla/score.py` (`reduce_band`, `mtla_score`, `svar_score`) |
| Self-consistency voting / NMS fusion | `mtla/voting.py` (`nms_fuse`) |
| AUROC / COCO mAP | `mtla/eval.py` |
