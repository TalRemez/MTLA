# Results

Headline numbers reproduced by the example pipelines. MTLA is the inside-region attention
score (ours); SVAR is the global-attention baseline (Jiang et al.). All use the default
middle-layer band (L8–21) except AudioSet, which uses all 28 layers. IoU ≥ 0.5 throughout.

## Hallucination detection — AUROC (single rollout)

How well the score separates grounded from hallucinated predictions.

| Benchmark | Model | MTLA | SVAR |
|---|---|--:|--:|
| COCO detection | Qwen3-VL-8B | **0.902** | 0.763 |
| COCO detection | InternVL3.5-8B | **0.873** | 0.803 |
| COCO detection | Gemma-4 E4B | **0.753** | 0.671 |
| QVHighlights | Qwen3-VL-8B | **0.800** | 0.415 |
| Charades-STA | Qwen3-VL-8B | **0.684** | 0.512 |
| AudioSet-Strong | Audio Flamingo 3 | **0.813** | 0.608 |

`run.py --config configs/coco_internvl.yaml --stage score` reproduces the InternVL row
(0.873 / 0.779); the CPU demo (`examples/demo.py`) reproduces it on an 800-prediction
subsample (≈0.87 / ≈0.79).

## Task metrics after MTLA re-ranking / self-consistency voting

**COCO detection** (InternVL3.5-8B, val2017, sum-of-cluster fusion):

| N (rollouts) | mAP | AP50 | AP75 |
|---|--:|--:|--:|
| 1 | 36.1 | 55.9 | 37.1 |
| 5 | 40.8 | 64.6 | 40.9 |
| **16** | **41.9** | **66.9** | **41.5** |

**QVHighlights** (Qwen3-VL-8B, N=16, NMS-MTLA): mAP **36.6**, R@1@0.5 **55.1**,
R@1@0.7 **39.5** (SVAR baseline mAP 28.1).

**Charades-STA** (Qwen3-VL-8B, N=16, max selection, `--slot first_digit`): R@1@0.3 **76.3**,
R@1@0.5 **55.4**, R@1@0.7 **29.4**, mIoU 0.508 (SVAR R@1@0.5 43.8).

**AudioSet-Strong** (Audio Flamingo 3, N=16, PSDS1 @ DCASE Task 4): NMS-MTLA **0.255**,
NMS-SVAR 0.229.

## Layer-band ablation (MTLA)

MTLA is not tuned to the L8–21 band. Using **all** layers (parameter-free) costs little:

| Benchmark | Metric | band L8–21 | all layers | Δ |
|---|---|--:|--:|--:|
| COCO (InternVL) | AUROC | 0.873 | 0.867 | −0.006 |
| COCO (InternVL) | mAP N=16 | 41.90 | 41.45 | −0.45 |
| QVHighlights | AUROC | 0.800 | 0.754 | −0.046 |
| QVHighlights | mAP N=16 | 36.71 | 36.15 | −0.56 |
| Charades | AUROC | 0.684 | 0.632 | −0.051 |
| Charades | R@1@0.5 N=16 | 55.40 | 53.66 | −1.74 |
| AudioSet | — | already all-layers | 0.813 (AUROC) | — |

COCO is nearly band-insensitive; video AUROC drops ~0.05 because the video grounding signal
concentrates in early/middle layers and the late third decays toward chance. The MTLA ≫ SVAR
ranking holds under either choice on every benchmark.

> Numbers are from the project's evaluation logs; the COCO and QVHighlights rows are
> reproducible end-to-end with the example scripts. The paper is the citable source of
> record (link to be added on release).
