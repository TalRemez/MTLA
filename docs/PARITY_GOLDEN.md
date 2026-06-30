# Parity golden numbers (Phase 0 baseline)

Captured through the pre-refactor `run.py --stage score` on the validated feature shards.
Every refactor phase must reproduce these. Source shards:
- COCO: `coco_det_attn_5k_t07_internvl` (seed0) + `..._voting_rawfix/seed1`
- QVH:  `qvhighlights_attn_v3_sample` (16 seeds)
- Charades: `charades_attn_v2_sample` (16 seeds)

Signal names below use the pre-refactor record fields; post-refactor the MTLA signal is
`local_attention` (images, all-Q_p mean) / `first_digit` (video), band L8-21. SVAR was removed
from the core (its baseline numbers are kept here as the documented historical targets).

## COCO detection (InternVL3.5-8B), MTLA=local_attention, band L8-21
- Hallucination AUROC (seed 0, 43139 preds): MTLA **0.8730**, SVAR **0.8031**

## QVHighlights (Qwen3-VL-8B), N=16, agg=max, MTLA=first_digit
- Per-window AUROC (seed 0): MTLA **0.8151**, SVAR **0.4149**
- NMS-MTLA: mAP **36.60**, R1@.5 **55.10**, R1@.7 **39.48**
- NMS-SVAR: mAP **28.13**, R1@.5 **39.42**, R1@.7 **27.94**

## Charades-STA (Qwen3-VL-8B), N=16, agg=max, MTLA=first_digit
- AUROC (seed 0): MTLA **0.6634**, SVAR **0.5177**
- max-MTLA: R@.3 **76.26**, R@.5 **55.40**, R@.7 **29.41**, mIoU **0.5076**
- max-SVAR: R@.3 68.82, R@.5 43.79, R@.7 18.90, mIoU 0.4316

## CPU demo (fixtures/coco_demo.pt)
- MTLA AUROC ≈ 0.870, SVAR ≈ 0.813
