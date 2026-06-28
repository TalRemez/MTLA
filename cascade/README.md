# AudioSet-Strong / Audio Flamingo 3 (separate cascade)

This directory holds the **audio** sound-event-detection pipeline. It is kept **separate from
the unified `run.py` pipeline** on purpose: unlike COCO / QVHighlights / Charades, the headline
AudioSet result comes from a multi-stage *cascade*, one stage of which calls an external
service (Claude on AWS Bedrock) for label normalization. That does not fit a clean,
self-contained `generate → extract → score` flow, so we document it here as research code
rather than forcing it into the unified abstraction.

The MTLA scoring idea is identical to the rest of the repo — restrict attention to the audio
tokens **inside** a predicted `[onset, offset]` span — only the surrounding plumbing differs.

## Model

Audio Flamingo 3 (`nvidia/audio-flamingo-3-hf`); its language model is Qwen2.5-7B (28 layers).
Audio tokens are emitted at a fixed **25 Hz**, so token `i` maps to time `i / 25` seconds
(`casc_d_extract_batched.py` uses `torch.arange(n_audio) / 25.0`).

## The cascade

| Stage | Script | What it does | External? |
|---|---|---|---|
| A. Gate | `casc_a_gate.py` | open-vocabulary free-text event labels via AF3 (N temperature-sampled runs, union) | no (local AF3) |
| B. Normalize | `casc_b_normalize.py` | map free-text labels → 456 canonical AudioSet classes | **YES — Claude/Bedrock** |
| C. Localize | `casc_c_localize.py` | per-candidate temporal localization via AF3 (N rollouts) → `[onset,offset]` windows | no (local AF3) |
| D. Extract | `casc_d_extract_batched.py` | HF-eager pass: attention from response tokens to audio tokens inside/over each window → `[L,H]` sums | no |
| E. Score | `casc_e_score.py` | PSDS1 (DCASE Task 4) with NMS / voting fusion over seeds | no |

`run_audioset_af3.py` is a **single-pass alternative** (prompt AF3 directly for
`{label, [onset,offset]}` with *local* heuristic label normalization, no Bedrock); pair it
with `casc_d_extract_batched.py` + `score_af3_mtla.py` for a 3-stage generate→extract→score
flow if you want to avoid the external dependency (it may not reproduce the exact headline
PSDS, which used the full cascade).

## ⚠️ External dependency (Stage B)

`casc_b_normalize.py` calls `boto3` `bedrock-runtime` with an Anthropic Claude model. It needs
AWS credentials and Bedrock access; it is **not runnable out of the box** and is the reason the
cascade lives outside the unified pipeline. Swap in any label-normalization method (including
the local heuristic in `run_audioset_af3.py`) if you don't have Bedrock.

## Scoring (PSDS1)

`casc_e_score.py` / `score_af3_mtla.py` compute PSDS1 with the DCASE Task 4 protocol
(`dtc = gtc = 0.7`, `alpha_st = 0`, 415-class macro, `max_efpr = 100`) via `psds_eval`. Note:
`psds_eval` 0.5.3 needs a pandas-3 fix — `psds.py` `fillna(method='ffill')` → `ffill()`.

Reduction is the same MTLA score as elsewhere (mean over heads, sum over layers); AudioSet uses
**all 28 layers** (no middle-layer band). Reported: NMS-MTLA PSDS1 ≈ 0.255, NMS-SVAR ≈ 0.229;
per-prediction hallucination AUROC MTLA 0.81 / SVAR 0.61.

## Paths

Scripts default to `$CASCADE_ROOT` (current dir) and `$AUDIOSET_PARQUET`; set those env vars,
or edit the constants at the top of each script, to point at your AudioSet-Strong data.
