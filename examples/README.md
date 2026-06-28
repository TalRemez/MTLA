# Examples

## CPU demo (no GPU, no downloads)

```bash
pip install -e ".[demo,coco]"
python examples/demo.py
```

Scores ~800 bundled predictions (InternVL3.5-8B on COCO) — MTLA ≈ 0.87 vs SVAR ≈ 0.79 AUROC —
and renders attention heatmaps to `examples/output/`.

## Full reproduction via `run.py`

All benchmarks run through the unified pipeline; the config picks the model + dataset:

```bash
# image detection (InternVL3.5-8B)
python run.py --config configs/coco_internvl.yaml --stage generate   # vLLM, GPU
python run.py --config configs/coco_internvl.yaml --stage extract    # HF eager attn, GPU
python run.py --config configs/coco_internvl.yaml --stage score      # CPU

# video grounding (Qwen3-VL-8B) — generate+extract are fused in one HF pass
python run.py --config configs/qvhighlights_qwen3vl.yaml --stage extract --slot first_digit
python run.py --config configs/qvhighlights_qwen3vl.yaml --stage score   --slot first_digit
python run.py --config configs/charades_qwen3vl.yaml     --stage score   --slot first_digit
```

Edit the `paths:` in each `configs/*.yaml` to point at your data (see `docs/DATA.md`). For
self-consistency voting, run the GPU stages once per seed (set `generate.extra.seed` / pass
multiple seeds), then `score --n 16`.

### Gotchas that change the numbers

- **Video needs `--slot first_digit`.** The score stage's slot selects which timestamp token's
  attention is used; `first_digit` is the paper setting. Other slots give worse numbers.
- **Fusion: COCO = `--agg sum`, video = `--agg max`.** COCO (many boxes/image) benefits from
  summing cluster scores across rollouts; single-/few-span video uses max. The configs already
  set the right default; the flag is for sweeps.
- **`score` is CPU-only** and reads the feature shards written by `extract` — no GPU or model
  needed to reproduce the metrics.

Expected numbers per benchmark are in [`../docs/RESULTS.md`](../docs/RESULTS.md).

## Audio

AudioSet-Strong / Audio Flamingo 3 lives under [`../cascade/`](../cascade/) — a separate,
documented multi-stage pipeline (it uses an external Bedrock normalization step that doesn't
fit the unified flow).
