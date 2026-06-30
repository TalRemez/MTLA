# Propose and Attend: Training-free MLLM Grounding Confidence via Multi-Token Localized Attention

**Does a multimodal LLM actually look where it says it's looking?**

When a vision, video, or audio LLM grounds a prediction (a bounding box, a time span) it also
produces internal attention over the input. MTLA reads that attention and asks a simple
question: *did the prediction's tokens attend to evidence **inside** the region they claim?*
Grounded predictions do, while hallucinations attend elsewhere. The result is a
**training-free, post-hoc** confidence score that needs no fine-tuning, no extra model, and
no labels, just one forward pass of the model you already have.

<p align="center">
  <img src="docs/assets/method_pipeline.png" width="100%" alt="MTLA pipeline: an MLLM localizes objects, then MTLA reads the prediction tokens' attention restricted to the proposed region to score each prediction"/>
</p>
<p align="center"><em>An MLLM emits box and label predictions. MTLA reads the decoder's attention
from each prediction's tokens, restricts it to the patches inside the proposed region, and scores
the prediction by how much attention falls inside, high when grounded and low when hallucinated.</em></p>

## Why it works

```
SVAR baseline:  sum attention over ALL input tokens          (global)
MTLA (ours):    sum attention over tokens INSIDE the region  (localized)  ... averaged over
                the prediction's tokens, over heads, over a band of middle layers.
```

A hallucination can still attend to *something*; what it cannot do is attend to evidence
that isn't there, inside the box it invented. Restricting the attention sum to the proposal
region is what separates the two. See [`docs/METHOD.md`](docs/METHOD.md) for the full method.

## Results at a glance (hallucination-detection AUROC)

| Benchmark | Model | MTLA | SVAR baseline |
|---|---|--:|--:|
| COCO detection | Qwen3-VL-8B | **0.90** | 0.76 |
| QVHighlights (video) | Qwen3-VL-8B | **0.80** | 0.42 |
| Charades-STA (video) | Qwen3-VL-8B | **0.68** | 0.51 |
| AudioSet-Strong (audio) | Audio Flamingo 3 | **0.81** | 0.61 |

The same idea also improves task metrics via self-consistency voting (COCO **41.9** mAP at
N=16; QVHighlights **36.6** mAP). One method, four modalities, no training. Full table:
[`docs/RESULTS.md`](docs/RESULTS.md).

## Quickstart (60 seconds, CPU, no GPU / no downloads)

```bash
pip install -e ".[demo,coco]"
python examples/demo.py
```

This loads a small bundled fixture of pre-extracted attention (InternVL3.5-8B on COCO) and:

1. scores ~800 predictions, printing MTLA vs SVAR AUROC (MTLA ≈ 0.87, SVAR ≈ 0.79);
2. renders attention heatmaps for one image into `examples/output/`, showing the model
   looking inside a grounded box but scattering on a hallucinated one.

## Use it on your own predictions

MTLA scores a prediction from a `[L, H]` array (layers × heads): the attention its tokens pay
to the modality tokens **inside** its proposed region. `reduce_band` reduces it to one scalar
(mean over heads, sum over the layer band) — higher means more grounded.

**If you have the attention arrays** (the extract stage produces them), scoring is CPU-only.
Run it now on the bundled fixture — no GPU, no download:

```python
import numpy as np, torch
from mtla import reduce_band, auroc

data   = torch.load("fixtures/coco_demo.pt", weights_only=False)["scoring"]   # ~800 preds
one    = reduce_band(data[0]["image_inside_sum"])                              # single -> scalar
scores = reduce_band(np.stack([r["image_inside_sum"] for r in data]))         # batch  -> [N]
labels = [r["is_hallucinated"] for r in data]                                  # IoU>=0.5 flags
print(f"AUROC = {auroc(scores, labels):.3f}")                                  # ~0.87
```

**Starting from your own model?** Run it to write a `predictions.json`
(`[{id, status, response, pred_bboxes:[{box, label}]}, ...]`), then let the extract stage do the
GPU forward pass that captures attention, and score:

```bash
python run.py --config configs/coco_internvl.yaml --stage extract   # GPU: predictions -> [L,H] shards
python run.py --config configs/coco_internvl.yaml --stage score     # CPU: shards -> AUROC / mAP
```

To plug in a different model or task, see [Extending](#extending-add-a-new-model-or-task).

The package is small and modular:

| Module | What it does |
|---|---|
| `mtla.score` | `reduce_band`, `mtla_score` — the layer-band + head reduction |
| `mtla.mask` | map a box / time-span to the modality-token indices inside it (`M(R_p)`) |
| `mtla.mtla_attn` | the eager-attention monkeypatch + per-item driver that capture localized attention |
| `mtla.voting` | self-consistency NMS fusion across rollouts (`max` / `sum` / ...) |
| `mtla.eval` | hallucination AUROC and COCO mAP |
| `mtla.utils` | shared primitives: `iou` / `tiou`, `repeat_kv`, token-span helpers |
| `mtla.viz` | attention heatmap overlays |

## Full reproduction — one config-driven pipeline

Every benchmark runs through the same three stages; a YAML config picks the model + dataset.
No bulk features are shipped; you regenerate them (GPU needed for `generate`/`extract`,
`score` is CPU-only).

```bash
python run.py --config configs/coco_internvl.yaml      --stage generate   # GPU
python run.py --config configs/coco_internvl.yaml      --stage extract    # GPU
python run.py --config configs/coco_internvl.yaml      --stage score      # CPU
```

Swap the config to run another benchmark — same commands. Configs default to a **single
rollout**; the headline numbers below use N=16 self-consistency voting (run the GPU stages
once per seed, e.g. `--seeds 0 1 ... 15`, then `score` with `n_rollouts: 16`):

| Config | Benchmark / model | Single rollout | N=16 voting |
|---|---|---|---|
| `configs/coco_internvl.yaml` (+ `_voting`) | COCO det / InternVL3.5-8B | AUROC 0.873; mAP 36.2 | **mAP 41.9** |
| `configs/coco_qwen3vl.yaml` | COCO det / Qwen3-VL-8B | AUROC 0.902 | — |
| `configs/qvhighlights_qwen3vl.yaml` | QVHighlights / Qwen3-VL-8B | mAP 24.5 | **mAP 36.6, R@1@0.5 55.1** |
| `configs/charades_qwen3vl.yaml` | Charades-STA / Qwen3-VL-8B | R@1@0.5 44 | **R@1@0.5 55.4, R@1@0.3 76.3** |

COCO runs on **either model** — same `CocoDataset`, just a different `model:` — because models and
datasets are independent adapters (see [Extending](#extending-add-a-new-model-or-task)). Every
benchmark uses the same **decoupled** stages (`generate` may use vLLM or HF; `extract` is always
HF-eager). `configs/coco_internvl_voting.yaml` ships the 16-seed COCO setup ready to run. CLI flags
override any config for quick sweeps: `--seeds 0 1 2 3`, `--n 16`, `--agg sum`.
Datasets and paths: [`docs/DATA.md`](docs/DATA.md).

## Extending: add a new model or task

Models and datasets are **independent registries**; any valid (model × dataset) pair runs from a
config. You ask for a `(model, dataset)` pair and, if either is missing, the error tells you how
to add it. A model adapter never appears in a dataset adapter and vice versa — the dataset asks
the model, per `task` family (`"image_det"` | `"video_span"`), how to parse, mask, and read its
attention.

Adapters **self-register** with a decorator and are auto-discovered, so adding one is just a new
file — no central registry to edit:

```python
from mtla.registry import register_model
@register_model("myvlm")
class MyVLMAdapter(ModelAdapter):
    tasks = ("image_det",)
    ...
```

```python
from mtla import resolve
model, dataset = resolve("myvlm", "coco")   # unknown key -> error listing what's available
```

**A new model family** (`mtla/models/<name>.py`): subclass `ModelAdapter`, decorate it with
`@register_model("<name>")`, declare `tasks` / `model_id` / `attn_module_path`, implement `parse`,
the `ext_*` extraction callbacks (build inputs + enumerate predictions, find each prediction's
tokens `Q_p`, mask the proposal region, assemble the record), and `generate_script`/`extract_script`.

**A new task / dataset** (`mtla/data/<name>.py`): subclass `DatasetAdapter`, decorate it with
`@register_dataset("<name>")`, set `task`, and implement `load_items(cfg)` (owns file I/O),
`ground_truth`, `score(cfg, model)` (reuse `mtla.reduce_band` + `mtla.nms_fuse` + `mtla.eval`), and
`stage_cmd`. Add a `configs/<name>.yaml`.

**Decoupled stages.** Every stage is split: `generate` (vLLM or HF) writes `predictions.json`;
`extract` (always HF-eager) re-runs the model to capture attention. vLLM can't expose attention,
so this split is what lets generation stay fast while extraction stays faithful.

Full walkthrough with the exact contract and the smallest examples: [`docs/EXTENDING.md`](docs/EXTENDING.md).

## Repository layout

```
mtla/                  core package
  mtla_attn.py         the MTLA computation: eager-attn monkeypatch + per-item driver (image+video)
  score / mask / voting / eval / utils / viz     parameter-free MTLA building blocks
  registry.py          @register_model / @register_dataset + resolve(model, dataset)
  config.py            YAML -> RunConfig
  models/              model adapters: internvl, qwen3vl  (parse, region mask, attn hook)
  data/                dataset adapters: coco, qvhighlights, charades  (load, score)
  stages/              GPU drivers: image_extract / video_extract + per-model generate scripts
run.py                 unified CLI: --config <yaml> --stage {generate,extract,score}
configs/               one YAML per model x dataset
examples/demo.py       CPU demo on the bundled fixture (no GPU, no download)
fixtures/              small committed demo fixture
docs/                  METHOD.md, DATA.md, RESULTS.md, EXTENDING.md
third_party/           vendored Moment-DETR evaluation (MIT)
```

## Citation

A paper describing MTLA is in preparation; citation and link will be added here on release.
See [`CITATION.cff`](CITATION.cff).

## Acknowledgements

Builds on **SVAR** (Jiang et al., *Devils in Middle Layers ...*) as the global-attention
baseline, and uses **Qwen3-VL**, **InternVL**, and **Audio Flamingo 3** as the grounding
models. Video evaluation uses the **Moment-DETR** standalone evaluator (MIT, vendored under
`third_party/`).
