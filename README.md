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

## Method

MTLA is a **training-free, post-hoc** confidence score for grounding predictions from
multimodal LLMs. It needs no extra parameters, no fine-tuning, and no auxiliary model: it
reads the model's own attention from the same forward pass that produced the prediction.

**Problem setup.** A grounding MLLM autoregressively emits one or more localized predictions.
Each prediction `p` has a **proposal region** `R_p` and a label. Depending on the modality, `R_p` is

- a bounding box `[x1, y1, x2, y2]` in an image (coordinates in `[0, 1000]`), or
- a temporal interval `[t_start, t_end]` in video or audio.

A prediction is **hallucinated** if its region matches no ground-truth region under the task's
criterion (IoU ≥ 0.5 throughout). The goal is a scalar score `s(p)`, computed from the model's
attention, that is high for grounded predictions and low for hallucinations.

**Localized Attention (LA).** Let the transformer have `L` layers and `H` heads, and let
`X = {k_1, ..., k_N}` be the input-modality tokens (image patches, video frames, or audio
frames). For a response token at position `q`, the model produces attention weights
`a[l,h](q -> k)` over `k ∈ X`. The key idea is to restrict the attention sum to the tokens
*inside* the proposal region, `M(R_p)` (image patches overlapping the box; frames whose
timestamps fall in the span):

```
LA[l,h](q) = sum over k in M(R_p) of  a[l,h](q -> k)
```

A grounded prediction attends strongly to evidence inside its own region; a hallucination
relies on context scattered elsewhere, so its localized attention stays low even when its
*global* attention (the [SVAR](#acknowledgements) baseline, which sums over all of `X`) is comparable.

**Multi-token aggregation.** A prediction spans several tokens (the digits of each coordinate,
plus the label). Any single token's attention is noisy; averaging across the prediction's
tokens `Q_p` is far more robust — this is **Multi-Token Localized Attention**:

```
MTLA[l,h](p) = (1/|Q_p|) * sum over q in Q_p of  LA[l,h](q)
```

**Layer and head reduction.** Average over heads and over a fixed band of middle layers to get
one scalar:

```
s(p) = mean over l in band of  ( (1/H) * sum over h of  MTLA[l,h](p) )
```

The default band is **layers 8–21** (`mtla.DEFAULT_BAND`), used for every image and video model
tested (Qwen3-VL, InternVL: 36 layers; Gemma-4: 42). Audio (Audio Flamingo 3, 28 layers) uses
**all** layers. MTLA is not very sensitive to the exact band (see the paper's ablation). The whole
reduction is `mtla.reduce_band`.

**Self-consistency voting.** Sampling `N` stochastic rollouts per input enlarges the candidate
pool (better recall). We pool predictions across rollouts, merge overlaps with non-maximum
suppression, and score each kept prediction from its cluster's MTLA values (`mtla.nms_fuse`):

- **max** (default): keep the single highest-scoring rollout. Used for video and audio.
- **sum**: sum the cluster's scores, rewarding regions that recur across rollouts. Used for
  **COCO detection only**, where each image yields many predictions; there it beats max.

## Results

Headline numbers reproduced by the example pipelines. MTLA is the inside-region attention score
(ours); **SVAR** (Jiang et al.) is the global-attention baseline we compare against. All use the
default middle-layer band (L8–21) except AudioSet, which uses all 28 layers. IoU ≥ 0.5 throughout.

### Task accuracy after MTLA re-ranking / self-consistency voting

One method, four modalities, no training. Re-ranking each benchmark's `N=16` stochastic rollouts
by MTLA also improves the **standard task metric** — the paper reports AP for COCO detection and
QVHighlights, R@1@0.5 for Charades-STA, and PSDS1 for AudioSet-Strong:

| Benchmark | Model | Metric | MTLA | SVAR baseline | Supervised reference |
|---|---|---|--:|--:|--:|
| COCO detection | InternVL3.5-8B | AP | **41.9** | 32.7 | 42.0 *(DETR)* |
| QVHighlights (video) | Qwen3-VL-8B | mAP | **36.6** | 28.1 | — |
| Charades-STA (video) | Qwen3-VL-8B | R@1@0.5 | **55.4** | 43.8 | — |
| AudioSet-Strong (audio) | Audio Flamingo 3 | PSDS1 | **0.26** | 0.23 | 0.33 *(BEATs)* |

Zero-shot and training-free, MTLA reaches the supervised range: it matches DETR on COCO, lands
between Moment-DETR and QD-DETR on the video benchmarks, and approaches a supervised sound-event
detector on AudioSet.

### Hallucination detection — AUROC (single rollout)

How well the score separates grounded from hallucinated predictions.

| Benchmark | Model | MTLA | SVAR baseline |
|---|---|--:|--:|
| COCO detection | Qwen3-VL-8B | **0.902** | 0.763 |
| COCO detection | InternVL3.5-8B | **0.873** | 0.803 |
| COCO detection | Gemma-4 E4B | **0.753** | 0.671 |
| QVHighlights (video) | Qwen3-VL-8B | **0.800** | 0.415 |
| Charades-STA (video) | Qwen3-VL-8B | **0.684** | 0.512 |
| AudioSet-Strong (audio) | Audio Flamingo 3 | **0.813** | 0.608 |

`run.py --config configs/coco_internvl.yaml --stage score` reproduces the InternVL row and
`configs/coco_qwen3vl.yaml` the Qwen3-VL row — same `CocoDataset`, different `model:`.

> Numbers are from the paper (the citable source of record; link to be added on release); the COCO
> and QVHighlights rows are reproducible end-to-end with the example scripts.

## Use it on your own predictions

The extract stage saves, per prediction, a `[L, H]` array `local_attention`: the attention its
tokens pay to the modality tokens **inside** its proposed region. Scoring it is CPU-only —
`mtla_score` reduces the array to one scalar (mean over heads, mean over the layer band); higher
means more grounded.

```python
import glob, torch
from mtla import mtla_score, auroc

objs = [o for f in glob.glob("runs/coco/features/seed0/shard*.pt")
        for r in torch.load(f, weights_only=False) for o in r["objects"]]
scores = [mtla_score(o) for o in objs]                 # [L,H] -> scalar per prediction
labels = [o["is_hallucinated"] for o in objs]          # IoU>=0.5 flags
print(f"AUROC = {auroc(scores, labels):.3f}")
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
| `mtla.mtla_attn` | the eager-attention monkeypatch + per-item driver that capture localized attention |
| `mtla.voting` | self-consistency NMS fusion across rollouts (`max` / `sum` / ...) |
| `mtla.eval` | hallucination AUROC and COCO mAP |
| `mtla.utils` | shared primitives: `iou` / `tiou`, `repeat_kv`, token-span helpers |
| `mtla.viz` | attention heatmap overlays |

## Full reproduction — one config-driven pipeline

Every benchmark runs through the same three stages; a YAML config picks the model + dataset.
No data or bulk features are shipped; you download the datasets and regenerate the features
(GPU needed for `generate`/`extract`, `score` is CPU-only).

**Prepare the data.** One script per dataset downloads the source and writes the files the
adapters load, into a repo-relative `data/` directory (the default the configs point at). Each
script prints the exact `paths:` to copy into your config; video clips are large and fetched
separately (the scripts say where). Details: [`docs/DATA.md`](docs/DATA.md).

```bash
python scripts/prepare_coco.py          # images + instances_val2017.json + the open-vocab JSON
python scripts/prepare_qvhighlights.py  # val annotations (jsonl); add the videos yourself
python scripts/prepare_charades.py --from-annotations charades_sta_test.txt   # test parquet
```

`prepare_coco.py` builds the open-vocabulary detection JSON straight from the official COCO
annotations (per-image present classes, GT boxes scaled to `[0, 1000]`, `iscrowd` excluded), so
the dataset is fully reproducible rather than a shipped blob.

**Run the three stages.** Swap the config to run another benchmark — same commands:

```bash
python run.py --config configs/coco_internvl.yaml      --stage generate   # GPU
python run.py --config configs/coco_internvl.yaml      --stage extract    # GPU
python run.py --config configs/coco_internvl.yaml      --stage score      # CPU
```

Swap the config to run another benchmark — same commands. Configs default to a **single
rollout**; the headline numbers above use N=16 self-consistency voting (run the GPU stages
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
mtla/                  core importable library (pip install mtla)
  mtla_attn.py         the MTLA computation: eager-attn monkeypatch + per-item driver (image+video)
  score / mask / voting / eval / utils / viz     parameter-free MTLA building blocks
  registry.py          @register_model / @register_dataset + resolve(model, dataset)
  config.py            YAML -> RunConfig
  pipeline.py          run_stage: launch a GPU stage script as a subprocess
  models/              model adapters: internvl, qwen3vl  (parse, region mask, attn hook)
  data/                dataset adapters: coco, qvhighlights, charades  (load, score)
run.py                 unified CLI: --config <yaml> --stage {generate,extract,score}
configs/               one YAML per model x dataset
scripts/
  prepare_*.py         one dataset-prep script per benchmark (download + build)
  stages/              GPU pipeline drivers: image/video_{generate,extract} (config-driven)
docs/                  DATA.md, EXTENDING.md
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
