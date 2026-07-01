# Extending MTLA: add a model or a dataset

MTLA resolves a run from two independent registries — **models** (`mtla/models/`) and **datasets**
(`mtla/data/`). A YAML config names one of each; `resolve("qwen3vl", "coco")` returns the pair
after checking the dataset's `task` family is one the model supports. Adapters **self-register**
with a decorator and are auto-discovered, so adding one is just a new file — there is no central
list to edit.

```python
from mtla import resolve, available_models, available_datasets
available_models()      # ['internvl', 'qwen3vl']
available_datasets()    # ['charades', 'coco', 'qvhighlights']
model, dataset = resolve("qwen3vl", "coco")
# resolve("foo", "coco") -> ValueError listing what's available + how to add an adapter
```

A `task` is a coarse modality family, not a benchmark:
- `image_det` — image object detection (proposal region = a bbox)
- `video_span` — temporal grounding (proposal region = a `[start, end]` time span)

Using a family (not a benchmark) is what lets one dataset run on multiple models.

---

## The pipeline in one paragraph

Three stages — `generate.py`, `extract.py`, `score.py`, each `--config <yaml>`:
- **generate** (GPU; vLLM) writes `<predictions>/seed{K}/predictions.json` — a list of uniform,
  model-agnostic records `{id, prompt, response, gt, extra}`, where `response` is the **raw** model
  output. No parsing happens here.
- **extract** (GPU; HF eager attention) re-runs the model, parses each `response`, and writes
  `<features>/seed{K}/shard*.pt` — each prediction's localized attention as a `[L, H]` array
  (paper eqs. 2–3), plus its region, label, and hallucination flag.
- **score** (CPU; no model) reduces each `[L, H]` over heads + a middle-layer band (eq. 4), votes
  across rollouts, and computes the benchmark metric.

`generate.py`/`extract.py` are config-driven: the stage script reloads the config, `resolve()`s the
adapters, and asks the **dataset** for its items (`load_items`) and the **model** for the per-item
work (`build_request` + the extraction callbacks). The shared MTLA core (`mtla/mtla_attn.py`) owns
the common per-item flow for both image and video; the kernel is the same because "the modality
tokens inside the proposal region" is a flat index set either way. All of the score-stage
computation lives in `mtla/evaluate.py` + `mtla/metrics.py`, so dataset adapters do no computation.

---

## Add a new model family

Create `mtla/models/<name>.py`, subclass `ModelAdapter`, decorate it, and implement the pieces
below. `mtla/models/internvl.py` is the smallest complete `image_det` example;
`mtla/models/qwen3vl.py` shows one adapter serving both `image_det` and `video_span`.

```python
from .base import ModelAdapter, Prediction, hallucinated
from ..registry import register_model
from ..mtla_attn import CaptureState, install_capture

@register_model("myvlm")
class MyVLMAdapter(ModelAdapter):
    model_id = "org/MyVLM"
    attn_module_path = "transformers.models.myvlm.modeling_myvlm"  # whose eager_attention_forward we capture
    tasks = ("image_det",)

    def parse(self, response, task=None) -> list: ...              # raw text -> [Prediction]

    # vLLM generation
    def gen_processor(self): ...                                   # load the processor once per worker
    def build_request(self, proc, item, dataset, cfg): ...         # -> {prompt, multi_modal_data, ...}

    # HF-eager MTLA extraction (extract_one is inherited; it calls the callbacks below)
    def load_for_extract(self, gpu_id, task="image_det") -> dict: ...   # -> ctx (model, tokenizer, state, ...)
```

Install the attention-capture hook inside `load_for_extract`, and record each decoder layer's id:

```python
state = CaptureState()
install_capture(self.attn_module_path, state)                      # wraps the model's own eager forward
# ... load model/processor with attn_implementation="eager" ...
layers = model.model.language_model.layers
state.layer_ids = {id(L.self_attn): i for i, L in enumerate(layers)}
```

The extraction callbacks the shared driver (`compute_mtla`) calls — all plain data; the driver owns
Q_p assembly, the single captured forward, the MTLA math, and the buffer→record step:

| callback | returns |
|---|---|
| `build_inputs(record, ctx, rank)` | dict with `prompt_ids`, `response`, `modality_idx_l` (modality-token positions), `predictions` (from `parse`), `hallu_flags` (via `hallucinated(...)`), `meta`, plus anything `forward_kwargs` needs — or `None` to skip |
| `query_tokens(response, predictions, tokenizer)` | per prediction `{first_label_tok, label_toks, coord_toks}` (the tokens `Q_p`) or `None` |
| `region_mask(prediction, meta)` | the modality-token indices inside `prediction.region` — specific to your model's token layout |
| `forward_kwargs(full_ids, total_len, device, inp)` | kwargs for the single captured `model(**fk)` forward |

`prediction_record` / `item_record` are inherited from the base (they emit the generic
`{region, label, ...}` object and `{id, gt, extra, objects}` record); override only if your model
needs extra per-prediction fields.

---

## Add a new task / dataset

Create `mtla/data/<name>.py`, subclass `DatasetAdapter`, decorate it, and set the **declarative**
fields — the adapter does no computation, it only supplies data and *declares* how it is scored.

```python
from .base import DatasetAdapter
from ..registry import register_dataset

@register_dataset("mybench")
class MyBench(DatasetAdapter):
    name = "mybench"
    task = "image_det"                     # must be a family the chosen model supports
    # scoring descriptors (read by mtla.evaluate):
    signal = "local_attention"             # or "first_digit" (video)
    overlap = "iou"                        # "iou" (boxes) | "tiou" (spans)
    select = "fuse"                        # "fuse" (NMS pool across rollouts) | "argmax" (single-span)
    metric = "coco_map"                    # a pure computer in mtla.metrics
    gen_strategy = "pooled"                # "pooled" (many small requests) | "sharded" (heavy per item)

    def load_items(self, cfg) -> list:     # owns file I/O; reads cfg.path(...)
        ...
    def prompt(self, item) -> str: ...
    def ground_truth(self, item) -> list:  # [{"region": [...], "label": "..."}] (label "" for spans)
        ...
    def gen_record(self, cfg, item, response, truncated=False) -> dict:
        return {"id": item["id"], "prompt": self.prompt(item), "response": response,
                "gt": self.ground_truth(item), "extra": {"image": item["image"]}}
```

Then add `configs/mybench_myvlm.yaml`:

```yaml
model: myvlm
dataset: mybench
paths:   {data: ..., predictions: runs/mybench/predictions, features: runs/mybench/features}
n_rollouts: 1                          # one knob: generate/extract produce seeds 0..n-1, score votes
generate: {gpus: null, temperature: 0.7}   # gpus: null = all visible GPUs
extract:  {gpus: null, n_items: 5000}
score:    {agg: max}
band: [8, 21]
```

`python -m score --config configs/mybench_myvlm.yaml` now works. `mtla/data/coco.py` is the
smallest complete `image_det` example; `mtla/data/charades.py` the smallest `video_span` one.

### Reusing vs. adding a metric
`select` + `metric` cover the existing benchmarks (`coco_map`, `moment_retrieval`, `recall_at_iou`).
If your benchmark reuses one of these, there is nothing to add in `mtla/evaluate.py`. A genuinely
new metric is one pure function in `mtla/metrics.py` (fused candidates → numbers) plus one handler
in `mtla/evaluate.py`.

### Video specifics
A `video_span` run declares its vision **preprocessing** in the config (not the dataset), so the
same frames feed both the generate and extract stages:

```yaml
preprocess: {fps: 2.0, min_pixels: 4096, max_pixels: 131072}
```

`select="fuse"` marks a multi-window benchmark (QVHighlights); `select="argmax"` a single-span one
(Charades). The video MTLA signal defaults to `first_digit` (the validated choice); each predicted
window stores both `first_digit` and `local_attention` so either is available at score time.
