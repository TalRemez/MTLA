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
- `video_span` — temporal grounding (proposal region = a [start, end] time span)

Using a family (not a benchmark) is what lets one dataset run on multiple models.

---

## The pipeline in one paragraph

Three stages, all driven by `run.py --config <yaml> --stage {generate,extract,score}`:
- **generate** (GPU; vLLM or HF) writes `<predictions>/seed{K}/predictions.json`.
- **extract** (GPU; HF eager attention) re-runs the model and writes `<features>/seed{K}/shard*.pt`
  — each prediction's localized attention as a `[L, H]` array (paper eqs. 2–3).
- **score** (CPU; no model) reduces `[L, H]` over heads + a middle-layer band (eq. 4) to one
  scalar per prediction and computes the benchmark metrics.

`generate`/`extract` are config-driven subprocesses: the stage script reloads the config,
`resolve()`s the adapters, and asks the **dataset** for its items (`load_items`) and the **model**
for the per-item MTLA work (`ext_*`). The shared MTLA core (`mtla/mtla_attn.py`,
`compute_mtla`) owns the common per-item flow for both image and video; the kernel is the same
because "the modality tokens inside the proposal region" is a flat index set either way.

---

## Add a new model family

Create `mtla/models/<name>.py`, subclass `ModelAdapter`, decorate it, and implement the pieces
below. `mtla/models/internvl.py` is the smallest complete `image_det` example;
`mtla/models/qwen3vl.py` shows one adapter serving both `image_det` and `video_span`.

```python
from .base import ModelAdapter, Prediction
from ..registry import register_model

@register_model("myvlm")
class MyVLMAdapter(ModelAdapter):
    model_id = "org/MyVLM"
    attn_module_path = "transformers.models.myvlm.modeling_myvlm"  # whose eager_attention_forward we hook
    tasks = ("image_det",)

    def parse(self, response, task=None, **kw) -> list: ...        # raw text -> [Prediction]
    def generate_script(self, task, engine) -> str: ...            # filename in mtla/stages/
    def extract_script(self, task) -> str: return "image_extract.py"

    # HF-eager extraction: delegate to the shared driver, which calls the ext_* below.
    def load_for_extract(self, gpu_id, task="image_det") -> dict: ...   # -> ctx (model, tokenizer, MTLAState, ...)
    def extract_one(self, p, ds_by_id, ctx, svar_shift, rank=0):
        from ..mtla_attn import compute_mtla
        return compute_mtla(self, p, ds_by_id, ctx, svar_shift, rank)
```

The `ext_*` callbacks the shared driver calls (all plain data; the driver owns Q_p assembly, the
single forward, and the buffer→record step):

| callback | returns |
|---|---|
| `ext_build_inputs(p, ds_by_id, ctx, rank)` | dict with `prompt_ids`, `response`, `modality_idx_l` (the modality-token positions), `predictions` (boxes/windows), `hallu_flags` (aligned bools), `meta`, plus anything `ext_forward_kwargs` needs — or `None` to skip |
| `ext_token_ranges(response, predictions, tokenizer)` | per prediction `{first_label_tok, label_toks, coord_toks}` (the tokens `Q_p`) or `None` |
| `ext_region_mask(prediction, meta)` | the modality-token indices inside that prediction's region `M(R_p)` (delegate to `mtla.mask`) |
| `ext_forward_kwargs(full_ids, total_len, device, inp)` | kwargs for the single patched `model(**fk)` forward |
| `ext_obj_record(prediction, pred_idx, meta)` | the per-prediction record fields (the driver adds `is_hallucinated` / `n_qp_tokens` / `local_attention` / `first_digit`) |
| `ext_record(p, meta, objects, n_predictions)` | the top-level saved record (id keys + counts + `objects`) |

Install the MTLA attention forward inside `load_for_extract`:

```python
from ..mtla_attn import MTLAState, make_mtla_attention_forward, install
state = MTLAState()
install(self.attn_module_path, make_mtla_attention_forward(state))   # repeat_kv defaults to mtla.utils.repeat_kv
# ... load model/tokenizer, record decoder-layer ids on `state` ...
```

---

## Add a new task / dataset

Create `mtla/data/<name>.py`, subclass `DatasetAdapter`, decorate it, set `task`, and implement:

```python
from .base import DatasetAdapter
from ..registry import register_dataset

@register_dataset("mybench")
class MyBench(DatasetAdapter):
    name = "mybench"
    task = "image_det"                     # must be a family the chosen model supports

    def load_items(self, cfg) -> list:     # owns file I/O; reads cfg.path(...)
        ...
    def ground_truth(self, item): ...
    def stage_cmd(self, cfg, model, seed, mode):
        # uniform, config-driven: the model names the script, we pass --config/--seed
        args = ["--config", cfg.config_path, "--seed", str(seed)]
        script = (model.generate_script(self.task, cfg.generate.engine) if mode == "generate"
                  else model.extract_script(self.task))
        return script, args
    def score(self, cfg, model) -> dict:   # CPU; read shards via self.load_shards(cfg.feat_dir(s))
        ...                                 # reduce_band(obj["local_attention"]) + nms_fuse + eval
```

Then add `configs/mybench_myvlm.yaml`:

```yaml
model: myvlm
dataset: mybench
paths:   {data: ..., predictions: runs/mybench/predictions, features: runs/mybench/features}
generate: {engine: vllm, gpus: [0,1]}
extract:  {gpus: [0,1], n_items: 5000}
score:    {n_rollouts: 1, agg: max}
band: [8, 21]
```

`python run.py --config configs/mybench_myvlm.yaml --stage score` now works. `mtla/data/coco.py`
is the smallest complete `image_det` example; `mtla/data/charades.py` the smallest `video_span` one.

### Video specifics
A `video_span` dataset also declares its sampling on the adapter (read by the Qwen3-VL video
`ext_*`), and normalizes its own prediction records:

```python
video = {"fps": 2.0, "min_pixels": 4*32*32, "max_pixels": 128*32*32,
         "max_new_tokens": 128, "multi": False}   # multi=True for multi-window benchmarks (QVH)
def video_item(self, p, video_dir) -> dict:        # -> {video_path, query, pred_windows, gt_windows}
    ...
```

The video MTLA signal defaults to `first_digit` (the validated choice); each predicted window
stores both `first_digit` and `local_attention` so either is available at score time.
