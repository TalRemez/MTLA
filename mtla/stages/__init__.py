"""GPU stage scripts (generation + attention extraction).

Large, GPU-only, paper-faithful scripts kept standalone (not imported) and invoked as
subprocesses by `run_stage(...)`. Dataset adapters' `stage_cmd` declare which script + args each
stage needs. The shared extract drivers are config-driven (`--config <yaml> --seed <K>`): they
resolve the (model, dataset) adapters and delegate the model/task specifics to `ext_*` callbacks.

  image_extract.py          shared image_det MTLA extraction (any image model) — HF eager
  video_extract.py          shared video_span MTLA extraction (any video model) — HF eager
  internvl_generate.py      COCO detection generation, InternVL (vLLM; engine: vllm)
  internvl_generate_hf.py   COCO detection generation, InternVL (HF; engine: hf)
  qwen3vl_det_generate.py   COCO detection generation, Qwen3-VL (vLLM)

  qwen3vl_video.py          LEGACY QVHighlights generate|extract monolith (kept as the parity
  qwen3vl_charades.py       LEGACY Charades-STA   generate|extract monolith   reference for the
                            video_extract.py GPU equivalence gate; not on the run path).
"""
import os
import subprocess
import sys

STAGES_DIR = os.path.dirname(os.path.abspath(__file__))


def run_stage(script: str, args: list):
    """Run a stage script (by filename in this dir) with string args. Raises on failure."""
    path = os.path.join(STAGES_DIR, script)
    cmd = [sys.executable, path, *[str(a) for a in args]]
    print(f"[stage] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
