"""GPU stage scripts (generation + attention extraction).

Large, GPU-only, paper-faithful scripts kept standalone (not imported) and invoked as
subprocesses by `run_stage(...)`. Dataset adapters' `stage_cmd` declare which script + args each
stage needs. All four shared drivers are config-driven (`--config <yaml> --seed <K>`): they
resolve the (model, dataset) adapters and delegate the model/task specifics to adapter callbacks
(`ext_*` for extract; `make_vllm_prep` / `make_hf_generate` / `generate_video` for generate).

  image_generate.py         shared image_det generation (any image model; vLLM or HF)
  image_extract.py          shared image_det MTLA extraction (any image model) — HF eager
  video_generate.py         shared video_span generation (any video model; vLLM or HF)
  video_extract.py          shared video_span MTLA extraction (any video model) — HF eager

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
