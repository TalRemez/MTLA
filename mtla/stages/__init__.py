"""GPU stage scripts (generation + attention extraction).

These are the validated, paper-faithful pipeline scripts, lightly cleaned of hardcoded paths
so they are driven by command-line args. They are large and GPU-only, so they are kept as
standalone scripts (not imported) and invoked by `run_stage(...)`. Dataset adapters declare
which script + args each stage needs.

  internvl_generate.py      COCO detection generation, InternVL (vLLM; engine: vllm)
  internvl_generate_hf.py   COCO detection generation, InternVL (HF; engine: hf)
  internvl_extract.py       COCO detection attention extraction, InternVL (HF eager)
  qwen3vl_det_generate.py   COCO detection generation, Qwen3-VL (vLLM)
  qwen3vl_det_extract.py    COCO detection attention extraction, Qwen3-VL (HF eager)
  qwen3vl_video.py          QVHighlights generate|extract via --mode (HF eager)
  qwen3vl_charades.py       Charades-STA generate|extract via --mode (HF eager)
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
