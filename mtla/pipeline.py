"""Run a GPU stage script as a subprocess.

The generate/extract stage drivers are large, GPU-only, paper-faithful executables — not library
code. They live at the repo root under ``scripts/stages/`` (outside the importable ``mtla``
package) and are invoked as subprocesses, never imported. A dataset adapter's ``stage_cmd``
declares which script + args each stage needs; ``run_stage`` runs it.

The four shared drivers are config-driven (``--config <yaml> --seed <K>``): they resolve the
(model, dataset) adapters and delegate the model/task specifics to adapter callbacks.

  scripts/stages/image_generate.py   shared image_det generation (any image model; vLLM or HF)
  scripts/stages/image_extract.py    shared image_det MTLA extraction (any image model) — HF eager
  scripts/stages/video_generate.py   shared video_span generation (any video model; vLLM or HF)
  scripts/stages/video_extract.py    shared video_span MTLA extraction (any video model) — HF eager
  scripts/stages/qwen3vl_video.py    LEGACY QVHighlights monolith (parity reference; off the run path)
  scripts/stages/qwen3vl_charades.py LEGACY Charades-STA monolith (parity reference; off the run path)
"""
import os
import subprocess
import sys

# scripts/stages/ sits at the repo root, one level above the mtla package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES_DIR = os.path.join(_REPO_ROOT, "scripts", "stages")


def run_stage(script: str, args: list):
    """Run a stage script (by filename in ``scripts/stages/``) with string args. Raises on failure."""
    path = os.path.join(STAGES_DIR, script)
    cmd = [sys.executable, path, *[str(a) for a in args]]
    print(f"[stage] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
