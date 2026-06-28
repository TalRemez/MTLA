# Vendored: Moment-DETR standalone evaluation

`eval.py` and `utils.py` here are the official QVHighlights / moment-retrieval evaluation code
from **Moment-DETR** (Lei, Berg, Bansal, *Detecting Moments and Highlights in Videos via
Natural Language Queries*, NeurIPS 2021):

- Upstream: https://github.com/jayleicn/moment_detr (`standalone_eval/`)
- License: MIT (see `LICENSE`)

They are vendored unmodified so the QVHighlights example can compute official mAP / R@1
without an extra clone. All credit to the original authors; please cite Moment-DETR if you
use this evaluation.
