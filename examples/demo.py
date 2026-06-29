"""60-second CPU demo: no GPU, no model weights, no dataset download.

Loads a small committed fixture of pre-extracted attention (InternVL3.5-8B on COCO) and:
  1. scores every prediction with MTLA (inside-region attention) and the SVAR baseline
     (global attention), and reports how well each separates grounded vs. hallucinated
     predictions (AUROC);
  2. renders an attention heatmap for one image, showing the model looking inside a
     grounded box but *not* inside a hallucinated one.

Run:
    pip install -e ".[demo,coco]"
    python examples/demo.py
"""
import os

import numpy as np
import torch

from mtla import auroc, reduce_band

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "fixtures", "coco_demo.pt")
OUT_DIR = os.path.join(HERE, "output")


def main():
    data = torch.load(FIXTURE, weights_only=False)
    scoring, viz, meta = data["scoring"], data["viz"], data["meta"]
    band = meta["band_default"]
    print(f"MTLA demo  |  {meta['model']} on {meta['benchmark']}  |  layer band {band[0]}-{band[-1]}")
    print(f"{len(scoring)} pre-extracted predictions "
          f"({sum(not p['is_hallucinated'] for p in scoring)} grounded, "
          f"{sum(p['is_hallucinated'] for p in scoring)} hallucinated)\n")

    # ---- 1. hallucination AUROC: MTLA vs SVAR ----
    # Each record stores two [L,H] arrays: image_inside_sum (MTLA, inside-region attention over
    # all the prediction's tokens) and image_sum (SVAR baseline, global attention).
    labels = [p["is_hallucinated"] for p in scoring]
    mtla = [reduce_band(p["image_inside_sum"], band) for p in scoring]
    svar = [reduce_band(p["image_sum"], band) for p in scoring]
    auroc_mtla = auroc(mtla, labels)
    auroc_svar = auroc(svar, labels)
    print("Hallucination detection AUROC (higher = better separation)")
    print(f"   MTLA  (inside-region attention, ours):  {auroc_mtla:.3f}")
    print(f"   SVAR  (global attention, baseline):     {auroc_svar:.3f}")
    print(f"   -> MTLA improves separation by {auroc_mtla - auroc_svar:+.3f} AUROC\n")

    # ---- 2. attention heatmap on one image ----
    try:
        from PIL import Image
        from mtla.viz import overlay
    except ImportError:
        print("(install the 'demo' extra for the heatmap: pip install -e '.[demo]')")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    img_path = os.path.join(HERE, "..", "fixtures", viz["image_file"])
    img = np.asarray(Image.open(img_path).convert("RGB"))
    print(f"Rendering attention heatmaps for image {viz['image_id']} -> {OUT_DIR}/")
    for i, p in enumerate(viz["preds"]):
        tag = "HALLUCINATION" if p["is_hallucinated"] else "grounded"
        out = os.path.join(OUT_DIR, f"{viz['image_id']}_{i}_{p['label']}_{tag}.png")
        overlay(img, p["mean_map"], box=p["box"], out_path=out,
                title=f"{p['label']}  ({tag})")
        print(f"   {p['label']:<10} {tag:<14} -> {os.path.basename(out)}")
    print("\nOpen the PNGs: grounded predictions concentrate attention inside the box;")
    print("the hallucinated one scatters it across the scene.")


if __name__ == "__main__":
    main()
