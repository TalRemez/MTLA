"""Render an MTLA attention heatmap over an image, with the predicted box drawn on top.

Given a coarse ``grid_h x grid_w`` attention map (the per-token attention to image patches,
averaged over the prediction's tokens), this upsamples and smooths it into a turbo overlay
so you can *see* whether the model looked inside the box it drew. Distilled from the figure
script used for the paper's qualitative panels.
"""
from __future__ import annotations

import numpy as np

RED = "#E7263F"
SIGMA = 1.2       # gaussian smoothing (in upscaled-patch units)
CLIP_PCT = 60     # percentile below which the overlay is fully transparent
FLOOR, AMAX = 0.45, 0.88  # min / max overlay opacity


def heatmap(grid_map, H: int, W: int):
    """Upsample a ``[grid_h, grid_w]`` attention map to image size ``(H, W)``, smooth it, and
    normalize to ``[0, 1]`` by the map's own peak.

    No upper clipping: this previously divided by the 99th percentile and clipped to 1, which
    flattened the hottest ~1% of cells into a saturated plateau and hid the falloff at the edges of
    the attended region. We instead keep the full positive range, then renormalize the *smoothed*
    map to its own maximum — the peak sits at 1.0 and the whole gradient (edges included) shows."""
    from scipy.ndimage import gaussian_filter, zoom

    g = np.asarray(grid_map, dtype=np.float32)
    gh, gw = g.shape
    up = zoom(np.clip(g, 0, None), (H / gh, W / gw), order=0)   # keep full range, drop negatives only
    up = gaussian_filter(up, SIGMA * max(H / gh, W / gw))
    return up / (up.max() + 1e-9)


def overlay(image, grid_map, box=None, out_path=None, title=None):
    """Save a turbo attention overlay of ``grid_map`` on ``image``.

    Args:
        image: HxWx3 uint8 array (or anything ``np.asarray`` turns into one).
        grid_map: coarse ``[grid_h, grid_w]`` attention map for one prediction.
        box: optional ``[x1,y1,x2,y2]`` in ``[0,1000]`` to draw (drawn in white).
        out_path: where to write the PNG. If ``None``, returns the composited array.
        title: optional title string.

    Returns:
        ``out_path`` if given, else the composited HxWx3 uint8 array.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = np.asarray(image)
    H, W = img.shape[:2]
    hm = heatmap(grid_map, H, W)
    cmap = plt.get_cmap("turbo")

    lo = np.percentile(hm, CLIP_PCT)
    t = np.clip((hm - lo) / (hm.max() - lo + 1e-9), 0, 1)
    alpha = np.where(hm <= lo, 0.0, FLOOR + (AMAX - FLOOR) * t)[..., None]
    comp = ((1 - alpha) * img.astype(np.float32)
            + alpha * (cmap(np.clip(hm, 0, 1))[..., :3] * 255)).clip(0, 255).astype(np.uint8)

    if out_path is None and box is None and title is None:
        return comp

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(comp, interpolation="none")
    if box is not None:
        x1, y1, x2, y2 = box[0] * W / 1000, box[1] * H / 1000, box[2] * W / 1000, box[3] * H / 1000
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, ec="white", fc="none", lw=2.5))
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=130)
        plt.close(fig)
        return out_path
    plt.close(fig)
    return comp
