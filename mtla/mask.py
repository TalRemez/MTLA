"""Region masks: map a prediction's proposal region to the modality-token indices inside it.

This is the only modality-specific part of MTLA. Given a predicted region we return the
indices of the input tokens that fall inside it (``M(R_p)`` in the paper); the extractor
then restricts attention aggregation to those indices.

  * images, fixed patch grid (Qwen3-VL):      ``bbox_to_patch_indices``
  * images, dynamic tiling (InternVL3.5):      ``bbox_to_internvl_token_indices``
  * video / audio, 1-D token timeline:         ``span_to_token_indices``

Bounding boxes are in the ``[0, 1000]`` normalized space these grounding MLLMs are trained
to emit. Temporal spans are in seconds.
"""
from __future__ import annotations

import numpy as np


def bbox_to_patch_indices(bbox, grid_h: int, grid_w: int):
    """Indices of patches overlapping ``bbox`` on a fixed ``grid_h x grid_w`` row-major grid.

    Used for models that encode an image as a single rectangular patch grid (e.g. Qwen3-VL,
    where ``grid_h``/``grid_w`` come from ``image_grid_thw``). ``bbox`` is ``[x1,y1,x2,y2]``
    in ``[0, 1000]``. Returns ``(inside, outside)`` index lists into ``range(grid_h*grid_w)``.
    """
    x1, y1, x2, y2 = bbox
    total = grid_h * grid_w
    if x2 <= x1 or y2 <= y1:
        return [], list(range(total))
    col_min = int(np.floor(x1 * grid_w / 1000.0))
    col_max = int(np.floor((x2 - 1e-6) * grid_w / 1000.0))
    row_min = int(np.floor(y1 * grid_h / 1000.0))
    row_max = int(np.floor((y2 - 1e-6) * grid_h / 1000.0))
    col_min, col_max = max(0, min(grid_w - 1, col_min)), max(0, min(grid_w - 1, col_max))
    row_min, row_max = max(0, min(grid_h - 1, row_min)), max(0, min(grid_h - 1, row_max))
    inside = [r * grid_w + c for r in range(row_min, row_max + 1)
              for c in range(col_min, col_max + 1)]
    inside_set = set(inside)
    outside = [i for i in range(total) if i not in inside_set]
    return inside, outside


def bbox_to_internvl_token_indices(bbox, tile_grid, has_thumb: bool, patch_grid: int = 16):
    """Indices of image tokens overlapping ``bbox`` under InternVL dynamic tiling.

    InternVL splits an image into ``n_cols x n_rows`` tiles plus an optional thumbnail. The
    image-token sequence is ``tile_0[0..G-1], tile_1[0..G-1], ..., [thumbnail[0..G-1]]``, each
    tile a ``patch_grid x patch_grid`` row-major grid of ``G = patch_grid**2`` tokens.
    ``tile_grid`` is ``(n_cols, n_rows)``; ``patch_grid`` is the per-tile side (16 for
    InternVL3.5, i.e. 16x16=256 tokens per 448px tile after pixel-shuffle 0.5).
    ``bbox`` is ``[x1,y1,x2,y2]`` in ``[0, 1000]``. Returns ``(inside, outside)``.
    """
    x1, y1, x2, y2 = bbox
    n_cols, n_rows = tile_grid
    n_tiles = n_cols * n_rows
    per_tile = patch_grid * patch_grid
    total = n_tiles * per_tile + (per_tile if has_thumb else 0)
    if x2 <= x1 or y2 <= y1:
        return [], list(range(total))

    def clamp(v):
        return max(0, min(patch_grid - 1, v))

    bx1, by1, bx2, by2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
    inside = []
    for tile_idx in range(n_tiles):
        col, row = tile_idx % n_cols, tile_idx // n_cols
        tx0, tx1 = col / n_cols, (col + 1) / n_cols
        ty0, ty1 = row / n_rows, (row + 1) / n_rows
        if tx1 <= bx1 or tx0 >= bx2 or ty1 <= by1 or ty0 >= by2:
            continue  # tile does not overlap bbox
        # bbox in tile-local [0,1] coordinates
        lx0 = max(0.0, (bx1 - tx0) / (tx1 - tx0))
        lx1 = min(1.0, (bx2 - tx0) / (tx1 - tx0))
        ly0 = max(0.0, (by1 - ty0) / (ty1 - ty0))
        ly1 = min(1.0, (by2 - ty0) / (ty1 - ty0))
        col_min = clamp(int(np.floor(lx0 * patch_grid)))
        col_max = clamp(int(np.floor((lx1 - 1e-6) * patch_grid)))
        row_min = clamp(int(np.floor(ly0 * patch_grid)))
        row_max = clamp(int(np.floor((ly1 - 1e-6) * patch_grid)))
        off = tile_idx * per_tile
        for pr in range(row_min, row_max + 1):
            for pc in range(col_min, col_max + 1):
                inside.append(off + pr * patch_grid + pc)
    if has_thumb:
        col_min = clamp(int(np.floor(bx1 * patch_grid)))
        col_max = clamp(int(np.floor((bx2 - 1e-6) * patch_grid)))
        row_min = clamp(int(np.floor(by1 * patch_grid)))
        row_max = clamp(int(np.floor((by2 - 1e-6) * patch_grid)))
        off = n_tiles * per_tile
        for pr in range(row_min, row_max + 1):
            for pc in range(col_min, col_max + 1):
                inside.append(off + pr * patch_grid + pc)
    inside_set = set(inside)
    outside = [i for i in range(total) if i not in inside_set]
    return inside, outside


def span_to_token_indices(span, duration_s: float, n_tokens: int):
    """Indices of timeline tokens whose time falls within a temporal ``span``.

    For video and audio the modality tokens form a 1-D timeline of ``n_tokens`` frames over
    ``duration_s`` seconds; token ``t`` covers time ``t * duration_s / n_tokens``. ``span`` is
    ``[t_start, t_end]`` in seconds. Returns ``(inside, outside)`` index lists.

    (Audio Flamingo 3 instead emits tokens at a fixed 25 Hz; there ``duration_s`` should be
    ``n_tokens / 25`` so the mapping is content-proportional.)
    """
    if span is None or duration_s <= 0 or n_tokens <= 0:
        return [], list(range(max(0, n_tokens)))
    s, e = span
    fs = max(0, int(np.floor(s * n_tokens / duration_s)))
    fe = min(n_tokens, int(np.ceil(e * n_tokens / duration_s)))
    if fe <= fs:
        return [], list(range(n_tokens))
    inside = list(range(fs, fe))
    inside_set = set(inside)
    outside = [i for i in range(n_tokens) if i not in inside_set]
    return inside, outside
