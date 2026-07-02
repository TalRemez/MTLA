"""Reproduce the paper's per-token attention figure (Fig. 3).

For each chosen COCO prediction, one row shows WHERE each of its response tokens attends over the
image — the four bounding-box coordinate sub-tokens `x1,y1,x2,y2`, the label token, and (right of
a dashed rule) their per-token mean. Grounded predictions concentrate attention inside the proposed
box `R_p`; hallucinations spread it across the scene. This is the qualitative motivation for MTLA:
the grounding signal is multi-token and lives inside the region.

This is a standalone GPU script (Qwen3-VL, HF-eager): it runs one forward per image and captures
the full per-token spatial attention `[n_token, L, H, n_patch]`, then reduces each token group to a
`grid_h x grid_w` map (mean over the group's sub-tokens, the middle-layer band L8–21, and heads).
Unlike the `extract` stage it keeps the map *spatial* (no inside-region masking) — the figure is
about where attention lands. Rendering reuses `mtla.utils.heatmap` (upsample + smooth) under a
turbo overlay.

    python -m scripts.figure_pertoken --config configs/coco_qwen3vl.yaml
    python -m scripts.figure_pertoken --config configs/coco_qwen3vl.yaml \
        --targets 64499:0:grounded 60823:3:hallu --out fig3.pdf

Default targets are the paper's Fig. 3 (grounded zebra + bird; hallucinated cow→"horse" + phantom
dining-table); pass `--targets <image_id>:<pred_idx>:<grounded|hallu> ...` to render your own.
"""

import argparse
import json
import os
import sys

import numpy as np

# Paper Fig. 3 examples: (image_id, pred_idx, grounded?). COCO val2017 ids; both land on the
# paper's object by index under the public parser (zebra grounded; a cow labeled "horse").
DEFAULT_TARGETS = [
    (64499, 0, True),  # grounded: zebra
    (60823, 3, False),
]  # hallucinated: cow→"horse"

BAND = list(range(8, 22))  # middle-layer band (mtla.DEFAULT_BAND)
COLS = ["prediction", "$x_1$", "$y_1$", "$x_2$", "$y_2$", "label", "mean"]
GROUNDED_C = "#1F6FEB"
HALLU_C = "#E7263F"

# Turbo-overlay opacity ramp: transparent below the CLIP_PCT percentile, then ramping
# linearly from FLOOR to AMAX opacity up to the peak.
CLIP_PCT = 60  # percentile below which the overlay is fully transparent
FLOOR, AMAX = 0.45, 0.88  # min / max overlay opacity


def parse_targets(specs):
    """Parse ``--targets`` spec strings into figure target tuples.

    Args:
        specs: a list of ``"<image_id>:<pred_idx>:<grounded|hallu>"`` strings; the third
            field is optional and defaults to grounded (any value starting with ``g`` is
            treated as grounded).

    Returns:
        A list of ``(image_id, pred_idx, grounded_bool)`` tuples.
    """
    out = []
    for s in specs:
        iid, pi, *rest = s.split(":")
        grounded = (rest[0].lower().startswith("g")) if rest else True
        out.append((int(iid), int(pi), grounded))
    return out


def find_coord_groups(response, pred_bboxes, tokenizer):
    """Locate the response token indices of each prediction's coords and label.

    For each predicted box, regex-matches its ``"bbox_2d":[...],"label":"..."`` span in
    the response, then maps the four coordinate fields and the label back to the
    response token indices that overlap each span (using the tokenizer's offset
    mapping). Mirrors the paper's figure extractor.

    Args:
        response: the model's raw response text.
        pred_bboxes: the parsed predicted boxes, each a dict with a ``label`` field; the
            search advances left-to-right so repeated labels are matched in order.
        tokenizer: a fast tokenizer supporting ``return_offsets_mapping``.

    Returns:
        A list aligned with ``pred_bboxes``; each entry is either ``None`` (the span was
        not found) or a dict with ``coord_groups`` (four lists of token indices, one per
        coordinate) and ``label_toks`` (the label's token indices).
    """
    import re

    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    tmpl = (
        r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,'
        r'\s*"label"\s*:\s*"{l}"'
    )
    out, sp = [], 0
    for pb in pred_bboxes:
        lab = pb["label"]
        pat = re.compile(tmpl.format(l=re.escape(lab)))
        m = pat.search(response, sp) or pat.search(response)
        if not m:
            out.append(None)
            continue
        coord_groups = [
            [ti for ti, (ts, te) in enumerate(off) if ts < m.end(g) and te > m.start(g)]
            for g in range(1, 5)
        ]
        ls, le = m.end() - 1 - len(lab), m.end() - 1
        label_toks = [ti for ti, (ts, te) in enumerate(off) if ts < le and te > ls]
        sp = m.end()
        out.append({"coord_groups": coord_groups, "label_toks": label_toks})
    return out


def build_hook(state):
    """Build an eager-attention hook that captures per-query attention to image tokens.

    The returned hook replaces Qwen3-VL's ``eager_attention_forward``: it recomputes the
    softmax attention (numerically identical to the stock forward and returns the same
    output), and, when armed, stashes the selected query positions' attention over the
    image tokens into ``state["buf"]`` shaped ``[Nq, L, H, n_img]``.

    Args:
        state: a mutable dict the caller arms per image, with keys ``active`` (whether
            to capture), ``ids``/``order`` (the attention modules to record and their
            layer order), ``qpos`` (query positions), ``imgidx`` (image-token indices),
            and ``buf`` (the preallocated output tensor).

    Returns:
        The ``hook(module, query, key, value, ...)`` callable to install.
    """
    import torch
    import torch.nn as nn
    from mtla.utils import repeat_kv

    def hook(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = repeat_kv(key, module.num_key_value_groups)
        vs = repeat_kv(value, module.num_key_value_groups)
        aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None:
            aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
        aw = nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
        s = state
        if s["active"] and id(module) in s["ids"]:
            li = s["order"].index(id(module))
            rows = (
                aw[0].index_select(1, s["qpos"]).transpose(0, 1).float()
            )  # [Nq, H, K]
            s["buf"][:, li, :, :] = rows.index_select(
                2, s["imgidx"]
            ).cpu()  # [Nq, L, H, n_img]
        out = (
            torch.matmul(
                nn.functional.dropout(aw, p=dropout, training=module.training), vs
            )
            .transpose(1, 2)
            .contiguous()
        )
        return out, None

    return hook


def main():
    """Render the per-token attention figure for the chosen COCO predictions.

    Loads Qwen3-VL with the eager-attention capture hook, runs one forward per target
    image, reduces each token group (coords, label, and their mean) to a spatial
    attention map, and lays the rows out into the paper's Fig. 3, grouped by
    grounded/hallucinated. Requires the Qwen3-VL COCO config and existing predictions;
    writes the figure to ``--out`` (plus a ``.png``).
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", required=True, help="a COCO config (for model + dataset paths)"
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="<image_id>:<pred_idx>:<grounded|hallu> ... (default: the paper's Fig. 3)",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="figure_pertoken.pdf")
    ap.add_argument(
        "--norm",
        choices=["per-column", "per-map"],
        default="per-column",
        help="per-column: shared scale per token across rows (paper); per-map: each panel by its own peak",
    )
    args = ap.parse_args()

    import torch
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_mod
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch

    from mtla.config import load_config
    from mtla.registry import resolve
    from mtla.utils import heatmap

    targets = parse_targets(args.targets) if args.targets else DEFAULT_TARGETS
    cfg = load_config(args.config)
    model, dataset = resolve(cfg.model, cfg.dataset)
    # This figure is Qwen3-VL-specific: it loads the model with the Qwen3-VL classes and patches the
    # Qwen3-VL attention module. Other model configs would load the wrong weights into that class.
    if cfg.model != "qwen3vl_image" or cfg.dataset != "coco":
        raise SystemExit(
            f"figure_pertoken only supports the Qwen3-VL COCO config, got model={cfg.model} "
            f"dataset={cfg.dataset}. Run: python -m scripts.figure_pertoken --config configs/coco_qwen3vl.yaml"
        )

    preds_path = os.path.join(cfg.pred_dir(0), "predictions.json")
    if not os.path.exists(preds_path):
        raise SystemExit(
            f"no predictions at {preds_path}. Generate them first:\n"
            f"    python -m generate --config {args.config}"
        )

    ds_by_id = {d["id"]: d for d in dataset.load_items(cfg)}
    preds_by_id = {p["id"]: p for p in json.load(open(preds_path))}

    # load model + install the capture hook
    dev = f"cuda:{args.gpu}"
    state = {
        "active": False,
        "ids": set(),
        "order": [],
        "qpos": None,
        "imgidx": None,
        "buf": None,
    }
    qwen3_mod.eager_attention_forward = build_hook(state)
    proc = AutoProcessor.from_pretrained(model.model_id)
    net = Qwen3VLForConditionalGeneration.from_pretrained(
        model.model_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=dev,
    ).eval()
    layers = net.model.language_model.layers
    nL = len(layers)
    nH = net.config.text_config.num_attention_heads
    state["ids"] = {id(L.self_attn) for L in layers}
    state["order"] = [id(L.self_attn) for L in layers]
    img_pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    rows = []
    for iid, pi, grounded in targets:
        if iid not in ds_by_id or iid not in preds_by_id:
            print(f"[skip] {iid}: not in dataset/predictions")
            continue
        di, pr = ds_by_id[iid], preds_by_id[iid]
        # Parse the boxes from the stored response with the model's own parser (the record holds the
        # raw response, not pre-parsed boxes), and read the prompt AS STORED (do not regenerate it —
        # it must match what generation sent, exactly as the extract stage teacher-forces it).
        pbs = [
            {"label": p.label, "box": p.region}
            for p in model.parse_response(pr["response"])
        ]
        if pi >= len(pbs):
            print(f"[skip] {iid} pi={pi}: only {len(pbs)} predictions")
            continue
        prompt = pr["prompt"]
        img = Image.open(di["image"]).convert("RGB")
        text = proc.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inp = proc(text=[text], images=[img], return_tensors="pt").to(dev)
        pid = inp["input_ids"][0]
        plen = pid.shape[0]
        h_, w_ = inp["image_grid_thw"][0].tolist()[1:]
        gh, gw = h_ // 2, w_ // 2
        nimg = gh * gw
        resp = pr["response"]
        rids = torch.tensor(
            proc.tokenizer(resp, add_special_tokens=False)["input_ids"],
            dtype=pid.dtype,
            device=dev,
        )
        full = torch.cat([pid, rids]).unsqueeze(0)
        total = full.shape[1]
        imgidx = [i for i, t in enumerate(pid.cpu().tolist()) if t == img_pad]
        if len(imgidx) != nimg:
            print(f"[skip] {iid}: image-token count {len(imgidx)} != {nimg}")
            continue

        ranges = find_coord_groups(resp, pbs, proc.tokenizer)
        r = ranges[pi]
        if r is None:
            print(f"[skip] {iid} pi={pi}: could not locate the prediction's tokens")
            continue
        # read attention at each token's PREDICTING position (t-1), clamped into range
        cg = [
            [max(0, plen + t - 1) for t in g if plen + t < total]
            for g in r["coord_groups"]
        ]
        lp = [max(0, plen + t - 1) for t in r["label_toks"] if plen + t < total]
        if any(len(g) == 0 for g in cg) or not lp:
            print(f"[skip] {iid} pi={pi}: empty coord/label token group")
            continue

        allpos = sorted({x for g in cg for x in g} | set(lp))
        p2r = {x: i for i, x in enumerate(allpos)}
        state["qpos"] = torch.tensor(allpos, dtype=torch.long, device=dev)
        state["imgidx"] = torch.tensor(imgidx, dtype=torch.long, device=dev)
        state["buf"] = torch.zeros(len(allpos), nL, nH, nimg, dtype=torch.float32)
        state["active"] = True
        with torch.no_grad():
            net(
                input_ids=full,
                pixel_values=inp["pixel_values"],
                image_grid_thw=inp["image_grid_thw"],
                attention_mask=torch.ones(1, total, device=dev, dtype=torch.long),
            )
        state["active"] = False
        buf = state["buf"].numpy()

        def red(
            pos,
        ):  # [n_tok, L, H, nimg] -> [gh, gw]: mean sub-tokens, band, mean heads, sum layers
            mats = np.stack([buf[p2r[x]] for x in pos])
            return mats.mean(0)[BAND].mean(1).sum(0).reshape(gh, gw)

        variants = [red(cg[0]), red(cg[1]), red(cg[2]), red(cg[3]), red(lp)]
        variants.append(np.mean(variants, axis=0))  # the per-token mean column
        rows.append(
            {
                "iid": iid,
                "grounded": grounded,
                "box": pbs[pi]["box"],
                "label": pbs[pi]["label"],
                "variants": variants,
                "image": np.asarray(img),
            }
        )
        torch.cuda.empty_cache()

    if not rows:
        print("no rows rendered; check --targets / that the COCO predictions exist")
        sys.exit(1)

    # per-column normalization divisor (shared scale per token across rows), like the paper's colref
    ncol = len(rows[0]["variants"])
    if args.norm == "per-column":
        colref = [
            float(
                np.percentile(
                    np.concatenate([r["variants"][c].ravel() for r in rows]), 99
                )
            )
            + 1e-9
            for c in range(ncol)
        ]
    else:
        colref = None

    nr = len(rows)
    fig, axes = plt.subplots(
        nr, 7, figsize=(7 * 1.6, nr * 1.6 * 0.85 + 0.4), squeeze=False
    )
    cmap = plt.get_cmap("turbo")
    for ri, r in enumerate(rows):
        col = GROUNDED_C if r["grounded"] else HALLU_C
        img = r["image"]
        H, W = img.shape[:2]
        b = r["box"]
        x1, y1 = b[0] * W / 1000, b[1] * H / 1000
        x2, y2 = b[2] * W / 1000, b[3] * H / 1000
        ax0 = axes[ri, 0]
        ax0.imshow(img, interpolation="none")
        ax0.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, ec=col, fc="none", lw=2.2))
        yt, va = (y1 - 2, "bottom") if y1 > 14 else (y1 + 2, "top")
        ax0.text(
            x1,
            yt,
            f" {r['label']} ",
            color="white",
            fontsize=7,
            fontweight="bold",
            va=va,
            ha="left",
            bbox=dict(facecolor=col, edgecolor="none", boxstyle="round,pad=0.12"),
            clip_on=False,
        )
        ax0.set_xticks([])
        ax0.set_yticks([])
        if ri == 0:
            ax0.set_title(COLS[0], fontsize=13)
        for ci, gmap in enumerate(r["variants"]):
            ax = axes[ri, ci + 1]
            g = gmap / colref[ci] if colref is not None else gmap
            hm = heatmap(g, H, W)
            # opacity ramp: transparent below the CLIP percentile, ramping to AMAX
            lo = np.percentile(hm, CLIP_PCT)
            tt = np.clip((hm - lo) / (hm.max() - lo + 1e-9), 0, 1)
            al = np.where(hm <= lo, 0.0, FLOOR + (AMAX - FLOOR) * tt)[..., None]
            comp = (
                (
                    (1 - al) * img.astype(np.float32)
                    + al * (cmap(np.clip(hm, 0, 1))[..., :3] * 255)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            ax.imshow(comp, interpolation="none")
            ax.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, ec="white", fc="none", lw=1.2)
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(COLS[ci + 1], fontsize=13)

    fig.subplots_adjust(
        left=0.055, right=0.997, top=0.94, bottom=0.01, wspace=0.03, hspace=0.06
    )
    # open a gap before the aggregate "mean" column
    for ri in range(nr):
        bb = axes[ri, 6].get_position()
        axes[ri, 6].set_position([bb.x0 + 0.022, bb.y0, bb.width, bb.height])

    # left side-bands grouping grounded vs hallucinated rows
    def band(idx, color, text):
        if not idx:
            return
        tops = [axes[i, 0].get_position().y1 for i in idx]
        bots = [axes[i, 0].get_position().y0 for i in idx]
        fig.add_artist(
            FancyBboxPatch(
                (0.012, min(bots)),
                0.02,
                max(tops) - min(bots),
                boxstyle="round,pad=0.002",
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
                zorder=5,
            )
        )
        fig.text(
            0.022,
            (min(bots) + max(tops)) / 2,
            text,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    gi = [i for i, r in enumerate(rows) if r["grounded"]]
    hi = [i for i, r in enumerate(rows) if not r["grounded"]]
    band(gi, GROUNDED_C, "Grounded")
    band(hi, HALLU_C, "Hallu.")
    # dashed separator before the mean column
    xsep = (axes[0, 5].get_position().x1 + axes[0, 6].get_position().x0) / 2
    fig.add_artist(
        plt.Line2D(
            [xsep, xsep],
            [axes[nr - 1, 6].get_position().y0, axes[0, 6].get_position().y1],
            transform=fig.transFigure,
            color="#bbb",
            lw=0.8,
            zorder=7,
        )
    )

    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"saved {args.out} (+ .png)  rows={nr}")


if __name__ == "__main__":
    main()
