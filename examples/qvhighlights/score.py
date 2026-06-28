"""QVHighlights self-consistency voting with MTLA, scored by the OFFICIAL
Moment-DETR standalone_eval (no hand-rolled mAP).

QVH is MULTI-segment, so this mirrors the COCO voting recipe:
  1. Pool every predicted window across N stochastic rollouts for a query.
  2. Cluster by temporal IoU >= 0.5 (greedy union-find).
  3. Each cluster -> mean window, vote = #distinct seeds, mean per-seg MTLA / SVAR.
  4. Score each cluster and RANK -> the ranked window list (with scores) is the
     fused multi-segment prediction.

PER-SEGMENT MTLA = inside-attention over a SINGLE window's frames (mask that window
only), L8-21 mean-H sum-L -- same band/reduction as analyze_qvhighlights_v3.py.
SVAR/GA = global attention over ALL video frames (video_sum; per-rollout scalar,
constant across a rollout's windows -> single-rollout SVAR ranking == emission order).

Scoring: builds official submission dicts
  {"qid", "pred_relevant_windows": [[s, e, score], ...]}  (ranked, score desc)
and calls compute_mr_ap / compute_mr_r1 from
third_party/moment_detr_eval/eval.py (vendored from jayleicn/moment_detr, MIT).
mAP = avg over IoU 0.5:0.05:0.95; R1@{0.5,0.7} on the top-ranked window.
"""
import argparse, glob, sys, os
from collections import defaultdict
import numpy as np
import torch

FEAT_ROOT = "features"                     # --feat_root : attention shards from generate_and_extract.py
ANN_PATH = "highlight_val_release.jsonl"   # --ann       : QVHighlights val annotations (jsonl)
SEEDS = list(range(16))
LF_BAND = list(range(8, 22))
SLOT = {"first_digit": 0, "first2_mean": 1, "last_digit": 2, "all_mean": 3}
CLUSTER_IOU = 0.5
MAX_PRED_WINDOWS = 10

# official Moment-DETR eval
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "third_party"))
from moment_detr_eval.eval import compute_mr_ap, compute_mr_r1

ap = argparse.ArgumentParser()
ap.add_argument("--slot", default="first_digit", choices=list(SLOT))
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--perwin", action="store_true",
                help="Use per-window coord-token attention (frame_sum_perwin) instead "
                     "of the shared response-level slot. Requires _perwin shards.")
ap.add_argument("--feat_root", default=FEAT_ROOT,
                help="attention-shards dir from generate_and_extract.py")
ap.add_argument("--ann", default=ANN_PATH, help="QVHighlights val annotations jsonl")
args = ap.parse_args()
SLOT_IDX = SLOT[args.slot]
USE_SEEDS = SEEDS[: args.n]
PERWIN = args.perwin
FEAT_ROOT = args.feat_root
ANN_PATH = args.ann


def reduce_band(arr_l_h):
    return float(arr_l_h[LF_BAND, :].astype(np.float32).mean(axis=1).sum())


def seg_mtla(frame_sum_LHT, window, duration, T_tokens):
    if window is None or duration <= 0 or T_tokens <= 0:
        return 0.0
    s, e = window
    fs = max(0, int(np.floor(s * T_tokens / duration)))
    fe = min(T_tokens, int(np.ceil(e * T_tokens / duration)))
    if fe <= fs:
        return 0.0
    return reduce_band(frame_sum_LHT[:, :, fs:fe].sum(axis=2))


def svar_global(video_sum_LH):
    return reduce_band(video_sum_LH)


def tiou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 1e-9 else 0.0


def cluster(idx_windows, iou_th=CLUSTER_IOU):
    n = len(idx_windows)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if tiou(idx_windows[i], idx_windows[j]) >= iou_th:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


# ---------- load ----------
print(f"loading {len(USE_SEEDS)} seeds, slot={args.slot} (idx {SLOT_IDX}) ...")
by_query = defaultdict(list)
gt_of = {}
for sd in USE_SEEDS:
    recs = []
    for sp in sorted(glob.glob(f"{FEAT_ROOT}/seed{sd}/shard*.pt")):
        recs.extend(torch.load(sp, weights_only=False, map_location="cpu"))
    for r in recs:
        qid = r["qid"]
        fs = r["attn"]["frame_sum"][SLOT_IDX].astype(np.float32)        # shared 4-slot map
        gsvar = svar_global(r["attn"]["video_sum"][SLOT_IDX].astype(np.float32))
        # per-window maps [n_win, L, H, T] if present (image-case parity); else None.
        # first_digit slot -> window first coord token; else -> mean over window coords.
        _pw_key = "frame_sum_perwin_first" if args.slot == "first_digit" else "frame_sum_perwin"
        fpw = r["attn"].get(_pw_key)
        if fpw is not None:
            fpw = np.asarray(fpw).astype(np.float32)
        rollout = []
        for ei, w in enumerate(r["pred_windows"] or []):
            w = list(w)
            if PERWIN and fpw is not None and ei < fpw.shape[0]:
                fs_w = fpw[ei]                     # this window's own coord-token attention
            else:
                fs_w = fs                          # fallback: shared response-level map
            rollout.append((w, seg_mtla(fs_w, w, r["duration_s"], r["T_tokens"]), gsvar, ei))
        by_query[qid].append({"seed": sd, "segs": rollout})
        gt_of[qid] = [list(w) for w in r["gt_windows"]]

# official GT from the annotation file (authoritative relevant_windows)
ground_truth = []
import json
with open(ANN_PATH) as f:
    for ln in f:
        d = json.loads(ln)
        if d["qid"] in by_query:
            ground_truth.append({"qid": d["qid"], "relevant_windows": d["relevant_windows"]})
gt_qids = {d["qid"] for d in ground_truth}

queries = sorted(q for q in by_query if q in gt_qids)
cov = {len(by_query[q]) for q in queries}
print(f"queries={len(queries)}  rollouts-per-query set={sorted(cov)}  (GT matched)")


# ---------- ranking -> official submission ----------
def clusters_for(qid):
    segs = [(s[0], s[1], s[2], s[3], roll["seed"])
            for roll in by_query[qid] for s in roll["segs"]]
    windows = [s[0] for s in segs]
    out = []
    for members in cluster(windows):
        ms = [segs[i] for i in members]
        out.append({
            "window": np.mean([m[0] for m in ms], axis=0).tolist(),
            "vote": len({m[4] for m in ms}),
            "mean_mtla": float(np.mean([m[1] for m in ms])),
            "mean_svar": float(np.mean([m[2] for m in ms])),
        })
    return out


def sub_single(seed, key):
    """Submission for single-rollout `seed`, windows ranked by key (mtla/svar/emit)."""
    out = []
    for qid in queries:
        roll = next((r for r in by_query[qid] if r["seed"] == seed), by_query[qid][0])
        rows = []
        for (w, m, gv, ei) in roll["segs"]:
            sc = m if key == "mtla" else (gv - ei * 1e-6 if key == "svar" else -ei)
            rows.append([w[0], w[1], float(sc)])
        rows.sort(key=lambda x: -x[2])
        out.append({"qid": qid, "pred_relevant_windows": rows[:MAX_PRED_WINDOWS]
                    if rows else [[0.0, 0.0, 0.0]]})
    return out


def sub_clusters(score_fn):
    out = []
    for qid in queries:
        cl = clusters_for(qid)
        rows = [[c["window"][0], c["window"][1], float(score_fn(c))] for c in cl]
        rows.sort(key=lambda x: -x[2])
        out.append({"qid": qid, "pred_relevant_windows": rows[:MAX_PRED_WINDOWS]
                    if rows else [[0.0, 0.0, 0.0]]})
    return out


def sub_nms(key, iou_th=CLUSTER_IOU, use_support=False):
    """Detection NMS over ALL candidate windows pooled across rollouts.
    key='mtla' -> per-segment MTLA; 'svar' -> per-rollout global attn.
    Rank by score desc, greedily keep top + suppress remaining windows that
    overlap a kept one at IoU >= iou_th. If use_support, the kept window absorbs
    its suppressed overlaps and its final score = (#distinct seeds absorbed) x attn."""
    out = []
    for qid in queries:
        cands = []
        for roll in by_query[qid]:
            for (w, m, gv, ei) in roll["segs"]:
                sc = m if key == "mtla" else gv
                cands.append((w, float(sc), roll["seed"]))
        order = sorted(range(len(cands)), key=lambda i: -cands[i][1])
        taken = [False] * len(cands)
        rows = []
        for i in order:
            if taken[i]:
                continue
            taken[i] = True
            w_i, sc_i, _ = cands[i]
            seeds = {cands[i][2]}
            for j in order:
                if taken[j] or j == i:
                    continue
                if tiou(w_i, cands[j][0]) >= iou_th:
                    taken[j] = True
                    seeds.add(cands[j][2])
            final = (len(seeds) * sc_i) if use_support else sc_i
            rows.append([w_i[0], w_i[1], float(final)])
            if len(rows) >= MAX_PRED_WINDOWS:
                break
        rows = rows or [[0.0, 0.0, 0.0]]
        out.append({"qid": qid, "pred_relevant_windows": rows})
    return out


def evaluate(submission):
    apd = compute_mr_ap(submission, ground_truth, num_workers=8)
    r1 = compute_mr_r1(submission, ground_truth)
    return {
        "mAP": apd["average"], "mAP@0.5": apd["0.5"], "mAP@0.75": apd["0.75"],
        "R1@0.5": r1["0.5"], "R1@0.7": r1["0.7"],
    }


def mean_over_seeds(key):
    rows = [evaluate(sub_single(sd, key)) for sd in USE_SEEDS]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


single_emit = mean_over_seeds("emit")
single_svar = mean_over_seeds("svar")
single_mtla = mean_over_seeds("mtla")

variants = [
    ("SVAR alone (no vote)",  sub_clusters(lambda c: c["mean_svar"])),
    ("MTLA alone (no vote)",  sub_clusters(lambda c: c["mean_mtla"])),
    ("vote alone",            sub_clusters(lambda c: c["vote"])),
    ("vote x mean SVAR",      sub_clusters(lambda c: c["vote"] * c["mean_svar"])),
    ("vote x mean MTLA",      sub_clusters(lambda c: c["vote"] * c["mean_mtla"])),
    ("NMS-SVAR",              sub_nms("svar")),
    ("NMS-MTLA",              sub_nms("mtla")),
    ("NMS-SVAR x support",    sub_nms("svar", use_support=True)),
    ("NMS-MTLA x support",    sub_nms("mtla", use_support=True)),
]

print("\n" + "=" * 78)
print(f"QVHighlights voting  N={args.n} seeds  slot={args.slot}  cluster IoU>={CLUSTER_IOU}")
print(f"OFFICIAL Moment-DETR standalone_eval  (mAP avg over IoU 0.5:0.95)")
print(f"{'variant':<28s} {'mAP':>7s} {'mAP@.5':>7s} {'mAP@.75':>8s} {'R1@.5':>7s} {'R1@.7':>7s}")
print("-" * 78)
def show(name, m):
    print(f"{name:<28s} {m['mAP']:>7.2f} {m['mAP@0.5']:>7.2f} {m['mAP@0.75']:>8.2f} "
          f"{m['R1@0.5']:>7.2f} {m['R1@0.7']:>7.2f}")
print(f"-- single rollout (mean of {len(USE_SEEDS)} seeds) --")
show("  emission order", single_emit)
show("  SVAR (global attn)", single_svar)
show("  MTLA (inside attn)", single_mtla)
print(f"-- N={len(USE_SEEDS)} voting --")
for name, sub in variants:
    show(name, evaluate(sub))
print("=" * 78)
