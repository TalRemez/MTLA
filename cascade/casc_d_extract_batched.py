"""Cascade Stage D (BATCHED): MTLA/SVAR extraction with RIGHT-padded response
batching. All responses of one (clip,label) share an identical [prompt+audio]
prefix; we right-pad their response suffixes and run up to --batch at once in a
single forward. Because attention is causal and padding is on the RIGHT, the
real response query-rows never attend to pad keys -> no contamination, no custom
position_ids, and the prefix/audio/response token positions are uniform across
the batch.

Hook captures [B, n_resp, H, n_audio] per layer (audio columns only).
Out: <out>/attn_shard{ID}.pt  same schema (objects carry seed).
Env: spotsound (transformers-git AF3, eager).
"""
import argparse, glob, io, json, re, time
from collections import defaultdict
from pathlib import Path
import soundfile as sf, torch
import torch.nn as nn
import pyarrow.parquet as pqm

# Cap CPU threads: 8 concurrent procs/machine on 192 cores oversubscribe badly
# (the c2t loop + .float() reductions are CPU-bound). Without this, loadavg hit
# ~550 and GPUs sat at 0%. Env vars may be set by the orchestrator too.
import os as _os
torch.set_num_threads(int(_os.environ.get("OMP_NUM_THREADS", "6")))

LOCAL_PARQUET = os.environ.get("AUDIOSET_PARQUET", "data/audioset_strong_parquet/data")
AUDIO_TOKEN_ID = 151669
CLIP = 10.0
AUDIO_TOKEN_HZ = 25.0  # AF3 emits audio tokens at a fixed 25 Hz (content-proportional,
                       # verified by probe_af3_audio_rate.py). token i -> i/HZ seconds, so
                       # short (<10s) clips map correctly instead of being stretched to 10s.
PROMPT = ('This is a sequence of audio stream. Your task is to identify the temporal window '
          '(start and end timestamps) when the given query appears. The query is: ')
SUFFIX = " Answer: "
WIN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s?\s*(?:to|[-–])\s*([0-9]+(?:\.[0-9]+)?)")

_EXTRACT = {"active": False, "layer_ids": set(), "layer_order": [],
            "resp_lo": 0, "resp_hi": 0, "audio_idx": None, "collected": None}


def hook_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    s = _EXTRACT
    if s["active"] and id(module) in s["layer_ids"]:
        li = s["layer_order"].index(id(module))
        # attn_weights: [B, H, seq, seq] -> keep response rows + audio cols
        rows = attn_weights[:, :, s["resp_lo"]:s["resp_hi"], :].index_select(3, s["audio_idx"])
        # [B, H, n_resp, n_audio] -> [B, n_resp, H, n_audio]
        s["collected"][li] = rows.permute(0, 2, 1, 3).contiguous().to(torch.float16).cpu()
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    out = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return out, attn_weights


DILATE_K = 0   # set from --dilate; widen the inside-window token mask by +-K tokens
               # (40 ms/token at AF3 25 Hz). d2 (+-80 ms) is the chosen headline mask.


def _dilate(mask, k):
    if k <= 0:
        return mask
    m = mask.clone()
    for s in range(1, k + 1):
        m[:-s] |= mask[s:]
        m[s:] |= mask[:-s]
    return m


def reduce_obj(coll_row, secs, lo, hi, tokrange):
    """coll_row: [L, n_resp, H, n_audio] for ONE batch row."""
    t_lo, t_hi = tokrange
    inside = (secs >= lo) & (secs <= hi)
    if not inside.any():
        inside[(secs - 0.5 * (lo + hi)).abs().argmin()] = True
    inside = _dilate(inside, DILATE_K)
    sub = coll_row[:, t_lo:t_hi + 1, :, :].float()
    mt = sub.mean(1); ft = sub[:, 0, :, :]
    first = {"image_global_sum": ft.sum(-1), "image_inside_sum": ft[..., inside].sum(-1)}
    allm = {"image_global_sum": mt.sum(-1), "image_inside_sum": mt[..., inside].sum(-1),
            "image_outside_sum": mt[..., ~inside].sum(-1), "image_max": mt.max(-1).values}
    return first, allm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--loc_file_glob", default="/tmp/af3_cascade/loc_shard*.jsonl")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--out", default="/tmp/af3_cascade")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--clip_start", type=int, default=0, help="slice this shard's clips [start:end)")
    ap.add_argument("--clip_end", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for output .pt name (e.g. batch range)")
    ap.add_argument("--dilate", type=int, default=0, help="widen inside-window mask by +-K tokens (d2 = headline)")
    args = ap.parse_args()
    global DILATE_K; DILATE_K = args.dilate

    # Read ONLY this shard's own localization file (one line per clip that
    # Stage C assigned to this shard). Reading all files + modulo would break
    # when shards finish at different times (per-shard orchestration), so we
    # key off the shard-specific file directly.
    recs = []
    # own-file = the glob with the trailing "*" replaced by this shard's 2-digit id
    # (works for loc_shard*.jsonl AND loc_fix_shard*.jsonl etc.)
    own = args.loc_file_glob.replace("*.jsonl", f"{args.shard_id:02d}.jsonl")
    import os as _os
    if _os.path.exists(own):
        for line in open(own):
            recs.append(json.loads(line))
    else:  # fallback: glob + modulo (back-compat for all-at-once runs)
        all_recs = []
        for f in sorted(glob.glob(args.loc_file_glob)):
            for line in open(f):
                all_recs.append(json.loads(line))
        recs = [r for i, r in enumerate(all_recs) if i % args.num_shards == args.shard_id]
    recs = recs[args.clip_start:args.clip_end]   # batch slice
    want = {r["video_id"] for r in recs}

    audio_by = {}
    for pf in sorted(glob.glob(f"{LOCAL_PARQUET}/test-*.parquet")):
        for rec in pqm.read_table(pf).to_pylist():
            if rec["video_id"] in want and rec["video_id"] not in audio_by:
                arr, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(1)
                audio_by[rec["video_id"]] = (arr, sr)
        if len(audio_by) >= len(want):
            break

    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    import transformers.models.qwen2.modeling_qwen2 as q2
    import tempfile, os
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model_path)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", attn_implementation="eager")
    model.eval()
    tok = proc.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    q2.eager_attention_forward = hook_eager
    llm = model.model.language_model
    attn_mods = [layer.self_attn for layer in llm.layers]
    _EXTRACT["layer_order"] = [id(m) for m in attn_mods]
    _EXTRACT["layer_ids"] = set(_EXTRACT["layer_order"])
    L = len(attn_mods)
    print(f"[t={time.time()-t0:.0f}s] AF3 eager+hook ready (batched), {len(recs)} clips, L={L}", flush=True)

    out_records = []
    for ci, rec in enumerate(recs):
        vid = rec["video_id"]; au = audio_by.get(vid)
        if au is None:
            out_records.append({"video_id": vid, "gt_events": rec["gt_events"], "objects": []})
            continue
        arr, sr = au
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); sf.write(tf.name, arr, sr)
        by_label = defaultdict(list)
        for qr in rec["query_results"]:
            by_label[qr["label"]].append(qr)
        objects = []
        for label, qrs in by_label.items():
            conv = [{"role": "user", "content": [
                {"type": "text", "text": PROMPT + label + SUFFIX}, {"type": "audio", "path": tf.name}]}]
            enc = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
            prompt_ids = enc["input_ids"]               # [1, P]
            P = prompt_ids.shape[1]
            feats = {k: v for k, v in enc.items() if k in ("input_features", "input_features_mask")}
            audio_pos = (prompt_ids[0] == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
            if audio_pos.numel() == 0:
                continue
            n_audio = audio_pos.numel(); secs = torch.arange(n_audio, dtype=torch.float32) / AUDIO_TOKEN_HZ

            # tokenize each response; keep (qr, resp_ids, pieces, coff)
            items = []
            for qr in qrs:
                rid = tok(qr["raw"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                pieces = [tok.decode([int(t)]) for t in rid.tolist()]
                coff = [0]
                for p in pieces:
                    coff.append(coff[-1] + len(p))
                items.append((qr, rid, pieces, coff))

            # batch in chunks of args.batch
            for b0 in range(0, len(items), args.batch):
                chunk = items[b0:b0 + args.batch]
                max_r = max(it[1].shape[0] for it in chunk)
                B = len(chunk)
                # build [B, P+max_r] right-padded; audio feats repeated per row
                input_ids = torch.full((B, P + max_r), pad_id, dtype=prompt_ids.dtype)
                attn_mask = torch.zeros((B, P + max_r), dtype=torch.long)
                for bi, (qr, rid, _, _) in enumerate(chunk):
                    rlen = rid.shape[0]
                    input_ids[bi, :P] = prompt_ids[0].cpu()
                    input_ids[bi, P:P + rlen] = rid
                    attn_mask[bi, :P + rlen] = 1
                input_ids = input_ids.to(model.device); attn_mask = attn_mask.to(model.device)
                bfeats = {k: v.repeat(B, *([1] * (v.dim() - 1))) for k, v in feats.items()}
                _EXTRACT.update(active=True, resp_lo=P, resp_hi=P + max_r,
                                audio_idx=audio_pos.to(model.device), collected=[None] * L)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask, **bfeats, use_cache=False)
                _EXTRACT["active"] = False
                coll = torch.stack(_EXTRACT["collected"], 0)  # [L, B, max_r, H, n_audio]
                for bi, (qr, rid, pieces, coff) in enumerate(chunk):
                    row = coll[:, bi]  # [L, max_r, H, n_audio]

                    def c2t(c_lo, c_hi, pieces=pieces, coff=coff):
                        t_lo = next((i for i in range(len(pieces)) if coff[i + 1] > c_lo), None)
                        t_hi = next((i for i in range(len(pieces)) if coff[i + 1] >= c_hi), None)
                        return (t_lo, t_hi) if (t_lo is not None and t_hi is not None) else None

                    for m in WIN.finditer(qr["raw"]):
                        lo, hi = float(m.group(1)), float(m.group(2))
                        if not (hi > lo and hi <= 10.5):
                            continue
                        tr = c2t(m.start(), m.end())
                        if tr is None:
                            continue
                        first, allm = reduce_obj(row, secs, lo, hi, tr)
                        objects.append({"label": label, "bbox_2d": [lo, hi], "seed": qr.get("seed", 0),
                                        "attn_first": first, "attn_all": allm})
        os.unlink(tf.name)
        out_records.append({"video_id": vid, "gt_events": rec["gt_events"], "objects": objects})
        print(f"[t={time.time()-t0:.0f}s] shard{args.shard_id} {ci+1}/{len(recs)} {vid}: {len(objects)} dets", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save(out_records, f"{args.out}/attn_shard{args.shard_id:02d}{args.tag}.pt")
    print(f"shard {args.shard_id}: saved {len(out_records)} clips", flush=True)


if __name__ == "__main__":
    main()
