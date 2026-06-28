import os
"""AudioSet-Strong open-vocab SED generation with base Audio Flamingo 3.

AF3 cannot follow the rigid JSON SED schema (it loops), but with a light
"native" prompt it emits parseable, AudioSet-named, timestamped events:
    Sound of <label>: [start-end, start-end, ...]; Sound of <label>: [...]
We parse that, fuzzy-normalize each label to the 456 canonical AudioSet-Strong
classes, and write the SAME JSONL record schema as the Qwen3-Omni pipeline so
the existing attention extractor + PSDS1 scorer join by (video_id, pred-index):
    {video_id, predictions:[{label, bbox_2d:[start,end]}], raw_output,
     gt_events, n_gen_tokens}

One JSONL per seed: <out_root>/seed{S}/predictions_shard{ID}.jsonl
Run one process per GPU (CUDA_VISIBLE_DEVICES=S), env spotsound (HF eager,
transformers 5.x with AudioFlamingo3).
"""
import argparse
import glob
import io
import json
import re
import sys
import time
import difflib
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

ROOT = Path(os.environ.get("CASCADE_ROOT", "."))
LOCAL_PARQUET = os.environ.get("AUDIOSET_PARQUET", "data/audioset_strong_parquet/data")
CLASSES_PATH = ROOT / "audioset_strong_456_classes.json"
DEFAULT_OUT = ROOT / "artifacts/predictions/audioset_strong_af3"
TARGET_SR = 16000

PROMPT = ("Detect all the sound events in this audio and list each one with the times it "
          "occurs, using the format: Sound of <event>: [start-end, start-end, ...].")

# AF3 emits "Sound of <label>: [a-b, c-d, ...]" segments separated by ';'.
SEG_RE = re.compile(r"Sound of\s*(.+?):\s*\[([^\]]*)\]", re.IGNORECASE)
WIN_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s?\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)")


def build_normalizer(classes):
    lc = {c.lower(): c for c in classes}

    keys = list(lc.keys())

    def norm(raw):
        r = raw.strip().strip(".").lower()
        if not r:
            return None
        if r in lc:
            return lc[r]
        # AF3 frequently TRUNCATES the canonical AudioSet name, dropping the
        # comma-suffix: "Heart sounds" for "Heart sounds, heartbeat", "Female
        # speech" for "Female speech, woman speaking". Prefer a canonical class
        # whose comma-delimited head equals the raw label, or that starts with
        # it -- this is far safer than difflib (which mapped "Heart sounds" to
        # "Human sounds").
        for cl_low, cl in lc.items():
            if cl_low.split(",")[0].strip() == r:
                return cl
        for cl_low, cl in lc.items():
            if cl_low.startswith(r) or r.startswith(cl_low):
                return cl
        # AF3 sometimes adds descriptors ("Rock music with a guitar and a
        # female singer"); fall back to substring containment of a canonical
        # name, longest first so "Male singing" beats "Singing".
        for cl_low in sorted(keys, key=len, reverse=True):
            if cl_low in r and len(cl_low) >= 4:
                return lc[cl_low]
        # last resort: fuzzy
        m = difflib.get_close_matches(r, keys, n=1, cutoff=0.82)
        return lc[m[0]] if m else None

    return norm


def parse_response(text, norm):
    """text -> list of {label, bbox_2d:[start,end]} (one per window), normalized."""
    preds = []
    for m in SEG_RE.finditer(text):
        raw_label, windows = m.group(1), m.group(2)
        canon = norm(raw_label)
        if canon is None:
            continue
        for wm in WIN_RE.finditer(windows):
            lo, hi = float(wm.group(1)), float(wm.group(2))
            if hi > lo and hi <= 10.5:
                preds.append({"label": canon, "bbox_2d": [lo, hi], "raw_label": raw_label})
    return preds


def decode_audio(row):
    au = row["audio"]
    arr, sr = (sf.read(io.BytesIO(au["bytes"]), dtype="float32")
               if au.get("bytes") is not None else sf.read(au["path"], dtype="float32"))
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="AF3 base snapshot dir")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--max_clips", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    classes = json.loads(CLASSES_PATH.read_text())
    norm = build_normalizer(classes)

    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model_path)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto")
    model.eval()
    print(f"[t={time.time()-t0:.0f}s] AF3 loaded", flush=True)

    # Read parquet directly via pyarrow (immune to datasets/feature-type skew;
    # the parquet was written with a newer datasets than this env has).
    import pyarrow.parquet as pq_mod

    def iter_rows():
        for pf in sorted(glob.glob(f"{LOCAL_PARQUET}/test-*.parquet")):
            tbl = pq_mod.read_table(pf)
            for rec in tbl.to_pylist():
                yield rec

    ds = iter_rows()

    out_paths = {}
    for s in seeds:
        d = Path(args.out_root) / f"seed{s}"
        d.mkdir(parents=True, exist_ok=True)
        out_paths[s] = open(d / f"predictions_shard{args.shard_id:02d}.jsonl", "w")

    n_done = 0
    for gi, row in enumerate(ds):
        if gi % args.num_shards != args.shard_id:
            continue
        if n_done >= args.max_clips:
            break
        vid = row["video_id"]
        arr = decode_audio(row)
        gt = row.get("events") or []
        conv = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "audio", "audio": arr}]}]
        inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        for s in seeds:
            torch.manual_seed(s)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                      do_sample=(len(seeds) > 1), temperature=0.7, top_p=0.95)
            resp = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
            preds = parse_response(resp, norm)
            rec = {"video_id": vid, "seed": s, "raw_output": resp,
                   "predictions": preds, "predictions_normalized": preds,
                   "gt_events": gt, "n_gen_tokens": int(out.shape[1] - inputs["input_ids"].shape[1])}
            out_paths[s].write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_paths[s].flush()
        n_done += 1
        if n_done % 5 == 0:
            print(f"[t={time.time()-t0:.0f}s] {n_done} clips", flush=True)
    for f in out_paths.values():
        f.close()
    print(f"shard {args.shard_id}: {n_done} clips x {len(seeds)} seeds in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
