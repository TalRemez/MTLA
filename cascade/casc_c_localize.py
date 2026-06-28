"""Cascade Stage C: single-class localization (vLLM) of each clip's candidate
classes. Reads candidates.jsonl, queries AF3 single-class per candidate, parses
[start,end] windows. Sharded by clip.

Out: <out>/loc_shard{ID}.jsonl  {video_id, gt_events, query_results:[{label,windows,raw}]}
Env: audio_grounding. torch.compile on.
"""
import argparse, glob, io, json, re, time
import soundfile as sf, librosa
import pyarrow.parquet as pqm

LOCAL_PARQUET = os.environ.get("AUDIOSET_PARQUET", "data/audioset_strong_parquet/data")
PROMPT = ('This is a sequence of audio stream. Your task is to identify the temporal window '
          '(start and end timestamps) when the given query appears. The query is: ')
SUFFIX = " Answer: "
WIN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s?\s*(?:to|[-–])\s*([0-9]+(?:\.[0-9]+)?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cand_file", default="/tmp/af3_cascade/candidates.jsonl")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--out", default="/tmp/af3_cascade")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--n_rollouts", type=int, default=1,
                    help="N stochastic localization rollouts per (clip,class). "
                         "1 = greedy; >1 = temp-sampled with distinct seeds.")
    args = ap.parse_args()

    cand_by = {}
    for i, line in enumerate(open(args.cand_file)):
        if i % args.num_shards == args.shard_id:
            r = json.loads(line)
            cand_by[r["video_id"]] = r
    # audio for this shard's clips
    audio_by = {}
    for pf in sorted(glob.glob(f"{LOCAL_PARQUET}/test-*.parquet")):
        for rec in pqm.read_table(pf).to_pylist():
            if rec["video_id"] in cand_by and rec["video_id"] not in audio_by:
                arr, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(1)
                if sr != 16000:
                    arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=16000)
                audio_by[rec["video_id"]] = arr
        if len(audio_by) >= len(cand_by):
            break

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, gpu_memory_utilization=0.90, max_model_len=4096,
              limit_mm_per_prompt={"audio": 1}, trust_remote_code=True, dtype="bfloat16")
    N = args.n_rollouts
    print(f"[t={time.time()-t0:.0f}s] localize vLLM ready, {len(cand_by)} clips, N={N}", flush=True)

    import os
    os.makedirs(args.out, exist_ok=True)
    fout = open(f"{args.out}/loc_shard{args.shard_id:02d}.jsonl", "w")
    for ci, (vid, rec) in enumerate(cand_by.items()):
        arr = audio_by.get(vid)
        if arr is None:
            continue
        prompts = []
        for c in rec["candidates"]:
            conv = [{"role": "user", "content": [
                {"type": "text", "text": PROMPT + c + SUFFIX}, {"type": "audio", "audio": arr}]}]
            prompts.append((c, proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)))
        # per-seed batched generation; results carry the seed so the extractor
        # and NMS voting can count distinct-seed support.
        results = []  # {label, windows, raw, seed}
        for seed in range(N):
            sp = (SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens) if N == 1
                  else SamplingParams(temperature=0.7, top_p=0.95, seed=seed,
                                      max_tokens=args.max_new_tokens))
            reqs = [{"prompt": p, "multi_modal_data": {"audio": [(arr, 16000)]}} for _, p in prompts]
            outs = llm.generate(reqs, sp)
            for (c, _), o in zip(prompts, outs):
                t = o.outputs[0].text
                wins = [[float(m.group(1)), float(m.group(2))] for m in WIN.finditer(t)
                        if float(m.group(2)) > float(m.group(1)) and float(m.group(2)) <= 10.5]
                if wins:
                    results.append({"label": c, "windows": wins, "raw": t, "seed": seed})
        fout.write(json.dumps({"video_id": vid, "gt_events": rec["gt_events"],
                               "query_results": results}, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[t={time.time()-t0:.0f}s] shard{args.shard_id} {ci+1}/{len(cand_by)} {vid}: "
              f"{len(results)} (class,seed) results", flush=True)
    fout.close()
    print(f"shard {args.shard_id}: done", flush=True)


if __name__ == "__main__":
    main()
