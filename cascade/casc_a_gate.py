"""Cascade Stage A: AF3 multi-run gate (vLLM). Per clip, run the open
'what do you hear' prompt N times (temp-sampled) and store the union of raw
free-text labels. Sharded by clip across GPUs. torch.compile on (no enforce_eager).

Out: <out>/gate_shard{ID}.jsonl  {video_id, gt_events, runs:[raw,...]}
Env: audio_grounding (vllm 0.19 + transformers-git AF3). One proc/GPU.
"""
import argparse, glob, io, json, time
import soundfile as sf, librosa
import pyarrow.parquet as pqm

LOCAL_PARQUET = os.environ.get("AUDIOSET_PARQUET", "data/audioset_strong_parquet/data")
PROMPT = ("List all the distinct sound events you can hear in this audio clip. "
          "Give only short sound-event labels, comma separated.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--total_clips", type=int, default=30)
    ap.add_argument("--n_runs", type=int, default=6)
    ap.add_argument("--out", default="/tmp/af3_cascade")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, gpu_memory_utilization=0.90, max_model_len=4096,
              limit_mm_per_prompt={"audio": 1}, trust_remote_code=True, dtype="bfloat16")
    print(f"[t={time.time()-t0:.0f}s] gate vLLM ready", flush=True)

    rows, gi = [], 0
    for pf in sorted(glob.glob(f"{LOCAL_PARQUET}/test-*.parquet")):
        for rec in pqm.read_table(pf).to_pylist():
            if gi >= args.total_clips:
                break
            if gi % args.num_shards == args.shard_id:
                rows.append(rec)
            gi += 1
        if gi >= args.total_clips:
            break

    import os
    os.makedirs(args.out, exist_ok=True)
    fout = open(f"{args.out}/gate_shard{args.shard_id:02d}.jsonl", "w")
    for ci, rec in enumerate(rows):
        arr, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(1)
        if sr != 16000:
            arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=16000)
        conv = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT}, {"type": "audio", "audio": arr}]}]
        p = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        req = {"prompt": p, "multi_modal_data": {"audio": [(arr, 16000)]}}
        runs = []
        for k in range(args.n_runs):
            sp = SamplingParams(temperature=0.0 if k == 0 else 0.9, top_p=0.95,
                                max_tokens=200, seed=k)
            runs.append(llm.generate([req], sp)[0].outputs[0].text.strip())
        fout.write(json.dumps({"video_id": rec["video_id"],
                               "gt_events": rec.get("events") or [], "runs": runs},
                              ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[t={time.time()-t0:.0f}s] shard{args.shard_id} {ci+1}/{len(rows)}", flush=True)
    fout.close()
    print(f"shard {args.shard_id}: gated {len(rows)} clips", flush=True)


if __name__ == "__main__":
    main()
