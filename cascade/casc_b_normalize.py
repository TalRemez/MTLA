import os
"""Cascade Stage B: Claude-normalize gate output -> candidate canonical classes.
Runs on the machine with Bedrock access (this CPU box). Reads all gate shards,
unions each clip's N runs, asks Claude (conservative/recall-oriented) to map the
free-text labels to canonical AudioSet classes.

Out: <out>/candidates.jsonl  {video_id, gt_events, candidates:[canonical_class,...]}
"""
import argparse, glob, json, os, re, time
from concurrent.futures import ThreadPoolExecutor

import boto3

ROOT = os.environ.get("CASCADE_ROOT", ".")
CLASSES = json.load(open(f"{ROOT}/audioset_strong_456_classes.json"))
CLSET = set(CLASSES)
CLASS_STR = ", ".join(CLASSES)
MODEL = "global.anthropic.claude-opus-4-8"


CONSERVATIVE = ("Map to canonical classes. RECALL-oriented gate: be conservative, "
                "if unsure output ALL plausible canonical candidates (ontology ancestors AND specific "
                "children). Better to include an extra plausible class than miss the right one. "
                "Output ONLY a JSON list of canonical class strings, deduplicated.")
PRECISE = ("Map each free-text label to the SINGLE closest canonical class from the vocabulary "
           "(best-guess, no extra candidates). Output ONLY a JSON list of canonical class strings, "
           "deduplicated.")


def normalize(free, client, mode="conservative"):
    instr = CONSERVATIVE if mode == "conservative" else PRECISE
    prompt = (f"Canonical AudioSet vocabulary (456 classes):\n{CLASS_STR}\n\n"
              f"An audio model heard these free-text labels (unioned over several listens): "
              f"\"{free}\"\n\n{instr}")
    for _ in range(3):
        try:
            r = client.invoke_model(modelId=MODEL, body=json.dumps(
                {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 800,
                 "messages": [{"role": "user", "content": prompt}]}))
            t = json.loads(r["body"].read())["content"][0]["text"]
            m = re.search(r"\[.*\]", t, re.S)
            lst = json.loads(m.group(0)) if m else []
            return [c for c in lst if c in CLSET]
        except Exception:
            time.sleep(2)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="/tmp/af3_cascade")
    ap.add_argument("--out", default="/tmp/af3_cascade")
    ap.add_argument("--mode", choices=["conservative", "precise"], default="conservative")
    ap.add_argument("--out_name", default="candidates.jsonl")
    args = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(f"{args.gate_dir}/gate_shard*.jsonl")):
        for line in open(f):
            recs.append(json.loads(line))
    print(f"loaded {len(recs)} gated clips", flush=True)

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    def work(rec):
        union_free = ", ".join(set(", ".join(rec["runs"]).replace("\n", " ").split(",")))
        cand = sorted(set(normalize(union_free, client, args.mode)))
        return {"video_id": rec["video_id"], "gt_events": rec["gt_events"], "candidates": cand}

    from concurrent.futures import as_completed
    import time as _t
    # resume: skip clips already in the output file
    done_vids = set()
    outpath = f"{args.out}/{args.out_name}"
    if os.path.exists(outpath):
        for l in open(outpath):
            try: done_vids.add(json.loads(l)["video_id"])
            except Exception: pass
    todo = [r for r in recs if r["video_id"] not in done_vids]
    print(f"resume: {len(done_vids)} done, {len(todo)} todo", flush=True)
    fo = open(outpath, "a")
    t0 = _t.time(); n = 0
    with ThreadPoolExecutor(max_workers=64) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for fut in as_completed(futs):  # as_completed: slow calls don't block
            r = fut.result()
            fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            n += 1
            if n % 50 == 0:
                rate = n / (_t.time() - t0)
                print(f"normalized {n}/{len(todo)} ({rate:.1f}/s, eta {(len(todo)-n)/max(rate,0.1)/60:.0f}m)", flush=True)
    fo.close()
    out = [json.loads(l) for l in open(outpath)]
    nc = sum(len(r["candidates"]) for r in out) / max(len(out), 1)
    print(f"wrote {args.out_name}: {len(out)} clips, {nc:.1f} candidates/clip", flush=True)


if __name__ == "__main__":
    main()
