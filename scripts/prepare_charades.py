"""Prepare Charades-STA data for the video_span benchmark.

Builds the Charades-STA test parquet the `charades` adapter loads (`CharadesDataset.load_items`
reads it with pandas; each row: `video` = `{id}.mp4`, `caption`, `timestamp = [start, end]`).

Two ways to get the annotations (the script tries the first that is available):
  1. `--from-annotations charades_sta_test.txt` — convert the ORIGINAL Charades-STA annotations
     (Gao et al., TALL). Each line is `VIDID start end##caption`. Fully open, no auth.
  2. HuggingFace `lmms-lab/Charades-STA` (gated — needs `huggingface_hub` + an HF token / accepted
     terms): `pip install huggingface_hub` then `huggingface-cli login`, and run with `--hf`.

Produces, under `--out` (default `data/charades/`):
  - charades_sta_test.parquet      test annotations         (config `paths.data`)
  - Charades_v1_480/               you provide the clips    (config `paths.video_dir`)

Videos (Charades_v1_480, ~16GB) are from the Charades authors (https://prior.allenai.org/projects/charades);
place the `{video_id}.mp4` clips under the `Charades_v1_480/` dir.

    python scripts/prepare_charades.py --from-annotations charades_sta_test.txt
    python scripts/prepare_charades.py --hf      # if you have HF access to lmms-lab/Charades-STA
"""
import argparse
import os
import re

from _prep_utils import out_dir, done_banner

HF_REPO = "lmms-lab/Charades-STA"
HF_FILE = "data/test-00000-of-00001.parquet"
VIDEOS_INFO = "https://prior.allenai.org/projects/charades"


def from_annotations(txt_path: str, out_parquet: str) -> int:
    """Convert original Charades-STA `VIDID start end##caption` lines -> the test parquet."""
    import pandas as pd
    rows = []
    with open(txt_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            head, _, caption = ln.partition("##")
            m = re.match(r"(\S+)\s+([\d.]+)\s+([\d.]+)", head)
            if not m:
                continue
            vid, s, e = m.group(1), float(m.group(2)), float(m.group(3))
            rows.append({"video": f"{vid}.mp4", "caption": caption.strip(),
                         "timestamp": [s, e]})
    df = pd.DataFrame(rows, columns=["video", "caption", "timestamp"])
    df.to_parquet(out_parquet)
    return len(df)


def from_hf(out_parquet: str) -> int:
    """Fetch the lmms-lab/Charades-STA test parquet via huggingface_hub (needs auth/accepted terms)."""
    import shutil
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    shutil.copy(src, out_parquet)
    import pandas as pd
    return len(pd.read_parquet(out_parquet))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default: <repo>/data/charades)")
    ap.add_argument("--from-annotations", default=None,
                    help="path to the original charades_sta_test.txt (open, no auth)")
    ap.add_argument("--hf", action="store_true",
                    help="fetch from HuggingFace lmms-lab/Charades-STA (needs HF auth)")
    args = ap.parse_args()
    root = out_dir(args.out, "charades")
    parquet = os.path.join(root, "charades_sta_test.parquet")

    n = None
    if args.from_annotations:
        n = from_annotations(args.from_annotations, parquet)
        print(f"  built {parquet} from {args.from_annotations}  ({n} queries)")
    elif args.hf:
        try:
            n = from_hf(parquet)
            print(f"  fetched {parquet} from HF {HF_REPO}  ({n} queries)")
        except Exception as e:
            print(f"  [error] HF fetch failed ({type(e).__name__}: {str(e)[:120]}).")
            print(f"          {HF_REPO} is gated — accept its terms + `huggingface-cli login`,")
            print(f"          or use --from-annotations charades_sta_test.txt instead.")
    else:
        print("  No source given. Provide ONE of:")
        print("    --from-annotations charades_sta_test.txt   (original Charades-STA; open)")
        print(f"    --hf                                        (HF {HF_REPO}; needs auth)")
        print(f"  Expected parquet columns: video ({{id}}.mp4), caption, timestamp [start,end].")

    videos = os.path.join(root, "Charades_v1_480")
    os.makedirs(videos, exist_ok=True)
    print(f"\n  Videos: download Charades_v1_480 (~16GB) from {VIDEOS_INFO}")
    print(f"  and place the {{video_id}}.mp4 clips under: {videos}")

    done_banner("Charades-STA", [f"paths.data:      {parquet}",
                                 f"paths.video_dir: {videos}"])


if __name__ == "__main__":
    main()
